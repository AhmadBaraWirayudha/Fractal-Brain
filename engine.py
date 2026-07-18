from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from decomposer import DecompositionResult, KNOWN_INTENTS, TaskDecomposer
from decoder import PlanConditionedDecoder
from lkg import LivingKnowledgeGraph
from memory import InteractionRecord, MemoryDocument, VectorMemoryStore
from moe_model import KG_CAVEAT_PREFIX, SharedMoEBackbone
from planner import MarkovChainPlanner, PlanResult
from tokenizer import QueryTokenizer, SentenceSplitter, TextEmbedder

SESSION_INTENT_ENTITY = 'session_intent'


def _document_source(doc: MemoryDocument) -> str:
    """Best-effort provenance label for a memory document, used as the
    knowledge graph's per-source reliability key.

    Documents get an explicit ``metadata['source']`` when added via
    ``ai_pipeline.py``'s learning-memory path (``'pipeline_feedback'``) or
    this file's success path (``'successful_interaction'``), but bootstrap
    records (``data/bootstrap_dataset.jsonl`` via
    ``VectorMemoryStore.bootstrap()``) only carry a boolean ``'bootstrap'``
    flag, no ``'source'`` string -- this normalizes both cases into one
    label space so the knowledge graph tracks reliability per *origin*
    rather than splitting bootstrap data across an inconsistent key.
    """
    explicit = doc.metadata.get('source')
    if explicit:
        return str(explicit)
    if doc.metadata.get('bootstrap'):
        return 'bootstrap'
    return 'unknown'


@dataclass
class EngineState:
    last_interaction_id: Optional[str] = None
    last_input_text: Optional[str] = None
    last_retrieved_ids: list[str] = field(default_factory=list)
    last_plan: Optional[PlanResult] = None
    last_output: Optional[str] = None
    last_decomposition: Optional[DecompositionResult] = None


@dataclass
class OpenClosedLoopResult:
    interaction_id: str
    intent: str
    subtasks: list[str]
    retrieved: list[dict[str, Any]]
    plan: dict[str, Any]
    output: str


class OpenClosedLoopEngine:
    def __init__(self, config_path: str | Path = 'config.yaml') -> None:
        self.config_path = Path(config_path)
        self.base_path = self.config_path.resolve().parent
        self.config = self._load_config(self.config_path)
        logging.basicConfig(
            level=getattr(
                logging,
                self.config.get('runtime', {}).get('log_level', 'INFO').upper(),
                logging.INFO,
            )
        )
        self.logger = logging.getLogger('open_closed_loop_engine')
        self.tokenizer = QueryTokenizer.from_config(self.config)
        self.embedder = TextEmbedder.from_config(self.config)
        self.memory = VectorMemoryStore.from_config(self.config, self.embedder, base_path=self.base_path)
        self.planner = MarkovChainPlanner.from_config(self.config, self.embedder)
        self.decomposer = TaskDecomposer.from_config(self.config)
        self.backbone = SharedMoEBackbone.from_config(self.config)
        self.decoder = PlanConditionedDecoder.from_config(self.config, self.backbone)
        self.knowledge_graph = LivingKnowledgeGraph.from_config(self.config)
        self.state = EngineState()
        self.initialized = False
        # Guards define_entity_states() from re-running on a second
        # initialize() call. define_entity_states() unconditionally
        # rebuilds fresh DiscreteMarkovChain instances (see lkg.py), so
        # calling it again would silently wipe every intent-transition
        # observed so far -- and initialize() is explicitly re-called
        # against the same engine elsewhere (test_bootstrap_is_idempotent),
        # so this needs to be safe the same way bootstrap() already is.
        self._kg_entities_defined = False

    @classmethod
    def from_default_config(cls) -> 'OpenClosedLoopEngine':
        return cls(Path(__file__).with_name('config.yaml'))

    def _load_config(self, path: Path) -> dict[str, Any]:
        text = path.read_text(encoding='utf-8')
        return parse_simple_yaml(text)

    def _resolve_path(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.base_path / p).resolve()

    def initialize(self) -> None:
        self.memory.initialize()
        self.planner.initialize()
        self._bootstrap()
        if not self._kg_entities_defined:
            self.knowledge_graph.define_entity_states(SESSION_INTENT_ENTITY, list(KNOWN_INTENTS))
            self._kg_entities_defined = True
        self.initialized = True

    def _bootstrap(self) -> None:
        ds = self._resolve_path(self.config['paths']['bootstrap_dataset'])
        if not ds.exists():
            return
        records = self.memory.load_bootstrap_records(ds)
        self.memory.bootstrap(records)
        self.planner.fit(records)

    def run(self, input_text: str, cognitive_context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.initialized:
            self.initialize()
        interaction_id = str(uuid.uuid4())
        sentences = SentenceSplitter.split(input_text)
        token_batch = self.tokenizer.tokenize(input_text)
        query_embedding = self.embedder.embed_text(input_text)
        retrieved = self.memory.retrieve(query_embedding, top_k=int(self.config['retrieval']['top_k']))
        decomposition = self.decomposer.decompose(input_text, [d.text for d in retrieved])

        # Knowledge-graph update: this turn's intent is a directly observed
        # session_intent state (not latent). Retrieved documents are NOT
        # recorded as matched_intent evidence here -- only close_loop()
        # does that, from actual feedback. See CHANGELOG.md ("matched_intent
        # confidence made purely feedback-driven") for why: an earlier
        # version recorded retrieval itself as positive evidence weighted by
        # doc.score, which created a per-document confidence floor of
        # score/(score+1) that 100% negative feedback could never cross --
        # worst for documents with the highest retrieval score, which are
        # exactly the ones surfaced verbatim (no caveat needed, by this
        # metric alone) by the retrieval_confidence_threshold gate below.
        self.knowledge_graph.observe_entity_transition(SESSION_INTENT_ENTITY, decomposition.intent)
        # Computed once, immediately after the update above, so both the
        # decoder (trust signal for generation) and the returned payload
        # below see the same values instead of re-querying the graph twice.
        kg_snapshot = self._knowledge_graph_snapshot(decomposition.intent, retrieved)
        doc_confidence = {
            item['doc_id']: {'mean': item['mean'], 'variance': item['variance']}
            for item in kg_snapshot['retrieved_doc_confidence']
        }

        plan = self.planner.plan(query_embedding)
        output = self.decoder.generate(input_text, sentences, token_batch, retrieved, decomposition, plan, cognitive_context=cognitive_context, doc_confidence=doc_confidence)
        self.memory.store_pending_interaction(
            InteractionRecord(
                interaction_id,
                input_text,
                [d.doc_id for d in retrieved],
                plan.to_dict(),
                output,
                None,
                {
                    'sentences': sentences,
                    'intent': decomposition.intent,
                    'subtasks': decomposition.subtasks,
                },
            )
        )
        self.state = EngineState(interaction_id, input_text, [d.doc_id for d in retrieved], plan, output, decomposition)
        return {
            'interaction_id': interaction_id,
            'intent': decomposition.intent,
            'subtasks': decomposition.subtasks,
            'retrieved': [d.to_dict() for d in retrieved],
            'plan': plan.to_dict(),
            'output': output,
            'knowledge_graph': kg_snapshot,
        }

    def _knowledge_graph_snapshot(self, intent: str, retrieved: list[MemoryDocument]) -> dict[str, Any]:
        """Summarize the knowledge graph's view of this turn: which intent it
        expects next given the session's history so far, and how much it
        currently trusts each retrieved document for the intent just
        observed. Read-only -- this doesn't itself call add_fact/
        observe_entity_transition, both already applied above."""
        predicted = self.knowledge_graph.predict_entity_state(SESSION_INTENT_ENTITY, horizon=1)
        distribution = dict(zip(KNOWN_INTENTS, (round(float(p), 4) for p in predicted)))
        predicted_next_intent = max(distribution, key=distribution.get) if distribution else None
        return {
            'predicted_next_intent': predicted_next_intent,
            'predicted_intent_distribution': distribution,
            'retrieved_doc_confidence': [
                {
                    'doc_id': doc.doc_id,
                    'source': _document_source(doc),
                    **self.knowledge_graph.get_confidence(doc.doc_id, 'matched_intent', intent),
                }
                for doc in retrieved
            ],
        }

    def close_loop(self, feedback: dict[str, Any]) -> dict[str, Any]:
        if not self.state.last_interaction_id:
            raise ValueError('No active interaction to close')
        interaction_id = feedback.get('interaction_id', self.state.last_interaction_id)
        success = bool(feedback.get('success', False))
        corrected_output = feedback.get('corrected_output')
        notes = feedback.get('notes')
        stored = self.memory.finalize_interaction(interaction_id, success, corrected_output, notes)

        if self.state.last_decomposition is not None:
            doc_by_id = {d.doc_id: d for d in self.memory.documents}
            for doc_id in self.state.last_retrieved_ids:
                doc = doc_by_id.get(doc_id)
                source = _document_source(doc) if doc is not None else 'unknown'
                self.knowledge_graph.add_fact(
                    doc_id, 'matched_intent', self.state.last_decomposition.intent,
                    source=source,
                    positive=success,
                )

        if success and self.state.last_plan is not None and self.state.last_input_text:
            solution_text = corrected_output or self.state.last_output or ''
            solution_text = '\n'.join(
                line for line in solution_text.splitlines()
                if not line.startswith(KG_CAVEAT_PREFIX)
            )
            self.memory.add_document(
                solution_text,
                self.embedder.embed_text(solution_text),
                {
                    'source': 'successful_interaction',
                    'input_text': self.state.last_input_text,
                    'interaction_id': interaction_id,
                    'intent': self.state.last_decomposition.intent if self.state.last_decomposition else None,
                },
            )
            self.planner.update_from_success(self.embedder.embed_text(self.state.last_input_text), self.state.last_plan, solution_text)
        return stored


def _strip_comment(raw: str) -> str:
    """Strip a trailing ``# comment``, but ignore ``#`` characters that
    appear inside a single- or double-quoted value (e.g. ``color: "#FF0000"``).
    The previous version split on the first ``#`` unconditionally, which
    would silently truncate any quoted value containing one. See CHANGELOG.
    """
    in_single = False
    in_double = False
    for idx, ch in enumerate(raw):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == '#' and not in_single and not in_double:
            return raw[:idx].rstrip()
    return raw.rstrip()


def parse_simple_yaml(text: str) -> dict[str, Any]:
    lines = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw)
        if stripped.strip():
            lines.append(stripped)
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(0, root)]

    i = 0
    while i < len(lines):
        line = lines[i]
        indent = len(line) - len(line.lstrip(' '))
        content = line.strip()

        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError('Invalid YAML indentation')

        current = stack[-1][1]

        if content.startswith('- '):
            item = parse_scalar(content[2:].strip())
            if not isinstance(current, list):
                raise ValueError('List item without list container')
            current.append(item)
            i += 1
            continue

        if ':' not in content:
            raise ValueError(f'Invalid YAML line: {content!r}')

        key, value = content.split(':', 1)
        key = key.strip()
        value = value.strip()

        if isinstance(current, list):
            raise ValueError('Mapping item inside list not supported in this config')

        if value == '':
            next_is_list = False
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                nxt_indent = len(nxt) - len(nxt.lstrip(' '))
                nxt_content = nxt.strip()
                next_is_list = nxt_indent > indent and nxt_content.startswith('- ')
            container: Any = [] if next_is_list else {}
            current[key] = container
            stack.append((indent + 1, container))
        else:
            current[key] = parse_scalar(value)
        i += 1

    return root


def parse_scalar(value: str) -> Any:
    if value in {'true', 'True'}:
        return True
    if value in {'false', 'False'}:
        return False
    if value in {'null', 'None', '~'}:
        return None
    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(',')]
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    if re.fullmatch(r'[-+]?\d+', value):
        return int(value)
    if re.fullmatch(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', value) or re.fullmatch(r'[-+]?\d+(?:[eE][-+]?\d+)', value):
        return float(value)
    return value


def dump_scalar(value: Any) -> str:
    """Inverse of parse_scalar for one value. Quotes strings that would
    otherwise round-trip as the wrong type (looks like a number/bool/null,
    contains ': ', or is empty)."""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return '[' + ', '.join(dump_scalar(v) for v in value) + ']'
    text = str(value)
    needs_quoting = (
        text == ''
        or parse_scalar(text) != text
        or ': ' in text
        or text.startswith('- ')
    )
    if needs_quoting:
        return "'" + text.replace("'", "''") + "'"
    return text


def dump_simple_yaml(data: dict[str, Any], indent: int = 0) -> str:
    """Inverse of parse_simple_yaml: dict -> the same indentation-based
    format it parses. Lists are written inline (`[a, b, c]`) rather than as
    `- item` blocks -- parse_simple_yaml accepts both, and inline is simpler
    to get right for the flat string/number lists this config actually has
    (e.g. planner.terminal_actions). Round-trip-tested against the shipped
    config.yaml in tests/test_regressions.py."""
    pad = '  ' * indent
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f'{pad}{key}:')
            lines.append(dump_simple_yaml(value, indent + 1))
        else:
            lines.append(f'{pad}{key}: {dump_scalar(value)}')
    return '\n'.join(lines)


def main() -> None:
    engine = OpenClosedLoopEngine.from_default_config()
    engine.initialize()
    print(json.dumps(engine.run('Solve the integral of 2x from 0 to 4.'), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
