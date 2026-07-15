from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from decomposer import DecompositionResult, TaskDecomposer
from decoder import PlanConditionedDecoder
from memory import InteractionRecord, VectorMemoryStore
from moe_model import SharedMoEBackbone
from planner import MarkovChainPlanner, PlanResult
from tokenizer import QueryTokenizer, SentenceSplitter, TextEmbedder


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
        self.state = EngineState()
        self.initialized = False

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
        plan = self.planner.plan(query_embedding)
        output = self.decoder.generate(input_text, sentences, token_batch, retrieved, decomposition, plan, cognitive_context=cognitive_context)
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
        }

    def close_loop(self, feedback: dict[str, Any]) -> dict[str, Any]:
        if not self.state.last_interaction_id:
            raise ValueError('No active interaction to close')
        interaction_id = feedback.get('interaction_id', self.state.last_interaction_id)
        success = bool(feedback.get('success', False))
        corrected_output = feedback.get('corrected_output')
        notes = feedback.get('notes')
        stored = self.memory.finalize_interaction(interaction_id, success, corrected_output, notes)
        if success and self.state.last_plan is not None and self.state.last_input_text:
            solution_text = corrected_output or self.state.last_output or ''
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


def main() -> None:
    engine = OpenClosedLoopEngine.from_default_config()
    engine.initialize()
    print(json.dumps(engine.run('Solve the integral of 2x from 0 to 4.'), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
