from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Any

from quantization import build_quantization_settings

# Shared with engine.py's close_loop(): a successful interaction's output
# gets stored as a future retrievable "solved example" (see CHANGELOG.md,
# "matched_intent confidence made purely feedback-driven" -- same entry
# covers this). A caveat about *that specific past interaction's* knowledge-
# graph confidence would go stale and misleading the moment it's echoed back
# verbatim as someone else's "closest known solved example", possibly long
# after the real confidence has moved on -- so engine.py strips any line
# starting with this prefix before storing. Defined once, here, so the text
# that builds the caveat and the text that strips it can't drift apart.
KG_CAVEAT_PREFIX = 'Caution: this source has a weak track record'


@dataclass
class GenerationOutput:
    text: str
    prompt: str
    used_fallback: bool = False


class SharedMoEBackbone:
    def __init__(self, model_name: str, fallback_model_name: str, use_trust_remote_code: bool = False, quantization_mode: str = 'auto', retrieval_confidence_threshold: float = 0.5, low_confidence_threshold: float = 0.4) -> None:
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.use_trust_remote_code = use_trust_remote_code
        self.quantization_mode = quantization_mode
        self.retrieval_confidence_threshold = retrieval_confidence_threshold
        # Below this, a retrieved document's knowledge-graph confidence
        # (accumulated, feedback-adjusted trust in that specific document for
        # this intent -- see lkg.py / engine.py) is treated as low enough to
        # caveat even when its cosine similarity cleared
        # retrieval_confidence_threshold above. Same threshold value
        # ai_pipeline.py's reflection uses for the same signal, read from
        # the same config field, so the two don't quietly drift apart.
        self.low_confidence_threshold = low_confidence_threshold
        self.tokenizer = None
        self.model = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'SharedMoEBackbone':
        model_cfg = config.get('model', {})
        q = config.get('quantization', {})
        kg_cfg = config.get('knowledge_graph', {})
        return cls(
            model_cfg.get('model_name', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
            model_cfg.get('fallback_model_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'),
            bool(model_cfg.get('use_trust_remote_code', False)),
            str(q.get('mode', 'auto')),
            float(model_cfg.get('retrieval_confidence_threshold', 0.5)),
            float(kg_cfg.get('low_confidence_threshold', 0.4)),
        )

    def initialize(self) -> None:
        # Pure-Python build: always use the fallback rule-based backend.
        # model_name/fallback_model_name are accepted for forward-compat with
        # a future real-backend swap-in, but nothing loads them today -- see
        # CHANGELOG and docs/UNIFIED_PIPELINE.md.
        if self.model_name != 'fallback':
            warnings.warn(
                f"SharedMoEBackbone.model_name is {self.model_name!r}, which looks like "
                "a real model identifier, but this pure-Python build has no code path "
                "that loads one -- the rule-based fallback generator is used regardless "
                "of this setting. See docs/UNIFIED_PIPELINE.md.",
                stacklevel=2,
            )
        build_quantization_settings(self.quantization_mode)
        self.tokenizer = None
        self.model = None

    def build_prompt(self, query_text: str, retrieved_docs: list[str], intent: str, subtasks: list[str], plan_actions: list[str], cognitive_context: dict[str, Any] | None = None) -> str:
        ctx = '\n'.join(f'- {x}' for x in retrieved_docs[:3]) or '- none'
        tasks = '\n'.join(f'{i+1}. {x}' for i, x in enumerate(subtasks)) or '1. solve directly'
        plan = ' -> '.join(plan_actions) or 'finalize'
        cognitive = '- none'
        if cognitive_context:
            cognitive = '\n'.join(f'- {key}: {value}' for key, value in cognitive_context.items())
        return f'''You are a production text inference engine for mathematics, engineering, and coding.

Intent: {intent}
Open-loop plan: {plan}

Retrieved context:
{ctx}

Subtasks:
{tasks}

Cognitive context:
{cognitive}

Task: {query_text}
{query_text}

Answer with a concise, correct solution. Show the essential steps and final result.
'''

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        structured_context: dict[str, Any] | None = None,
    ) -> GenerationOutput:
        return GenerationOutput(self._fallback_generate(prompt, structured_context), prompt, True)

    def encode_prompt(self, prompt: str) -> list[float]:
        return self._fallback_embedding(prompt)

    @staticmethod
    def _fallback_embedding(prompt: str) -> list[float]:
        vec = [0.0] * 384
        for tok in prompt.lower().split():
            # A deterministic hash (not the builtin hash(), which is
            # randomized per-process) so this embedding is stable across
            # runs. See CHANGELOG.
            digest = hashlib.sha256(tok.encode('utf-8')).digest()
            vec[int.from_bytes(digest[:8], 'big') % 384] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _extract_task(prompt: str) -> str:
        task = 'unknown'
        lines = prompt.splitlines()
        for index, line in enumerate(lines):
            if line.startswith('Task:'):
                task = line.split('Task:', 1)[1].strip()
                if not task and index + 1 < len(lines):
                    task = lines[index + 1].strip() or task
                break
        return task

    def _fallback_generate(self, prompt: str, structured_context: dict[str, Any] | None = None) -> str:
        structured_context = structured_context or {}
        task = self._extract_task(prompt)
        retrieved = structured_context.get('retrieved') or []
        plan_actions = [a for a in (structured_context.get('plan_actions') or []) if a]
        subtasks = [s for s in (structured_context.get('subtasks') or []) if s]

        best = max(retrieved, key=lambda r: r.get('score', 0.0), default=None)
        lines: list[str]
        if best is not None and best.get('score', 0.0) >= self.retrieval_confidence_threshold:
            # A close enough match exists in memory -- surface it instead of
            # a generic template. This is still a rule-based fallback (no
            # real model is loaded in this pure-Python build), just one that
            # actually uses what retrieval/planning found. See CHANGELOG.
            lines = [
                f"Fallback backend active (retrieval-grounded, similarity {best['score']:.2f}).",
                f'Task summary: {task}',
                'Closest known solved example:',
                f"- {best['text']}",
            ]
            kg_confidence = best.get('kg_confidence')
            if kg_confidence is not None and kg_confidence < self.low_confidence_threshold:
                # High lexical similarity doesn't mean the document has
                # actually held up -- retrieval/embeddings here are
                # hash-based keyword overlap (see docs/UNIFIED_PIPELINE.md's
                # "Known limitations"), which a close match can satisfy while
                # still being the wrong answer. The knowledge graph tracks
                # that from real feedback (engine.close_loop), so a low
                # score here means this specific document has a track record
                # of not panning out for this intent, not just a hunch.
                lines.append(
                    f"{KG_CAVEAT_PREFIX} for this kind of "
                    f"request (knowledge-graph confidence {kg_confidence:.2f}); verify "
                    f"before relying on it."
                )
            step_texts = [s for s in (best.get('metadata') or {}).get('step_texts') or [] if s]
            if step_texts:
                lines.append('Steps from that example:')
                lines.extend(f'- {s}' for s in step_texts)
            lines.append(f"Proposed answer (adapted from the closest match above): {best['text']}")
        else:
            # No confident match -- fall back to the plan/subtasks the
            # pipeline actually computed, rather than an invariant canned
            # list unrelated to the request.
            steps = plan_actions or subtasks or ['identify the structure', 'apply the matching rule', 'simplify and verify']
            lines = [
                'Fallback backend active (no confident retrieval match; plan-only).',
                f'Task summary: {task}',
                'Essential plan:',
            ]
            lines.extend(f'- {s}' for s in steps)
        return '\n'.join(lines)
