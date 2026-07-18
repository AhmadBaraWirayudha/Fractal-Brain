from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from decomposer import DecompositionResult
from memory import MemoryDocument
from moe_model import SharedMoEBackbone
from planner import PlanResult
from tokenizer import TokenBatch


@dataclass
class DecodingResult:
    prompt: str
    output_text: str
    used_fallback: bool


class PlanConditionedDecoder:
    def __init__(self, backbone: SharedMoEBackbone, max_new_tokens: int = 256, temperature: float = 0.2, top_p: float = 0.95, repetition_penalty: float = 1.05) -> None:
        self.backbone = backbone
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    @classmethod
    def from_config(cls, config: dict[str, Any], backbone: SharedMoEBackbone) -> 'PlanConditionedDecoder':
        model_cfg = config.get('model', {})
        return cls(
            backbone,
            int(model_cfg.get('max_new_tokens', 256)),
            float(model_cfg.get('temperature', 0.2)),
            float(model_cfg.get('top_p', 0.95)),
            float(model_cfg.get('repetition_penalty', 1.05)),
        )

    def generate(self, query_text: str, sentences: list[str], token_batch: TokenBatch, retrieved_docs: list[MemoryDocument], decomposition: DecompositionResult, plan: PlanResult, cognitive_context: dict[str, Any] | None = None, doc_confidence: dict[str, dict[str, float]] | None = None) -> str:
        self.backbone.initialize()
        prompt = self.backbone.build_prompt(
            query_text,
            [d.text for d in retrieved_docs],
            decomposition.intent,
            decomposition.subtasks,
            [s.action for s in plan.steps],
            cognitive_context=cognitive_context,
        )
        doc_confidence = doc_confidence or {}
        structured_context = {
            'retrieved': [
                {
                    'doc_id': d.doc_id,
                    'text': d.text,
                    'score': float(d.score),
                    'metadata': d.metadata,
                    'kg_confidence': doc_confidence.get(d.doc_id, {}).get('mean'),
                }
                for d in retrieved_docs
            ],
            'plan_actions': [s.action for s in plan.steps],
            'subtasks': decomposition.subtasks,
            'intent': decomposition.intent,
        }
        return self.backbone.generate(
            prompt,
            self.max_new_tokens,
            self.temperature,
            self.top_p,
            self.repetition_penalty,
            structured_context=structured_context,
        ).text
