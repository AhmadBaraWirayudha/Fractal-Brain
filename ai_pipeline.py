from __future__ import annotations

import json
import re
import textwrap
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from engine import OpenClosedLoopEngine
from fractal_brain import FractalBrain, set_seed
from fractal_brain.math_utils import Vector, softmax


@dataclass
class PipelineStageTrace:
    name: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineReflection:
    confidence: float
    summary: str
    strengths: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    next_action: str = ''
    should_train: bool = False
    training_note: str = ''


@dataclass
class UnifiedRunState:
    last_input_text: Optional[str] = None
    last_normalized_text: Optional[str] = None
    last_closed_loop_result: Optional[dict[str, Any]] = None
    last_fractal_logits: Optional[list[list[float]]] = None
    last_fractal_loss: Optional[float] = None
    last_token_ids: list[int] = field(default_factory=list)
    last_trace: list[PipelineStageTrace] = field(default_factory=list)
    last_reflection: Optional[PipelineReflection] = None
    session_history: list[dict[str, Any]] = field(default_factory=list)
    session_summary: str = ''


@dataclass
class UnifiedPipelineResult:
    interaction_id: str
    input_text: str
    normalized_text: str
    final_output: str
    closed_loop: dict[str, Any]
    fractal: dict[str, Any]
    reflection: dict[str, Any]
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UnifiedAIPipeline:
    """One end-to-end AI pipeline that fuses retrieval, planning, generation, and
    fractal reasoning into a single request lifecycle.

    The flow is:
        1. normalize input
        2. run the closed-loop engine for retrieve/decompose/plan/generate
        3. run FractalBrain as the cognitive scorer and adaptive memory stage
        4. produce a reflection layer that can guide learning or clarification
        5. optionally learn from feedback
    """

    def __init__(
        self,
        config_path: str | Path = 'config.yaml',
        brain_vocab_size: int = 4096,
        brain_d_model: int = 64,
        num_experts: int = 4,
        seed: int = 42,
    ) -> None:
        set_seed(seed)
        self.closed_loop = OpenClosedLoopEngine(config_path)
        self.brain = FractalBrain(
            vocab_size=brain_vocab_size,
            d_model=brain_d_model,
            num_experts=num_experts,
            num_heads=2,
            d_ff=128,
            num_layers=1,
            num_markov_nodes=5,
            markov_states=3,
            max_level=2,
        )
        self.state = UnifiedRunState()

    def initialize(self) -> None:
        self.closed_loop.initialize()

    @property
    def vocab_size(self) -> int:
        return self.brain.vocab_size

    def _normalize_text(self, text: str) -> str:
        collapsed = ' '.join(text.strip().split())
        return collapsed

    def _stable_hash(self, text: str) -> int:
        return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'big')

    def _token_ids_from_text(self, text: str) -> list[int]:
        batch = self.closed_loop.tokenizer.tokenize(text)
        return [tid % self.brain.vocab_size for tid in batch.token_ids] or [0]

    def _distribution_from_text(self, text: str) -> list[float]:
        dist = [0.0] * self.brain.vocab_size
        for token in re.findall(r'[A-Za-z0-9_]+|[^\w\s]', text.lower()):
            dist[self._stable_hash(token) % self.brain.vocab_size] += 1.0
        total = sum(dist) or 1.0
        return [x / total for x in dist]

    @staticmethod
    def _preview_text(text: str, limit: int = 160) -> str:
        compact = ' '.join(text.strip().split())
        return textwrap.shorten(compact, width=limit, placeholder='...') if compact else ''

    def _build_session_context(self) -> dict[str, Any]:
        recent = self.state.session_history[-4:]
        if recent:
            avg_loss = sum(float(item.get('loss', 0.0)) for item in recent) / len(recent)
            success_rate = sum(1 for item in recent if (item.get('feedback') or {}).get('success')) / len(recent)
        else:
            avg_loss = 0.0
            success_rate = 0.0
        return {
            'turn_count': len(self.state.session_history),
            'recent_turns': [
                {
                    'input_preview': item.get('input_preview', ''),
                    'intent': item.get('intent', 'unknown'),
                    'confidence': item.get('confidence', 0.0),
                    'loss': item.get('loss', 0.0),
                    'output_preview': item.get('output_preview', ''),
                    'feedback_success': bool(item.get('feedback', {}).get('success')) if item.get('feedback') else None,
                }
                for item in recent
            ],
            'average_loss': round(float(avg_loss), 6),
            'success_rate': round(float(success_rate), 6),
            'session_summary': self.state.session_summary or 'No prior session summary yet.',
        }

    def _refresh_session_summary(self) -> str:
        recent = self.state.session_history[-3:]
        if not recent:
            self.state.session_summary = 'No prior interactions.'
            return self.state.session_summary
        lines = []
        for item in recent:
            fb = item.get('feedback') or {}
            feedback_tag = 'success' if fb.get('success') else 'open'
            lines.append(f"{item.get('intent', 'unknown')} | {feedback_tag} | loss={float(item.get('loss', 0.0)):.4f}")
        self.state.session_summary = '; '.join(lines)
        return self.state.session_summary

    def _append_session_turn(
        self,
        input_text: str,
        normalized: str,
        closed_loop_result: dict[str, Any],
        reflection: PipelineReflection,
        loss: float,
        top_tokens: list[dict[str, float]],
    ) -> None:
        entry = {
            'interaction_id': closed_loop_result.get('interaction_id'),
            'input_text': input_text,
            'input_preview': self._preview_text(input_text),
            'normalized_text': normalized,
            'intent': closed_loop_result.get('intent', 'unknown'),
            'loss': float(loss),
            'confidence': float(reflection.confidence),
            'output_preview': self._preview_text(closed_loop_result.get('output', '')),
            'retrieved_doc_ids': [item.get('doc_id') for item in closed_loop_result.get('retrieved', [])],
            'top_tokens': top_tokens[:5],
            'feedback': None,
        }
        self.state.session_history.append(entry)
        self.state.session_history = self.state.session_history[-12:]
        self._refresh_session_summary()

    def _update_last_session_feedback(self, feedback: dict[str, Any], closed_feedback: dict[str, Any], lesson_doc_id: str | None) -> None:
        if not self.state.session_history:
            return
        entry = self.state.session_history[-1]
        entry['feedback'] = {
            'success': bool(feedback.get('success', False)),
            'notes': feedback.get('notes'),
            'corrected_output': feedback.get('corrected_output'),
            'interaction_id': feedback.get('interaction_id'),
            'lesson_doc_id': lesson_doc_id,
            'closed_loop': closed_feedback,
        }
        self._refresh_session_summary()

    @staticmethod
    def _topk_from_logits(logits: list[float], k: int = 8) -> list[dict[str, float]]:
        indexed = sorted(enumerate(logits), key=lambda item: item[1], reverse=True)[:k]
        if not indexed:
            return []
        scores = Vector([score for _, score in indexed])
        probs = softmax(scores).to_list()
        return [
            {'token_id': int(token_id), 'score': float(score), 'probability': float(prob)}
            for (token_id, score), prob in zip(indexed, probs)
        ]

    def _estimate_confidence(self, loss: float, top_tokens: list[dict[str, float]]) -> float:
        if not top_tokens:
            return max(0.0, min(1.0, 1.0 / (1.0 + float(loss))))
        top_prob = float(top_tokens[0].get('probability', 0.0))
        inverse_loss = 1.0 / (1.0 + max(0.0, float(loss)))
        confidence = (inverse_loss * 0.65) + (top_prob * 0.35)
        return max(0.0, min(1.0, confidence))

    def _build_cognitive_context(
        self,
        closed_loop_result: dict[str, Any],
        loss: float,
        top_tokens: list[dict[str, float]],
        token_ids: list[int],
    ) -> dict[str, Any]:
        return {
            'loss': round(float(loss), 6),
            'top_tokens': top_tokens[:5],
            'sequence_length': len(token_ids),
            'retrieved_docs': len(closed_loop_result.get('retrieved', [])),
            'intent': closed_loop_result.get('intent'),
            'subtasks': closed_loop_result.get('subtasks', []),
        }

    def _build_reflection(
        self,
        input_text: str,
        normalized: str,
        closed_loop_result: dict[str, Any],
        cognitive_context: dict[str, Any],
        loss: float,
        top_tokens: list[dict[str, float]],
        feedback: dict[str, Any] | None = None,
    ) -> PipelineReflection:
        confidence = self._estimate_confidence(loss, top_tokens)
        retrieved_count = len(closed_loop_result.get('retrieved', []))
        intent = closed_loop_result.get('intent') or 'unknown'
        plan_steps = closed_loop_result.get('plan', {}).get('steps', [])

        strengths: list[str] = []
        if retrieved_count:
            strengths.append(f'Retrieved {retrieved_count} supporting memory items.')
        if intent != 'unknown':
            strengths.append(f'Identified intent as {intent}.')
        if plan_steps:
            strengths.append(f'Built a {len(plan_steps)}-step plan before generation.')
        if top_tokens:
            strengths.append(f'Highest signal token id: {top_tokens[0]["token_id"]}.')

        risks: list[str] = []
        if loss > 1.5:
            risks.append('Fractal loss is elevated, so the answer may need tighter grounding.')
        if not retrieved_count:
            risks.append('No retrieved memory was available, so the response may rely heavily on heuristics.')
        if len(normalized) < 12:
            risks.append('Input is very short, so ambiguity is likely.')

        if feedback and not bool(feedback.get('success', False)):
            risks.append('Recent feedback indicates the previous answer should be improved.')

        should_train = bool(feedback and feedback.get('success', False)) or confidence < 0.45
        if should_train:
            training_note = 'Strengthen the memory trace with the corrected output and keep the successful pattern.'
        elif confidence < 0.7:
            training_note = 'Consider asking for clarification or tightening the prompt constraints.'
        else:
            training_note = 'Confidence is acceptable; use the current answer as the working solution.'

        next_action = 'Proceed with the generated answer.' if confidence >= 0.7 else 'Refine with a more specific correction or follow-up constraint.'
        summary = (
            f"Intent={intent}; confidence={confidence:.2f}; "
            f"retrieved={retrieved_count}; loss={loss:.4f}; "
            f"token_count={cognitive_context.get('sequence_length', 0)}"
        )
        return PipelineReflection(
            confidence=confidence,
            summary=summary,
            strengths=strengths,
            risks=risks,
            next_action=next_action,
            should_train=should_train,
            training_note=training_note,
        )

    def _trace_from_closed_loop(self, normalized_text: str, closed_loop_result: dict[str, Any], reflection: PipelineReflection) -> list[PipelineStageTrace]:
        plan = closed_loop_result.get('plan', {})
        retrieved = closed_loop_result.get('retrieved', [])
        return [
            PipelineStageTrace(
                name='normalize',
                summary='Collapse whitespace and standardize the incoming request text.',
                data={'normalized_text': normalized_text},
            ),
            PipelineStageTrace(
                name='retrieve',
                summary=f"Retrieve {len(retrieved)} memory documents and match them against the request.",
                data={'doc_ids': [item.get('doc_id') for item in retrieved]},
            ),
            PipelineStageTrace(
                name='decompose',
                summary='Infer the task intent and break it into subgoals.',
                data={
                    'intent': closed_loop_result.get('intent'),
                    'subtasks': closed_loop_result.get('subtasks', []),
                },
            ),
            PipelineStageTrace(
                name='plan',
                summary='Select a Markov plan and terminal action sequence.',
                data={
                    'start_state_id': plan.get('start_state_id'),
                    'terminal': plan.get('terminal'),
                    'steps': plan.get('steps', []),
                },
            ),
            PipelineStageTrace(
                name='generate',
                summary='Produce the final answer from the closed-loop decoder.',
                data={'output_preview': textwrap.shorten(closed_loop_result.get('output', ''), width=220, placeholder='...')},
            ),
            PipelineStageTrace(
                name='fractal_cognition',
                summary='Score the request with FractalBrain and expose the cognitive control signal.',
                data={
                    'confidence': reflection.confidence,
                    'summary': reflection.summary,
                    'strengths': reflection.strengths,
                    'risks': reflection.risks,
                    'next_action': reflection.next_action,
                    'should_train': reflection.should_train,
                },
            ),
        ]

    def _record_learning_memory(
        self,
        source_text: str,
        reflection: PipelineReflection,
        corrected_output: str,
        feedback: dict[str, Any],
    ) -> str | None:
        memory = self.closed_loop.memory
        if getattr(memory, 'conn', None) is None:
            return None
        payload = corrected_output.strip() or source_text.strip()
        if not payload:
            return None
        embedding = self.closed_loop.embedder.embed_text(payload)
        metadata = {
            'source': 'pipeline_feedback',
            'reflection': asdict(reflection),
            'success': bool(feedback.get('success', False)),
            'interaction_id': feedback.get('interaction_id') or self.state.last_closed_loop_result.get('interaction_id') if self.state.last_closed_loop_result else None,
        }
        return memory.add_document(payload, embedding, metadata)

    def run(self, input_text: str, feedback: dict[str, Any] | None = None) -> UnifiedPipelineResult:
        if not self.closed_loop.initialized:
            self.initialize()

        normalized = self._normalize_text(input_text)
        token_ids = self._token_ids_from_text(normalized)
        target = self._distribution_from_text(normalized)
        logits, loss = self.brain.evaluate(token_ids, target)
        last_row = logits.data[-1] if getattr(logits, 'data', None) else []
        top_tokens = self._topk_from_logits(last_row, k=8) if last_row else []

        session_context = self._build_session_context()
        # Reuse the same TaskDecomposer the closed-loop engine will run
        # later, instead of a separate ad hoc keyword check -- this used to
        # be its own 2-bucket guess (math vs. "engineering" for everything
        # else, no "coding" option at all) that could disagree with the
        # real decomposition computed a few lines down. See CHANGELOG.
        intent_guess = self.closed_loop.decomposer.decompose(normalized, []).intent
        prompt_context = {
            'stage': 'pre-generation',
            'normalized_length': len(normalized),
            'token_count': len(token_ids),
            'intent_guess': intent_guess,
            'fractal_loss': round(float(loss), 6),
            'top_tokens': top_tokens[:5],
            'session': session_context,
        }
        closed_loop_result = self.closed_loop.run(normalized, cognitive_context=prompt_context)
        cognitive_context = self._build_cognitive_context(closed_loop_result, float(loss), top_tokens, token_ids)
        cognitive_context['session'] = session_context
        reflection = self._build_reflection(input_text, normalized, closed_loop_result, cognitive_context, float(loss), top_tokens, feedback=feedback)
        trace = self._trace_from_closed_loop(normalized, closed_loop_result, reflection)

        result = UnifiedPipelineResult(
            interaction_id=closed_loop_result['interaction_id'],
            input_text=input_text,
            normalized_text=normalized,
            final_output=closed_loop_result['output'],
            closed_loop=closed_loop_result,
            fractal={
                'loss': float(loss),
                'top_tokens': top_tokens,
                'sequence_length': len(token_ids),
                'cognitive_context': cognitive_context,
            },
            reflection=asdict(reflection),
            trace=[asdict(stage) for stage in trace],
        )

        self.state.last_input_text = input_text
        self.state.last_normalized_text = normalized
        self.state.last_closed_loop_result = closed_loop_result
        self.state.last_fractal_logits = logits.data
        self.state.last_fractal_loss = float(loss)
        self.state.last_token_ids = token_ids
        self.state.last_trace = trace
        self.state.last_reflection = reflection
        self._append_session_turn(input_text, normalized, closed_loop_result, reflection, float(loss), top_tokens)

        if feedback is not None:
            self.observe_feedback(feedback)

        return result

    def observe_feedback(self, feedback: dict[str, Any]) -> dict[str, Any]:
        if not self.state.last_input_text:
            raise ValueError('No active interaction to close')

        closed = self.closed_loop.close_loop(feedback)
        corrected_output = feedback.get('corrected_output') or ''
        success = bool(feedback.get('success', False))

        if self.state.last_closed_loop_result is not None:
            current_reflection = self.state.last_reflection or PipelineReflection(
                confidence=0.0,
                summary='No prior reflection.',
            )
            lesson_doc_id = self._record_learning_memory(
                self.state.last_input_text,
                current_reflection,
                corrected_output,
                feedback,
            )
        else:
            lesson_doc_id = None

        if success and self.state.last_input_text:
            token_ids = self._token_ids_from_text(self.state.last_input_text)
            target = self._distribution_from_text(corrected_output or self.state.last_input_text)
            logits, loss = self.brain.step(token_ids, target)
            self.state.last_fractal_logits = logits.data
            self.state.last_fractal_loss = float(loss)
            self.state.last_token_ids = token_ids

        merged = dict(closed)
        merged['fractal_training'] = {
            'success': success,
            'trained': bool(success and self.state.last_input_text),
            'loss': self.state.last_fractal_loss,
            'lesson_doc_id': lesson_doc_id,
        }
        if self.state.last_reflection is not None:
            merged['reflection'] = asdict(self.state.last_reflection)

        self._update_last_session_feedback(feedback, merged, lesson_doc_id)
        return merged

    def teach_from_example(self, input_text: str, ideal_output: str, notes: str | None = None) -> dict[str, Any]:
        result = self.run(input_text)
        feedback = {
            'interaction_id': result.interaction_id,
            'success': True,
            'corrected_output': ideal_output,
            'notes': notes or 'teacher-provided example',
        }
        return self.observe_feedback(feedback)

    def run_session(self, inputs: Sequence[str], feedbacks: Sequence[dict[str, Any] | None] | None = None) -> dict[str, Any]:
        turns: list[dict[str, Any]] = []
        feedback_list = list(feedbacks or [])
        for index, text in enumerate(inputs):
            turn = self.run(text)
            applied_feedback = feedback_list[index] if index < len(feedback_list) else None
            if applied_feedback is not None:
                turn_feedback = self.observe_feedback(applied_feedback)
                turn_dict = turn.to_dict()
                turn_dict['feedback'] = turn_feedback
            else:
                turn_dict = turn.to_dict()
                turn_dict['feedback'] = None
            turns.append(turn_dict)

        avg_loss = sum(float(turn['fractal']['loss']) for turn in turns) / max(len(turns), 1)
        success_rate = 0.0
        if turns:
            success_rate = sum(1 for turn in turns if turn['feedback'] and turn['feedback']['fractal_training']['success']) / len(turns)

        return {
            'turn_count': len(turns),
            'average_loss': avg_loss,
            'success_rate': success_rate,
            'turns': turns,
            'last_reflection': asdict(self.state.last_reflection) if self.state.last_reflection else None,
            'summary': self.summary(),
        }

    def summary(self) -> str:
        payload = {
            'closed_loop': self.state.last_closed_loop_result,
            'fractal_loss': self.state.last_fractal_loss,
            'token_ids': self.state.last_token_ids[:32],
            'reflection': asdict(self.state.last_reflection) if self.state.last_reflection else None,
            'session_summary': self.state.session_summary,
            'session_turn_count': len(self.state.session_history),
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def export_session_history(self) -> dict[str, Any]:
        return {
            'turn_count': len(self.state.session_history),
            'session_summary': self.state.session_summary,
            'turns': list(self.state.session_history),
        }


HybridCognitiveEngine = UnifiedAIPipeline
HybridRunState = UnifiedRunState
HybridResult = UnifiedPipelineResult


def main() -> None:
    engine = UnifiedAIPipeline()
    engine.initialize()
    result = engine.run('Solve the integral of 2x from 0 to 4.')
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
