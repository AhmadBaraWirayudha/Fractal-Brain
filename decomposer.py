from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_MATH_KEYWORDS = ('integral', 'differentiate', 'solve', 'equation', 'derivative', 'matrix')
_CODING_KEYWORDS = ('python', 'code', 'function', 'bug', 'sql', 'loop', 'class')
_ENGINEERING_KEYWORDS = (
    'force', 'stress', 'strain', 'torque', 'voltage', 'current', 'circuit',
    'load', 'bearing', 'beam', 'pressure', 'velocity', 'material', 'gear',
    'motor', 'conveyor', 'shaft', 'thermal', 'fluid', 'structural', 'wear',
)


@dataclass
class DecompositionResult:
    intent: str
    subtasks: list[str]
    confidence: float
    raw_summary: str


class TaskDecomposer:
    def __init__(self, max_subtasks: int = 5) -> None:
        self.max_subtasks = max_subtasks

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'TaskDecomposer':
        return cls(int(config.get('decomposition', {}).get('max_subtasks', 5)))

    def decompose(self, input_text: str, retrieved_contexts: list[str]) -> DecompositionResult:
        text = input_text.lower()
        math_hits = [k for k in _MATH_KEYWORDS if k in text]
        coding_hits = [k for k in _CODING_KEYWORDS if k in text]
        engineering_hits = [k for k in _ENGINEERING_KEYWORDS if k in text]

        if math_hits:
            intent = 'math_symbolic'
            hits = math_hits
            subtasks = [
                'identify symbolic structure',
                'select the relevant rule or theorem',
                'transform the expression step by step',
                'simplify and verify the result',
            ]
        elif coding_hits:
            intent = 'coding'
            hits = coding_hits
            subtasks = [
                'inspect the code behavior',
                'locate the defect or target change',
                'apply the smallest correct edit',
                'verify the corrected output',
            ]
        elif engineering_hits:
            intent = 'engineering'
            hits = engineering_hits
            subtasks = [
                'identify the governing physical law',
                'substitute the known parameters',
                'compute the engineering quantity',
                'check units and plausibility',
            ]
        else:
            # No keyword signal for any known domain. Previously this
            # silently fell back to "engineering" with physics-specific
            # subtasks regardless of what was actually asked; say "general"
            # honestly instead. See CHANGELOG.
            intent = 'general'
            hits = []
            subtasks = [
                'identify exactly what is being asked',
                'gather the relevant facts or context',
                'work through the reasoning step by step',
                'state the answer clearly',
            ]

        subtasks = subtasks[: self.max_subtasks]
        # Confidence now reflects how much keyword signal actually supported
        # this classification (more distinct matching keywords -> higher
        # confidence) plus a modest bonus for having retrieved context,
        # rather than a flat 0.78/0.62 constant that didn't depend on the
        # input at all. See CHANGELOG.
        keyword_strength = min(len(hits) / 2.0, 1.0)
        base = 0.35 + 0.45 * keyword_strength if hits else 0.3
        confidence = min(0.95, base + (0.1 if retrieved_contexts else 0.0))
        return DecompositionResult(
            intent=intent,
            subtasks=subtasks,
            confidence=round(confidence, 3),
            raw_summary=' | '.join(subtasks),
        )
