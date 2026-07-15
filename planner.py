from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import math
import random


@dataclass
class PlanStep:
    state_id: int
    action: str
    next_state_id: int
    probability: float
    rationale: str


@dataclass
class PlanResult:
    start_state_id: int
    steps: list[PlanStep] = field(default_factory=list)
    terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            'start_state_id': self.start_state_id,
            'terminal': self.terminal,
            'steps': [asdict(s) for s in self.steps],
        }


class SimpleKMeans:
    def __init__(self, n_clusters: int, random_state: int = 42, max_iter: int = 50) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.cluster_centers_: list[list[float]] | None = None

    def fit(self, X: list[list[float]]) -> 'SimpleKMeans':
        rng = random.Random(self.random_state)
        X = [list(map(float, row)) for row in X]
        if len(X) < self.n_clusters:
            self.cluster_centers_ = [row[:] for row in X]
            return self
        centers = [X[i][:] for i in rng.sample(range(len(X)), self.n_clusters)]
        for _ in range(self.max_iter):
            labels = self.predict(X, centers)
            new_centers: list[list[float]] = []
            for i in range(self.n_clusters):
                group = [X[j] for j, lab in enumerate(labels) if lab == i]
                if group:
                    new_centers.append(_mean_vector(group))
                else:
                    new_centers.append(centers[i][:])
            if _allclose(new_centers, centers):
                break
            centers = new_centers
        self.cluster_centers_ = centers
        return self

    def predict(self, X: list[list[float]], centers: list[list[float]] | None = None) -> list[int]:
        centers = centers if centers is not None else self.cluster_centers_
        if centers is None:
            raise RuntimeError('KMeans not fitted')
        out: list[int] = []
        for row in X:
            scores = [dot(row, center) for center in centers]
            out.append(max(range(len(scores)), key=lambda i: scores[i]))
        return out


class MarkovChainPlanner:
    def __init__(self, n_states: int = 12, max_plan_steps: int = 6, terminal_actions: Optional[list[str]] = None, embedder: Any | None = None) -> None:
        self.n_states = n_states
        self.max_plan_steps = max_plan_steps
        self.terminal_actions = set(terminal_actions or ['finalize', 'verify', 'return_answer'])
        self.embedder = embedder
        self.kmeans: Any | None = None
        self.state_centroids: list[list[float]] | None = None
        self.action_vocab: list[str] = []
        self.action_to_id: dict[str, int] = {}
        self.transition_counts: list[list[list[float]]] | None = None
        self.transition_probs: list[list[list[float]]] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any], embedder: Any) -> 'MarkovChainPlanner':
        p = config.get('planner', {})
        return cls(
            int(p.get('n_states', 12)),
            int(p.get('max_plan_steps', 6)),
            list(p.get('terminal_actions', [])),
            embedder,
        )

    def initialize(self) -> None:
        self.transition_counts = [[[0.0] for _ in range(self.n_states)] for _ in range(self.n_states)]
        self.transition_probs = [[[0.0] for _ in range(self.n_states)] for _ in range(self.n_states)]

    def fit(self, records: list[dict[str, Any]]) -> None:
        texts: list[str] = []
        actions: list[str] = []
        for r in records:
            texts.extend(r.get('step_texts', []))
            actions.extend(r.get('actions', []))
        if not texts:
            return

        embs = [self.embedder.embed_text(t) for t in texts]
        n_states = min(self.n_states, max(2, len(embs)))
        km = SimpleKMeans(n_states, random_state=42)
        km.fit(embs)
        self.kmeans = km
        self.state_centroids = km.cluster_centers_ or []
        self.action_vocab = sorted(set(actions + ['finalize']))
        self.action_to_id = {a: i for i, a in enumerate(self.action_vocab)}
        self.transition_counts = [[[0.0 for _ in range(n_states)] for _ in range(len(self.action_vocab))] for _ in range(n_states)]

        for r in records:
            steps, acts = r.get('step_texts', []), r.get('actions', [])
            if len(steps) < 2:
                continue
            states = self._assign([self.embedder.embed_text(s) for s in steps])
            for i, a in enumerate(acts):
                aid = self.action_to_id.get(a)
                if aid is None:
                    continue
                s0 = int(states[i])
                s1 = int(states[min(i + 1, len(states) - 1)])
                self.transition_counts[s0][aid][s1] += 1.0
        self._recompute()

    def _recompute(self) -> None:
        if self.transition_counts is None:
            return
        self.transition_probs = []
        for state_counts in self.transition_counts:
            state_probs: list[list[float]] = []
            for action_counts in state_counts:
                total = sum(action_counts) + 1e-8
                state_probs.append([c / total for c in action_counts])
            self.transition_probs.append(state_probs)

    def _assign(self, embs: list[list[float]]) -> list[int]:
        if self.kmeans is None:
            raise RuntimeError('Planner not fitted')
        return self.kmeans.predict(embs, self.state_centroids)

    def _nearest(self, emb: list[float]) -> int:
        if self.state_centroids is None:
            return 0
        scores = [dot(center, emb) / ((norm(center) * norm(emb)) + 1e-8) for center in self.state_centroids]
        return int(max(range(len(scores)), key=lambda i: scores[i]))

    def plan(self, query_embedding: list[float]) -> PlanResult:
        if self.transition_probs is None or self.state_centroids is None or not self.action_vocab:
            self._fallback_init()
        assert self.transition_probs is not None
        state = self._nearest(query_embedding)
        steps: list[PlanStep] = []
        terminal = False
        for _ in range(self.max_plan_steps):
            aid, nxt, prob = self._best(state)
            action = self.action_vocab[aid]
            steps.append(PlanStep(state, action, nxt, prob, self._why(action)))
            state = nxt
            if action in self.terminal_actions:
                terminal = True
                break
        return PlanResult(steps[0].state_id if steps else state, steps, terminal)

    def _fallback_init(self) -> None:
        dim = max(4, self.n_states)
        self.state_centroids = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(self.n_states)]
        self.action_vocab = ['identify', 'transform', 'compute', 'finalize']
        self.action_to_id = {a: i for i, a in enumerate(self.action_vocab)}
        self.transition_counts = [[[1.0 for _ in range(self.n_states)] for _ in range(len(self.action_vocab))] for _ in range(self.n_states)]
        self._recompute()

    def _best(self, state: int) -> tuple[int, int, float]:
        assert self.transition_probs is not None
        assert self.transition_counts is not None
        sl = self.transition_probs[state]
        counts = self.transition_counts[state]
        # Score each action by how many times it was actually observed being
        # taken from this state. Summing the action's own next-state
        # distribution (as this used to do) doesn't work as a score: each
        # row is independently normalized in _recompute(), so it sums to
        # ~1.0 for *any* action with data, which means it can't discriminate
        # between two competing actions once a state has seen more than
        # one -- ties get broken by action_vocab order rather than by
        # anything learned. Raw counts don't have that problem. See
        # CHANGELOG.
        scores = [sum(action_counts) for action_counts in counts]
        aid = max(range(len(scores)), key=lambda i: scores[i])
        nxt = max(range(len(sl[aid])), key=lambda i: sl[aid][i])
        return aid, nxt, float(sl[aid][nxt])

    def _why(self, action: str) -> str:
        return {
            'identify_integrand': 'identify the symbolic object to transform',
            'integrate_term': 'apply the relevant calculus rule',
            'evaluate_bounds': 'substitute endpoints and simplify',
            'rewrite_equation': 'normalize the expression before solving',
            'factor_expression': 'factor the polynomial or structure',
            'solve_factor': 'solve each factor independently',
            'identify_law': 'select the governing physics law',
            'substitute_values': 'insert the known engineering parameters',
            'compute_result': 'calculate the numeric result',
            'inspect_loop': 'inspect the control flow and bounds',
            'edit_bounds': 'repair the loop range or index logic',
            'append_values': 'apply the intended data mutation',
            'define_function': 'create a function signature and body',
            'use_string_slice': 'use a direct string transformation',
            'return_value': 'return the transformed result',
            'identify_rule': 'select the calculus rule that matches the form',
            'differentiate_terms': 'differentiate each factor or term',
            'apply_rule': 'combine partial derivatives correctly',
            'inspect_query': 'inspect the repeated scan or bottleneck',
            'rewrite_query': 'refactor the query structure',
            'validate_plan': 'confirm the plan with the execution engine',
            'finalize': 'produce the final answer',
        }.get(action, f'apply action: {action}')

    def update_from_success(self, start_embedding: list[float], plan: PlanResult, solution_text: str) -> None:
        if self.transition_counts is None or self.state_centroids is None:
            return
        state = self._nearest(start_embedding)
        for step in plan.steps:
            aid = self.action_to_id.get(step.action)
            if aid is None:
                continue
            self.transition_counts[state][aid][step.next_state_id] += 1.0
            state = step.next_state_id
        self._recompute()


def dot(a: list[float], b: list[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def norm(v: list[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _mean_vector(group: list[list[float]]) -> list[float]:
    dim = len(group[0])
    return [sum(row[i] for row in group) / len(group) for i in range(dim)]


def _allclose(a: list[list[float]], b: list[list[float]], tol: float = 1e-6) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            if abs(x - y) > tol:
                return False
    return True
