"""
fractal_brain/markov.py
Fractal Markov chain with bootstrap validation gates.
No external dependencies beyond math_utils.
"""
import random
from .math_utils import Matrix, Vector, softmax, sample_multinomial

class BootstrapGate:
    """
    Validates a candidate next state by bootstrap resampling recent history.
    If the 95% CI (lower bound) for the candidate's frequency excludes zero,
    the transition is allowed; otherwise it is held (remains in current state).
    """
    def __init__(self, window_size=100, n_bootstrap=20, ci=0.95):
        self.history = []          # list of state indices
        self.window_size = window_size
        self.n_bootstrap = n_bootstrap
        self.ci = ci
        self.alpha = (1 - ci) / 2   # lower percentile

    def should_transition(self, candidate: int) -> bool:
        """
        Decide whether to allow a move to `candidate` based on bootstrap test.
        Returns True if transition allowed, else False.
        """
        if len(self.history) < 10:
            # Not enough data – allow transition and record it
            self.history.append(candidate)
            if len(self.history) > self.window_size:
                self.history.pop(0)
            return True

        # bootstrap resampling of history
        hist = self.history[:]
        n = len(hist)
        counts = []
        for _ in range(self.n_bootstrap):
            # sample with replacement of same size
            sample = [hist[int(random.random() * n)] for _ in range(n)]
            prop = sum(1 for s in sample if s == candidate) / n
            counts.append(prop)
        counts.sort()
        # lower bound of CI: index floor(alpha * n_bootstrap)
        lower_idx = int(self.alpha * self.n_bootstrap)
        lower = counts[lower_idx]
        allow = (lower > 0.0)

        if allow:
            self.history.append(candidate)
            if len(self.history) > self.window_size:
                self.history.pop(0)
        return allow


class FractalMarkovNode:
    """
    A node in a fractal Markov chain.
    Level 0: atomic states with a transition matrix.
    Higher levels: contains child nodes, transitions between them.
    """
    def __init__(self, level: int, num_states: int = 3, child_factory=None):
        self.level = level
        self.num_states = num_states
        if level == 0:
            self.is_leaf = True
            # transition matrix (num_states x num_states) – initially equal probabilities
            self.P = Matrix([[1.0/num_states] * num_states for _ in range(num_states)])
        else:
            self.is_leaf = False
            # children are nodes of level-1
            if child_factory is None:
                # fallback recursive creation (will cause infinite recursion if not careful)
                # but we use external factory to avoid that
                raise ValueError("child_factory must be provided for non-leaf nodes")
            self.children = [child_factory(level-1) for _ in range(num_states)]
            # transition matrix between children (num_states x num_states)
            self.P = Matrix([[1.0/num_states] * num_states for _ in range(num_states)])
        # each node has its own bootstrap gate for transition validation
        self.bootstrap_gate = BootstrapGate()

    def forward(self, current_state_idx: int, external_context=None) -> tuple:
        """
        Perform one step: decide next state, return (next_idx, embedding).
        external_context is not used in basic Markov chain, but passed for future RAG.
        Returns: (next_state_idx, embedding_vector as list)
        """
        # get row of transition matrix as probability vector
        row = Vector(self.P.data[current_state_idx])
        # softmax not strictly needed if we keep rows normalized, but we do it for safety
        probs = softmax(row)
        # decide candidate via sampling
        candidate = sample_multinomial(probs)
        # validate with bootstrap gate
        if self.bootstrap_gate.should_transition(candidate):
            next_idx = candidate
        else:
            next_idx = current_state_idx

        if self.is_leaf:
            emb = [1.0 if i == next_idx else 0.0 for i in range(self.num_states)]
            return next_idx, emb
        else:
            # transition to child node occurred; recurse into that child
            child = self.children[next_idx]
            child_next, child_emb = child.forward(next_idx, external_context)
            # For simplicity, return the child's embedding as this node's embedding
            # (could be concatenated later, but here we just pass through)
            return next_idx, child_emb


def build_fractal_chain(max_level: int, num_states: int = 3) -> FractalMarkovNode:
    """
    Factory to build a root node of the given max_level.
    The root's children are level max_level-1, etc., down to leaves at level 0.
    """
    def factory(level):
        return FractalMarkovNode(level, num_states, child_factory=factory)
    # Create the root (max_level) – its children are built via factory recursion
    root = factory(max_level)
    return root