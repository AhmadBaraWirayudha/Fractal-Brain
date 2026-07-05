"""
fractal_brain/logic_folding.py
Logic‑based folding of state vectors.
Implements a fuzzy logical fold operation: given a state vector and a rule tree,
it reduces the vector to a scalar or a new state.
For simplicity, we implement a fold that applies logical connectives (AND, OR, NOT)
to selected dimensions.
"""
from .math_utils import Vector

def fuzzy_and(x, y):
    return min(x, y)

def fuzzy_or(x, y):
    return max(x, y)

def fuzzy_not(x):
    return 1.0 - x

class LogicFolder:
    """
    Fold a state vector through a logical expression defined by a binary tree.
    Leaves are indices into the state vector.
    """
    def __init__(self, expression_tree):
        """
        expression_tree: a tuple like ('and', left, right) or ('not', child) or an int (leaf index).
        """
        self.tree = expression_tree

    def evaluate(self, state):
        """
        state: Vector.
        Returns scalar (0~1).
        """
        return self._eval(self.tree, state)

    def _eval(self, node, state):
        if isinstance(node, int):
            return state[node] if node < len(state) else 0.0
        elif isinstance(node, tuple):
            op = node[0]
            if op == 'and':
                left = self._eval(node[1], state)
                right = self._eval(node[2], state)
                return fuzzy_and(left, right)
            elif op == 'or':
                left = self._eval(node[1], state)
                right = self._eval(node[2], state)
                return fuzzy_or(left, right)
            elif op == 'not':
                child = self._eval(node[1], state)
                return fuzzy_not(child)
            else:
                return 0.0
        else:
            return 0.0

def fold_states(states, binary_op):
    """
    General fold: repeatedly apply binary_op to combine a list of Vectors into one.
    """
    if not states:
        return None
    result = states[0]
    for s in states[1:]:
        result = Vector([binary_op(a, b) for a, b in zip(result.data, s.data)])
    return result