"""
fractal_brain/wormhole.py
Wormhole connections: direct linear shortcuts between distant layers/state spaces.
A wormhole maps a source vector to a target space and adds to the target.
"""
from .math_utils import Matrix, Vector

class Wormhole:
    """
    A learned linear shortcut: out = (src @ W) + b added to target.
    """
    def __init__(self, src_dim, target_dim):
        self.W = Matrix.he_init(src_dim, target_dim)
        self.b = Vector.zeros(target_dim)

    def transform(self, src_vector):
        """
        src_vector: Vector of length src_dim.
        Returns Vector of length target_dim.
        """
        out = self.W.linear(src_vector)
        return Vector([out[i] + self.b[i] for i in range(len(self.b))])