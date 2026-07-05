"""
fractal_brain/recursive_matrices.py
Recursive fractal nested matrices.
A FractalMatrix is defined recursively as a block matrix where each block is itself
a FractalMatrix of the next level, or a leaf matrix at level 0.
"""
from .math_utils import Matrix

class FractalMatrix:
    """
    A recursive block matrix with a power‑of‑two structure.
    For simplicity, we require size = 2^depth, and each block is a FractalMatrix.
    Leaf level contains ordinary Matrices.
    """
    def __init__(self, depth, base_size=2):
        self.depth = depth
        self.size = base_size ** depth   # total rows/cols (square)
        if depth == 0:
            self.is_leaf = True
            # self.size at depth 0 is base_size**0 == 1, so the leaf must be 1x1 to match --
            # get_element()/to_dense() one level up only ever address index [0][0] of this
            # leaf (blk_rows == base_size**(depth-1) == 1 there), so anything larger here
            # would just be unused random data.
            self.leaf = Matrix.random(1, 1)
        else:
            self.is_leaf = False
            # Blocks stored in row-major order (list of length base_size*base_size)
            self.blocks = [FractalMatrix(depth-1, base_size) for _ in range(base_size * base_size)]
            self.base_size = base_size

    def get_element(self, i, j):
        """Slow element retrieval for demonstration; not optimized."""
        if self.is_leaf:
            return self.leaf.data[i][j]
        blk_rows = self.base_size ** (self.depth-1)
        blk_idx = (i // blk_rows) * self.base_size + (j // blk_rows)
        local_i = i % blk_rows
        local_j = j % blk_rows
        return self.blocks[blk_idx].get_element(local_i, local_j)

    def to_dense(self):
        """Convert to a dense Matrix."""
        if self.is_leaf:
            return Matrix([row[:] for row in self.leaf.data])
        blk_size = self.base_size ** (self.depth-1)
        full_size = self.size
        dense = Matrix.zeros(full_size, full_size)
        for r in range(self.base_size):
            for c in range(self.base_size):
                blk = self.blocks[r * self.base_size + c]
                blk_dense = blk.to_dense()
                for i in range(blk_size):
                    for j in range(blk_size):
                        dense.data[r*blk_size + i][c*blk_size + j] = blk_dense.data[i][j]
        return dense