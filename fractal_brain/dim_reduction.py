"""
fractal_brain/dim_reduction.py
Dimensional reduction via truncated SVD (power iteration method).
Pure Python, no external dependencies.
"""
from .math_utils import Matrix, Vector
import math
import random

def power_iteration(A: Matrix, num_iters=100):
    """
    Compute the dominant eigenvector of A^T A (i.e., first right singular vector).
    Returns a Vector (length A.cols).
    """
    n = A.cols
    v = Vector.random(n)
    for _ in range(num_iters):
        # v = A^T (A v)
        Av = A.dot_vector(v)                 # (A.rows,)
        v_new = A.transpose().dot_vector(Av) # (A.cols,)
        norm = math.sqrt(sum(x*x for x in v_new.data))
        if norm == 0:
            break
        v = Vector([x / norm for x in v_new.data])
    return v

def truncated_svd(A: Matrix, k: int, max_iter=100):
    """
    Return top k singular vectors (U_vectors, S_values, Vt_vectors).
    Uses power iteration with deflation.
    """
    A_T = A.transpose()
    m, n = A.rows, A.cols
    U = []
    S = []
    V = []
    A_residual = Matrix([row[:] for row in A.data])  # copy
    for i in range(k):
        # Find one singular vector of residual matrix
        v = power_iteration(A_residual, max_iter)
        # Compute u = A_residual v / ||A_residual v||
        Av = A_residual.dot_vector(v)
        sigma = math.sqrt(sum(x*x for x in Av.data))
        if sigma < 1e-12:
            break
        u = Vector([x / sigma for x in Av.data])
        U.append(u)
        S.append(sigma)
        V.append(v)
        # Deflate: A_residual = A_residual - sigma * u v^T
        # Construct rank-1 matrix
        for i_row in range(m):
            for j_col in range(n):
                A_residual.data[i_row][j_col] -= sigma * u[i_row] * v[j_col]
    # Convert to Matrix if needed
    U_mat = Matrix([[u[i] for i in range(m)] for u in U]).transpose() if U else None
    Vt_mat = Matrix([v.data for v in V]) if V else None   # (k x n)
    return U_mat, S, Vt_mat

class PCA:
    """
    Reduce dimension using top principal components (computed via SVD of centered data).
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None   # Matrix (n_components x original_dim)

    def fit(self, data_vectors):
        """
        data_vectors: list of Vectors.
        """
        if not data_vectors:
            return
        dim = len(data_vectors[0])
        n = len(data_vectors)
        # compute mean
        self.mean = Vector([sum(vec[i] for vec in data_vectors) / n for i in range(dim)])
        # center data
        centered = []
        for vec in data_vectors:
            centered.append([vec[i] - self.mean[i] for i in range(dim)])
        # build matrix (n x dim)
        X = Matrix(centered)
        # SVD of X (power iteration) to get top n_components of V^T
        U, S, Vt = truncated_svd(X, self.n_components)
        self.components = Vt   # (n_components x dim)
        self.explained_variance = [s**2 / n for s in S]

    def transform(self, vec: Vector):
        """
        Project vec onto principal components.
        Returns reduced Vector.
        """
        if self.components is None:
            raise ValueError("PCA not fitted.")
        centered = Vector([vec[i] - self.mean[i] for i in range(len(self.mean))])
        reduced = self.components.dot_vector(centered)   # (n_components,)
        return reduced

    def inverse_transform(self, reduced_vec: Vector):
        """
        Reconstruct original vector from reduced representation.
        """
        if self.components is None:
            raise ValueError("PCA not fitted.")
        # original ≈ components^T * reduced + mean
        rec = self.components.transpose().dot_vector(reduced_vec)
        return Vector([rec[i] + self.mean[i] for i in range(len(self.mean))])