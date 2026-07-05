"""
fractal_brain/math_utils.py
Pure‑Python linear algebra, activation functions, and random helpers.
No external dependencies.
"""
import math
import random

class Vector:
    """1‑D list‑backed vector."""
    def __init__(self, data):
        self.data = list(data)
        self.shape = (len(self.data),)

    def __getitem__(self, i): return self.data[i]
    def __setitem__(self, i, val): self.data[i] = val
    def __len__(self): return len(self.data)
    def __iter__(self): return iter(self.data)

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector([a + b for a, b in zip(self.data, other.data)])
        return Vector([a + other for a in self.data])

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector([a - b for a, b in zip(self.data, other.data)])
        return Vector([a - other for a in self.data])

    def __mul__(self, scalar):
        return Vector([a * scalar for a in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def sum(self):
        return sum(self.data)

    def exp(self):
        return Vector([math.exp(x) for x in self.data])

    def log(self):
        return Vector([math.log(max(x, 1e-12)) for x in self.data])

    def to_list(self):
        return self.data

    @staticmethod
    def zeros(n):
        return Vector([0.0] * n)

    @staticmethod
    def ones(n):
        return Vector([1.0] * n)

    @staticmethod
    def random(n):
        return Vector([random.random() for _ in range(n)])

    def __repr__(self):
        return f"Vector({self.data})"


class Matrix:
    """2‑D list‑backed matrix."""
    def __init__(self, data):
        """data: list of lists (rows)"""
        self.data = [list(row) for row in data]
        self.shape = (len(self.data), len(self.data[0]) if self.data else 0)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.data[idx[0]][idx[1]]
        return Vector(self.data[idx])  # row as Vector

    def __setitem__(self, idx, val):
        if isinstance(idx, tuple):
            self.data[idx[0]][idx[1]] = val
        else:
            self.data[idx] = list(val)

    @property
    def rows(self): return self.shape[0]
    @property
    def cols(self): return self.shape[1]

    def dot_vector(self, vec):
        """Matrix × Vector → Vector (classical A·v, treating vec as a column vector).
        Requires self.cols == len(vec); returns a Vector of length self.rows.
        This is the correct operation for e.g. power iteration / SVD (A applied to v).
        It is NOT what you want for a (in_dim, out_dim)-shaped neural-net weight matrix
        applied to an input row-vector -- for that, use `linear()` below instead."""
        assert self.cols == len(vec), f"dot_vector: matrix has {self.cols} cols but vec has length {len(vec)}"
        return Vector([sum(self.data[i][j] * vec[j] for j in range(self.cols)) for i in range(self.rows)])

    def linear(self, vec):
        """Row-vector × Matrix → Vector (computes vec @ self).
        Use this for weight matrices built as Matrix.random(in_dim, out_dim) -- the
        convention used throughout this library for embeddings, projections, and gates.
        Requires self.rows == len(vec) (= in_dim); returns a Vector of length self.cols (= out_dim).
        (dot_vector implements the *other* convention, self @ vec for an (out_dim, in_dim)
        matrix -- mixing the two up is a shape bug that either asserts or silently transposes.)"""
        assert self.rows == len(vec), f"linear: expected input of length {self.rows} (in_dim), got {len(vec)}"
        return Vector([sum(vec[i] * self.data[i][j] for i in range(self.rows)) for j in range(self.cols)])

    def matmul(self, other):
        """Matrix × Matrix → Matrix"""
        assert self.cols == other.rows
        result = [[sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                   for j in range(other.cols)] for i in range(self.rows)]
        return Matrix(result)

    def transpose(self):
        return Matrix([[self.data[i][j] for i in range(self.rows)] for j in range(self.cols)])

    def add(self, other):
        if isinstance(other, Matrix):
            return Matrix([[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(self.data, other.data)])
        # scalar
        return Matrix([[a + other for a in row] for row in self.data])

    def mul_scalar(self, scalar):
        return Matrix([[a * scalar for a in row] for row in self.data])

    def sum(self):
        return sum(sum(row) for row in self.data)

    def abs_sum(self):
        return sum(sum(abs(a) for a in row) for row in self.data)

    def apply(self, func):
        return Matrix([[func(a) for a in row] for row in self.data])

    def to_list(self):
        return self.data

    @staticmethod
    def zeros(rows, cols):
        return Matrix([[0.0] * cols for _ in range(rows)])

    @staticmethod
    def ones(rows, cols):
        return Matrix([[1.0] * cols for _ in range(rows)])

    @staticmethod
    def random(rows, cols):
        return Matrix([[random.random() for _ in range(cols)] for _ in range(rows)])

    @staticmethod
    def eye(n):
        return Matrix([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])

    @staticmethod
    def he_init(in_dim, out_dim):
        """Weight matrix for a neural-net layer mapping in_dim -> out_dim: zero-centered
        Gaussian scaled by 1/sqrt(in_dim) (a simple He/Xavier-style init). Prefer this over
        Matrix.random(...) for anything used as a linear-layer weight: Matrix.random's
        U[0,1) values have mean 0.5 and no fan-in scaling, so stacking even two or three
        such layers without an intervening LayerNorm (as JEPA's encoder/predictor do) can
        amplify activations by orders of magnitude within a handful of layers."""
        scale = 1.0 / math.sqrt(max(in_dim, 1))
        return Matrix([[random.gauss(0.0, scale) for _ in range(out_dim)] for _ in range(in_dim)])

    def __repr__(self):
        return f"Matrix({self.data})"


def softmax(vec: Vector) -> Vector:
    """Numerically stable softmax for a Vector."""
    max_val = max(vec.data)
    exps = [math.exp(x - max_val) for x in vec.data]
    s = sum(exps)
    return Vector([e / s for e in exps])


def softmax_rows(mat: Matrix) -> Matrix:
    """Apply softmax to each row of a Matrix."""
    result = []
    for i in range(mat.rows):
        row_vec = Vector(mat.data[i])
        result.append(softmax(row_vec).to_list())
    return Matrix(result)


def kl_divergence(p_log, q):
    """KL(P||Q) = sum(P * (log P - log Q)).

    p_log: Vector of genuine log-probabilities, i.e. exp(p_log) must already sum to ~1
           (e.g. the output of softmax(...) followed by .log(), or math.log applied to
           a softmax result). q: Vector of probabilities (target distribution), also
           expected to sum to ~1.

    IMPORTANT: passing raw (unnormalized) logits instead of true log-probabilities will
    NOT raise an error but will silently return the wrong number -- the renormalization
    below is only a numerical-stability safety net for tiny floating-point drift in an
    already-normalized input, it does not turn logits into log-probabilities.
    """
    # p = exp(p_log), but we compute directly: sum(p * (log(p) - log(q))) = sum(p*(p_log - log(q)))
    # Use numerical stability: p_log already log. Compute p = exp(p_log)
    max_log = max(p_log.data)
    p = [math.exp(lp - max_log) for lp in p_log.data]
    s = sum(p)
    p = [pi / s for pi in p]  # normalized
    kl = 0.0
    for pi, lpi, qi in zip(p, p_log.data, q.data):
        if pi > 0:
            kl += pi * (lpi - math.log(max(qi, 1e-12)))
    return kl


def sample_multinomial(probs: Vector):
    """Sample an index from a probability vector."""
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            return i
    return len(probs) - 1


# Seed for reproducibility
def set_seed(seed):
    random.seed(seed)