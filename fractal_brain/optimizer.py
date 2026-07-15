"""
fractal_brain/optimizer.py
Generic, dependency-free optimizers operating on this project's own Matrix type and on
plain scalars, plus gradient clipping and a few learning-rate schedules.

Covers the "A proper optimizer" item from To-Do.md. What existed before this (see
core.FractalBrain._update_expert_output_layers / _meta_update_pid_gains, per
CHANGELOG's earlier entries) computed a gradient and immediately applied it with a
fixed learning rate in the same breath -- no momentum, no per-parameter adaptive rates,
no weight decay, no schedule, no clipping. core.py now delegates the *application* of a
gradient to one of these; it still computes the gradients itself (this module knows
nothing about FractalBrain, attention, or losses -- it only knows how to turn a
gradient into a parameter update).

Each optimizer keeps per-parameter state (momentum buffers, or Adam's moment
estimates) keyed by a caller-supplied string, since this codebase's parameters aren't
uniform tensors living in one flat list the way a framework's `parameters()` would
return -- they're specific named things (an expert's W_out, a PID gain) that the
caller already knows how to name.
"""
import math

from .math_utils import Matrix


class SGD:
    """SGD, optionally with momentum and/or (L2-style) weight decay."""
    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = {}   # key -> Matrix (matrix params) or float (scalar params)

    def step_matrix(self, key, weight_matrix, grad_matrix):
        """In-place update of weight_matrix.data using grad_matrix (same shape)."""
        rows, cols = weight_matrix.rows, weight_matrix.cols
        wd = self.weight_decay
        if self.momentum:
            v = self._velocity.setdefault(key, Matrix.zeros(rows, cols))
            for i in range(rows):
                vi, wi, gi = v.data[i], weight_matrix.data[i], grad_matrix.data[i]
                for j in range(cols):
                    g = gi[j] + wd * wi[j] if wd else gi[j]
                    vi[j] = self.momentum * vi[j] + g
                    wi[j] -= self.lr * vi[j]
        else:
            for i in range(rows):
                wi, gi = weight_matrix.data[i], grad_matrix.data[i]
                for j in range(cols):
                    g = gi[j] + wd * wi[j] if wd else gi[j]
                    wi[j] -= self.lr * g

    def step_scalar(self, key, value, grad):
        """Returns the updated scalar (floats are immutable, so this can't update in place)."""
        if self.weight_decay:
            grad = grad + self.weight_decay * value
        if self.momentum:
            v = self.momentum * self._velocity.get(key, 0.0) + grad
            self._velocity[key] = v
            return value - self.lr * v
        return value - self.lr * grad

    def step_vector(self, key, vec, grad_vec):
        """In-place update of vec.data using grad_vec.data (same length). Vector
        counterpart to step_matrix, for e.g. a bias term."""
        n = len(vec.data)
        wd = self.weight_decay
        if self.momentum:
            v = self._velocity.setdefault(key, [0.0] * n)
            for i in range(n):
                g = grad_vec.data[i] + wd * vec.data[i] if wd else grad_vec.data[i]
                v[i] = self.momentum * v[i] + g
                vec.data[i] -= self.lr * v[i]
        else:
            for i in range(n):
                g = grad_vec.data[i] + wd * vec.data[i] if wd else grad_vec.data[i]
                vec.data[i] -= self.lr * g


class Adam:
    """Adam (Kingma & Ba, 2015): per-parameter adaptive learning rates from running
    estimates of the gradient's mean (m) and uncentered variance (v), with bias
    correction for the early steps. Optionally decoupled weight decay (AdamW-style:
    applied directly to the parameter, not folded into the gradient before the moment
    estimates see it)."""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = {}
        self._v = {}
        self._t = {}

    def step_matrix(self, key, weight_matrix, grad_matrix):
        rows, cols = weight_matrix.rows, weight_matrix.cols
        m = self._m.setdefault(key, Matrix.zeros(rows, cols))
        v = self._v.setdefault(key, Matrix.zeros(rows, cols))
        t = self._t.get(key, 0) + 1
        self._t[key] = t
        b1, b2, eps, wd = self.beta1, self.beta2, self.eps, self.weight_decay
        bias1 = 1 - b1 ** t
        bias2 = 1 - b2 ** t
        for i in range(rows):
            mi, vi, wi, gi = m.data[i], v.data[i], weight_matrix.data[i], grad_matrix.data[i]
            for j in range(cols):
                g = gi[j]
                mi[j] = b1 * mi[j] + (1 - b1) * g
                vi[j] = b2 * vi[j] + (1 - b2) * g * g
                m_hat = mi[j] / bias1
                v_hat = vi[j] / bias2
                if wd:
                    wi[j] -= self.lr * wd * wi[j]   # decoupled weight decay
                wi[j] -= self.lr * m_hat / (math.sqrt(v_hat) + eps)

    def step_scalar(self, key, value, grad):
        m = self._m.get(key, 0.0)
        v = self._v.get(key, 0.0)
        t = self._t.get(key, 0) + 1
        self._t[key] = t
        b1, b2 = self.beta1, self.beta2
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        self._m[key] = m
        self._v[key] = v
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        if self.weight_decay:
            value -= self.lr * self.weight_decay * value
        return value - self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def step_vector(self, key, vec, grad_vec):
        """In-place update of vec.data using grad_vec.data (same length). Vector
        counterpart to step_matrix, for e.g. a bias term."""
        n = len(vec.data)
        m = self._m.setdefault(key, [0.0] * n)
        v = self._v.setdefault(key, [0.0] * n)
        t = self._t.get(key, 0) + 1
        self._t[key] = t
        b1, b2, eps, wd = self.beta1, self.beta2, self.eps, self.weight_decay
        bias1 = 1 - b1 ** t
        bias2 = 1 - b2 ** t
        for i in range(n):
            g = grad_vec.data[i]
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            m_hat = m[i] / bias1
            v_hat = v[i] / bias2
            if wd:
                vec.data[i] -= self.lr * wd * vec.data[i]
            vec.data[i] -= self.lr * m_hat / (math.sqrt(v_hat) + eps)


def clip_grad_norm_matrix(grad_matrix, max_norm):
    """Return a rescaled copy of grad_matrix whose overall L2 norm is at most max_norm
    (a no-op copy if it's already within budget). Does not mutate grad_matrix."""
    total = sum(x * x for row in grad_matrix.data for x in row)
    norm = math.sqrt(total)
    if norm <= max_norm or norm == 0.0:
        return grad_matrix
    scale = max_norm / norm
    return Matrix([[x * scale for x in row] for row in grad_matrix.data])


class ConstantLR:
    """Always returns base_lr, regardless of step. The default -- matches "no schedule"."""
    def __init__(self, base_lr):
        self.base_lr = base_lr

    def get_lr(self, step):
        return self.base_lr


class StepLR:
    """Multiply base_lr by gamma every step_size steps (a standard "staircase" decay)."""
    def __init__(self, base_lr, step_size, gamma=0.5):
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, step):
        return self.base_lr * (self.gamma ** (step // self.step_size))


class CosineAnnealingLR:
    """Smoothly decays from base_lr to min_lr following a cosine curve over total_steps,
    then holds at min_lr."""
    def __init__(self, base_lr, total_steps, min_lr=0.0):
        self.base_lr = base_lr
        self.total_steps = max(total_steps, 1)
        self.min_lr = min_lr

    def get_lr(self, step):
        step = min(step, self.total_steps)
        cos_inner = math.pi * step / self.total_steps
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(cos_inner))


class LinearWarmupLR:
    """Linearly ramps from 0 to base_lr over warmup_steps, then delegates to `after`
    (another schedule, evaluated as if step 0 were the first post-warmup step;
    defaults to holding steady at base_lr)."""
    def __init__(self, base_lr, warmup_steps, after=None):
        self.base_lr = base_lr
        self.warmup_steps = max(warmup_steps, 1)
        self.after = after if after is not None else ConstantLR(base_lr)

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        return self.after.get_lr(step - self.warmup_steps)
