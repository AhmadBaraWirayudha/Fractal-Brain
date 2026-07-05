"""
fractal_brain/synaptic.py
BCM‑like synaptic plasticity rule.
Operates on Matrix weights using pre‑ and post‑synaptic activities.
No external dependencies beyond math_utils.
"""
from .math_utils import Vector, Matrix

class BCMPlasticity:
    """
    Bienenstock–Cooper–Munro (BCM) plasticity rule.
    Weight change: Δw = lr * pre * post * (post - θ)
    with sliding threshold: θ ← θ + (E[post²] - θ) / τ
    """
    def __init__(self, learning_rate=0.001, initial_threshold=0.5, tau=1000.0):
        self.lr = learning_rate
        self.threshold = initial_threshold
        self.tau = tau

    def update(self, weight_matrix, pre_activation, post_activation):
        """
        Apply BCM weight update in-place.
        weight_matrix: Matrix (shape: num_pre x num_post)
        pre_activation: Vector (length num_pre)
        post_activation: Vector (length num_post)
        """
        num_pre = weight_matrix.rows
        num_post = weight_matrix.cols
        if not isinstance(pre_activation, Vector):
            pre_activation = Vector(pre_activation)
        if not isinstance(post_activation, Vector):
            post_activation = Vector(post_activation)

        # pre * post * (post - threshold)  -> outer product of pre and modified post
        # modified_post[j] = post[j] * (post[j] - threshold)
        modified_post = Vector([post_activation[j] * (post_activation[j] - self.threshold)
                                for j in range(num_post)])

        # delta matrix = lr * outer(pre, modified_post)
        for i in range(num_pre):
            for j in range(num_post):
                dw = self.lr * pre_activation[i] * modified_post[j]
                weight_matrix.data[i][j] += dw

        # slide threshold: θ <- θ + (mean(post²) - θ) / tau
        if num_post > 0:
            mean_post_sq = sum(p * p for p in post_activation.data) / num_post
            self.threshold += (mean_post_sq - self.threshold) / self.tau

    def compute_delta(self, pre_activation, post_activation):
        """
        Return the weight change Matrix without applying it.
        Useful for inspection or manual application.
        """
        num_pre = len(pre_activation)
        num_post = len(post_activation)
        if not isinstance(pre_activation, Vector):
            pre_activation = Vector(pre_activation)
        if not isinstance(post_activation, Vector):
            post_activation = Vector(post_activation)

        modified_post = Vector([post_activation[j] * (post_activation[j] - self.threshold)
                                for j in range(num_post)])
        delta = Matrix.zeros(num_pre, num_post)
        for i in range(num_pre):
            for j in range(num_post):
                delta.data[i][j] = self.lr * pre_activation[i] * modified_post[j]
        return delta