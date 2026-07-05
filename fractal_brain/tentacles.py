"""
fractal_brain/tentacles.py
Lasso‑tentacles: sparse linear mapping from Markov‑node state to MoE gate logits,
with L1 penalty and dynamic pruning.
Uses only math_utils.
"""
from .math_utils import Matrix, Vector

class LassoTentacles:
    """
    A linear layer (no bias) whose weights are penalized by L1 norm and can be
    pruned by zeroing out connections where the sum of absolute weights is small.
    The mask is applied multiplicatively during forward.
    """
    def __init__(self, input_dim: int, num_experts: int, l1_lambda: float = 0.01):
        self.num_input = input_dim
        self.num_output = num_experts
        self.l1_lambda = l1_lambda
        # Weight matrix: (num_input x num_output)
        self.W = Matrix.he_init(input_dim, num_experts)
        # Mask: vector of ones, length num_output (controls each output channel)
        self.mask = Vector.ones(num_experts)

    def forward(self, state_emb):
        """
        state_emb: Vector (or list) of length input_dim (num_input).
        Returns: list of gate logits (length num_experts).
        """
        if not isinstance(state_emb, Vector):
            state_emb = Vector(state_emb)
        # raw = state_emb @ W  => (1 x num_input) × (num_input x num_output) -> (1 x num_output)
        raw = self.W.linear(state_emb)   # Vector of length num_output
        # apply mask (element‑wise multiply)
        masked = Vector([raw[i] * self.mask[i] for i in range(len(raw))])
        return masked.to_list()

    def l1_loss(self):
        """Compute L1 penalty: lambda * sum of absolute weights."""
        total = 0.0
        for row in self.W.data:
            total += sum(abs(w) for w in row)
        return self.l1_lambda * total

    def prune(self, threshold=1e-3):
        """
        Update mask: for each output column j, if sum of absolute weights in that
        column is below threshold, set mask[j]=0, otherwise 1.
        """
        num_input = self.num_input
        new_mask = []
        for j in range(self.num_output):
            col_abs_sum = sum(abs(self.W.data[i][j]) for i in range(num_input))
            new_mask.append(0.0 if col_abs_sum < threshold else 1.0)
        self.mask = Vector(new_mask)

    def get_weights_matrix(self):
        """Return a copy of the weight matrix (for inspection)."""
        return Matrix([row[:] for row in self.W.data])

    def update_weights(self, gradient_matrix):
        """
        Manually apply gradient (Matrix) to weights. Used by plasticity rules.
        gradient_matrix shape must be (num_input, num_output).
        """
        for i in range(self.num_input):
            for j in range(self.num_output):
                self.W.data[i][j] += gradient_matrix.data[i][j]