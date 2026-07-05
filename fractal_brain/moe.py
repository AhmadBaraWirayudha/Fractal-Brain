"""
fractal_brain/moe.py
Mixture of Experts: gating network + multiple small Transformer experts.
No external dependencies beyond math_utils and attention.
"""
from .math_utils import Matrix, Vector, softmax
from .attention import TransformerEncoderLayer

class TransformerExpert:
    """
    A single expert: token embedding + stack of Transformer encoder layers + output projection.
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # token embedding matrix: (vocab_size, d_model)
        self.embedding = Matrix.he_init(vocab_size, d_model)
        # stack of encoder layers
        self.layers = [TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        # output projection: (d_model, vocab_size)
        self.W_out = Matrix.he_init(d_model, vocab_size)
        # populated after the first forward() call; see forward() below
        self._last_hidden = None
        self._last_token_ids = None

    def forward(self, token_ids):
        """
        token_ids: list of ints (seq_len)
        Returns: Matrix (seq_len, vocab_size) of logits
        """
        seq_len = len(token_ids)
        # embed
        emb_rows = [self.embedding.data[idx] for idx in token_ids]
        x = Matrix(emb_rows)  # (seq_len, d_model)

        # pass through transformer layers
        for layer in self.layers:
            x = layer.forward(x)

        # cache pre-projection hidden states (seq_len, d_model): needed to train W_out
        # via an analytic gradient (see FractalBrain._update_expert_output_layers).
        # Attention/feed-forward layers upstream stay fixed random projections for now --
        # full backprop through them is still future work (see To-Do.md).
        self._last_hidden = x
        self._last_token_ids = list(token_ids)

        # project to vocabulary logits
        logits = x.matmul(self.W_out)  # (seq_len, vocab_size)
        return logits


class GatedMoE:
    """
    Mixture of Experts with a linear gating network.
    The gate logits can be modified externally (by PID correction and lasso mask).
    """
    def __init__(self, num_experts, vocab_size, d_model, num_heads, d_ff, num_layers):
        self.num_experts = num_experts
        self.experts = [TransformerExpert(vocab_size, d_model, num_heads, d_ff, num_layers)
                        for _ in range(num_experts)]
        # gate: linear layer (d_model -> num_experts)
        self.W_gate = Matrix.he_init(d_model, num_experts)

    def forward(self, token_ids, pid_correction=0.0, lasso_mask=None):
        """
        token_ids: list of ints (seq_len)
        pid_correction: scalar added to each gate logit (same for all experts)
        lasso_mask: list of 0/1 floats, length num_experts, applied multiplicatively
        Returns: combined logits (Matrix: seq_len x vocab_size), expert weights (list)
        """
        seq_len = len(token_ids)

        # ---- 1. Compute gate logits ----
        # use the average embedding across tokens as the query vector
        # (in real systems you might use a special token or mean pooling)
        # embed each token to get mean vector
        # To avoid a separate embedding, we'll take the first expert's embedding
        # (since they all share the same token embedding space conceptually)
        emb_rows = [self.experts[0].embedding.data[idx] for idx in token_ids]
        # mean pool
        if seq_len > 0:
            mean_vec = Vector([sum(col) / seq_len for col in zip(*emb_rows)])
        else:
            mean_vec = Vector.zeros(self.experts[0].d_model)

        # raw gate logits (num_experts)
        gate_logits = self.W_gate.linear(mean_vec)  # Vector

        # ---- 2. Apply external corrections ----
        # Hard-mask pruned experts with a large negative logit so they truly drop out of
        # the softmax. Multiplying by 0 only neutralizes a logit to exactly 0, which can
        # still out-compete a surviving expert whose own (unmasked) score is negative --
        # the identical bug and fix as in core.FractalBrain.forward; see CHANGELOG.md.
        if lasso_mask is not None:
            NEG_INF = -1e9
            gate_logits = Vector([gate_logits[i] if lasso_mask[i] > 0.5 else NEG_INF
                                  for i in range(self.num_experts)])

        # pid_correction modulates gate *temperature* (divide) rather than shifting every
        # logit by the same additive amount, which cancels out exactly under softmax and
        # would otherwise make this parameter a complete no-op -- same fix as core.py.
        temperature = max(0.2, min(5.0, 1.0 + pid_correction))
        expert_weights = softmax(Vector([g / temperature for g in gate_logits.data]))

        # ---- 3. Compute expert outputs and combine ----
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            logits_i = expert.forward(token_ids)  # (seq_len, vocab_size)
            expert_outputs.append(logits_i)

        # weighted sum: output = sum_i weight[i] * expert_output_i
        combined = Matrix.zeros(seq_len, expert_outputs[0].cols)
        for t in range(seq_len):
            for v in range(combined.cols):
                s = 0.0
                for i in range(self.num_experts):
                    s += expert_weights[i] * expert_outputs[i].data[t][v]
                combined.data[t][v] = s

        return combined, expert_weights.to_list()