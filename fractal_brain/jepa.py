"""
fractal_brain/jepa.py
Joint Embedding Predictive Architecture.
Two encoders (context and target), a predictor, with EMA on target encoder.
Pure Python, using math_utils.
"""
import math
from .math_utils import Matrix, Vector, softmax, set_seed

class JEPA:
    """
    Context encoder and target encoder map input to embedding.
    Predictor takes context embedding and predicts target embedding.
    Loss: L2 distance between predicted and target embedding (stop‑gradient on target).
    """
    def __init__(self, input_dim, embed_dim, hidden_dim=128):
        # Context encoder
        self.Wc = Matrix.he_init(input_dim, hidden_dim)
        self.Wc2 = Matrix.he_init(hidden_dim, embed_dim)
        # Target encoder (will be EMA updated)
        self.Wt = Matrix.he_init(input_dim, hidden_dim)
        self.Wt2 = Matrix.he_init(hidden_dim, embed_dim)
        # Predictor (context -> target embedding)
        self.Wp1 = Matrix.he_init(embed_dim, hidden_dim)
        self.Wp2 = Matrix.he_init(hidden_dim, embed_dim)
        # EMA decay
        self.ema_decay = 0.99

    def encode_context(self, x: Vector):
        h = self.Wc.linear(x)
        h = Vector([max(0.0, val) for val in h.data])  # ReLU
        return self.Wc2.linear(h)

    def encode_target(self, x: Vector):
        h = self.Wt.linear(x)
        h = Vector([max(0.0, val) for val in h.data])
        return self.Wt2.linear(h)

    def predict(self, context_emb: Vector):
        h = self.Wp1.linear(context_emb)
        h = Vector([max(0.0, val) for val in h.data])
        return self.Wp2.linear(h)

    def loss(self, context_x: Vector, target_x: Vector):
        # stop-grad on target encoder: we will not backprop into Wt/Wt2 by manually computing loss
        # In this native setup, we compute target embedding and then detach (just don't update Wt).
        target_emb = self.encode_target(target_x)  # no gradient tracking, but we manually update EMA later
        context_emb = self.encode_context(context_x)
        pred_emb = self.predict(context_emb)
        # L2 loss
        diff = [pred_emb[i] - target_emb[i] for i in range(len(pred_emb))]
        loss_val = sum(d*d for d in diff)
        return loss_val, pred_emb, target_emb

    def update_ema(self):
        """After each step, update target weights towards context weights."""
        for param_c, param_t in [(self.Wc, self.Wt), (self.Wc2, self.Wt2)]:
            for i in range(param_c.rows):
                for j in range(param_c.cols):
                    param_t.data[i][j] = (self.ema_decay * param_t.data[i][j] +
                                          (1 - self.ema_decay) * param_c.data[i][j])