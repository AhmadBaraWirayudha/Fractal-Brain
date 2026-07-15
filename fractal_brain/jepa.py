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

    def train_step(self, context_x: Vector, target_x: Vector, optimizer, key_prefix="jepa"):
        """
        One real training step for this JEPA's own weights (Wc, Wc2, Wp1, Wp2): forward
        pass with cached intermediates, then exact analytic backprop through the
        predictor and the context encoder, via the standard 2-layer-MLP chain rule
        (applied twice, chained, since predict()'s input is encode_context()'s output).

        The target encoder (Wt, Wt2) is deliberately NOT backpropagated into here --
        that's the whole point of a JEPA/BYOL-style target network: it's a slowly-moving
        copy of the context encoder, updated only via update_ema() (called at the end of
        this method), never by its own gradient. This mirrors loss()'s existing "stop
        gradient on target" comment, just actually implementing the non-target half of
        that sentence, which previously wasn't implemented at all.

        optimizer: anything exposing step_matrix(key, weight, grad) (e.g. optimizer.SGD
            or optimizer.Adam). Returns the scalar loss (float, computed before this
            step's updates are applied, matching the convention used elsewhere).
        """
        # ---- forward, caching what backprop needs ----
        pre_relu_c = self.Wc.linear(context_x)
        h_c = Vector([max(0.0, v) for v in pre_relu_c.data])
        context_emb = self.Wc2.linear(h_c)

        target_emb = self.encode_target(target_x)   # stop-gradient target network

        pre_relu_p = self.Wp1.linear(context_emb)
        h_p = Vector([max(0.0, v) for v in pre_relu_p.data])
        pred_emb = self.Wp2.linear(h_p)

        diff = [pred_emb[k] - target_emb[k] for k in range(len(pred_emb))]
        loss_val = sum(d * d for d in diff)

        # ---- backprop through predict(): pred = Wp2 . ReLU(Wp1 . context_emb) ----
        embed_dim = len(context_emb)
        hidden_dim_p = len(h_p)
        d_pred = [2.0 * d for d in diff]                                        # dL/d(pred_emb)
        grad_Wp2 = Matrix([[h_p[j] * d_pred[k] for k in range(embed_dim)] for j in range(hidden_dim_p)])
        d_h_p = [sum(d_pred[k] * self.Wp2.data[j][k] for k in range(embed_dim)) for j in range(hidden_dim_p)]
        d_pre_relu_p = [d_h_p[j] if pre_relu_p.data[j] > 0 else 0.0 for j in range(hidden_dim_p)]
        grad_Wp1 = Matrix([[context_emb[d] * d_pre_relu_p[j] for j in range(hidden_dim_p)]
                           for d in range(embed_dim)])
        # dL/d(context_emb): needed to keep backpropagating into the context encoder
        d_context_emb = [sum(d_pre_relu_p[j] * self.Wp1.data[d][j] for j in range(hidden_dim_p))
                         for d in range(embed_dim)]

        # ---- backprop into encode_context(): context_emb = Wc2 . ReLU(Wc . context_x) ----
        hidden_dim_c = len(h_c)
        grad_Wc2 = Matrix([[h_c[j] * d_context_emb[e] for e in range(embed_dim)] for j in range(hidden_dim_c)])
        d_h_c = [sum(d_context_emb[e] * self.Wc2.data[j][e] for e in range(embed_dim)) for j in range(hidden_dim_c)]
        d_pre_relu_c = [d_h_c[j] if pre_relu_c.data[j] > 0 else 0.0 for j in range(hidden_dim_c)]
        input_dim = len(context_x)
        grad_Wc = Matrix([[context_x[d] * d_pre_relu_c[j] for j in range(hidden_dim_c)]
                          for d in range(input_dim)])

        optimizer.step_matrix(f"{key_prefix}.Wc", self.Wc, grad_Wc)
        optimizer.step_matrix(f"{key_prefix}.Wc2", self.Wc2, grad_Wc2)
        optimizer.step_matrix(f"{key_prefix}.Wp1", self.Wp1, grad_Wp1)
        optimizer.step_matrix(f"{key_prefix}.Wp2", self.Wp2, grad_Wp2)

        # target encoder follows the (just-updated) context encoder via EMA, not its own gradient
        self.update_ema()

        return loss_val

    def update_ema(self):
        """After each step, update target weights towards context weights."""
        for param_c, param_t in [(self.Wc, self.Wt), (self.Wc2, self.Wt2)]:
            for i in range(param_c.rows):
                for j in range(param_c.cols):
                    param_t.data[i][j] = (self.ema_decay * param_t.data[i][j] +
                                          (1 - self.ema_decay) * param_c.data[i][j])