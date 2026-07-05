"""
fractal_brain/attention.py
Multi‑head scaled dot‑product attention and Transformer encoder layer.
Pure Python, using only math_utils and the standard library.
"""
import math
from .math_utils import Matrix, Vector, softmax_rows

# ---------- Helper functions ----------

def gelu(x):
    """Gaussian Error Linear Unit approximation."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

def layer_norm(x, eps=1e-6):
    """
    Layer normalization for a vector x (list or Vector).
    Returns normalized vector as list.
    """
    if isinstance(x, Vector):
        arr = x.to_list()
    else:
        arr = list(x)
    mean = sum(arr) / len(arr)
    var = sum((a - mean) ** 2 for a in arr) / len(arr)
    std = math.sqrt(var + eps)
    return [(a - mean) / std for a in arr]


# ---------- Attention classes ----------

class MultiHeadAttention:
    """
    Multi‑head scaled dot‑product attention.
    Assumes input dimension d_model is divisible by num_heads.
    """
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Weight matrices: Q, K, V and output projection
        self.W_q = Matrix.he_init(d_model, d_model)
        self.W_k = Matrix.he_init(d_model, d_model)
        self.W_v = Matrix.he_init(d_model, d_model)
        self.W_o = Matrix.he_init(d_model, d_model)

    def _split_heads(self, x):
        """
        x: Matrix of shape (seq_len, d_model)
        Returns: Matrix of shape (seq_len * num_heads, d_k) by reshaping.
        We'll implement by creating a new matrix with rows for each head of each token.
        """
        seq_len = x.rows
        result = Matrix.zeros(seq_len * self.num_heads, self.d_k)
        for t in range(seq_len):
            row = x.data[t]
            for h in range(self.num_heads):
                head_start = h * self.d_k
                for k in range(self.d_k):
                    result.data[t * self.num_heads + h][k] = row[head_start + k]
        return result

    def _combine_heads(self, x):
        """
        x: Matrix of shape (seq_len * num_heads, d_k)
        Returns: Matrix (seq_len, d_model)
        """
        seq_len = x.rows // self.num_heads
        result = Matrix.zeros(seq_len, self.d_model)
        for t in range(seq_len):
            for h in range(self.num_heads):
                src_row = x.data[t * self.num_heads + h]
                for k in range(self.d_k):
                    result.data[t][h * self.d_k + k] = src_row[k]
        return result

    def _scaled_dot_product(self, Q, K, V, mask=None):
        """
        Q, K, V: Matrix (seq_len, d_k)
        Returns: Matrix (seq_len, d_k)
        """
        d_k = Q.cols
        # scores = Q @ K^T / sqrt(d_k)
        K_T = K.transpose()  # (d_k, seq_len)
        scores = Q.matmul(K_T)  # (seq_len, seq_len)
        # scale
        scale = 1.0 / math.sqrt(d_k)
        scores = scores.mul_scalar(scale)

        # apply mask if provided (add -inf to masked positions)
        if mask is not None:
            for i in range(scores.rows):
                for j in range(scores.cols):
                    if mask[i][j] == 0:
                        scores.data[i][j] = -1e9

        # softmax over rows
        attn_weights = softmax_rows(scores)

        # output = attn_weights @ V
        output = attn_weights.matmul(V)
        return output

    def forward(self, x, mask=None):
        """
        x: Matrix (seq_len, d_model)
        Returns: Matrix (seq_len, d_model)
        """
        seq_len = x.rows

        # Linear projections
        Q = x.matmul(self.W_q)  # (seq_len, d_model)
        K = x.matmul(self.W_k)
        V = x.matmul(self.W_v)

        # Split heads
        Q_heads = self._split_heads(Q)   # (seq_len*num_heads, d_k)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        # Apply attention per head in a batched manner: process each head separately
        head_outputs = []
        for h in range(self.num_heads):
            # extract head slices
            q_h = Matrix([Q_heads.data[t] for t in range(h, seq_len * self.num_heads, self.num_heads)])
            k_h = Matrix([K_heads.data[t] for t in range(h, seq_len * self.num_heads, self.num_heads)])
            v_h = Matrix([V_heads.data[t] for t in range(h, seq_len * self.num_heads, self.num_heads)])
            out_h = self._scaled_dot_product(q_h, k_h, v_h, mask)
            head_outputs.append(out_h)

        # Concatenate heads: interleave them to match _combine_heads expectation
        # Build matrix (seq_len*num_heads, d_k) by placing each head's output appropriately
        concat = Matrix.zeros(seq_len * self.num_heads, self.d_k)
        for h in range(self.num_heads):
            for t in range(seq_len):
                concat.data[t * self.num_heads + h] = head_outputs[h].data[t]
        combined = self._combine_heads(concat)

        # Final output projection
        output = combined.matmul(self.W_o)
        return output


class TransformerEncoderLayer:
    """
    One layer of a Transformer encoder: MultiHeadAttention + FeedForward with residuals and LayerNorm.
    """
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        # Feed‑forward: two linear layers with GELU activation
        self.W1 = Matrix.he_init(d_model, d_ff)
        self.b1 = Vector.zeros(d_ff)   # bias
        self.W2 = Matrix.he_init(d_ff, d_model)
        self.b2 = Vector.zeros(d_model)

    def forward(self, x, mask=None):
        """
        x: Matrix (seq_len, d_model)
        Returns: Matrix (seq_len, d_model)
        """
        # Attention sub‑layer
        attn_out = self.attention.forward(x, mask)
        # Residual connection & layer norm
        # apply norm to each token (each row)
        normed_attn = []
        for i in range(x.rows):
            row_sum = [x.data[i][j] + attn_out.data[i][j] for j in range(x.cols)]
            normed_attn.append(layer_norm(row_sum))
        x1 = Matrix(normed_attn)

        # Feed‑forward sub‑layer
        # ff1 = GELU(x1 @ W1 + b1)
        ff1 = x1.matmul(self.W1)
        # add bias
        for i in range(ff1.rows):
            for j in range(ff1.cols):
                ff1.data[i][j] += self.b1[j]
        # activation
        for i in range(ff1.rows):
            for j in range(ff1.cols):
                ff1.data[i][j] = gelu(ff1.data[i][j])

        # ff2 = ff1 @ W2 + b2
        ff2 = ff1.matmul(self.W2)
        for i in range(ff2.rows):
            for j in range(ff2.cols):
                ff2.data[i][j] += self.b2[j]

        # Residual & norm
        normed_out = []
        for i in range(x1.rows):
            row_sum = [x1.data[i][j] + ff2.data[i][j] for j in range(x1.cols)]
            normed_out.append(layer_norm(row_sum))
        return Matrix(normed_out)