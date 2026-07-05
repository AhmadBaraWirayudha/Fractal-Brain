"""
orchestrator.py
Training loop / experiment runner tying everything together.
Generates synthetic data, runs FractalBrain.step in a loop, logs losses,
periodically quantizes weights (TurboQuant) and reduces dimensionality (PCA) for inspection.

This file lives *outside* the fractal_brain/ package, as a sibling -- matching how the
package is meant to be used (see README.md). Run it with:
    python orchestrator.py
from the directory that contains both this file and the fractal_brain/ folder.

(Previously this file lived *inside* fractal_brain/ and did `from fractal_brain import
...`, which cannot work from inside the package's own directory -- see CHANGELOG.md.
It also referenced `Vector` in _inspect() without ever importing it, and used a
DummyTeacher that returned fresh independent random noise on every call rather than a
consistent function of its input -- not a meaningful distillation teacher. Both are
fixed below.)
"""
import random
from fractal_brain import FractalBrain, TurboQuant, PCA, set_seed
from fractal_brain.math_utils import Vector
from fractal_brain.moe import TransformerExpert


class FrozenTeacherExpert:
    """
    A real (if small and randomly-initialized) transformer expert used as a distillation
    teacher, with its weights fixed at construction time and never updated afterwards.

    This replaces the previous placeholder, which returned fresh independent random noise
    on every single call -- not even a consistent function of its own input, so there was
    no way it could meaningfully "teach" anything. A frozen-but-real network at least gives
    consistent, input-dependent soft targets, which is what distillation needs. It isn't a
    pretrained model, but as an architectural placeholder it now behaves like a teacher is
    supposed to; swap in a real pretrained checkpoint here for actual distillation.
    """
    def __init__(self, vocab_size, d_model=64, num_heads=2, d_ff=128, num_layers=2):
        self.expert = TransformerExpert(vocab_size, d_model, num_heads, d_ff, num_layers)

    def forward(self, token_ids):
        return self.expert.forward(token_ids)


class Runner:
    def __init__(self, vocab_size=500, d_model=64, num_experts=4, seed=123):
        set_seed(seed)
        self.brain = FractalBrain(vocab_size=vocab_size, d_model=d_model, num_experts=num_experts,
                                  num_heads=2, d_ff=128, num_layers=1,
                                  num_markov_nodes=5, markov_states=3, max_level=2,
                                  teacher_model=FrozenTeacherExpert(vocab_size))
        self.vocab_size = vocab_size
        self.losses = []

    def generate_dataset(self, n_samples=200):
        """Generate synthetic token sequences and one-hot targets."""
        data = []
        for _ in range(n_samples):
            seq_len = random.randint(2, 10)
            token_ids = [random.randint(0, self.vocab_size-1) for _ in range(seq_len)]
            target_idx = random.randint(0, self.vocab_size-1)
            target = [0.0]*self.vocab_size
            target[target_idx] = 1.0
            data.append((token_ids, target))
        return data

    def train(self, epochs=3, n_samples=200, log_every=20):
        data = self.generate_dataset(n_samples)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i, (token_ids, target) in enumerate(data):
                logits, loss = self.brain.step(token_ids, target)
                epoch_loss += loss
                if (i+1) % log_every == 0:
                    print(f"Epoch {epoch+1} sample {i+1}/{len(data)} loss={loss:.4f}")
            avg_loss = epoch_loss/len(data)
            self.losses.append(avg_loss)
            print(f"=== Epoch {epoch+1} avg loss: {avg_loss:.4f} ===")

            # Periodic maintenance: quantize weights for inspection, reduce dims
            self._inspect()

    def _inspect(self):
        """Quantize the gate weight matrix and print compression stats; also run PCA
        on the embedding matrix for visualization purposes."""
        tq = TurboQuant()
        gate_w = self.brain.moe.W_gate
        quantized, scale, min_val, shape = tq.quantize_matrix(gate_w)
        dequantized = tq.dequantize_matrix(quantized, scale, min_val, shape)
        err = sum(abs(gate_w.data[i][j]-dequantized.data[i][j])
                 for i in range(gate_w.rows) for j in range(gate_w.cols))
        print(f"Gate weight quantization error (sum abs): {err:.4f}")

        # PCA on embedding matrix of first expert
        emb = self.brain.moe.experts[0].embedding
        pca = PCA(n_components=2)
        # Fit on a subset of rows to keep it fast
        sample_rows = [Vector(emb.data[i]) for i in range(min(50, emb.rows))]
        pca.fit(sample_rows)
        reduced = [pca.transform(r) for r in sample_rows]
        print(f"PCA reduced first 3 embedding rows to 2D: {[r.to_list() for r in reduced[:3]]}")


if __name__ == "__main__":
    runner = Runner(vocab_size=300, d_model=48, num_experts=4, seed=7)
    runner.train(epochs=2, n_samples=100, log_every=25)
