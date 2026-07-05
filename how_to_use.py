"""
Quick-start example for fractal_brain.

Run this from the directory that contains both this file and the fractal_brain/
folder:
    python how_to_use.py

(Previously this file lived *inside* fractal_brain/ and imported `fractal_brain`
absolutely, which cannot work: a package cannot import itself by name from within
its own directory. See CHANGELOG.md.)
"""
from fractal_brain import FractalBrain, set_seed

set_seed(42)
brain = FractalBrain(vocab_size=1000, d_model=128, num_experts=4,
                     num_heads=2, d_ff=256, num_layers=1,
                     num_markov_nodes=7, markov_states=3, max_level=3)

# Example step
token_ids = [12, 45, 78]
target = [0.0]*1000
target[99] = 1.0   # one‑hot target
logits, loss = brain.step(token_ids, target)
print("Loss:", loss)

# Sample next token
probs = brain.sample(token_ids, temperature=0.8)
print("Top probability token:", max(range(len(probs)), key=lambda i: probs[i]))
