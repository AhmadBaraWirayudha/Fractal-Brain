# fractal_brain

A self‑contained, zero‑dependency Python library implementing a hybrid cognitive architecture that combines:

- PID control for stable expert gating
- Fractal Markov chains (4th‑iteration) with bootstrap validation gates
- Lasso‑tentacle sparse connections
- Mixture of Transformers (MoE)
- Retrieval‑Augmented Generation (RAG) with in‑memory vector store
- BCM synaptic plasticity
- Joint Embedding Predictive Architecture (JEPA)
- Knowledge distillation
- Autograd engine, convolutional delay lines, wormholes, logic folding, and more

Everything is written in pure Python – no NumPy, PyTorch, or any external packages.

> **First time here?** See `CHANGELOG.md` for a detailed list of what was fixed in this
> version (several import bugs and shape bugs meant the library couldn't actually run as
> originally shipped) and what new capability was added (the experts and the PID gains
> now receive real gradient-based training; previously nothing but two small weight
> matrices ever got any learning signal at all).

## Project layout

```
your_project/
  fractal_brain/        <- the importable package -- copy this folder as-is
    __init__.py
    core.py
    tokenizer.py         <- BPE tokenizer
    dataset.py           <- next-token-prediction dataset + train/val/test split
    checkpoint.py         <- save/load a full FractalBrain (weights + training state)
    storage.py            <- SQLite persistence (vocab, samples, documents, checkpoints, metrics)
    ...
  how_to_use.py           <- minimal single-step example
  orchestrator.py         <- fuller training-loop example (synthetic data, distillation, TurboQuant, PCA)
  train_on_text.py        <- tokenizer -> dataset -> train/val split -> training -> generation
  persistence_demo.py      <- SQLite storage + checkpoint save/load, end to end
  tests/
    test_smoke.py         <- regression test suite (python tests/test_smoke.py)
```

The example scripts and the package are siblings, not nested -- `fractal_brain/`
is imported *by* them, so it can't live inside itself.

## Installation

Copy the `fractal_brain/` folder into your project (as a sibling of your own code, not
inside it). No installation required.

## Quick Start

```python
from fractal_brain import FractalBrain, set_seed

set_seed(42)
brain = FractalBrain(vocab_size=1000, d_model=128, num_experts=4)

# Training step
token_ids = [12, 45, 78]
target = [0.0]*1000
target[99] = 1.0
logits, loss = brain.step(token_ids, target)

# Generate
probs = brain.sample([12, 45], temperature=0.7)
```

See `how_to_use.py` for a runnable version of this, `orchestrator.py` for a full
training loop on synthetic data (distillation against a frozen teacher, periodic
`TurboQuant` compression stats and `PCA` inspection of the embedding table),
`train_on_text.py` for the full data pipeline on real text: a from-scratch BPE
tokenizer (`tokenizer.BPETokenizer`), a next-token-prediction dataset with a proper
train/val/test split (`dataset.TextDataset`), and validation via `FractalBrain.evaluate()`
(a read-only counterpart to `step()` -- calling `step()` itself on held-out data would
silently train on it), and `persistence_demo.py` for saving/reloading a trained model
and querying training history back out of a SQLite database:

```python
from fractal_brain.checkpoint import save_checkpoint, load_checkpoint

save_checkpoint(brain, "model.json")
brain2 = load_checkpoint("model.json")   # every weight, PID state, markov chain
                                          # state, RAG store, etc. -- ready to keep
                                          # training or to sample from
```

## A note on performance

This is pure Python with no vectorization, so it's meant for learning and
experimentation at small-to-moderate scale, not throughput. As a rough reference point,
the exact config from `how_to_use.py` (`vocab_size=1000, d_model=128, num_experts=4,
num_markov_nodes=7, max_level=3`) runs at roughly half a second per `step()` call on a
typical machine. Runs scale in the config parameters about how you'd expect (vocabulary
size and `d_model` matter most); see the Long-Term section of `To-Do.md` for the natural
path to real speed (compiling the hot loops as C extensions).

## Testing

```
python tests/test_smoke.py
```

Runs a dependency-free suite of 95 checks covering every module, including regression
tests for each bug listed in `CHANGELOG.md`.

## Status

See `To-Do.md` for what's implemented, what's genuinely trained vs. still a fixed random
projection, and what's next.
