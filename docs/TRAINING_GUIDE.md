# Training Guide

## Minimal run

1. Import the package.
2. Build `FractalBrain`.
3. Call `step()` or `train_batch()` with token ids and targets.
4. Save checkpoints when training state matters.

## Example pattern

```python
from fractal_brain import FractalBrain, set_seed

set_seed(42)
brain = FractalBrain(vocab_size=1000, d_model=128, num_experts=4)

token_ids = [12, 45, 78]
target = [0.0] * 1000
target[99] = 1.0

logits, loss = brain.step(token_ids, target)
```

## Recommended usage

### Small experiments
- Use `step()` for one example at a time.
- Keep `vocab_size`, `d_model`, and `num_experts` small.

### Mini-batch training
- Use `TextDataset.batches(...)`.
- Pass each batch into `brain.train_batch(...)`.
- Use `Adam` and gradient clipping for more stable updates.

### Persistence
- Use `save_checkpoint()` after a run.
- Use `load_checkpoint()` to resume with the same optimizer and RNG state.

## Limitations

- No external tensor library.
- No GPU path.
- Retrieval is in-memory only.
- Scaling is limited by pure Python execution.
