# API Reference

## Package exports

`fractal_brain.__init__` re-exports the public surface.

### Core
- `FractalBrain`
- `set_seed`
- `__version__`

### Math and utilities
- `Vector`, `Matrix`
- `softmax`, `softmax_rows`
- `kl_divergence`, `sample_multinomial`

### Control and search
- `PIDController`
- `BootstrapGate`, `FractalMarkovNode`, `build_fractal_chain`
- `LassoTentacles`
- `Wormhole`
- `LogicFolder`, `fold_states`, `fuzzy_and`, `fuzzy_or`, `fuzzy_not`

### Learning and memory
- `TransformerExpert`, `GatedMoE`
- `VectorStore`, `StateRAGFusion`
- `BCMPlasticity`
- `JEPA`
- `Value`
- `distillation_loss`

### Representation and persistence
- `BPETokenizer`, `normalize_text`
- `TextDataset`, `DatasetView`
- `save_checkpoint`, `load_checkpoint`, `serialize_brain`, `deserialize_brain`
- `Storage`

## Behavioral contracts

### `FractalBrain.step(token_ids, target_distribution)`
- Input:
  - `token_ids`: list of token ids
  - `target_distribution`: dense target vector or `None`
- Output:
  - `(logits, loss)`
- Side effects:
  - advances internal state
  - updates trainable parameters when a target is present

### `FractalBrain.evaluate(token_ids, target_distribution)`
- Read-only forward pass.
- Does not update weights or internal training state.

### `FractalBrain.train_batch(batch)`
- Accepts a list of `(token_ids, target_distribution)` pairs.
- Averages the trainable gradients over the batch before applying one optimizer step.
- BCM plasticity remains example-wise because it is local, not gradient-based.

### `save_checkpoint(path)` / `load_checkpoint(path)`
- Serialize and restore model weights, optimizer state, PID state, storage state, and RNG state.
- Intended for exact resume, not approximate reconstruction.
