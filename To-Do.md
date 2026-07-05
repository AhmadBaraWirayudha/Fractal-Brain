# To‑Do & Future Enhancements

> See `CHANGELOG.md` for the full list of bugs fixed and why. This file tracks what's
> left, not what changed.

## Beyond the architecture: data, training & productionization infrastructure

*(Added from a separate infrastructure review, folded in here rather than kept as a
one-off note. The short version: this project is a prototype hybrid architecture that
operates on already-tokenized `list[int]` / `list[float]` input and in-memory state --
solid and now-working as that, but missing everything around it that would make it
usable as more than that.)*

### Data pipeline
- [x] ~~Tokenizer~~ -- done: `tokenizer.BPETokenizer`, a real from-scratch BPE tokenizer (train/encode/decode/save/load)
- [x] ~~Vocabulary builder~~ -- done: built into `BPETokenizer.train()` (learns the vocab + merges together, same as most real tokenizer libraries)
- [x] ~~Dataset loader (from raw text)~~ -- done: `dataset.TextDataset`
- [x] ~~Text cleaning / normalization~~ -- done: `tokenizer.normalize_text()` (lowercasing, whitespace collapsing)
- [x] ~~Train / validation / test split~~ -- done: `TextDataset.split()`
- [ ] Batching and padding -- `FractalBrain.step()` currently takes one sequence at a time

### Training completeness
- [ ] A real, general-purpose optimizer. To be precise about where things actually stand
      (see CHANGELOG #22, #23): there *is* now a real per-step gradient update on the
      output projections and the PID gains, but it's plain fixed-learning-rate gradient
      descent, not an optimizer object -- no momentum, no Adam-style moment estimates, no
      weight decay, no LR schedule, no gradient clipping. Attention/feed-forward weights
      still have no gradient signal at all (see "Add proper momentum / optimizer for
      expert parameters" below, which this generalizes).
- [x] ~~Checkpoint resume (depends on serialization below)~~ -- done: `checkpoint.load_checkpoint()` reconstructs a fully-trained `FractalBrain` and `persistence_demo.py` demonstrates continuing training on it (verified bit-for-bit identical to an uninterrupted run, across two separate processes -- see CHANGELOG). A full optimizer with momentum/schedule/decay is still the open part, per the item above.

### Persistence
- [x] ~~Storage schema~~ -- done: `storage.Storage`, a thin SQLite wrapper implementing exactly the schema suggested below (vocab, samples, documents, checkpoints, metrics, plus a generic memory table)
- [ ] The RAG `VectorStore` is in-memory and brute-force by default -- `Storage` can now
      persist and reload its documents (`save_document`/`load_documents`/
      `load_into_vector_store`), but `VectorStore` itself still has no built-in index
      structure, so search stays brute-force even when the vectors are loaded from disk.
      Fine for a toy corpus, not for scaling past one. (Also recall it's not currently
      wired into the gate/logits at all yet -- see "Known gaps" below.)
- [x] ~~Model serialization: save/load weights, vocabulary, training state, retrieval memory~~
      -- weights + training state done: `checkpoint.save_checkpoint`/`load_checkpoint`
      (every weight matrix, PID gains *and* internal state, markov chain state,
      step_count, lasso mask, RAG vector store contents -- verified bit-for-bit across
      two separate processes, see CHANGELOG). Vocabulary save/load was already covered
      by `BPETokenizer.save`/`.load`; `Storage.save_vocab`/`load_vocab` adds a queryable
      copy. Versioned checkpoints: done via `Storage.save_checkpoint_blob`/
      `list_checkpoints`. Not saved: `FractalBrain.teacher` (arbitrary externally-injected
      object, no generally-correct way to reconstruct -- documented in CHANGELOG).

### Scale
- [ ] Pure Python / nested lists throughout -- fine for learning and experimentation (see
      README's "A note on performance"), slow for large vocab/matrices/corpora, no
      vectorized backend. Same underlying gap as "Compile critical parts to C
      extensions" in Long-Term Vision below, not a separate one.

### If persistence is added: suggested storage
- SQLite is the natural first choice (built into Python, single-file, easy to inspect):
  tables for `vocab`, `samples`, `documents` (embeddings as BLOB), `checkpoints`,
  `metrics`, and retrieval `memory`.
- Simpler alternative at small scale: JSONL for samples/vocab plus a flat file or
  separate vector cache for embeddings.

### If a GUI is added
- Not part of the original design -- flagging only because it came up in review, not
  because one is assumed wanted. If pursued (e.g. Tkinter): keep the training loop off
  the GUI's main thread. A mainloop-based GUI blocks on its own event loop, so running
  training directly on it will freeze the UI.

### Suggested build order (if pursuing the above)
1. ~~Tokenizer~~ -- done
2. ~~Dataset loader~~ -- done
3. ~~Storage schema~~ -- done: `storage.Storage` (SQLite)
4. ~~Checkpoint save/load~~ -- done: `checkpoint.py`
5. A proper optimizer
6. Batching
7. Retrieval persistence -- **partially done**: `Storage.save_document`/`load_documents`/
   `load_into_vector_store` persist and reload a `VectorStore`'s contents; it's still
   brute-force search once loaded (no index structure), and still not wired into the
   gate/logits (see "Known gaps").
8. Evaluation loop -- **partially done**: `FractalBrain.evaluate()` is the core read-only
   primitive (forward pass + loss, no weight updates); a fuller harness (aggregate
   metrics across a dataset, perplexity, etc.) is still open. See `train_on_text.py` for
   it in use.
9. GUI wrapper, if wanted, last -- once there's something stable underneath to expose

## Implemented Features
- [x] Native matrix/vector library (`math_utils.py`)
- [x] PID controller with anti‑windup, now with a `compute_output()` probe for meta-gradient estimation
- [x] Fractal Markov chains (4 levels) with bootstrap validation gates
- [x] Lasso‑tentacles (sparse linear + L1 + pruning) -- pruning now genuinely excludes an expert (previously it didn't; CHANGELOG #8)
- [x] Multi‑head attention & Transformer encoder layer
- [x] Mixture of Transformer experts with gating
- [x] In‑memory vector store and RAG fusion cross‑attention (runs every step; see "Known gaps" below -- its output isn't consumed yet)
- [x] BCM synaptic plasticity
- [x] Core `FractalBrain` integrating all components
- [x] 8‑bit quantization (`TurboQuant`)
- [x] PCA via power iteration / truncated SVD
- [x] Wormhole linear shortcuts
- [x] Logic folding (fuzzy AND/OR/NOT)
- [x] Recursive fractal matrices
- [x] JEPA (Joint Embedding Predictive Architecture) -- loss is now numerically sane (CHANGELOG #10); see "Known gaps" -- its own weights still aren't trained
- [x] Scalar autograd engine (`Value`)
- [x] Signal processing (delay line, 1D convolution)
- [x] Knowledge distillation loss, now against a real (if untrained) frozen teacher instead of pure random noise
- [x] Orchestrator / training loop example
- [x] Gradient descent for each expert's output projection (`W_out`), via the exact softmax-cross-entropy gradient (CHANGELOG #22)
- [x] Real meta-gradient descent for the PID gains, via cheap finite differences (CHANGELOG #9, #23)
- [x] Adaptive temperature in softmax based on PID (previously listed under Medium-Term; this is what the PID gains now actually control -- CHANGELOG #7)
- [x] Unit tests for each module (`tests/test_smoke.py`, 95 checks)
- [x] BPE tokenizer with train/encode/decode/save/load (`tokenizer.BPETokenizer`)
- [x] Text dataset with sliding-window examples and train/val/test split (`dataset.TextDataset`)
- [x] Read-only `evaluate()` for validation/test metrics without training on them (`core.FractalBrain.evaluate()`)
- [x] End-to-end real-text training example (`train_on_text.py`)
- [x] Full model checkpointing -- weights, PID state, markov chain state, RAG store, everything (`checkpoint.py`), verified bit-for-bit reproducible across separate processes
- [x] SQLite-backed persistence for vocab, samples, RAG documents, checkpoints, and metrics (`storage.py`)
- [x] End-to-end persistence example (`persistence_demo.py`)

## Known gaps (found while fixing the above -- flagging rather than silently leaving)
- [ ] JEPA's own encoder/predictor weights are never trained: the loss is computed and
      reported correctly now, but `JEPA.update_ema()` is never called, and the context
      encoder has no gradient step of its own either. Properly wiring this up means either
      (a) calling `update_ema()` *and* adding a real gradient step for the context encoder
      + predictor (analytic backprop through two ReLU-linear layers each, similar in
      spirit to what's now done for each expert's `W_out`), or at minimum (b) calling
      `update_ema()` so the target encoder isn't permanently frozen at its initial values.
- [ ] `FractalBrain._retrieve_and_fuse()`'s output (`fused_state`) is computed every step
      -- exercising the RAG/cross-attention path -- but isn't fed into the gate or the
      logits anywhere. Worth deciding where it should plug in (additively into the gate
      alongside the lasso/wormhole contributions? concatenated into an expert's query?)
      before wiring it up -- that's a design choice, not a one-line fix.

## Short‑Term To‑Do
- [x] ~~Replace dummy teacher in `orchestrator.py` with a real frozen expert~~ -- done: `FrozenTeacherExpert`
- [ ] Add proper momentum / optimizer for expert parameters -- **partially done**: each
      expert's output projection (`W_out`) now trains via real gradient descent
      (CHANGELOG #22). Attention and feed-forward weights are still fixed random
      projections; extending real gradient descent to those is the natural next step, and
      a substantially bigger undertaking (full backprop through multi-head attention).
- [x] ~~Implement full forward graph using `Value` to allow meta‑gradient on PID gains~~
      -- done via a cheaper route instead: central-difference estimation that reuses
      cached expert outputs, rather than literally rebuilding the whole forward pass out
      of `Value` nodes (which would also make an already-slow pure-Python library slower
      still). See CHANGELOG #9, #23. Building the full `Value`-based graph remains an
      option if you want gradients through the *entire* network rather than just the
      pieces covered here.
- [ ] Add more sophisticated convolution kernels over the delay line (the delay line
      itself now behaves correctly -- CHANGELOG #12 -- but `convolve1d` still isn't
      applied anywhere; only the single most-delayed value is used)
- [ ] Integrate autograd with plasticity rules (differentiable BCM)
- [x] ~~Write unit tests for each module~~ -- done: `tests/test_smoke.py`

## Medium‑Term
- [ ] Add multi‑GPU / parallel simulation (pure Python multiprocessing)
- [x] ~~Implement adaptive temperature in softmax based on PID~~ -- done (CHANGELOG #7); this is also what made the PID meta-gradient in #9 meaningful to compute at all
- [ ] Extend fractal chain to higher depths or branching factors
- [ ] Add online clustering for RAG memory (incremental FAISS‑like index)
- [ ] Self‑repair scenario replication (kill an expert, observe recovery) -- the pruning
      fix (CHANGELOG #8) makes this a more meaningful experiment now that pruning
      actually excludes an expert instead of only nudging its score toward 0
- [ ] Wire `_retrieve_and_fuse`'s output into the gate or logits (see "Known gaps" above)
- [ ] Train JEPA's own weights (see "Known gaps" above)

## Long‑Term Vision
- [ ] Compile critical parts to C extensions (optional) -- the single biggest pure-Python
      hot loop was the bootstrap-gate resampling (CHANGELOG #20, now ~5x cheaper by
      default, but still the natural first target if you profile a real run)
- [x] ~~Export trained meta‑parameters (PID gains, lasso masks) as a compact config~~ -- superseded: `checkpoint.py` saves these (and everything else) as part of a full checkpoint rather than a separate config export; see `Storage.save_checkpoint_blob` for a compact, named/versioned way to keep several.
- [ ] Interface with external LLMs as "teachers" for distillation (the teacher interface
      is just `.forward(token_ids) -> Matrix(seq_len, vocab_size)`, so any wrapper around
      an external model implementing that method should slot in directly)
- [ ] Publish as a research library with documentation and tutorials
