# To‑Do & Future Enhancements

> See `CHANGELOG.md` for the full list of bugs fixed and why. This file tracks what's
> left, not what changed.

## External review: packaging, testing depth, documentation, operational maturity

*(Folded in from a second external review, the same way the first one was -- see
"Beyond the architecture" below. Two small corrections to keep the record accurate
rather than accepting every claim at face value: (1) "no reproducibility package
beyond a seed" undersells what's actually there -- `checkpoint.py` captures the full
model state *and* Python's global RNG state, verified bit-for-bit reproducible across
separate processes, not just a seed argument; (2) "no failure-mode tests" isn't quite
right either -- there are tests for empty input, unregistered checkpoint classes,
missing checkpoint versions, division by zero in the PID controller, and a few others.
Neither correction changes the review's overall verdict, which was otherwise accurate
and matched what this file already tracked before it arrived: several subsystems
weren't part of the actual learning path. That part is now addressed -- see
CHANGELOG's newest entry. Everything below is new ground the review raised, not
previously tracked here.)*

### Packaging
- [ ] `pyproject.toml` (would make this pip-installable; "copy the folder in" remains
      valid too, and is what the README currently documents)
- [ ] `requirements.txt` -- would be empty/trivial today (zero external dependencies is
      the whole point), but worth having for the record and for whatever a packaging
      tool expects to find
- [ ] `LICENSE` -- none exists yet, which matters for anyone deciding whether they can
      use or fork this
- [ ] CI config (run `tests/test_smoke.py` on push, at minimum)
- [ ] Versioning / release process (no version number exists anywhere right now)

### Testing depth
- [ ] Property-based tests (e.g. Hypothesis-style: random valid configs/inputs, assert
      invariants like "loss is always finite", "expert_weights always sums to 1")
      rather than only fixed example-based regression tests
- [ ] Benchmark tests (wall-clock/memory, tracked over time, not just the informal
      timing note in README)
- [ ] Load/stress tests (large vocab, long sequences, many training steps back-to-back)
- [ ] Cross-Python-version compatibility testing (everything so far has only run
      against whatever Python version was available in one environment)
- [ ] More real-data integration tests as actual pass/fail assertions, not just example
      scripts -- `train_on_text.py`/`persistence_demo.py` exercise real pipelines but
      aren't part of `tests/test_smoke.py`'s automated assertions

### Documentation
- [ ] A structured API reference (per-module/per-function contract: inputs, outputs,
      invariants), rather than relying on docstrings plus CHANGELOG/To-Do narrative
- [ ] An architecture diagram (there's a lot of "how four sources combine into the
      gate" type structure that's easier to see than to read)
- [ ] A dedicated training guide walking through common usage patterns end to end,
      distinct from the example scripts

### Operational maturity
- [ ] Monitoring hooks / integration with an experiment tracker (`Storage.log_metric`
      is a basic version of this -- a flat table, not a dashboard or a named-experiment
      concept)
- [ ] Dataset versioning
- [ ] Schema migration plan for `storage.py` (currently `CREATE TABLE IF NOT EXISTS`,
      no `ALTER TABLE` path if the schema needs to change later)

### On the architecture's naming
"Fractal," "wormhole," "tentacles," and similar names are fine as identifiers, but they
name custom heuristics this project invented, not established peer-reviewed methods --
worth being clear-eyed about that distinction when describing what this is.

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
- [x] ~~Batching and padding~~ -- done (batching, not padding -- see below): `FractalBrain.train_batch()` + `dataset.TextDataset.batches()`/`DatasetView.batches()`

### Training completeness
- [x] ~~A real, general-purpose optimizer~~ -- done: `optimizer.py` (`SGD` w/ momentum
      and weight decay, `Adam`, gradient clipping, four LR schedules), wired into
      `core.py` via `output_optimizer`/`pid_optimizer` (defaulting to plain `SGD` at the
      old hard-coded rates, so existing code is unaffected unless you opt in to
      something else). What's still true: attention/feed-forward weights have no
      gradient signal of any kind yet, optimizer or not -- see "Add proper momentum /
      optimizer for expert parameters" below, which this item generalizes.
- [x] ~~Checkpoint resume (depends on serialization below)~~ -- done: `checkpoint.load_checkpoint()` reconstructs a fully-trained `FractalBrain` and `persistence_demo.py` demonstrates continuing training on it (verified bit-for-bit identical to an uninterrupted run, across two separate processes -- see CHANGELOG). A full optimizer with momentum/schedule/decay is now also done, see above.

### Persistence
- [x] ~~Storage schema~~ -- done: `storage.Storage`, a thin SQLite wrapper implementing exactly the schema suggested below (vocab, samples, documents, checkpoints, metrics, plus a generic memory table)
- [ ] The RAG `VectorStore` is in-memory and brute-force by default -- `Storage` can now
      persist and reload its documents (`save_document`/`load_documents`/
      `load_into_vector_store`), but `VectorStore` itself still has no built-in index
      structure, so search stays brute-force even when the vectors are loaded from disk.
      Fine for a toy corpus, not for scaling past one. (Its output is now wired into the
      gate and trained -- `W_rag_gate`, see CHANGELOG -- so this is purely a scaling gap now.)
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
5. ~~A proper optimizer~~ -- done: `optimizer.py`
6. ~~Batching~~ -- done: `FractalBrain.train_batch()` + `dataset.*.batches()`
7. Retrieval persistence -- **partially done**: `Storage.save_document`/`load_documents`/
   `load_into_vector_store` persist and reload a `VectorStore`'s contents; it's still
   brute-force search once loaded (no index structure). Its output is now wired into
   the gate and trained (`W_rag_gate` -- see CHANGELOG), so what's left here is purely
   the scaling/indexing gap, not an integration gap.
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
- [x] In‑memory vector store and RAG fusion cross‑attention -- its output now genuinely participates in (and is trained as part of) expert selection via `W_rag_gate`, not computed and discarded (see CHANGELOG)
- [x] BCM synaptic plasticity
- [x] Core `FractalBrain` integrating all components
- [x] 8‑bit quantization (`TurboQuant`)
- [x] PCA via power iteration / truncated SVD
- [x] Wormhole linear shortcuts
- [x] Logic folding (fuzzy AND/OR/NOT)
- [x] Recursive fractal matrices
- [x] JEPA (Joint Embedding Predictive Architecture) -- own encoder/predictor now actually trained via real backprop, target encoder EMA-updated (see CHANGELOG), not just computing a reported-but-unminimized loss
- [x] Scalar autograd engine (`Value`)
- [x] Signal processing (delay line, 1D convolution)
- [x] Knowledge distillation loss, now against a real (if untrained) frozen teacher instead of pure random noise
- [x] Orchestrator / training loop example
- [x] Gradient descent for each expert's output projection (`W_out`), via the exact softmax-cross-entropy gradient (CHANGELOG #22)
- [x] Real meta-gradient descent for the PID gains, via cheap finite differences (CHANGELOG #9, #23)
- [x] Adaptive temperature in softmax based on PID (previously listed under Medium-Term; this is what the PID gains now actually control -- CHANGELOG #7)
- [x] BPE tokenizer with train/encode/decode/save/load (`tokenizer.BPETokenizer`)
- [x] Text dataset with sliding-window examples, train/val/test split, and batching (`dataset.TextDataset`, `DatasetView`)
- [x] Read-only `evaluate()` for validation/test metrics without training on them (`core.FractalBrain.evaluate()`)
- [x] End-to-end real-text training example (`train_on_text.py`)
- [x] Full model checkpointing -- weights, PID state, markov chain state, RAG store, optimizer state, everything (`checkpoint.py`), verified bit-for-bit reproducible across separate processes
- [x] SQLite-backed persistence for vocab, samples, RAG documents, checkpoints, and metrics (`storage.py`)
- [x] End-to-end persistence example (`persistence_demo.py`)
- [x] Real optimizers -- `SGD` (momentum, weight decay), `Adam`, gradient clipping, 4 LR schedules (`optimizer.py`)
- [x] Mini-batch training with gradient averaging (`core.FractalBrain.train_batch()`)
- [x] Unit tests for each module (`tests/test_smoke.py`, 137 checks)
- [x] RAG fusion wired into the gate (`W_rag_gate`) and trained via a real analytic gradient, alongside `moe.W_gate`, `tentacles.W`, and `wormhole.W`/`.b` -- the whole gate is now trained, not just the expert output projections (`core.FractalBrain._compute_gate_gradients`/`_apply_gate_gradients`, verified against numerical gradients)
- [x] JEPA's own encoder/predictor trained via real analytic backprop, with `update_ema()` finally wired up (`jepa.JEPA.train_step()`, verified against numerical gradients)

## Previously-known gaps (both now resolved)
- ~~JEPA's own encoder/predictor weights are never trained~~ -- resolved: `jepa.JEPA.train_step()`, see CHANGELOG.
- ~~`_retrieve_and_fuse()`'s output isn't fed into the gate~~ -- resolved: `W_rag_gate`, see CHANGELOG.

## Short‑Term To‑Do
- [x] ~~Replace dummy teacher in `orchestrator.py` with a real frozen expert~~ -- done: `FrozenTeacherExpert`
- [x] ~~Add proper momentum / optimizer for expert parameters~~ -- done: `optimizer.py`'s
      `SGD` (momentum, weight decay) and `Adam` are wired in via `output_optimizer` (see
      "A proper optimizer" above). What's still open, and is a materially bigger task:
      attention and feed-forward weights have no gradient signal *at all* yet, so there's
      no optimizer state for them to benefit from until that's built (full backprop
      through multi-head attention).
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

## Long‑Term Vision
- [ ] Compile critical parts to C extensions (optional) -- the single biggest pure-Python
      hot loop was the bootstrap-gate resampling (CHANGELOG #20, now ~5x cheaper by
      default, but still the natural first target if you profile a real run)
- [x] ~~Export trained meta‑parameters (PID gains, lasso masks) as a compact config~~ -- superseded: `checkpoint.py` saves these (and everything else) as part of a full checkpoint rather than a separate config export; see `Storage.save_checkpoint_blob` for a compact, named/versioned way to keep several.
- [ ] Interface with external LLMs as "teachers" for distillation (the teacher interface
      is just `.forward(token_ids) -> Matrix(seq_len, vocab_size)`, so any wrapper around
      an external model implementing that method should slot in directly)
- [ ] Publish as a research library with documentation and tutorials
