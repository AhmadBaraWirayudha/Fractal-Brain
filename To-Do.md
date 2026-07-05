# To‑Do & Future Enhancements

> See `CHANGELOG.md` for the full list of bugs fixed and why. This file tracks what's
> left, not what changed.

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
- [x] Unit tests for each module (`tests/test_smoke.py`, 50 checks)

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
- [ ] Export trained meta‑parameters (PID gains, lasso masks) as a compact config
- [ ] Interface with external LLMs as "teachers" for distillation (the teacher interface
      is just `.forward(token_ids) -> Matrix(seq_len, vocab_size)`, so any wrapper around
      an external model implementing that method should slot in directly)
- [ ] Publish as a research library with documentation and tutorials
