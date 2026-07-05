# Changelog

## Persistence: checkpointing and SQLite-backed storage (new capability)

This is new capability, not a bug fix -- covering the "Storage schema" and "Checkpoint
save/load" items from `To-Do.md` (items 3-4 of its suggested build order), and the
weights/training-state half of "Model serialization".

- **`checkpoint.py`**: save/load a full `FractalBrain` -- every weight matrix, the PID
  controller's gains *and* internal integral/derivative state, the fractal Markov
  chain's current states, the lasso mask, `step_count`, the RAG vector store's contents,
  the embedding delay line's buffer, everything -- to/from JSON, with no external
  dependencies. Rather than hand-writing a bespoke serializer for each of the ~15
  classes involved (a large surface that would drift out of sync the moment any of them
  gained, lost, or renamed an attribute), it walks the object graph generically:
  `Matrix`/`Vector` are special-cased, plain values and containers recurse, and any
  other object is reconstructed via `object.__new__(cls)` + restoring `__dict__`
  directly (bypassing `__init__` entirely) -- safe here because none of this project's
  classes do anything in `__init__` beyond building sub-objects and assigning
  attributes, which I confirmed by reading every `__init__` rather than assuming it.
  Unregistered custom classes raise a clear `TypeError` rather than failing silently or
  corrupting the save.

  Verified two ways. First, the obvious way: save, reload into a fresh instance, and
  check every weight/PID-gain/counter matches (it does). Second, the way that actually
  matters: does resuming training after a reload reproduce what an *uninterrupted* run
  would have done, bit-for-bit? This requires capturing Python's *global* `random`
  module state too, not just the brain's own attributes -- `BootstrapGate` (the fractal
  Markov chain's novelty gate) draws from it directly rather than from a per-instance
  RNG, which I only discovered by writing this test and watching a naive version of it
  fail. With that fixed, I ran two genuinely separate OS processes (not just two Python
  objects in one process, which would share that same global state unfairly) -- one
  training straight through, one saving a checkpoint partway and reloading it in a fresh
  process to continue -- and diffed the resulting losses: identical, all 40
  post-checkpoint steps, to the last floating-point digit.

  `FractalBrain.teacher` is deliberately *not* saved: it's an arbitrary,
  externally-injected object (e.g. `orchestrator.FrozenTeacherExpert` isn't even part of
  this package), so there's no generally-correct way to reconstruct it.
  `save_checkpoint()` warns if one is attached; reattach it manually after loading.

- **`storage.py`**: a thin SQLite wrapper (`sqlite3` is standard library, so this
  doesn't compromise the zero-dependency design) implementing the exact schema
  suggested in `To-Do.md`: `vocab`, `samples`, `documents` (RAG persistence, embeddings
  as BLOB), `checkpoints` (versioned blobs, e.g. from `checkpoint.serialize_brain`), and
  `metrics`. Also includes a generic `memory` key-value table for anything else (config,
  experiment notes) that didn't need its own table.

- **`persistence_demo.py`**: a new end-to-end example -- train a tokenizer and dataset,
  persist both to SQLite, train while logging metrics and periodically checkpointing
  (both as a standalone file and as versioned blobs inside the same database), reload a
  standalone checkpoint into a fresh instance and continue training, and pull a specific
  earlier checkpoint back out of the database by name.

  One thing this script's first draft got wrong, worth recording: it originally compared
  `original_brain.sample(...)` against `reloaded_brain.sample(...)` right after loading
  and reported them as different, which -- read carelessly -- looks like a checkpoint
  bug. It isn't one: both calls draw on that same shared global random state mentioned
  above, and calling one first advances the shared state before the other runs. Two live
  `FractalBrain` instances in one process are simply not independent that way, with or
  without checkpointing involved. The demo now compares static attributes (weights, PID
  gains) instead, which is the fair comparison, and says so in a comment; the stochastic
  part is what the two-separate-processes test above actually verifies.

Verified: 28 new checks in `tests/test_smoke.py` (95 total, all passing), plus the
cross-process checkpoint-fidelity test described above (not itself part of the unit
test suite, since it needs two separate process invocations to be a fair test -- see
this section).

## Data pipeline: tokenizer, dataset, and a read-only `evaluate()` (new capability)

This is new capability, not a bug fix -- covering the "Tokenizer", "Vocabulary
builder", "Dataset loader", "Text cleaning / normalization", and "Train / validation /
test split" items from `To-Do.md`'s data-pipeline section (items 1-2 of its suggested
build order).

- **`tokenizer.BPETokenizer`**: a real byte-pair-encoding tokenizer, trained from raw
  text with no external dependencies -- not a placeholder. Learns merges from a corpus,
  encodes text to token ids and decodes back, with save/load to JSON. Two limitations
  are documented rather than glossed over: it splits words into Unicode code points (not
  grapheme clusters), and `decode()` is an *approximate* detokenizer (a simple heuristic
  reconstructs spacing around punctuation) rather than an exact inverse of `encode()` --
  a fully exact round trip would need whitespace itself folded into the vocabulary
  (byte-level BPE, as in GPT-2), which is a larger undertaking than warranted here.
- **`dataset.TextDataset`**: turns a tokenizer + raw text into sliding-window
  `(context_token_ids, target_one_hot)` next-token-prediction examples, with
  `.split(train_frac, val_frac, seed)` for a shuffled, non-overlapping train/val/test
  split (verified: `tests/test_smoke.py` checks the three splits never share an index).
  Deliberately produces one example at a time rather than batching -- `FractalBrain.step()`
  itself only accepts one sequence per call today, so batching is out of scope until that
  changes too (see `To-Do.md`'s "Batching" item, still open).
- **`core.FractalBrain.evaluate()`**: while wiring up a demo that trains on `train` and
  reports loss on `val`/`test`, I found that doing the obvious thing -- calling `step()`
  on the validation examples too -- would silently *train* on them, since `step()` has no
  read-only mode. `evaluate()` fixes that: the identical forward pass and loss
  computation as `step()` (both now share a `_compute_losses()` helper, extracted from
  `step()` so the two can't drift apart on what "the loss" means), but without applying
  the output-projection gradient step, the PID meta-update, BCM plasticity, or pruning.
  It deliberately still advances the fractal Markov chain's internal state and the
  embedding delay line, same as `step()` does -- those are the model's ongoing internal
  dynamics, not something being fit to whatever you evaluate on, so freezing them seemed
  more wrong than right; see the method's own docstring for the full reasoning. This is a
  partial answer to the "Evaluation loop" item in the suggested build order -- the core
  read-only primitive exists now, a fuller harness (aggregate metrics, perplexity, etc.)
  is still open.
- **`train_on_text.py`**: a new end-to-end example (tokenizer -> dataset -> train/val
  split -> training -> greedy generation) demonstrating all of the above together. Worth
  noting what it actually shows: on the tiny demo corpus, train_loss drops from ~4.4 to
  ~0.25 over 25 epochs while val_loss *rises* from ~4.5 to ~5.4 -- textbook overfitting on
  a too-small corpus, and also a good sign that `evaluate()` genuinely isn't leaking
  training signal into the validation numbers (if it were, val_loss would track
  train_loss downward instead of diverging from it).

Verified: 17 new checks in `tests/test_smoke.py` (67 total, all passing), plus manual
confirmation that `how_to_use.py` and `orchestrator.py` produce byte-identical loss
values to before this round's `core.py` refactor (i.e. `step()`'s behavior is unchanged;
only `evaluate()` and the internal `_compute_losses()` extraction are new).

---

# Original fixes

This describes what changed from the version you originally uploaded, and why.
Everything below was verified empirically (not just read and reasoned about) via
`tests/test_smoke.py` and the extended training runs described inline.

The short version: **the library could not be imported at all** in either of the two
ways it's meant to be used (see below), and once that's fixed, two of the architecture's
headline mechanisms (PID-controlled gating, and pruning) turned out to be complete
no-ops even though the surrounding code ran without error. Everything here is fixed,
plus the library now actually trains (previously nothing but two small weight matrices
ever received any learning signal at all).

## Crash bugs (the library did not run, as shipped)

1. **Every submodule except `__init__.py` used absolute imports** (e.g. `from math_utils
   import ...`) instead of relative ones (`from .math_utils import ...`). This meant the
   package raised `ModuleNotFoundError` the moment you tried to use it the way the README
   documents (`from fractal_brain import FractalBrain` from a parent directory) -- and it
   *also* failed if you ran a script from inside the `fractal_brain/` folder itself (there's
   no nested `fractal_brain` package inside itself). I checked both invocations before
   changing anything; both failed. Fixed in every file that had it: `markov.py`,
   `tentacles.py`, `attention.py`, `moe.py`, `rag.py`, `synaptic.py`, `core.py`,
   `wormhole.py`, `jepa.py`, `distillation.py`, `logic_folding.py`, `dim_reduction.py`,
   `recursive_matrices.py`, `signal.py`, `turbo_quant.py`.

   One variant is worth calling out: `core.py` had `from signal import DelayLine,
   convolve1d`. `signal` is also a real standard-library module name, so depending on
   `sys.path`, this absolute import could silently resolve to Python's *actual* built-in
   `signal` module (OS signal handling) instead of your local file -- a much more
   confusing failure than a clean `ModuleNotFoundError`.

2. **`markov.BootstrapGate.should_transition()` called a bare `random()`**, but the file
   only ever imported `random` under the alias `_random`. This didn't fail immediately --
   the buggy branch only runs once a node's history exceeds 10 entries -- so it silently
   worked for the first few calls per node and then threw `NameError` partway through any
   real run. Fixed (and cleaned up the import to a normal top-of-file `import random`).

3. **A systematic shape bug: weight matrices built as `(in_dim, out_dim)` were multiplied
   with `Matrix.dot_vector()`**, which implements the *other* convention (`self @ vec`,
   requiring an `(out_dim, in_dim)` matrix). Wherever `in_dim != out_dim` -- which is
   almost everywhere, including literally the config in your own `how_to_use.py` /
   README.md walkthrough -- this raised `AssertionError` the moment the code path ran.
   Where the matrix happened to be square (attention's Q/K/V/O, RAG's Q/K/V/O), it didn't
   crash, but it silently computed `W @ x` instead of the `x @ W` the surrounding comments
   describe. Affected: `tentacles.py`, `moe.py` (gate), `wormhole.py`, `jepa.py` (3
   methods, 6 call sites), `rag.py` (4 call sites), `core.py` (state projection + gate).

   Fix: added `Matrix.linear(vec)`, implementing the correct `vec @ self` convention, and
   switched every affected call site to it. Left `dot_vector` itself untouched --
   `dim_reduction.py`'s power-iteration/SVD code genuinely needs the classical `A @ v`
   convention for an `(m, n)` data matrix, and was already using it correctly.

4. **`orchestrator.py` and `how to use.py` were shipped *inside* `fractal_brain/`**, but
   both import the package by its top-level name, which cannot work from inside the
   package's own folder, and doesn't match the layout your own README describes either.
   Moved both to the project root (siblings of `fractal_brain/`); renamed `how to use.py`
   -> `how_to_use.py` (a space in a filename isn't importable and is awkward to invoke).

5. **`orchestrator.Runner._inspect()` referenced `Vector(...)` without ever importing
   it** -- guaranteed `NameError` after the first epoch. Fixed.

6. **`orchestrator.py` unpacked `TurboQuant.quantize_matrix()` as a 3-tuple**, but it
   actually returns 4 values (`shape` is required by `dequantize_matrix`) --
   guaranteed `ValueError`. Fixed the call site and the docstring, which had drifted out
   of sync with the actual `return` statement.

## Bugs that ran, but silently did nothing (or the wrong thing)

7. **The PID controller had zero effect on the model, at all.** `pid_correction` was
   added as the *same constant* to every expert's gate logit before a softmax. Softmax is
   exactly invariant to a constant shift applied to all of its inputs, so this could never
   change which experts got selected regardless of the PID gains -- the "PID control for
   stable expert gating" headline feature was a no-op. Fixed by having `pid_correction`
   modulate the gate's *temperature* instead (dividing the logits, via a smooth squashing
   function -- see #9). This happens to be exactly the "adaptive temperature in softmax
   based on PID" item from your Medium-Term list, so that's checked off now too.

8. **Pruning didn't actually prune.** The lasso mask multiplied a pruned expert's gate
   score by 0 -- but 0 isn't necessarily a *low* score. If a surviving expert's own score
   was negative, the "pruned" expert (sitting at a neutral 0) could still out-compete it
   in the softmax. Fixed by giving masked-out experts a large negative logit instead, so
   they're genuinely excluded regardless of the other experts' score signs. Fixed both in
   `core.py` (the path `FractalBrain` actually uses) and in `moe.GatedMoE.forward` (not
   currently called by `FractalBrain`, but a public method with the identical bug).

9. **The PID gain "meta-learning" block was dead code**: it perturbed `self.pid_Kp.data`
   by a fixed delta and then immediately restored it, without recomputing anything in
   between -- and the perturbed copy was a separate `Value`-wrapped duplicate the real
   `PIDController` never read from anyway. Replaced with a real central-difference
   estimate of d(loss)/d(Kp, Ki, Kd), made cheap by reusing the already-computed
   (expensive) per-expert transformer outputs and only recomputing the (cheap) gating +
   recombination + cross-entropy per probe. This needed `PIDController` to expose a
   side-effect-free `compute_output(...)` probe plus cached `last_error` /
   `last_integral` / `last_derivative`, so gains can be probed without corrupting the
   controller's real integral / `prev_error` history.

   Wiring this up surfaced a second bug: the temperature mapping in #7 originally used a
   hard `max(0.1, min(10.0, ...))` clamp. A hard clamp has *exactly* zero gradient once
   saturated, and a KL-divergence error of 15-25 (routine for an early-training model
   against a random one-hot target) times Kp≈0.8 alone saturates it almost immediately --
   silently zeroing the new meta-gradient right back out. Replaced with a smooth logistic
   squash, which has a small but genuinely nonzero gradient everywhere. (I only caught
   this because I ran a 150-step training loop and printed the gains every 10 steps and
   noticed they weren't moving at all -- worth knowing in case you touch this code again:
   trust the printed numbers over the algebra when clamps are involved.)

10. **JEPA's embeddings blew up by several orders of magnitude.** `Matrix.random()` draws
    from U[0,1) -- not zero-centered, not scaled by fan-in. Every *other* module that
    stacks linear layers (attention, the transformer experts) follows each one with
    LayerNorm, which resets the activation scale regardless of init -- invisible there.
    JEPA's encoder/predictor (two unnormalized linear layers, twice) has no such reset, so
    the effect compounds: in one test, a `~0.5`-scale input reached magnitude ~540 after
    the encoder and ~284,000 after the predictor, producing a JEPA loss around 10^12 that
    swamped the total loss entirely. (Invisible before fix #3, since the shape-mismatch
    crash always fired first; this is a good example of one bug hiding a second one.)
    Fixed by adding `Matrix.he_init(in_dim, out_dim)` (zero-centered Gaussian, scaled by
    `1/sqrt(in_dim)`) and switching every neural-net weight matrix in the library to it --
    JEPA's six matrices, attention's four projections + two FFN weights, MoE's embedding
    table / output projection / gate, tentacles' weight, wormhole's weight, RAG fusion's
    four projections, and `core.py`'s state projection. Biases (previously also
    `Vector.random()`, i.e. U[0,1)) now zero-initialize, which is standard practice.

11. **`recursive_matrices.FractalMatrix`'s leaf didn't match its own size accounting.**
    `self.size = base_size ** depth` means a depth-0 node should be 1x1, but the leaf was
    built as a full `base_size x base_size` random matrix. Every level above a leaf only
    ever reads index `[0][0]` of it (the modulo arithmetic collapses to 0 one level up),
    so `base_size**2 - 1` random values were silently generated and never read. Fixed by
    making the leaf a proper 1x1 matrix, consistent with `self.size`.

12. **`signal.DelayLine`'s buffer was seeded with float `0.0` placeholders**, but
    `core.py` pushes `Vector` objects into it. For the first `max_delay` calls, `push()`
    returned a bare `0.0` instead of a `Vector` -- which "worked" only because `core.py`'s
    `if delayed_mean:` treats `0.0` as falsy and skips the (type-mismatched) branch,
    entirely by coincidence. Fixed by seeding with `None` and checking `is not None`
    explicitly.

## Smaller correctness / robustness / clarity fixes

13. `autograd.Value` was missing `__rsub__` and `__rtruediv__` (only `__radd__` /
    `__rmul__` existed), so `1 - some_value` or `10 / some_value` raised `TypeError`.
    Added both.
14. `distillation.py` computed `math.log(p)` on a raw softmax output with no floor,
    risking a rare `ValueError: math domain error`. Guarded with `max(p, 1e-12)`.
15. `core.py step()` computed `probs` inside one `if token_ids and target_distribution:`
    block and read it again inside a second, separately-written copy of the same
    condition a few lines later. They happen to always agree today, but nothing enforces
    that. Combined into a single `has_target` check computed once.
16. `turbo_quant.py`'s docstring claimed "symmetric" quantization; it's actually
    affine/min-max (it stores a `min_val` offset, which symmetric quantization doesn't
    use by definition). Docstring corrected.
17. `pid.PIDController.step()` divided by `dt` with no guard; `dt=0` raised
    `ZeroDivisionError`. Guarded. Also clarified the previously-ambiguous docstring
    around how `error` and `setpoint` combine.
18. `tentacles.LassoTentacles.__init__`'s first parameter was named `num_markov_nodes`,
    but callers actually pass the full concatenated state-vector dimension
    (`num_markov_nodes * markov_states`), not the raw node count. Renamed to `input_dim`.
19. `synaptic.BCMPlasticity.update()` divided by `num_post` with no guard against a
    zero-width weight matrix. Guarded (cheap insurance; not reachable with any shipped
    config today).
20. `markov.BootstrapGate`'s default `n_bootstrap=100` meant every one of the
    `num_markov_nodes * (max_level+1)` gates resampled its history 100 times on *every*
    `step()` call -- by far the biggest hot loop in the library (tens of seconds of pure
    overhead per 1,000 steps at default settings, by rough measurement). Lowered the
    default to 20 (~5x speedup on this specific bottleneck); it's a heuristic novelty
    gate, not a place that needs that much precision. Still fully overridable per
    instance.
21. Documentation only: `kl_divergence`'s docstring didn't make clear that passing raw
    logits instead of true log-probabilities silently returns a wrong number rather than
    raising -- the internal renormalization only protects against float drift in an
    *already*-normalized input, not against the wrong input type. Every existing call
    site already complies with the correct contract; this is a docstring fix only.

## New capability (previously flagged in your To-Do as missing)

22. **Real gradient descent for each expert's output projection.** Before this, none of
    the ~9 weight matrices per expert (embeddings, attention, feed-forward, output
    projection) were updated by anything resembling gradient descent -- only
    `tentacles.W` and `moe.W_gate` got any learning signal at all, via BCM plasticity.
    Added an exact analytic softmax-cross-entropy gradient step for each expert's `W_out`,
    scaled by that expert's current gate weight (so an expert the gate barely uses gets
    barely any update -- correct MoE behavior, and it composes with the pruning fix: a
    pruned expert has weight ~0 and is skipped). Needed caching each expert's
    pre-projection hidden state as a new, backward-compatible side effect of
    `TransformerExpert.forward()`. Verified end-to-end: on a small memorizable task, loss
    drops from ~2.6 to ~0.03 over 300 steps and the model reaches 8/8 accuracy (see
    `tests/test_smoke.py`).

    Everything upstream of `W_out` (attention, feed-forward, embeddings) is still a fixed
    random projection -- deliberately the "random features" end of the spectrum, not full
    backprop through attention. That's still future work (see To-Do.md), but it's now
    much better scoped, since the hidden-state caching groundwork is already there.

23. **Real meta-gradient descent for the PID gains** -- see #9 above.

24. **A real distillation teacher.** `orchestrator.DummyTeacher` returned fresh,
    independent random noise on *every single call* -- not even a consistent function of
    its own input, so it had no way to teach anything. Replaced with
    `FrozenTeacherExpert`: a real (if small and untrained) `TransformerExpert` whose
    weights are fixed at construction and never updated -- at least giving consistent,
    input-dependent soft targets. Still not a pretrained model, but it behaves like a
    teacher is supposed to; swap in a real checkpoint for actual distillation.

## What I deliberately did *not* change

- **JEPA's own encoder/predictor weights are still never trained.** The JEPA loss is
  computed and reported every step, and `JEPA.update_ema()` exists, but nothing calls it,
  so the target encoder never moves toward the context encoder -- and the context encoder
  itself has no gradient step either. This is a pre-existing gap, not something I
  introduced; I fixed the *numerical* bug (the blow-up in #10) but stopped short of
  wiring up real training for JEPA's own weights, since that's a similarly-sized chunk of
  new backprop math to what I already added for the output projections and PID gains, and
  I wanted to keep the amount of *new* untested machinery in one pass bounded. Noted in
  To-Do.md.
- **`_retrieve_and_fuse`'s output (`fused_state`) is computed every step but not consumed
  anywhere** -- it exercises the RAG/cross-attention path but doesn't feed into the gate
  or the logits. This was already true before my changes; I didn't invent a new
  integration point for it, since where it should plug in (the gate? the logits directly?
  a wormhole-style addition?) is a design decision, not a bug fix. Noted in To-Do.md.
- I did not attempt full backprop through attention/feed-forward layers. That's a
  substantially larger undertaking than anything else here, and the existing To-Do
  already correctly scopes it as such.

## Testing

Added `tests/test_smoke.py`: a dependency-free suite (50 checks, runs in a few seconds)
exercising every module at least once, with a specific regression test for each bug
above. Run with `python tests/test_smoke.py` from the project root. Also manually
verified (not just unit-tested): the exact config from your `how_to_use.py` runs
end-to-end; a 150-step run with `num_layers=2` produces no NaN/Inf and finite, non-negative,
adapting PID gains; `orchestrator.py`'s full training loop (including the new frozen
teacher, TurboQuant round-trip, and PCA) runs to completion.
