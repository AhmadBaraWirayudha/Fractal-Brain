# Changelog

## Full-codebase review of the v3 unification layer: five bugs fixed, testing/docs/CI gaps closed

Requested review of `hybrid_ai_pipeline_unified_v3` covering both correctness ("what's
wrong") and completeness ("what's insufficient"). `fractal_brain/` itself came out of
this clean (its own 137-check smoke suite still passes unchanged); everything below is
in the newer OCLE-plus-glue layer (`ai_pipeline.py`, `engine.py` and neighbors,
`ocle_clean_build/`), which this CHANGELOG had never covered before now.

### Bugs fixed

- **The final answer was always a canned template, regardless of what retrieval,
  decomposition, and planning had already found.** `SharedMoEBackbone._fallback_generate()`
  only ever read the `Task:` line back out of the assembled prompt and returned a fixed
  "Fallback backend active. / Essential plan: / - identify the structure / ..." string
  for every input -- verified live: asking it to solve `2x` from 0 to 4 returned that
  template even though memory had retrieved the exact answer ("...is 16."). Fixed by
  threading a `structured_context` dict (retrieved docs *with their similarity scores*,
  plan actions, subtasks) from `PlanConditionedDecoder.generate()` down into the
  backbone: when the best retrieved match scores at or above the new
  `model.retrieval_confidence_threshold` config field (default `0.5`), the answer now
  leads with that document's text and any `step_texts` metadata instead of discarding
  them; otherwise it lists the actual plan actions/subtasks instead of an
  input-independent constant. Still an honestly-labeled rule-based fallback, not a real
  model -- see `docs/UNIFIED_PIPELINE.md`. Regression test:
  `tests/test_pipeline.py::test_unified_pipeline_returns_trace` now asserts `'16'` is
  actually present in `final_output`, not just that it's non-empty.

- **Every `engine.initialize()` re-inserted the whole bootstrap dataset as fresh
  duplicate rows.** `VectorMemoryStore.bootstrap()` called `add_document()` (fresh
  `uuid4()` id, unconditional `INSERT`) for every record on every init, with no check
  for whether that content was already loaded. The `data/engine.db` previously shipped
  in this repo had accumulated 38 rows from a 4-record dataset (9 copies of each
  solution) purely from repeated development runs, and a single additional run visibly
  took it to 42. Fixed by deriving a stable id per bootstrap record
  (`bootstrap:<sha256(query+text)[:24]>`) and switching to `INSERT OR IGNORE`, checked
  via `cursor.rowcount` so the in-memory index doesn't duplicate the entry either.
  `data/engine.db` regenerates cleanly now (4 rows) and is no longer tracked (see
  "Housekeeping" below). Regression tests: `test_bootstrap_is_idempotent`,
  `test_repeated_retrieval_returns_distinct_documents`.

- **`ocle_clean_build`'s re-export shims could silently import themselves.** Files like
  `ocle_clean_build/engine.py` do `from engine import *` -- and because that submodule
  is named identically to the top-level `engine.py` it wraps, running Python with
  `ocle_clean_build/` itself as the working directory made a bare `import engine`
  resolve to the shim file instead of the real one, producing an empty module with no
  error (`ocle_clean_build/tests/` never actually exercised the package's own re-export
  path, only the root modules directly, so this had zero coverage). Fixed with a new
  `ocle_clean_build/_pathsetup.py`: each shim now inserts the project root at the front
  of `sys.path` before its wildcard import, and asserts the expected names actually
  landed afterward. Because the sys.path fix-up is imported *relatively*
  (`from ._pathsetup import ...`), the previously-silent failure mode now raises an
  immediate, clear `ImportError` ("attempted relative import with no known parent
  package") if this file is ever executed outside proper package context. Regression
  tests: `test_ocle_clean_build_reexport_matches_root_engine`,
  `test_ocle_clean_build_shim_fails_loudly_if_self_imported`.

- **The pipeline trace's `normalize` stage always showed the *previous* turn's text**
  (or `None` on the first turn). `UnifiedAIPipeline._trace_from_closed_loop()` read
  `self.state.last_normalized_text`, which wasn't updated until after the trace was
  already built. Fixed by passing the current turn's `normalized` value into the trace
  builder directly instead of reading stale state. Regression test:
  `test_pipeline_trace_normalize_stage_is_current_turn`.

- **The hand-rolled YAML parser truncated any value containing a literal `#`,** even
  inside quotes, because it split every line on the first `#` unconditionally.
  `engine.parse_simple_yaml()` now uses a new `_strip_comment()` that tracks single-
  and double-quote state and only treats `#` as a comment outside of both. Didn't
  currently affect `config.yaml`'s content, but was a live landmine for the next edit.
  Regression test: `test_yaml_parser_preserves_hash_inside_quotes`.

### Gaps addressed

- **CI never ran the pytest suite at all** -- `.github/workflows/ci.yml` only ran
  `python tests/test_smoke.py` (fractal_brain's own 137-check script). All 14
  pytest-based tests, including the only coverage that existed for the unified
  pipeline and for `ocle_clean_build`, never executed in CI. `ci.yml` now runs both.
- **The existing pipeline tests wouldn't have caught the canned-template bug even if
  CI had run them** -- they only checked shape (`assert payload['final_output']` is
  satisfied by any non-empty string). Strengthened with a content assertion (see
  above), and added `test_hybrid_cli_pipeline_mode_runs` since `hybrid_cli.py` itself
  -- the documented entry point -- previously had no test at all (`test_cli.py` only
  covered `python -m fractal_brain`).
- **The unified layer had zero documentation.** Added `docs/UNIFIED_PIPELINE.md`
  (component map, fallback-generator behavior, CLI usage, config reference, and an
  explicit "known limitations" section written in the same direct style as
  `ARCHITECTURE.md`'s own notes) and linked it and the existing `fractal_brain` docs
  from the README, which previously linked to neither.
- **`config.yaml`'s `model.model_name`/`quantization.*` fields implied real-model and
  quantization support that doesn't exist anywhere in this pure-Python build.** Rather
  than fake it, `SharedMoEBackbone.initialize()` and `build_quantization_settings()`
  now emit a `UserWarning` when these are set to anything other than the honest
  placeholder values (`fallback` / `none` / `auto`), so misconfiguration is visible
  instead of silently doing nothing.
- **The decomposer mislabeled anything outside its math/coding keyword lists as
  "engineering"** with the same four physics-flavored subtasks regardless of actual
  content, and `confidence` was a hardcoded `0.78`/`0.62` that didn't depend on the
  input at all. Added an explicit engineering keyword set (so "engineering" is now a
  real match, not an unconditional catch-all), a genuine `general` bucket for text
  matching none of the known domains, and confidence computed from how many keywords
  actually matched. `ai_pipeline.py`'s separate, cruder inline intent guess (used for
  FractalBrain's pre-generation cognitive context, and which didn't even consider
  "coding" as an option) was replaced with a direct call into this same classifier,
  removing a second, inconsistent copy of the same logic.
- **The planner's action-selection score couldn't discriminate between two competing
  actions from the same state** -- `_best()` summed each action's own next-state
  probability row, which is independently renormalized to ~1.0 for *any* action with
  data, so the comparison degenerated to near-ties broken by `action_vocab` order. The
  current 4-record bootstrap set happens not to expose this (every populated state has
  exactly one observed action), but a synthetic two-action test confirms the old
  scoring would pick a once-observed action over one seen 20 times. Fixed by scoring
  on the raw observed transition count instead of the normalized row sum.
- **`persistence_demo.py` wrote a 2.2MB db and a 1MB checkpoint into the repo root**
  with no `.gitignore` coverage for either -- the same class of issue that let the
  stale `data/engine.db` above accumulate and ship. The demo now writes under
  `var/persistence_demo/`, and `.gitignore` covers `*.db`/`*.sqlite`/`var/`.
- **`train_on_text.py` trained to the last epoch and generated from those weights**
  even though the printed curve shows validation loss rising well before training
  ends (expected on an 83-example corpus, but the final epoch is the *worst*
  validation point, not the best). It now keeps an in-memory snapshot
  (`fractal_brain.checkpoint.serialize_brain`, no disk I/O) whenever validation loss
  improves and restores that snapshot before final test-loss evaluation and
  generation. The full train/val curve is still printed in full -- the overfitting
  pattern is real and worth seeing, this just stops silently reporting metrics from
  the most-overfit checkpoint as if it were the model's best.
- **`SharedMoEBackbone._fallback_embedding` used Python's builtin `hash()`**, which is
  randomized per-process, so the same text could embed differently across separate
  runs. Currently dead code (nothing calls `encode_prompt()`), so no live behavior
  changed, but switched to the same `hashlib.sha256`-based approach
  `tokenizer.TextEmbedder` already uses correctly, removing the landmine.

### Housekeeping

- `data/engine.db` regenerated clean (4 rows, matching `bootstrap_dataset.jsonl`) and
  added to `.gitignore` -- it's a runtime artifact, not source, and shouldn't have
  been tracked in the first place.
- New `tests/test_regressions.py` holds one test per bug above, named after the bug it
  guards against, so a future change can't silently reintroduce any of them.


This responds to an external review's central critique directly: several subsystems
existed as modules but weren't actually part of the learning path -- specifically,
`_retrieve_and_fuse()`'s output was computed and discarded rather than affecting expert
selection, and JEPA's loss was computed and reported without ever training JEPA's own
weights. Both are fixed here, not just acknowledged.

- **RAG fusion is now a real, trained contribution to expert selection.** A new
  `W_rag_gate` weight projects `_retrieve_and_fuse()`'s output into gate space and adds
  it to the existing `raw_gate + lasso_gate + worm_gate` sum -- the same additive
  pattern the other three sources already use. Deliberately not left as an untrained
  random projection sitting next to trained ones (which would arguably be worse than
  not wiring it in at all -- random noise added to a decision, dressed up as
  "retrieval-augmented"): see the gate-gradient training below.

- **The entire gate is now trained via a real analytic gradient**, not just the
  expert output projections. Derived the standard softmax-mixture-of-experts gating
  gradient for this architecture's specific combination of four additive sources:

      d(CE)/d(gated_logits[i]) = (expert_weights[i] / temperature) * (s_i - s_bar)

  where `s_i = sum_v dz[v] * expert_outputs[i][v]` (how much increasing expert i's
  share of the mixture would increase the loss) and `s_bar` is its
  `expert_weights`-weighted average. `core.FractalBrain._compute_gate_gradients()` /
  `_apply_gate_gradients()` use this to train `moe.W_gate`, `tentacles.W`,
  `wormhole.W`/`wormhole.b`, and the new `W_rag_gate`, via a new `gate_optimizer` slot
  (defaults to plain SGD, same pattern as the other optimizer slots). Pruned experts'
  own gate weights correctly get exactly zero gradient (not just numerically
  negligible) -- `forward()` gives a masked-out expert a constant `-1e9` logit
  regardless of its own weights, so the true gradient there is zero, and the code
  checks the mask explicitly rather than relying on `expert_weights[i]` having
  underflowed to ~0.

  This coexists with BCM's existing Hebbian updates on `tentacles.W`/`moe.W_gate` --
  both a task-driven (gradient) and an activity-driven (Hebbian) learning signal now
  act on those same two matrices, which is an unusual but not incoherent design (not
  unlike architectures that combine Hebbian and error-driven learning); noted rather
  than silently allowed to look like an oversight.

  **How this was verified is worth detailing, because the first verification attempt
  failed in a misleading way.** Numerically gradient-checking the analytic formula
  against finite differences on an actual `FractalBrain.forward()` call gave wildly
  erratic results -- errors in the hundreds, for a formula that should be accurate to
  ~1e-4. That's not what a small bug looks like; it's what a broken *test* looks like.
  The cause: `BootstrapGate` (part of the fractal Markov chain) keeps a `history` list
  that grows across calls, and the test's state snapshot/restore (markov chain
  position, PID state, RNG state) didn't include it -- so each finite-difference probe
  saw a different accumulated bootstrap history, unrelated to the weight perturbation
  being measured, and the "numerical gradient" was mostly noise from that. Fixed by
  testing the pure math in isolation instead: fixed synthetic `expert_outputs`, gate
  query, state vector, and fused state (no stochastic `FractalBrain` internals
  involved at all), computing `expert_weights`/loss as a direct function of the gate
  weight matrices. Against that, the analytic and numerical gradients agree to ~1e-11
  (float64 precision) across all four weight sources and the bias vector, including
  confirming the pruned-expert-zero-gradient property. Also verified end-to-end: all
  four gate sources (`W_rag_gate`, `moe.W_gate`, `wormhole.W`, `jepa.Wc` -- see below)
  demonstrably change during real training (not just theoretically differentiable),
  task loss still drops substantially with gate training active, and `train_batch()`'s
  gate-gradient averaging (across four sources, one of which -- `wormhole.b` -- is a
  Vector rather than a Matrix, needing separate accumulation logic) matches an
  independent manual replication exactly.

  Added `Adam`/`SGD.step_vector()` for this (a Vector-shaped parameter --
  `wormhole.b` -- needed the same momentum/Adam machinery as `step_matrix`, just for
  a 1D shape).

- **JEPA's own encoder and predictor are now actually trained.** `JEPA.train_step()`
  runs the forward pass with cached intermediates, then exact analytic backprop
  through the predictor and the context encoder (two chained 2-layer ReLU-MLP
  backprop derivations -- textbook, but verified anyway: see below), and applies the
  result via a new `jepa_optimizer` slot. The target encoder (`Wt`/`Wt2`) is
  deliberately *not* backpropagated into -- that's the defining feature of a
  JEPA/BYOL-style target network, a slowly-moving copy of the context encoder updated
  only by `update_ema()` (now actually called, at the end of every `train_step()` --
  previously implemented correctly but never invoked at all, so the target encoder
  was permanently frozen at its random initial values no matter how much JEPA "trained").

  Verified by numerical gradient checking against finite differences on all four
  weight matrices (`Wc`, `Wc2`, `Wp1`, `Wp2`) -- agreement to ~1e-4, and this
  derivation didn't need the isolation workaround above, since JEPA's own forward pass
  has no stochastic internal state to worry about. Also verified: `train_step()` drops
  JEPA's loss by >99% over 150 steps on fixed synthetic inputs; the target encoder
  visibly drifts toward the context encoder via EMA (not a hard copy, and not frozen);
  and wired into a real `FractalBrain`, JEPA's own weights change and its reported
  loss drops by more than half over 120 steps of ordinary training -- the loss that
  was already being computed and added to `total_loss` every step is now actually
  being minimized, not just reported.

- Both new training paths are wired into `train_batch()` as well as `step()`. Gate
  gradients accumulate across a batch the same way output-projection gradients do
  (average, then one optimizer step); JEPA trains per-example within a batch, the same
  as BCM, since `train_step()` is a complete, self-contained update rather than a
  compute/apply-split gradient -- there's nothing to accumulate across examples the
  way there is for the others.

- **Honest note on backward compatibility, since this one is different from the
  optimizer refactor two rounds ago:** that refactor changed only *how* an existing
  gradient got applied, and was verified bit-identical in the ways that matter. This
  change alters the forward pass itself -- gating now has a fourth additive input that
  didn't exist before -- so loss values differ from prior runs starting on the very
  first `step()` call, before any training happens at all, not just after many steps
  of accumulated drift. `how_to_use.py`'s single-step loss legitimately moved from
  `8.252234047434488` to `8.389949004136914`; this is not a bug, and no amount of
  further verification will make it match the old number, because the model
  genuinely computes something different now. Every existing threshold-based test
  (loss decreases by at least X%, accuracy reaches at least N/M) still passes --
  checked, not assumed -- since those don't depend on any specific value, only on the
  qualitative behavior, which is unchanged or improved.

Verified: 14 new checks in `tests/test_smoke.py` (137 total, all passing), plus the
cross-process checkpoint-fidelity test from two rounds ago was re-run against this
much larger training surface and is still bit-identical.

## A real optimizer, and batching (new capability)

This is new capability, not a bug fix -- covering "A proper optimizer" and "Batching"
from `To-Do.md` (items 5-6 of its suggested build order).

- **`optimizer.py`**: `SGD` (with optional momentum and weight decay) and `Adam`
  (Kingma & Ba, with bias correction), plus `clip_grad_norm_matrix` and four learning
  rate schedules (`ConstantLR`, `StepLR`, `CosineAnnealingLR`, `LinearWarmupLR`). Each
  optimizer keeps per-parameter momentum/moment-estimate state keyed by a caller-
  supplied string (an expert's `W_out`, a PID gain, ...), since this codebase's
  parameters are specific named things, not a flat list the way a framework's
  `.parameters()` would return.

  `core.py`'s two training methods (`_update_expert_output_layers`,
  `_meta_update_pid_gains` -- introduced two rounds ago) previously computed a
  gradient and immediately applied it at a fixed rate in the same breath: no momentum,
  no adaptive per-parameter rates, no weight decay, no schedule, no clipping, exactly
  as this To-Do item said. Both are now split into a pure `_compute_*_gradients()`
  (returns the raw gradient, doesn't touch any weights) and a separate
  `_apply_*_gradients()` (hands it to `self.output_optimizer` / `self.pid_optimizer`).
  This split is also what makes batching possible (below): computing several examples'
  gradients before deciding how to combine them requires them to exist as values you
  can hold onto, not side effects that already happened.

  Backward compatibility: `FractalBrain`'s constructor defaults
  (`output_optimizer=None`, `pid_optimizer=None`) construct a plain `SGD` at the exact
  same learning rates the old hard-coded update used, so existing code needs no
  changes to keep working. One honest caveat, which I initially misdiagnosed and want
  to correct rather than leave wrong: separating "compute the gradient" from "apply the
  update" changes the order grouping of the underlying floating-point multiplications
  (e.g. `(lr*w_i)*x*dz` before, vs `lr*(w_i*x*dz)` now). My first assumption was that
  this explained a `0.2508` -> `0.2507` shift I'd noticed in `train_on_text.py`'s final
  epoch. It doesn't: directly reconstructing the old inline computation and diffing it
  against the new split version, step by step, over up to 1625 pure per-example
  `step()` calls with no batching involved, the two never differ by more than
  ~1e-15 -- literally float64 epsilon, invisible at any display precision anyone would
  use, and `orchestrator.py`'s unrelated 200-step run (still per-example `step()`,
  untouched by any of this) shows zero drift at all for exactly that reason. The
  `0.2508`/`0.2507` difference is real but has a different, mundane cause: I also
  switched `train_on_text.py`'s training loop from one `step()` call per example to
  `train_batch()` (see below) in the same pass, and averaging several examples'
  gradients before one update is a genuinely different computation from applying them
  one at a time -- not a bug, not float noise, just a different (and intentional)
  training dynamic. Both explanations are recorded here rather than just the corrected
  one, since "I checked and my first explanation was wrong" is more useful to know than
  a clean story that skips it. What I did verify and can stand behind: the cross-process
  checkpoint-fidelity test from the previous round (train straight through vs.
  save/reload-and-continue in a separate process) was re-run after this refactor and is
  still bit-identical.

  Also verified: `SGD` with momentum converges faster than plain `SGD` on the same
  task and produces a genuinely different trajectory (not just a relabeled copy);
  `Adam` converges to the analytic minimum of a simple convex test function from both
  its matrix and scalar code paths; `clip_grad_norm_matrix` keeps an artificially huge
  learning rate (50.0) from blowing up a real training run; `StepLR` changes
  `output_optimizer.lr` at the exact expected step boundaries; and a checkpoint
  correctly round-trips `Adam`'s accumulated moment estimates *and* its per-parameter
  step counters (so bias correction resumes from the right `t`, not `t=0`) -- which
  required registering `SGD`/`Adam`/the four schedule classes in `checkpoint.py`'s
  registry, the same way any new stateful class needs to be.

- **`FractalBrain.train_batch(batch)`**: train on several `(token_ids,
  target_distribution)` examples at once, averaging their output-projection and PID
  gradients before taking *one* optimizer step each, rather than reacting to every
  example individually (what mini-batch SGD has always bought you). Sequences in a
  batch may be different lengths -- deliberately no padding: nothing here needs a
  single stacked tensor, so each example still gets its own independent forward pass
  (advancing the fractal Markov chain / delay line / RAG index once per example,
  exactly as repeated `step()` calls would). What batching buys you here is the
  averaging itself, not a vectorized speedup -- pure Python doesn't have one of those
  to offer regardless of batch size. BCM plasticity stays per-example (it's a local
  Hebbian rule reacting to that example's own activity, not a loss gradient -- "batching"
  it wouldn't mean the same thing), and `step_count`/pruning still advance once per
  example.

  Verified the averaging itself is exactly correct, not just "runs without crashing":
  independently replicated `train_batch()`'s forward-pass-plus-gradient-plus-BCM logic
  by hand for a mixed batch (including one example with no target and two different
  sequence lengths) and confirmed the resulting `W_out` matrices match to the last
  float. This took two attempts -- the first manual replication skipped BCM's
  per-example update to keep the comparison "simple", and quietly failed, because BCM
  updates `moe.W_gate` after every example, which changes the *next* example's gate
  weights within the same batch. Once the replication matched the real code path
  exactly (BCM included), it matched. Also verified: real learning through
  `train_batch()` alone reaches 10/12 or better on a small memorizable task (separately
  from the per-example `step()` version verified two rounds ago), and the accumulated
  gradient is what actually gets applied (not silently dropped or double-applied) via
  the same manual-replication check.

- **`dataset.TextDataset.batches(batch_size, shuffle=False, seed=None,
  drop_last=False)`** (and the same method on `DatasetView`, i.e. also available on a
  `.split()` result): chunks a dataset into lists of `(context, target)` pairs sized
  for `train_batch()`, with optional reproducible shuffling.

- **`train_on_text.py`** now trains via `train.batches(...)` + `brain.train_batch(...)`
  instead of one `step()` call per example, and uses `Adam` instead of the default
  plain SGD, to actually show the new pieces being used together rather than just
  existing unused. Loss curves show the same overfitting signature as before (expected
  on a corpus this small -- see the previous round's entry); generation is noticeably
  more coherent with Adam, though that's a side effect of trying it here, not a claim
  that Adam is definitively better for this architecture.

Verified: 28 new checks in `tests/test_smoke.py` (123 total, all passing).

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
