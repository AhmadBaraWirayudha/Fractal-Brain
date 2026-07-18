# Adaptive hyperparameter optimizer

`adaptive_optimizer.py` is a large (2900+ line), independently-authored,
pure-stdlib hyperparameter optimization engine: 21+ search strategies, a
meta-controller that switches between them, Bayesian belief tracking per
parameter, Gaussian-process-based acquisition, constraints, multi-objective
scoring, early stopping, and a full experiment-tracking data model (trials,
context, telemetry). Its own module docstring describes the architecture in
detail; this doc covers `pipeline_optimizer.py`, the integration layer that
connects it to *this* pipeline, and what was found and fixed while building
that connection.

## Two files, two jobs

- **`adaptive_optimizer.py`** -- the library, merged in close to as-is.
  Verified against a toy quadratic objective (converged to within ~0.06 of
  the true optimum in 40 trials) and a hard-constraint scenario (correctly
  rejected out-of-bounds proposals) before any integration work started.
- **`pipeline_optimizer.py`** -- everything specific to wiring it into this
  codebase: adapters bridging its dependency-injection points to the real
  `lkg.py` graph and `fractal_brain` storage, a parameter registry mapping to
  actual `config.yaml` values, and an objective function that scores a
  candidate configuration by really running the pipeline.

## Why adapters, not passing the real objects in directly

`adaptive_optimizer.LivingKnowledgeGraphInterface` requires `add_node`/
`add_edge`/`record_trial`/`get_parameter_correlations`; `lkg.
LivingKnowledgeGraph` has none of those (it speaks facts/entities/confidence,
not nodes/edges/trials) -- passing a real `lkg.LivingKnowledgeGraph` in
directly would satisfy nothing and fail at the first call. Grepping the
whole 2900-line file confirmed `record_trial` is the only one of those four
`AdaptiveOptimizer` ever calls internally; `add_node`/`add_edge`/
`get_parameter_correlations` exist for external callers.

`PipelineKnowledgeGraphAdapter` therefore delegates the bookkeeping
`adaptive_optimizer.py` actually needs (nodes, edges, Pearson correlations
across trial history) to one of its own default `LivingKnowledgeGraph`
instances -- reusing that rather than reimplementing it -- and additionally
pushes one real, repeatedly-observable fact into the shared `lkg.py` graph
per completed trial: whether this optimization run is still improving on its
own recent average (`(source, "trial_improves_on_recent_average", "true")`,
`positive=` whether the trial beat the trailing mean of the last 10
completed trials). That's a deliberate choice: `TrialResult` doesn't carry a
search-strategy name, so a per-strategy reliability fact (the first design
tried, mirroring `matched_intent`'s per-source tracking) wasn't available,
and a per-trial-id fact would only ever get one observation each -- Beta
confidence tracking can't build a signal from a subject it never sees twice.
The aggregate "still improving" fact does get observed repeatedly and shares
the same graph instance the engine's `session_intent`/`matched_intent`
tracking uses, per `lkg.py`'s own "Living Knowledge Graph ecosystem" framing.

A separate `OptimizerStateObserver` (registered via
`AdaptiveOptimizer.add_observer()`, not part of the KG interface at all)
tracks the optimizer's own
`INITIALIZING`/`EXPLORING`/`EXPLOITING`/`REFINING`/`CONVERGED`/`STOPPED`/
`ERROR` progression as an `optimizer_state` entity -- the exact same
`observe_entity_transition` mechanism `engine.py` already uses for
`session_intent`. It exists separately because state-change events arrive
through `OptimizationObserver.on_state_change`, a different interface than
`LivingKnowledgeGraphInterface`.

`PipelineFractalMemoryAdapter` similarly bridges `FractalMemoryInterface`'s
`store_elite_configuration`/`store_failed_configuration`/`get_elites` to
`fractal_brain.storage.Storage`'s generic key-value memory table
(`set_memory`/`get_memory`), so elite/failed configurations persist across
restarts the way `lkg.py` facts and `VectorMemoryStore` documents already do.
This uses a **dedicated db file** (`paths.optimizer_db` in `config.yaml`,
default `data/optimizer.db`), not `data/engine.db`: `Storage`'s own
`documents` table schema (`id, source, text, embedding_blob, created_at`)
collides column-for-column with `memory.VectorMemoryStore`'s (`doc_id, text,
embedding, metadata, success_count, ...`) -- sharing one db file would
corrupt whichever connects second. `tests/test_pipeline_optimizer.py::
test_optimizer_db_path_is_not_the_engine_db_path` guards against these two
paths being pointed at each other again.

## The parameter registry: real, not aspirational

`adaptive_optimizer.py` ships its own
`ParameterRegistry.populate_default_ecosystem_parameters()` with names like
`decoder.temperature`, `controller.confidence_target`,
`fractal_engine.recursion_budget` -- a plausible-sounding but imagined
ecosystem that doesn't match what this rule-based, non-sampling fallback
generator actually has wired up (there's no temperature/top-p sampling
anywhere in `moe_model.py`; "confidence_target" isn't read by anything).
Only one name coincidentally matches a real config key exactly
(`knowledge_graph.forgetting_factor`).

`pipeline_optimizer.build_pipeline_parameter_registry()` registers a
different, smaller set instead -- every one a dotted name that maps to an
actual key in the shipped `config.yaml`
(`tests/test_pipeline_optimizer.py::test_parameter_registry_matches_real_config_keys`
checks this holds):

| Parameter | Range | Module |
|---|---|---|
| `retrieval.top_k` | 1-10 | Retrieval |
| `model.retrieval_confidence_threshold` | 0.1-0.95 | Model |
| `planner.n_states` | 4-30 | Planner |
| `planner.max_plan_steps` | 2-15 | Planner |
| `decomposition.max_subtasks` | 2-10 | Decomposition |
| `knowledge_graph.num_particles` | 5-100 | KnowledgeGraph |
| `knowledge_graph.forgetting_factor` | 0.5-0.999 | KnowledgeGraph |
| `knowledge_graph.low_confidence_threshold` | 0.1-0.6 | KnowledgeGraph |

## The objective function -- and the bug that shaped it

`build_pipeline_objective()`/`_score_config()` apply a candidate's parameter
values on top of the real `config.yaml`, write the result to a real temp
file (`UnifiedAIPipeline`/`OpenClosedLoopEngine` only accept a `config_path`,
not a pre-parsed dict), build a fresh pipeline from it, and run a small fixed
set of benchmark prompts (one per `decomposer.KNOWN_INTENTS` category).

**The first version scored mean reflection confidence minus a risk-count
penalty** -- both already-computed signals, reused rather than inventing a
new metric. Testing it (running the same prompt at
`retrieval_confidence_threshold=0.1` and `=0.99`) found the score came out
*bit-identical* every time. Traced it to `ai_pipeline.py`'s own `run()`
order: `FractalBrain.evaluate()` runs first, from the raw input tokens only,
*before* the closed loop (which is what every registered parameter
actually affects) runs at all -- so reflection confidence, which is derived
from that loss, is mathematically independent of every parameter in the
registry. A second, related bug compounded this while debugging it: the
temp config file lives in a throwaway temp directory, and
`OpenClosedLoopEngine` resolves relative paths (`paths.bootstrap_dataset`)
against its *own config file's* directory -- so every trial's memory corpus
was silently empty, regardless of any setting, which is its own reason nothing could vary. Both are fixed now
(`_resolve_paths_section` resolves `paths.*` to absolute against the real
project directory before writing the temp config) and covered by
regression tests (`test_score_config_finds_bootstrap_data_from_a_temp_directory`,
`test_score_config_is_sensitive_to_retrieval_threshold`).

**The current objective**: for each benchmark prompt, reward the retrieved
document's cosine score when it clears `model.retrieval_confidence_threshold`
(scaling the reward by how good the match actually was), or a flat `-0.1`
when it doesn't (a missed opportunity to give a grounded answer) -- minus a
small penalty for reflection risks. This is mechanically, verifiably
sensitive to the threshold (confirmed: permissive settings consistently
outscore strict ones across the benchmark set).

**Known, stated-plainly scope limit**: this reward is directly sensitive to
`model.retrieval_confidence_threshold` (it's the entire branching condition)
but only weakly or not at all to several of the other registered parameters
in a single-shot, no-history trial. `retrieval.top_k` changes how many
candidates are *returned*, not which one is the single best-scoring match
this reward reads (confirmed empirically: identical scores across
`top_k=1..8`) -- it would matter more to an objective that also considered
retrieval diversity or decomposer/planner output quality. Most
`knowledge_graph.*` parameters govern behavior that accumulates over many
turns (entity-transition learning, per-source reliability), which a single
fresh, throwaway pipeline instance per trial never has a chance to
accumulate. `AdaptiveOptimizer` still searches those dimensions safely -- a
flat objective just means no informative gradient there, not an error or
crash -- and a richer, multi-turn, or session-replay objective is a
reasonable follow-up, not something this integration claims to already do.

## Running it

```bash
python hybrid_cli.py --mode tune --trials 20
```

Programmatically:

```python
from pathlib import Path
from engine import parse_simple_yaml
from pipeline_optimizer import build_pipeline_optimizer

config = parse_simple_yaml(Path('config.yaml').read_text())
optimizer, objective = build_pipeline_optimizer(config, Path('.').resolve())
best = optimizer.optimize(objective, max_trials=20, batch_size=1)
print(best.config, best.composite_score)
```

**Performance**: each trial builds a full `UnifiedAIPipeline` (including a
`FractalBrain` instance) and runs it against 4 benchmark prompts -- budget
roughly 5-10 seconds per trial. 20 trials is a reasonable starting point for
exploring; scale up for a longer, unattended tuning run. Temp directories
are created and removed per trial (`tempfile.mkdtemp()`, never a hardcoded
path -- see `tests/test_regressions.py::test_no_hardcoded_unix_temp_paths`),
so a long run doesn't leak disk space or open sqlite handles.

## What this doesn't do

- Doesn't change `config.yaml` itself. `optimizer.optimize()` returns the
  best configuration found; applying it is a manual step (edit `config.yaml`,
  or pass the winning dict through `_set_dotted` yourself).
- Doesn't tune anything about `FractalBrain`'s own architecture or training
  (`d_model`, `num_experts`, learning rates, ...) -- the registry above is
  scoped to the closed-loop/OCLE side and the knowledge graph, not the
  trained model's own hyperparameters, which is a substantially larger
  undertaking (would need training-loop integration, not single-shot
  evaluation).
- The elite/failed configurations persisted via `PipelineFractalMemoryAdapter`
  aren't currently read back by anything else in the pipeline (e.g. to seed
  future search) -- they're durable, inspectable history, not yet a closed
  loop back into search seeding.
