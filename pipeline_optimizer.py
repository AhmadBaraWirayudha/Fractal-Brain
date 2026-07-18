"""pipeline_optimizer.py

Wires adaptive_optimizer.py's AdaptiveOptimizer into this specific pipeline:
real config.yaml parameters to search, a real objective function that scores
a candidate configuration by actually running the pipeline, and adapters
bridging AdaptiveOptimizer's LivingKnowledgeGraphInterface/
FractalMemoryInterface to the real lkg.py graph and fractal_brain's storage
instead of adaptive_optimizer.py's own standalone default implementations.

Why adapters, not just passing the real objects in directly
-------------------------------------------------------------
adaptive_optimizer.LivingKnowledgeGraphInterface requires add_node/add_edge/
record_trial/get_parameter_correlations; lkg.LivingKnowledgeGraph has none of
those (it speaks facts/entities/confidence, not nodes/edges/trials). Passing
a real lkg.LivingKnowledgeGraph in directly would satisfy nothing and fail at
the first call. PipelineKnowledgeGraphAdapter below delegates the
bookkeeping adaptive_optimizer.py actually needs (nodes, edges, correlations)
to one of its own default LivingKnowledgeGraph instances -- reusing that
rather than reimplementing Pearson correlation and best/worst tracking from
scratch -- and additionally pushes one real, repeatedly-observable fact into
the shared lkg.py graph per completed trial: whether this optimization run
is still improving on its own recent average. That's meaningfully different
from a single-observation-per-subject fact (which Beta-confidence tracking
can't build a real signal from), and it shares the same graph instance the
engine's session_intent/matched_intent tracking uses, per lkg.py's own
"Living Knowledge Graph ecosystem" framing. An OptimizerStateObserver
(registered separately, since state changes arrive via OptimizationObserver.
on_state_change, not through LivingKnowledgeGraphInterface at all) tracks the
optimizer's own INITIALIZING/EXPLORING/EXPLOITING/... progression as an
entity, the same mechanism engine.py already uses for session_intent.

Similarly, FractalMemoryInterface's store_elite_configuration/
store_failed_configuration/get_elites bridge to fractal_brain.storage.
Storage's generic key-value memory table (set_memory/get_memory) -- a
dedicated db file (paths.optimizer_db in config.yaml), not data/engine.db:
Storage's own `documents` table schema collides column-for-column with
memory.VectorMemoryStore's `documents` table (see CHANGELOG.md), so sharing
one db file between the two would corrupt whichever connects second.
"""
from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path
from typing import Any

import adaptive_optimizer as ao
from ai_pipeline import UnifiedAIPipeline
from engine import dump_simple_yaml
from fractal_brain.storage import Storage
from lkg import LivingKnowledgeGraph

OPTIMIZER_STATE_ENTITY = 'optimizer_state'

# Dotted parameter name -> (config section, key, ParameterSpec constructor args).
# Every entry maps to a real config.yaml value actually read somewhere in this
# pipeline (see CHANGELOG.md for the specific line each one is read on) --
# deliberately not adaptive_optimizer.py's own populate_default_ecosystem_
# parameters() defaults, which describe an aspirational ecosystem (e.g.
# "decoder.temperature", "controller.confidence_target") that doesn't match
# what this rule-based fallback generator actually has wired up.
PIPELINE_PARAMETER_SPECS: list[tuple[str, str, Any]] = [
    ('retrieval.top_k', 'Retrieval', lambda: ao.IntegerParameter('retrieval.top_k', 1, 10, prior_mean=3, prior_std=2)),
    ('model.retrieval_confidence_threshold', 'Model', lambda: ao.ContinuousParameter('model.retrieval_confidence_threshold', 0.1, 0.95, prior_mean=0.5, prior_std=0.15)),
    ('planner.n_states', 'Planner', lambda: ao.IntegerParameter('planner.n_states', 4, 30, prior_mean=12, prior_std=6)),
    ('planner.max_plan_steps', 'Planner', lambda: ao.IntegerParameter('planner.max_plan_steps', 2, 15, prior_mean=6, prior_std=3)),
    ('decomposition.max_subtasks', 'Decomposition', lambda: ao.IntegerParameter('decomposition.max_subtasks', 2, 10, prior_mean=5, prior_std=2)),
    ('knowledge_graph.num_particles', 'KnowledgeGraph', lambda: ao.IntegerParameter('knowledge_graph.num_particles', 5, 100, prior_mean=20, prior_std=15)),
    ('knowledge_graph.forgetting_factor', 'KnowledgeGraph', lambda: ao.ContinuousParameter('knowledge_graph.forgetting_factor', 0.5, 0.999, prior_mean=0.95, prior_std=0.1)),
    ('knowledge_graph.low_confidence_threshold', 'KnowledgeGraph', lambda: ao.ContinuousParameter('knowledge_graph.low_confidence_threshold', 0.1, 0.6, prior_mean=0.4, prior_std=0.1)),
]

# A handful of fixed, varied benchmark prompts (spanning every decomposer.
# KNOWN_INTENTS category) to score a candidate configuration against. Small
# and fixed on purpose: the objective function below constructs and tears
# down a full pipeline per trial, so this is a real but deliberately cheap
# evaluation, not a claim that 4 prompts fully characterize quality.
DEFAULT_BENCHMARK_PROMPTS: tuple[str, ...] = (
    'Solve the integral of 2x from 0 to 4.',
    'Fix a Python loop that appends numbers 0 to 4.',
    'Estimate motor torque and load on a conveyor shaft.',
    'What is a reasonable morning routine?',
)


def _set_dotted(config: dict[str, Any], dotted_name: str, value: Any) -> None:
    section, key = dotted_name.split('.', 1)
    config.setdefault(section, {})[key] = value


def build_pipeline_parameter_registry() -> 'ao.ParameterRegistry':
    """A real, scoped ParameterRegistry: every parameter maps to an actual
    config.yaml value this pipeline reads, not adaptive_optimizer.py's own
    aspirational defaults."""
    registry = ao.ParameterRegistry()
    for dotted_name, module_name, make_spec in PIPELINE_PARAMETER_SPECS:
        registry.register(make_spec(), module_name)
    return registry


class PipelineKnowledgeGraphAdapter(ao.LivingKnowledgeGraphInterface):
    """Bridges AdaptiveOptimizer to the real lkg.py graph. See module
    docstring above for why this delegates bookkeeping to an internal
    default LivingKnowledgeGraph rather than reimplementing it, and pushes a
    single aggregate improvement fact to the real graph rather than one fact
    per trial-id (which Beta-confidence tracking can't build a signal from)."""

    def __init__(self, real_kg: LivingKnowledgeGraph, source: str = 'adaptive_optimizer', recent_window: int = 10) -> None:
        self._raw = ao.LivingKnowledgeGraph()  # adaptive_optimizer.py's own default: reused for node/edge/correlation bookkeeping only
        self.real_kg = real_kg
        self.source = source
        self.recent_window = recent_window

    def add_node(self, node: 'ao.KGNode') -> None:
        self._raw.add_node(node)

    def add_edge(self, edge: 'ao.KGEdge') -> None:
        self._raw.add_edge(edge)

    def get_parameter_correlations(self, param_names: list[str]) -> dict[tuple[str, str], float]:
        return self._raw.get_parameter_correlations(param_names)

    def record_trial(self, trial: 'ao.TrialResult') -> None:
        self._raw.record_trial(trial)
        completed = [t for t in self._raw.historical_experiments if t.status == ao.TrialStatus.COMPLETED]
        recent = completed[-(self.recent_window + 1):-1] or completed[:1]
        recent_avg = sum(t.composite_score for t in recent) / len(recent) if recent else trial.composite_score
        improved = trial.status == ao.TrialStatus.COMPLETED and trial.composite_score > recent_avg
        self.real_kg.add_fact(
            self.source, 'trial_improves_on_recent_average', 'true',
            source=self.source, positive=improved,
        )


class PipelineFractalMemoryAdapter(ao.FractalMemoryInterface):
    """Bridges AdaptiveOptimizer to fractal_brain.storage.Storage's generic
    memory table, so elite/failed configurations persist across restarts
    the way lkg.py facts and VectorMemoryStore documents already do. See
    module docstring above for why this is a dedicated db file rather than
    data/engine.db."""

    ELITES_KEY = 'adaptive_optimizer:elites'
    FAILED_KEY = 'adaptive_optimizer:failed'

    def __init__(self, storage: Storage, max_elites: int = 20, max_failed: int = 200) -> None:
        self.storage = storage
        self.max_elites = max_elites
        self.max_failed = max_failed
        self._elites: list[dict[str, Any]] = self.storage.get_memory(self.ELITES_KEY, default=[])
        self._failed: list[dict[str, Any]] = self.storage.get_memory(self.FAILED_KEY, default=[])

    @staticmethod
    def _serialize(trial: 'ao.TrialResult') -> dict[str, Any]:
        return {
            'trial_id': trial.trial_id,
            'config': trial.config,
            'composite_score': trial.composite_score,
            'status': trial.status.value,
            'end_time': trial.end_time,
        }

    @staticmethod
    def _deserialize(data: dict[str, Any]) -> 'ao.TrialResult':
        return ao.TrialResult(
            trial_id=data['trial_id'],
            config=data['config'],
            objective_scores={'main': data['composite_score']},
            composite_score=data['composite_score'],
            status=ao.TrialStatus(data['status']),
            start_time=data['end_time'],
            end_time=data['end_time'],
        )

    def store_elite_configuration(self, trial: 'ao.TrialResult') -> None:
        self._elites.append(self._serialize(trial))
        self._elites.sort(key=lambda t: t['composite_score'], reverse=True)
        self._elites = self._elites[:self.max_elites]
        self.storage.set_memory(self.ELITES_KEY, self._elites)

    def store_failed_configuration(self, trial: 'ao.TrialResult') -> None:
        self._failed.append(self._serialize(trial))
        self._failed = self._failed[-self.max_failed:]
        self.storage.set_memory(self.FAILED_KEY, self._failed)

    def get_elites(self, top_k: int = 10) -> list['ao.TrialResult']:
        return [self._deserialize(d) for d in self._elites[:top_k]]


def _ensure_optimizer_state_entity(real_kg: LivingKnowledgeGraph) -> None:
    """Define the optimizer_state entity if it isn't already, without
    reaching into lkg.py's private internals to check: get_current_state_
    distribution's documented KeyError is the public-API signal that an
    entity hasn't been defined yet. Guarding matters because
    define_entity_states() unconditionally rebuilds fresh
    DiscreteMarkovChain instances (see lkg.py, and engine.py's own
    _kg_entities_defined guard for session_intent) -- calling it twice on a
    real_kg shared across more than one OptimizerStateObserver (e.g.
    building more than one optimizer against the same persistent graph)
    would silently wipe any state-transition history already observed."""
    try:
        real_kg.get_current_state_distribution(OPTIMIZER_STATE_ENTITY)
    except KeyError:
        real_kg.define_entity_states(OPTIMIZER_STATE_ENTITY, [s.value for s in ao.OptimizerState])


class OptimizerStateObserver(ao.OptimizationObserver):
    """Tracks AdaptiveOptimizer's own INITIALIZING/EXPLORING/EXPLOITING/...
    lifecycle as an entity in the real knowledge graph -- the same
    observe_entity_transition mechanism engine.py uses for session_intent.
    Registered via AdaptiveOptimizer.add_observer(), separately from
    PipelineKnowledgeGraphAdapter, because state-change events arrive
    through OptimizationObserver, not LivingKnowledgeGraphInterface."""

    def __init__(self, real_kg: LivingKnowledgeGraph) -> None:
        self.real_kg = real_kg
        _ensure_optimizer_state_entity(real_kg)

    def on_trial_start(self, trial_id: str, config: dict[str, Any]) -> None:
        pass

    def on_trial_complete(self, result: 'ao.TrialResult') -> None:
        pass

    def on_strategy_switch(self, old_strategy: str, new_strategy: str) -> None:
        pass

    def on_state_change(self, old_state: 'ao.OptimizerState', new_state: 'ao.OptimizerState') -> None:
        self.real_kg.observe_entity_transition(OPTIMIZER_STATE_ENTITY, new_state.value)


def _resolve_paths_section(config: dict[str, Any], base_path: Path) -> None:
    """Resolve every paths.* entry to an absolute path against the real
    project base_path. Needed because OpenClosedLoopEngine resolves relative
    paths against its *own config file's* directory (config_path.resolve().
    parent) -- a trial's config file lives in a throwaway temp directory, so
    without this, paths.bootstrap_dataset silently resolves to a temp path
    with nothing there, leaving every trial's memory corpus empty regardless
    of any parameter value (found by testing: retrieval_confidence_threshold
    had zero effect on final_output at any setting from 0.1 to 0.99, traced
    to 0 documents ever loading -- see CHANGELOG.md)."""
    paths = config.get('paths', {})
    for key, value in paths.items():
        if isinstance(value, str) and not Path(value).is_absolute():
            paths[key] = str((base_path / value).resolve())


def _score_config(base_config: dict[str, Any], base_path: Path, overrides: dict[str, Any], benchmark_prompts: tuple[str, ...]) -> float:
    """Apply `overrides` (dotted-name -> value, one entry per registered
    parameter) on top of a deep copy of `base_config`, run a fresh pipeline
    against `benchmark_prompts`, and score it.

    Scoring signal: mean, over the benchmark prompts, of the best retrieved
    document's cosine score when it clears model.retrieval_confidence_
    threshold (rewarding a quality-scaled, retrieval-grounded answer), or a
    flat -0.1 when it doesn't (a missed opportunity to give a grounded
    answer, falling back to generic plan text) -- minus a small penalty for
    reflection risks. This replaced an earlier version scored on reflection
    confidence directly; testing found that signal came from FractalBrain's
    loss, which is evaluated from the raw input tokens *before* the closed
    loop runs at all (see ai_pipeline.py's run()), so it was mathematically
    identical across every configuration tried -- confirmed by running the
    same prompt at threshold=0.1 and threshold=0.99 and getting bit-identical
    loss and confidence both times. See CHANGELOG.md ("objective function
    redesigned after discovering it couldn't discriminate between
    configurations").

    Known scope limit, stated plainly rather than glossed over: this reward
    is mechanically sensitive to model.retrieval_confidence_threshold (its
    entire branching condition) but only weakly or not at all to several of
    the other registered parameters in a single-shot, no-history trial --
    e.g. retrieval.top_k changes how many candidates are returned, not which
    one is the single best-scoring match this reward reads, and knowledge_
    graph.* parameters mostly affect behavior that accumulates over many
    turns, which one fresh, throwaway pipeline instance per trial never has.
    AdaptiveOptimizer still searches those dimensions safely (a flat
    objective just means no informative gradient there, not an error), and a
    richer or multi-turn objective is a reasonable follow-up -- see
    docs/ADAPTIVE_OPTIMIZER.md.

    UnifiedAIPipeline/OpenClosedLoopEngine only accept a config_path (see
    CHANGELOG.md), not a pre-parsed dict, so this writes the overridden
    config out to a real temp file (engine.dump_simple_yaml) and constructs
    through the normal, already-tested constructor rather than reaching
    around it. Each trial gets its own tempfile.mkdtemp() directory (never
    a hardcoded /tmp path -- see
    tests/test_regressions.py::test_no_hardcoded_unix_temp_paths), removed
    in a finally block so a long optimize() run doesn't leak temp dirs or
    open sqlite file handles.
    """
    config = copy.deepcopy(base_config)
    for dotted_name, value in overrides.items():
        _set_dotted(config, dotted_name, value)
    _resolve_paths_section(config, base_path)

    tmp_dir = Path(tempfile.mkdtemp(prefix='pipeline_optimizer_trial_'))
    try:
        config.setdefault('paths', {})['sqlite_db'] = str(tmp_dir / 'trial.db')
        config_path = tmp_dir / 'trial_config.yaml'
        config_path.write_text(dump_simple_yaml(config), encoding='utf-8')

        pipeline = UnifiedAIPipeline(config_path=config_path)
        pipeline.initialize()
        threshold = float(config.get('model', {}).get('retrieval_confidence_threshold', 0.5))

        rewards = []
        risk_counts = []
        for prompt in benchmark_prompts:
            result = pipeline.run(prompt)
            retrieved = result.closed_loop['retrieved']
            best_score = max((d['score'] for d in retrieved), default=0.0)
            rewards.append(best_score if best_score >= threshold else -0.1)
            risk_counts.append(len(result.reflection['risks']))
        if pipeline.closed_loop.memory.conn is not None:
            pipeline.closed_loop.memory.conn.close()

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        mean_risks = sum(risk_counts) / len(risk_counts) if risk_counts else 0.0
        return mean_reward - 0.05 * mean_risks
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def build_pipeline_objective(base_config: dict[str, Any], base_path: Path, benchmark_prompts: tuple[str, ...] = DEFAULT_BENCHMARK_PROMPTS) -> 'ao.SingleObjective':
    def eval_fn(config_values: dict[str, Any], context: 'ao.ContextMetadata') -> float:
        return _score_config(base_config, base_path, config_values, benchmark_prompts)
    return ao.SingleObjective('main', eval_fn)


def build_pipeline_optimizer(
    base_config: dict[str, Any],
    base_path: str | Path,
    real_kg: LivingKnowledgeGraph | None = None,
    optimizer_db_path: str | Path | None = None,
) -> tuple['ao.AdaptiveOptimizer', 'ao.SingleObjective']:
    """Construct a fully-wired AdaptiveOptimizer for this pipeline: the real
    parameter registry above, adapters bridging to the real lkg.py graph and
    fractal_brain storage, and a real objective function. `base_path` is the
    real project directory paths.* should resolve against (see
    _resolve_paths_section) -- normally Path(config_path).resolve().parent
    for whatever config.yaml this was loaded from. `optimizer_db_path`
    defaults to paths.optimizer_db from base_config, resolved against
    base_path if relative. Returns (optimizer, objective) -- pass both to
    optimizer.optimize(objective, ...) or optimizer.step(objective, ...)."""
    base_path = Path(base_path)
    if optimizer_db_path is None:
        optimizer_db_path = base_config.get('paths', {}).get('optimizer_db', 'data/optimizer.db')
    optimizer_db_path = Path(optimizer_db_path)
    if not optimizer_db_path.is_absolute():
        optimizer_db_path = base_path / optimizer_db_path

    kg = real_kg if real_kg is not None else LivingKnowledgeGraph.from_config(base_config)
    storage = Storage(str(optimizer_db_path))
    optimizer = ao.AdaptiveOptimizer(
        parameter_registry=build_pipeline_parameter_registry(),
        knowledge_graph=PipelineKnowledgeGraphAdapter(kg),
        fractal_memory=PipelineFractalMemoryAdapter(storage),
    )
    optimizer.add_observer(OptimizerStateObserver(kg))
    objective = build_pipeline_objective(base_config, base_path)
    return optimizer, objective
