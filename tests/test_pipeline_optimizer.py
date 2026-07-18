"""Tests for pipeline_optimizer.py -- the integration layer bridging
adaptive_optimizer.py's AdaptiveOptimizer to the real lkg.py graph,
fractal_brain storage, and this pipeline's actual config/objective.
adaptive_optimizer.py's own machinery (search strategies, belief tracking,
constraints, scoring) is a large, independently-authored library verified
separately (see CHANGELOG.md); these tests cover the wiring specific to
this integration. Trial counts are kept small throughout (2-3 per test) --
each trial builds and runs a full pipeline instance, at roughly 5s/trial
(see docs/ADAPTIVE_OPTIMIZER.md), so this suite stays reasonably fast.
"""
from __future__ import annotations

from pathlib import Path

import adaptive_optimizer as ao
from engine import parse_simple_yaml
from fractal_brain.storage import Storage
from lkg import LivingKnowledgeGraph
from pipeline_optimizer import (
    DEFAULT_BENCHMARK_PROMPTS,
    PIPELINE_PARAMETER_SPECS,
    OPTIMIZER_STATE_ENTITY,
    OptimizerStateObserver,
    PipelineFractalMemoryAdapter,
    PipelineKnowledgeGraphAdapter,
    _score_config,
    build_pipeline_optimizer,
    build_pipeline_parameter_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _real_config() -> dict:
    return parse_simple_yaml((REPO_ROOT / 'config.yaml').read_text(encoding='utf-8'))


def test_parameter_registry_matches_real_config_keys() -> None:
    """Every registered parameter must map to a key that actually exists in
    the shipped config.yaml -- guards against drifting to describe an
    aspirational config, the mistake adaptive_optimizer.py's own
    populate_default_ecosystem_parameters() makes for this codebase (see
    CHANGELOG.md)."""
    config = _real_config()
    for dotted_name, _module_name, _make_spec in PIPELINE_PARAMETER_SPECS:
        section, key = dotted_name.split('.', 1)
        assert section in config, f'{dotted_name}: section {section!r} missing from config.yaml'
        assert key in config[section], f'{dotted_name}: key {key!r} missing from config.yaml[{section!r}]'


def test_optimizer_db_path_is_not_the_engine_db_path() -> None:
    """Storage's own `documents` table schema collides column-for-column
    with memory.VectorMemoryStore's -- sharing one db file would corrupt
    whichever connects second (see CHANGELOG.md). Guards against the two
    paths accidentally being pointed at each other again."""
    config = _real_config()
    assert config['paths']['optimizer_db'] != config['paths']['sqlite_db']


def test_score_config_finds_bootstrap_data_from_a_temp_directory(tmp_path) -> None:
    """Regression test: an earlier version didn't resolve paths.* against
    the real project directory, so a trial config living in a temp
    directory silently found zero bootstrap documents regardless of any
    parameter value (see CHANGELOG.md, "objective function redesigned")."""
    config = _real_config()
    # A permissive threshold: if retrieval found nothing, this would score
    # -0.1 (or worse with the risk penalty) for every prompt.
    score = _score_config(config, REPO_ROOT, {'model.retrieval_confidence_threshold': 0.1, 'retrieval.top_k': 3}, DEFAULT_BENCHMARK_PROMPTS[:1])
    assert score > -0.05


def test_score_config_is_sensitive_to_retrieval_threshold() -> None:
    """Regression test for the bug this replaced: the original objective
    (reflection confidence) was mathematically identical across every
    configuration, since FractalBrain evaluates before the closed loop runs
    at all. This one must actually discriminate."""
    config = _real_config()
    permissive = _score_config(config, REPO_ROOT, {'model.retrieval_confidence_threshold': 0.1, 'retrieval.top_k': 3}, DEFAULT_BENCHMARK_PROMPTS)
    strict = _score_config(config, REPO_ROOT, {'model.retrieval_confidence_threshold': 0.99, 'retrieval.top_k': 3}, DEFAULT_BENCHMARK_PROMPTS)
    assert permissive > strict


def test_score_config_cleans_up_its_temp_directory() -> None:
    import tempfile
    before = set(Path(tempfile.gettempdir()).glob('pipeline_optimizer_trial_*'))
    config = _real_config()
    _score_config(config, REPO_ROOT, {'model.retrieval_confidence_threshold': 0.5, 'retrieval.top_k': 3}, DEFAULT_BENCHMARK_PROMPTS[:1])
    after = set(Path(tempfile.gettempdir()).glob('pipeline_optimizer_trial_*'))
    assert after == before


def test_kg_adapter_pushes_a_real_fact_and_delegates_bookkeeping() -> None:
    real_kg = LivingKnowledgeGraph(num_particles=10, seed=0)
    adapter = PipelineKnowledgeGraphAdapter(real_kg, source='test_source')

    registry = ao.ParameterRegistry()
    registry.register(ao.ContinuousParameter('x', 0.0, 1.0), 'toy')
    trial = ao.TrialResult(trial_id='t1', config={'x': 0.5}, objective_scores={'main': 1.0}, composite_score=1.0, status=ao.TrialStatus.COMPLETED, start_time=0.0, end_time=0.0)

    adapter.record_trial(trial)
    # Delegated bookkeeping (adaptive_optimizer.py's own default impl):
    assert len(adapter._raw.historical_experiments) == 1
    # Real, repeatedly-observable fact pushed to the actual lkg.py graph:
    conf = real_kg.get_confidence('test_source', 'trial_improves_on_recent_average', 'true')
    assert 0.0 <= conf['mean'] <= 1.0


def test_fractal_memory_adapter_persists_across_instances(tmp_path) -> None:
    db_path = tmp_path / 'optimizer.db'
    trial = ao.TrialResult(trial_id='t1', config={'x': 0.5}, objective_scores={'main': 0.9}, composite_score=0.9, status=ao.TrialStatus.COMPLETED, start_time=0.0, end_time=0.0)

    adapter1 = PipelineFractalMemoryAdapter(Storage(str(db_path)))
    adapter1.store_elite_configuration(trial)

    # A fresh adapter, same db file: elites must survive the round-trip
    # through fractal_brain.storage.Storage, not just live in memory.
    adapter2 = PipelineFractalMemoryAdapter(Storage(str(db_path)))
    elites = adapter2.get_elites(5)
    assert len(elites) == 1
    assert elites[0].trial_id == 't1'
    assert elites[0].composite_score == 0.9


def test_optimizer_state_observer_defines_entity_idempotently() -> None:
    """define_entity_states() unconditionally rebuilds fresh
    DiscreteMarkovChain instances (see lkg.py) -- constructing a second
    observer against the same real_kg must not wipe a transition already
    observed by the first."""
    real_kg = LivingKnowledgeGraph(num_particles=10, seed=0)
    observer1 = OptimizerStateObserver(real_kg)
    observer1.on_state_change(ao.OptimizerState.INITIALIZING, ao.OptimizerState.EXPLORING)
    before = real_kg.get_current_state_distribution(OPTIMIZER_STATE_ENTITY)

    OptimizerStateObserver(real_kg)  # second observer, same graph
    after = real_kg.get_current_state_distribution(OPTIMIZER_STATE_ENTITY)
    assert before == after


def test_build_pipeline_optimizer_runs_end_to_end(tmp_path) -> None:
    config = _real_config()
    optimizer, objective = build_pipeline_optimizer(config, REPO_ROOT, optimizer_db_path=tmp_path / 'optimizer.db')
    results = optimizer.step(objective, batch_size=2)
    assert len(results) == 2
    assert all(r.status in (ao.TrialStatus.COMPLETED, ao.TrialStatus.FAILED) for r in results)
    assert len(optimizer.memory._elites) + len(optimizer.memory._failed) >= 1
