"""Regression tests for bugs found and fixed in a full-codebase review.

Each test below is named after, and guards against, one specific bug. See
CHANGELOG for the corresponding entry and root-cause writeup.
"""
from __future__ import annotations

import sqlite3
import subprocess
import sys

from engine import OpenClosedLoopEngine, parse_simple_yaml
from decomposer import TaskDecomposer
from planner import MarkovChainPlanner


def test_bootstrap_is_idempotent(tmp_path) -> None:
    """engine.initialize() used to re-insert the entire bootstrap dataset as
    brand-new duplicate rows every single time, with no check for whether a
    given record was already present. Initializing repeatedly against the
    same db must not grow the documents table past the dataset size."""
    db_path = tmp_path / 'engine.db'

    for _ in range(3):
        engine = OpenClosedLoopEngine.from_default_config()
        engine.config['paths']['sqlite_db'] = str(db_path)
        engine.memory.sqlite_path = engine.memory._resolve_path(str(db_path))
        engine.initialize()

    bootstrap_records = engine.memory.load_bootstrap_records(
        engine.memory._resolve_path(engine.config['paths']['bootstrap_dataset'])
    )
    conn = sqlite3.connect(db_path)
    row_count = conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
    assert row_count == len(bootstrap_records)


def test_repeated_retrieval_returns_distinct_documents(tmp_path) -> None:
    """Direct consequence of the duplication bug: retrieval used to return
    several copies of the same document instead of distinct ones."""
    db_path = tmp_path / 'engine.db'
    engine = OpenClosedLoopEngine.from_default_config()
    engine.config['paths']['sqlite_db'] = str(db_path)
    engine.memory.sqlite_path = engine.memory._resolve_path(str(db_path))
    engine.initialize()
    engine.initialize()  # second init against the same db on purpose

    result = engine.run('Solve the integral of 2x from 0 to 4.')
    texts = [r['text'] for r in result['retrieved']]
    assert len(texts) == len(set(texts))


def test_pipeline_trace_normalize_stage_is_current_turn(tmp_path) -> None:
    """The 'normalize' trace stage used to read state that was only updated
    *after* the trace was built, so it always showed the previous turn's
    text (or None on the first turn)."""
    from ai_pipeline import UnifiedAIPipeline

    pipeline = UnifiedAIPipeline(config_path='config.yaml')
    pipeline.closed_loop.config['paths']['sqlite_db'] = str(tmp_path / 'engine.db')
    pipeline.closed_loop.memory.sqlite_path = tmp_path / 'engine.db'
    pipeline.initialize()

    r1 = pipeline.run('First question about gears').to_dict()
    r2 = pipeline.run('Second question about torque').to_dict()

    norm1 = next(s for s in r1['trace'] if s['name'] == 'normalize')
    norm2 = next(s for s in r2['trace'] if s['name'] == 'normalize')
    assert norm1['data']['normalized_text'] == 'First question about gears'
    assert norm2['data']['normalized_text'] == 'Second question about torque'


def test_planner_prefers_well_evidenced_action_over_rare_one() -> None:
    """_best() used to score actions by summing their own next-state
    probability row, which is independently re-normalized to ~1.0 for any
    action with data -- so it couldn't tell a well-evidenced action from a
    rarely-taken one and effectively broke ties by vocab order."""
    planner = MarkovChainPlanner(n_states=3, max_plan_steps=4, terminal_actions=['finalize'])
    planner.state_centroids = [[1, 0], [0, 1], [0.5, 0.5]]
    planner.action_vocab = ['rare_action', 'common_action', 'finalize']
    planner.action_to_id = {a: i for i, a in enumerate(planner.action_vocab)}
    n = planner.n_states
    planner.transition_counts = [[[0.0] * n for _ in range(len(planner.action_vocab))] for _ in range(n)]
    planner.transition_counts[0][0][1] = 1.0    # rare_action: seen once
    planner.transition_counts[0][1][2] = 20.0   # common_action: seen 20x
    planner._recompute()

    aid, _nxt, _prob = planner._best(0)
    assert planner.action_vocab[aid] == 'common_action'


def test_decomposer_does_not_mislabel_unrelated_text_as_engineering() -> None:
    """Anything that didn't match the math or coding keyword sets used to
    fall through to 'engineering' unconditionally, with the same four
    physics-flavored subtasks regardless of actual content."""
    decomposer = TaskDecomposer()
    result = decomposer.decompose("What's a good gift idea for my partner's birthday?", [])
    assert result.intent == 'general'


def test_decomposer_confidence_is_not_a_flat_constant() -> None:
    """confidence used to be exactly 0.78 or 0.62 (retrieved-context present
    or not) regardless of anything about the input text itself."""
    decomposer = TaskDecomposer()
    strong_match = decomposer.decompose('Solve the integral and derivative of this equation', [])
    no_match = decomposer.decompose("What's a good gift idea?", [])
    assert strong_match.confidence != 0.78
    assert no_match.confidence != 0.62
    assert strong_match.confidence > no_match.confidence


def test_yaml_parser_preserves_hash_inside_quotes() -> None:
    """parse_simple_yaml used to split every line on the first '#'
    unconditionally, silently truncating any quoted value that legitimately
    contained one (e.g. a hex color)."""
    sample = '''
project:
  color: "#FF0000"   # a real trailing comment
  note: 'hash # inside single quotes'
'''
    parsed = parse_simple_yaml(sample)
    assert parsed['project']['color'] == '#FF0000'
    assert parsed['project']['note'] == 'hash # inside single quotes'


def test_ocle_clean_build_reexport_matches_root_engine() -> None:
    """ocle_clean_build's shims are named identically to the top-level
    modules they wildcard-import (e.g. ocle_clean_build/engine.py does
    `from engine import *`). This checks the normal, supported usage path
    still resolves to the *same* class object as the root module."""
    import engine as root_engine
    from ocle_clean_build import engine as shim_engine

    assert shim_engine.OpenClosedLoopEngine is root_engine.OpenClosedLoopEngine


def test_ocle_clean_build_shim_fails_loudly_if_self_imported() -> None:
    """The specific failure mode this guards against: running Python with
    ocle_clean_build/ itself as the working directory used to make a bare
    `import engine` silently import the shim file itself (empty module, no
    error) instead of the real top-level engine.py. It must now fail loudly
    with a clear ImportError instead."""
    result = subprocess.run(
        [sys.executable, '-c', 'import engine'],
        capture_output=True,
        text=True,
        cwd='ocle_clean_build',
    )
    assert result.returncode != 0
    assert 'ImportError' in result.stderr


def test_no_hardcoded_unix_temp_paths() -> None:
    """tests/test_smoke.py used to hardcode "/tmp/_test_tokenizer_vocab.json"
    for a save/load round-trip, which crashed on Windows (no /tmp there).
    Guard against that whole class of bug reappearing anywhere: nothing in
    the source tree should hardcode /tmp, /var, /home, /etc, or /usr as an
    absolute path -- use tempfile.mkdtemp()/gettempdir() instead."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    this_file = Path(__file__).resolve()
    pattern = re.compile(r'''['"]/(?:tmp|var|home|etc|usr)/''')
    offenders = []
    for path in root.rglob('*.py'):
        if '__pycache__' in path.parts or '.pytest_cache' in path.parts:
            continue
        if path.resolve() == this_file:
            continue
        text = path.read_text(encoding='utf-8', errors='ignore')
        if pattern.search(text):
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f'hardcoded Unix-only absolute paths found in: {offenders}'


def test_lkg_entity_tracking_had_no_public_observation_path(tmp_path) -> None:
    """Pins down the gap found while integrating lkg.py: the shipped
    LivingKnowledgeGraph could define entity states and predict from them,
    but had no public method to tell it what state was actually observed.
    DiscreteMarkovChain.observe_transition existed but was reachable only
    from the module's own tests, never from ParticleFilter or
    LivingKnowledgeGraph itself -- so step() alone could never move a
    prediction away from uniform. This guards the general property (not
    just the specific fixed method, which has its own coverage in
    tests/test_lkg.py) so a future refactor of lkg.py can't quietly
    reopen it without a public entity-observation method of some kind."""
    from lkg import LivingKnowledgeGraph

    kg = LivingKnowledgeGraph(num_particles=10, forgetting_factor=0.99, seed=0)
    kg.define_entity_states('probe', ['a', 'b'])
    public_methods = [name for name in dir(kg) if not name.startswith('_')]
    observation_methods = [
        name for name in public_methods
        if 'observe' in name.lower() or 'transition' in name.lower()
    ]
    assert observation_methods, (
        'LivingKnowledgeGraph has no public method for recording an observed '
        'entity transition -- entity-state tracking would be unusable from '
        'real data again (see CHANGELOG.md, "Living Knowledge Graph merged '
        'in and wired into the closed loop").'
    )


def test_knowledge_graph_initialize_is_idempotent(tmp_path) -> None:
    """Same shape of bug as test_bootstrap_is_idempotent above, in the new
    knowledge-graph wiring added alongside it: LivingKnowledgeGraph.
    define_entity_states() unconditionally builds fresh DiscreteMarkovChain
    instances, wiping any accumulated history, so OpenClosedLoopEngine must
    guard against calling it again on a repeated initialize()."""
    from engine import OpenClosedLoopEngine

    db_path = tmp_path / 'engine.db'
    engine = OpenClosedLoopEngine.from_default_config()
    engine.config['paths']['sqlite_db'] = str(db_path)
    engine.memory.sqlite_path = engine.memory._resolve_path(str(db_path))
    engine.initialize()
    engine.run('Solve the integral of 2x from 0 to 4.')
    engine.run('Solve x^2 - 5x + 6 = 0.')

    before = engine.knowledge_graph.get_current_state_distribution('session_intent')
    engine.initialize()
    after = engine.knowledge_graph.get_current_state_distribution('session_intent')
    assert before == after
