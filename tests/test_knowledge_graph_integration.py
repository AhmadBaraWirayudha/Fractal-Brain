"""Integration tests for the Living Knowledge Graph wiring in engine.py and
ai_pipeline.py (lkg.py's own unit tests live in tests/test_lkg.py).

These cover the actual wiring: that the engine updates the graph from real
retrieval/decomposition/feedback events, that the result of doing so is
correctly surfaced up through UnifiedAIPipeline, and that the
initialize()-repeatability guarantee the rest of this project already
relies on (test_bootstrap_is_idempotent) still holds with the knowledge
graph attached. See CHANGELOG.md for the full design writeup.
"""
from __future__ import annotations

from ai_pipeline import UnifiedAIPipeline
from decomposer import KNOWN_INTENTS
from engine import OpenClosedLoopEngine


def _fresh_engine(tmp_path) -> OpenClosedLoopEngine:
    engine = OpenClosedLoopEngine.from_default_config()
    db_path = tmp_path / 'engine.db'
    engine.config['paths']['sqlite_db'] = str(db_path)
    engine.memory.sqlite_path = engine.memory._resolve_path(str(db_path))
    engine.initialize()
    return engine


def test_engine_run_includes_knowledge_graph_field(tmp_path) -> None:
    engine = _fresh_engine(tmp_path)
    result = engine.run('Solve the integral of 2x from 0 to 4.')
    assert 'knowledge_graph' in result
    kg = result['knowledge_graph']
    assert set(kg) == {'predicted_next_intent', 'predicted_intent_distribution', 'retrieved_doc_confidence'}
    assert kg['predicted_next_intent'] in {'math_symbolic', 'coding', 'engineering', 'general'}


def test_session_intent_tracking_reflects_actual_intents(tmp_path) -> None:
    """The whole point of observe_entity_transition: after a session that's
    mostly math questions, the graph's own tracked state -- not just this
    turn's guess -- should say so, not sit at a uniform prior forever."""
    engine = _fresh_engine(tmp_path)
    prompts = [
        'Solve the integral of 2x from 0 to 4.',
        'Solve x^2 - 5x + 6 = 0.',
        'Differentiate 3x^3 + 2x.',
    ]
    for prompt in prompts:
        engine.run(prompt)

    distribution = engine.knowledge_graph.get_current_state_distribution('session_intent')
    intents = list(KNOWN_INTENTS)
    math_share = distribution[intents.index('math_symbolic')]
    assert math_share == max(distribution)
    assert math_share > 1.0 / len(intents)


def test_knowledge_graph_survives_repeated_initialize(tmp_path) -> None:
    """Mirrors test_bootstrap_is_idempotent's pattern: initialize() is
    legitimately called more than once against the same engine/db elsewhere
    in this project, and must not silently wipe the knowledge graph's
    accumulated session_intent history when it does."""
    engine = _fresh_engine(tmp_path)
    engine.run('Solve the integral of 2x from 0 to 4.')
    engine.run('Solve x^2 - 5x + 6 = 0.')
    before = engine.knowledge_graph.get_current_state_distribution('session_intent')

    engine.initialize()
    engine.initialize()

    after = engine.knowledge_graph.get_current_state_distribution('session_intent')
    assert before == after


def test_close_loop_feedback_reaches_knowledge_graph(tmp_path) -> None:
    """Positive feedback should be able to move a retrieved document's
    matched_intent confidence for this session's intent away from its
    pre-feedback value; the fact must exist under the same (doc_id,
    'matched_intent', intent) key close_loop() reinforces."""
    engine = _fresh_engine(tmp_path)
    result = engine.run('Solve the integral of 2x from 0 to 4.')
    retrieved = result['retrieved']
    assert retrieved, 'expected at least one retrieved document for this to be a meaningful test'

    before = engine.knowledge_graph.get_confidence(retrieved[0]['doc_id'], 'matched_intent', result['intent'])
    engine.close_loop({'interaction_id': result['interaction_id'], 'success': True})
    after = engine.knowledge_graph.get_confidence(retrieved[0]['doc_id'], 'matched_intent', result['intent'])

    assert after['mean'] != before['mean']


def test_pipeline_surfaces_knowledge_graph_in_cognitive_context_and_trace(tmp_path) -> None:
    pipeline = UnifiedAIPipeline(config_path='config.yaml')
    db_path = tmp_path / 'engine.db'
    pipeline.closed_loop.config['paths']['sqlite_db'] = str(db_path)
    pipeline.closed_loop.memory.sqlite_path = pipeline.closed_loop.memory._resolve_path(str(db_path))
    pipeline.initialize()

    result = pipeline.run('Solve the integral of 2x from 0 to 4.').to_dict()

    assert 'knowledge_graph' in result['fractal']['cognitive_context']
    trace_names = [stage['name'] for stage in result['trace']]
    assert 'knowledge_graph' in trace_names
    assert trace_names.index('knowledge_graph') == trace_names.index('decompose') + 1
    assert trace_names.index('knowledge_graph') == trace_names.index('plan') - 1


def test_reflection_flags_topic_shift(tmp_path) -> None:
    """End-to-end check of the topic-shift reflection heuristic across a
    real two-turn session: math_symbolic then general should trigger it,
    since the tracked session history points toward math_symbolic."""
    pipeline = UnifiedAIPipeline(config_path='config.yaml')
    db_path = tmp_path / 'engine.db'
    pipeline.closed_loop.config['paths']['sqlite_db'] = str(db_path)
    pipeline.closed_loop.memory.sqlite_path = pipeline.closed_loop.memory._resolve_path(str(db_path))
    pipeline.initialize()

    session = pipeline.run_session([
        'Solve the integral of 2x from 0 to 4.',
        'Explain the method in one sentence.',
    ])
    second_turn_risks = session['turns'][1]['reflection']['risks']
    assert any('topic shift' in risk for risk in second_turn_risks)


def test_matched_intent_confidence_is_neutral_before_feedback(tmp_path) -> None:
    """matched_intent confidence is purely feedback-driven (see CHANGELOG.md,
    "matched_intent confidence made purely feedback-driven"): retrieval alone
    must not move it away from the Beta(1,1) neutral prior."""
    engine = _fresh_engine(tmp_path)
    result = engine.run('Solve the integral of 2x from 0 to 4.')
    for doc_conf in result['knowledge_graph']['retrieved_doc_confidence']:
        assert doc_conf['mean'] == 0.5


def test_sustained_negative_feedback_has_no_retrieval_score_floor(tmp_path) -> None:
    """Regression test for the structural bug found while wiring generation
    up to this signal: an earlier version had retrieval itself vote
    positive-evidence weighted by doc.score, which put a floor of
    score/(score+1) under confidence -- worst for the highest-scoring
    (most confidently retrieved) documents, since a near-1.0 score pushes
    that floor toward 0.5. Confirms confidence can now drop meaningfully
    below the shared low_confidence_threshold under sustained failure,
    for a document whose retrieval score is high."""
    engine = _fresh_engine(tmp_path)
    threshold = engine.config['knowledge_graph']['low_confidence_threshold']
    last_mean = 1.0
    for _ in range(15):
        r = engine.run('Solve the integral of 2x from 0 to 4.')
        assert r['retrieved'][0]['score'] > 0.7  # a high-confidence retrieval, the case that matters most
        engine.close_loop({'interaction_id': r['interaction_id'], 'success': False})
        last_mean = r['knowledge_graph']['retrieved_doc_confidence'][0]['mean']
    assert last_mean < threshold


def test_caveat_reflects_live_confidence_not_frozen_stored_text(tmp_path) -> None:
    """End-to-end check of both the generation caveat and the fix that
    keeps it from going stale: drive a document's confidence down (caveat
    should appear), then drive it back up with real successes (caveat
    should disappear, tracking the live value) -- and confirm the caveat
    text itself never ends up stored as part of a future "solved example"
    (see CHANGELOG.md, "keep the caveat out of what gets stored")."""
    from moe_model import KG_CAVEAT_PREFIX

    engine = _fresh_engine(tmp_path)
    for _ in range(8):
        r = engine.run('Solve the integral of 2x from 0 to 4.')
        engine.close_loop({'interaction_id': r['interaction_id'], 'success': False})
    assert KG_CAVEAT_PREFIX in r['output']

    for _ in range(3):
        r = engine.run('Solve the integral of 2x from 0 to 4.')
        engine.close_loop({'interaction_id': r['interaction_id'], 'success': True})
    assert KG_CAVEAT_PREFIX not in r['output']

    assert not any(KG_CAVEAT_PREFIX in doc.text for doc in engine.memory.documents)
