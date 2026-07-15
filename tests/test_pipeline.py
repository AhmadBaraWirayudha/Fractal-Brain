from __future__ import annotations

from ai_pipeline import UnifiedAIPipeline


def test_unified_pipeline_returns_trace(tmp_path) -> None:
    pipeline = UnifiedAIPipeline(config_path='config.yaml')
    pipeline.closed_loop.config['paths']['sqlite_db'] = str(tmp_path / 'engine.db')
    pipeline.closed_loop.memory.sqlite_path = tmp_path / 'engine.db'
    pipeline.initialize()

    result = pipeline.run('Solve the integral of 2x from 0 to 4.')
    payload = result.to_dict()

    assert payload['final_output']
    assert payload['closed_loop']['intent']
    assert payload['fractal']['cognitive_context']['sequence_length'] >= 1
    assert any(stage['name'] == 'fractal_cognition' for stage in payload['trace'])
    # Content check, not just shape: the bootstrap dataset has an exact
    # solved example for this query ("...is 16."), so a confident retrieval
    # match exists and the final answer must actually surface it -- not a
    # generic template that ignores what was retrieved. This is the
    # regression test for the canned-fallback-output bug. See CHANGELOG.
    assert '16' in payload['final_output']
    top_doc = payload['closed_loop']['retrieved'][0]
    assert top_doc['score'] >= pipeline.closed_loop.backbone.retrieval_confidence_threshold


def test_feedback_updates_pipeline_state(tmp_path) -> None:
    pipeline = UnifiedAIPipeline(config_path='config.yaml')
    pipeline.closed_loop.config['paths']['sqlite_db'] = str(tmp_path / 'engine.db')
    pipeline.closed_loop.memory.sqlite_path = tmp_path / 'engine.db'
    pipeline.initialize()

    result = pipeline.run('Solve the integral of 2x from 0 to 4.')
    feedback = pipeline.observe_feedback({
        'interaction_id': result.interaction_id,
        'success': True,
        'corrected_output': '16',
        'notes': 'verified',
    })

    assert feedback['success'] is True
    assert feedback['fractal_training']['trained'] is True
