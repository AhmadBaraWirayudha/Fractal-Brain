from __future__ import annotations

from ai_pipeline import UnifiedAIPipeline


def test_session_pipeline_tracks_reflection(tmp_path) -> None:
    pipeline = UnifiedAIPipeline(config_path='config.yaml')
    pipeline.closed_loop.config['paths']['sqlite_db'] = str(tmp_path / 'engine.db')
    pipeline.closed_loop.memory.sqlite_path = tmp_path / 'engine.db'
    pipeline.initialize()

    result = pipeline.run_session([
        'Solve the integral of 2x from 0 to 4.',
        'Explain the method in one sentence.',
    ])

    assert result['turn_count'] == 2
    assert result['average_loss'] >= 0
    assert result['last_reflection'] is not None
    assert 'confidence' in result['last_reflection']
    assert all('reflection' in turn for turn in result['turns'])
    assert all('feedback' in turn for turn in result['turns'])


def test_teach_from_example_updates_memory(tmp_path) -> None:
    pipeline = UnifiedAIPipeline(config_path='config.yaml')
    pipeline.closed_loop.config['paths']['sqlite_db'] = str(tmp_path / 'engine.db')
    pipeline.closed_loop.memory.sqlite_path = tmp_path / 'engine.db'
    pipeline.initialize()

    feedback = pipeline.teach_from_example(
        'Solve the integral of 2x from 0 to 4.',
        'The answer is 16.',
        notes='ground-truth example',
    )

    assert feedback['success'] is True
    assert feedback['fractal_training']['trained'] is True
    assert feedback['fractal_training']['lesson_doc_id']
