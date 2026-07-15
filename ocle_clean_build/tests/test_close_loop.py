from engine import OpenClosedLoopEngine


def test_close_loop_updates_memory_and_planner(tmp_path):
    engine = OpenClosedLoopEngine.from_default_config()
    engine.config['paths']['sqlite_db'] = str(tmp_path / 'engine.db')
    engine.memory.sqlite_path = tmp_path / 'engine.db'
    engine.initialize()

    before_docs = len(engine.memory.documents)
    result = engine.run('Solve the integral of 2x from 0 to 4.')
    response = engine.close_loop({
        'interaction_id': result['interaction_id'],
        'success': True,
        'corrected_output': '16',
        'notes': 'Verified by direct integration',
    })

    assert response['success'] is True
    assert len(engine.memory.documents) == before_docs + 1
    assert engine.planner.transition_counts is not None
    assert engine.planner.transition_probs is not None
