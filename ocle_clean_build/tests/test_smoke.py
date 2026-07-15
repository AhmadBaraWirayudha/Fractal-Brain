from engine import OpenClosedLoopEngine

def test_smoke(tmp_path):
    engine = OpenClosedLoopEngine.from_default_config()
    engine.config['paths']['sqlite_db'] = str(tmp_path / 'engine.db')
    engine.memory.sqlite_path = tmp_path / 'engine.db'
    engine.initialize()
    result = engine.run('Solve the integral of 2x from 0 to 4.')
    assert result['output']
    assert result['retrieved']
