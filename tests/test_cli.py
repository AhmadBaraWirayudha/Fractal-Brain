from __future__ import annotations

import json
import subprocess
import sys


def test_module_demo_runs() -> None:
    result = subprocess.run(
        [sys.executable, '-m', 'fractal_brain', '--demo'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert 'logits_rows=' in result.stdout
    assert 'loss=' in result.stdout


def test_module_version_runs() -> None:
    result = subprocess.run(
        [sys.executable, '-m', 'fractal_brain', '--version'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip()


def test_hybrid_cli_pipeline_mode_runs() -> None:
    # hybrid_cli.py is the documented entry point for the unified pipeline
    # but previously had zero test coverage (only `python -m fractal_brain`
    # was subprocess-tested here). Runs against the real config/bootstrap
    # dataset -- safe to run repeatedly because bootstrap() is now
    # idempotent (see CHANGELOG / test_bootstrap_is_idempotent).
    import os

    env = dict(os.environ)
    env['PYTHONPATH'] = '.'
    result = subprocess.run(
        [
            sys.executable, 'hybrid_cli.py',
            '--mode', 'pipeline',
            '--text', 'Solve the integral of 2x from 0 to 4.',
            '--config', 'config.yaml',
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    payload = json.loads(result.stdout)
    assert payload['final_output']
    assert '16' in payload['final_output']
