"""
run_tests.py -- runs the full test suite, installing pytest first if it
isn't already available.

requirements.txt is intentionally empty (this project has zero *runtime*
dependencies), but pytest is a *test-only* dependency and won't be present
on a fresh machine unless something installs it. Rather than making that a
manual prerequisite, this script checks for it and installs it automatically
so `run_demo.bat` / `demo_showcase.py --run-tests` stay one-click.

Runs both halves of the suite, matching .github/workflows/ci.yml exactly:
    1. tests/test_smoke.py   (fractal_brain's own 137-check script)
    2. pytest -q             (unified pipeline, OCLE engine, CLI, regressions)

Usage:
    python run_tests.py
"""
from __future__ import annotations

import subprocess
import sys


def ensure_pytest_installed() -> bool:
    try:
        import pytest  # noqa: F401

        return True
    except ImportError:
        print("  pytest isn't installed yet -- installing it now...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "pytest"])
        if result.returncode != 0:
            print()
            print("  Automatic install failed. Install it yourself with:")
            print(f"    {sys.executable} -m pip install pytest")
            return False
        print()
        return True


def main() -> int:
    print("--- fractal_brain smoke checks (137 checks, standalone script) ---")
    smoke_result = subprocess.run([sys.executable, "tests/test_smoke.py"])

    print()
    print("--- pytest suite (unified pipeline, OCLE engine, CLI, regressions) ---")
    if not ensure_pytest_installed():
        return 1
    pytest_result = subprocess.run([sys.executable, "-m", "pytest", "-q"])

    return 0 if (smoke_result.returncode == 0 and pytest_result.returncode == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
