# Packaging Notes

- `pyproject.toml` makes the project installable with `pip install -e .`.
- `MANIFEST.in` keeps documentation, tests, and source files in source distributions.
- `.gitignore` excludes build artifacts, caches, and virtual environments.
- `tests/test_pytest_smoke.py` gives `pytest` a real collected test.
