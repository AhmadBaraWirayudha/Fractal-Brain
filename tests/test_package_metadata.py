from fractal_brain import __version__, FractalBrain, set_seed


def test_version_and_imports() -> None:
    assert __version__ == "0.1.0"
    set_seed(1)
    brain = FractalBrain(vocab_size=16, d_model=8, num_experts=2)
    assert brain.vocab_size == 16


from pathlib import Path


def test_packaging_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists()
    assert (root / "LICENSE").exists()
    assert (root / "MANIFEST.in").exists()
