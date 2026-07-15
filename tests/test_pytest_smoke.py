"""Pytest-compatible smoke test for standard tooling."""
from fractal_brain import FractalBrain, set_seed, __version__


def test_basic_import_and_step() -> None:
    assert __version__ == "0.1.0"
    set_seed(7)
    brain = FractalBrain(vocab_size=32, d_model=8, num_experts=2)
    logits, loss = brain.step([1, 2, 3], [0.0] * 32)
    assert (logits.rows, logits.cols) == (3, 32)
    assert isinstance(loss, float)
