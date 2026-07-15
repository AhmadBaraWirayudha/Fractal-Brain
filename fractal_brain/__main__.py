from __future__ import annotations

import argparse
from typing import Sequence

from . import FractalBrain, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m fractal_brain',
        description='Run a tiny fractal_brain demo or inspect the package version.',
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run a small deterministic forward pass and print the output summary.',
    )
    parser.add_argument(
        '--version',
        action='store_true',
        help='Print the installed package version.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from .version import __version__
        print(__version__)
        return 0

    if args.demo:
        set_seed(7)
        brain = FractalBrain(vocab_size=32, d_model=8, num_experts=4)
        logits, loss = brain.step([1, 2, 3], [0.0] * 32)
        print(f'logits_rows={logits.rows} logits_cols={logits.cols} loss={loss:.6f}')
        return 0

    parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
