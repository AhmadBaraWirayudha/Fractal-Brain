# Hybrid Cognitive AI Pipeline

This workspace now runs as one end-to-end AI system rather than two separate demos.

## Pipeline stages

1. **Normalize** the input text.
2. **Retrieve** relevant memory records from the closed-loop engine.
3. **Decompose** the task into subgoals.
4. **Plan** the action sequence.
5. **Generate** the final answer with the closed-loop decoder.
6. **Score and adapt** with FractalBrain.
7. **Reflect** on confidence, retrieved memory, and correction signals.
8. **Learn from feedback** when the result is corrected.

## Run the unified pipeline

```bash
python hybrid_cli.py --mode pipeline
```

## Run the legacy subsystem demos

```bash
python hybrid_cli.py --mode closed-loop
python hybrid_cli.py --mode fractal
python -m fractal_brain --demo
```

## Compatibility

- `hybrid_engine.py` now points to the unified pipeline.
- Root-level closed-loop modules are preserved for `from engine import ...`.
- `ocle_clean_build/` still re-exports the root closed-loop modules for package-style imports.
- Relative paths in `config.yaml` resolve from the file location, so the bundled bootstrap dataset works from any working directory.

## Session mode

Run a multi-turn session by separating prompts with `||`:

```bash
python hybrid_cli.py --mode session --text "First question || Second question || Third question"
```

## Quick demo / one-click launcher

- **Windows:** double-click `run_demo.bat` for a menu (showcase demo, full
  test suite, ask your own question, or open the interview prep notes).
- **Any OS:** `python demo_showcase.py` runs the same showcase directly --
  four short, live scenes (retrieval-grounded answers, honest fallback on
  out-of-domain questions, a multi-turn session, and teaching it a new fact
  on the spot), using its own throwaway database so it's reproducible and
  never touches `data/engine.db`. Add `--run-tests` to also run the full
  test suite afterwards, or `--pause` to step through it manually.
- `docs/INTERVIEW_PREP.md` -- talking points, likely questions, and honest
  limitations if you're presenting or discussing this project.

## Docs

- [`docs/UNIFIED_PIPELINE.md`](docs/UNIFIED_PIPELINE.md) -- component map, CLI
  usage, config reference, and known limitations for the layer described
  above (`ai_pipeline.py` / `hybrid_cli.py` / `ocle_clean_build/`).
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md),
  [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) -- the `fractal_brain/` package itself.

