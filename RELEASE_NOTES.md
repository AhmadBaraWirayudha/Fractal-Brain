# Release Notes

## 0.3.0

Added:
- `adaptive_optimizer.py` -- a pure-stdlib hyperparameter optimization engine (21+
  search strategies, Bayesian belief tracking, GP-based acquisition, constraints,
  multi-objective scoring), merged in from a standalone module and wired to the real
  ecosystem via new `pipeline_optimizer.py`: adapters bridge its knowledge-graph and
  memory dependency-injection points to the real `lkg.py` graph (an `optimizer_state`
  entity tracked the same way `session_intent` is, plus a real "still improving" fact
  per completed trial) and `fractal_brain.storage.Storage` (elite/failed configuration
  persistence, in a dedicated `data/optimizer.db` -- its schema collides with
  `VectorMemoryStore`'s if pointed at the same file as `data/engine.db`, see
  `CHANGELOG.md`). A real, scoped parameter registry (8 parameters, every one mapping
  to an actual `config.yaml` key) replaces the module's own aspirational defaults, and
  a real objective function scores a candidate configuration by actually building and
  running the pipeline against benchmark prompts. See `docs/ADAPTIVE_OPTIMIZER.md` for
  the full design and the two bugs found and fixed while building the objective
  function (it was initially insensitive to every registered parameter; a path
  resolution bug independently left every trial's memory corpus empty).
- `--mode tune` in `hybrid_cli.py` (`python hybrid_cli.py --mode tune --trials 20`).
- `engine.dump_simple_yaml`, the inverse of the existing `parse_simple_yaml`, needed to
  write a trial's overridden configuration to a real temp file.
- `paths.optimizer_db` in `config.yaml`.
- `tests/test_pipeline_optimizer.py`.

## 0.2.0

Added:
- `lkg.py` -- a Living Knowledge Graph (Beta-Bernoulli fact confidence,
  per-source reliability, particle-filtered entity-state tracking), merged in
  from a standalone module and wired into `OpenClosedLoopEngine`/
  `UnifiedAIPipeline`: a `session_intent` entity tracks each turn's task
  intent, and each retrieved document's match to that intent is a fact
  reinforced or penalized once real feedback closes the loop (confidence is
  purely feedback-driven -- a document with no feedback yet reads as
  neutral, not inflated by retrieval alone), attributed to the document's
  origin (bootstrap data, teacher feedback, or a prior successful
  interaction). Surfaced in the engine/pipeline result (`knowledge_graph`),
  the reflection (topic-shift and low-confidence-source risk notes), the
  pipeline trace, and generation itself: `SharedMoEBackbone` now appends a
  caution line when a retrieved document it's about to surface verbatim has
  a poor knowledge-graph track record for this intent, even if its cosine
  similarity was strong. See `docs/UNIFIED_PIPELINE.md` and `CHANGELOG.md`
  for the full design and the functional gaps found and fixed along the way
  (entity-state observation had no public entry point; an earlier version
  of the confidence design had a per-document floor tied to retrieval score;
  the caution line could go stale if echoed back from stored text).
- `knowledge_graph:` section in `config.yaml`, including a
  `low_confidence_threshold` shared by the reflection and the generation
  caveat above.
- `tests/test_lkg.py` (the module's own test suite, plus new coverage for
  the methods added during integration), `tests/test_knowledge_graph_integration.py`,
  and knowledge-graph regression tests in `tests/test_regressions.py`.

## 0.1.0

Added:
- `pyproject.toml`
- `requirements.txt`
- `LICENSE`
- CI workflow
- API reference
- architecture diagram
- training guide
- package version export

These files cover the main missing project-level scaffolding identified in the review.
