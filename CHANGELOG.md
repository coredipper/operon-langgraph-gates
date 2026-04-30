# Changelog

All notable changes to `operon-langgraph-gates` are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0.0 alpha releases follow [PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

### Added

- Public theorem-name constants `STAGNATION_THEOREM` and `INTEGRITY_THEOREM`
  re-exported from the package root, so downstream consumers can key on
  the emitted theorem names without hard-coding strings that drift if
  `operon-ai` renames the underlying theorem.

### Fixed

- Package docstring (`src/operon_langgraph_gates/__init__.py`) named the wrong
  theorems for both gates: `behavioral_stability` instead of the windowed
  variant emitted since `0.1.0a2`, and `state_integrity_verified` (a name
  that has never existed elsewhere in the codebase) instead of
  `langgraph_state_integrity`. Both corrected.
- HuggingFace Space app (`huggingface/space-stagnation-gate/app.py`) referenced
  the pre-`0.1.0a2` theorem name `behavioral_stability` in its top-level
  docstring. README maintainer-checklist item 4 had explicitly flagged this
  as known drift; corrected to `behavioral_stability_windowed`.
- Example-rendering script (`scripts/build_examples.py`) had two prose blocks
  describing emitted certificates as `behavioral_stability` rather than the
  windowed name. Re-rendering example notebooks will pick up the fix.
- `stagnation.py` module docstring and one inline comment named
  `behavioral_stability` where the windowed variant was meant; updated to
  `behavioral_stability_windowed`. The intentional comparative reference
  at `stagnation.py:278` ("distinct from the shared `behavioral_stability`")
  is preserved as-is — it is naming the older theorem on purpose.

## [0.1.0a2] — 2026-04-19

### Changed

- **Breaking** for any consumer that keyed on the pre-alpha theorem name:
  `StagnationGate` now emits certificates with theorem name
  `behavioral_stability_windowed` rather than `behavioral_stability`.
  The shared core verifier `behavioral_stability` is flat-mean-based;
  the windowed variant is `max(per_window_severity_means) <=
  stability_threshold`, which mirrors the gate's detection logic
  exactly. Round-trip serialization through `Certificate.from_theorem`
  resolves the correct verifier without the gate package needing to
  bind to any upstream symbol beyond the public `Certificate` class.
- Cert semantics aligned with `operon-ai>=0.36.0` public API
  (`resolve_verify_fn`, `_THEOREM_FN_PATHS`).
- Ephemeral-thread fallback (`__ephemeral__`) narrowed to fire only when
  no `thread_id` can be extracted from the LangGraph config or runtime
  at the wrapped node — explicitly internal, not public API.
- `EPHEMERAL_THREAD` constant marked internal (per Roborev #903).

### Added

- "Theoretical basis" section in README citing the factor-graph SLAM
  lineage (Kaess et al. 2012, Dellaert & Kaess 2017, Dellaert 2026 STAG)
  with hard scope line against learned factors.
- Maintainer follow-up checklist keyed to stable symbols (not line
  numbers), surviving ordinary refactors.
- CI hardening: workflow actions pinned to SHAs, `ruff format --check`
  enforced on Python 3.11 / 3.13.

## [0.1.0a1] — 2026-04-17

### Added

- First public alpha. Two reliability primitives for LangGraph:
  - `StagnationGate` — Bayesian stagnation detection on conditional
    edges or node outputs. Per-thread state, async + class-callable
    aware. Emits `behavioral_stability` certificates (renamed in
    `0.1.0a2` — see above).
  - `IntegrityGate` — runtime invariant check on wrapped node outputs.
    First-violation-per-thread certificate emission with the
    `(name, passed)` invariant-result vector. Detection-and-
    certification only; does not repair state.
- Two example notebooks:
  - `examples/01_stagnation_breaks_loop.ipynb`
  - `examples/02_integrity_catches_drift.ipynb`
- HuggingFace Space deployed at
  https://huggingface.co/spaces/coredipper/operon-stagnation-gate.
- Backed by Paper 4 §4.3 (96% real-embedding accuracy) and §4.1 (3/3
  DNA-repair integrity benchmarks); cert preservation framework from
  Paper 5 §3.

[Unreleased]: https://github.com/coredipper/operon-langgraph-gates/compare/v0.1.0a2...HEAD
[0.1.0a2]: https://github.com/coredipper/operon-langgraph-gates/releases/tag/v0.1.0a2
[0.1.0a1]: https://github.com/coredipper/operon-langgraph-gates/releases/tag/v0.1.0a1
