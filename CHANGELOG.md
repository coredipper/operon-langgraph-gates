# Changelog

All notable changes to `operon-langgraph-gates` are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0.0 alpha releases follow [PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

_No changes since `0.1.0`._

## [0.1.0] — 2026-04-30

**First stable release.** This release un-alphas the package: theorem-name
SSoT exposure (Wave 1), A2A round-trip enforcement and Public API
documentation (Wave 2), and version + classifier cutover (Wave 3) all
land here. The committed `0.1.x` public surface is documented in the
README "Public API" section; breaking changes from this point increment
to `0.2.0` per SemVer.

### Added

- Public theorem-name constants `STAGNATION_THEOREM` and `INTEGRITY_THEOREM`
  re-exported from the package root, so downstream consumers can key on
  the emitted theorem names without hard-coding strings that drift if
  `operon-ai` renames the underlying theorem.
- A2A round-trip test (`tests/test_a2a_round_trip.py`) promoting the
  cross-repo binding with `operon-ai`'s certificate codec from
  *informational* to *enforced*. The test drives both gates to emission,
  encodes certificates via `certificate_to_a2a_part`, decodes via
  `certificate_from_a2a_part`, and asserts that `Certificate.verify()`
  returns the same result and that `parameters` are preserved exactly.
  The `safe_certificate_from_a2a_part` graceful-degradation path is
  also covered.
- README "Public API" section documenting the committed `0.1.x` surface
  (classes, methods, theorem-name constants, version attributes) and
  the explicitly-internal complement (underscore-prefixed names,
  `EPHEMERAL_THREAD`, internal gate machinery). Includes the SemVer
  stability commitment for `0.1.0` stable and beyond.
- `tests/test_public_api.py` — focused public-API surface tests:
  theorem constants importable, listed in `__all__`, and equal to
  the module-internal SSoT they re-export.

### Changed

- **Version bump:** `0.1.0a2` → `0.1.0`. Development Status classifier
  bumped from `3 - Alpha` to `4 - Beta` (conservative — alpha-1 was 13
  days ago and there is no production user count yet, so Beta is the
  honest level for the first stable release).
- `pyproject.toml` `operon-ai` dependency tightened from `>=0.36.1`
  (no upper bound) to `>=0.36.1,<0.40`, matching the range the new
  A2A round-trip test was authored against. Widening the pin requires
  re-running the round-trip test against the new range.
- README "Install" line synced with `pyproject.toml`:
  `operon-ai>=0.36.1,<0.40`, `langgraph>=1.0,<2.0`.
- README "Ecosystem note" section relabeled from "out of this repository"
  to "A2A round-trip is enforced" and rewritten to document the binding.
  Maintainer checklist item 3 updated accordingly: A2A is no longer
  conditional on a future test, it's the current state.

### Fixed

- Package docstring (`src/operon_langgraph_gates/__init__.py`) named the wrong
  theorems for both gates: `behavioral_stability` instead of the windowed
  variant emitted since `0.1.0a2`, and `state_integrity_verified` (a name
  that has never existed elsewhere in the codebase) instead of
  `langgraph_state_integrity`. Both corrected.
- `integrity.py` module docstring claimed Paper 5 proves preservation of
  "the paper's `state_integrity_verified` theorem" — but no Operon paper
  has a theorem with that name. Replaced with the honest claim that
  Paper 5 §3 proves preservation of certificate-bearing morphisms in
  general, and this module registers a distinct LangGraph-flavored
  theorem (`langgraph_state_integrity`) that is structurally analogous
  but not itself the subject of a preservation proof in the paper.
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

[Unreleased]: https://github.com/coredipper/operon-langgraph-gates/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/coredipper/operon-langgraph-gates/releases/tag/v0.1.0
[0.1.0a2]: https://github.com/coredipper/operon-langgraph-gates/releases/tag/v0.1.0a2
[0.1.0a1]: https://github.com/coredipper/operon-langgraph-gates/releases/tag/v0.1.0a1
