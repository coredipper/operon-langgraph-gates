# operon-langgraph-gates

> **In-graph** reliability gates for LangGraph — drop-in, cert-emitting.

[LangGraph issue #6731](https://github.com/langchain-ai/langgraph/issues/6731) — *"agent infinite-loops until recursion limit, burns tokens invisibly"* — was closed as **NOT_PLANNED**. LangChain's answer: *"use tool-call limits in middleware."*

This package ships that missing native gate, plus a second one for checkpointer-write integrity. Two primitives. No framework adoption. ~10-line diff on an existing `StateGraph`.

Both gates run **inside** the graph: they route conditional edges and can short-circuit the next step. This is a deliberate contrast to post-hoc observability tools — an observer tells you an agent looped after the fact; a gate stops the loop mid-run and emits a replayable certificate of what fired it.

**At a glance:**

- `StagnationGate` — Bayesian stagnation detection (Paper 4 §4.3, 0.960 on convergence / false-stagnation scenarios) with per-thread state, async + class-callable aware. [See it live.](https://huggingface.co/spaces/coredipper/operon-stagnation-gate)
- `IntegrityGate` — user-defined invariants checked on every wrapped node's output; violations emit a `langgraph_state_integrity` certificate with replayable evidence. Detection-and-certification only; does not repair state.

## Install

```bash
pip install operon-langgraph-gates
```

Requires `operon-ai>=0.36.1,<0.40` and `langgraph>=1.0,<2.0`.

## Quickstart

### Break infinite loops (`StagnationGate`)

```python
from langgraph.graph import StateGraph
from operon_langgraph_gates import StagnationGate

graph = StateGraph(State)
gate = StagnationGate(threshold=0.2, critical_duration=3)
graph.add_node("think", gate.wrap(think_fn))

# Or attach to a conditional edge (route loop-detected runs elsewhere)
graph.add_conditional_edges(
    "think",
    gate.edge(forward="answer", break_to="escalate"),
)

# All certificates the gate has emitted (across all threads)
certs = gate.certificates
# For thread-scoped views use gate.is_stagnant_for(thread_id_) /
# gate.integrals_for(thread_id_). The gate falls back to an internal
# ephemeral-thread id (__ephemeral__) whenever no thread_id can be
# extracted from the LangGraph config/runtime at the wrapped node
# (e.g. no config argument, or a config with no configurable.thread_id);
# this id is an implementation detail, not part of the public API.
```

Backed by [Paper 4 §4.3](https://github.com/coredipper/operon/blob/main/article/paper4/main.pdf): convergence/false-stagnation accuracy **0.960** with real sentence embeddings (all-MiniLM-L6-v2, N = 300 trials). See [`docs/paper-citations.md`](./docs/paper-citations.md) for the full citation record, including the loop-detection caveat and a pointer to the archived benchmark data.

### Catch state drift (`IntegrityGate`)

```python
from langgraph.graph import StateGraph
from operon_langgraph_gates import IntegrityGate

gate = IntegrityGate(invariants=[has_required_user, budget_not_exceeded])

graph = StateGraph(State)
graph.add_node("tool_call", gate.wrap(tool_call_fn))

# Route runs that violated an invariant to a recovery node
graph.add_conditional_edges(
    "tool_call",
    gate.edge(forward="process", break_to="recover"),
)

# Certificates (one per thread, on first violation) are on the gate
certs = gate.certificates
```

Backed by [Paper 4 §4, Table 3](https://github.com/coredipper/operon/blob/main/article/paper4/main.pdf): *in the paper's setup*, the FULL variant (with `DNARepair`) achieves 100% detection and 100% repair of injected state corruption, vs 0%/0% for RAW and GUARDED. **This package is detection-and-certification only — it does not repair state.** It reformulates the idea as a LangGraph-native invariant gate. [Paper 5 §3](https://github.com/coredipper/operon/blob/main/article/paper5/main.pdf) establishes the preservation-under-compilation framework that the gate's certificate follows. See [`docs/paper-citations.md`](./docs/paper-citations.md) for verbatim quotes and the honest caveat.

## Theoretical basis

Both gates are a discrete-state port of the rolling-past reliability loop that has been standard in robotics state estimation since [Kaess et al. 2012](https://dspace.mit.edu/handle/1721.1/71582) (iSAM2) and is formalised in [Dellaert & Kaess 2017](https://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.pdf) (*Factor Graphs for Robot Perception*). A recent [GTSAM blog post](https://gtsam.org/2026/04/21/factor-graphs-and-world-models.html) frames that loop — **STAG**: Sense-Think-Act with Graphs — as a concrete structured instance of an energy-based world model.

### Scope (normative for this package)

What this repository actually commits to under the factor-graph framing:

- **Binding**: `StagnationGate` replay equivalence — a `behavioral_stability_windowed` certificate emitted by the gate must verify identically offline when passed its own parameters. The emitted certificate parameters are in the **severity domain**, not the integral domain: `signal_values = tuple(1.0 - i for i in state.integrals[-critical_duration:])` and `threshold = 1.0 - detection_threshold` (see `src/operon_langgraph_gates/stagnation.py:266-269`), which keeps the replay predicate aligned with `operon_ai`'s shared `behavioral_stability_windowed` verifier semantics (`max(signal_values) ≤ threshold`).
- **Binding**: `IntegrityGate` frozen-result replay — a `langgraph_state_integrity` certificate's parameters are exactly `invariant_results: tuple[(str, bool), ...]` and `all_passed: False` (see `_make_certificate` in `src/operon_langgraph_gates/integrity.py`); replay must reproduce the same boolean vector. Node output is **not** captured in the certificate payload. Two notes on the `name` field in `(name, passed)`: (i) names are **descriptive labels only**, derived from `inv.__name__` at emit time; replay never resolves them back to live invariants, so lambdas (`<lambda>`) and duplicate names do not break replay correctness — they just collapse to lossy labels in logs and UIs. Treat them as descriptive, not as stable identifiers. (ii) exceptions raised inside an invariant are coerced to `False`, and neither the certificate nor the replay surfaces the offending state or the exception detail.
- **Binding**: A2A round-trip equivalence — under the pinned `operon-ai` range in `pyproject.toml` (`>=0.36.1,<0.40`), a certificate emitted by either gate must round-trip through `operon_ai.convergence.a2a_certificate.{certificate_to_a2a_part, certificate_from_a2a_part}` such that `Certificate.verify()` returns the same result and the parameter dict is preserved exactly. Enforced by `tests/test_a2a_round_trip.py`. Tightening the upper bound on the `operon-ai` pin requires re-running this test against the new range.
- **Non-binding**: no cross-agent inference or factor-graph joining is implemented in this package. Nothing below promises any behaviour beyond what is replayable from the emitted certificates.
- **Non-binding**: the factor-graph vocabulary is explanatory. Any analogy mapping in the next subsection can be re-tagged as *analogy only* without breaking any contract in this repository.

### Mapping (explanatory, not normative)

- `StagnationGate` is the discrete-state analogue of a **past-graph fixed-lag smoother**. Concretely: on each turn, `EpiplexityMonitor.measure()` emits a scalar `epiplexic_integral` that is stored in `state.integrals`; the detector fires when the last `critical_duration` integrals fall below `detection_threshold` (`src/operon_langgraph_gates/stagnation.py:250-256`). At emission, that integral slice is converted to the severity domain via `1.0 - integral` and shipped in the certificate as `signal_values`, with the threshold also translated to `1.0 - detection_threshold` (`stagnation.py:266-269`). The offline replay predicate `max(signal_values) ≤ threshold` is therefore a translated, not raw, replay of the gate's decision — the translation is part of the replay contract and is required to match `operon_ai`'s shared `behavioral_stability_windowed` verifier semantics.
- `IntegrityGate` is a **dynamics-residual check**: user-defined invariants play the role of the dynamics model's consistency factors, and a violation at a wrapped node is a positive residual routed onto a conditional edge. The certificate is the replayable record of that residual — specifically the `(name, passed)` vector over invariants and the `all_passed` flag, as emitted by `_make_certificate`. Capturing the offending node output is a deliberate non-goal in the current schema (privacy/redaction would need a separate design); the cert fixes *which* invariants failed, not *what* triggered them.

### Ecosystem note (A2A round-trip is enforced)

Operon's A2A certificate codec (in the [operon repo](https://github.com/coredipper/operon), `operon_ai/convergence/a2a_certificate.py`) transports certificates as `DataPart` payloads and handles graceful degradation for unknown theorems. Under the STAG framing that is a transport-layer analogy of factor joining along shared theorem variables, but the codec does *not* maintain an internal factor graph or perform joint inference.

The `Certificate` objects this package emits are guaranteed to round-trip through that codec under the pinned `operon-ai` range in `pyproject.toml` (`>=0.36.1,<0.40`). The guarantee is enforced by `tests/test_a2a_round_trip.py`, which drives both gates to emission, encodes their certificates via `certificate_to_a2a_part`, decodes via `certificate_from_a2a_part`, and asserts that `Certificate.verify()` returns the same result and that `parameters` are preserved exactly. The same test verifies that `safe_certificate_from_a2a_part` recovers the cert (not `None`) for both gate-emitted theorem names — i.e., the round-trip works even when the receiver opts into graceful-degradation mode.

This was previously informational; promoted to a binding claim in `0.1.0` per the maintainer checklist. If the upper bound on the `operon-ai` pin is widened, the round-trip test must be re-run against the new range before merging.

### Follow-up checklist for maintainers

If the code changes, the bindings in *Scope (normative for this package)* above must be re-verified. Checklist items are keyed to stable symbols, not line numbers, so they survive ordinary refactors:

1. `StagnationGate` — if *either* `_emit_certificate` (the integral-to-severity translation and emitted parameter names `signal_values`, `threshold`) *or* `_observe` (the detection predicate `integral < threshold`, the streak counting, and the `critical_duration` slice at emission) changes, update the *StagnationGate replay equivalence* scope entry and add or update an end-to-end test that exercises both the detection path and the replay predicate so detection changes cannot silently drift past the checklist.
2. `IntegrityGate` — if `_make_certificate` (in `integrity.py`) changes the shape of `invariant_results` or `all_passed`, update the *IntegrityGate frozen-result replay* scope entry; and in the same change, update or add tests that cover: (a) label derivation from `inv.__name__` including the lambda / duplicate-name cases (treated as lossy labels, not identifiers); (b) exception-coerced-to-`False` behaviour; (c) serialization round-trip of `(name, passed)` pairs. If callers ever start round-tripping `name` back to a callable (e.g. an executable replay), introduce an explicit caller-supplied identifier and upgrade the binding before merging.
3. A2A — the round-trip is binding under the current pinned `operon-ai` range (`>=0.36.1,<0.40`) and is enforced by `tests/test_a2a_round_trip.py`. If `certificate_to_a2a_part` / `certificate_from_a2a_part` change shape upstream, re-run the test against the new range; if the upper bound on the pin is widened, the test must be re-run against the new range before merging. If the `safe_certificate_from_a2a_part` graceful-degradation behaviour is removed upstream, the corresponding sub-test must be removed in lockstep with the pin update.
4. Theorem-name changes are **repo-wide**, not README-only. If either `behavioral_stability_windowed` or `langgraph_state_integrity` is renamed in `operon-ai`, the following surfaces must all change in lockstep: this README (the quickstart text and the *Certificate theorem name and verification* section), `src/operon_langgraph_gates/__init__.py` and `stagnation.py` / `integrity.py` (emission sites), `scripts/build_examples.py` (currently references `langgraph_state_integrity` at 3 places), and `huggingface/space-stagnation-gate/app.py` (currently still references the pre-0.36 `behavioral_stability` name — known drift, fix when the theorem-rename PR next touches it). Prefer a single source of truth (e.g. a `_THEOREM` constant re-exported from the package) to prevent this list from growing.

### Honest scope

This is a porting exercise, not new math. What is new is running the loop over symbolic LLM-agent state with a fixed verifier instead of gradient-based smoothing; see [paper 6 appendix §8](https://github.com/coredipper/operon/blob/main/article/paper6/sections/08-factor-graphs.tex) for the full term-by-term mapping, the worked example, and an explicit record of where the analogy stops. The scope-discipline rule inherited from the framing: **factors and topology are fixed; only the theorem set grows** — no learned factors, no horizon > 1 planning graphs.

### Position vs. existing taxonomies (informational)

[Meng et al. 2026](https://www.preprints.org/manuscript/202604.0428/v2) formalize the agent harness as a six-component tuple `H = (E, T, C, S, L, V)` and survey 110+ papers and 23 systems on a Harness Completeness Matrix. Their matrix scores LangGraph at `E✓ T≈ C≈ S≈ L✗ V✗` — Lifecycle Hooks (`L`) and Evaluation Interface (`V`) absent. `operon-langgraph-gates` provides complementary in-graph hooks and certificate-typed evidence at wrapped-node boundaries: `StagnationGate` emits Bayesian stagnation evidence (Paper 4 §4.3), `IntegrityGate` emits state-integrity invariant results (Paper 4 §4.1). The package is an ecosystem complement, not a wholesale L+V retrofit — replacing LangGraph orchestration is an explicit non-goal (see *Scope (normative)* above).

## Certificate theorem name and verification

`StagnationGate` emits certificates with theorem name `behavioral_stability_windowed` (not the core's shared `behavioral_stability`). The two differ in how they verify:

- `behavioral_stability` (shared core): `mean(severities) < threshold`. Loses the per-window structure rolling-integral detection operates on.
- `behavioral_stability_windowed` (shared core, since operon-ai 0.36.0): `max(per_window_severity_means) <= stability_threshold`. Mirrors detection exactly.

Both verifiers are registered in `operon_ai.core.certificate._THEOREM_FN_PATHS`, so deserialized certificates resolve through `_resolve_verify_fn` without this package needing to be imported. Any consumer with `operon-ai>=0.36.0` can round-trip a `behavioral_stability_windowed` certificate correctly.

### Sibling-adapter consumption contract (enforced cross-repo)

The supported way for a downstream Operon adapter to consume this package is exactly: the two public theorem-name constants — `STAGNATION_THEOREM`, `INTEGRITY_THEOREM` — plus the import side-effect of `operon_langgraph_gates.integrity`, which registers `langgraph_state_integrity` in `operon-ai`'s verifier registry. Nothing else (no underscore internals) is part of this contract.

It is enforced from the *consumer* side, not just asserted here: `operon-ai`'s `TestOperonLanggraphGatesDogfood` (`tests/unit/convergence/test_gascity_adapter.py` and `test_agentflow_adapter.py`, [operon-ai #182](https://github.com/coredipper/operon/pull/182)) installs this package as a CI dependency and asserts that a certificate produced through the gascity / agentflow adapters under each constant survives the Beads/Dolt audit-trail JSON render **and** a `certificate_to_dict` → `certificate_from_dict` → `verify()` round-trip with agreement. That test is the executable proof of the portability claim in operon-ai's [external-frameworks §8.4 landscape memo](https://github.com/coredipper/operon/blob/main/docs/site/external-frameworks.md); this section is its consumer-side counterpart. Widening the `operon-ai` pin in `pyproject.toml` requires that contract test (and `tests/test_a2a_round_trip.py`) to pass against the new range.

### Breaking change from pre-alpha prototypes

Earlier builds emitted certificates with theorem name `behavioral_stability`, bound to a locally-attached `_verify_fn`. That shape was semantically wrong — the shared verifier is flat-mean-based, so any cert round-tripped through serialization would silently revert to the wrong replay logic. Consumers that key on `certificate.theorem == "behavioral_stability"` must update to `"behavioral_stability_windowed"`. No migration path; alpha.

## Try it — HuggingFace Space

[**Operon StagnationGate Demo**](https://huggingface.co/spaces/coredipper/operon-stagnation-gate) — interactive page: pick a preset (identical, diverse, noisy, slow drift), tune the gate parameters, watch `is_stagnant` flip and the certificate appear. Deterministic text trajectories — no LLM calls.

## Examples

- [`examples/01_stagnation_breaks_loop.ipynb`](./examples/01_stagnation_breaks_loop.ipynb) — reproduces issue #6731 pathology, then fixes it with a ten-line diff.
- [`examples/02_integrity_catches_drift.ipynb`](./examples/02_integrity_catches_drift.ipynb) — a three-node graph silently corrupts state; `IntegrityGate` catches it with replayable evidence.

## Public API

The committed surface for `0.1.x` is small and explicit:

- **Classes:** `StagnationGate`, `IntegrityGate`.
- **Methods on each gate:** `wrap(node_fn)`, `edge(forward, break_to)`, `certificates`, `reset()`. `StagnationGate` additionally exposes the global property `is_stagnant`, the per-thread methods `is_stagnant_for(thread_id)` and `integrals_for(thread_id)`. `IntegrityGate` exposes the global property `is_violated` and the per-thread method `is_violated_for(thread_id)`.
- **Theorem-name constants:** `STAGNATION_THEOREM` (`"behavioral_stability_windowed"`), `INTEGRITY_THEOREM` (`"langgraph_state_integrity"`). Use these instead of hard-coding strings; the underlying private constants `_THEOREM` / `_WINDOWED_THEOREM` are the SSoT and the public re-exports forward to them.
- **Module attributes:** `__version__`, `__all__`.

Anything underscore-prefixed is **internal** and may change without notice within `0.1.x`. Notable internals callers sometimes touch:

- `_THEOREM` / `_WINDOWED_THEOREM` — module-internal SSoT for theorem names; use `STAGNATION_THEOREM` / `INTEGRITY_THEOREM` instead.
- `EPHEMERAL_THREAD` (in `_common.py`) — the internal fallback thread-id used whenever no `thread_id` can be extracted from the LangGraph config or runtime at the wrapped node. Confirmed internal in `0.1.0a2` (Roborev #903); see `_common.py:thread_id()` for exact semantics.
- `_observe`, `_emit_certificate`, `_make_certificate`, `_thread_state` — internal gate machinery; touching these from a caller breaks the binding contract in *Scope (normative)*.

**Stability commitment:**

- `0.1.0` is the first stable release; the `0.1.x` series preserves the public surface above. Any deprecations of theorem names will go through `operon-ai`'s theorem registry and be documented in `CHANGELOG.md`.
- Breaking changes increment to `0.2.0` per [SemVer](https://semver.org/).
- The cross-repo binding to `operon-ai`'s A2A codec is enforced by `tests/test_a2a_round_trip.py` against the pinned range in `pyproject.toml`; widening that pin requires re-running the test against the new range.

## Status

**Beta — `0.1.0`.** First stable release. Public surface is documented in the section above; `0.1.x` patches preserve it, breaking changes increment to `0.2.0` per [SemVer](https://semver.org/). The cross-repo A2A binding is enforced under the pinned `operon-ai>=0.36.1,<0.40` range. Feedback welcome via Issues.

## License

MIT — see [LICENSE](./LICENSE).
