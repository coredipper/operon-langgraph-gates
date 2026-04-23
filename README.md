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

Requires `operon-ai>=0.34.4` and `langgraph>=1.0`.

## Quickstart

### Break infinite loops (`StagnationGate`)

```python
from langgraph.graph import StateGraph
from operon_langgraph_gates import StagnationGate

graph = StateGraph(State)
graph.add_node("think", StagnationGate.wrap(think_fn, threshold=0.1, history=10))

# Or attach to a conditional edge (route loop-detected runs elsewhere)
graph.add_conditional_edges(
    "think",
    StagnationGate.edge(forward="answer", break_to="escalate", threshold=0.1),
)

# Certificates collected from the run
certs = StagnationGate.collect(graph)
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
- **Binding**: `IntegrityGate` frozen-result replay — a `langgraph_state_integrity` certificate carries `(invariant_name, passed)` pairs for the first violating node output, and replay must reproduce the same boolean vector. Two preconditions callers must satisfy: (i) invariants must be callables with unique, stable `__name__` attributes — lambdas and duplicate-named callables are **out of the replay contract** because the codec identifies invariants by `inv.__name__` (`src/operon_langgraph_gates/integrity.py:99, 204`); (ii) exceptions raised inside an invariant are coerced to `False`, and neither the certificate nor the replay surfaces the offending state or the exception detail.
- **Non-binding**: no cross-agent inference or factor-graph joining is implemented in this package. Nothing below promises any behaviour beyond what is replayable from the emitted certificates.
- **Non-binding**: the factor-graph vocabulary is explanatory. Any analogy mapping in the next subsection can be re-tagged as *analogy only* without breaking any contract in this repository.

### Mapping (explanatory, not normative)

- `StagnationGate` is the discrete-state analogue of a **past-graph fixed-lag smoother**. Concretely: on each turn, `EpiplexityMonitor.measure()` emits a scalar `epiplexic_integral` that is stored in `state.integrals`; the detector fires when the last `critical_duration` integrals fall below `detection_threshold` (`src/operon_langgraph_gates/stagnation.py:250-256`). At emission, that integral slice is converted to the severity domain via `1.0 - integral` and shipped in the certificate as `signal_values`, with the threshold also translated to `1.0 - detection_threshold` (`stagnation.py:266-269`). The offline replay predicate `max(signal_values) ≤ threshold` is therefore a translated, not raw, replay of the gate's decision — the translation is part of the replay contract and is required to match `operon_ai`'s shared `behavioral_stability_windowed` verifier semantics.
- `IntegrityGate` is a **dynamics-residual check**: user-defined invariants play the role of the dynamics model's consistency factors, and a violation at a wrapped node is a positive residual routed onto a conditional edge. The certificate is the replayable record of that residual — specifically, the first violating node's output plus the `(name, passed)` vector over invariants.

### Ecosystem note (out of this repository)

Operon's A2A certificate codec (in the [operon repo](https://github.com/coredipper/operon), `operon_ai/convergence/a2a_certificate.py`) transports certificates as `DataPart` payloads and handles graceful degradation for unknown theorems. Under the STAG framing that is a transport-layer analogy of factor joining along shared theorem variables, but the codec does *not* maintain an internal factor graph or perform joint inference, and **no A2A integration exists in this gates repository**. The `Certificate` objects this package emits are structurally compatible with that codec because both sides depend on `operon-ai` for the serialisation shape, but **there is no cross-repo compatibility test or pinned version pair in this package**, so treat this as motivation only — not as an enforced interface contract. If you need A2A-round-trip guarantees, pin a specific `operon-ai` version and add your own compatibility test.

### Follow-up checklist for maintainers

If the code changes, the bindings in *Scope (normative for this package)* above must be re-verified. Concretely:

1. If `stagnation.py:266-269` changes (integral-to-severity translation or threshold remapping), update the *StagnationGate replay equivalence* bullet and add or update a certificate-schema test.
2. If `integrity.py:204` changes the `(inv.__name__, passed)` shape, update the *IntegrityGate frozen-result replay* bullet; if the identifier scheme becomes unstable (e.g. `id(inv)`), loosen the binding or introduce an explicit caller-supplied identifier.
3. If A2A compatibility becomes a binding claim (e.g. a pinned `operon-ai` version range + test), move the ecosystem note out of *out-of-repo* and into *Scope (normative)* with the pinned range listed; otherwise keep it informational.

### Honest scope

This is a porting exercise, not new math. What is new is running the loop over symbolic LLM-agent state with a fixed verifier instead of gradient-based smoothing; see [paper 6 appendix §8](https://github.com/coredipper/operon/blob/main/article/paper6/sections/08-factor-graphs.tex) for the full term-by-term mapping, the worked example, and an explicit record of where the analogy stops. The scope-discipline rule inherited from the framing: **factors and topology are fixed; only the theorem set grows** — no learned factors, no horizon > 1 planning graphs.

## Certificate theorem name and verification

`StagnationGate` emits certificates with theorem name `behavioral_stability_windowed` (not the core's shared `behavioral_stability`). The two differ in how they verify:

- `behavioral_stability` (shared core): `mean(severities) < threshold`. Loses the per-window structure rolling-integral detection operates on.
- `behavioral_stability_windowed` (shared core, since operon-ai 0.36.0): `max(per_window_severity_means) <= stability_threshold`. Mirrors detection exactly.

Both verifiers are registered in `operon_ai.core.certificate._THEOREM_FN_PATHS`, so deserialized certificates resolve through `_resolve_verify_fn` without this package needing to be imported. Any consumer with `operon-ai>=0.36.0` can round-trip a `behavioral_stability_windowed` certificate correctly.

### Breaking change from pre-alpha prototypes

Earlier builds emitted certificates with theorem name `behavioral_stability`, bound to a locally-attached `_verify_fn`. That shape was semantically wrong — the shared verifier is flat-mean-based, so any cert round-tripped through serialization would silently revert to the wrong replay logic. Consumers that key on `certificate.theorem == "behavioral_stability"` must update to `"behavioral_stability_windowed"`. No migration path; alpha.

## Try it — HuggingFace Space

[**Operon StagnationGate Demo**](https://huggingface.co/spaces/coredipper/operon-stagnation-gate) — interactive page: pick a preset (identical, diverse, noisy, slow drift), tune the gate parameters, watch `is_stagnant` flip and the certificate appear. Deterministic text trajectories — no LLM calls.

## Examples

- [`examples/01_stagnation_breaks_loop.ipynb`](./examples/01_stagnation_breaks_loop.ipynb) — reproduces issue #6731 pathology, then fixes it with a ten-line diff.
- [`examples/02_integrity_catches_drift.ipynb`](./examples/02_integrity_catches_drift.ipynb) — a three-node graph silently corrupts state; `IntegrityGate` catches it with replayable evidence.

## Status

**Alpha.** API may change before `0.1.0` stable. Feedback welcome via Issues.

## License

MIT — see [LICENSE](./LICENSE).
