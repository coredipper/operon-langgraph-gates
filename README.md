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

### Catch checkpointer drift (`IntegrityGate`)

```python
from langgraph.checkpoint.postgres import PostgresSaver
from operon_langgraph_gates import IntegrityGate

checkpointer = IntegrityGate.wrap(
    PostgresSaver.from_conn_string("..."),
    invariants=[MyState.__annotations__, my_schema_check],
)
graph = workflow.compile(checkpointer=checkpointer)
```

Backed by [Paper 4 §4, Table 3](https://github.com/coredipper/operon/blob/main/article/paper4/main.pdf): *in the paper's setup*, the FULL variant (with `DNARepair`) achieves 100% detection and 100% repair of injected state corruption, vs 0%/0% for RAW and GUARDED. **This package is detection-and-certification only — it does not repair state.** It reformulates the idea as a LangGraph-native invariant gate. [Paper 5 §3](https://github.com/coredipper/operon/blob/main/article/paper5/main.pdf) establishes the preservation-under-compilation framework that the gate's certificate follows. See [`docs/paper-citations.md`](./docs/paper-citations.md) for verbatim quotes and the honest caveat.

## Try it — HuggingFace Space

[**Operon StagnationGate Demo**](https://huggingface.co/spaces/coredipper/operon-stagnation-gate) — interactive page: pick a preset (identical, diverse, noisy, slow drift), tune the gate parameters, watch `is_stagnant` flip and the certificate appear. Deterministic text trajectories — no LLM calls.

## Examples

- [`examples/01_stagnation_breaks_loop.ipynb`](./examples/01_stagnation_breaks_loop.ipynb) — reproduces issue #6731 pathology, then fixes it with a ten-line diff.
- [`examples/02_integrity_catches_drift.ipynb`](./examples/02_integrity_catches_drift.ipynb) — a three-node graph silently corrupts state; `IntegrityGate` catches it with replayable evidence.

## Status

**Alpha.** API may change before `0.1.0` stable. Feedback welcome via Issues.

## License

MIT — see [LICENSE](./LICENSE).
