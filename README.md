# operon-langgraph-gates

> Reliability primitives for LangGraph — drop-in, cert-emitting.

[LangGraph issue #6731](https://github.com/langchain-ai/langgraph/issues/6731) — *"agent infinite-loops until recursion limit, burns tokens invisibly"* — was closed as **NOT_PLANNED**. LangChain's answer: *"use tool-call limits in middleware."*

This package ships that missing native gate, plus a second one for checkpointer-write integrity. Two primitives. No framework adoption. ~10-line diff on an existing `StateGraph`.

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

Backed by [Paper 4 §4, Table 3](https://github.com/coredipper/operon/blob/main/article/paper4/main.pdf): the FULL integrity variant detects and repairs **100%** of injected state-corruption sites (vs 0% for RAW and GUARDED variants). [Paper 5 §3](https://github.com/coredipper/operon/blob/main/article/paper5/main.pdf) proves the certificate is preserved under compilation. See [`docs/paper-citations.md`](./docs/paper-citations.md) for verbatim quotes and the honest caveat: this gate is a LangGraph reformulation of the paper's idea, not a direct reuse of `DNARepair`.

## Status

**Alpha.** API may change before `0.1.0` stable. Feedback welcome via Issues.

## License

MIT — see [LICENSE](./LICENSE).
