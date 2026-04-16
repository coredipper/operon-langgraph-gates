"""Build the example Jupyter notebooks from code-as-data.

Run with:

    .venv/bin/python scripts/build_examples.py            # source only
    .venv/bin/python scripts/build_examples.py --execute  # also run cells

The ``--execute`` flag runs each notebook end-to-end and strips the
per-run ``metadata.execution`` timestamp fields (``iopub.execute_input``,
``shell.execute_reply``, etc.) so a no-op regenerate-and-execute is a
true byte-for-byte no-op on git. Keep the notebook *source* reviewable
as plain Python strings rather than pages of JSON; generated ``.ipynb``
files live under ``examples/``.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(dedent(text).strip())


def _code(text: str) -> nbformat.NotebookNode:
    return new_code_cell(dedent(text).strip())


def _assign_stable_ids(cells: list[nbformat.NotebookNode], prefix: str) -> None:
    """Override ``nbformat``'s per-run random cell IDs with stable ones so
    regenerating a notebook produces a diff only on actual content changes.
    """
    for idx, cell in enumerate(cells):
        cell["id"] = f"{prefix}-{idx:02d}"


# ---------------------------------------------------------------------------
# 01 — Stagnation breaks a loop
# ---------------------------------------------------------------------------

NB1_CELLS = [
    _md("""
        # Breaking a LangGraph infinite loop with `StagnationGate`

        [LangGraph issue #6731](https://github.com/langchain-ai/langgraph/issues/6731) —
        *"agent infinite-loops until recursion limit, burns tokens invisibly"* —
        was closed as **NOT_PLANNED**. LangChain's suggested answer is
        *"use tool-call limits in middleware."*

        `StagnationGate` is the missing native gate: it measures output novelty
        on each node invocation and, once it sees repeated low-novelty
        readings in a row, flips a per-thread flag that a conditional edge
        can route on — plus it emits a signed `behavioral_stability`
        certificate with the evidence that fired it.

        This notebook shows the pathology in plain LangGraph, then adds the
        gate with a ten-line diff.
    """),
    _code("""
        import sys
        import langgraph
        import operon_langgraph_gates as olg

        print(f"Python     : {sys.version.split()[0]}")
        print(f"langgraph  : {getattr(langgraph, '__version__', 'unknown')}")
        print(f"olg        : {olg.__version__}")
    """),
    _md("""
        ## Baseline — the loop hits the recursion limit

        A single `think` node that returns the same output every turn, with
        a conditional edge that always routes back to itself. No gate.
    """),
    _code("""
        from typing_extensions import TypedDict

        from langgraph.errors import GraphRecursionError
        from langgraph.graph import END, START, StateGraph


        class LoopState(TypedDict):
            turn: int
            answer: str


        def think(state: LoopState) -> LoopState:
            return {"turn": state["turn"] + 1, "answer": "same output every turn"}


        def always_loop(_state: LoopState) -> str:
            return "think"


        baseline = StateGraph(LoopState)
        baseline.add_node("think", think)
        baseline.add_edge(START, "think")
        baseline.add_conditional_edges("think", always_loop)
        baseline_app = baseline.compile()

        try:
            baseline_app.invoke({"turn": 0, "answer": ""}, {"recursion_limit": 5})
        except GraphRecursionError as e:
            print(f"GraphRecursionError fired: {e}")
    """),
    _md("""
        ## With `StagnationGate` — same graph, one gate, terminates

        Only two calls change: `gate.wrap(think, text_extractor=...)` replaces
        the raw node function, and `gate.edge(forward=..., break_to=...)`
        replaces the always-loop router. The `text_extractor` picks the field
        we want the gate to measure — here, just the `answer` field, so the
        incrementing `turn` counter doesn't add string drift that masks
        repetition.
    """),
    _code("""
        from operon_langgraph_gates import StagnationGate


        def escalate(state: LoopState) -> LoopState:
            return {"turn": state["turn"], "answer": "escalated"}


        gate = StagnationGate(threshold=0.2, critical_duration=2, window_size=3)

        gated = StateGraph(LoopState)
        gated.add_node(
            "think",
            gate.wrap(think, text_extractor=lambda out: out["answer"]),
        )
        gated.add_node("escalate", escalate)
        gated.add_edge(START, "think")
        gated.add_conditional_edges(
            "think",
            gate.edge(forward="think", break_to="escalate"),
        )
        gated.add_edge("escalate", END)
        gated_app = gated.compile()

        result = gated_app.invoke({"turn": 0, "answer": ""}, {"recursion_limit": 25})
        print(f"final answer: {result['answer']!r}")
        print(f"turns taken : {result['turn']}")
    """),
    _md("""
        ## Inspect the certificate

        On the turn where stagnation was first detected, the gate emits one
        `behavioral_stability` certificate into `gate.certificates`. The
        certificate is a frozen evidence snapshot — `cert.verify()` replays
        the saved severity values against the saved threshold and returns
        `(holds, details)`. For a stagnation cert, `holds == False` — the
        claim "stability held" did *not* hold.
    """),
    _code("""
        assert len(gate.certificates) == 1, "expected exactly one certificate"
        cert = gate.certificates[0]
        print(f"theorem    : {cert.theorem}")
        print(f"source     : {cert.source}")
        print(f"conclusion : {cert.conclusion}")

        result = cert.verify()
        print(f"verify.holds: {result.holds}")
        print(f"verify.evidence: {result.evidence}")
    """),
    _md("""
        ## Takeaway

        Ten lines of diff turned a loop-until-crash into a routed exit
        with a replayable evidence artifact. The gate is per-thread and
        async-aware out of the box; certificates are serializable, so the
        same evidence can be re-verified downstream (audit log, test
        assertion, alert) without re-running the agent.

        See `02_integrity_catches_drift.ipynb` for the companion
        `IntegrityGate` that checks invariants at node boundaries.
    """),
]


# ---------------------------------------------------------------------------
# 02 — Integrity catches drift
# ---------------------------------------------------------------------------

NB2_CELLS = [
    _md("""
        # Catching state drift with `IntegrityGate`

        Long agent runs silently corrupt state: a node forgets to set a
        required field, a tool mutates a value out of range, a schema
        assumption breaks after a refactor and the downstream nodes keep
        happily running on garbage. The pathology Paper 4 §4 calls out —
        "RAW and GUARDED are completely blind to genome corruption" — maps
        straight onto LangGraph state graphs.

        `IntegrityGate` runs a list of invariants after each wrapped node and
        emits a `langgraph_state_integrity` certificate the first time any
        one fails (per thread). A conditional edge can then route the run
        to a recovery / escalation path instead of letting corruption
        propagate.
    """),
    _code("""
        import sys
        import langgraph
        import operon_langgraph_gates as olg

        print(f"Python     : {sys.version.split()[0]}")
        print(f"langgraph  : {getattr(langgraph, '__version__', 'unknown')}")
        print(f"olg        : {olg.__version__}")
    """),
    _md("""
        ## Baseline — a node writes invalid values into state

        A three-node pipeline: `fetch` populates state, `transform` mutates
        it, `respond` formats the final output. `transform` has a bug: it
        explicitly overwrites `user_id` with `None` and pushes `budget`
        negative. Under LangGraph's state semantics those partial-update
        values are then merged into the graph state, so the downstream
        `respond` node reads the corrupted values and emits a broken
        response without any error.
    """),
    _code("""
        from typing_extensions import TypedDict

        from langgraph.graph import END, START, StateGraph


        class AppState(TypedDict, total=False):
            user_id: str | None
            budget: int
            answer: str


        def fetch(state: AppState) -> AppState:
            return {"user_id": "u_42", "budget": 100}


        def bad_transform(state: AppState) -> AppState:
            # bug: explicitly overwrites user_id with None and exceeds budget.
            # LangGraph merges this into the existing state, so both values
            # become visible downstream — not a silent omission.
            return {"user_id": None, "budget": state["budget"] - 500}


        def respond(state: AppState) -> AppState:
            return {"answer": f"hi {state.get('user_id')!r}, balance={state.get('budget')}"}


        baseline = StateGraph(AppState)
        baseline.add_node("fetch", fetch)
        baseline.add_node("transform", bad_transform)
        baseline.add_node("respond", respond)
        baseline.add_edge(START, "fetch")
        baseline.add_edge("fetch", "transform")
        baseline.add_edge("transform", "respond")
        baseline.add_edge("respond", END)
        baseline_app = baseline.compile()

        result = baseline_app.invoke({})
        print(f"broken answer: {result['answer']!r}")
        print(f"budget       : {result['budget']}  (should never go negative)")
    """),
    _md("""
        ## With `IntegrityGate` — invariants fire on the node's partial output

        Define two invariants and wrap the offending node. The gate runs
        the invariants against each wrapped node's **partial output**
        (the dict returned from the node, before LangGraph merges it into
        graph state). That scoping is deliberate: a partial update that
        emits an invalid value is easier to pin on a specific node than
        inspecting merged state after multiple reducers. When either
        invariant fails, the gate flips the per-thread flag and emits a
        `langgraph_state_integrity` certificate; the conditional edge
        routes to `recover` instead of `respond`.
    """),
    _code("""
        from operon_langgraph_gates import IntegrityGate


        def has_user_id(state: AppState) -> bool:
            return bool(state.get("user_id"))


        def budget_not_exceeded(state: AppState) -> bool:
            return state.get("budget", 0) >= 0


        def recover(state: AppState) -> AppState:
            return {"answer": "I can't answer right now — please retry."}


        gate = IntegrityGate(invariants=[has_user_id, budget_not_exceeded])

        gated = StateGraph(AppState)
        gated.add_node("fetch", fetch)
        gated.add_node("transform", gate.wrap(bad_transform))
        gated.add_node("respond", respond)
        gated.add_node("recover", recover)
        gated.add_edge(START, "fetch")
        gated.add_edge("fetch", "transform")
        gated.add_conditional_edges(
            "transform",
            gate.edge(forward="respond", break_to="recover"),
        )
        gated.add_edge("recover", END)
        gated.add_edge("respond", END)
        gated_app = gated.compile()

        result = gated_app.invoke({})
        print(f"gated answer: {result['answer']!r}")
    """),
    _md("""
        ## Inspect the certificate

        One `langgraph_state_integrity` cert was emitted on the first
        violation. `cert.verify()` replays the evidence: `holds` is
        `False` (integrity did not hold), and `evidence` lists which
        invariants failed.
    """),
    _code("""
        assert len(gate.certificates) == 1, "expected exactly one certificate"
        cert = gate.certificates[0]
        print(f"theorem    : {cert.theorem}")
        print(f"source     : {cert.source}")
        print(f"conclusion : {cert.conclusion}")

        result = cert.verify()
        print(f"verify.holds: {result.holds}")
        print(f"failed      : {result.evidence.get('failed_invariants')}")
        print(f"all results : {result.evidence.get('invariant_results')}")
    """),
    _md("""
        ## Takeaway

        Invariants live as ordinary callables in your code; evidence lives
        in a signed artifact the first time any one of them fails. Both the
        code invariants and the cert survive a restart / audit / replay —
        you can feed an old cert back through `cert.verify()` and
        re-confirm the finding without re-running the graph.

        Paired with `StagnationGate` (see `01_stagnation_breaks_loop.ipynb`),
        this gives a LangGraph `StateGraph` two reliability primitives:
        **loops stop**, **drift gets caught**, and both surface as
        replayable certificates.
    """),
]


def _write(path: Path, cells: list[nbformat.NotebookNode], id_prefix: str) -> None:
    _assign_stable_ids(cells, id_prefix)
    nb = new_notebook(cells=cells)
    # Deep copy so nested dicts in _CANONICAL_NB_METADATA are never shared
    # between source-only writes and executed-then-normalized writes.
    nb["metadata"] = deepcopy(_CANONICAL_NB_METADATA)
    # Sort keys so regenerated notebooks are diff-friendly.
    path.write_text(json.dumps(nb, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


# The canonical top-level notebook metadata. **Single source of truth**
# for both source-only (``_write``) and executed (``_normalize_notebook``)
# notebooks — they must stay identical so a regenerate-then-execute is
# byte-stable. Kept narrow so kernel-populated fields (e.g.
# ``language_info.version``, ``.codemirror_mode``) don't cause per-machine
# diffs. Callers deep-copy before assigning to a notebook so nested dicts
# aren't shared across instances.
_CANONICAL_NB_METADATA: dict[str, object] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}


def _normalize_notebook(data: dict[str, object]) -> None:
    """Scrub per-run noise from a notebook dict in place.

    Removes:
    - Per-cell ``metadata.execution`` (iopub/shell timestamps written by
      every execution).
    - Kernel-populated top-level ``metadata`` fields beyond the canonical
      whitelist (e.g. ``language_info.version``, ``.file_extension``).

    After this, a regenerate-and-execute produces a byte-for-byte no-op
    in git regardless of the executing Python/kernel version.
    """
    for cell in data.get("cells", []):  # type: ignore[union-attr]
        meta = cell.get("metadata", {})
        if "execution" in meta:
            del meta["execution"]
    # Deep copy so nested dicts in _CANONICAL_NB_METADATA are never shared
    # with this notebook instance — later mutations through ``data`` won't
    # leak back into the module-level constant.
    data["metadata"] = deepcopy(_CANONICAL_NB_METADATA)


def _strip_execution_timestamps(path: Path) -> None:
    """Public wrapper retained for compatibility with pre-existing docs.

    Delegates to :func:`_normalize_notebook`; the deep copy lives in the
    normalizer itself so all callers — not only this one — get the same
    isolation from the module-level canonical constant.
    """
    data = json.loads(path.read_text())
    _normalize_notebook(data)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def _execute(path: Path) -> None:
    """Run the notebook in place and scrub per-run timestamps after."""
    # Import lazily so the bare `python scripts/build_examples.py` path
    # doesn't require nbclient + ipykernel.
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=60, kernel_name="python3").execute()
    nbformat.write(nb, path)
    _strip_execution_timestamps(path)
    print(f"executed {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="also execute the notebooks in place and scrub timestamps",
    )
    args = parser.parse_args()

    EXAMPLES.mkdir(exist_ok=True)
    paths = [
        (EXAMPLES / "01_stagnation_breaks_loop.ipynb", NB1_CELLS, "stagnation"),
        (EXAMPLES / "02_integrity_catches_drift.ipynb", NB2_CELLS, "integrity"),
    ]
    for path, cells, prefix in paths:
        _write(path, cells, id_prefix=prefix)
    if args.execute:
        for path, _, _ in paths:
            _execute(path)


if __name__ == "__main__":
    main()
