"""End-to-end: StagnationGate breaks a real LangGraph infinite loop.

The scenario mirrors the pathology described in ``langchain-ai/langgraph``
issue #6731 (agent infinite-loops until recursion limit, burning tokens
invisibly). A node that always emits the same output is wired with a
self-loop conditional edge. Without the gate the graph recurses until
``GraphRecursionError``; with the gate the conditional edge routes out
to an ``escalate`` node once stagnation is detected.
"""

from __future__ import annotations

import pytest
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from operon_langgraph_gates import StagnationGate


class LoopState(TypedDict):
    turn: int
    answer: str


_STAGNANT_ANSWER = "same output every turn"


def _think(state: LoopState) -> LoopState:
    return {"turn": state["turn"] + 1, "answer": _STAGNANT_ANSWER}


def _escalate(state: LoopState) -> LoopState:
    return {"turn": state["turn"], "answer": "escalated"}


def _build_gated_graph(gate: StagnationGate) -> object:
    graph = StateGraph(LoopState)
    # Measure only the `answer` field — the turn counter increments each call
    # and would otherwise contaminate the stagnation signal with string drift
    # unrelated to the repeated model output.
    graph.add_node(
        "think",
        gate.wrap(_think, text_extractor=lambda out: out["answer"]),
    )
    graph.add_node("escalate", _escalate)
    graph.add_edge(START, "think")
    graph.add_conditional_edges(
        "think",
        gate.edge(forward="think", break_to="escalate"),
    )
    graph.add_edge("escalate", END)
    return graph.compile()


def test_stagnation_gate_breaks_infinite_loop() -> None:
    """Gate detects stagnation and routes out of the self-loop promptly."""
    # window_size=3, critical_duration=2 => detection requires ~5 warmup turns
    # for the integral to slide below threshold for 2 consecutive readings.
    # Upper bound proves the gate fires *promptly*, not just eventually.
    gate = StagnationGate(threshold=0.2, critical_duration=2, window_size=3)
    app = _build_gated_graph(gate)

    # Without the gate, this self-loop would recurse to the default limit.
    result = app.invoke({"turn": 0, "answer": ""}, {"recursion_limit": 25})

    # Graph terminated via the escalate branch.
    assert result["answer"] == "escalated"
    # Detection must happen in a narrow window: enough warmup for evidence,
    # but well before the recursion limit (or the baseline would dominate).
    assert 3 <= result["turn"] <= 10, (
        f"Expected gate to fire between turns 3 and 10; saw turn={result['turn']}"
    )
    # A behavioral_stability certificate was emitted at the moment of detection.
    certs = gate.certificates
    assert len(certs) >= 1
    assert certs[0].theorem == "behavioral_stability"


def test_without_gate_loop_hits_recursion_limit() -> None:
    """Baseline: the same self-loop without the gate hits RecursionError.

    Proves the gate is load-bearing — not an accidental side effect of
    some other graph property.
    """

    def _always_loop(_state: LoopState) -> str:
        return "think"

    graph = StateGraph(LoopState)
    graph.add_node("think", _think)
    graph.add_edge(START, "think")
    graph.add_conditional_edges("think", _always_loop)
    app = graph.compile()

    with pytest.raises(GraphRecursionError):
        app.invoke({"turn": 0, "answer": ""}, {"recursion_limit": 5})
