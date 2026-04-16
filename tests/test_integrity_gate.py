"""Tests for IntegrityGate.

Verifies state invariants at node output boundaries. Emits a
``langgraph_state_integrity`` certificate on first violation, per thread.

Backed by Paper 4 §4.1 (state-integrity certification, 3/3 on canonical
benchmarks) and Paper 5 §3 (certificate preservation under compilation).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from operon_langgraph_gates import IntegrityGate


def _has_required_answer(state: Mapping[str, Any]) -> bool:
    return "answer" in state and isinstance(state["answer"], str) and state["answer"] != ""


def _positive_turn(state: Mapping[str, Any]) -> bool:
    return state.get("turn", 0) >= 0


def _make_gate() -> IntegrityGate:
    return IntegrityGate(invariants=[_has_required_answer, _positive_turn])


def test_wrap_returns_callable() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: state)
    assert callable(wrapped)


def test_wrap_passes_output_through_unchanged() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "ok", "turn": 1})
    assert wrapped({}) == {"answer": "ok", "turn": 1}


def test_no_violation_when_invariants_pass() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "ok", "turn": 1})
    wrapped({})
    assert gate.is_violated is False
    assert gate.certificates == []


def test_violation_detected_when_invariant_fails() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "", "turn": 1})  # empty answer
    wrapped({})
    assert gate.is_violated is True


def test_edge_routes_forward_when_clean() -> None:
    gate = _make_gate()
    edge = gate.edge(forward="process", break_to="recover")
    assert edge({}) == "process"


def test_edge_routes_break_to_when_violated() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "", "turn": 1})
    wrapped({})
    edge = gate.edge(forward="process", break_to="recover")
    assert edge({}) == "recover"


def test_certificate_emitted_on_violation() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "", "turn": 1})
    wrapped({})
    certs = gate.certificates
    assert len(certs) == 1
    assert certs[0].theorem == "langgraph_state_integrity"


def test_certificate_verify_replays_correctly() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "", "turn": 1})
    wrapped({})
    cert = gate.certificates[0]
    verification = cert.verify()
    assert verification.holds is False  # violation => verification fails
    # Evidence enumerates which named invariants failed.
    assert "invariant_results" in verification.evidence or verification.evidence


def test_certificate_only_emitted_once_per_thread() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "", "turn": 1})
    for _ in range(5):
        wrapped({})
    assert len(gate.certificates) == 1


def test_reset_clears_violation() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "", "turn": 1})
    wrapped({})
    assert gate.is_violated is True
    gate.reset()
    assert gate.is_violated is False
    assert gate.certificates == []


async def test_wrap_handles_async_fn() -> None:
    gate = _make_gate()

    async def afn(state: dict[str, Any]) -> dict[str, Any]:
        return {"answer": "", "turn": 1}

    wrapped = gate.wrap(afn)
    out = await wrapped({})
    assert out == {"answer": "", "turn": 1}
    assert gate.is_violated is True


def test_edge_routes_per_thread() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state, config=None: {"answer": "", "turn": 1})
    cfg_a = {"configurable": {"thread_id": "A"}}
    cfg_b = {"configurable": {"thread_id": "B"}}
    edge = gate.edge(forward="process", break_to="recover")

    wrapped({}, cfg_a)
    assert edge({}, cfg_a) == "recover"
    assert edge({}, cfg_b) == "process"
