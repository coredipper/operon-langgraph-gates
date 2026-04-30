"""A2A round-trip test for gate-emitted certificates.

Pins the cross-repo binding between ``operon-langgraph-gates`` and
``operon-ai``'s A2A codec at ``operon_ai.convergence.a2a_certificate``.
A certificate emitted by either gate must encode via
``certificate_to_a2a_part`` and decode via ``certificate_from_a2a_part``
such that ``Certificate.verify()`` returns the same result and the
parameter dict is preserved.

This is the test that promotes the README's former "Ecosystem note (out
of this repository)" claim from informational to enforced under the
package's pinned ``operon-ai`` range in ``pyproject.toml``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from operon_ai.convergence.a2a_certificate import (
    certificate_from_a2a_part,
    certificate_to_a2a_part,
    safe_certificate_from_a2a_part,
)

from operon_langgraph_gates import (
    INTEGRITY_THEOREM,
    STAGNATION_THEOREM,
    IntegrityGate,
    StagnationGate,
)


def _has_answer(state: Mapping[str, Any]) -> bool:
    return bool(state.get("answer"))


def _drive_integrity_gate_to_violation() -> IntegrityGate:
    """Wrap a node that returns an empty answer; the invariant fails on first call."""
    gate = IntegrityGate(invariants=[_has_answer])
    wrapped = gate.wrap(lambda state: {"answer": "", "turn": 1})
    wrapped({})
    assert gate.certificates, "fixture failed to drive IntegrityGate to emission"
    return gate


def _drive_stagnation_gate_to_emission() -> StagnationGate:
    """Wrap a node returning identical text; drive enough turns for the windowed
    detector to fire. Mirrors the fixture in ``test_stagnation_gate.py`` —
    same config (``threshold=0.2, critical_duration=1, window_size=20``) and
    same 40-turn loop, since the windowed verifier needs sufficient history
    before the rolling integrals drop below threshold."""
    gate = StagnationGate(threshold=0.2, critical_duration=1, window_size=20)
    wrapped = gate.wrap(lambda state: {"answer": "identical saturating text"})
    for _ in range(40):
        wrapped({"input": "q"})
    assert gate.certificates, "fixture failed to drive StagnationGate to emission"
    return gate


def test_integrity_certificate_round_trips_through_a2a() -> None:
    gate = _drive_integrity_gate_to_violation()
    original = gate.certificates[0]
    pre = original.verify()

    part = certificate_to_a2a_part(original)
    decoded = certificate_from_a2a_part(part)
    post = decoded.verify()

    assert decoded.theorem == original.theorem == INTEGRITY_THEOREM
    assert post.holds == pre.holds
    # Parameter dict is the entire replay payload — must be identical.
    assert decoded.parameters == original.parameters


def test_stagnation_certificate_round_trips_through_a2a() -> None:
    gate = _drive_stagnation_gate_to_emission()
    original = gate.certificates[0]
    pre = original.verify()

    part = certificate_to_a2a_part(original)
    decoded = certificate_from_a2a_part(part)
    post = decoded.verify()

    assert decoded.theorem == original.theorem == STAGNATION_THEOREM
    assert post.holds == pre.holds
    assert decoded.parameters == original.parameters


def test_safe_decode_returns_certificate_for_known_theorems() -> None:
    """``safe_certificate_from_a2a_part`` must return the cert (not None) for
    both gate-emitted theorem names — the round-trip works even when the
    receiver opts into graceful-degradation mode."""
    integ_gate = _drive_integrity_gate_to_violation()
    stag_gate = _drive_stagnation_gate_to_emission()

    integ_part = certificate_to_a2a_part(integ_gate.certificates[0])
    stag_part = certificate_to_a2a_part(stag_gate.certificates[0])

    assert safe_certificate_from_a2a_part(integ_part) is not None
    assert safe_certificate_from_a2a_part(stag_part) is not None
