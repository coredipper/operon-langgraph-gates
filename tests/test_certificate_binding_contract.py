"""Parameter-shape contract tests, propagated from
``specs/certificate-binding.allium``.

These fill the gap left by the existing behavioural tests. The hand-written
suites (``test_a2a_round_trip.py``, ``test_stagnation_gate.py``,
``test_integrity_gate.py``) thoroughly cover the *behavioural* obligations —
replay equivalence and A2A round-trip — so those are deliberately NOT
duplicated here (per the propagate skill: do not regenerate working tests).

What they do NOT assert is the *exact emitted parameter-dict shape*. A
regression that adds, renames, or drops a parameter key — or flips
``all_passed`` — would pass every existing test (verify() still replays,
round-trip still equal) yet silently break the cross-package consumption
contract that operon-ai's sibling adapters depend on. These tests close that
gap, mapping 1:1 to the structural obligations from ``allium plan``.

No PBT framework is present in this repo (pyproject declares only pytest),
so these are assertion-based per the propagate fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from operon_langgraph_gates import (
    INTEGRITY_THEOREM,
    STAGNATION_THEOREM,
    IntegrityGate,
    StagnationGate,
)

_STAGNATION_DETECTION_THRESHOLD = 0.2


def _has_answer(state: Mapping[str, Any]) -> bool:
    return bool(state.get("answer"))


def _drive_integrity_gate_to_violation() -> IntegrityGate:
    gate = IntegrityGate(invariants=[_has_answer])
    wrapped = gate.wrap(lambda state: {"answer": "", "turn": 1})
    wrapped({})
    assert gate.certificates, "fixture failed to drive IntegrityGate to emission"
    return gate


def _drive_stagnation_gate_to_emission() -> StagnationGate:
    gate = StagnationGate(
        threshold=_STAGNATION_DETECTION_THRESHOLD,
        critical_duration=1,
        window_size=20,
    )
    wrapped = gate.wrap(lambda state: {"answer": "identical saturating text"})
    for _ in range(40):
        wrapped({"input": "q"})
    assert gate.certificates, "fixture failed to drive StagnationGate to emission"
    return gate


# spec: Certificate.theorem discriminator (obligation 0) — the sum-type
# discriminator must equal exactly the two public wire constants.
def test_certificate_theorem_discriminator_strings() -> None:
    stag = _drive_stagnation_gate_to_emission().certificates[0]
    integ = _drive_integrity_gate_to_violation().certificates[0]
    assert stag.theorem == STAGNATION_THEOREM == "behavioral_stability_windowed"
    assert integ.theorem == INTEGRITY_THEOREM == "langgraph_state_integrity"


# spec: variant Stagnation fields + StagnationCertificateEmitted ensures
# (obligations 1, 8) — parameters are EXACTLY {signal_values, threshold}.
def test_stagnation_certificate_parameter_shape_is_exact() -> None:
    cert = _drive_stagnation_gate_to_emission().certificates[0]
    assert set(cert.parameters.keys()) == {"signal_values", "threshold"}
    signal_values = cert.parameters["signal_values"]
    assert isinstance(signal_values, tuple) and len(signal_values) >= 1
    assert all(isinstance(v, (int, float)) for v in signal_values)
    assert isinstance(cert.parameters["threshold"], (int, float))


# spec: StagnationCertificateEmitted ensures threshold = 1 - detection_threshold
# (obligation 8) — the severity-domain translation binding.
def test_stagnation_threshold_is_translated_one_minus_detection_threshold() -> None:
    cert = _drive_stagnation_gate_to_emission().certificates[0]
    assert cert.parameters["threshold"] == 1.0 - _STAGNATION_DETECTION_THRESHOLD


# spec: variant Integrity fields + IntegrityCertificateEmitted ensures
# (obligations 2, 11) — parameters are EXACTLY {invariant_results, all_passed}.
def test_integrity_certificate_parameter_shape_is_exact() -> None:
    cert = _drive_integrity_gate_to_violation().certificates[0]
    assert set(cert.parameters.keys()) == {"invariant_results", "all_passed"}


# spec: IntegrityCertificateEmitted ensures all_passed: false (obligation 11)
# — a certificate is emitted ONLY on violation, so this is invariantly False.
def test_integrity_all_passed_is_exactly_false_on_emission() -> None:
    cert = _drive_integrity_gate_to_violation().certificates[0]
    assert cert.parameters["all_passed"] is False


# spec: value InvariantResult fields + structural equality (obligations 3, 4)
# — each result is a (name: str, passed: bool) pair; equal pairs compare equal.
def test_integrity_invariant_results_are_name_passed_pairs() -> None:
    cert = _drive_integrity_gate_to_violation().certificates[0]
    results = cert.parameters["invariant_results"]
    # Exact emitted shape: a tuple whose entries are each strictly a
    # 2-tuple (str, bool) — not merely "unpackable into two", which a
    # list or custom object would also satisfy and which would let a
    # regression away from tuple[(str, bool), ...] pass silently.
    assert isinstance(results, tuple) and len(results) >= 1
    for entry in results:
        assert type(entry) is tuple and len(entry) == 2
        name, passed = entry
        assert type(name) is str
        assert type(passed) is bool
    # Structural-equality / exact-value check against an independently
    # constructed expected value (not a re-projection of the emitted
    # entry). The fixture is deterministic: the single invariant
    # `_has_answer` returns False for an empty answer, so the emitted
    # vector is exactly this tuple-of-tuples.
    assert results == (("_has_answer", False),)
