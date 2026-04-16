"""Tests for StagnationGate.

Behavior backed by Paper 4 §4.3 (Bayesian stagnation detection, 96%
accuracy with real embeddings); tests here use the zero-dep NGramEmbedder
which trades accuracy for install-free determinism.
"""

from __future__ import annotations

from operon_langgraph_gates import StagnationGate


def _make_gate() -> StagnationGate:
    # Short window + low critical_duration so tests stay fast and deterministic
    # against the n-gram embedder.
    return StagnationGate(threshold=0.2, critical_duration=2, window_size=3)


# Real English sentences — sufficient trigram divergence to keep
# epiplexic_integral above the threshold across calls.
_DIVERSE_RESPONSES = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "sphinx of black quartz judge my vow",
    "two driven jocks help fax my big quiz",
    "the five boxing wizards jump quickly",
    "waltz bad nymph for quick jigs vex",
    "jackdaws love my big sphinx of quartz",
    "bright vixens jump dozy fowl quack",
    "crazy Fredrick bought many very exquisite opal jewels",
]


def test_wrap_returns_callable() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: state)
    assert callable(wrapped)


def test_wrap_passes_output_through_unchanged() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"x": state["x"] + 1})
    assert wrapped({"x": 1}) == {"x": 2}


def test_no_stagnation_detected_on_first_call() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "first response"})
    wrapped({"input": "q"})
    assert gate.is_stagnant is False


def test_stagnation_detected_on_identical_outputs() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "same response every turn"})
    for _ in range(6):
        wrapped({"input": "q"})
    assert gate.is_stagnant is True


def test_no_stagnation_on_diverse_outputs() -> None:
    gate = _make_gate()
    responses = iter([{"answer": r} for r in _DIVERSE_RESPONSES])
    wrapped = gate.wrap(lambda state: next(responses))
    for _ in range(6):
        wrapped({"input": "q"})
    assert gate.is_stagnant is False


def test_edge_routes_forward_when_healthy() -> None:
    gate = _make_gate()
    edge = gate.edge(forward="answer", break_to="escalate")
    assert edge({"x": 1}) == "answer"


def test_edge_routes_break_to_when_stagnant() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "same response every turn"})
    for _ in range(6):
        wrapped({"input": "q"})
    edge = gate.edge(forward="answer", break_to="escalate")
    assert edge({}) == "escalate"


def test_certificate_emitted_on_stagnation() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "same response every turn"})
    for _ in range(6):
        wrapped({"input": "q"})
    certs = gate.certificates
    assert len(certs) >= 1
    # The cert carries the theorem name used by operon-ai's behavioral cert.
    assert certs[0].theorem == "behavioral_stability"


def test_certificates_empty_on_healthy_run() -> None:
    gate = _make_gate()
    responses = iter([{"answer": r} for r in _DIVERSE_RESPONSES])
    wrapped = gate.wrap(lambda state: next(responses))
    for _ in range(6):
        wrapped({"input": "q"})
    assert gate.certificates == []


def test_text_extractor_used_when_provided() -> None:
    gate = _make_gate()
    responses = iter(_DIVERSE_RESPONSES)
    wrapped = gate.wrap(
        lambda state: {"answer": next(responses), "noise": state["noise"]},
        text_extractor=lambda out: out["answer"],
    )
    # Noise varies per state but the extractor picks "answer" for measurement.
    for i in range(6):
        wrapped({"noise": i})
    assert gate.is_stagnant is False
