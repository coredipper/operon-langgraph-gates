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
    # The cert uses a distinct theorem name — ``behavioral_stability_windowed``
    # — rather than the shared ``behavioral_stability``, so round-tripped
    # certs don't fall back to the core's flat-mean verifier.
    assert certs[0].theorem == "behavioral_stability_windowed"


def test_certificates_empty_on_healthy_run() -> None:
    gate = _make_gate()
    responses = iter([{"answer": r} for r in _DIVERSE_RESPONSES])
    wrapped = gate.wrap(lambda state: next(responses))
    for _ in range(6):
        wrapped({"input": "q"})
    assert gate.certificates == []


async def test_wrap_handles_async_fn() -> None:
    gate = _make_gate()

    async def afn(state: dict[str, object]) -> dict[str, str]:
        return {"answer": "same response every turn"}

    wrapped = gate.wrap(afn)
    for _ in range(6):
        out = await wrapped({"input": "q"})
        assert out == {"answer": "same response every turn"}
    assert gate.is_stagnant is True


def test_sync_wrap_raises_if_fn_returns_awaitable() -> None:
    """Defensive: fn advertised as sync but returns an awaitable is a bug; raise
    loud rather than measure ``str(coroutine)``, and don't leak a
    ``coroutine was never awaited`` RuntimeWarning."""
    import warnings

    import pytest

    gate = _make_gate()

    async def inner(state: dict[str, object]) -> dict[str, str]:
        return {"answer": "x"}

    # Declare the outer sync-looking but actually returns a coroutine.
    def sync_returning_coro(state: dict[str, object]) -> object:
        return inner(state)

    wrapped = gate.wrap(sync_returning_coro)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="awaitable"):
            wrapped({"input": "q"})

    # The coroutine must be closed before TypeError is raised, so no
    # "coroutine was never awaited" RuntimeWarning should escape.
    leaks = [w for w in captured if "coroutine" in str(w.message).lower()]
    assert leaks == [], f"Leaked coroutine warnings: {[str(w.message) for w in leaks]}"


def test_wrap_forwards_extra_args_to_fn() -> None:
    gate = _make_gate()
    received: list[object] = []

    def fn(state: dict[str, object], config: object) -> dict[str, str]:
        received.append(config)
        return {"answer": "x"}

    wrapped = gate.wrap(fn)
    wrapped({"q": "q"}, {"thread_id": "A"})
    assert received == [{"thread_id": "A"}]


def test_thread_id_extracted_from_runtime_like_object() -> None:
    """LangGraph 1.x passes a Runtime object (attribute access for .config),
    not the raw dict config. The gate must unwrap it to extract thread_id."""
    gate = _make_gate()

    class FakeRuntime:
        def __init__(self, thread_id: str) -> None:
            self.config = {"configurable": {"thread_id": thread_id}}

    wrapped = gate.wrap(lambda state, rt: {"answer": "same response every turn"})
    edge = gate.edge(forward="answer", break_to="escalate")

    rt_a = FakeRuntime("A")
    for _ in range(6):
        wrapped({"q": "q"}, rt_a)

    # Thread A must be the one that accumulated the stagnation.
    assert edge({}, rt_a) == "escalate"
    assert edge({}, FakeRuntime("B")) == "answer"


def test_thread_id_ignores_business_objects_with_config_attribute() -> None:
    """A non-LangGraph object with a ``.config`` attribute shaped like a
    RunnableConfig must NOT hijack thread routing. Only explicit
    ``config`` / ``runtime`` kwarg names (or positional args) are
    scanned for thread id.
    """
    gate = _make_gate()

    class SettingsContainer:
        # A realistic-ish business dependency with a ``config`` attribute
        # that happens to match the RunnableConfig shape.
        def __init__(self) -> None:
            self.config = {"configurable": {"thread_id": "hijacked-by-object"}}

    wrapped = gate.wrap(
        lambda state, **kw: {"answer": "same response every turn"},
    )

    for _ in range(6):
        wrapped({"q": "q"}, settings=SettingsContainer())

    assert gate.is_stagnant_for("hijacked-by-object") is False
    assert gate.is_stagnant is True


def test_thread_id_ignores_business_kwargs_shaped_like_config() -> None:
    """A plain-dict business kwarg that happens to contain a
    ``configurable.thread_id`` path must NOT be interpreted as LangGraph
    config. Only known LangGraph kwarg names (``config``, ``runtime``) or
    Runtime-like objects with ``.config`` should be scanned for thread id.
    """
    gate = _make_gate()
    wrapped = gate.wrap(
        lambda state, **kw: {"answer": "same response every turn"},
    )

    # Bogus business payload with the same shape as a RunnableConfig.
    business = {"configurable": {"thread_id": "hijacked"}}
    for _ in range(6):
        wrapped({"q": "q"}, business_payload=business)

    # The "hijacked" bucket must not have been populated — measurements
    # should have landed in the ephemeral bucket.
    assert gate.is_stagnant_for("hijacked") is False
    assert gate.is_stagnant is True


def test_thread_id_extracted_from_runtime_keyword_arg() -> None:
    """LangGraph can pass ``runtime`` as a keyword-only argument; the gate
    must find the thread id regardless of which kwarg name carries it."""
    gate = _make_gate()

    class FakeRuntime:
        def __init__(self, thread_id: str) -> None:
            self.config = {"configurable": {"thread_id": thread_id}}

    wrapped = gate.wrap(lambda state, *, runtime: {"answer": "same response every turn"})
    edge = gate.edge(forward="answer", break_to="escalate")

    for _ in range(6):
        wrapped({"q": "q"}, runtime=FakeRuntime("A"))

    assert edge({}, runtime=FakeRuntime("A")) == "escalate"
    assert edge({}, runtime=FakeRuntime("B")) == "answer"


async def test_wrap_handles_callable_class_with_async_call() -> None:
    """Async detection must catch classes with ``async def __call__``."""
    gate = _make_gate()

    class AsyncNode:
        async def __call__(self, state: dict[str, object]) -> dict[str, str]:
            return {"answer": "same response every turn"}

    wrapped = gate.wrap(AsyncNode())
    for _ in range(6):
        out = await wrapped({"input": "q"})
        assert out == {"answer": "same response every turn"}
    assert gate.is_stagnant is True


def test_is_stagnant_for_reads_specific_thread() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state, config=None: {"answer": "same response every turn"})
    cfg_a = {"configurable": {"thread_id": "A"}}
    for _ in range(6):
        wrapped({"q": "q"}, cfg_a)
    assert gate.is_stagnant_for("A") is True
    assert gate.is_stagnant_for("B") is False


def test_edge_routes_per_thread() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state, config=None: {"answer": "same response every turn"})
    cfg_a = {"configurable": {"thread_id": "A"}}
    cfg_b = {"configurable": {"thread_id": "B"}}
    edge = gate.edge(forward="answer", break_to="escalate")

    for _ in range(6):
        wrapped({"q": "q"}, cfg_a)

    # Thread A is stagnant; thread B has never run and must stay healthy.
    assert edge({}, cfg_a) == "escalate"
    assert edge({}, cfg_b) == "answer"


def test_integrals_for_records_the_value_the_gate_routed_on() -> None:
    """``integrals_for`` must surface the exact per-turn integral the
    detection logic compared against ``threshold`` — not a re-derivation
    from severities that could drift once α-mixing is in play."""
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "same response every turn"})
    for _ in range(6):
        wrapped({"q": "q"})
    integrals = gate.integrals_for()
    assert len(integrals) == 6, f"expected one integral per turn; got {integrals}"
    # All values must lie in the valid epiplexity range [0, 1].
    assert all(0.0 <= v <= 1.0 for v in integrals)
    # The gate flipped to is_stagnant within these 6 turns, so at least
    # critical_duration of the trailing integrals must be < threshold.
    assert sum(1 for v in integrals[-3:] if v < 0.2) >= 2


def test_integrals_for_is_empty_before_any_observation() -> None:
    gate = _make_gate()
    assert gate.integrals_for() == []
    assert gate.integrals_for("unknown-thread") == []


def test_reset_clears_stagnation() -> None:
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "same response every turn"})
    for _ in range(6):
        wrapped({"q": "q"})
    assert gate.is_stagnant is True
    gate.reset()
    assert gate.is_stagnant is False
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


# ---------------------------------------------------------------------------
# Cert-semantics sibling-sync tests — mirror the regression suite that
# operon-openhands-gates converged on through roborev reviews 760–767.
# Each test anchors one historical finding so the same class of bug
# cannot be reintroduced on either sibling.
# ---------------------------------------------------------------------------


def test_certificate_evidence_is_per_window_severity_means() -> None:
    """Cert ``signal_values`` is one mean per violating rolling window —
    ``critical_duration`` values — not a flattened severity history."""
    gate = _make_gate()
    wrapped = gate.wrap(lambda state: {"answer": "same response every turn"})
    for _ in range(6):
        wrapped({"input": "q"})
    assert gate.certificates, "expected a certificate on sustained stagnation"
    cert = gate.certificates[0]
    signal_values = cert.parameters["signal_values"]
    assert len(signal_values) == gate._critical_duration


def test_certificate_verify_uses_max_not_flat_mean_under_overlapping_windows() -> None:
    """Reviewer 764 counterexample, ported verbatim.

    With ``window=2, critical_duration=2`` and severities ``[0.61, 1.0, 0.61]``,
    both rolling-window means are ``0.805`` — detection fires against
    stability threshold ``0.8``. The flattened mean over the union is only
    ``0.74``, which would incorrectly say stability held under the
    shared ``_verify_behavioral_stability``. The ported
    ``_verify_window_max_stability`` must return ``holds=False`` because
    ``max(window_means) > threshold``.
    """
    from operon_langgraph_gates.stagnation import _emit_certificate

    cert = _emit_certificate(
        window_severity_means=(0.805, 0.805),
        threshold=0.8,
        detection_index=3,
        source="test",
    )
    verification = cert.verify()
    assert verification.holds is False
    assert verification.evidence["max"] == 0.805
    assert verification.evidence["n"] == 2


def test_certificate_verify_treats_threshold_equality_as_stable() -> None:
    """Boundary: detection uses strict ``integral < τ`` so the severity-domain
    complement is the inclusive ``mean(severity) <= 1 - τ``."""
    from operon_langgraph_gates.stagnation import _emit_certificate

    at_boundary = _emit_certificate(
        window_severity_means=(0.8,),
        threshold=0.8,
        detection_index=1,
        source="test",
    )
    assert at_boundary.verify().holds is True

    just_below = _emit_certificate(
        window_severity_means=(0.799,),
        threshold=0.8,
        detection_index=1,
        source="test",
    )
    assert just_below.verify().holds is True

    just_above = _emit_certificate(
        window_severity_means=(0.801,),
        threshold=0.8,
        detection_index=1,
        source="test",
    )
    assert just_above.verify().holds is False


def test_certificate_empty_evidence_is_rejected_not_vacuously_stable() -> None:
    """Empty evidence is malformed, not vacuous. Two-layer defense:
    emit raises, verify rejects."""
    import pytest

    from operon_langgraph_gates.stagnation import (
        _emit_certificate,
        _verify_window_max_stability,
    )

    with pytest.raises(ValueError, match="non-empty"):
        _emit_certificate(
            window_severity_means=(),
            threshold=0.8,
            detection_index=42,
            source="test",
        )

    holds, evidence = _verify_window_max_stability(
        {"signal_values": (), "threshold": 0.8}
    )
    assert holds is False
    assert evidence["reason"] == "empty_evidence"


def test_certificate_threshold_is_severity_domain_complement() -> None:
    """Stored cert threshold is the stability threshold ``1 - τ_detect``,
    so verify agrees with detection at every τ in [0,1] — not just τ <= 0.5."""
    gate = _make_gate()  # threshold=0.2 by default in this helper
    wrapped = gate.wrap(lambda state: {"answer": "stuck stuck stuck"})
    for _ in range(6):
        wrapped({"input": "q"})
    assert gate.certificates, "expected a certificate on sustained stagnation"
    cert = gate.certificates[0]
    assert cert.parameters["threshold"] == 1.0 - gate._threshold


def test_certificate_conclusion_reports_total_detection_index() -> None:
    """Regression for roborev job 771 Low.

    The conclusion text must quote the total evaluation count at
    detection, not the evidence-slice length. Parse the number out of
    the "after N measurements" fragment and assert it equals the
    integrals-list length at emission (which is what was passed as
    ``detection_index``).
    """
    import re

    gate = StagnationGate(threshold=0.2, critical_duration=1, window_size=20)
    wrapped = gate.wrap(lambda state: {"answer": "identical saturating text"})
    for _ in range(25):
        wrapped({"input": "q"})
    assert gate.certificates

    conclusion = gate.certificates[0].conclusion
    # Conclusion must contain the exact "after N measurements" fragment.
    match = re.search(r"after (\d+) measurements", conclusion)
    assert match is not None, f"conclusion lacks measurement count: {conclusion!r}"
    reported_n = int(match.group(1))

    # Reported N must match the evaluation count at which the cert fired.
    # With window=20, cd=1, detection fires at the first integral that
    # crosses threshold, which happens once enough context exists — so
    # N is strictly greater than critical_duration and at least window-sized.
    assert reported_n > gate._critical_duration
    assert reported_n >= gate._window_size


def test_windowed_theorem_is_registered_for_round_trip_verify() -> None:
    """Regression for roborev job 771 High.

    The custom verifier must be registered against a unique theorem name,
    so that a deserialized certificate resolves back to our verifier
    rather than silently falling back to the core's flat-mean
    ``_verify_behavioral_stability`` (which is what's registered for the
    shared ``behavioral_stability`` theorem name).
    """
    from operon_ai.core.certificate import _resolve_verify_fn

    from operon_langgraph_gates.stagnation import _verify_window_max_stability

    resolved = _resolve_verify_fn("behavioral_stability_windowed")
    assert resolved is _verify_window_max_stability

    # And the shared name still resolves to the core's (correctly-named but
    # flat-mean) verifier — we didn't clobber it.
    from operon_ai.core.certificate import _verify_behavioral_stability

    assert _resolve_verify_fn("behavioral_stability") is _verify_behavioral_stability
