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
