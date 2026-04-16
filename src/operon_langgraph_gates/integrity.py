"""IntegrityGate — runtime invariant checks on LangGraph node output.

Backed by Paper 4 §4.1 (state integrity, 3/3 on canonical benchmarks) and
Paper 5 §3 (certificate preservation under compilation). Conceptually a
thin LangGraph adapter over the same *state-integrity* idea Operon uses
for DNA-repair checkpoints — but the public API uses LangGraph terms
(``invariants``, ``thread_id``, ``schema``) rather than the biology
vocabulary of the upstream implementation.

Usage::

    gate = IntegrityGate(invariants=[has_required_user, budget_not_exceeded])
    graph.add_node("tool_call", gate.wrap(tool_call_fn))
    graph.add_conditional_edges(
        "tool_call",
        gate.edge(forward="process", break_to="recover"),
    )

A certificate with theorem ``langgraph_state_integrity`` is emitted once
per thread on the first violation. ``cert.verify()`` replays the frozen
invariant-result evidence and returns (False, details) on replay.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from operon_ai.core.certificate import Certificate, register_verify_fn

from ._common import EPHEMERAL_THREAD, thread_id

_NodeIn = Any
_NodeOut = Any
NodeFn = Callable[..., _NodeOut]
Invariant = Callable[[Mapping[str, Any]], bool]

_THEOREM = "langgraph_state_integrity"


def _verify_state_integrity(params: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Replay: all invariants recorded in ``invariant_results`` must hold."""
    results = tuple(params.get("invariant_results", ()))
    all_passed = all(passed for _, passed in results)
    return all_passed, {
        "invariant_results": tuple((name, bool(passed)) for name, passed in results),
        "failed_invariants": tuple(name for name, passed in results if not passed),
        "n": len(results),
    }


# Register our theorem so Certificate.verify() can find the function when
# replaying a cert that was serialized and reloaded (Paper 5 §3 — preservation
# under compilation relies on the verify-function registry).
register_verify_fn(_THEOREM, _verify_state_integrity)


@dataclass
class _ThreadState:
    """Per-thread integrity state."""

    is_violated: bool = False
    certificates: list[Certificate] = field(default_factory=list)


class IntegrityGate:
    """Checks state invariants at node output boundaries.

    Parameters
    ----------
    invariants:
        Iterable of callables ``(state) -> bool``. Each is run after every
        wrapped-node invocation; a ``False`` return counts as a violation.
        Callables are identified in certificates by their ``__name__``.
    """

    def __init__(self, invariants: list[Invariant] | tuple[Invariant, ...]) -> None:
        if not invariants:
            raise ValueError("IntegrityGate requires at least one invariant")
        self._invariants: tuple[Invariant, ...] = tuple(invariants)
        self._threads: dict[str, _ThreadState] = {}

    # -- public API ---------------------------------------------------------

    @property
    def is_violated(self) -> bool:
        """True if the ephemeral (default-thread) state has been violated."""
        state = self._threads.get(EPHEMERAL_THREAD)
        return state is not None and state.is_violated

    @property
    def certificates(self) -> list[Certificate]:
        """All integrity certificates emitted across all threads."""
        return [c for s in self._threads.values() for c in s.certificates]

    def reset(self, thread_id_: str | None = None) -> None:
        """Reset integrity state. ``None`` clears all threads."""
        if thread_id_ is None:
            self._threads.clear()
        else:
            self._threads.pop(thread_id_, None)

    def wrap(self, fn: NodeFn) -> NodeFn:
        """Wrap a LangGraph node function with invariant checking.

        Supports sync and async ``fn``; forwards ``*args, **kwargs`` so
        LangGraph's ``config`` / ``runtime`` are preserved and the gate can
        scope state per ``thread_id``.
        """
        if inspect.iscoroutinefunction(fn):

            async def async_wrapped(state: _NodeIn, *args: Any, **kwargs: Any) -> _NodeOut:
                output = await fn(state, *args, **kwargs)
                self._check(output, thread_id(args, kwargs))
                return output

            return async_wrapped

        def sync_wrapped(state: _NodeIn, *args: Any, **kwargs: Any) -> _NodeOut:
            output = fn(state, *args, **kwargs)
            self._check(output, thread_id(args, kwargs))
            return output

        return sync_wrapped

    def edge(self, forward: str, break_to: str) -> Callable[..., str]:
        """Conditional-edge router: returns ``break_to`` after a violation."""

        def route(_state: _NodeIn, *args: Any, **kwargs: Any) -> str:
            tid = thread_id(args, kwargs)
            state = self._threads.get(tid)
            return break_to if state is not None and state.is_violated else forward

        return route

    # -- internals ----------------------------------------------------------

    def _thread_state(self, tid: str) -> _ThreadState:
        state = self._threads.get(tid)
        if state is None:
            state = _ThreadState()
            self._threads[tid] = state
        return state

    def _check(self, output: _NodeOut, tid: str) -> None:
        state = self._thread_state(tid)
        # Don't re-emit after the first violation — the cert carries the full
        # evidence snapshot, subsequent calls would just duplicate it.
        if state.is_violated:
            return

        results: list[tuple[str, bool]] = []
        for inv in self._invariants:
            try:
                passed = bool(inv(output))
            except Exception:
                # A raising invariant is as much a violation as returning False.
                passed = False
            results.append((inv.__name__, passed))

        all_passed = all(p for _, p in results)
        if not all_passed:
            state.is_violated = True
            state.certificates.append(_make_certificate(results))


def _make_certificate(results: list[tuple[str, bool]]) -> Certificate:
    params = MappingProxyType(
        {
            "invariant_results": tuple(results),
            "all_passed": False,
        }
    )
    failed = [name for name, passed in results if not passed]
    return Certificate(
        theorem=_THEOREM,
        parameters=params,
        conclusion=(
            f"State-integrity violation: {len(failed)} of {len(results)} "
            f"invariants failed ({', '.join(failed)})."
        ),
        source="operon_langgraph_gates.integrity",
        _verify_fn=_verify_state_integrity,
    )
