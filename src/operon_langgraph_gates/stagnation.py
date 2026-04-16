"""StagnationGate — Bayesian stagnation detection for LangGraph nodes and edges.

Backed by Paper 4 §4.3 (Bayesian stagnation detection, 96% accuracy with real
embeddings; Operon Papers). This module is a thin LangGraph-friendly wrapper
over ``operon_ai.health.epiplexity.EpiplexityMonitor`` — it adds:

- A text-extraction step (LangGraph node outputs are typically dicts, not text).
- A zero-dep default embedder (``NGramEmbedder``) so the gate works without
  a neural model install. Pass ``embedder=...`` for a real one.
- Per-thread state scoping: each LangGraph ``config["configurable"]["thread_id"]``
  gets its own monitor and certificates, so reusing a compiled graph across
  invocations or running concurrent threads doesn't leak stagnation state.
- Async node support via ``inspect.iscoroutinefunction`` detection.
- ``behavioral_stability`` certificate emission on first detection per thread.

Usage::

    gate = StagnationGate(threshold=0.2, critical_duration=3)
    graph.add_node("think", gate.wrap(think_fn))
    graph.add_conditional_edges("think", gate.edge(forward="answer", break_to="escalate"))
    # after graph.invoke(...)
    if gate.certificates:
        # stagnation was detected during the run
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from operon_ai.core.certificate import Certificate, _verify_behavioral_stability
from operon_ai.health.epiplexity import EmbeddingProvider, EpiplexityMonitor

from ._common import EPHEMERAL_THREAD, thread_id
from .embedders import NGramEmbedder

# LangGraph node functions consume and return arbitrary mappings (TypedDict,
# dict, or dataclass). We keep the typing loose on purpose — the gate does not
# care about the schema, only the extracted text.
_NodeIn = Any
_NodeOut = Any
NodeFn = Callable[..., _NodeOut]
TextExtractor = Callable[[_NodeOut], str]


@dataclass
class _ThreadState:
    """Per-thread state holder — one per LangGraph ``thread_id``."""

    monitor: EpiplexityMonitor
    severities: list[float] = field(default_factory=list)
    certificates: list[Certificate] = field(default_factory=list)
    is_stagnant: bool = False
    low_integral_streak: int = 0


class StagnationGate:
    """Detects stagnation on a per-node basis and routes conditional edges.

    Parameters mirror :class:`operon_ai.health.epiplexity.EpiplexityMonitor`
    but with LangGraph-friendly defaults:

    - ``threshold``: epiplexic integral below this counts toward stagnation.
    - ``window_size``: sliding-window size for the epiplexity calculation.
    - ``critical_duration``: consecutive sub-threshold readings required.
    - ``embedder``: any object with ``embed(text) -> list[float]``. Defaults to
      :class:`NGramEmbedder` — zero-dependency, deterministic, good enough to
      catch verbatim-repeat pathologies.
    """

    def __init__(
        self,
        threshold: float = 0.2,
        critical_duration: int = 2,
        window_size: int = 10,
        *,
        alpha: float = 0.5,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._threshold = threshold
        self._critical_duration = critical_duration
        self._window_size = window_size
        self._alpha = alpha
        self._embedder: EmbeddingProvider = embedder or NGramEmbedder()
        self._threads: dict[str, _ThreadState] = {}

    # -- public API ---------------------------------------------------------

    @property
    def is_stagnant(self) -> bool:
        """Whether the ephemeral (default-thread) measurement is stagnant.

        For per-thread checks inside a LangGraph run, use :meth:`edge` — it
        reads the correct thread from the passed-in ``config`` automatically.
        """
        state = self._threads.get(EPHEMERAL_THREAD)
        return state is not None and state.is_stagnant

    @property
    def certificates(self) -> list[Certificate]:
        """All certificates emitted across all threads during this gate's life."""
        return [c for s in self._threads.values() for c in s.certificates]

    def reset(self, thread_id: str | None = None) -> None:
        """Reset stagnation state. ``None`` clears all threads."""
        if thread_id is None:
            self._threads.clear()
        else:
            self._threads.pop(thread_id, None)

    def wrap(self, fn: NodeFn, *, text_extractor: TextExtractor | None = None) -> NodeFn:
        """Wrap a LangGraph node function with stagnation measurement.

        Supports both sync and async ``fn``. Forwards all positional and
        keyword arguments to ``fn`` so LangGraph's ``config`` / ``runtime``
        are preserved. Uses the same arg list to scope state per thread.
        """
        extract = text_extractor or (lambda out: str(out))

        if inspect.iscoroutinefunction(fn):

            async def async_wrapped(state: _NodeIn, *args: Any, **kwargs: Any) -> _NodeOut:
                output = await fn(state, *args, **kwargs)
                self._observe(extract(output), thread_id(args, kwargs))
                return output

            return async_wrapped

        def sync_wrapped(state: _NodeIn, *args: Any, **kwargs: Any) -> _NodeOut:
            output = fn(state, *args, **kwargs)
            self._observe(extract(output), thread_id(args, kwargs))
            return output

        return sync_wrapped

    def edge(self, forward: str, break_to: str) -> Callable[..., str]:
        """Conditional-edge router: returns ``break_to`` when stagnant."""

        def route(_state: _NodeIn, *args: Any, **kwargs: Any) -> str:
            tid = thread_id(args, kwargs)
            state = self._threads.get(tid)
            return break_to if state is not None and state.is_stagnant else forward

        return route

    # -- internals ----------------------------------------------------------

    def _thread_state(self, thread_id: str) -> _ThreadState:
        state = self._threads.get(thread_id)
        if state is None:
            state = _ThreadState(
                monitor=EpiplexityMonitor(
                    embedding_provider=self._embedder,
                    alpha=self._alpha,
                    window_size=self._window_size,
                    threshold=self._threshold,
                    critical_duration=self._critical_duration,
                ),
            )
            self._threads[thread_id] = state
        return state

    def _observe(self, text: str, thread_id: str) -> None:
        state = self._thread_state(thread_id)
        result = state.monitor.measure(text)

        # severity = 1 - epiplexity so "mean < threshold" means healthy; large
        # values flag stagnation pathology (matches the existing operon_ai
        # ``behavioral_stability`` verify semantics).
        severity = max(0.0, min(1.0, 1.0 - float(result.epiplexity)))
        state.severities.append(severity)

        # Detect on sustained low epiplexic_integral rather than the monitor's
        # built-in status classifier, which depends on a perplexity
        # approximation that varies with text shape and can mask stagnation
        # as CONVERGING. The integral is a direct, stable signal.
        if result.epiplexic_integral < self._threshold:
            state.low_integral_streak += 1
        else:
            state.low_integral_streak = 0

        was_stagnant = state.is_stagnant
        state.is_stagnant = state.low_integral_streak >= self._critical_duration

        if state.is_stagnant and not was_stagnant:
            _emit_certificate(state, self._threshold)


def _emit_certificate(state: _ThreadState, threshold: float) -> None:
    params = MappingProxyType(
        {
            "signal_values": tuple(state.severities),
            "threshold": float(threshold),
        }
    )
    cert = Certificate(
        theorem="behavioral_stability",
        parameters=params,
        conclusion=(
            f"Stagnation detected after {len(state.severities)} measurements; "
            f"severity evidence captured for replay verification."
        ),
        source="operon_langgraph_gates.stagnation",
        _verify_fn=_verify_behavioral_stability,
    )
    state.certificates.append(cert)
