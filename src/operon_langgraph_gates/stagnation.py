"""StagnationGate — Bayesian stagnation detection for LangGraph nodes and edges.

Backed by Paper 4 §4.3 (Bayesian stagnation detection, 96% accuracy with real
embeddings; Operon Papers). This module is a thin LangGraph-friendly wrapper
over ``operon_ai.health.epiplexity.EpiplexityMonitor`` — it adds:

- A text-extraction step (LangGraph node outputs are typically dicts, not text).
- A zero-dep default embedder (``NGramEmbedder``) so the gate works without
  a neural model install. Pass ``embedder=...`` for a real one.
- Instance-scoped state shared between ``wrap`` (records measurements) and
  ``edge`` (routes based on the latest status).
- ``behavioral_stability`` certificate emission on first detection.

Usage::

    gate = StagnationGate(threshold=0.2, critical_duration=3)
    graph.add_node("think", gate.wrap(think_fn))
    graph.add_conditional_edges("think", gate.edge(forward="answer", break_to="escalate"))
    # after graph.invoke(...)
    if gate.certificates:
        # stagnation was detected during the run
"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

from operon_ai.core.certificate import Certificate, _verify_behavioral_stability
from operon_ai.health.epiplexity import EmbeddingProvider, EpiplexityMonitor

from .embedders import NGramEmbedder

# LangGraph node functions consume and return arbitrary mappings (TypedDict,
# dict, or dataclass). We keep the typing loose on purpose — the gate does not
# care about the schema, only the extracted text.
_NodeIn = Any
_NodeOut = Any
NodeFn = Callable[[_NodeIn], _NodeOut]
TextExtractor = Callable[[_NodeOut], str]


class StagnationGate:
    """Detects stagnation on a per-node basis and routes conditional edges.

    Parameters mirror :class:`operon_ai.health.epiplexity.EpiplexityMonitor`
    but with LangGraph-friendly defaults:

    - ``threshold``: epiplexity below this for ``critical_duration`` readings
      in a row triggers a ``STAGNANT`` status.
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
        self._monitor = EpiplexityMonitor(
            embedding_provider=embedder or NGramEmbedder(),
            alpha=alpha,
            window_size=window_size,
            threshold=threshold,
            critical_duration=critical_duration,
        )
        self._threshold = threshold
        self._critical_duration = critical_duration
        self._low_integral_streak = 0
        self._is_stagnant = False
        self._severities: list[float] = []
        self._certificates: list[Certificate] = []

    # -- public API ---------------------------------------------------------

    @property
    def is_stagnant(self) -> bool:
        """Whether the most recent measurement indicated stagnation."""
        return self._is_stagnant

    @property
    def certificates(self) -> list[Certificate]:
        """Certificates emitted by this gate during its lifetime."""
        return list(self._certificates)

    def wrap(self, fn: NodeFn, *, text_extractor: TextExtractor | None = None) -> NodeFn:
        """Wrap a LangGraph node function with stagnation measurement."""
        extract = text_extractor or (lambda out: str(out))

        def wrapped(state: _NodeIn) -> _NodeOut:
            output = fn(state)
            self._observe(extract(output))
            return output

        return wrapped

    def edge(self, forward: str, break_to: str) -> Callable[[_NodeIn], str]:
        """Conditional-edge router: returns ``break_to`` when stagnant."""

        def route(_state: _NodeIn) -> str:
            return break_to if self._is_stagnant else forward

        return route

    # -- internals ----------------------------------------------------------

    def _observe(self, text: str) -> None:
        result = self._monitor.measure(text)
        # severity = 1 - epiplexity so "mean < threshold" means healthy and
        # large values indicate stagnation pathology (matches the existing
        # operon_ai ``behavioral_stability`` verify semantics).
        severity = max(0.0, min(1.0, 1.0 - float(result.epiplexity)))
        self._severities.append(severity)

        # Detect on sustained low epiplexic_integral rather than the monitor's
        # built-in status classifier, which depends on a perplexity
        # approximation that varies with text shape and can mask stagnation
        # as CONVERGING. The integral is a direct, stable signal.
        if result.epiplexic_integral < self._threshold:
            self._low_integral_streak += 1
        else:
            self._low_integral_streak = 0

        was_stagnant = self._is_stagnant
        self._is_stagnant = self._low_integral_streak >= self._critical_duration

        if self._is_stagnant and not was_stagnant:
            self._emit_certificate()

    def _emit_certificate(self) -> None:
        params = MappingProxyType(
            {
                "signal_values": tuple(self._severities),
                "threshold": float(self._threshold),
            }
        )
        cert = Certificate(
            theorem="behavioral_stability",
            parameters=params,
            conclusion=(
                f"Stagnation detected after {len(self._severities)} measurements; "
                f"severity evidence captured for replay verification."
            ),
            source="operon_langgraph_gates.stagnation",
            _verify_fn=_verify_behavioral_stability,
        )
        self._certificates.append(cert)
