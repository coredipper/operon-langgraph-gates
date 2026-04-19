"""StagnationGate — Bayesian stagnation detection for LangGraph nodes and edges.

Backed by Operon Paper 4 §4.3 "Epiplexity: Embedding Quality Determines
Outcome". With real sentence embeddings (all-MiniLM-L6-v2, N = 3 seeds
× 100 trials = 300), the biological two-signal monitor achieves:

- **convergence discrimination**: 0.960 accuracy (vs 0.401 naive baseline)
- **false-stagnation rejection**: 0.960 accuracy, 0.000 FP rate
  (vs 0.020 accuracy, 0.980 FP rate for the naive baseline)
- **loop detection**: 0.631 accuracy (the naive detector scores 0.940
  here, so the strength of this gate is in convergence/false-stagnation
  rather than loop detection per se)

Authoritative source for these numbers (path relative to the Operon
repo root, commit ``339875e``):
``eval/results/benchmarks_real_embeddings/multi_model_summary.json`` at
``github.com/coredipper/operon``. A full citation record with verbatim
paper quotes and reproduction commands is at ``docs/paper-citations.md``.

In practice the gate is effective at breaking stuck loops (see
``tests/integration/test_loop_break.py``) because most real-world
stagnation presents as identical or near-identical outputs that trip the
epiplexic integral rather than the classifier's loop scenario. Using
``epiplexic_integral`` directly — rather than the monitor's status
classifier — gives the gate its stable detection signal.

This module is a thin LangGraph-friendly wrapper over
``operon_ai.health.epiplexity.EpiplexityMonitor``. It adds:

- A text-extraction step (LangGraph node outputs are typically dicts, not
  text).
- A zero-dep default embedder (``NGramEmbedder``) so the gate works
  without a neural model install. Pass ``embedder=...`` for a real one;
  the Paper 4 numbers above were measured with ``all-MiniLM-L6-v2``.
- Per-thread state scoping keyed on
  ``config["configurable"]["thread_id"]`` so reusing a compiled graph
  across invocations or running concurrent threads doesn't leak state.
- Async node support via ``_common.is_async_callable`` (also catches
  classes with ``async def __call__``).
- ``behavioral_stability`` certificate emission on first detection per
  thread.

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

from operon_ai.core.certificate import Certificate
from operon_ai.health.epiplexity import EmbeddingProvider, EpiplexityMonitor

from ._common import EPHEMERAL_THREAD, is_async_callable, thread_id
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
    # ``integrals[i]`` is the ``EpiplexityResult.epiplexic_integral`` the
    # monitor produced on turn ``i``. This is the exact signal the gate's
    # detection logic routes on — surfacing it lets UI clients (e.g. the
    # demo Space) render the same number the gate acted on, rather than
    # re-deriving something that can drift (e.g. ``1 - mean(severity)``
    # does not match the α-mixed formula once perplexity is in play).
    integrals: list[float] = field(default_factory=list)
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

        For per-thread checks inside a LangGraph run, use :meth:`edge` (which
        extracts thread_id from config automatically) or
        :meth:`is_stagnant_for`.
        """
        state = self._threads.get(EPHEMERAL_THREAD)
        return state is not None and state.is_stagnant

    def is_stagnant_for(self, thread_id_: str) -> bool:
        """Whether a specific LangGraph thread is currently stagnant."""
        state = self._threads.get(thread_id_)
        return state is not None and state.is_stagnant

    @property
    def certificates(self) -> list[Certificate]:
        """All certificates emitted across all threads during this gate's life."""
        return [c for s in self._threads.values() for c in s.certificates]

    def integrals_for(self, thread_id_: str | None = None) -> list[float]:
        """Per-turn ``epiplexic_integral`` values observed on a thread.

        Returns the exact integral sequence the detection logic routed on,
        so UI clients (e.g. the demo Space) can render the same number the
        gate acted on rather than re-deriving one that may not match the
        monitor's α-mixed formula.

        ``thread_id_ is None`` returns the ephemeral thread's history —
        matching :attr:`is_stagnant` semantics for single-thread callers.
        """
        key = EPHEMERAL_THREAD if thread_id_ is None else thread_id_
        state = self._threads.get(key)
        return list(state.integrals) if state is not None else []

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

        if is_async_callable(fn):

            async def async_wrapped(state: _NodeIn, *args: Any, **kwargs: Any) -> _NodeOut:
                output = await fn(state, *args, **kwargs)
                self._observe(extract(output), thread_id(args, kwargs))
                return output

            return async_wrapped

        def sync_wrapped(state: _NodeIn, *args: Any, **kwargs: Any) -> _NodeOut:
            output = fn(state, *args, **kwargs)
            if inspect.isawaitable(output):
                # Close the coroutine before raising so we don't leak a
                # "coroutine was never awaited" RuntimeWarning on top of the
                # error we're already surfacing.
                if inspect.iscoroutine(output):
                    output.close()
                raise TypeError(
                    f"{fn!r} returned an awaitable but was not detected as async. "
                    "Declare it as ``async def`` or call it via the ainvoke path; "
                    "the sync gate wrapper cannot measure an unawaited coroutine."
                )
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
        # Record the exact integral the detection logic routes on, so
        # callers rendering the gate's decision see the same number.
        integral = float(result.epiplexic_integral)
        state.integrals.append(integral)

        # Detect on sustained low epiplexic_integral rather than the monitor's
        # built-in status classifier, which depends on a perplexity
        # approximation that varies with text shape and can mask stagnation
        # as CONVERGING. The integral is a direct, stable signal.
        if integral < self._threshold:
            state.low_integral_streak += 1
        else:
            state.low_integral_streak = 0

        was_stagnant = state.is_stagnant
        should_be_stagnant = state.low_integral_streak >= self._critical_duration

        if should_be_stagnant and not was_stagnant:
            # Emit the certificate *before* flipping ``state.is_stagnant`` so a
            # failure (e.g. unregistered theorem) leaves the thread in its
            # prior non-stagnant state and a later call can retry emission
            # once the underlying problem is corrected. If we flipped state
            # first, the ``was_stagnant/not was_stagnant`` guard above would
            # suppress retry and the thread would be permanently stuck in
            # ``is_stagnant=True`` without a corresponding certificate.
            violating_integrals = state.integrals[-self._critical_duration :]
            window_severity_means = tuple(1.0 - i for i in violating_integrals)
            cert = _emit_certificate(
                window_severity_means=window_severity_means,
                threshold=1.0 - self._threshold,
                detection_index=len(state.severities),
                source="operon_langgraph_gates.stagnation",
            )
            state.certificates.append(cert)
        state.is_stagnant = should_be_stagnant


# A theorem name distinct from the shared ``behavioral_stability`` (which is
# flat-mean-based). Upstream ``operon_ai.core.certificate`` registers this
# theorem in its theorem registry, so ``Certificate.from_theorem`` resolves
# the correct verifier without this package binding to any upstream symbol
# beyond the public ``Certificate`` class itself.
_WINDOWED_THEOREM = "behavioral_stability_windowed"


def _emit_certificate(
    *,
    window_severity_means: tuple[float, ...],
    threshold: float,
    detection_index: int,
    source: str,
) -> Certificate:
    """Emit a ``behavioral_stability_windowed`` certificate backed by per-window means.

    ``signal_values`` is the sequence of mean severities over each violating
    rolling window — one value per window, ``critical_duration`` values total.
    The replay check is ``max(signal_values) <= threshold``, which mirrors
    detection's "every one of the last ``critical_duration`` rolling windows
    was violating" predicate.

    ``threshold`` is the stability threshold in the severity domain (the
    complement of the detection threshold, ``1 - τ_detect``). This makes the
    verifier agree with detection at every τ in [0, 1] — the severity-domain
    and epiplexity-domain predicates are equivalent only when the threshold
    is translated.

    ``detection_index`` is the total number of evaluations the critic had
    performed when stagnation was detected (``len(state.severities)`` at emit
    time). Used only in the human-readable conclusion text so "after N
    measurements" remains accurate after a long healthy prefix — the
    evidence slice is narrower than total history.

    Raises ``ValueError`` if ``window_severity_means`` is empty: a stagnation
    certificate without evidence is a contradiction in terms, and the Pydantic
    ``critical_duration >= 1`` guarantee makes this unreachable from the gate.
    """
    if not window_severity_means:
        raise ValueError(
            "window_severity_means must be non-empty; "
            "a stagnation certificate requires at least one violating window"
        )
    return Certificate.from_theorem(
        theorem=_WINDOWED_THEOREM,
        parameters=MappingProxyType(
            {
                "signal_values": tuple(window_severity_means),
                "threshold": float(threshold),
            }
        ),
        conclusion=(
            f"Stagnation detected after {detection_index} measurements; "
            f"{len(window_severity_means)} violating rolling windows "
            f"captured for replay verification."
        ),
        source=source,
    )
