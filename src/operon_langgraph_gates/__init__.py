"""Reliability primitives for LangGraph.

Two gates, both thin wrappers over Operon's certificate framework:

- ``StagnationGate``: Bayesian stagnation detection on conditional edges or
  node outputs. Emits ``behavioral_stability_windowed`` certificates.
- ``IntegrityGate``: runtime invariant check on checkpointer writes. Emits
  ``langgraph_state_integrity`` certificates.

Backed by Paper 4 §4.3 (stagnation, 96% real-embedding accuracy) and §4.1
(DNA-repair integrity, 3/3 benchmarks) + Paper 5 §3 (certificate preservation
under compilation).

The theorem names emitted by each gate are also exposed as public
constants — ``STAGNATION_THEOREM`` and ``INTEGRITY_THEOREM`` — so
downstream consumers can key on them without hard-coding strings that
might drift if the underlying theorem is renamed in ``operon-ai``.
"""

from typing import TYPE_CHECKING

from .integrity import _THEOREM as INTEGRITY_THEOREM
from .integrity import IntegrityGate
from .stagnation import _WINDOWED_THEOREM as STAGNATION_THEOREM
from .stagnation import StagnationGate

if TYPE_CHECKING:
    # Re-export alias (``X as X``) marks this as intentional for type checkers /
    # IDEs resolving ``operon_langgraph_gates.StagnationMiddleware`` statically,
    # without listing it in ``__all__`` (see note below).
    from .middleware import StagnationMiddleware as StagnationMiddleware

__version__ = "0.2.0"

# ``StagnationMiddleware`` is intentionally *not* in ``__all__``: it requires the
# optional ``langchain`` extra, so listing it would make ``from … import *`` fail
# in a core-only install. The explicit convenience import
# ``from operon_langgraph_gates import StagnationMiddleware`` still works via
# ``__getattr__`` below; the canonical path is ``…middleware.StagnationMiddleware``.
__all__ = [
    "INTEGRITY_THEOREM",
    "IntegrityGate",
    "STAGNATION_THEOREM",
    "StagnationGate",
    "__version__",
]


def __getattr__(name: str) -> object:
    """Lazily expose the ``create_agent`` adapter.

    ``StagnationMiddleware`` lives in :mod:`operon_langgraph_gates.middleware`
    and pulls in the optional ``langchain`` dependency. Importing it lazily here
    keeps ``import operon_langgraph_gates`` langchain-free for StateGraph-only
    users, while still allowing the convenience import
    ``from operon_langgraph_gates import StagnationMiddleware`` (and a friendly
    error if the extra is missing).
    """
    if name == "StagnationMiddleware":
        try:
            from .middleware import StagnationMiddleware
        except ImportError as exc:  # pragma: no cover - requires missing extra
            raise ImportError(
                "StagnationMiddleware requires the 'langchain' extra: "
                "pip install operon-langgraph-gates[langchain]"
            ) from exc
        return StagnationMiddleware
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
