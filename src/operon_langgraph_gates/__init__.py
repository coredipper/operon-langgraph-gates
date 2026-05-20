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

from .integrity import _THEOREM as INTEGRITY_THEOREM
from .integrity import IntegrityGate
from .stagnation import _WINDOWED_THEOREM as STAGNATION_THEOREM
from .stagnation import StagnationGate

__version__ = "0.1.1"

__all__ = [
    "INTEGRITY_THEOREM",
    "IntegrityGate",
    "STAGNATION_THEOREM",
    "StagnationGate",
    "__version__",
]
