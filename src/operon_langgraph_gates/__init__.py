"""Reliability primitives for LangGraph.

Two gates, both thin wrappers over Operon's certificate framework:

- ``StagnationGate``: Bayesian stagnation detection on conditional edges or
  node outputs. Emits ``behavioral_stability`` certificates.
- ``IntegrityGate``: runtime invariant check on checkpointer writes. Emits
  ``state_integrity_verified`` certificates.

Backed by Paper 4 §4.3 (stagnation, 96% real-embedding accuracy) and §4.1
(DNA-repair integrity, 3/3 benchmarks) + Paper 5 §3 (certificate preservation
under compilation).
"""

from .integrity import IntegrityGate
from .stagnation import StagnationGate

__version__ = "0.1.0a1"

__all__ = ["IntegrityGate", "StagnationGate", "__version__"]
