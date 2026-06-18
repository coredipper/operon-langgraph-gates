"""Public API surface tests.

Verifies the package-root exports are importable, equal their expected
values, and are listed in ``__all__``. This file is the home for future
public-API stability assertions (Wave 2 of the un-alpha track).
"""

from __future__ import annotations

import pytest

import operon_langgraph_gates as olg


def test_stagnation_theorem_constant_exported() -> None:
    assert olg.STAGNATION_THEOREM == "behavioral_stability_windowed"


def test_integrity_theorem_constant_exported() -> None:
    assert olg.INTEGRITY_THEOREM == "langgraph_state_integrity"


def test_theorem_constants_in_all() -> None:
    assert "STAGNATION_THEOREM" in olg.__all__
    assert "INTEGRITY_THEOREM" in olg.__all__


def test_theorem_constants_match_internal_ssot() -> None:
    """Public constants must equal the module-internal SSoT they re-export.

    Guards against accidental drift between the private ``_THEOREM`` /
    ``_WINDOWED_THEOREM`` constants and the public re-exports if either
    side is renamed without the other.
    """
    from operon_langgraph_gates.integrity import _THEOREM as integrity_internal
    from operon_langgraph_gates.stagnation import (
        _WINDOWED_THEOREM as stagnation_internal,
    )

    assert stagnation_internal == olg.STAGNATION_THEOREM
    assert integrity_internal == olg.INTEGRITY_THEOREM


def test_stagnation_gate_observe_is_public() -> None:
    """``observe`` is the public seam the create_agent middleware drives."""
    assert callable(olg.StagnationGate.observe)


def test_stagnation_middleware_not_in_all_but_lazily_importable() -> None:
    """The middleware needs the optional ``langchain`` extra, so it is kept out
    of ``__all__`` (``import *`` must stay langchain-free) yet remains importable
    via the explicit convenience path and the canonical submodule path."""
    assert "StagnationMiddleware" not in olg.__all__

    pytest.importorskip("langchain")
    from operon_langgraph_gates.middleware import StagnationMiddleware

    assert olg.StagnationMiddleware is StagnationMiddleware


def test_unknown_attribute_raises_attribute_error() -> None:
    """The package ``__getattr__`` must not swallow genuinely missing names."""
    with pytest.raises(AttributeError):
        olg.DefinitelyNotAThing  # noqa: B018
