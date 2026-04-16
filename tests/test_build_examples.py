"""Regression test for ``scripts/build_examples.py``.

Guards the two properties that prior reviews (roborev #715, #717)
flagged as easy to regress silently:

- Cell IDs are deterministic across regenerations (no random slugs),
  so ``git diff`` on a no-op rebuild is empty.
- The integrity example demonstrates *real* LangGraph state corruption
  (``user_id`` is written as ``None``, not merely omitted from the
  partial update).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_examples.py"


def _load_builder() -> object:
    spec = importlib.util.spec_from_file_location("build_examples", SCRIPT)
    assert spec is not None and spec.loader is not None, "script must be loadable"
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_examples"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> object:
    return _load_builder()


def test_stagnation_cells_have_stable_ids(builder: object) -> None:
    cells = list(builder.NB1_CELLS)  # type: ignore[attr-defined]
    builder._assign_stable_ids(cells, "stagnation")  # type: ignore[attr-defined]
    assert [c["id"] for c in cells] == [f"stagnation-{i:02d}" for i in range(len(cells))]


def test_integrity_cells_have_stable_ids(builder: object) -> None:
    cells = list(builder.NB2_CELLS)  # type: ignore[attr-defined]
    builder._assign_stable_ids(cells, "integrity")  # type: ignore[attr-defined]
    assert [c["id"] for c in cells] == [f"integrity-{i:02d}" for i in range(len(cells))]


def test_id_assignment_is_idempotent(builder: object) -> None:
    """Running the ID assignment twice produces the same IDs — proves
    regeneration is a no-op on IDs rather than rotating them."""
    cells_a = list(builder.NB1_CELLS)  # type: ignore[attr-defined]
    cells_b = list(builder.NB1_CELLS)  # type: ignore[attr-defined]
    builder._assign_stable_ids(cells_a, "stagnation")  # type: ignore[attr-defined]
    builder._assign_stable_ids(cells_b, "stagnation")  # type: ignore[attr-defined]
    assert [c["id"] for c in cells_a] == [c["id"] for c in cells_b]


def test_integrity_example_uses_explicit_none(builder: object) -> None:
    """The integrity notebook must demonstrate *real* LangGraph-visible
    corruption: setting user_id to None in the partial update so the
    value survives the merge, not just omitting it."""
    joined = "\n".join(
        c["source"]
        for c in builder.NB2_CELLS
        if c["cell_type"] == "code"  # type: ignore[attr-defined]
    )
    assert '"user_id": None' in joined, (
        "bad_transform must explicitly write user_id=None so the corruption "
        "survives LangGraph's partial-update merge"
    )
    assert "bad_transform" in joined
    # Both invariants must be wired in.
    assert "has_user_id" in joined and "budget_not_exceeded" in joined
    # IntegrityGate must be imported and used.
    assert "IntegrityGate" in joined


def test_integrity_example_explains_scoping(builder: object) -> None:
    """Prose must make it explicit that invariants run on the node's
    partial output (not merged state) — fix from roborev #715."""
    joined = "\n".join(
        c["source"]
        for c in builder.NB2_CELLS
        if c["cell_type"] == "markdown"  # type: ignore[attr-defined]
    )
    assert "partial output" in joined.lower()
