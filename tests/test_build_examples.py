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
    """Running the ID assignment twice on independent copies produces
    the same IDs — proves regeneration is a no-op on IDs rather than
    rotating them. Uses deepcopy so mutations on one copy can't leak
    into the other (shallow ``list()`` over NotebookNode references
    would share mutable cells and mask non-idempotence)."""
    from copy import deepcopy

    cells_a = deepcopy(builder.NB1_CELLS)  # type: ignore[attr-defined]
    cells_b = deepcopy(builder.NB1_CELLS)  # type: ignore[attr-defined]
    builder._assign_stable_ids(cells_a, "stagnation")  # type: ignore[attr-defined]
    builder._assign_stable_ids(cells_b, "stagnation")  # type: ignore[attr-defined]
    assert [c["id"] for c in cells_a] == [c["id"] for c in cells_b]
    # And also a no-op second pass on the same list must not change IDs.
    ids_before = [c["id"] for c in cells_a]
    builder._assign_stable_ids(cells_a, "stagnation")  # type: ignore[attr-defined]
    assert [c["id"] for c in cells_a] == ids_before


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


def test_normalize_notebook_strips_cell_execution_and_kernel_noise(
    builder: object,
) -> None:
    """``_normalize_notebook`` must remove both per-cell ``metadata.execution``
    timestamps and the kernel-populated top-level metadata fields (e.g.
    ``language_info.version``) that make executed notebooks diff per
    Python/kernel version."""
    polluted = {
        "cells": [
            {
                "cell_type": "code",
                "source": "print('hi')",
                "metadata": {
                    "execution": {
                        "iopub.execute_input": "2026-04-17T12:00:00.000Z",
                        "shell.execute_reply": "2026-04-17T12:00:00.100Z",
                    },
                    "something_the_user_set": True,
                },
                "outputs": [],
            }
        ],
        "metadata": {
            "kernelspec": {"name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.11.15",  # noisy — varies per machine
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
            },
            "orig_nbformat": 4,
        },
    }

    builder._normalize_notebook(polluted)  # type: ignore[attr-defined]

    # Cell-level: execution stripped, user-set metadata preserved.
    assert "execution" not in polluted["cells"][0]["metadata"]
    assert polluted["cells"][0]["metadata"]["something_the_user_set"] is True

    # Top-level: reset to canonical whitelist; version/extension gone.
    assert sorted(polluted["metadata"].keys()) == ["kernelspec", "language_info"]
    assert polluted["metadata"]["language_info"] == {"name": "python"}
    assert "version" not in polluted["metadata"]["language_info"]


def test_execute_flag_produces_byte_identical_output(
    tmp_path: Path, builder: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``--execute`` workflow must be byte-for-byte idempotent: writing
    the same notebook twice (with a stub executor that produces
    deterministic outputs) yields the same file. Uses a monkeypatched
    ``NotebookClient`` so this test is instant and doesn't need an
    ipykernel."""
    # Stub NotebookClient so .execute() just sets deterministic outputs and
    # injects the noisy kernel metadata that _normalize_notebook should strip.
    import sys as _sys
    import types as _types

    from nbformat.v4 import new_output

    def _mk_client(nb: object, timeout: int, kernel_name: str) -> object:
        class _StubClient:
            def __init__(self) -> None:
                self.nb = nb

            def execute(self) -> None:
                for cell in self.nb.cells:  # type: ignore[attr-defined]
                    if cell["cell_type"] == "code":
                        cell["execution_count"] = 1
                        cell["outputs"] = [
                            new_output("stream", name="stdout", text="ok\n"),
                        ]
                        cell.setdefault("metadata", {})["execution"] = {
                            "iopub.execute_input": "ignored-timestamp",
                        }
                # Simulate kernel populating version fields.
                self.nb.metadata["language_info"] = {  # type: ignore[attr-defined]
                    "name": "python",
                    "version": "3.11.15",
                    "file_extension": ".py",
                }

        return _StubClient()

    fake_nbclient = _types.ModuleType("nbclient")
    fake_nbclient.NotebookClient = _mk_client  # type: ignore[attr-defined]
    monkeypatch.setitem(_sys.modules, "nbclient", fake_nbclient)

    # Redirect ROOT + EXAMPLES to a tmp dir so we don't touch the real files
    # and the _write() relative_to(ROOT) call stays in-tree.
    monkeypatch.setattr(builder, "ROOT", tmp_path)  # type: ignore[attr-defined]
    monkeypatch.setattr(builder, "EXAMPLES", tmp_path)  # type: ignore[attr-defined]
    # Stub argv to include --execute so main() actually exercises the CLI
    # branch (not just the source-only path + a direct _execute call).
    monkeypatch.setattr(_sys, "argv", ["build_examples.py", "--execute"])

    # Run main() twice; both outputs must be byte-identical, and the
    # if args.execute: branch must have produced populated outputs on each
    # run without a direct _execute_all() call here.
    builder.main()  # type: ignore[attr-defined]
    run1 = (tmp_path / "01_stagnation_breaks_loop.ipynb").read_bytes()

    builder.main()  # type: ignore[attr-defined]
    run2 = (tmp_path / "01_stagnation_breaks_loop.ipynb").read_bytes()

    assert run1 == run2, "regenerate-and-execute must be byte-identical"

    # And verify the notebook actually has populated outputs + no timestamps.
    import json

    data = json.loads(run1)
    code_cells = [c for c in data["cells"] if c["cell_type"] == "code"]
    assert all(c["outputs"] for c in code_cells)
    assert all("execution" not in c.get("metadata", {}) for c in code_cells)
    assert data["metadata"]["language_info"] == {"name": "python"}


