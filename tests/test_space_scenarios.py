"""Guard the StagnationGate demo Space's preset trajectories.

The Space's value proposition is that a user can click each preset and
see a predictable outcome. If the gate defaults or the NGramEmbedder
change in a way that shifts a preset's trajectory (fires too late,
doesn't fire at all, false-positives on diverse outputs), the demo
regresses silently. These tests are the guard.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from operon_langgraph_gates import StagnationGate

ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "huggingface" / "space-stagnation-gate"


def _load_scenarios() -> object:
    spec = importlib.util.spec_from_file_location("scenarios", SPACE_DIR / "scenarios.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["scenarios"] = module
    spec.loader.exec_module(module)
    return module


def _replay(outputs: tuple[str, ...]) -> tuple[StagnationGate, list[bool]]:
    """Feed a scenario's outputs into a gate configured with the Space's
    UI defaults and record ``is_stagnant`` after each turn."""
    mod = _load_scenarios()
    defaults = mod.DEMO_DEFAULTS  # type: ignore[attr-defined]
    gate = StagnationGate(**defaults)
    stream = iter(outputs)
    wrapped = gate.wrap(lambda _state: {"answer": next(stream)})
    flips: list[bool] = []
    for _ in outputs:
        wrapped({})
        flips.append(gate.is_stagnant)
    return gate, flips


def test_identical_preset_fires_within_6_turns() -> None:
    mod = _load_scenarios()
    scenario = mod.IDENTICAL_OUTPUTS  # type: ignore[attr-defined]
    _, flips = _replay(scenario.outputs)
    # Must have flipped to True by turn 6 (index 5) at the latest.
    assert any(flips[:6]), f"expected stagnation by turn 6; trajectory={flips}"


def test_diverse_preset_never_fires() -> None:
    mod = _load_scenarios()
    scenario = mod.DIVERSE_OUTPUTS  # type: ignore[attr-defined]
    _, flips = _replay(scenario.outputs)
    assert not any(flips), f"diverse outputs must not trip the gate; trajectory={flips}"


def test_noisy_repeat_preset_fires() -> None:
    mod = _load_scenarios()
    scenario = mod.NEAR_IDENTICAL_WITH_NOISE  # type: ignore[attr-defined]
    _, flips = _replay(scenario.outputs)
    assert any(flips), (
        f"near-identical outputs with surface noise must still trip the gate; trajectory={flips}"
    )


def test_slow_drift_preset_fires_after_warmup() -> None:
    mod = _load_scenarios()
    scenario = mod.SLOW_DRIFT  # type: ignore[attr-defined]
    _, flips = _replay(scenario.outputs)
    # First 4 turns are diverse — must still be healthy then.
    assert not flips[0] and not flips[3], (
        f"slow-drift preset must stay healthy through the exploratory phase; flips[:4]={flips[:4]}"
    )
    # By the end of the 10-turn trajectory the gate must have fired.
    assert flips[-1], f"slow-drift must end in stagnation; trajectory={flips}"


def test_all_presets_registered_in_scenarios_dict() -> None:
    mod = _load_scenarios()
    registered = mod.SCENARIOS  # type: ignore[attr-defined]
    expected_keys = {"identical", "diverse", "noisy_repeat", "slow_drift"}
    assert set(registered.keys()) == expected_keys
    for key, scenario in registered.items():
        assert scenario.key == key
        assert scenario.label and scenario.narration and scenario.outputs
