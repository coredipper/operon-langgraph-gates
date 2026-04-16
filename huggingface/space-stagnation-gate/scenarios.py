"""Hardcoded preset output trajectories for the StagnationGate demo Space.

Each preset is a list of agent-output strings that, when fed turn-by-turn
into a ``StagnationGate`` configured with :data:`DEMO_DEFAULTS`, produces
a specific + deterministic stagnation trajectory. No LLM calls — zero
API cost, fully reproducible.

Trajectories are empirically tuned against the zero-dep ``NGramEmbedder``.
Re-tuning is required if you swap in a neural embedder or change
:data:`DEMO_DEFAULTS`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Gate parameters used both by the Space's UI sliders (as initial values)
# and by the regression test that guards the preset trajectories. Tuned
# for short demo sequences (10 turns): window_size smaller than production
# default so the sliding integral reacts within the demo window.
DEMO_DEFAULTS: dict[str, float | int] = {
    "threshold": 0.2,
    "critical_duration": 2,
    "window_size": 3,
}


@dataclass(frozen=True)
class Scenario:
    """A named demo trajectory."""

    key: str
    label: str
    short_description: str
    narration: str
    outputs: tuple[str, ...]


IDENTICAL_OUTPUTS = Scenario(
    key="identical",
    label="Identical outputs (gate fires)",
    short_description="Agent returns the exact same answer every turn.",
    narration=(
        "The agent emits an identical output on every turn — the worst case "
        "of LangGraph issue #6731. The gate's epiplexic integral stays below "
        "threshold after a short warmup and the gate flips "
        "``is_stagnant=True`` around turn 4–5."
    ),
    outputs=tuple(["I need to think about this more carefully."] * 10),
)

DIVERSE_OUTPUTS = Scenario(
    key="diverse",
    label="Diverse outputs (healthy)",
    short_description="Agent produces a genuinely different answer each turn.",
    narration=(
        "The outputs share no common templates. Trigram overlap between "
        "successive turns is low, the epiplexic integral stays well above "
        "threshold, and the gate never fires — the desired healthy behavior."
    ),
    outputs=(
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump.",
        "Sphinx of black quartz, judge my vow.",
        "Two driven jocks help fax my big quiz.",
        "The five boxing wizards jump quickly.",
        "Waltz, bad nymph, for quick jigs vex.",
        "Jackdaws love my big sphinx of quartz.",
        "Bright vixens jump dozy fowl quack.",
        "Crazy Fredrick bought many very exquisite opal jewels.",
    ),
)

NEAR_IDENTICAL_WITH_NOISE = Scenario(
    key="noisy_repeat",
    label="Near-identical with noise (gate still fires)",
    short_description="Same template with a tiny bit of changing text per turn.",
    narration=(
        "A common real-world pathology: the agent is stuck but adds a "
        "surface-level variation (a counter, a timestamp, a reworded phrase) "
        "on every turn. The gate looks through the noise because the n-gram "
        "distribution is dominated by the shared template; stagnation still "
        "fires, just a turn or two later than the pure-identical case."
    ),
    outputs=tuple(
        f"Let me re-check this answer carefully. Attempt {i} is similar to the last one."
        for i in range(10)
    ),
)

SLOW_DRIFT = Scenario(
    key="slow_drift",
    label="Slow drift into repetition",
    short_description="Healthy at first, then the agent starts looping.",
    narration=(
        "The agent explores normally for four turns, then falls into a loop. "
        "The gate's sliding-window integral stays healthy through the "
        "exploratory phase, then drops below threshold once the repeats "
        "dominate the window. Shows the gate's reaction time on a realistic "
        "collapse trajectory rather than an all-identical pathology."
    ),
    outputs=(
        "The sunrise paints the eastern sky with orange hues.",
        "Coffee shops bustle with early-morning commuter energy.",
        "Commuters rush through the subway tunnels to work.",
        "Office lights flicker on in the tall glass buildings.",
        "I need to think about this more carefully.",
        "I need to think about this more carefully.",
        "I need to think about this more carefully.",
        "I need to think about this more carefully.",
        "I need to think about this more carefully.",
        "I need to think about this more carefully.",
    ),
)


SCENARIOS: dict[str, Scenario] = {
    s.key: s for s in (IDENTICAL_OUTPUTS, DIVERSE_OUTPUTS, NEAR_IDENTICAL_WITH_NOISE, SLOW_DRIFT)
}
