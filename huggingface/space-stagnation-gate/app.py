"""Interactive StagnationGate demo for HuggingFace Spaces.

Walks a visitor through the LangGraph issue #6731 pathology: an agent
repeats itself, the `StagnationGate` sees the repeat, the gate flips
``is_stagnant=True``, a `behavioral_stability` certificate is emitted
with replayable evidence. Pick a preset, adjust the sliders, click Run —
no LLM calls, fully deterministic.

Runs locally with ``python app.py`` (binds 127.0.0.1:7860) or deployed
as a HuggingFace Space (see ``README.md`` frontmatter).
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import gradio as gr
from scenarios import DEMO_DEFAULTS, SCENARIOS, Scenario

from operon_langgraph_gates import StagnationGate
from operon_langgraph_gates._common import EPHEMERAL_THREAD

_REPO_URL = "https://github.com/coredipper/operon-langgraph-gates"
_ISSUE_URL = "https://github.com/langchain-ai/langgraph/issues/6731"
_PAPER4_URL = "https://github.com/coredipper/operon/blob/main/article/paper4/main.pdf"


# ---------------------------------------------------------------------------
# Core replay — no LLM, no LangGraph graph construction; the demo is about
# the gate's own behavior, the notebooks cover graph integration.
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    index: int
    output: str
    integral: float
    severity: float
    is_stagnant: bool


def replay(
    outputs: list[str],
    threshold: float,
    critical_duration: int,
    window_size: int,
) -> tuple[list[Turn], StagnationGate]:
    """Feed outputs turn-by-turn through a freshly-configured gate."""
    gate = StagnationGate(
        threshold=threshold,
        critical_duration=critical_duration,
        window_size=window_size,
    )
    stream = iter(outputs)
    wrapped = gate.wrap(lambda _s: {"answer": next(stream)})

    turns: list[Turn] = []
    for i, out in enumerate(outputs):
        wrapped({})
        state = gate._threads[EPHEMERAL_THREAD]
        severity = state.severities[-1] if state.severities else 0.0
        # Re-derive the window-mean integral the monitor just used.
        integral = _window_integral(state, window_size)
        turns.append(
            Turn(
                index=i,
                output=out,
                integral=integral,
                severity=severity,
                is_stagnant=gate.is_stagnant,
            )
        )
    return turns, gate


def _window_integral(state: object, window_size: int) -> float:
    """Mean of the last ``window_size`` epiplexity readings = 1 - severity."""
    severities = state.severities[-window_size:]  # type: ignore[attr-defined]
    if not severities:
        return 0.0
    mean_severity = sum(severities) / len(severities)
    return max(0.0, 1.0 - mean_severity)


# ---------------------------------------------------------------------------
# HTML rendering — pure strings, no matplotlib (matches operon-ai Spaces).
# ---------------------------------------------------------------------------

_STATUS_STYLES = {
    "HEALTHY": ("#0a7a3a", "#d4f5dd"),
    "WATCHING": ("#7a5a00", "#fff2b8"),
    "STAGNANT": ("#8a0f0f", "#fcd1d1"),
}


def _status_label(turn: Turn, threshold: float) -> str:
    if turn.is_stagnant:
        return "STAGNANT"
    if turn.integral < threshold:
        return "WATCHING"
    return "HEALTHY"


def _badge(status: str) -> str:
    fg, bg = _STATUS_STYLES[status]
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f"border-radius:10px;font-family:monospace;font-size:12px;"
        f'font-weight:600;">{status}</span>'
    )


def _escape_preview(text: str, limit: int = 80) -> str:
    t = text if len(text) <= limit else text[: limit - 1] + "…"
    return html.escape(t)


def render_turn_table(turns: list[Turn], threshold: float) -> str:
    rows = []
    for t in turns:
        status = _status_label(t, threshold)
        rows.append(
            "<tr>"
            f'<td style="padding:4px 10px;text-align:right;">{t.index}</td>'
            f'<td style="padding:4px 10px;font-family:monospace;">'
            f"{_escape_preview(t.output)}</td>"
            f'<td style="padding:4px 10px;text-align:right;font-family:monospace;">'
            f"{t.integral:.3f}</td>"
            f'<td style="padding:4px 10px;">{_badge(status)}</td>'
            "</tr>"
        )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:14px;">'
        '<thead><tr style="border-bottom:2px solid #444;">'
        '<th style="padding:6px 10px;text-align:right;">#</th>'
        '<th style="padding:6px 10px;text-align:left;">output (preview)</th>'
        '<th style="padding:6px 10px;text-align:right;">ε-integral</th>'
        '<th style="padding:6px 10px;text-align:left;">status</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def render_cert_card(gate: StagnationGate) -> str:
    certs = gate.certificates
    if not certs:
        return (
            '<div style="padding:16px;border:1px dashed #888;border-radius:8px;'
            'color:#888;">No certificate emitted — gate stayed healthy.</div>'
        )
    cert = certs[0]
    verification = cert.verify()
    evidence_items = "".join(
        f'<li style="padding:2px 0;font-family:monospace;font-size:12px;">'
        f"<b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>"
        for k, v in verification.evidence.items()
    )
    return (
        '<div style="padding:16px;border:2px solid #8a0f0f;background:#fcd1d1;'
        'border-radius:8px;">'
        f'<div style="font-weight:700;font-size:16px;margin-bottom:6px;">'
        f"Certificate: <code>{html.escape(cert.theorem)}</code></div>"
        f'<div style="margin-bottom:6px;">{html.escape(cert.conclusion)}</div>'
        f'<div style="font-size:12px;color:#444;margin-bottom:4px;">'
        f"<b>verify().holds</b>: <code>{verification.holds}</code></div>"
        f'<ul style="margin:0;padding-left:20px;">{evidence_items}</ul>'
        "</div>"
    )


def render_narration(scenario: Scenario, turns: list[Turn]) -> str:
    fired_at = next((t.index for t in turns if t.is_stagnant), None)
    if fired_at is None:
        fired_sentence = "The gate did not fire on this trajectory."
    else:
        fired_sentence = f"The gate flipped ``is_stagnant=True`` on turn {fired_at}."
    return f"**{scenario.label}** — {scenario.narration}  \n\n{fired_sentence}"


# ---------------------------------------------------------------------------
# Gradio wiring
# ---------------------------------------------------------------------------


def on_run(
    preset_key: str,
    custom_text: str,
    threshold: float,
    critical_duration: int,
    window_size: int,
) -> tuple[str, str, str]:
    if preset_key == "custom":
        outputs = [line for line in custom_text.splitlines() if line.strip()]
        scenario = Scenario(
            key="custom",
            label="Custom trajectory",
            short_description="User-supplied outputs.",
            narration=(
                "One output per non-empty line. The gate is run with the "
                "slider parameters exactly as set."
            ),
            outputs=tuple(outputs),
        )
    else:
        scenario = SCENARIOS[preset_key]
        outputs = list(scenario.outputs)

    if not outputs:
        empty_msg = (
            '<div style="padding:16px;color:#888;">'
            "Provide at least one output line to run the gate.</div>"
        )
        return empty_msg, empty_msg, "_No outputs provided._"

    turns, gate = replay(outputs, threshold, int(critical_duration), int(window_size))
    return (
        render_turn_table(turns, threshold),
        render_cert_card(gate),
        render_narration(scenario, turns),
    )


_HEADER_MD = f"""
# 🔁 StagnationGate — Interactive Demo

**Problem**: LangGraph [issue #6731]({_ISSUE_URL}) — *agent infinite-loops
until recursion limit, burns tokens invisibly* — was closed as NOT_PLANNED.

**Fix**: [`operon-langgraph-gates`]({_REPO_URL}) ships a drop-in
`StagnationGate` that detects the repetition, flips a routing flag on a
conditional edge, and emits a replayable certificate. This page is the
gate running on a handful of hand-picked trajectories. Pick a preset,
tune the sliders, and watch the gate react.

*Backed by [Paper 4 §4.3]({_PAPER4_URL}): convergence / false-stagnation
accuracy 0.960 with real sentence embeddings. See
[`docs/paper-citations.md`]({_REPO_URL}/blob/main/docs/paper-citations.md)
for the full record and the loop-detection caveat.*
"""

_FOOTER_MD = f"""
---

**See also**: the two runnable example notebooks in the repo:
[`01_stagnation_breaks_loop.ipynb`]({_REPO_URL}/blob/main/examples/01_stagnation_breaks_loop.ipynb)
shows the gate breaking a real LangGraph `StateGraph` self-loop;
[`02_integrity_catches_drift.ipynb`]({_REPO_URL}/blob/main/examples/02_integrity_catches_drift.ipynb)
covers the companion `IntegrityGate`.
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Operon StagnationGate Demo") as app:
        gr.Markdown(_HEADER_MD)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Controls")
                preset_choices = [(s.label, k) for k, s in SCENARIOS.items()]
                preset_choices.append(("Custom (paste your own)", "custom"))
                preset = gr.Dropdown(
                    choices=preset_choices,
                    value="identical",
                    label="Preset",
                )
                custom_text = gr.Textbox(
                    lines=6,
                    label="Custom outputs (one per line)",
                    placeholder="Line 1 is turn 0\nLine 2 is turn 1\n…",
                    visible=False,
                )
                threshold = gr.Slider(
                    0.05,
                    0.5,
                    value=float(DEMO_DEFAULTS["threshold"]),
                    step=0.01,
                    label="threshold (ε-integral below this counts as low)",
                )
                critical_duration = gr.Slider(
                    1,
                    5,
                    value=int(DEMO_DEFAULTS["critical_duration"]),
                    step=1,
                    label="critical_duration (consecutive low readings)",
                )
                window_size = gr.Slider(
                    2,
                    10,
                    value=int(DEMO_DEFAULTS["window_size"]),
                    step=1,
                    label="window_size (sliding window for integral)",
                )
                run_btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Per-turn gate state")
                turn_table = gr.HTML()
                gr.Markdown("### Certificate")
                cert_card = gr.HTML()
                gr.Markdown("### What just happened")
                narration = gr.Markdown()

        gr.Markdown(_FOOTER_MD)

        def _toggle_custom(preset_key: str) -> gr.Textbox:
            return gr.Textbox(visible=(preset_key == "custom"))

        preset.change(_toggle_custom, inputs=preset, outputs=custom_text)

        run_btn.click(
            on_run,
            inputs=[preset, custom_text, threshold, critical_duration, window_size],
            outputs=[turn_table, cert_card, narration],
        )

        # Run once on load so the page is not empty.
        app.load(
            on_run,
            inputs=[preset, custom_text, threshold, critical_duration, window_size],
            outputs=[turn_table, cert_card, narration],
        )

    return app


if __name__ == "__main__":
    build_app().launch()
