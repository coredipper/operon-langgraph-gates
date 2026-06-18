"""``StagnationMiddleware`` — a ``create_agent`` adapter for :class:`StagnationGate`.

LangGraph `issue #6731 <https://github.com/langchain-ai/langgraph/issues/6731>`_:
a prebuilt agent (LangChain's ``create_agent`` / langgraph's
``create_react_agent``) re-issues the *same failing tool call* every turn until
the recursion limit fires, burning tokens with no early exit.

:class:`StagnationGate`'s ``wrap`` / ``edge`` API attaches to a ``StateGraph``
*you* build — but a prebuilt agent builds its graph internally, so there is no
node seam to attach to. This middleware bridges that gap. It implements the
LangChain v1 ``AgentMiddleware`` ``after_model`` hook: after every model call it
measures the model's output novelty and, once the output stagnates (repeated
near-verbatim tool calls / content — exactly the #6731 shape), it halts the loop
with ``jump_to="end"`` and emits the same ``behavioral_stability_windowed``
certificate the gate emits on a ``StateGraph``.

Detection, per-thread state, and certificate emission are delegated to an
internal :class:`StagnationGate`; this adapter only changes the *driver* (a
model-output message rather than a node return value). The graceful-exit
contract (``@hook_config(can_jump_to=["end"])`` + a ``jump_to`` state update)
mirrors LangChain's own ``ModelCallLimitMiddleware`` / ``ToolCallLimitMiddleware``.

Requires the ``langchain`` extra::

    pip install operon-langgraph-gates[langchain]

Usage::

    from langchain.agents import create_agent
    from operon_langgraph_gates.middleware import StagnationMiddleware

    mw = StagnationMiddleware(threshold=0.2, critical_duration=2)
    agent = create_agent(model=llm, tools=[...], middleware=[mw])
    result = agent.invoke({"messages": [...]})
    if mw.certificates:
        # the loop was broken early — replayable evidence is on the cert
        cert = mw.certificates[0]
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, BaseMessage
from operon_ai.core.certificate import Certificate
from operon_ai.health.epiplexity import EmbeddingProvider

from ._common import EPHEMERAL_THREAD, _extract_thread_id
from .stagnation import StagnationGate

MessageText = Callable[[BaseMessage], str]


def _default_message_text(message: BaseMessage) -> str:
    """Extract the text the gate measures from a model-output message.

    A #6731 loop repeats the *tool call* (same name + args) on every turn, so we
    serialize tool calls when present and fall back to the message content. This
    keeps the measured signal on what actually repeats rather than on incidental
    fields that might drift turn to turn.
    """
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return " ".join(f"{tc['name']}({tc['args']})" for tc in tool_calls)
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


class StagnationMiddleware(AgentMiddleware):
    """Halt a ``create_agent`` loop when the model's output stagnates.

    The constructor mirrors :class:`StagnationGate` (``threshold``,
    ``critical_duration``, ``window_size``, ``alpha``, ``embedder``) plus two
    adapter-specific options:

    - ``message_text``: how to turn a model-output message into the string the
      gate measures. Defaults to :func:`_default_message_text` (tool calls, else
      content).
    - ``stop_message``: the assistant message appended when the loop is broken.

    Inspect :attr:`certificates` after a run to see whether (and why) the loop
    was broken; the emitted certificate is the same replayable
    ``behavioral_stability_windowed`` artifact produced by the StateGraph gate.
    """

    DEFAULT_STOP_MESSAGE = (
        "Stopped early: the agent's output stagnated — operon StagnationGate "
        "detected repeated low-novelty model responses (LangGraph issue #6731). "
        "See the emitted behavioral_stability_windowed certificate for "
        "replayable evidence."
    )

    def __init__(
        self,
        threshold: float = 0.2,
        critical_duration: int = 2,
        window_size: int = 10,
        *,
        alpha: float = 0.5,
        embedder: EmbeddingProvider | None = None,
        message_text: MessageText | None = None,
        stop_message: str | None = None,
    ) -> None:
        super().__init__()
        self._gate = StagnationGate(
            threshold=threshold,
            critical_duration=critical_duration,
            window_size=window_size,
            alpha=alpha,
            embedder=embedder,
        )
        self._message_text: MessageText = message_text or _default_message_text
        self._stop_message = stop_message or self.DEFAULT_STOP_MESSAGE

    @property
    def gate(self) -> StagnationGate:
        """The underlying gate (for ``is_stagnant_for`` / ``integrals_for`` / ``reset``)."""
        return self._gate

    @property
    def certificates(self) -> list[Certificate]:
        """``behavioral_stability_windowed`` certificates emitted across all threads."""
        return self._gate.certificates

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Measure the just-produced model output; end the loop if it stagnated."""
        messages = state["messages"]
        if not messages:
            return None
        text = self._message_text(messages[-1])
        thread_id = _extract_thread_id(runtime) or EPHEMERAL_THREAD
        if self._gate.observe(text, thread_id=thread_id):
            return {
                "jump_to": "end",
                "messages": [AIMessage(content=self._stop_message)],
            }
        return None
