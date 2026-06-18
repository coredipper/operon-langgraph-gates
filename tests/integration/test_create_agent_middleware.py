"""Integration: ``StagnationMiddleware`` breaks a ``create_agent`` loop (#6731).

Mirrors ``examples/03_stagnation_middleware_create_agent.ipynb`` with a
deterministic stand-in chat model so no API key is required. The whole module
is skipped when the optional ``langchain`` extra is not installed.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langchain")

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402

from operon_langgraph_gates import StagnationMiddleware  # noqa: E402
from operon_langgraph_gates._common import EPHEMERAL_THREAD  # noqa: E402


class _LoopingModel(BaseChatModel):
    """Always proposes the same failing tool call — the #6731 pathology."""

    n: int = 0

    @property
    def _llm_type(self) -> str:
        return "looping-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        # The call is hardcoded; we intentionally ignore the bound tools.
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.n += 1
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "query_databricks",
                    "args": {"sql": "SELECT * FROM sales"},
                    "id": f"call_{self.n}",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


@tool
def query_databricks(sql: str) -> str:
    """Run a SQL query against Databricks."""
    return "[REQUIRES_SINGLE_PART_NAMESPACE] Catalog/schema not allowed here."


def _user_input() -> dict[str, list[dict[str, str]]]:
    return {"messages": [{"role": "user", "content": "get the sales rows"}]}


def test_baseline_create_agent_loops_to_recursion_limit() -> None:
    """Without the middleware, the agent loops until GraphRecursionError."""
    agent = create_agent(model=_LoopingModel(), tools=[query_databricks])
    with pytest.raises(GraphRecursionError):
        agent.invoke(_user_input(), {"recursion_limit": 8})


def test_middleware_breaks_loop_and_emits_certificate() -> None:
    """With the middleware, the loop ends gracefully and one cert is emitted."""
    guard = StagnationMiddleware(threshold=0.2, critical_duration=2, window_size=3)
    agent = create_agent(model=_LoopingModel(), tools=[query_databricks], middleware=[guard])

    result = agent.invoke(_user_input(), {"recursion_limit": 25})

    # Terminated without a recursion error, with the stop message last.
    assert "stagnat" in result["messages"][-1].content.lower()

    # Exactly one behavioral_stability_windowed certificate, and it does not hold.
    certs = guard.certificates
    assert len(certs) == 1
    assert certs[0].theorem == "behavioral_stability_windowed"
    assert certs[0].verify().holds is False


def test_middleware_certificate_matches_gate_path() -> None:
    """The middleware emits the same theorem the StateGraph gate exposes."""
    from operon_langgraph_gates import STAGNATION_THEOREM

    guard = StagnationMiddleware(threshold=0.2, critical_duration=2, window_size=3)
    agent = create_agent(model=_LoopingModel(), tools=[query_databricks], middleware=[guard])
    agent.invoke(_user_input(), {"recursion_limit": 25})

    assert guard.certificates[0].theorem == STAGNATION_THEOREM


def test_observations_are_scoped_per_thread_not_leaked() -> None:
    """Regression for roborev #1258: the middleware ``Runtime`` exposes the
    thread id at ``runtime.execution_info.thread_id`` (it has no ``.config``).
    Observations must land under the real ``thread_id`` — not collapse into the
    shared ephemeral bucket — so state does not leak across agent threads/runs.
    """
    guard = StagnationMiddleware(threshold=0.2, critical_duration=2, window_size=3)
    agent = create_agent(
        model=_LoopingModel(),
        tools=[query_databricks],
        middleware=[guard],
        checkpointer=InMemorySaver(),
    )

    cfg_a = {"recursion_limit": 25, "configurable": {"thread_id": "thread-a"}}
    cfg_b = {"recursion_limit": 25, "configurable": {"thread_id": "thread-b"}}
    agent.invoke(_user_input(), cfg_a)
    agent.invoke(_user_input(), cfg_b)

    gate = guard.gate
    # Each thread has its own observation history...
    assert gate.integrals_for("thread-a"), "thread-a must have its own history"
    assert gate.integrals_for("thread-b"), "thread-b must have its own history"
    # ...and is independently flagged stagnant (state is not shared).
    assert gate.is_stagnant_for("thread-a") is True
    assert gate.is_stagnant_for("thread-b") is True
    # The ephemeral bucket stays empty: nothing leaked into it (this is exactly
    # what failed before the fix, when every observation collapsed there).
    assert gate.integrals_for(EPHEMERAL_THREAD) == []
