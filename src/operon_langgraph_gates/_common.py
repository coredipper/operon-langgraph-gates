"""Shared internals for both gates.

Keep this tiny — only things that both ``stagnation`` and ``integrity``
genuinely need. Any growth beyond a handful of helpers should trigger a
rethink of the abstraction rather than accretion here.
"""

from __future__ import annotations

from typing import Any

EPHEMERAL_THREAD = "__ephemeral__"


def thread_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Extract ``thread_id`` from a LangGraph config/runtime passed to a node.

    LangGraph passes a ``RunnableConfig`` as the node's second positional
    argument (after ``state``) or as the ``config`` keyword argument. Its
    ``configurable`` inner dict holds ``thread_id``. Fall back to
    :data:`EPHEMERAL_THREAD` when no thread id is present — that keeps
    one-shot calls (tests, simple scripts without persistence) working
    without forcing a thread id.
    """
    candidates: list[Any] = list(args)
    cfg = kwargs.get("config")
    if cfg is not None:
        candidates.append(cfg)
    for c in candidates:
        if isinstance(c, dict):
            configurable = c.get("configurable")
            if isinstance(configurable, dict):
                tid = configurable.get("thread_id")
                if tid is not None:
                    return str(tid)
    return EPHEMERAL_THREAD
