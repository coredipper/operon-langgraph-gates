"""Shared internals for both gates.

Keep this tiny — only things that both ``stagnation`` and ``integrity``
genuinely need. Any growth beyond a handful of helpers should trigger a
rethink of the abstraction rather than accretion here.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

EPHEMERAL_THREAD = "__ephemeral__"


def thread_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Extract ``thread_id`` from a LangGraph config/runtime passed to a node.

    LangGraph passes either:

    - A ``RunnableConfig`` dict as the node's second positional argument or
      as the ``config`` keyword argument. Its ``configurable`` inner dict
      holds ``thread_id``.
    - A ``Runtime`` object (LangGraph 1.x newer nodes) with attribute access
      for ``.config``, which is itself a ``RunnableConfig`` dict. Runtime may
      arrive positionally or as any keyword (common names: ``runtime``,
      ``config``).

    We scan every positional arg and every kwarg value; whichever one yields
    a ``configurable.thread_id`` wins. Fall back to :data:`EPHEMERAL_THREAD`
    when no thread id is present — that keeps one-shot calls (tests, scripts
    without persistence) working without forcing a thread id.
    """
    for c in (*args, *kwargs.values()):
        tid = _extract_thread_id(c)
        if tid is not None:
            return tid
    return EPHEMERAL_THREAD


def _extract_thread_id(value: Any) -> str | None:
    if isinstance(value, dict):
        return _from_config_dict(value)
    # Runtime-like object: unwrap via .config (LangGraph 1.x Runtime).
    inner = getattr(value, "config", None)
    if isinstance(inner, dict):
        return _from_config_dict(inner)
    return None


def _from_config_dict(cfg: dict[str, Any]) -> str | None:
    configurable = cfg.get("configurable")
    if isinstance(configurable, dict):
        tid = configurable.get("thread_id")
        if tid is not None:
            return str(tid)
    return None


def is_async_callable(fn: Callable[..., Any]) -> bool:
    """Detect async callables including callable classes with ``async __call__``."""
    if inspect.iscoroutinefunction(fn):
        return True
    # Callable instance of a class with ``async def __call__``: inspect the
    # class-level attribute so ``iscoroutinefunction`` sees the unbound method.
    if callable(fn):
        return inspect.iscoroutinefunction(type(fn).__call__)
    return False
