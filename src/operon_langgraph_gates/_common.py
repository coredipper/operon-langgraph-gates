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

# Kwarg names LangGraph is known to use when injecting a RunnableConfig
# dict or a Runtime object into a node. Only these names are trusted for
# dict-shaped extraction, so a business kwarg whose dict happens to have
# a ``configurable.thread_id`` key doesn't hijack thread routing.
_LANGGRAPH_KWARG_NAMES: tuple[str, ...] = ("config", "runtime")


def thread_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Extract ``thread_id`` from a LangGraph config/runtime passed to a node.

    Resolution order, most-trusted first:

    1. Positional args (LangGraph invocation protocol: ``state`` plus an
       optional ``config`` dict or ``runtime`` object).
    2. Known LangGraph kwarg names — ``config``, ``runtime``. Any shape
       (dict or Runtime-like) is accepted.
    3. Other kwargs — only *Runtime-like* objects (non-dict with a
       ``.config`` attribute). Plain dicts here are treated as business
       payloads and **not** interpreted as RunnableConfig, even if their
       shape happens to match.

    Falls back to :data:`EPHEMERAL_THREAD` when no thread id is present.
    """
    # (1) positional — LangGraph protocol slots.
    for c in args:
        tid = _extract_thread_id(c)
        if tid is not None:
            return tid

    # (2) explicit LangGraph kwarg names — trust any shape.
    for name in _LANGGRAPH_KWARG_NAMES:
        if name in kwargs:
            tid = _extract_thread_id(kwargs[name])
            if tid is not None:
                return tid

    # (3) remaining kwargs — Runtime-like (non-dict with ``.config``) only.
    for name, value in kwargs.items():
        if name in _LANGGRAPH_KWARG_NAMES or isinstance(value, dict):
            continue
        tid = _extract_thread_id(value)
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
