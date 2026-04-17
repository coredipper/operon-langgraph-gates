---
title: Operon StagnationGate Demo
emoji: 🔁
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: "6.12.0"
app_file: app.py
pinned: false
license: mit
---

# 🔁 StagnationGate — Interactive Demo

LangGraph [issue #6731](https://github.com/langchain-ai/langgraph/issues/6731) — *agent infinite-loops until recursion limit, burns tokens invisibly* — was closed as `NOT_PLANNED`. LangChain's answer: *"use tool-call limits in middleware."*

This Space demonstrates the missing native gate. [`operon-langgraph-gates`](https://github.com/coredipper/operon-langgraph-gates) ships a drop-in `StagnationGate` that watches a node's output for repetition, flips a routing flag on a conditional edge, and emits a replayable certificate with the evidence that fired it — all as a ~10-line diff on an existing `StateGraph`.

Pick a preset (identical outputs, diverse outputs, near-identical with noise, slow drift), tune threshold and window_size, and watch the gate react turn by turn. No LLM calls — every trajectory is deterministic text, so the demo is free to run and reproducible.

Backed by [Paper 4 §4.3](https://github.com/coredipper/operon/blob/main/article/paper4/main.pdf): convergence / false-stagnation accuracy **0.960** with real sentence embeddings (all-MiniLM-L6-v2). See [`docs/paper-citations.md`](https://github.com/coredipper/operon-langgraph-gates/blob/main/docs/paper-citations.md) for the full record and the loop-detection caveat.
