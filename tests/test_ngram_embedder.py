"""Tests for the zero-dependency character-n-gram embedder.

This is the default embedder for ``StagnationGate`` so users can adopt the
gate without installing a neural embedding model.
"""

from __future__ import annotations

import math

from operon_langgraph_gates.embedders import NGramEmbedder, cosine


def test_embed_returns_flat_list_of_floats() -> None:
    embedder = NGramEmbedder()
    vec = embedder.embed("hello world")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(x, float) for x in vec)


def test_embed_is_deterministic() -> None:
    embedder = NGramEmbedder()
    assert embedder.embed("the quick brown fox") == embedder.embed("the quick brown fox")


def test_embed_differs_for_different_text() -> None:
    embedder = NGramEmbedder()
    assert embedder.embed("hello") != embedder.embed("goodbye")


def test_embed_is_normalized() -> None:
    # Normalized vector => identical text gives cosine == 1.0 via dot product.
    embedder = NGramEmbedder()
    vec = embedder.embed("the quick brown fox")
    norm = math.sqrt(sum(x * x for x in vec))
    # Allow tiny float drift.
    assert abs(norm - 1.0) < 1e-9


def test_cosine_identical_text_is_one() -> None:
    embedder = NGramEmbedder()
    a = embedder.embed("the quick brown fox jumps")
    b = embedder.embed("the quick brown fox jumps")
    assert abs(cosine(a, b) - 1.0) < 1e-9


def test_cosine_unrelated_text_is_low() -> None:
    # "abcdef" and "xyzwvu" share no 3-grams -> cosine should be 0.0.
    embedder = NGramEmbedder()
    a = embedder.embed("abcdef")
    b = embedder.embed("xyzwvu")
    assert cosine(a, b) < 0.2


def test_cosine_similar_text_is_high() -> None:
    # Identical-ish text with a single-char edit should stay near 1.0.
    embedder = NGramEmbedder()
    a = embedder.embed("the quick brown fox")
    b = embedder.embed("the quick brown dox")  # one-char edit
    assert cosine(a, b) > 0.7


def test_empty_text_embeds_without_crashing() -> None:
    embedder = NGramEmbedder()
    vec = embedder.embed("")
    # Empty text has no n-grams; returns a zero vector rather than NaN.
    assert all(x == 0.0 for x in vec)
