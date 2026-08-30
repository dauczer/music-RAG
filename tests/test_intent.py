"""Tests for intent routing and Pydantic validation in rag/chain.py."""

import pytest

from rag.chain import _Intent, route_and_ask


def test_intent_valid():
    intent = _Intent.model_validate({"mode": "single", "artists": ["Damso"]})
    assert intent.mode == "single"
    assert intent.artists == ["Damso"]


def test_intent_invalid_mode():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _Intent.model_validate({"mode": "invalid_mode", "artists": []})


def test_intent_unknown_mode():
    intent = _Intent.model_validate({"mode": "unknown", "artists": []})
    assert intent.mode == "unknown"


def test_route_and_ask_malformed_json(monkeypatch):
    """Malformed JSON from Groq falls back to the French fallback message."""
    monkeypatch.setattr("rag.chain._call_groq", lambda *args, **kwargs: "NOT JSON AT ALL !!!")
    monkeypatch.setattr("rag.chain.list_indexed_artists", lambda: ["damso"])
    monkeypatch.setattr("rag.chain.list_artist_summaries", lambda: [])
    result = route_and_ask("Quels sont les thèmes de Damso ?")
    assert "Je n'ai pas compris" in result


def test_route_and_ask_valid_single(monkeypatch):
    """Valid single-mode intent routes to ask() and returns its result."""
    import json

    monkeypatch.setattr(
        "rag.chain._call_groq",
        lambda *args, **kwargs: json.dumps({"mode": "single", "artists": ["damso"]}),
    )
    monkeypatch.setattr("rag.chain.list_indexed_artists", lambda: ["damso"])
    monkeypatch.setattr("rag.chain.list_artist_summaries", lambda: [])
    monkeypatch.setattr("rag.chain._find_indexed_artist", lambda name: "damso")
    monkeypatch.setattr("rag.chain.ask", lambda artist, question: "stub answer")
    result = route_and_ask("Quels sont les thèmes de Damso ?")
    assert result == "stub answer"
