"""Tests for CORS middleware configuration."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_client(allowed_origins: str = "http://localhost:5173") -> TestClient:
    """Reload the app module with a specific ALLOWED_ORIGINS env value."""
    import importlib

    import api.main as main_module

    with patch.dict(os.environ, {"ALLOWED_ORIGINS": allowed_origins}):
        importlib.reload(main_module)
        return TestClient(main_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Patch out the heavy RAG imports so tests don't need ChromaDB / Groq loaded
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _patch_rag(monkeypatch):
    monkeypatch.setattr("rag.chain.ask", MagicMock(return_value="stub"))
    monkeypatch.setattr("rag.chain.compare_artists", MagicMock(return_value="stub"))
    monkeypatch.setattr("rag.chain.route_and_ask", MagicMock(return_value="stub"))
    monkeypatch.setattr("rag.vectorstore.index_artist", MagicMock(return_value=True))


def test_preflight_allowed_origin():
    """A listed origin should receive the Access-Control-Allow-Origin header."""
    client = _make_client("http://localhost:5173")
    resp = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_preflight_unknown_origin():
    """An origin not in the allowlist must NOT receive the Allow-Origin header."""
    client = _make_client("http://localhost:5173")
    resp = client.options(
        "/health",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in resp.headers
