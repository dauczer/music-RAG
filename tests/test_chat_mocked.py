"""Baseline test: /chat happy path with Groq monkeypatched."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    with (
        patch("rag.chain._call_groq", return_value="Damso parle souvent de la solitude."),
        patch(
            "rag.chain.retrieve",
            return_value=["chunk1", "chunk2"],
        ),
    ):
        from api.main import app

        yield TestClient(app)


def test_chat_returns_answer(client):
    resp = client.post("/chat", json={"artist": "Damso", "question": "quels sont ses thèmes?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
