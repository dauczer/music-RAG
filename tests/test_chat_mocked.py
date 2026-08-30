"""The portfolio chat endpoint returns its complete public contract."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from api.main import app
    from rag.chain import ArtistRef, RagResult, RagSource

    result = RagResult(
        status="answered",
        answer="Damso parle souvent de la solitude.",
        artist=ArtistRef(slug="damso", name="Damso"),
        sources=[RagSource(id="damso:song-0001", artist="Damso", title="Mort")],
    )
    with patch("api.main.ask_result", return_value=result):
        yield TestClient(app)


def test_chat_returns_answer(client):
    resp = client.post("/chat", json={"artist": "Damso", "question": "quels sont ses thèmes?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "answered"
    assert data["artist"] == {"slug": "damso", "name": "Damso"}
    assert data["sources"][0]["title"] == "Mort"
    assert data["request_id"]
