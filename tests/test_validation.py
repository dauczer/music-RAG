"""Tests for request model input validation (step 2)."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from api.main import app
    from rag.chain import ArtistRef, RagResult

    result = RagResult(
        status="answered",
        mode="single",
        answer="stub",
        artists=[ArtistRef(slug="damso", name="Damso")],
        sources=[],
    )
    with (
        patch("api.main.ask_result", MagicMock(return_value=result)),
        patch("api.main.compare_result", MagicMock(return_value=result)),
        patch("api.main.route_and_ask_result", MagicMock(return_value=result)),
    ):
        yield TestClient(app)


# --- /chat ---


def test_chat_empty_artist_rejected(client):
    resp = client.post("/chat", json={"artist": "", "question": "test"})
    assert resp.status_code == 422


def test_chat_oversize_question_rejected(client):
    resp = client.post("/chat", json={"artist": "Damso", "question": "x" * 501})
    assert resp.status_code == 422


def test_chat_valid_passes(client):
    resp = client.post("/chat", json={"artist": "Damso", "question": "quels thèmes?"})
    assert resp.status_code == 200


# --- /ask ---


def test_ask_empty_question_rejected(client):
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_oversize_question_rejected(client):
    resp = client.post("/ask", json={"question": "q" * 501})
    assert resp.status_code == 422


# --- /compare ---


def test_compare_empty_artist1_rejected(client):
    resp = client.post("/compare", json={"artist1": "", "artist2": "Nekfeu", "question": "compare"})
    assert resp.status_code == 422


def test_compare_oversize_artist2_rejected(client):
    resp = client.post(
        "/compare",
        json={"artist1": "Damso", "artist2": "x" * 501, "question": "compare"},
    )
    assert resp.status_code == 422
