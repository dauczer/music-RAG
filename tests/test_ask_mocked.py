"""The selector-free endpoint exposes scope, answer, and cited tracks."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from rag.chain import ArtistRef, RagResult, RagSource


def test_ask_returns_unified_global_contract():
    result = RagResult(
        status="answered",
        mode="global",
        answer="Plusieurs artistes abordent la solitude de façons différentes.",
        artists=[],
        sources=[
            RagSource(id="damso:song-0001", artist="Damso", title="Mort"),
            RagSource(id="orelsan:song-0002", artist="Orelsan", title="Notes pour trop tard"),
        ],
    )

    from api.main import app

    with patch("api.main.route_and_ask_result", return_value=result):
        response = TestClient(app).post(
            "/ask",
            json={"question": "Quels artistes parlent de solitude ?"},
        )

    assert response.status_code == 200
    assert response.json()["mode"] == "global"
    assert response.json()["artists"] == []
    assert [source["artist"] for source in response.json()["sources"]] == [
        "Damso",
        "Orelsan",
    ]
    assert response.json()["request_id"]


def test_ask_returns_detected_artist_scope():
    result = RagResult(
        status="answered",
        mode="single",
        answer="Réponse ciblée.",
        artists=[ArtistRef(slug="damso", name="Damso")],
        sources=[RagSource(id="damso:song-0001", artist="Damso", title="Mort")],
    )

    from api.main import app

    with patch("api.main.route_and_ask_result", return_value=result):
        response = TestClient(app).post(
            "/ask",
            json={"question": "Comment Damso parle-t-il de solitude ?"},
        )

    assert response.status_code == 200
    assert response.json()["mode"] == "single"
    assert response.json()["artists"] == [{"slug": "damso", "name": "Damso"}]
