"""The public demo is limited to three generations per minute and IP."""

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
    with patch("api.main.ask_result", MagicMock(return_value=result)):
        yield TestClient(app, raise_server_exceptions=False)


def test_rate_limit_429_on_fourth_request(client):
    payload = {"artist": "Damso", "question": "quels thèmes?"}
    responses = [client.post("/chat", json=payload) for _ in range(4)]
    status_codes = [r.status_code for r in responses]
    assert status_codes[:3] == [200, 200, 200]
    assert status_codes[3] == 429
