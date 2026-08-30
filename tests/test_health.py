"""Baseline test: /health endpoint returns 200."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    with (
        patch("rag.chain.ask", MagicMock(return_value="stub")),
        patch("rag.chain.compare_artists", MagicMock(return_value="stub")),
        patch("rag.chain.route_and_ask", MagicMock(return_value="stub")),
    ):
        from api.main import app

        return TestClient(app)


def test_health_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
