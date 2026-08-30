from unittest.mock import patch

from fastapi.testclient import TestClient

from rag.vectorstore import ArtistSummary


def test_artists_endpoint_returns_only_vectorstore_summaries():
    from api.main import app

    summaries = [
        ArtistSummary(slug="damso", name="Damso", song_count=156, chunk_count=521),
        ArtistSummary(slug="nekfeu", name="Nekfeu", song_count=142, chunk_count=477),
    ]
    with patch("api.main.list_artist_summaries", return_value=summaries):
        response = TestClient(app).get("/artists")

    assert response.status_code == 200
    assert response.json() == {
        "artists": [
            {"slug": "damso", "name": "Damso", "song_count": 156},
            {"slug": "nekfeu", "name": "Nekfeu", "song_count": 142},
        ]
    }
