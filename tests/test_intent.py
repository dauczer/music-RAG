"""Tests for deterministic artist routing in rag/chain.py."""

from rag.chain import ArtistRef, RagResult, _detect_artists, route_and_ask_result
from rag.vectorstore import ArtistSummary


def _artist(slug: str, name: str) -> ArtistSummary:
    return ArtistSummary(slug=slug, name=name, song_count=1, chunk_count=1)


ARTISTS = [
    _artist("booba", "Booba"),
    _artist("damso", "Damso"),
    _artist("mc_solaar", "MC Solaar"),
    _artist("orelsan", "Orelsan"),
]


def test_detects_one_artist_without_llm():
    detected = _detect_artists("Quels sont les thèmes de Damso ?", ARTISTS)
    assert [artist.slug for artist in detected] == ["damso"]


def test_detects_comparison_in_mention_order():
    detected = _detect_artists("Compare Orelsan et Damso sur la famille", ARTISTS)
    assert [artist.slug for artist in detected] == ["orelsan", "damso"]


def test_detects_curated_alias_and_clear_typo():
    assert _detect_artists("Que raconte Solaar ?", ARTISTS)[0].slug == "mc_solaar"
    assert _detect_artists("Que raconte Boba ?", ARTISTS)[0].slug == "booba"


def test_no_artist_means_global_scope():
    assert _detect_artists("Quels morceaux parlent de solitude ?", ARTISTS) == []


def test_route_without_artist_dispatches_global(monkeypatch):
    expected = RagResult(
        status="answered",
        mode="global",
        answer="stub answer",
        artists=[],
        sources=[],
    )
    monkeypatch.setattr("rag.chain.list_artist_summaries", lambda: ARTISTS)
    monkeypatch.setattr("rag.chain.global_result", lambda question: expected)

    result = route_and_ask_result("Quels morceaux parlent de solitude ?")

    assert result is expected


def test_route_with_artist_dispatches_single(monkeypatch):
    expected = RagResult(
        status="answered",
        mode="single",
        answer="stub answer",
        artists=[ArtistRef(slug="damso", name="Damso")],
        sources=[],
    )
    monkeypatch.setattr("rag.chain.list_artist_summaries", lambda: ARTISTS)
    monkeypatch.setattr("rag.chain.ask_result", lambda artist, question: expected)

    result = route_and_ask_result("Quels sont les thèmes de Damso ?")

    assert result is expected
