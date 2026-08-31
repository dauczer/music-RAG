from rag import vectorstore


class _Collection:
    def count(self):
        return 1

    def get(self, **kwargs):
        return {"ids": ["damso:song-0001:chunk-000"], "metadatas": [{"title": "Amnésie"}]}

    def query(self, **kwargs):
        return {
            "ids": [["damso:song-0001:chunk-000"]],
            "documents": None,
            "metadatas": [
                [
                    {
                        "artist": "Damso",
                        "title": "Amnésie",
                        "album": "Unknown",
                        "year": "Unknown",
                        "chunk_index": 0,
                    }
                ]
            ],
            "distances": [[0.12]],
        }


class _Client:
    def get_collection(self, name):
        assert name == "damso"
        return _Collection()


def test_retrieval_loads_document_from_compressed_passage_store(monkeypatch):
    monkeypatch.setattr(vectorstore, "_get_client", lambda: _Client())
    monkeypatch.setattr(vectorstore, "_embed_queries", lambda texts: [[0.0, 1.0]])
    monkeypatch.setattr(
        vectorstore,
        "_load_passage_store",
        lambda: {"damso:song-0001:chunk-000": "stored lyrics passage"},
    )

    chunks = vectorstore.retrieve_chunks("Damso", "question", n_results=5)

    assert chunks[0].document == "stored lyrics passage"
    assert chunks[0].album is None
    assert chunks[0].source_id == "damso:song-0001"


def test_explicit_song_title_is_detected_with_diacritics():
    assert (
        vectorstore._matching_title(
            _Collection(),
            "Dans Amnesie, comment Damso parle-t-il de la mémoire ?",
        )
        == "Amnésie"
    )


def test_retrieve_across_artists_embeds_shared_query_once(monkeypatch):
    calls = []

    monkeypatch.setattr(
        vectorstore,
        "_embed_queries",
        lambda texts: calls.append(texts) or [[0.1, 0.2]],
    )
    monkeypatch.setattr(
        vectorstore,
        "retrieve_chunks",
        lambda artist, query, n_results, query_embedding, match_title: [
            (artist, query, n_results, query_embedding)
        ],
    )

    results = vectorstore.retrieve_across_artists(["booba", "orelsan"], "la réussite")

    assert calls == [["la réussite"]]
    assert [result[0][0] for result in results] == ["booba", "orelsan"]
    assert all(result[0][3] == [0.1, 0.2] for result in results)


def _chunk(artist: str, song: int, distance: float) -> vectorstore.RetrievedChunk:
    slug = artist.casefold()
    return vectorstore.RetrievedChunk(
        id=f"{slug}:song-{song:04d}:chunk-000",
        document="lyrics",
        artist=artist,
        title=f"Song {song}",
        album=None,
        year=None,
        chunk_index=0,
        distance=distance,
    )


def test_global_retrieval_merges_by_distance_with_artist_diversity(monkeypatch):
    monkeypatch.setattr(vectorstore, "list_indexed_artists", lambda: ["A", "B", "C"])
    monkeypatch.setattr(
        vectorstore,
        "retrieve_across_artists",
        lambda artists, query, n_results, match_titles: [
            [_chunk("A", 1, 0.10), _chunk("A", 2, 0.11)],
            [_chunk("B", 1, 0.12), _chunk("B", 2, 0.13)],
            [_chunk("C", 1, 0.14), _chunk("C", 2, 0.15)],
        ],
    )

    chunks = vectorstore.retrieve_global_chunks(
        "question",
        n_results=4,
        max_chunks_per_artist=1,
    )

    assert [(chunk.artist, chunk.title) for chunk in chunks] == [
        ("A", "Song 1"),
        ("B", "Song 1"),
        ("C", "Song 1"),
    ]
