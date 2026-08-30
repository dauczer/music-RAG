from ingestion.build_chunks import build_chunks_from_songs


def test_long_song_is_split_with_overlap():
    words = [f"word{index}" for index in range(350)]
    chunks = build_chunks_from_songs(
        "Test Artist",
        [
            {
                "title": "Long Song",
                "album": "Album",
                "year": "2026",
                "lyrics": " ".join(words),
            }
        ],
        chunk_words=160,
        overlap_words=30,
    )

    assert len(chunks) == 3
    assert chunks[0]["metadata"]["chunk_count"] == 3
    assert chunks[1]["metadata"]["chunk_index"] == 1
    assert "word130" in chunks[0]["text"]
    assert "word130" in chunks[1]["text"]


def test_empty_song_is_skipped():
    chunks = build_chunks_from_songs(
        "Test Artist",
        [{"title": "Silence", "lyrics": "  "}],
    )
    assert chunks == []


def test_invalid_overlap_is_rejected():
    try:
        build_chunks_from_songs(
            "Test Artist",
            [{"title": "Song", "lyrics": "some words"}],
            chunk_words=10,
            overlap_words=10,
        )
    except ValueError as exc:
        assert "overlap_words" in str(exc)
    else:
        raise AssertionError("Expected invalid overlap to raise ValueError")
