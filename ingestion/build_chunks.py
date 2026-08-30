import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DEFAULT_CHUNK_WORDS = 160
DEFAULT_OVERLAP_WORDS = 30


def _split_words(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Split lyrics into bounded, overlapping word windows.

    multilingual-e5-small accepts up to 512 tokens. A 160-word window leaves
    comfortable room for the required passage prefix, the metadata header, and
    words that split into multiple wordpieces. A small overlap preserves context
    across boundaries.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than zero")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be between 0 and chunk_words - 1")

    words = text.split()
    if not words:
        return []

    step = chunk_words - overlap_words
    return [" ".join(words[start : start + chunk_words]) for start in range(0, len(words), step)]


def build_chunks_from_songs(
    artist_name: str,
    lyrics_data: list[dict[str, Any]],
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    """Build embedding-sized chunks from already loaded Genius song data."""
    chunks: list[dict[str, Any]] = []
    skipped = 0

    for song_index, song in enumerate(lyrics_data):
        title = str(song.get("title") or "Unknown")
        album = str(song.get("album") or "Unknown")
        year = str(song.get("year") or "Unknown")
        lyrics = str(song.get("lyrics") or "").strip()

        if not lyrics:
            skipped += 1
            continue

        passages = _split_words(lyrics, chunk_words, overlap_words)
        for chunk_index, passage in enumerate(passages):
            chunk_text = "\n".join(
                [
                    f"Artist: {artist_name}",
                    f"Title: {title}",
                    f"Album: {album}",
                    f"Year: {year}",
                    "Lyrics excerpt:",
                    passage,
                ]
            )
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "title": title,
                        "album": album,
                        "year": year,
                        "artist": artist_name,
                        "song_index": song_index,
                        "chunk_index": chunk_index,
                        "chunk_count": len(passages),
                    },
                }
            )

    logger.info(
        "Built %d chunks from %d songs (%d songs skipped — no lyrics)",
        len(chunks),
        len(lyrics_data),
        skipped,
    )
    return chunks


def build_chunks(artist_name: str) -> list[dict]:
    lyrics_path = _RAW_DATA_DIR / f"{artist_name}_lyrics.json"

    try:
        with open(lyrics_path, encoding="utf-8") as f:
            lyrics_data: list[dict] = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"No scraped data found for {artist_name}. Run the scraper first.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in {lyrics_path}: {e}") from e

    return build_chunks_from_songs(artist_name, lyrics_data)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    chunks = build_chunks("Damso")
    for c in chunks[:3]:
        print(c["text"])
        print("---")
