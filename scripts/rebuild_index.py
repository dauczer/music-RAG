"""Build a fresh Chroma index without touching the currently deployed one.

Usage:
    python -m scripts.rebuild_index
    python -m scripts.rebuild_index --output /tmp/music-rag-chroma --resume

The command downloads the embedding model weights when needed, then computes all
lyrics embeddings locally. Corpus text is not sent to an inference API. After
evaluation, point CHROMA_DB_PATH at the new directory or replace the old index
in a separate, explicit deployment step.
"""

import argparse
import gzip
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from ingestion.build_chunks import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS, build_chunks
from rag.vectorstore import _chunk_id, create_client, index_artist

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw"
_CURRENT_INDEX = _PROJECT_ROOT / "chroma_db"


def _artist_names() -> list[str]:
    suffix = "_lyrics.json"
    return sorted(
        path.name.removesuffix(suffix)
        for path in _RAW_DATA_DIR.glob(f"*{suffix}")
        if path.is_file()
    )


def rebuild(output: Path, *, resume: bool = False) -> None:
    output = output.resolve()
    if output == _CURRENT_INDEX.resolve():
        raise ValueError("Refusing to rebuild in-place. Choose a separate output directory.")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError(f"{output} is not empty. Use --resume or choose another directory.")

    artists = _artist_names()
    if not artists:
        raise ValueError(f"No scraped lyrics files found in {_RAW_DATA_DIR}.")

    output.mkdir(parents=True, exist_ok=True)
    client = create_client(output)
    indexed = 0
    skipped = 0
    passages: dict[str, str] = {}

    for position, artist in enumerate(artists, 1):
        logger.info("[%d/%d] Rebuilding %s", position, len(artists), artist)
        chunks = build_chunks(artist)
        passages.update({_chunk_id(artist, chunk["metadata"]): chunk["text"] for chunk in chunks})
        if index_artist(
            artist,
            client=client,
            chunks=chunks,
            store_documents=False,
        ):
            indexed += 1
        else:
            skipped += 1

    with gzip.open(output / "passages.json.gz", "wt", encoding="utf-8", compresslevel=9) as file:
        json.dump(passages, file, ensure_ascii=False, separators=(",", ":"))

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "artist_count": len(artists),
        "indexed": indexed,
        "skipped": skipped,
        "embedding_model": "intfloat/multilingual-e5-small",
        "chunk_words": DEFAULT_CHUNK_WORDS,
        "overlap_words": DEFAULT_OVERLAP_WORDS,
        "passage_count": len(passages),
        "documents_stored_outside_chroma": True,
    }
    (output / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Fresh index ready at %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "chroma_db_v2",
        help="Destination for the fresh index (default: chroma_db_v2)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep already indexed collections in a partial destination",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    rebuild(args.output, resume=args.resume)


if __name__ == "__main__":
    main()
