"""
Scrape and index a curated list of French rap artists.
Safe to re-run: skips artists already scraped or indexed.

Usage:
    python -m scripts.bulk_ingest
"""

import logging
from pathlib import Path

from ingestion.genius_scraper import scrape_artist
from rag.vectorstore import index_artist

logger = logging.getLogger(__name__)

ARTISTS = [
    "MC Solaar",
    "IAM",
    "Suprême NTM",
    "Oxmo Puccino",
    "Booba",
    "Rohff",
    "Kaaris",
    "Lacrim",
    "La Fouine",
    "Orelsan",
    "Nekfeu",
    "SCH",
    "Vald",
    "Lomepal",
    "Laylow",
    "PLK",
    "Ninho",
    "Freeze Corleone",
    "Gazo",
    "Hamza",
    "Niska",
    "Alpha Wann",
    "Kekra",
    "Luther",
    "Winterzuko",
]


def _already_scraped(artist: str) -> bool:
    return Path(f"data/raw/{artist}_lyrics.json").exists()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    failed = []

    for i, artist in enumerate(ARTISTS, 1):
        logger.info("[%d/%d] %s", i, len(ARTISTS), artist)
        logger.info("-" * 40)

        try:
            if _already_scraped(artist):
                logger.info("Scraping skipped — data/raw/%s_lyrics.json already exists", artist)
            else:
                scrape_artist(artist, max_songs=200)

            index_artist(artist)

        except Exception as e:
            logger.error("ERROR processing %s: %s", artist, e)
            failed.append((artist, str(e)))
            continue

    logger.info("=" * 40)
    logger.info("Done. %d/%d artists ingested.", len(ARTISTS) - len(failed), len(ARTISTS))
    if failed:
        logger.warning("Failed artists:")
        for artist, err in failed:
            logger.warning("  - %s: %s", artist, err)


if __name__ == "__main__":
    main()
