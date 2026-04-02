"""
Scrape and index a curated list of French rap artists.
Safe to re-run: skips artists already scraped or indexed.

Usage:
    python -m scripts.bulk_ingest
"""

from pathlib import Path

from ingestion.genius_scraper import scrape_artist
from rag.vectorstore import index_artist

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
    failed = []

    for i, artist in enumerate(ARTISTS, 1):
        print(f"\n[{i}/{len(ARTISTS)}] {artist}")
        print("-" * 40)

        try:
            if _already_scraped(artist):
                print(f"  Scraping skipped — data/raw/{artist}_lyrics.json already exists")
            else:
                scrape_artist(artist, max_songs=200)

            index_artist(artist)

        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append((artist, str(e)))
            continue

    print("\n" + "=" * 40)
    print(f"Done. {len(ARTISTS) - len(failed)}/{len(ARTISTS)} artists ingested.")
    if failed:
        print("Failed:")
        for artist, err in failed:
            print(f"  - {artist}: {err}")


if __name__ == "__main__":
    main()
