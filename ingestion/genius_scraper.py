import json
import logging
import os
import random
import re
import time

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = "https://api.genius.com"


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _find_artist_id(token: str, artist_name: str) -> int | None:
    resp = requests.get(
        f"{BASE_URL}/search",
        params={"q": artist_name},
        headers=_auth_headers(token),
        timeout=10,
    )
    resp.raise_for_status()
    hits = resp.json()["response"]["hits"]
    for hit in hits:
        if hit["type"] == "song":
            artist = hit["result"]["primary_artist"]
            if artist["name"].lower() == artist_name.lower():
                logger.info("Collecting songs for %s: Artist ID %s", artist_name, artist["id"])
                return artist["id"]
    logger.warning("Artist '%s' not found on Genius.", artist_name)
    return None


def _get_song_stubs(
    token: str, artist_id: int, max_songs: int, artist_name: str = ""
) -> list[dict]:
    """Fetch ALL primary songs for the artist, shuffle, then return up to max_songs."""
    all_stubs = []
    page = 1
    while True:
        resp = requests.get(
            f"{BASE_URL}/artists/{artist_id}/songs",
            params={"per_page": 50, "page": page},
            headers=_auth_headers(token),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()["response"]

        for song in data["songs"]:
            # Only songs where this artist is the primary artist (skip features)
            if song["primary_artist"]["id"] == artist_id:
                all_stubs.append(
                    {
                        "id": song["id"],
                        "title": song["title"],
                        "path": song["path"],
                    }
                )

        if data["next_page"] is None:
            break
        page += 1

    logger.info("%d primary songs found in full catalog", len(all_stubs))
    random.Random(artist_name).shuffle(all_stubs)
    return all_stubs[:max_songs]


def _get_song_meta(token: str, song_id: int) -> dict:
    resp = requests.get(
        f"{BASE_URL}/songs/{song_id}",
        headers=_auth_headers(token),
        timeout=10,
    )
    resp.raise_for_status()
    song = resp.json()["response"]["song"]
    album = song.get("album") or {}
    components = song.get("release_date_components") or {}
    year = str(components["year"]) if components.get("year") else "Unknown"
    return {
        "album": album.get("name", "Unknown"),
        "year": year,
    }


def _scrape_lyrics(path: str) -> str:
    """Scrape lyrics from a Genius song page (adapted from elliebirbeck/genius-lyrics-scraper)."""
    page = requests.get("https://genius.com" + path, timeout=15)
    if not page.ok:
        return ""

    html = BeautifulSoup(page.text, "html.parser")
    # Remove script tags before parsing (as in the reference repo)
    for script in html("script"):
        script.extract()

    # Try old layout (div.lyrics), then new layout (data-lyrics-container)
    old = html.find("div", class_="lyrics")
    if old:
        text = old.get_text()
    else:
        containers = html.find_all("div", attrs={"data-lyrics-container": "true"})
        if not containers:
            return ""
        for br in html.find_all("br"):
            br.replace_with("\n")
        text = "\n".join(c.get_text() for c in containers)

    if "Lyrics will be available" in text:
        return ""

    return _clean_lyrics(text)


MIN_LYRICS_CHARS = 600


def _clean_lyrics(text: str) -> str:
    """Strip Genius page noise: contributor header, editorial descriptions, 'Paroles de' marker."""
    # Remove everything up to and including [Paroles de "..."] or [Paroles issues d'un extrait]
    # Covers noise like: "N ContributorsTranslationsEnglish... Read More [Paroles de ...]"
    text = re.sub(r"^.*?\[Paroles[^\]]*\]\n?", "", text, flags=re.DOTALL)
    return text.strip()


def scrape_artist(artist_name: str, max_songs: int = 50) -> None:
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        raise ValueError("GENIUS_ACCESS_TOKEN not set in .env")

    logger.info("Scraping %s (up to %d songs)...", artist_name, max_songs)

    try:
        artist_id = _find_artist_id(token, artist_name)
    except requests.RequestException as e:
        logger.exception("Error finding artist %s: %s", artist_name, e)
        return

    if not artist_id:
        return

    try:
        stubs = _get_song_stubs(token, artist_id, max_songs, artist_name)
    except requests.RequestException as e:
        logger.exception("Error fetching song list for %s: %s", artist_name, e)
        return

    songs = []
    for i, stub in enumerate(stubs, 1):
        title = stub["title"]
        logger.info("  [%d/%d] %s", i, len(stubs), title)

        try:
            meta = _get_song_meta(token, stub["id"])
        except requests.RequestException:
            meta = {"album": "Unknown", "year": "Unknown"}

        try:
            lyrics = _scrape_lyrics(stub["path"])
        except requests.RequestException as e:
            logger.warning("    Error scraping lyrics for %s: %s", title, e)
            lyrics = ""

        # Skip Genius-marked snippets/unofficial leaks (title ends with *)
        if title.endswith("*"):
            logger.debug("    Skipping unofficial snippet: %s", title)
            continue
        if not lyrics:
            logger.debug("    No lyrics found for %s, skipping", title)
            continue
        if len(lyrics) < MIN_LYRICS_CHARS:
            logger.debug("    Lyrics too short (%d chars) for %s, skipping", len(lyrics), title)
            continue

        songs.append(
            {
                "title": title,
                "album": meta["album"],
                "year": meta["year"],
                "lyrics": lyrics,
            }
        )
        time.sleep(0.5)

    output_path = f"data/raw/{artist_name}_lyrics.json"
    os.makedirs("data/raw", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(songs, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d songs to %s", len(songs), output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    scrape_artist("Damso", max_songs=200)
