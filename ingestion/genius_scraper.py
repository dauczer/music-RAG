import json
import os
import random
import re
import time

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://api.genius.com"


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
                print(f"Collecting songs for {artist_name}: Artist ID {artist['id']}")
                return artist["id"]
    print(f"Artist '{artist_name}' not found on Genius.")
    return None


def _get_song_stubs(token: str, artist_id: int, max_songs: int) -> list[dict]:
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
                all_stubs.append({
                    "id": song["id"],
                    "title": song["title"],
                    "path": song["path"],
                })

        if data["next_page"] is None:
            break
        page += 1

    print(f"{len(all_stubs)} primary songs found in full catalog")
    random.shuffle(all_stubs)
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
    # This covers both: "N ContributorsTranslationsEnglishTitle LyricsOptional description Read More [Paroles de ...]"
    text = re.sub(r"^.*?\[Paroles[^\]]*\]\n?", "", text, flags=re.DOTALL)
    return text.strip()


def scrape_artist(artist_name: str, max_songs: int = 50) -> None:
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        raise ValueError("GENIUS_ACCESS_TOKEN not set in .env")

    print(f"Scraping {artist_name} (up to {max_songs} songs)...")

    try:
        artist_id = _find_artist_id(token, artist_name)
    except Exception as e:
        print(f"Error finding artist: {e}")
        return

    if not artist_id:
        return

    try:
        stubs = _get_song_stubs(token, artist_id, max_songs)
    except Exception as e:
        print(f"Error fetching song list: {e}")
        return

    songs = []
    for i, stub in enumerate(stubs, 1):
        title = stub["title"]
        print(f"  [{i}/{len(stubs)}] {title}")

        try:
            meta = _get_song_meta(token, stub["id"])
        except Exception:
            meta = {"album": "Unknown", "year": "Unknown"}

        try:
            lyrics = _scrape_lyrics(stub["path"])
        except Exception as e:
            print(f"    Error scraping lyrics: {e}")
            lyrics = ""

        # Skip Genius-marked snippets/unofficial leaks (title ends with *)
        if title.endswith("*"):
            print(f"    Skipping unofficial snippet (title ends with *)")
            continue
        if not lyrics:
            print(f"    Warning: no lyrics found, skipping")
            continue
        if len(lyrics) < MIN_LYRICS_CHARS:
            print(f"    Warning: lyrics too short ({len(lyrics)} chars), skipping")
            continue

        songs.append({
            "title": title,
            "album": meta["album"],
            "year": meta["year"],
            "lyrics": lyrics,
        })
        time.sleep(0.5)

    output_path = f"data/raw/{artist_name}_lyrics.json"
    os.makedirs("data/raw", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(songs, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(songs)} songs to {output_path}")


if __name__ == "__main__":
    scrape_artist("Damso", max_songs=200)
