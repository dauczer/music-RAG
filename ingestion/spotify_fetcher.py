import json
import os
import re

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()


def _normalize_title(title: str) -> str:
    """Lowercase and strip feat./ft. mentions for fuzzy matching."""
    title = title.lower().strip()
    title = re.sub(r"\s*[\(\[](feat|ft)\.?.*?[\)\]]", "", title)
    title = re.sub(r"\s*(feat|ft)\.?\s+.*", "", title)
    return title.strip()


def fetch_audio_features(artist_name: str) -> None:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET not set in .env")

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    ))

    print(f"Fetching Spotify data for {artist_name}...")

    # Find artist
    try:
        results = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
        artists = results["artists"]["items"]
    except Exception as e:
        print(f"Error searching for artist: {e}")
        return

    if not artists:
        print(f"Artist '{artist_name}' not found on Spotify.")
        return

    artist = artists[0]
    artist_id = artist["id"]
    print(f"Found artist: {artist['name']} (id={artist_id})")

    # Get all albums
    try:
        albums_response = sp.artist_albums(artist_id, album_type="album,single", limit=50)
        albums = albums_response["items"]
    except Exception as e:
        print(f"Error fetching albums: {e}")
        return

    # Deduplicate albums by normalized name (Spotify returns region duplicates)
    seen_albums: set[str] = set()
    unique_albums = []
    for album in albums:
        key = album["name"].lower().strip()
        if key not in seen_albums:
            seen_albums.add(key)
            unique_albums.append(album)

    # Collect all tracks with their album/year context
    all_tracks: list[dict] = []  # {"id": ..., "title": ..., "album": ..., "year": ...}
    for album in unique_albums:
        album_name = album["name"]
        year = album["release_date"][:4] if album.get("release_date") else "Unknown"
        print(f"  Processing album: {album_name} ({year})")
        try:
            tracks_response = sp.album_tracks(album["id"], limit=50)
            for track in tracks_response["items"]:
                all_tracks.append({
                    "id": track["id"],
                    "title": track["name"],
                    "album": album_name,
                    "year": year,
                })
        except Exception as e:
            print(f"  Error fetching tracks for '{album_name}': {e}")

    # Fetch audio features in batches of 50
    songs = []
    for i in range(0, len(all_tracks), 50):
        batch = all_tracks[i:i + 50]
        track_ids = [t["id"] for t in batch]
        try:
            features_list = sp.audio_features(track_ids)
        except Exception as e:
            print(f"  Error fetching audio features (batch {i}): {e}")
            features_list = [None] * len(batch)

        for track, features in zip(batch, features_list or []):
            if not features:
                continue
            songs.append({
                "title": track["title"],
                "album": track["album"],
                "year": track["year"],
                "energy": features.get("energy"),
                "valence": features.get("valence"),
                "tempo": features.get("tempo"),
                "danceability": features.get("danceability"),
            })

    output_path = f"data/raw/{artist_name}_spotify.json"
    os.makedirs("data/raw", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(songs, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(songs)} tracks to {output_path}")


if __name__ == "__main__":
    fetch_audio_features("Damso")
