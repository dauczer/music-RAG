import json


def build_chunks(artist_name: str) -> list[dict]:
    lyrics_path = f"data/raw/{artist_name}_lyrics.json"

    with open(lyrics_path, "r", encoding="utf-8") as f:
        lyrics_data: list[dict] = json.load(f)

    chunks: list[dict] = []
    skipped = 0

    for song in lyrics_data:
        title = song.get("title", "Unknown")
        album = song.get("album", "Unknown")
        year = song.get("year", "Unknown")
        lyrics = (song.get("lyrics") or "").strip()

        if not lyrics:
            skipped += 1
            continue

        chunk_text = "\n".join([
            f"Artist: {artist_name}",
            f"Title: {title}",
            f"Album: {album}",
            f"Year: {year}",
            "Lyrics:",
            lyrics[:3000],
        ])

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "title": title,
                "album": album,
                "year": year,
                "artist": artist_name,
            },
        })

    print(f"Built {len(chunks)} chunks ({skipped} songs skipped — no lyrics)")
    return chunks


if __name__ == "__main__":
    chunks = build_chunks("Damso")
    for c in chunks[:3]:
        print(c["text"])
        print("---")
