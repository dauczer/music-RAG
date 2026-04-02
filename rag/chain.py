import json
import os
from difflib import get_close_matches

from dotenv import load_dotenv
from groq import Groq

from rag.vectorstore import _collection_name, list_indexed_artists, retrieve

load_dotenv()

_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise EnvironmentError("GROQ_API_KEY is not set. Check your .env file.")

_client = Groq(api_key=_api_key)
_MODEL = "llama-3.3-70b-versatile"


def _call_groq(prompt: str) -> str:
    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Groq API error: {e}") from e


def _find_indexed_artist(name: str) -> str | None:
    """Fuzzy-match a detected artist name against indexed collection slugs."""
    slug = _collection_name(name)
    indexed = list_indexed_artists()
    matches = get_close_matches(slug, indexed, n=1, cutoff=0.8)
    return matches[0] if matches else None


def ask(artist_name: str, question: str) -> str:
    chunks = retrieve(artist_name, question, n_results=5)
    context = "\n---\n".join(chunks)
    prompt = f"""You are an expert analyst of {artist_name}'s lyrics and artistic universe.
Answer ONLY using the song excerpts below. Do not use external knowledge.
If the context is insufficient, say so clearly.

LYRICS CONTEXT:
{context}

Question: {question}"""
    return _call_groq(prompt)


def compare_artists(artist1: str, artist2: str, question: str) -> str:
    chunks1 = retrieve(artist1, question, n_results=4)
    chunks2 = retrieve(artist2, question, n_results=4)
    context1 = "\n---\n".join(chunks1)
    context2 = "\n---\n".join(chunks2)
    prompt = f"""You are an expert music analyst. Compare {artist1} and {artist2} using ONLY the lyrics excerpts below.
Do not use external knowledge. If the context is insufficient for one artist, say so.

--- {artist1.upper()} LYRICS ---
{context1}

--- {artist2.upper()} LYRICS ---
{context2}

Question: {question}"""
    return _call_groq(prompt)


def route_and_ask(question: str) -> str:
    indexed = list_indexed_artists()
    artist_list = ", ".join(indexed)

    intent_prompt = f"""You are a routing assistant for a French rap chatbot. Given a question, extract:
- mode: "single" if about one artist, "compare" if comparing two artists, "unknown" if unclear
- artists: list of artist names mentioned (exact spelling from the question)

Respond ONLY with valid JSON, no explanation.
Examples:
{{"mode": "single", "artists": ["Damso"]}}
{{"mode": "compare", "artists": ["Booba", "Nekfeu"]}}
{{"mode": "unknown", "artists": []}}

Question: {question}"""

    try:
        raw = _call_groq(intent_prompt)
        intent = json.loads(raw.strip().strip("```json").strip("```").strip())
    except Exception:
        return "Je n'ai pas compris ta question. Essaie par exemple : \"Quels sont les thèmes de Damso ?\" ou \"Compare Booba et Nekfeu\"."

    mode = intent.get("mode", "unknown")
    artists = intent.get("artists", [])

    if mode == "single" and len(artists) >= 1:
        match = _find_indexed_artist(artists[0])
        if not match:
            return f"Je n'ai pas de données sur \"{artists[0]}\". Artistes disponibles : {artist_list}."
        return ask(match, question)

    if mode == "compare" and len(artists) >= 2:
        match1 = _find_indexed_artist(artists[0])
        match2 = _find_indexed_artist(artists[1])
        missing = [a for a, m in [(artists[0], match1), (artists[1], match2)] if not m]
        if missing:
            return f"Je n'ai pas de données sur {', '.join(missing)}. Artistes disponibles : {artist_list}."
        return compare_artists(match1, match2, question)

    return f"Je n'ai pas compris ta question. Essaie par exemple : \"Quels sont les thèmes de Damso ?\" ou \"Compare Booba et Nekfeu\". Artistes disponibles : {artist_list}."
