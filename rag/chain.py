import os

from dotenv import load_dotenv
from groq import Groq

from rag.vectorstore import list_indexed_artists, rank_artists_by_relevance, retrieve

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


def ask_general(question: str) -> str:
    if not list_indexed_artists():
        return "No artists have been indexed yet."
    top_artists = rank_artists_by_relevance(question, top_n=5)
    all_chunks = []
    for artist in top_artists:
        all_chunks.extend(retrieve(artist, question, n_results=2))
    context = "\n---\n".join(all_chunks)
    prompt = f"""You are an expert music analyst. Answer using ONLY the song excerpts below.
Do not use external knowledge. If the context is insufficient, say so.

LYRICS CONTEXT:
{context}

Question: {question}"""
    return _call_groq(prompt)
