import unicodedata
from pathlib import Path

import chromadb

from ingestion.build_chunks import build_chunks

_DB_PATH = Path(__file__).parent.parent / "chroma_db"
_model = None
_client = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(_DB_PATH))
    return _client


def _collection_name(artist_name: str) -> str:
    normalized = unicodedata.normalize("NFD", artist_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_name.lower().replace(" ", "_")


def index_artist(artist_name: str) -> None:
    collection = _get_client().get_or_create_collection(_collection_name(artist_name))

    if collection.count() > 0:
        print(f"{artist_name} already indexed ({collection.count()} chunks). Skipping.")
        return False

    chunks = build_chunks(artist_name)
    if not chunks:
        raise ValueError(f"No chunks found for {artist_name}. Run the scraper first.")

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{artist_name}_{i}" for i in range(len(chunks))]

    print(f"Embedding {len(chunks)} chunks for {artist_name}...")
    embeddings = _get_model().encode(texts, show_progress_bar=True).tolist()

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Indexed {len(chunks)} chunks for {artist_name}")
    return True


def list_indexed_artists() -> list[str]:
    return [col.name for col in _get_client().list_collections()]


def rank_artists_by_relevance(query: str, top_n: int = 8) -> list[str]:
    """Return the top_n most relevant artist collection names for a given query."""
    query_embedding = _get_model().encode([query])[0].tolist()
    scored = []
    for col in _get_client().list_collections():
        results = col.query(query_embeddings=[query_embedding], n_results=1)
        if results["distances"] and results["distances"][0]:
            scored.append((col.name, results["distances"][0][0]))
    scored.sort(key=lambda x: x[1])
    return [name for name, _ in scored[:top_n]]


def retrieve(artist_name: str, query: str, n_results: int = 4) -> list[str]:
    try:
        collection = _get_client().get_collection(_collection_name(artist_name))
    except Exception:
        raise ValueError(f"{artist_name} is not indexed. Call index_artist() first.")

    query_embedding = _get_model().encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results["documents"][0]


if __name__ == "__main__":
    index_artist("Damso")
    print("\n--- Retrieval test ---")
    results = retrieve("Damso", "thèmes sur la mort et la solitude")
    for r in results:
        print(r[:300])
        print("---")
