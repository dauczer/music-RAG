import gzip
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import requests
from dotenv import load_dotenv

from ingestion.build_chunks import build_chunks

load_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "chroma_db_v2"
_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
_HF_API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    f"{_EMBEDDING_MODEL}/pipeline/feature-extraction"
)
_CHROMA_BATCH_SIZE = 500
_client = None
_client_path: Path | None = None
_passage_store: dict[str, str] | None = None
_passage_store_path: Path | None = None
_local_embedding_model: Any | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    document: str
    artist: str
    title: str
    album: str | None
    year: str | None
    chunk_index: int
    distance: float | None

    @property
    def source_id(self) -> str:
        marker = ":chunk-"
        return self.id.split(marker, maxsplit=1)[0] if marker in self.id else self.id


@dataclass(frozen=True)
class ArtistSummary:
    slug: str
    name: str
    song_count: int
    chunk_count: int


def _db_path() -> Path:
    configured = os.getenv("CHROMA_DB_PATH")
    return Path(configured).expanduser().resolve() if configured else _DEFAULT_DB_PATH


def _unit_normalize(embeddings: list[list[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for embedding in embeddings:
        magnitude = sum(value * value for value in embedding) ** 0.5
        if magnitude == 0:
            raise RuntimeError("The embedding service returned a zero vector.")
        normalized.append([value / magnitude for value in embedding])
    return normalized


def _embed_queries(texts: list[str]) -> list[list[float]]:
    """Embed visitor queries remotely; corpus passages are never sent here."""
    if not texts:
        return []

    token = os.getenv("HF_TOKEN")
    if not token:
        raise OSError("HF_TOKEN is not set. Check your .env file.")

    try:
        response = requests.post(
            _HF_API_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": [f"query: {text}" for text in texts]},
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError("The embedding service is temporarily unavailable.") from exc
    embeddings = response.json()
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise RuntimeError("Hugging Face returned an unexpected embedding response.")
    return _unit_normalize(embeddings)


def _get_local_embedding_model() -> Any:
    global _local_embedding_model
    if _local_embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local indexing dependencies are missing. Install requirements-index.txt."
            ) from exc
        _local_embedding_model = SentenceTransformer(_EMBEDDING_MODEL)
    return _local_embedding_model


def _embed_passages_locally(texts: list[str]) -> list[list[float]]:
    """Embed corpus text locally so lyrics are not uploaded to an inference API."""
    if not texts:
        return []
    embeddings = _get_local_embedding_model().encode(
        [f"passage: {text}" for text in texts],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.tolist()


def _get_client() -> chromadb.PersistentClient:
    global _client, _client_path
    path = _db_path()
    if _client is None or _client_path != path:
        _client = chromadb.PersistentClient(path=str(path))
        _client_path = path
    return _client


def create_client(path: Path) -> chromadb.PersistentClient:
    """Create an explicit client, primarily for safe staged index rebuilds."""
    return chromadb.PersistentClient(path=str(path.resolve()))


def _load_passage_store() -> dict[str, str]:
    global _passage_store, _passage_store_path
    path = _db_path() / "passages.json.gz"
    if _passage_store is None or _passage_store_path != path:
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as file:
                loaded = json.load(file)
            if not isinstance(loaded, dict):
                raise RuntimeError(f"Invalid passage store at {path}.")
            _passage_store = {str(key): str(value) for key, value in loaded.items()}
        else:
            _passage_store = {}
        _passage_store_path = path
    return _passage_store


def _collection_name(artist_name: str) -> str:
    normalized = unicodedata.normalize("NFD", artist_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_name.lower().replace(" ", "_")


def _chunk_id(artist_name: str, metadata: dict[str, Any]) -> str:
    return (
        f"{_collection_name(artist_name)}:song-{int(metadata['song_index']):04d}"
        f":chunk-{int(metadata['chunk_index']):03d}"
    )


def index_artist(
    artist_name: str,
    *,
    client: Any | None = None,
    chunks: list[dict[str, Any]] | None = None,
    store_documents: bool = True,
) -> bool:
    db_client = client or _get_client()
    collection = db_client.get_or_create_collection(_collection_name(artist_name))

    if collection.count() > 0:
        logger.info("%s already indexed (%d chunks). Skipping.", artist_name, collection.count())
        return False

    chunks = chunks if chunks is not None else build_chunks(artist_name)
    if not chunks:
        raise ValueError(f"No chunks found for {artist_name}. Run the scraper first.")

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [_chunk_id(artist_name, metadata) for metadata in metadatas]

    logger.info("Embedding %d chunks for %s...", len(chunks), artist_name)
    embeddings = _embed_passages_locally(texts)

    for start in range(0, len(chunks), _CHROMA_BATCH_SIZE):
        end = start + _CHROMA_BATCH_SIZE
        batch = {
            "ids": ids[start:end],
            "embeddings": embeddings[start:end],
            "metadatas": metadatas[start:end],
        }
        if store_documents:
            batch["documents"] = texts[start:end]
        collection.add(
            **batch,
        )

    logger.info("Indexed %d chunks for %s", len(chunks), artist_name)
    return True


def list_artist_summaries() -> list[ArtistSummary]:
    summaries: list[ArtistSummary] = []

    for collection in _get_client().list_collections():
        chunk_count = collection.count()
        if chunk_count == 0:
            continue

        stored = collection.get(include=["metadatas"])
        metadatas = [metadata or {} for metadata in (stored.get("metadatas") or [])]
        first = metadatas[0] if metadatas else {}
        name = str(first.get("artist") or collection.name.replace("_", " ").title())

        song_keys = {
            str(metadata.get("song_index"))
            if metadata.get("song_index") is not None
            else str(metadata.get("title") or "")
            for metadata in metadatas
        }
        song_keys.discard("")
        summaries.append(
            ArtistSummary(
                slug=collection.name,
                name=name,
                song_count=len(song_keys),
                chunk_count=chunk_count,
            )
        )

    return sorted(summaries, key=lambda artist: artist.name.casefold())


def list_indexed_artists() -> list[str]:
    return sorted(
        collection.name for collection in _get_client().list_collections() if collection.count() > 0
    )


def _normalize_search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").casefold()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_value).split())


def _matching_title(collection: Any, query: str) -> str | None:
    normalized_query = f" {_normalize_search_text(query)} "
    stored = collection.get(include=["metadatas"])
    titles = {
        str(metadata.get("title"))
        for metadata in (stored.get("metadatas") or [])
        if metadata and metadata.get("title")
    }
    matches = [
        title
        for title in titles
        if len(_normalize_search_text(title)) >= 4
        and f" {_normalize_search_text(title)} " in normalized_query
    ]
    return max(matches, key=lambda title: len(_normalize_search_text(title)), default=None)


def _query_rows(results: dict) -> list[tuple[str, str | None, dict, float | None]]:
    ids = results.get("ids", [[]])[0]
    document_batches = results.get("documents")
    documents = (
        document_batches[0] if document_batches and document_batches[0] else [None] * len(ids)
    )
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    return list(zip(ids, documents, metadatas, distances, strict=True))


def retrieve_chunks(
    artist_name: str,
    query: str,
    n_results: int = 5,
    *,
    query_embedding: list[float] | None = None,
    match_title: bool = True,
) -> list[RetrievedChunk]:
    try:
        collection = _get_client().get_collection(_collection_name(artist_name))
    except chromadb.errors.NotFoundError as exc:
        raise ValueError(f"{artist_name} is not indexed. Call index_artist() first.") from exc

    count = collection.count()
    if count == 0:
        raise ValueError(f"{artist_name} has an empty index.")

    query_embedding = query_embedding or _embed_queries([query])[0]
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"],
    )

    rows = _query_rows(dense_results)
    matched_title = _matching_title(collection, query) if match_title else None
    if matched_title:
        title_count = len(
            collection.get(where={"title": {"$eq": matched_title}}, include=[]).get("ids", [])
        )
        title_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(2, title_count),
            where={"title": {"$eq": matched_title}},
            include=["documents", "metadatas", "distances"],
        )
        rows = _query_rows(title_results) + rows

    retrieved: list[RetrievedChunk] = []
    seen_ids: set[str] = set()
    passage_store = _load_passage_store()
    for chunk_id, document, metadata, distance in rows:
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        metadata = metadata or {}
        document = document or passage_store.get(str(chunk_id))
        if not document:
            raise RuntimeError(f"Missing text for retrieved passage {chunk_id}.")
        album = str(metadata.get("album") or "").strip()
        year = str(metadata.get("year") or "").strip()
        retrieved.append(
            RetrievedChunk(
                id=str(chunk_id),
                document=str(document),
                artist=str(metadata.get("artist") or artist_name),
                title=str(metadata.get("title") or "Unknown"),
                album=None if album.casefold() in {"", "unknown", "none"} else album,
                year=None if year.casefold() in {"", "unknown", "none"} else year,
                chunk_index=int(metadata.get("chunk_index") or 0),
                distance=float(distance) if distance is not None else None,
            )
        )
        if len(retrieved) == n_results:
            break
    return retrieved


def retrieve_across_artists(
    artist_names: list[str],
    query: str,
    n_results: int = 4,
    *,
    match_titles: bool = True,
) -> list[list[RetrievedChunk]]:
    """Retrieve for several artists while embedding the shared query only once."""
    query_embedding = _embed_queries([query])[0]
    return [
        retrieve_chunks(
            artist_name,
            query,
            n_results,
            query_embedding=query_embedding,
            match_title=match_titles,
        )
        for artist_name in artist_names
    ]


def retrieve_global_chunks(
    query: str,
    *,
    n_results: int = 8,
    candidates_per_artist: int = 2,
    max_chunks_per_artist: int = 2,
) -> list[RetrievedChunk]:
    """Search every indexed artist with one query embedding and merge fairly."""
    if n_results < 1 or candidates_per_artist < 1 or max_chunks_per_artist < 1:
        raise ValueError("Global retrieval limits must be positive integers.")

    artist_names = list_indexed_artists()
    if not artist_names:
        return []

    grouped = retrieve_across_artists(
        artist_names,
        query,
        n_results=candidates_per_artist,
        match_titles=False,
    )
    candidates = sorted(
        (chunk for chunks in grouped for chunk in chunks),
        key=lambda chunk: float("inf") if chunk.distance is None else chunk.distance,
    )

    selected: list[RetrievedChunk] = []
    seen_songs: set[str] = set()
    chunks_per_artist: dict[str, int] = {}
    for chunk in candidates:
        artist_key = _collection_name(chunk.artist)
        if chunk.source_id in seen_songs:
            continue
        if chunks_per_artist.get(artist_key, 0) >= max_chunks_per_artist:
            continue

        selected.append(chunk)
        seen_songs.add(chunk.source_id)
        chunks_per_artist[artist_key] = chunks_per_artist.get(artist_key, 0) + 1
        if len(selected) == n_results:
            break

    return selected


def retrieve(artist_name: str, query: str, n_results: int = 4) -> list[str]:
    """Backward-compatible document-only retrieval helper."""
    return [chunk.document for chunk in retrieve_chunks(artist_name, query, n_results)]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    index_artist("Damso")
    logger.info("--- Retrieval test ---")
    for result in retrieve_chunks("Damso", "thèmes sur la mort et la solitude"):
        print(f"{result.title}: {result.document[:220]}")
        print("---")
