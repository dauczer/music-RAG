# Artist DNA — French Rap RAG

A RAG-powered chatbot that knows the lyrical universe of 22+ French rap artists. Ask questions about themes, compare artists, or explore patterns across the entire corpus.

**Live API:** `https://your-app.onrender.com` *(update after deployment)*

---

## What you can ask

```
# Single artist
"Quels sont les thèmes principaux de Nekfeu ?"
"Comment Damso parle-t-il de la mort ?"

# Cross-artist comparison
"Comment Booba et Orelsan parlent-ils de la réussite ?"
"Compare la vision de la rue chez SCH et Kaaris."

# General (across all artists)
"Qui parle le plus de spiritualité dans le rap français ?"
"Quels artistes évoquent leurs origines dans leurs textes ?"
```

---

## Architecture

```
User question
     │
     ▼
[Sentence Transformers]  ← embed query locally (no API call)
     │
     ▼
[ChromaDB]  ← vector similarity search → top 5 relevant lyrics chunks
     │
     ▼
[Groq / Llama 3.3 70B]  ← generate answer grounded in retrieved chunks
     │
     ▼
Answer
```

---

## Tech stack

| Role | Tool | Why |
|------|------|-----|
| Lyrics | Genius API + BeautifulSoup | Free, comprehensive French rap catalog |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Runs locally — no API cost, no rate limits |
| Vector DB | ChromaDB | Zero-setup, runs as a Python library, trivial to swap for Pinecone in prod |
| LLM | Groq / Llama 3.3 70B | Free tier, extremely fast inference |
| Backend | FastAPI | Lightweight, async-ready, auto-generates OpenAPI docs |

---

## Artist corpus (22 artists)

| Era | Artists |
|-----|---------|
| 90s classics | MC Solaar, Suprême NTM, Oxmo Puccino |
| 2000s | Booba, Rohff |
| 2010s | Kaaris, Lacrim, La Fouine, Nekfeu, Vald, Lomepal, Alpha Wann, Kekra |
| 2020s | Damso, SCH, Orelsan, Laylow, PLK, Ninho, Freeze Corleone, Gazo, Hamza, Niska |

~200 songs per artist, ~4,400 lyrics chunks total.

---

## API endpoints

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | — | Health check |
| `POST` | `/index` | `{"artist": "Nekfeu"}` | Scrape & index a new artist |
| `POST` | `/chat` | `{"artist": "Damso", "question": "..."}` | Single-artist RAG query |
| `POST` | `/compare` | `{"artist1": "...", "artist2": "...", "question": "..."}` | Cross-artist comparison |
| `POST` | `/chat/general` | `{"question": "..."}` | Query across all indexed artists |

Interactive docs available at `/docs` (FastAPI Swagger UI).

---

## Run locally

```bash
# 1. Clone & install
git clone https://github.com/<your-username>/music-rag.git
cd music-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Set env vars
cp .env.example .env
# Fill in GENIUS_ACCESS_TOKEN and GROQ_API_KEY

# 3. Start the API (vectors already committed, no indexing needed)
uvicorn api.main:app --reload
```

To add a new artist:
```bash
python -c "
from ingestion.genius_scraper import scrape_artist
from rag.vectorstore import index_artist
scrape_artist('Artist Name', max_songs=200)
index_artist('Artist Name')
"
```

---

## Project structure

```
music-rag/
├── ingestion/
│   ├── genius_scraper.py   # Genius API + BeautifulSoup scraper
│   └── build_chunks.py     # Formats lyrics into embedding-ready chunks
├── rag/
│   ├── vectorstore.py      # ChromaDB indexing & retrieval
│   └── chain.py            # RAG chain: retrieve → prompt → LLM
├── api/
│   └── main.py             # FastAPI app
├── scripts/
│   └── bulk_ingest.py      # Batch scrape & index multiple artists
├── data/raw/               # Scraped lyrics (JSON)
└── chroma_db/              # Persisted vector store
```

---

## Design decisions

**Why truncate lyrics to 3000 chars?**
Embedding models have a token limit. Explicit truncation avoids silent errors — 3000 chars ≈ 600 tokens, well within `all-MiniLM-L6-v2`'s limits.

**Why local embeddings instead of OpenAI?**
Zero cost, zero rate limits, no API dependency. `all-MiniLM-L6-v2` is 80MB and fast enough for this scale. The trade-off (slightly lower quality) is invisible at this scale.

**Why ChromaDB instead of Pinecone?**
For a local portfolio project: zero setup, runs as a Python library. Swapping to Pinecone in production would require ~10 lines of code change.

---

## What I'd improve in v2

- **Hybrid search** — combine vector similarity (semantic) with BM25 (keyword) so exact word queries ("does Nekfeu say X?") work alongside thematic queries
- **Async indexing** — `POST /index` currently blocks while scraping (~10 min). A background task queue (Celery or FastAPI `BackgroundTasks`) with a status endpoint would be better UX
- **Hosted vector DB** — swap ChromaDB for Pinecone to avoid committing the vector store to the repo
- **Streaming responses** — stream Groq output token-by-token for a better chat UX
