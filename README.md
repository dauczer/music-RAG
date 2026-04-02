# Artist DNA — French Rap RAG

A RAG-powered chatbot that knows the lyrical universe of 22 French rap artists. Ask questions about themes, compare artists, or explore patterns across the corpus — in natural language.

**Live API:** `https://music-rag.onrender.com`

---

## What you can ask

Just type naturally — the API detects intent automatically:

```
# Single artist
"Quels sont les thèmes principaux de Nekfeu ?"
"Comment Damso parle-t-il de la mort ?"

# Cross-artist comparison
"Compare Booba et Orelsan sur la thématique de la réussite"
"Compare la vision de la rue chez SCH et Kaaris"

# Unknown artist → helpful error
"Parle moi de Jul"
→ "Je n'ai pas de données sur Jul. Artistes disponibles : ..."
```

---

## Architecture

```
User question
     │
     ▼
[Intent detection]  ← Groq detects: single artist / compare / unknown
     │
     ▼
[HuggingFace Inference API]  ← embed query (all-MiniLM-L6-v2, remote)
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
| Embeddings | HuggingFace Inference API (`all-MiniLM-L6-v2`) | Same model quality as local, zero RAM on server |
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
| `POST` | `/ask` | `{"question": "..."}` | **Main endpoint** — free-text input, auto-routes |
| `POST` | `/index` | `{"artist": "Nekfeu"}` | Scrape & index a new artist |
| `POST` | `/chat` | `{"artist": "Damso", "question": "..."}` | Direct single-artist query |
| `POST` | `/compare` | `{"artist1": "...", "artist2": "...", "question": "..."}` | Direct comparison query |

Interactive docs available at `/docs` (FastAPI Swagger UI).

> **Note:** The free tier on Render sleeps after 15min of inactivity. First request after sleep takes ~30s.

---

## Run locally

```bash
# 1. Clone & install
git clone https://github.com/dauczer/music-RAG.git
cd music-RAG
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Set env vars
cp .env.example .env
# Fill in GENIUS_ACCESS_TOKEN, GROQ_API_KEY, HF_TOKEN

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

To bulk ingest a list of artists:
```bash
python -m scripts.bulk_ingest
```

---

## Project structure

```
music-RAG/
├── ingestion/
│   ├── genius_scraper.py   # Genius API + BeautifulSoup scraper
│   └── build_chunks.py     # Formats lyrics into embedding-ready chunks
├── rag/
│   ├── vectorstore.py      # ChromaDB indexing & retrieval (HF embeddings)
│   └── chain.py            # Intent detection, RAG chain, fuzzy artist matching
├── api/
│   └── main.py             # FastAPI app
├── scripts/
│   └── bulk_ingest.py      # Batch scrape & index multiple artists
├── data/raw/               # Scraped lyrics (JSON)
└── chroma_db/              # Persisted vector store
```

---

## Design decisions

**Why HuggingFace Inference API instead of local embeddings?**
`sentence-transformers` loads PyTorch (~300MB) which exceeds Render's free tier RAM limit. Moving embeddings to the HF Inference API keeps the server lightweight while using the exact same model (`all-MiniLM-L6-v2`) — so existing ChromaDB vectors remain valid.

**Why truncate lyrics to 3000 chars?**
Embedding models have a token limit. Explicit truncation avoids silent errors — 3000 chars ≈ 600 tokens, well within `all-MiniLM-L6-v2`'s limits.

**Why ChromaDB instead of Pinecone?**
Zero setup, runs as a Python library, vectors committed directly to the repo. Swapping to Pinecone in production would require ~10 lines of code change.

**Why intent detection instead of separate endpoints?**
Better UX for a portfolio chatbot — users type naturally rather than selecting modes. A lightweight Groq call parses the question before the main RAG call.

---

## What I'd improve in v2

- **Hybrid search** — combine vector similarity (semantic) with BM25 (keyword) so exact word queries ("does Nekfeu say X?") work alongside thematic queries
- **Async indexing** — `POST /index` currently blocks while scraping (~10 min). A background task queue with a status endpoint would be better UX
- **Hosted vector DB** — swap ChromaDB for Pinecone to avoid committing the vector store to the repo
- **Streaming responses** — stream Groq output token-by-token for a better chat UX
