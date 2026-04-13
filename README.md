# Artist DNA — French Rap RAG

Ask natural-language questions about French rap lyrics and get answers grounded in real songs, not hallucinations. The system scrapes ~200 songs per artist from Genius, embeds them into a vector database, and at query time retrieves the most relevant excerpts before sending them to an LLM that answers *only* from those lyrics.

**Live API:** `https://music-rag.onrender.com`
Free tier — first request after 15min of inactivity takes ~30s while the container wakes up.

---

## What you can ask

Just type naturally. An LLM-based intent detector figures out whether you're asking about one artist, comparing two, or asking about someone who isn't indexed — no buttons, no mode selection.

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

## How it works

There are two distinct phases: **ingestion** (offline, run once per artist) and **query** (real-time, on every request).

```
INGESTION (offline)

  Genius API  ──►  genius_scraper.py  ──►  data/raw/{artist}.json
                                                  │
                                            build_chunks.py
                                                  │
                              HF Inference API  ──►  vectorstore.py  ──►  chroma_db/


QUERY (real-time, POST /ask)

  User question
       │
       ▼
  Intent detection (Groq / Llama 3.3)  ──►  {mode: "single", artists: ["Damso"]}
       │
       ▼
  Fuzzy match artist name against indexed collections
       │
       ▼
  Embed query (HuggingFace Inference API)  ──►  ChromaDB similarity search  ──►  top 5 chunks
       │
       ▼
  Build prompt with retrieved lyrics  ──►  Groq / Llama 3.3  ──►  grounded answer
```

Each request makes **3 external API calls**: intent detection (Groq, ~200-400ms), query embedding (HuggingFace, ~100-300ms), and answer generation (Groq, ~500-1500ms). ChromaDB search is local and takes ~5-10ms. Total latency: **~1-2.5 seconds**.

---

## Tech stack

| Role | Tool | Why this one |
|------|------|-------------|
| Lyrics | Genius API + BeautifulSoup | Free, comprehensive French rap catalog. The API gives metadata; HTML scraping gets the actual lyrics. |
| Embeddings | HuggingFace Inference API (`all-MiniLM-L6-v2`) | I originally ran `sentence-transformers` locally, but PyTorch (~300MB) exceeded Render's free-tier RAM. The HF API runs the exact same model remotely — same vectors, zero local memory. |
| Vector DB | ChromaDB | Zero infrastructure. `pip install chromadb` and you have a vector database. No Docker, no hosted service, no credentials. |
| LLM | Groq / Llama 3.3 70B | Free tier with extremely fast inference. Llama 3.3 70B handles structured tasks (intent extraction, text analysis) well. Low temperature (0.3) for factual consistency. |
| Backend | FastAPI | Lightweight, async-ready, auto-generates OpenAPI docs at `/docs`. |

---

## Artist corpus (22 artists)

| Era | Artists |
|-----|---------|
| 90s classics | MC Solaar, Supreme NTM, Oxmo Puccino |
| 2000s | Booba, Rohff |
| 2010s | Kaaris, Lacrim, La Fouine, Nekfeu, Vald, Lomepal, Alpha Wann, Kekra |
| 2020s | Damso, SCH, Orelsan, Laylow, PLK, Ninho, Freeze Corleone, Gazo, Hamza, Niska |

~200 songs per artist, ~4,400 lyrics chunks total.

---

## API endpoints

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | — | Health check |
| `POST` | `/ask` | `{"question": "..."}` | **Main endpoint** — free-text input, auto-detects intent and routes |
| `POST` | `/chat` | `{"artist": "Damso", "question": "..."}` | Direct single-artist query (skips intent detection) |
| `POST` | `/compare` | `{"artist1": "...", "artist2": "...", "question": "..."}` | Direct comparison query (skips intent detection) |

All input fields are validated: `min_length=1`, `max_length=500`. Rate limited to 10 requests/minute per IP on the three main endpoints. Interactive docs at `/docs` (Swagger UI).

Three endpoints instead of one because `/ask` is the "smart" endpoint for end users, while `/chat` and `/compare` let the frontend skip the intent detection LLM call when it already knows what the user wants.

---

## Design decisions

**HuggingFace Inference API instead of local embeddings**
I started with `sentence-transformers` running locally — works great in dev. But PyTorch + model weights (~300MB) exceeded Render's free-tier memory limit. Switching to the HF Inference API was a ~10-line change: same model, same vectors, zero local memory. The existing ChromaDB data stayed valid because the model is identical. Lesson learned: consider deployment constraints early.

**Committing `chroma_db/` to git**
It's ~82MB of binary data in version control, which isn't pretty. But it means Render's free tier works immediately on cold start — no re-embedding step, no startup delay, no external vector DB to pay for. The alternative was Pinecone (adds a dependency and a cost) or re-indexing on startup (exceeds memory limits). For a portfolio project, I chose simplicity over repo hygiene.

**Intent detection via LLM instead of regex**
Regex would handle "Compare X and Y" but fail on "What's the difference between X's style and Y's approach?" The LLM handles arbitrary phrasing. It costs one extra API call (~200ms) but enables a much more natural UX. The output is validated with Pydantic — if the LLM returns garbage, the system falls back gracefully with a helpful message instead of crashing.

**Fuzzy matching with 0.8 cutoff**
Users type artist names inconsistently ("damso", "Damso", "DAMSO", "Boba"). The slug normalization handles casing and accents (NFD unicode normalization); `difflib.get_close_matches` with cutoff 0.8 handles typos. "Boba" matches "booba" (87% similar), but "xyz" doesn't match anything. The threshold balances tolerance with precision.

**One chunk per song**
Individual songs are the natural semantic unit for lyrics analysis. Splitting a song across multiple chunks would lose coherence. Each chunk includes a metadata header (artist, title, album, year) so the embedding captures both the structured context and the lyrics. Lyrics are truncated to 3000 chars (~600 tokens) to stay within the model's limits.

**Three layers of data quality filtering**
1. At scrape time: skip unofficial snippets (title ends with `*`), skip songs with no lyrics, skip songs under 600 characters (filters out intros, skits)
2. At chunk time: skip songs with empty lyrics after cleaning
3. At retrieval time: vector similarity naturally ranks irrelevant songs low — only the top N most relevant chunks reach the LLM

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
│   ├── chain.py            # Intent detection, RAG chain, fuzzy artist matching
│   └── messages.py         # User-facing error strings (French)
├── api/
│   └── main.py             # FastAPI app (CORS, rate limiting, validation)
├── scripts/
│   └── bulk_ingest.py      # Batch scrape & index multiple artists
├── evals/
│   ├── run.py              # Retrieval evaluation harness (recall@4)
│   └── questions.jsonl     # 10 hand-written eval cases
├── tests/                  # 18 tests (validation, CORS, rate limiting, routing)
├── data/raw/               # Scraped lyrics JSON (not committed)
├── chroma_db/              # Persisted vector store (committed)
└── .github/workflows/      # CI: lint, format, test, dependency audit
```

---

## What I'd improve next

- **Response caching** — identical questions hit the LLM every time. A Redis cache on common queries could eliminate a good chunk of LLM calls
- **Hybrid search** — combine vector similarity (semantic) with BM25 (keyword) so exact-word queries ("does Nekfeu say X?") work alongside thematic ones
- **Streaming responses** — SSE for token-by-token output, much better chat UX
- **Conversation memory** — currently stateless. Adding chat history would enable follow-up questions ("and what about on his second album?")
- **Auth + usage tracking** — API keys at minimum, per-user quotas, analytics
- **Async ingestion** — `POST /index` currently blocks while scraping (~10 min). A background task queue with status tracking would be better
- **Prompt injection mitigation** — user input goes directly into LLM prompts right now. Input sanitization or a guardrail model would help
- **Better eval** — RAGAS metrics, LLM-as-judge for answer quality, automated regression testing in CI
