# Artist DNA — French Rap RAG

Ask natural-language questions about French rap lyrics and get answers grounded in real songs. The system scrapes up to ~200 songs per artist from Genius, splits them into embedding-sized passages, and retrieves the most relevant evidence before asking an LLM to answer only from that context.

**Live API:** `https://music-rag.onrender.com`
Free tier — first request after 15min of inactivity takes ~30s while the container wakes up.

---

## What you can ask

Just type naturally. An LLM-based intent detector figures out whether you're asking about one artist, comparing two, or asking about someone who isn't indexed — no buttons, no mode selection.

```
# Single artist
"What are the main themes of Nekfeu ?"
"How does Damso mention death ?"

# Cross-artist comparison
"Compare Booba and Orelsan on success thematic"
"Compare the street vision in Booba and Kaaris"

# Unknown artist → helpful error
"Tell me about Jul"
→ "I don't have Jul informations. Artists available : ..."
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
                  multilingual-e5-small (local)  ──►  vectorstore.py  ──►  chroma_db/


QUERY (real-time, POST /ask)

  User question
       │
       ▼
  Intent detection (Groq / gpt-oss-120b)  ──►  {mode: "single", artists: ["Damso"]}
       │
       ▼
  Fuzzy match artist name against indexed collections
       │
       ▼
  Embed query (HuggingFace Inference API)  ──►  ChromaDB similarity search  ──►  top 5 chunks
       │
       ▼
  Build prompt with retrieved lyrics  ──►  Groq / gpt-oss-120b  ──►  grounded answer + sources
```

`POST /ask` makes three external calls: intent detection, query embedding, and answer generation. The portfolio uses `POST /chat`, where the artist is already selected, so it skips intent detection and makes only two. A comparison reuses one query embedding for both artists. ChromaDB search itself is local.

---

## Tech stack

| Role | Tool | Why this one |
|------|------|-------------|
| Lyrics | Genius API + BeautifulSoup | Free, comprehensive French rap catalog. The API gives metadata; HTML scraping gets the actual lyrics. |
| Embeddings | `intfloat/multilingual-e5-small` | Multilingual retrieval with a 512-token context. Corpus embeddings are built locally; production sends only visitor queries to Hugging Face. |
| Vector DB | ChromaDB | Zero infrastructure. `pip install chromadb` and you have a vector database. No Docker, no hosted service, no credentials. |
| LLM | Groq / `openai/gpt-oss-120b` | Fast hosted inference, structured JSON responses, and low reasoning effort for the public demo. |
| Backend | FastAPI | Lightweight, async-ready, auto-generates OpenAPI docs at `/docs`. |

---

## Artist corpus (22 artists)

| Era | Artists |
|-----|---------|
| 90s classics | MC Solaar, Supreme NTM, Oxmo Puccino |
| 2000s | Booba, Rohff |
| 2010s | Kaaris, Lacrim, La Fouine, Nekfeu, Vald, Lomepal, Alpha Wann, Kekra |
| 2020s | Damso, Orelsan, Laylow, PLK, Ninho, Freeze Corleone, Gazo, Hamza, Niska |

The current raw corpus contains about 3,300 songs. The public `/artists` endpoint derives availability and song counts from non-empty Chroma collections instead of hard-coding them.

---

## API endpoints

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | — | Health check |
| `GET` | `/artists` | — | Non-empty indexed artists and song counts |
| `POST` | `/ask` | `{"question": "..."}` | **Main endpoint** — free-text input, auto-detects intent and routes |
| `POST` | `/chat` | `{"artist": "Damso", "question": "..."}` | Portfolio endpoint — structured answer, status, artist and sources |
| `POST` | `/compare` | `{"artist1": "...", "artist2": "...", "question": "..."}` | Direct comparison query (skips intent detection) |

All input fields are validated with a maximum of 500 characters. Generation endpoints default to three requests per minute and IP; the limit can be configured with `DEMO_RATE_LIMIT`. Interactive docs are available at `/docs`.

Three endpoints instead of one because `/ask` is the "smart" endpoint for end users, while `/chat` and `/compare` let the frontend skip the intent detection LLM call when it already knows what the user wants.

---

## Design decisions

**Local corpus embeddings, remote query embeddings**
The full lyrics corpus is embedded locally during the offline rebuild, so lyrics are not uploaded to an inference service and bulk indexing consumes no Hugging Face inference credits. Render does not load PyTorch or the model: at query time it sends only the visitor's question to the hosted version of the same model. Both sides use the required `passage:` and `query:` prefixes and normalized 384-dimensional vectors. Each user request therefore consumes one Hugging Face inference call, independently of the number of Groq calls.

**Committing `chroma_db/` to git**
It's ~82MB of binary data in version control, which isn't pretty. But it means Render's free tier works immediately on cold start — no re-embedding step, no startup delay, no external vector DB to pay for. The alternative was Pinecone (adds a dependency and a cost) or re-indexing on startup (exceeds memory limits). For a portfolio project, I chose simplicity over repo hygiene.

The rebuilt index keeps passage text in a compressed `passages.json.gz` file beside Chroma instead of duplicating it in SQLite. Chroma contains only vectors and metadata. A local full-corpus measurement reduced the staged index from about 188 MiB to about 48 MiB while preserving all 15,496 passages.

**Intent detection via LLM instead of regex**
Regex would handle "Compare X and Y" but fail on "What's the difference between X's style and Y's approach?" The LLM handles arbitrary phrasing. It costs one extra API call (~200ms) but enables a much more natural UX. The output is validated with Pydantic — if the LLM returns garbage, the system falls back gracefully with a helpful message instead of crashing.

**Fuzzy matching with 0.8 cutoff**
Users type artist names inconsistently ("damso", "Damso", "DAMSO", "Boba"). The slug normalization handles casing and accents (NFD unicode normalization); `difflib.get_close_matches` with cutoff 0.8 handles typos. "Boba" matches "booba" (87% similar), but "xyz" doesn't match anything. The threshold balances tolerance with precision.

**Overlapping passages instead of one vector per song**
Songs are split into 160-word windows with a 30-word overlap. This fits comfortably within multilingual-e5-small's 512-token context while preserving precise passages for retrieval. Every passage repeats artist, title, album and year metadata, and keeps stable song/chunk identifiers for source display.

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

After changing the chunking strategy, build a fresh index alongside the current one:

```bash
pip install -r requirements-index.txt
python -m scripts.rebuild_index
CHROMA_DB_PATH=chroma_db_v2 python -m evals.run
```

The rebuild never overwrites `chroma_db/`. Model weights are downloaded once, then lyrics passages are embedded locally. Chroma stores vectors and metadata while the text is written to `passages.json.gz`. The application defaults to the validated `chroma_db_v2/`; `CHROMA_DB_PATH` remains available for explicit overrides.

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
│   ├── bulk_ingest.py      # Batch scrape & index multiple artists
│   └── rebuild_index.py    # Safe staged rebuild after chunking changes
├── evals/
│   ├── run.py              # Source-based Hit@5 and MRR@5 evaluation
│   └── questions.jsonl     # 18 hand-written retrieval/failure cases
├── tests/                  # API, retrieval-contract and chunking tests
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
- **Stronger prompt-injection evaluation** — expand adversarial cases as the public interface evolves
- **Broader eval** — expand the manually labelled source set as the corpus evolves, while keeping human answer review
