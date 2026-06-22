# CogniCore Q&A

A retrieval-augmented question-answering system over a local document corpus.
Ask natural-language questions about the CogniCore product documentation and get
answers grounded in the source material, with citations back to the documents
they came from.

## Architecture

A question flows from the browser to a FastAPI backend, which embeds the query,
retrieves the most relevant document chunks from a persistent vector store, and
asks a pluggable LLM to synthesize an answer using **only** that retrieved
context. The answer and its sources are returned to the frontend.

```
                    ┌─────────────────────────────────────────────┐
                    │                  FastAPI                     │
   ┌──────────┐     │  ┌───────────┐   ┌──────────────────────┐    │
   │ Frontend │────▶│  │ /ask      │──▶│ Retrieval            │    │
   │ (HTML/JS)│     │  │ endpoint  │   │  embed query ──┐     │    │
   │          │◀────│  │           │   │                ▼     │    │
   └──────────┘     │  └───────────┘   │           ┌─────────┐│    │
        ▲           │        │         │           │ChromaDB ││    │
        │           │        │         │           │(vectors)││    │
     answer +       │        │         │           └─────────┘│    │
     sources        │        │         └──────────────────────┘    │
                    │        │   context + question                │
                    │        ▼                                     │
                    │  ┌──────────────────────┐                    │
                    │  │ LLM (Ollama / Groq)  │── answer ──────────┘
                    │  └──────────────────────┘
                    └─────────────────────────────────────────────┘

Offline (one-time):  data/docs/*.md ──▶ ingestion (chunk) ──▶ embeddings ──▶ ChromaDB
```

### Modules

| Module | Responsibility |
| --- | --- |
| `app/ingestion.py` | Loads markdown docs and splits them into section-aware, size-bounded chunks. |
| `app/embeddings.py` | Wraps a sentence-transformers model; lazily loaded and reused to embed text. |
| `app/vectorstore.py` | Persistent ChromaDB collection (cosine space): upsert chunks and query by similarity. |
| `app/retrieval.py` | Embeds a question and returns the top-K most relevant chunks as typed results. |
| `app/llm.py` | LLM provider abstraction with an Ollama implementation, selected via config. |
| `app/rag.py` | Builds the grounded prompt from retrieved context and synthesizes the final answer. |
| `app/main.py` | FastAPI app: serves the frontend and the `/ask` and `/health` endpoints. |
| `app/config.py` | Pydantic-settings configuration loaded from `.env`. |
| `app/schemas.py` | Pydantic request/response models for the API. |
| `scripts/ingest.py` | CLI that builds, embeds, and indexes all chunks into the vector store. |

## How to run

### Option 1: Docker (recommended)

```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.2   # one-time, after first start
```

Then open <http://localhost:8000>. The vector store is built automatically on
container startup and persists in a named volume across restarts.

### Option 2: Local

```bash
# 1. Python 3.11+, create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure the LLM provider
cp .env.example .env        # then edit as needed

# 4. If using Ollama, install it (https://ollama.com), start it, and pull a model
ollama serve &
ollama pull llama3.1        # match OLLAMA_MODEL in your .env

# 5. Index the documents into the vector store
python -m scripts.ingest

# 6. Run the API + frontend
uvicorn app.main:app --reload
```

Then open <http://localhost:8000>.

### Running tests

```bash
python -m pytest tests/ -v
```

> Note: the retrieval tests read the persisted vector store, so run
> `python -m scripts.ingest` at least once before running the suite.

## Design decisions & trade-offs

- **Markdown-aware chunking vs naive splitting.** We split documents along their
  header hierarchy and attach a breadcrumb (e.g. `billing.md > Refunds >
  Eligibility`) to every chunk, only sub-splitting sections that exceed the size
  limit. This keeps semantically related text together and yields precise,
  citable sources, where blind fixed-size splitting would cut across topics and
  blur attribution.

- **Sentence-transformers (local) vs OpenAI embeddings.** We embed with a local
  `all-MiniLM-L6-v2` model so the system runs fully offline with no API key,
  no per-call cost, and no data leaving the machine. The trade-off is somewhat
  lower embedding quality and a one-time model download versus a hosted API.

- **ChromaDB with persistence vs in-memory FAISS.** ChromaDB gives us on-disk
  persistence, metadata storage, and a simple upsert/query API out of the box,
  so the index survives restarts without extra plumbing. FAISS can be faster at
  large scale but would require us to manage serialization and metadata
  ourselves — unnecessary complexity for this corpus size.

- **Pluggable LLM provider pattern.** Generation sits behind an `LLMProvider`
  interface chosen at runtime from config, so switching between a local Ollama
  model and a hosted Groq model is a configuration change, not a code change.
  This keeps the RAG logic backend-agnostic and makes the system easy to demo
  locally and later point at a faster hosted model.

- **Plain HTML/JS frontend vs React.** The UI is a single self-contained HTML
  file with vanilla JS and no build step, which is ideal for a proof of concept:
  zero toolchain, instant to serve from FastAPI, easy to read. A richer app
  would justify React, but here it would add tooling overhead for no benefit.



## Testing approach

- **What we test.** Ingestion logic (header splitting, size limits, empty-section
  handling, deterministic IDs) as fast unit tests; retrieval relevance (results
  are well-formed and the top hit matches the expected document) as integration
  tests against the real embedder and vector store; and the API contract
  (`/health` shape, input validation on `/ask`).

- **What we don't test, and why.** We don't assert on the LLM's generated answer
  text. LLM output is non-deterministic and model-dependent, so exact-match
  assertions would be brittle; we verify the grounded pipeline up to the model
  boundary instead.

- **What we'd add for production.** An evaluation framework such as RAGAS to
  measure faithfulness and answer relevance, a curated golden Q&A set to catch
  regressions in retrieval and generation quality, and a CI pipeline that runs
  the suite (plus the evaluation) on every change.

## Future improvements / path to production

- **Hybrid search** — combine keyword (BM25) and semantic retrieval to handle
  exact terms and rare identifiers that embeddings alone miss.
- **Cross-encoder re-ranking** — re-score the top candidates with a cross-encoder
  for sharper ordering before they reach the LLM.
- **Async + streaming** — make LLM calls async and stream tokens to the frontend
  so answers appear progressively instead of after a blocking wait.
- **Authentication & rate limiting** — add API keys/auth and per-client rate
  limits before any public exposure.
- **Evaluation pipeline** — golden Q&A pairs plus automated faithfulness/relevance
  scoring to track quality over time.
- **CI/CD** — automated tests, linting, and image builds on every commit, with
  ingestion run as part of deployment.
- **Monitoring & logging** — structured query logs, latency metrics, and
  retrieval relevance scores to observe and debug the system in production.

