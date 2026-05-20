# Production RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) system built with **LangChain**, **LangGraph**, **ChromaDB**, and **FastAPI**.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Layer                        │
│          POST /ingest   POST /query   GET /metrics          │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    Orchestration (LangGraph)                 │
│   ┌──────────┐        ┌──────────┐                          │
│   │ retrieve ├─docs──▶│ generate │                          │
│   └────┬─────┘        └──────────┘                          │
│        │ no docs                                            │
│   ┌────▼─────┐                                              │
│   │  rewrite │──▶ retrieve (up to MAX_QUERY_REWRITES)       │
│   └──────────┘                                              │
└──────────────────────────────┬──────────────────────────────┘
          ┌────────────────────┼────────────────────┐
          │                    │                    │
┌─────────▼──────┐  ┌──────────▼───────┐  ┌────────▼───────┐
│   Ingestion    │  │    Retrieval     │  │   Evaluation   │
│  load → clean  │  │  vector search   │  │   metrics +    │
│  → chunk       │  │  + prompting     │  │   monitoring   │
└─────────┬──────┘  └──────────────────┘  └────────────────┘
          │
┌─────────▼──────┐
│   Embedding    │
│  embed → index │
│  (ChromaDB)    │
└────────────────┘
```

### Components

| Component | Responsibility |
|-----------|---------------|
| **Ingestion** | Load PDFs, TXT, CSV, URLs; clean and chunk documents |
| **Embedding** | Convert chunks to vectors via OpenAI or HuggingFace |
| **Indexing** | Persist vectors in ChromaDB for fast similarity search |
| **Retrieval** | Fetch top-k relevant documents for a given query |
| **Orchestration** | Self-corrective LangGraph workflow with query rewriting on miss |
| **Evaluation** | Token-overlap metrics for context relevance and answer coverage |
| **Monitoring** | SQLite-backed query log tracking latency and retrieval stats |
| **API** | FastAPI REST interface with automatic OpenAPI docs |

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.template .env
# Edit .env — at minimum set OPENAI_API_KEY
# Or set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY
```

### 3. Run the API

```bash
PYTHONPATH=. uvicorn src.api.main:app --reload
```

API docs available at `http://localhost:8000/docs`

### 4. Ingest documents

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"sources": ["data/sample/sample.txt"]}'
```

### 5. Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG and how does it work?"}'
```

## Python API

```python
from src.orchestration.pipeline import RAGPipeline

pipeline = RAGPipeline()

# Ingest documents (files, directories, or URLs)
pipeline.ingest(["data/sample/sample.txt"])

# Query
result = pipeline.query("What are the benefits of RAG?")
print(result["answer"])
print(result["sources"])
```

## Docker

```bash
cp .env.template .env   # fill in your API keys
docker compose up --build
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/ingest` | Ingest documents into the vector store |
| `POST` | `/api/v1/query` | Query the RAG pipeline |
| `GET` | `/api/v1/metrics` | Retrieve usage statistics |
| `DELETE` | `/api/v1/collection` | Clear the vector store |

### POST /ingest

```json
{
  "sources": ["path/to/file.pdf", "https://example.com/page"],
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

### POST /query

```json
{ "question": "What is Retrieval-Augmented Generation?" }
```

Response:

```json
{
  "question": "What is Retrieval-Augmented Generation?",
  "answer": "RAG is a technique that...",
  "sources": ["data/sample/sample.txt"],
  "latency_ms": 843
}
```

## Running Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `huggingface` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Vector store persistence path |
| `COLLECTION_NAME` | `rag_documents` | ChromaDB collection name |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Documents retrieved per query |
| `SCORE_THRESHOLD` | `0.0` | Minimum similarity score (0–1) |
| `MAX_QUERY_REWRITES` | `2` | Self-correction retries before fallback |
| `API_PORT` | `8000` | Server port |
| `MONITOR_DB_PATH` | `./data/monitor.db` | SQLite monitoring database |

## Supported File Types

- **PDF** (`.pdf`) via PyPDF
- **Text / Markdown** (`.txt`, `.md`) via TextLoader
- **CSV** (`.csv`) via CSVLoader
- **Web pages** (`http://`, `https://`) via WebBaseLoader
- **Directories** — all supported files recursively

## Project Structure

```
src/
├── config.py              # Pydantic settings (all config from .env)
├── ingestion/
│   ├── loaders.py         # Load documents from files, dirs, URLs
│   ├── cleaners.py        # Normalise whitespace and unicode
│   └── chunkers.py        # Recursive character text splitting
├── embedding/
│   ├── embedder.py        # Embedding provider abstraction
│   └── indexer.py         # ChromaDB vector store management
├── retrieval/
│   ├── retriever.py       # Similarity search with score filtering
│   └── prompt_builder.py  # RAG prompt template
├── orchestration/
│   ├── _llm.py            # LLM provider factory
│   ├── graph.py           # LangGraph self-corrective RAG workflow
│   └── pipeline.py        # High-level ingest / query interface
├── evaluation/
│   ├── metrics.py         # Relevance and coverage scoring
│   └── monitor.py         # SQLite query logging and stats
└── api/
    ├── main.py            # FastAPI app factory
    ├── routes.py          # All route handlers
    └── schemas.py         # Pydantic request / response models

tests/                     # Pytest test suite (one file per module)
data/sample/sample.txt     # Sample document for quick testing
```

## Stack

- **LangChain / LangGraph** — orchestration and chain primitives
- **ChromaDB** — embedded vector store (no server required)
- **OpenAI / Anthropic** — LLM and embedding providers
- **FastAPI + Uvicorn** — async REST API
- **SQLite** — lightweight monitoring database
- **Docker** — containerised deployment
