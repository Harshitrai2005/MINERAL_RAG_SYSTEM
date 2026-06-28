# MEIS — Mineral Exploration Intelligence System

> A production-ready RAG system for geological intelligence — clean layered architecture,
> swappable vector stores & LLM providers, Prometheus metrics, and Grafana dashboards.

**Python 3.11+** | **FastAPI** | **LanceDB / Qdrant** | **Groq / OpenAI** | **Docker + Render**

**Live demo:** https://mineral-exploration-rag.onrender.com
*(first load ~30 s cold start on free tier — Render free tier spins down after 15 min)*

---

## What it does

Upload geological PDF reports and geochemical CSV/JSON datasets. Ask natural-language questions.
Get precise, source-cited answers grounded in your own data — the pipeline hard-blocks LLM calls
when context is empty, so hallucination is structurally impossible.

---

## Architecture

```
Browser --> FastAPI (backend/)
               |-- RAGService       --> VectorRepository  (LanceDB | Qdrant)
               |                   --> LLMProvider        (Groq    | OpenAI)
               |                   --> CrossEncoderReranker
               |                   --> HybridRetriever    (BM25 + dense fusion)
               |-- IngestionService --> PDFProcessor / MineralDatasetProcessor
               +-- IngestionQueue   (async background worker for large files)

Prometheus --> GET /api/metrics
Grafana    --> Prometheus datasource
```

Every infrastructure dependency sits behind an **abstract interface**.
Swap LanceDB for Qdrant — or Groq for OpenAI — by changing one environment variable.
Zero business-logic changes. This is the Dependency Inversion Principle in practice.

---

## Tech Stack

| Layer       | Technology                          | Notes                                        |
|-------------|-------------------------------------|----------------------------------------------|
| API         | FastAPI + Uvicorn                   | Async, auto OpenAPI docs at `/api/docs`      |
| RAG         | Hybrid BM25 + vector search         | Cross-encoder reranking (ms-marco-MiniLM)    |
| Vector DB   | LanceDB (local) / Qdrant Cloud      | Swappable via `VECTOR_BACKEND` env var       |
| LLM         | Groq llama-3.3-70b (free) / OpenAI  | Swappable via `LLM_PROVIDER` env var         |
| Embeddings  | sentence-transformers (local, CPU)  | No external embedding API needed             |
| Metrics     | Prometheus + Grafana                | Full observability stack via Docker Compose  |
| Persistence | SQLite                              | Persistent query metrics dashboard           |
| Deployment  | Docker + Render.com                 | `render.yaml` blueprint for one-click deploy |

---

## RAG Pipeline — 5 Explicit Steps

1. **Retrieve** — adaptive multi-collection vector search + BM25 hybrid fusion
2. **Re-rank** — cross-encoder scores every (query, chunk) pair with filename boosting
3. **Guard** — if no relevant context found, return a clear message; LLM is never called
4. **Generate** — structured prompt with strict instructions to cite sources, never hallucinate
5. **Assemble** — deduplicated source citations with similarity scores and page references

---

## Key Features

**Smart clarifying questions** — when a query is too broad or ambiguous, the system generates
3 clickable clarifying questions instead of a vague answer.

**Source-scoped queries** — `source_filter` restricts retrieval to a single uploaded file
so you can query one dataset without cross-contamination from others.

**Streaming responses** — `POST /api/query/` with `"stream": true` returns tokens in real-time.

**Async ingestion queue** — large files (100MB+ JSONs, multi-hundred-page PDFs) are queued
and processed in the background; upload returns immediately with a `job_id` to poll.

**Domain validation** — uploads are checked for geological/geochemical content; unrelated
files (legal, medical, financial) are rejected at the gate.

**Persistent metrics dashboard** — every query is logged to SQLite and survives restarts.
View, filter, and delete individual records from the UI or via API.

**Delete by file or by type** — remove a single document or all files of a given extension
(pdf, csv, json, txt) from the index in one call.

---

## RAG Evaluation

`POST /api/analysis/evaluate` scores any (query, answer, context) triple on four criteria:

| Criterion    | What it checks                             |
|--------------|--------------------------------------------|
| Relevance    | Does the answer address the question?      |
| Faithfulness | Is every claim grounded in the context?    |
| Completeness | Are all aspects of the question covered?   |
| Conciseness  | Is the answer free of padding?             |

Each score is 0-1 with a detailed explanation from the LLM judge.
Scores are also exposed as Prometheus metrics: `meis_evaluation_scores{criterion="..."}`.

---

## Quick Start (5 minutes, no Docker needed)

Prerequisites: Python 3.11+, free Groq API key from https://console.groq.com (no credit card)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/MEIS-GeoIntel.git
cd MEIS-GeoIntel/meis_v6

# 2. Install CPU-only PyTorch first (saves ~2 GB vs the CUDA wheel)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# 3. Install everything else
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env: set GROQ_API_KEY and API_KEY (any strong random string)

# 5. Start the backend — two equivalent options:
#    Option A: from inside backend/
cd backend && python main.py

#    Option B: from meis_v6/ root (recommended for dev, enables --reload)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# App UI:   http://localhost:8000
# API docs: http://localhost:8000/api/docs

# 6. Seed sample geological data (new terminal, from meis_v6/)
python scripts/seed_sample_data.py
```

---

## Full Observability Stack (Docker)

```bash
docker compose -f docker/docker-compose.yml up --build

# App:        http://localhost:8000
# API docs:   http://localhost:8000/api/docs
# Metrics:    http://localhost:8000/api/metrics
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000   (login: admin / admin)
```

---

## API Reference

### Ingestion

| Method | Endpoint                          | Description                                    |
|--------|-----------------------------------|------------------------------------------------|
| POST   | `/api/ingest/upload`              | Upload and index a file (synchronous)          |
| POST   | `/api/ingest/upload-async`        | Upload large file — returns job_id immediately |
| GET    | `/api/ingest/job/{job_id}`        | Poll status of an async ingestion job          |
| GET    | `/api/ingest/jobs`                | List recent ingestion jobs                     |
| GET    | `/api/ingest/files`               | List all indexed source files                  |
| GET    | `/api/ingest/stats`               | Document counts per collection                 |
| DELETE | `/api/ingest/document`            | Remove a single document from the index        |
| DELETE | `/api/ingest/document/by-type`    | Delete all files of a given extension          |

### Query & RAG

| Method | Endpoint                          | Description                                    |
|--------|-----------------------------------|------------------------------------------------|
| POST   | `/api/query/`                     | Natural language RAG query (supports stream)   |
| POST   | `/api/query/rock-formation`       | Specialized rock formation interpretation      |
| POST   | `/api/query/mineral-zone`         | Specialized mineral zone identification        |
| GET    | `/api/query/metrics-history`      | Persistent query metrics dashboard             |
| DELETE | `/api/query/metrics-history/{id}` | Delete a single metrics record                 |
| DELETE | `/api/query/metrics-history`      | Clear all metrics history                      |

### Analysis

| Method | Endpoint                          | Description                                    |
|--------|-----------------------------------|------------------------------------------------|
| POST   | `/api/analysis/mineral-zones`     | Mineral zone analysis with optional report     |
| POST   | `/api/analysis/exploration-decision` | Exploration decision support (phased plan)  |
| GET    | `/api/analysis/deposit-models`    | List all supported deposit models              |
| POST   | `/api/analysis/evaluate`          | Score a RAG answer on 4 quality criteria       |

### System

| Method | Endpoint        | Description                               |
|--------|-----------------|-------------------------------------------|
| GET    | `/api/health`   | Liveness probe + vector DB status         |
| GET    | `/api/metrics`  | Prometheus metrics scrape endpoint        |

---

## Deploy to Render (free tier)

1. Fork this repo and push to GitHub (make sure `.env` is in `.gitignore`)
2. Create a free Qdrant Cloud cluster at https://cloud.qdrant.io (1 GB free, no card)
3. Go to https://render.com --> New --> Blueprint --> connect your repo
4. Set these secrets in the Render dashboard:

| Variable         | Where to get it                          |
|------------------|------------------------------------------|
| `GROQ_API_KEY`   | https://console.groq.com                |
| `API_KEY`        | run: `openssl rand -hex 32`             |
| `QDRANT_URL`     | your Qdrant cluster URL                  |
| `QDRANT_API_KEY` | your Qdrant API key                      |

5. Click Deploy — Render picks up `render.yaml` automatically

> Free-tier note: 512 MB RAM, shared CPU. Spins down after 15 min of inactivity.
> First request after a cold start takes ~30 s while ML models load.
> Use VECTOR_BACKEND=qdrant on Render so data persists across restarts.

---

## Supported File Types

| Type    | Category param | Processor                                             |
|---------|----------------|-------------------------------------------------------|
| `.pdf`  | `report`       | PyMuPDF page-by-page extraction + section detection   |
| `.txt`  | `report`       | Semantic text chunker                                 |
| `.csv`  | `dataset`      | MineralDatasetProcessor — anomaly detection, zones    |
| `.json` | `dataset`      | MineralDatasetProcessor — flattens nested structures  |

---

## Run Tests

```bash
# Unit tests — no API key, no database needed
pytest tests/test_core.py -v

# Route integration tests
pytest tests/test_routes.py -v

# RAG quality evaluation suite
pytest tests/evaluation/test_rag_metrics.py -v
```

---

## Project Structure

```
meis_v6/
|-- backend/
|   |-- api/routes/          # query, ingest, analysis, health, metrics
|   |-- core/                # config, DI container, security, rate limiting
|   |-- infra/               # LanceDB, Qdrant, Groq, OpenAI, BM25, reranker
|   |-- ingestion/           # PDF processor, mineral dataset processor
|   |-- repositories/        # Abstract interfaces: VectorRepository, LLMProvider
|   |-- services/            # RAGService, IngestionService, IngestionQueue
|   +-- main.py
|-- frontend/
|   +-- index.html           # Single-file UI served by FastAPI
|-- tests/
|   |-- test_core.py
|   |-- test_routes.py
|   +-- evaluation/test_rag_metrics.py
|-- sample_data/             # Ready-to-use geological CSVs, JSONs, reports
|-- scripts/seed_sample_data.py
|-- docker/                  # Dockerfile + docker-compose (Prometheus + Grafana)
|-- monitoring/              # Grafana dashboard JSON + Prometheus config
|-- render.yaml              # One-click Render deploy blueprint
|-- requirements.txt
+-- .env.example
```

---

## Key Prometheus Metrics

```
meis_rag_query_duration_seconds    # RAG pipeline latency (histogram)
meis_rag_chunks_retrieved          # Chunks per query (histogram)
meis_evaluation_scores             # Quality scores 0-1 per criterion
meis_ingest_documents_total        # Ingestion success/error counts
meis_http_request_duration_seconds # Full HTTP latency
meis_errors_total                  # Errors by endpoint and type
```

Scrape endpoint: `GET /api/metrics`

---

## Environment Variables

See `.env.example` for a fully commented reference of all supported variables.
