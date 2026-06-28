# Quick Start — Mineral Exploration Intelligence System

## Prerequisites
- Python 3.11+
- A free Groq API key from https://console.groq.com (no credit card)

## Setup & Run (Local, ~5 minutes)

```bash
# 1. Clone and enter the project
git clone https://github.com/YOUR_USERNAME/MEIS-GeoIntel.git
cd MEIS-GeoIntel/meis_v6

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies (install PyTorch CPU-only first to save ~2GB)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — set GROQ_API_KEY and API_KEY at minimum

# 5. Start the backend — pick either:
#    From inside backend/:
cd backend && python main.py

#    OR from meis_v6/ root (recommended for development, enables --reload):
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 6. Seed the sample geological data (new terminal, venv active, from meis_v6/)
python scripts/seed_sample_data.py

# 7. Open http://localhost:8000
```

## Example Queries to Try
- "What gold anomalies were identified in the north zone?"
- "Which drill targets have the highest Cu-Au grades?"
- "Is this an epithermal or porphyry system?"
- "Compare copper grades between zones"
- "What are the recommended next exploration steps?"

## API Endpoints
- `GET  /api/health`           — Liveness probe + collection stats
- `GET  /api/ready`            — Readiness probe + doc counts per collection
- `POST /api/ingest/upload`    — Upload & index a file (requires X-API-Key header)
- `POST /api/query/`           — Natural language RAG query
- `POST /api/analysis/evaluate`— Score a (query, answer, context) triple on 4 criteria
- `GET  /api/metrics`          — Prometheus metrics

## Docker (full observability stack)
```bash
docker compose -f docker/docker-compose.yml up --build
# App: http://localhost:8000  |  Prometheus: http://localhost:9090  |  Grafana: http://localhost:3000
```

## Deploy to Render (free)
See [DEPLOYMENT.md](DEPLOYMENT.md) for full steps. Short version:
1. Push repo to GitHub (`.env` must be in `.gitignore`)
2. Create free Qdrant Cloud cluster at https://cloud.qdrant.io
3. `render.com` → New → Blueprint → connect repo → set 4 secrets → Deploy
