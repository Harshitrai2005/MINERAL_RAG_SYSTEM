# MEIS — Ordered Commands Reference

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | https://python.org |
| Docker Desktop | 24+ | https://docker.com |
| Git | any | https://git-scm.com |

---

## SECTION 1 — LOCAL DEVELOPMENT (no Docker)

### Step 1 — Clone

```bash
git clone https://github.com/YOUR_USERNAME/MEIS-GeoIntel.git
cd MEIS-GeoIntel/meis_v6
```

### Step 2 — Create virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Step 3 — Install CPU-only PyTorch first (saves ~2GB vs CUDA wheel)

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

### Step 4 — Install all dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Configure environment

```bash
cp .env.example .env
# Edit .env — set these three at minimum:
#   GROQ_API_KEY=gsk_...    ← free at https://console.groq.com
#   API_KEY=any-secret-key
#   VECTOR_BACKEND=lancedb  ← local, no signup required
```

### Step 6 — Start the backend

From the project root (`meis_v6/`), run either:

```bash
# Option A — python main.py (uses uvicorn internally, reads HOST/PORT from .env)
cd backend
python main.py

# Option B — uvicorn directly (from project root, recommended for dev with --reload)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Both options start the app at:
- App UI:      http://localhost:8000
- API docs:    http://localhost:8000/api/docs
- Metrics:     http://localhost:8000/api/metrics

> **Note:** `--reload` (Option B) auto-restarts the server on file changes — useful during development. `python main.py` also supports reload when `DEBUG=true` is set in `.env`.

### Step 7 — Seed sample data (new terminal, venv active)

```bash
# From meis_v6/ root:
python scripts/seed_sample_data.py
# Ingests all files from sample_data/ into the knowledge base
# Wait for "Backend healthy" before proceeding
```

### Step 8 — Run unit tests

```bash
pytest tests/ -v --tb=short
```

### Step 9 — Run evaluation tests (app must be running)

```bash
# In a second terminal with venv active:
pytest tests/evaluation/ -v --tb=short -s
# -s flag shows the per-criterion score output
```

### Step 10 — Query the API manually

```bash
# Replace YOUR_API_KEY with the value from .env
API_KEY="your-api-key-here"

# Health check
curl http://localhost:8000/api/health

# Ask a geological question
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"query": "What is the gold grade at Mount Centauri?", "query_type": "all", "top_k": 5}'

# Evaluate an answer
curl -X POST http://localhost:8000/api/analysis/evaluate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "query": "What is the gold grade at Mount Centauri?",
    "answer": "The inferred resource grades 0.42 g/t Au containing 2.5 Moz gold.",
    "context_chunks": ["An inferred resource of 185 Mt grading 0.42 g/t Au, 2.5 Moz contained."]
  }'

# Check Prometheus metrics
curl http://localhost:8000/api/metrics
```

---

## SECTION 2 — LOCAL DOCKER STACK (backend + Prometheus + Grafana)

### Step 1 — Configure environment

```bash
cp .env.example .env
# Set GROQ_API_KEY, API_KEY (VECTOR_BACKEND=lancedb works for local Docker)
```

### Step 2 — Build and start all services

```bash
docker compose -f docker/docker-compose.yml up --build
# First build takes ~5-10 minutes (downloading PyTorch CPU wheel ~250MB)
# Subsequent builds are fast (cached layers)
```

### Step 3 — Wait for healthy status

```bash
docker compose -f docker/docker-compose.yml ps
# meis-backend should show "healthy"
# App:        http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / admin)
```

### Step 4 — Seed data into the running container

```bash
python scripts/seed_sample_data.py
```

### Step 5 — Run evaluation tests against Docker stack

```bash
pytest tests/evaluation/ -v -s
```

### Step 6 — Open Grafana dashboard

1. Go to http://localhost:3000
2. Login: admin / admin
3. Navigate to Dashboards → MEIS Dashboards → "Mineral Exploration RAG — System Dashboard"
4. The dashboard auto-refreshes every 30s

### Step 7 — Stop the stack

```bash
docker compose -f docker/docker-compose.yml down
# To also delete volumes (resets all data):
docker compose -f docker/docker-compose.yml down -v
```

---

## SECTION 3 — FREE-TIER CLOUD DEPLOYMENT (Render + Qdrant Cloud)

### Pre-deployment checklist

- [ ] Groq API key (free): https://console.groq.com
- [ ] Qdrant Cloud account (free): https://cloud.qdrant.io
- [ ] Render account (free): https://render.com
- [ ] Code pushed to GitHub (with `.gitignore` in place — verify `.env` is NOT tracked)

### Step 1 — Set up Qdrant Cloud (free tier, 5 min)

```
1. Go to https://cloud.qdrant.io
2. Create account → New Cluster → Free tier (1 node, 0.5 vCPU, 1GB RAM)
3. Select region closest to you
4. Copy:  Cluster URL  (e.g. https://abc123.us-east-1-0.aws.cloud.qdrant.io)
          API Key      (from API Keys tab)
```

### Step 2 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial MEIS commit"
git remote add origin https://github.com/YOUR_USERNAME/MEIS-GeoIntel.git
git push -u origin main
```

### Step 3 — Deploy to Render (free web service)

```
1. Go to https://render.com → New → Blueprint
2. Connect your GitHub repo
3. Render detects render.yaml automatically
4. Set secret environment variables in the Render dashboard:
   GROQ_API_KEY    = gsk_...
   API_KEY         = (generate: python -c "import secrets; print(secrets.token_hex(32))")
   QDRANT_URL      = https://your-cluster.cloud.qdrant.io
   QDRANT_API_KEY  = your-qdrant-api-key
   ALLOWED_ORIGINS = ["https://your-render-app.onrender.com"]
5. Click "Apply"
6. First deploy takes ~8-12 min (Docker build + model download)
```

### Step 4 — Seed data to deployed app

```bash
export MEIS_BASE_URL=https://your-app.onrender.com
export MEIS_API_KEY=your-api-key
python scripts/seed_sample_data.py
```

### Step 5 — Run evaluation tests against deployed app

```bash
MEIS_BASE_URL=https://your-app.onrender.com \
MEIS_API_KEY=your-api-key \
pytest tests/evaluation/ -v -s
```

### Step 6 — Monitoring on free tier

**Option A — Manual scrape**
```bash
curl https://your-app.onrender.com/api/metrics
```

**Option B — UptimeRobot (free health monitoring)**
```
https://uptimerobot.com → New Monitor → HTTPS
URL: https://your-app.onrender.com/api/health
Interval: 5 minutes  (also keeps the service warm, preventing cold starts)
```

**Option C — Grafana Cloud (free 10k metrics)**
```
https://grafana.com/products/cloud/ → Start for free
Add Prometheus remote-write URL, then scrape /api/metrics locally and push.
```

---

## SECTION 4 — USEFUL COMMANDS

### Check logs

```bash
# Local Docker
docker compose -f docker/docker-compose.yml logs -f meis-backend

# Local Python
tail -f logs/app.log
```

### Reset the knowledge base

```bash
# Local LanceDB — delete the data directory
rm -rf data/lancedb

# Qdrant Cloud — via API
curl -X DELETE "https://your-cluster.qdrant.io/collections/geological_reports" \
  -H "api-key: your-key"
curl -X DELETE "https://your-cluster.qdrant.io/collections/mineral_datasets" \
  -H "api-key: your-key"
```

### Run only a specific test

```bash
pytest tests/evaluation/test_rag_metrics.py::TestRAGEvaluation::test_evaluation_scores -v -s
pytest tests/evaluation/test_rag_metrics.py::TestHealthAndMetrics -v
```

### Check what's in the knowledge base

```bash
curl http://localhost:8000/api/health | python -m json.tool
```

---

## TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| `RuntimeError: API_KEY is required` | Add `API_KEY=anything` to .env |
| `RuntimeError: GROQ_API_KEY missing` | Get free key at https://console.groq.com |
| Docker build OOM | Set `DOCKER_BUILDKIT=1` or increase Docker RAM to 4GB |
| Render deploy slow | Normal — first deploy downloads ~250MB model files. Subsequent deploys are faster. |
| `No context retrieved` | Run `python scripts/seed_sample_data.py` first |
| Grafana shows "No data" | Generate some traffic first: run seed + tests |
| Port 8000 already in use | `lsof -i :8000` and kill the process, or change PORT in .env |
| uvicorn: `ModuleNotFoundError` | Make sure you run from `meis_v6/` root, not from inside `backend/` |
