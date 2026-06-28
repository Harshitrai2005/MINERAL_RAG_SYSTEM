# Deployment Guide — Mineral Exploration Intelligence System

Two sections:
1. **Run locally on your laptop** (5 minutes, zero accounts needed)
2. **Deploy live on the internet for free** (15 minutes, 3 free accounts)

---

## PART 1 — Run on your laptop

### Prerequisites
- Python 3.11 or 3.12
- A free Groq API key: https://console.groq.com (no credit card)

### Steps

```bash
# 1. Clone the repo and enter the project folder
git clone https://github.com/YOUR_USERNAME/MEIS-GeoIntel.git
cd MEIS-GeoIntel/meis_v6

# 2. Create a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 3. Install dependencies (torch first saves ~2GB of CUDA wheels)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Set your keys
cp .env.example .env
# Edit .env and set at minimum:
#   GROQ_API_KEY=gsk_your_key_here
#   API_KEY=any_strong_random_string
#   VECTOR_BACKEND=lancedb

# 5. Start the server — two equivalent options:

# Option A: python main.py (from inside backend/)
cd backend && python main.py

# Option B: uvicorn directly (from meis_v6/ root — preferred for --reload in dev)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 6. Seed sample geological data (new terminal, from meis_v6/)
python scripts/seed_sample_data.py

# 7. Open your browser
#    App:          http://localhost:8000
#    Swagger docs: http://localhost:8000/api/docs
#    Metrics:      http://localhost:8000/api/metrics
#    Health:       http://localhost:8000/api/health
```

The first startup downloads the embedding model (~90MB) — subsequent starts are instant.

---

## PART 2 — Free live deployment (Render + Qdrant Cloud)

### Why two services?
- **Render** hosts the FastAPI app (free web service tier)
- **Qdrant Cloud** stores the vectors (free 1GB cluster, data persists forever)

Render's free tier has an **ephemeral filesystem** — anything written to local disk is wiped on restart. That's why we use Qdrant Cloud for the vector DB when deployed: your uploaded documents survive restarts and redeploys.

### Accounts you need (all free, no credit card required)
1. **GitHub** — https://github.com
2. **Render** — https://render.com
3. **Qdrant Cloud** — https://cloud.qdrant.io

---

### Step 1 — Push code to GitHub

```bash
# In meis_v6/ (make sure .env is in .gitignore before this!)
git init
git add .
git commit -m "Initial commit — MEIS RAG system"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/MEIS-GeoIntel.git
git branch -M main
git push -u origin main
```

> **Security check:** run `git status` and confirm `.env` does NOT appear in the list.

---

### Step 2 — Create a free Qdrant Cloud cluster

1. Go to https://cloud.qdrant.io and sign up (free, no card)
2. Click **"Create cluster"**
3. Choose: Name: `mineral-rag`, Cloud: AWS or GCP, Plan: **Free** (1GB)
4. Click **Create** and wait ~60 seconds
5. Copy the **Cluster URL** and **API Key** (API Keys tab → Create)

---

### Step 3 — Deploy on Render

1. Go to https://render.com → sign up (use GitHub login)
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub repo — Render auto-detects `render.yaml`
4. Set these **Environment Variables** in the Render dashboard (Environment tab):

   | Key | Value |
   |-----|-------|
   | `GROQ_API_KEY` | from https://console.groq.com |
   | `API_KEY` | `python -c "import secrets; print(secrets.token_hex(32))"` |
   | `VECTOR_BACKEND` | `qdrant` |
   | `QDRANT_URL` | your Qdrant cluster URL |
   | `QDRANT_API_KEY` | your Qdrant API key |
   | `ALLOWED_ORIGINS` | `["https://your-app.onrender.com"]` |

5. Click **"Apply"** — first build takes 8–12 minutes (Docker + model download)

6. Verify:
   - `https://your-app.onrender.com/api/health` → `{"status":"healthy"}`
   - `https://your-app.onrender.com` → full UI loads

---

### Step 4 — Seed data into the live deployment (optional)

```bash
export MEIS_BASE_URL=https://your-app.onrender.com
export MEIS_API_KEY=your-api-key
python scripts/seed_sample_data.py
```

Or just upload files through the live UI — it works identically.

---

## Free tier limits

| Service | Free limit | What happens |
|---------|-----------|-------------|
| Render web service | Sleeps after 15min inactivity | First request after sleep takes ~30s |
| Render bandwidth | 100GB/month | Unlikely to hit for a demo |
| Qdrant Cloud | 1GB storage | ~500k chunks of 384-dim vectors |
| Groq API | 14,400 req/day free | Returns 429 — the retry logic handles it |

### The "sleeping" question in interviews
> "Free tier Render services sleep after inactivity. The first request after sleep takes ~30 seconds while ML models reload. For a production deployment I'd use a paid tier or configure UptimeRobot (free) to ping `/api/health` every 5 minutes to keep it warm. The architecture itself is stateless — FastAPI + external Qdrant — so adding workers is a one-line config change."

---

## Switching vector backends

```bash
# Local dev (default — no accounts, data in ./data/lancedb)
VECTOR_BACKEND=lancedb

# Production (Qdrant Cloud — data survives restarts)
VECTOR_BACKEND=qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
```

That is the **entire switch** — zero code changes. This is the Dependency Inversion Principle in practice: the app depends on the `VectorRepository` interface, not on LanceDB or Qdrant directly.

---

## Useful URLs once deployed

| URL | What it shows |
|-----|--------------|
| `/` | Main application UI |
| `/api/docs` | Swagger — interactive API docs, test every endpoint |
| `/api/redoc` | ReDoc — clean API reference |
| `/api/health` | Liveness probe — used by Render's health checker |
| `/api/ready` | Readiness probe — vector DB connectivity + doc counts |
| `/api/metrics` | Prometheus metrics scrape endpoint |
