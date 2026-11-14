# StyleForge v2.0 — Production Deployment Guide

This guide details deployment procedures for StyleForge v2.0 across local development, multi-container Docker environments, and cloud GPU instances.

---

## 1. Quickstart (Local Development)

### Prerequisites
- Python 3.10+
- Node.js 20+ & pnpm
- NVIDIA GPU with CUDA 12.1+ (Optional, CPU fallback supported)

### Step 1: Install Backend Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
pip install fastapi uvicorn[standard] pydantic pydantic-settings sqlalchemy alembic asyncpg psycopg2-binary prometheus_client redis mlflow
```

### Step 2: Launch Backend API
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at: `http://localhost:8000/docs`

### Step 3: Launch Frontend Application
```bash
cd frontend
pnpm install
pnpm run dev
```
Frontend UI will be available at: `http://localhost:3000`

---

## 2. Multi-Container Docker Deployment

The multi-container configuration deploys:
1. `backend`: FastAPI with Uvicorn (Port 8000)
2. `frontend`: React SPA served by Nginx Alpine (Port 3000)
3. `postgres`: PostgreSQL 15 database (Port 5432)
4. `redis`: Redis 7 in-memory cache (Port 6379)
5. `prometheus`: Prometheus metrics scraper (Port 9090)
6. `grafana`: Preconfigured Grafana telemetry dashboard (Port 3001)
7. `nginx`: Global reverse proxy (Port 80)

### CPU Deployment (Default)
```bash
docker compose up -d --build
```

### GPU-Accelerated Deployment (NVIDIA Container Toolkit)
For hosts with NVIDIA drivers and the NVIDIA Container Toolkit installed:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

---

## 3. Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `ENV` | `development` | Deployment environment: `development`, `production`, `test` |
| `DATABASE_URL` | `sqlite:///./data/styleforge.db` | PostgreSQL connection string in production |
| `STORAGE_DIR` | `./data/artifacts` | Persistent directory for synthesized image binaries |
| `API_KEY` | `styleforge-dev-key-2026` | Master authentication key for backend API access |
| `ENABLE_AUTH` | `true` | Enforce API key verification on protected endpoints |
| `RATE_LIMIT_PER_MINUTE` | `120` | Maximum requests per minute per IP address |
| `CHECKPOINT_PATH` | `checkpoints/styleforge_final_model.pt` | Path to production generator weights |
| `MAX_BATCH_SIZE` | `16` | Maximum allowed samples per single generation request |
| `GPU_CONCURRENCY_LIMIT` | `1` | Maximum parallel GPU tasks (guards against VRAM spikes) |
| `MLFLOW_TRACKING_URI` | `file:./data/mlruns` | URI for remote or local MLflow tracking server |

---

## 4. Verification & Health Probes

Verify deployment readiness:
```bash
# Check service health
curl -f http://localhost/health

# Check readiness probe
curl -f http://localhost/ready

# Check Prometheus metrics
curl -f http://localhost/api/v1/metrics
```

Access services:
- **Application Studio:** `http://localhost` (or `http://localhost:3000`)
- **Interactive Swagger Docs:** `http://localhost/docs` (or `http://localhost:8000/docs`)
- **Prometheus Metrics:** `http://localhost:9090`
- **Grafana Telemetry:** `http://localhost:3001` (Credentials: `admin` / `admin`)
