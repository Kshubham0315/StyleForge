# StyleForge v2.0 — Production Deployment & Operations Runbook

**Document Version:** 2.0.0  
**Effective Date:** 2026-11-14  
**Audience:** Site Reliability Engineers (SRE), DevOps Engineers, Platform Administrators  
**Classification:** Operational Guide  

---

## 1. System Architecture Overview

StyleForge v2.0 is a modular, containerized Generative AI platform for real-time facial synthesis:

```
[ Clients / Web Browser ]
         │
         ▼
[ Reverse Proxy / Ingress ] (Port 80 / 443 - Nginx)
   ├───► Static Assets / SPA ─────────► [ Vite Frontend ] (Port 5173 / Nginx Dist)
   └───► API Requests (/api/*) ────────► [ FastAPI Backend ] (Port 8000)
                                             │
                                             ├──► [ PyTorch Inference Engine ] ◄── Checkpoint (SHA-256)
                                             │       (NVIDIA GPU / CUDA 12.1)
                                             ├──► [ Metadata Store ] (SQLite / PostgreSQL)
                                             ├──► [ Artifact Storage ] (Local Disk / S3 Volume)
                                             └──► [ Prometheus Exporter ] (/metrics)
                                                     │
                                                     ▼
                                          [ Prometheus Server ] (Port 9090)
                                                     │
                                                     ▼
                                          [ Grafana Dashboards ] (Port 3000)
```

---

## 2. Infrastructure & Prerequisites

### 2.1 Hardware Requirements
| Resource | Minimum (CPU Fallback) | Recommended (Production GPU) |
|---|---|---|
| **CPU** | 4 Cores (x86_64) | 8+ Cores (Intel / AMD) |
| **RAM** | 8 GB | 16+ GB |
| **GPU** | Not required | NVIDIA GPU with $\ge 4\text{ GB}$ VRAM (RTX 3060+, RTX 4050+, T4, L4, A10) |
| **Disk** | 10 GB free space | 50+ GB SSD (NVMe preferred for checkpoints & artifacts) |

### 2.2 Software Requirements
- **OS:** Linux (Ubuntu 22.04 LTS / Debian 12 / RHEL 9) or Windows 11 / Windows Server 2022
- **Python:** 3.10 or 3.11
- **Node.js:** v20.x or v22.x LTS (with `pnpm` or `npm`)
- **CUDA / cuDNN:** CUDA 12.1 or 12.4 with matching PyTorch build
- **Docker:** Docker Engine 24.0+ and Docker Compose v2.20+
- **NVIDIA Container Toolkit:** Required for GPU access inside containers (`nvidia-smi` accessible inside container)

---

## 3. Deployment Method A — Native Host Deployment (Bare Metal / VM)

### Step 1: Clone and Prepare Environment
```bash
git clone https://github.com/Kshubham0315/StyleForge.git
cd StyleForge

# Create Python virtual environment
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Verify Production Checkpoint
Ensure the production model checkpoint is present and matches the official SHA-256 hash:
```bash
python -c "from core.engine.checkpoint_manager import CheckpointManager, PRODUCTION_CHECKPOINT_SHA256; from pathlib import Path; p = Path('checkpoints/styleforge_final_model.pt'); h = CheckpointManager.compute_sha256(p); print('Hash:', h); assert h == PRODUCTION_CHECKPOINT_SHA256, 'Hash mismatch!'"
```

### Step 3: Configure Environment Variables
Create or edit `.env` in the project root:
```ini
ENV=production
APP_NAME=StyleForge
APP_VERSION=2.0.0
DEBUG=false
HOST=0.0.0.0
PORT=8000
API_KEY=YOUR_SECURE_STRONG_PRODUCTION_SECRET_KEY_HERE
DEVICE=cuda
GPU_CONCURRENCY_LIMIT=1
RATE_LIMIT_PER_MINUTE=120
DATABASE_URL=sqlite:///./data/styleforge.db
ARTIFACT_STORAGE_PATH=./data/artifacts
CORS_ORIGINS=["https://styleforge.yourcompany.com"]
```

### Step 4: Build and Serve Frontend SPA
```bash
cd frontend
pnpm install
pnpm run build
cd ..
```
The compiled bundle will reside in `frontend/dist/`.

### Step 5: Start Backend Service
Run via Uvicorn with production worker configuration:
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
```

---

## 4. Deployment Method B — Containerized Deployment (Docker Compose)

### Step 1: Standard CPU Container Deployment
```bash
# Build and start services in background
docker compose -f docker-compose.yml up -d --build

# View container status
docker compose ps
```

### Step 2: GPU-Accelerated Container Deployment (Production Recommended)
Prerequisites: Host must have NVIDIA drivers and `nvidia-container-toolkit` installed.
```bash
# Verify host Docker GPU accessibility
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Launch StyleForge with GPU acceleration
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### Step 3: Verify Running Services
```bash
# Check service logs
docker compose logs -f backend
docker compose logs -f frontend
```

---

## 5. Health Checks & Verification Procedures

### 5.1 Service Liveness & Readiness Probes
1. **Liveness Check:**
   ```bash
   curl -s http://localhost:8000/health | jq .
   ```
   *Expected Response:*
   ```json
   {
     "status": "healthy",
     "uptime_seconds": 12.4,
     "version": "2.0.0",
     "device": "cuda",
     "cuda_available": true,
     "database_connected": true,
     "models_loaded": true
   }
   ```

2. **Readiness Check (Kubernetes probe):**
   ```bash
   curl -s -i http://localhost:8000/ready
   ```
   *Expected Response:* `HTTP/1.1 200 OK`, body: `{"status": "ready"}`.

3. **Inference Smoke Test:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/generate \
     -H "Content-Type: application/json" \
     -H "X-API-Key: YOUR_SECURE_STRONG_PRODUCTION_SECRET_KEY_HERE" \
     -d '{"num_samples": 1, "seed": 42, "mode": "standard"}' | jq .
   ```
   *Expected Response:* `status: "completed"`, `latency_ms < 200`, `images` array with 1 item.

4. **Metrics Endpoint:**
   ```bash
   curl -s http://localhost:8000/metrics | grep styleforge_
   ```

---

## 6. Observability & Monitoring Setup

### 6.1 Prometheus Configuration
StyleForge exposes Prometheus metrics on both `/metrics` and `/api/v1/metrics`.
Prometheus scrape target in `monitoring/prometheus/prometheus.yml`:
```yaml
scrape_configs:
  - job_name: "styleforge-api"
    scrape_interval: 5s
    metrics_path: "/api/v1/metrics"
    static_configs:
      - targets: ["backend:8000"]
```

### 6.2 Key Prometheus Metrics
| Metric Name | Type | Description | Alert Threshold |
|---|---|---|---|
| `styleforge_active_inferences` | Gauge | Currently executing GPU inference calls | $> 4$ sustained |
| `styleforge_inference_duration_seconds` | Histogram | Forward-pass generation latency by mode | P95 $> 0.5\text{s}$ |
| `styleforge_requests_total` | Counter | Total API requests by method, route, and status | Status $500$ rate $> 1\%$ |
| `styleforge_generated_faces_total` | Counter | Cumulative face synthesis count | N/A |
| `styleforge_gpu_memory_bytes` | Gauge | Current allocated GPU VRAM | $> 5.0\text{ GB}$ (on 6GB card) |

### 6.3 Grafana Dashboard
Import dashboard definition from:
`monitoring/grafana/dashboards/styleforge_dashboard.json`
- Access Grafana at `http://localhost:3000` (Default credentials: `admin` / `admin`).
- Add Prometheus data source pointing to `http://prometheus:9090`.

---

## 7. Common Operational Incidents & Troubleshooting

### Incident 1: `CUDA out of memory` (OOM)
- **Symptom:** API returns HTTP 500 with `CUDA out of memory` log.
- **Root Cause:** Excessive concurrent batch sizes or uncollected PyTorch cache.
- **Remediation:**
  1. Confirm `GPU_CONCURRENCY_LIMIT=1` in `.env`.
  2. Inspect other host processes consuming VRAM using `nvidia-smi`.
  3. Restart backend service: `docker compose restart backend`.

### Incident 2: Checkpoint Missing or Corrupt
- **Symptom:** Startup fails with `CheckpointError: StyleForge checkpoint not found` or SHA-256 mismatch.
- **Remediation:**
  1. Verify file exists at `checkpoints/styleforge_final_model.pt`.
  2. Check hash: `ec8e4bf1e4f21f240f7793a1b3922a4510394831d0c39de521c4cd456c9eed64`.
  3. Re-export or download from official release: `python scripts/export_final_model.py`.

### Incident 3: Rate Limiting Alert (`HTTP 429`)
- **Symptom:** Clients report `RATE_LIMIT_EXCEEDED`.
- **Root Cause:** Client application burst exceeding `RATE_LIMIT_PER_MINUTE`.
- **Remediation:**
  1. If legitimate high-volume client, adjust `RATE_LIMIT_PER_MINUTE` in `.env`.
  2. Implement client-side exponential backoff based on `retry_after_seconds`.

### Incident 4: Port 8000 or 5173 Already in Use
- **Symptom:** `OSError: [Errno 98] Address already in use`.
- **Remediation:**
  1. Check occupying process: `netstat -ano | findstr 8000` (Windows) or `lsof -i :8000` (Linux).
  2. Terminate rogue process or modify `PORT` in `.env`.

---

## 8. Backup & Disaster Recovery Procedures

1. **Database Backup (SQLite):**
   ```bash
   sqlite3 ./data/styleforge.db ".backup ./backups/styleforge_$(date +%Y%m%d_%H%M%S).db"
   ```
2. **Artifact Backup:**
   Rsync or rclone the artifacts directory to persistent backup storage:
   ```bash
   rsync -avz ./data/artifacts/ /backup_storage/styleforge_artifacts/
   ```
3. **Rollback Strategy:**
   If a newly deployed version experiences defects:
   ```bash
   git checkout v1.0.1
   docker compose up -d --build
   ```
