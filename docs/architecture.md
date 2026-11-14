# StyleForge v2.0 — Production System Architecture

## 1. High-Level Architecture Overview

StyleForge v2.0 transforms an experimental StyleGAN research project into a multi-tier, production-grade Generative AI platform.

```
                              [ User Web Browser ]
                                       │
                                       ▼
                       [ Nginx Reverse Proxy (Port 80) ]
                        ├── /api/* ──► [ FastAPI Backend (Port 8000) ]
                        └── /*      ──► [ React SPA (Port 3000) ]
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [ Security & Ingress ]                                        [ Persistence Tier ]
     ├── API Key Auth (X-API-Key)                                  ├── PostgreSQL 15 (Metadata & Audit)
     ├── Sliding Rate Limiter                                      └── Local / S3 Storage (Artifacts)
     └── Request Context / Tracing                                             │
                 │                                                             │
                 ▼                                                             │
     [ Async GPU Worker Gateway ]                                              │
     └── Concurrency Semaphore (Bounded VRAM Safety)                          │
                 │                                                             │
                 ▼                                                             │
     [ Modular Inference Engine ]                                              │
     ├── ModelLoader (Raw & EMA StyleGenerators)                              │
     ├── CheckpointManager (SHA256 Integrity Verification)                    │
     ├── Mode Presets (Standard: 1.0, Clean Parity: 0.25)                      │
     └── ImagePostProcessor ([-1, 1] -> RGB uint8 -> Storage) ─────────────────┘
                 │
                 ▼
     [ Observability & Telemetry ]
     ├── Prometheus Exporter (/api/v1/metrics)
     ├── Grafana Dashboard (Port 3001)
     └── MLflow Run Tracker
```

---

## 2. Core Subsystems

### 2.1 Frontend Tier (`frontend/`)
- **Technology Stack:** React 18, TypeScript 5, Vite, modern CSS design system.
- **Visual Aesthetic:** Premium dark studio UI, glassmorphic card overlays, responsive layout.
- **Key Modules:**
  - `DashboardView`: High-level operational status, quick synthesis launch, recent generation feed.
  - `GeneratorView`: Full synthesis control center with noise scale sliders, presets (`Standard` vs `Clean Mode`), seed randomizer, truncation $\psi$ modulation, and live side-by-side comparison.
  - `HistoryView`: Multi-tenant generation audit trail with pagination and artifact retrieval.
  - `ModelInfoView`: Cryptographic SHA-256 verification card, architecture specifications, empirical benchmarks, and explicit native $64\times 64$ scope callout.
  - `SystemStatusView`: Hardware telemetry, live GPU VRAM meter, and streaming Prometheus metrics.

### 2.2 Backend REST API (`backend/`)
- **Framework:** FastAPI with Uvicorn ASGI server and Pydantic validation.
- **Concurrency & Memory Isolation:** Long-running PyTorch inference is isolated from the asyncio event loop using `asyncio.to_thread` and strictly guarded by an `asyncio.Semaphore` to serialize device execution and guarantee zero VRAM exhaustion.
- **Security & Middlewares:**
  - Request ID injection (`X-Request-ID`) and request processing latency timing (`X-Process-Time-Ms`).
  - Sliding-window in-memory rate limiter enforcing per-IP quotas.
  - API Key authentication enforcing `X-API-Key` or `Authorization: Bearer <key>` headers.
  - Defensive path traversal sanitization on all artifact downloads.

### 2.3 Modular Inference Engine (`core/engine/`)
- **`DeviceManager`:** Automatically detects CUDA capability, manages memory placement, tracks VRAM allocation, and issues safe cache flushes (`empty_cache`).
- **`CheckpointManager`:** Computes and verifies streaming SHA-256 digests against the official release checkpoint (`ec8e4bf1e4f21f240f7793a1b3922a4510394831d0c39de521c4cd456c9eed64`).
- **`ModelLoader`:** Loads dual weights (Raw and EMA StyleGenerators) with cleaned module prefixes into evaluation mode (`requires_grad_(False)`).
- **`InferenceEngine`:** Thread-safe singleton providing deterministic seed generation, W-space truncation modulation, controlled noise injection scaling, continuous latent interpolation (LERP/SLERP), and multi-scale style mixing.

### 2.4 Persistence & Storage (`backend/database/` & `backend/storage/`)
- **Relational Database:** PostgreSQL 15 (with SQLite fallback for local developer velocity). SQLAlchemy ORM models track `User`, `ModelVersion`, `Generation`, and `GenerationMetadata`.
- **Artifact Storage Abstraction:** Abstract `AbstractStorageService` with `LocalStorageService` implementation. High-resolution raw image bytes are stored on persistent disk volumes and referenced by sanitized relative keys; image binaries are never stored as database BLOBs.

### 2.5 Observability & MLOps (`backend/metrics/` & `monitoring/`)
- **Metrics Exporter:** Prometheus client library exposing `/api/v1/metrics`. Tracks request throughput, latency histograms, GPU memory utilization, and active worker gauges.
- **Dashboards:** Provisioned Grafana dashboard configured to visualize real-time request rates, P50/P90/P99 latency, and VRAM consumption.
- **Model Registry & Governance:** `ModelRegistry` catalog tracking checkpoint versions, validation metrics, activation state, and rollback mechanisms.
- **MLflow Tracker:** `MLflowTracker` module for logging hyperparameters, quantitative metrics (FID, KID, Laplacian variance, PSD ratios), and artifact images.
