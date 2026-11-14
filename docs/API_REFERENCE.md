# StyleForge v2.0 — REST API Reference

Base URL: `http://localhost:8000` (Direct) or `http://localhost` (Nginx Proxy)  
Version: `v1`  
Authentication: `X-API-Key: <your_key>` or `Authorization: Bearer <your_key>`  
Default Development Key: `styleforge-dev-key-2026`

---

## 1. System & Health Endpoints

### `GET /health`
Returns system operational state and runtime hardware telemetry.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "uptime_seconds": 124.5,
  "version": "2.0.0",
  "device": "cuda:0",
  "cuda_available": true,
  "database_connected": true,
  "models_loaded": true
}
```

### `GET /ready`
Readiness probe for container orchestrators (Docker / Kubernetes).

**Response (200 OK):**
```json
{
  "status": "ready"
}
```

### `GET /api/v1/metrics`
Prometheus metrics scrape target. Returns standard OpenMetrics formatted plain text.

---

## 2. Model Catalog Endpoints

### `GET /api/v1/models/current`
Returns active generative model specifications, hardware telemetry, and supported inference modes.

**Response (200 OK):**
```json
{
  "name": "StyleForge Production Face Generator",
  "version": "v1.0.0",
  "checkpoint_sha256": "ec8e4bf1e4f21f240f7793a1b3922a4510394831d0c39de521c4cd456c9eed64",
  "resolution": 64,
  "latent_dim": 512,
  "style_dim": 512,
  "device": "cuda:0",
  "is_official": true,
  "modes_supported": [
    "standard (noise 1.0)",
    "smooth (noise 0.25)",
    "custom"
  ],
  "vram_telemetry": {
    "cuda_available": true,
    "device_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
    "allocated_mb": 184.2,
    "reserved_mb": 240.0,
    "total_mb": 6144.0,
    "free_mb": 5959.8
  }
}
```

### `GET /api/v1/models`
Returns list of registered model catalog items.

---

## 3. Face Synthesis Endpoints

### `POST /api/v1/generate`
Synthesizes photographic human faces from latent space.

**Request Body:**
```json
{
  "seed": 2026,
  "num_samples": 4,
  "truncation_psi": 0.8,
  "mode": "smooth",
  "noise_scale": 0.25,
  "use_ema": false,
  "clamp_adain": false,
  "output_format": "png"
}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `seed` | `int \| null` | `null` | Deterministic random seed ($0 \le \text{seed} \le 2^{31}-1$). Set null for random sampling. |
| `num_samples` | `int` | `1` | Number of faces to generate ($1 \le N \le 16$). |
| `truncation_psi` | `float` | `0.8` | Truncation coefficient ($0.0 \le \psi \le 1.0$). $\psi < 1.0$ trades diversity for quality. |
| `mode` | `string` | `"standard"` | Preset mode: `"standard"` (noise 1.0), `"smooth"` (noise 0.25), or `"custom"`. |
| `noise_scale` | `float` | `1.0` | Injected stochastic noise multiplier ($0.0 \le \text{scale} \le 1.0$). Set to `0.25` for spectral parity. |
| `use_ema` | `bool` | `false` | Generate with Exponential Moving Average weights. (Raw weights achieve lowest FID: 294.75). |
| `clamp_adain` | `bool` | `false` | Experimental AdaIN clamping. Keep `false` to prevent diversity loss. |
| `output_format` | `string` | `"png"` | Target encoding format: `"png"`, `"jpeg"`, or `"webp"`. |

**Response (200 OK):**
```json
{
  "generation_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "completed",
  "num_samples": 4,
  "seed": 2026,
  "mode": "smooth",
  "noise_scale": 0.25,
  "use_ema": false,
  "latency_ms": 35.8,
  "latency_per_face_ms": 8.95,
  "resolution": "64x64",
  "images": [
    {
      "index": 0,
      "base64_data": "data:image/png;base64,iVBORw0KGgoAAA...",
      "download_url": "/api/v1/artifacts/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d_0.png",
      "artifact_path": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d_0.png"
    }
  ],
  "created_at": "2026-11-14T11:45:00.000Z"
}
```

---

## 4. History & Artifact Endpoints

### `GET /api/v1/generations`
Retrieves paginated audit log of face generations.

**Query Parameters:**
- `limit` (int, default `20`, max `100`): Results per page.
- `offset` (int, default `0`): Pagination offset.
- `mode` (string, optional): Filter by mode (`"standard"`, `"smooth"`, `"custom"`).

### `GET /api/v1/generations/{generation_id}`
Retrieves detailed metadata for a specific generation UUID.

### `GET /api/v1/artifacts/{filename}`
Securely streams stored image binary. Includes strict path sanitization against directory traversal.

---

## 5. Standard Error Format

All error responses adhere to a consistent JSON structure:

```json
{
  "error": {
    "code": "INVALID_API_KEY",
    "message": "The provided API key is invalid or deactivated.",
    "request_id": "3f9820f1-4f12-4ebc-b3a2-962649b1de12"
  }
}
```
