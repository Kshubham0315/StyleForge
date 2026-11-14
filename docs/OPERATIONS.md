# StyleForge v2.0 — Operations & Runbook

This guide contains operational playbooks for monitoring, backup/restore, troubleshooting, and model rollback procedures.

---

## 1. Routine Health Monitoring

StyleForge exposes comprehensive telemetry endpoints:

```bash
# 1. Health Probe
curl -i http://localhost:8000/health

# 2. Model State
curl -i -H "X-API-Key: styleforge-dev-key-2026" http://localhost:8000/api/v1/models/current

# 3. Prometheus Metrics Stream
curl -i http://localhost:8000/api/v1/metrics
```

Key Metrics to Monitor:
- `styleforge_requests_total`: Alert if 5xx rate > 1% over 5m.
- `styleforge_inference_latency_seconds`: Alert if P99 latency > 250ms.
- `styleforge_active_inferences`: Alert if continuously saturated at max concurrency.
- `styleforge_gpu_memory_bytes`: Alert if allocated VRAM exceeds 90% of physical capacity.

---

## 2. Backup & Disaster Recovery

### Database Backup (PostgreSQL)
```bash
docker exec -t styleforge_postgres pg_dump -U styleforge styleforge > backup_$(date +%Y%m%d_%H%M%S).sql
```

### Database Restore
```bash
docker exec -i styleforge_postgres psql -U styleforge -d styleforge < backup_20261114.sql
```

### Artifact Backup
Artifacts are stored in the `styleforge_artifacts` Docker volume (mapped to `./data/artifacts`).
```bash
tar -czvf artifacts_backup_$(date +%Y%m%d).tar.gz ./data/artifacts
```

---

## 3. Model Rollback Playbook

If a newly deployed checkpoint exhibits instability or regression:

1. **Verify Official Checkpoint Integrity:**
   ```bash
   python -c "from core.engine.checkpoint_manager import CheckpointManager, Path; print(CheckpointManager.inspect_checkpoint(Path('checkpoints/styleforge_final_model.pt')))"
   ```
2. **Execute Registry Rollback:**
   ```python
   from core.registry.model_registry import ModelRegistry
   reg = ModelRegistry()
   reg.rollback("v1.0.0")
   ```
3. **Restart Backend Service:**
   ```bash
   docker compose restart backend
   ```

---

## 4. Troubleshooting Playbook

| Symptom | Cause | Solution |
|---|---|---|
| `HTTP 401 Unauthorized` | Missing or mismatched API key | Pass `X-API-Key: styleforge-dev-key-2026` in headers or set `ENABLE_AUTH=false` in `.env`. |
| `HTTP 429 Too Many Requests` | Rate limit threshold exceeded | Check `RATE_LIMIT_PER_MINUTE` in `.env` (default: 120 req/min) or pause client traffic for 60s. |
| `HTTP 503 Model Not Ready` | Checkpoint missing or corrupted | Ensure `checkpoints/styleforge_final_model.pt` exists and matches SHA-256 `ec8e4bf1e4f21f240f7793a1b3922a4510394831d0c39de521c4cd456c9eed64`. |
| High Generation Latency | GPU fallback to CPU | Run `nvidia-smi` and verify NVIDIA container toolkit runtime configuration. |
