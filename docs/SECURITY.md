# StyleForge v2.0 — Security & Threat Model

This document outlines the security architecture, authentication protocols, defensive controls, and vulnerability mitigation implemented across StyleForge v2.0.

---

## 1. Authentication & Authorization

- **API Key Governance:** All mutation and generation endpoints require authentication via `X-API-Key` or `Authorization: Bearer <token>`.
- **Hashed Storage:** API keys in PostgreSQL are hashed using SHA-256 with constant-time equality comparisons to prevent timing attacks.
- **Configurable Dev Bypass:** In non-production environments (`ENABLE_AUTH=false`), authentication can be disabled for local development velocity.

---

## 2. Rate Limiting & Resource Protection

- **Sliding-Window Limiter:** Each client IP address is restricted to a configurable limit (default: 120 requests/minute) to mitigate Denial of Service (DoS) and brute-force attempts.
- **Bounded GPU Concurrency:** Generative forward passes are managed through an `asyncio.Semaphore(GPU_CONCURRENCY_LIMIT=1)`. Requests beyond this threshold queue gracefully, preventing CUDA Out-Of-Memory (OOM) fatal crashes.
- **Input Boundary Validation:** Pydantic validators enforce strict parameter limits:
  - `num_samples`: $1 \le N \le 16$
  - `truncation_psi`: $0.0 \le \psi \le 1.0$
  - `noise_scale`: $0.0 \le \text{scale} \le 1.0$
  - `seed`: $0 \le \text{seed} \le 2^{31}-1$

---

## 3. Storage Security & Path Traversal Prevention

- **Sanitized Filenames:** Generated artifact filenames are constructed using randomly generated UUIDs: `{generation_id}_{index}.{ext}`.
- **Path Confinement:** The `LocalStorageService` resolves all download paths and verifies that `target_path.startswith(base_storage_dir)`. Directory traversal attempts (e.g. `../../etc/passwd`) are rejected and logged.
- **Binary Isolation:** Raw image binaries are stored strictly in the persistent artifact volume; image data is never placed directly into relational database tables as raw BLOBs.

---

## 4. Error Handling & Information Disclosure

- **Sanitized Error Responses:** Internal stack traces, raw file paths, and database query parameters are stripped from user-facing error payloads.
- **Unique Request IDs:** Every HTTP transaction is assigned a unique `X-Request-ID` UUID returned in response headers and logged for audit correlation.
- **No Hardcoded Secrets:** Production credentials, database passwords, and API keys are injected exclusively through environment variables.
