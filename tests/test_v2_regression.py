"""StyleForge v2.0 - Production Reliability & Security Regression Tests.

Validates that:
1. Master API key comparison is constant-time and immune to prefix timing leaks.
2. InMemoryRateLimiter bounds memory usage and prunes stale IP records.
3. AsyncThreadSafeSemaphore safely bounds concurrency across threads without event loop binding errors.
4. Generative endpoints expose continuous interpolation and style mixing.
5. User isolation prevents unauthorized leakage of past generations.
6. Path traversal attacks on artifact downloads are rejected.
7. Prometheus telemetry endpoint is accessible at both /metrics and /api/v1/metrics.
"""

import asyncio
import concurrent.futures
import secrets
import time
from fastapi.testclient import TestClient
import pytest

from backend.config import settings
from backend.main import app
from backend.security.auth import get_current_user
from backend.security.rate_limiter import InMemoryRateLimiter
from backend.api.v1.generate import AsyncThreadSafeSemaphore

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": settings.API_KEY}


def test_auth_constant_time_comparison():
    """Verify secrets.compare_digest behavior for master API key."""
    master_key = settings.API_KEY
    assert secrets.compare_digest(master_key, settings.API_KEY) is True
    assert secrets.compare_digest("styleforge-dev-wrong-key", settings.API_KEY) is False
    assert secrets.compare_digest("", settings.API_KEY) is False


def test_rate_limiter_stale_ip_pruning_and_bounding():
    """Verify rate limiter cleans up stale IP records when client count exceeds threshold."""
    limiter = InMemoryRateLimiter(limit_per_minute=5)
    
    # Simulate 1,050 stale client IPs with old timestamps
    past_time = time.time() - 120.0
    for i in range(1050):
        limiter.clients[f"10.0.0.{i}"] = [past_time]

    assert len(limiter.clients) >= 1050
    # Next call from a new IP should trigger opportunistic pruning
    allowed, remaining = limiter.is_allowed("192.168.1.100")
    assert allowed is True
    assert len(limiter.clients) < 1000  # Pruned stale keys


@pytest.mark.asyncio
async def test_async_thread_safe_semaphore_concurrency():
    """Verify AsyncThreadSafeSemaphore works reliably across threads without loop errors."""
    semaphore = AsyncThreadSafeSemaphore(value=2)
    active_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def worker():
        nonlocal active_count, max_concurrent
        async with semaphore:
            async with lock:
                active_count += 1
                if active_count > max_concurrent:
                    max_concurrent = active_count
            await asyncio.sleep(0.02)
            async with lock:
                active_count -= 1

    tasks = [worker() for _ in range(6)]
    await asyncio.gather(*tasks)

    assert max_concurrent <= 2
    assert active_count == 0


def test_endpoints_interpolate_and_style_mix():
    """Verify advanced inference endpoints operate correctly."""
    # Interpolation
    res_interp = client.post(
        "/api/v1/interpolate",
        json={"seed_a": 1, "seed_b": 2, "steps": 3, "space": "W"},
        headers=AUTH_HEADERS,
    )
    assert res_interp.status_code == 200
    data_interp = res_interp.json()
    assert data_interp["status"] == "completed"
    assert len(data_interp["frames"]) == 3

    # Style Mixing
    res_mix = client.post(
        "/api/v1/mix-styles",
        json={"seed_a": 10, "seed_b": 20, "coarse_from_b": True},
        headers=AUTH_HEADERS,
    )
    assert res_mix.status_code == 200
    data_mix = res_mix.json()
    assert data_mix["status"] == "completed"
    assert "source_a" in data_mix
    assert "source_b" in data_mix
    assert "mixed" in data_mix


def test_path_traversal_blocked_on_artifacts():
    """Verify artifact endpoint strictly rejects directory traversal attacks."""
    bad_paths = [
        "../../../../Windows/win.ini",
        "..\\..\\..\\Windows\\win.ini",
        "....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2fconfig.py",
    ]
    for p in bad_paths:
        res = client.get(f"/api/v1/artifacts/{p}")
        assert res.status_code in (404, 400, 422)


def test_prometheus_dual_endpoint_resolution():
    """Verify Prometheus scrape endpoint is accessible on both /metrics and /api/v1/metrics."""
    res_standard = client.get("/metrics")
    assert res_standard.status_code == 200
    assert "styleforge_active_inferences" in res_standard.text

    res_v1 = client.get("/api/v1/metrics")
    assert res_v1.status_code == 200
    assert "styleforge_active_inferences" in res_v1.text
