"""Sliding window rate limiter middleware and dependency."""

import time
from collections import defaultdict
from typing import Dict, List
from fastapi import HTTPException, Request, status

from backend.config import settings


class InMemoryRateLimiter:
    """Sliding-window in-memory rate limiter per IP address."""

    def __init__(self, limit_per_minute: int = 120):
        self.limit = limit_per_minute
        self.window_seconds = 60
        self.clients: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> tuple[bool, int]:
        now = time.time()
        cutoff = now - self.window_seconds
        # Clean older timestamps
        timestamps = [t for t in self.clients[client_ip] if t > cutoff]
        self.clients[client_ip] = timestamps

        if len(timestamps) >= self.limit:
            remaining = 0
            return False, remaining

        self.clients[client_ip].append(now)
        remaining = self.limit - len(self.clients[client_ip])
        return True, remaining


rate_limiter = InMemoryRateLimiter(settings.RATE_LIMIT_PER_MINUTE)


async def check_rate_limit(request: Request) -> None:
    """FastAPI dependency enforcing request frequency limits."""
    client_ip = request.client.host if request.client else "unknown"
    allowed, remaining = rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "RATE_LIMIT_EXCEEDED",
                "message": f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_PER_MINUTE} requests per minute.",
                "retry_after_seconds": 60,
            },
        )
