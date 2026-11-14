"""Prometheus observability metrics for StyleForge Generative Platform."""

import logging
from typing import Optional
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
import torch

logger = logging.getLogger(__name__)

# Request counters
REQUEST_COUNT = Counter(
    "styleforge_requests_total",
    "Total incoming HTTP API requests",
    ["method", "endpoint", "status"],
)

GENERATE_COUNT = Counter(
    "styleforge_generations_total",
    "Total facial samples synthesized",
    ["mode", "use_ema"],
)

# Inference Latency Histogram
INFERENCE_LATENCY = Histogram(
    "styleforge_inference_latency_seconds",
    "Time consumed by neural network forward pass",
    ["mode"],
    buckets=[0.005, 0.010, 0.020, 0.050, 0.100, 0.250, 0.500, 1.000],
)

# Concurrency & GPU Gauges
ACTIVE_INFERENCES = Gauge(
    "styleforge_active_inferences",
    "Number of inference requests currently executing on device",
)

GPU_MEMORY_BYTES = Gauge(
    "styleforge_gpu_memory_bytes",
    "Current physical GPU allocated memory in bytes",
)


def update_gpu_metrics() -> None:
    """Updates GPU telemetry gauge if CUDA is active."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        GPU_MEMORY_BYTES.set(allocated)


def get_metrics_output() -> bytes:
    """Generates Prometheus formatted scrape payload."""
    update_gpu_metrics()
    return generate_latest()
