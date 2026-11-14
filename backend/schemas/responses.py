"""Pydantic API response models and schema definitions."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ImageItem(BaseModel):
    """Metadata and base64 encoded image."""
    index: int
    base64_data: str
    download_url: str
    artifact_path: str


class GenerateResponse(BaseModel):
    """Response returned upon completion of face generation."""
    generation_id: str
    status: str
    num_samples: int
    seed: Optional[int]
    mode: str
    noise_scale: float
    use_ema: bool
    latency_ms: float
    latency_per_face_ms: float
    resolution: str
    images: List[ImageItem]
    created_at: str


class GenerationListItem(BaseModel):
    """Summary item for paginated history listings."""
    id: str
    created_at: str
    num_samples: int
    seed: Optional[int]
    mode: str
    noise_scale: float
    use_ema: bool
    latency_ms: float
    download_url: Optional[str]


class GenerationListResponse(BaseModel):
    """Paginated collection of past generations."""
    total: int
    limit: int
    offset: int
    items: List[GenerationListItem]


class ModelInfoResponse(BaseModel):
    """Metadata regarding currently active generative model."""
    name: str
    version: str
    checkpoint_sha256: str
    resolution: int
    latent_dim: int
    style_dim: int
    device: str
    is_official: bool
    modes_supported: List[str]
    vram_telemetry: Dict[str, Any]


class HealthResponse(BaseModel):
    """Service health and readiness check."""
    status: str
    uptime_seconds: float
    version: str
    device: str
    cuda_available: bool
    database_connected: bool
    models_loaded: bool
