"""Model management and catalog endpoints."""

from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.repository import ModelRepository
from backend.database.session import get_db
from backend.schemas.responses import ModelInfoResponse
from core.engine.checkpoint_manager import CheckpointManager
from core.engine.device_manager import DeviceManager
from core.engine.inference_engine import InferenceEngine

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/current", response_model=ModelInfoResponse, summary="Get currently active model metadata")
def get_current_model():
    """Returns runtime telemetry and metadata of currently active model."""
    engine = InferenceEngine.get_instance(checkpoint_path=settings.CHECKPOINT_PATH, device=settings.DEVICE)
    meta = engine.checkpoint_meta
    vram = DeviceManager.get_vram_info(engine.device)

    return ModelInfoResponse(
        name=meta.get("model_name", "StyleForge"),
        version="v1.0.0",
        checkpoint_sha256=meta.get("sha256", "N/A"),
        resolution=meta.get("img_size", 64),
        latent_dim=meta.get("latent_dim", 512),
        style_dim=meta.get("style_dim", 512),
        device=str(engine.device),
        is_official=meta.get("is_official", False),
        modes_supported=["standard (noise 1.0)", "smooth (noise 0.25)", "custom"],
        vram_telemetry=vram,
    )


@router.get("", summary="List all registered model catalog versions")
def list_models(db: Session = Depends(get_db)):
    """Lists registered model catalog items."""
    models = ModelRepository.list_all(db)
    return [
        {
            "id": m.id,
            "name": m.name,
            "version_tag": m.version_tag,
            "checkpoint_sha256": m.checkpoint_sha256,
            "resolution": f"{m.resolution}x{m.resolution}",
            "latent_dim": m.latent_dim,
            "is_active": m.is_active,
            "created_at": m.created_at.isoformat(),
        }
        for m in models
    ]
