"""Generative synthesis endpoints for StyleForge v2.0 REST API."""

import asyncio
from datetime import datetime, timezone
import logging
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.models import User
from backend.database.repository import GenerationRepository, ModelRepository
from backend.database.session import get_db
from backend.metrics.prometheus import (
    ACTIVE_INFERENCES,
    GENERATE_COUNT,
    INFERENCE_LATENCY,
    REQUEST_COUNT,
)
from backend.schemas.requests import (
    BatchGenerateRequest,
    GenerateRequest,
    InterpolateRequest,
    StyleMixRequest,
)
from backend.schemas.responses import GenerateResponse, ImageItem
from backend.security.auth import get_current_user
from backend.security.rate_limiter import check_rate_limit
from backend.storage import storage_service
from core.engine.config import (
    GenerationConfig,
    InferenceMode,
    InterpolationConfig,
    OutputFormat,
    StyleMixConfig,
)
from core.engine.inference_engine import InferenceEngine
from core.engine.postprocessor import ImagePostProcessor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["generation"])

# Bounded concurrency semaphore for GPU memory safety (lazy initialized within event loop)
_gpu_semaphore: Optional[asyncio.Semaphore] = None


def get_gpu_semaphore() -> asyncio.Semaphore:
    global _gpu_semaphore
    if _gpu_semaphore is None:
        _gpu_semaphore = asyncio.Semaphore(settings.GPU_CONCURRENCY_LIMIT)
    return _gpu_semaphore


@router.post(
    "/generate",
    response_model=GenerateResponse,
    dependencies=[Depends(check_rate_limit)],
    summary="Synthesize human faces from latent space",
)
async def generate_faces(
    req: GenerateRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Executes controlled face synthesis using StyleForge generator."""
    engine = InferenceEngine.get_instance(checkpoint_path=settings.CHECKPOINT_PATH, device=settings.DEVICE)
    active_model = ModelRepository.get_active(db)

    # Map request to internal config
    mode_enum = InferenceMode.SMOOTH if req.mode == "smooth" else (
        InferenceMode.STANDARD if req.mode == "standard" else InferenceMode.CUSTOM
    )
    fmt_enum = OutputFormat.PNG if req.output_format == "png" else (
        OutputFormat.JPEG if req.output_format == "jpeg" else OutputFormat.WEBP
    )

    core_config = GenerationConfig(
        seed=req.seed,
        num_samples=req.num_samples,
        truncation_psi=req.truncation_psi,
        mode=mode_enum,
        noise_scale=req.noise_scale,
        use_ema=req.use_ema,
        clamp_adain=req.clamp_adain,
        output_format=fmt_enum,
    )

    generation_id = str(uuid.uuid4())

    async with get_gpu_semaphore():
        ACTIVE_INFERENCES.inc()
        try:
            # Run inference in worker thread to prevent blocking FastAPI event loop
            pil_images, telemetry = await asyncio.to_thread(engine.generate, core_config)
        except Exception as e:
            logger.error("Inference execution failed: %s", e, exc_info=True)
            REQUEST_COUNT.labels(method="POST", endpoint="/generate", status=500).inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "INFERENCE_FAILED", "message": f"Generation failed: {str(e)}"},
            )
        finally:
            ACTIVE_INFERENCES.dec()

    # Track metrics
    INFERENCE_LATENCY.labels(mode=core_config.mode).observe(telemetry["latency_ms"] / 1000.0)
    GENERATE_COUNT.labels(mode=core_config.mode, use_ema=str(req.use_ema)).inc(req.num_samples)
    REQUEST_COUNT.labels(method="POST", endpoint="/generate", status=200).inc()

    # Save artifact images to persistent storage & build response items
    image_items: List[ImageItem] = []
    saved_paths: List[str] = []

    for idx, pil_img in enumerate(pil_images):
        raw_bytes = ImagePostProcessor.pil_to_bytes(pil_img, format=req.output_format)
        ext = req.output_format.lower()
        filename = f"{generation_id}_{idx}.{ext}"
        saved_path = storage_service.save_image(raw_bytes, filename, content_type=f"image/{ext}")
        saved_paths.append(saved_path)

        b64 = ImagePostProcessor.pil_to_base64(pil_img, format=req.output_format)
        image_items.append(
            ImageItem(
                index=idx,
                base64_data=b64,
                download_url=f"/api/v1/artifacts/{saved_path}",
                artifact_path=saved_path,
            )
        )

    # Persist in Database
    client_ip = request.client.host if request.client else "unknown"
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    GenerationRepository.create_generation(
        db=db,
        generation_id=generation_id,
        user_id=current_user.id if current_user else None,
        model_version_id=active_model.id if active_model else None,
        seed=telemetry["seed"],
        num_samples=req.num_samples,
        truncation_psi=req.truncation_psi,
        mode=telemetry["mode"],
        noise_scale=telemetry["noise_scale"],
        use_ema=req.use_ema,
        clamp_adain=req.clamp_adain,
        output_format=req.output_format,
        latency_ms=telemetry["latency_ms"],
        artifact_paths=saved_paths,
        device=telemetry["device"],
        request_id=req_id,
        client_ip=client_ip,
    )

    return GenerateResponse(
        generation_id=generation_id,
        status="completed",
        num_samples=req.num_samples,
        seed=telemetry["seed"],
        mode=telemetry["mode"],
        noise_scale=telemetry["noise_scale"],
        use_ema=req.use_ema,
        latency_ms=telemetry["latency_ms"],
        latency_per_face_ms=telemetry["latency_per_face_ms"],
        resolution=telemetry["resolution"],
        images=image_items,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@router.post(
    "/generate/batch",
    response_model=GenerateResponse,
    dependencies=[Depends(check_rate_limit)],
    summary="Batch face generation",
)
async def generate_batch_faces(
    req: BatchGenerateRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Executes batch face generation."""
    gen_req = GenerateRequest(
        seed=req.seeds[0] if req.seeds and len(req.seeds) > 0 else None,
        num_samples=req.num_samples,
        truncation_psi=req.truncation_psi,
        mode=req.mode,
        noise_scale=req.noise_scale,
        use_ema=req.use_ema,
        output_format=req.output_format,
    )
    return await generate_faces(gen_req, request=request, current_user=current_user, db=db)
