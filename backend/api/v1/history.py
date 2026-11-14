"""Generation history and artifact download endpoints."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from backend.database.models import User
from backend.database.repository import GenerationRepository
from backend.database.session import get_db
from backend.schemas.responses import GenerationListItem, GenerationListResponse
from backend.security.auth import get_current_user
from backend.storage import storage_service

router = APIRouter(prefix="", tags=["history"])


@router.get("/generations", response_model=GenerationListResponse, summary="Query generation history")
def list_generations(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    mode: Optional[str] = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieves paginated history of face generations."""
    filter_user_id = None if (current_user and current_user.username == "admin") else (current_user.id if current_user else None)
    items, total = GenerationRepository.list_generations(
        db=db, limit=limit, offset=offset, mode=mode, user_id=filter_user_id
    )

    out_items = []
    for g in items:
        thumb = f"/api/v1/artifacts/{g.artifact_paths[0]}" if (g.artifact_paths and len(g.artifact_paths) > 0) else None
        out_items.append(
            GenerationListItem(
                id=g.id,
                created_at=g.created_at.isoformat(),
                num_samples=g.num_samples,
                seed=g.seed,
                mode=g.mode,
                noise_scale=g.noise_scale,
                use_ema=g.use_ema,
                latency_ms=g.latency_ms,
                download_url=thumb,
            )
        )

    return GenerationListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=out_items,
    )


@router.get("/generations/{generation_id}", summary="Get detailed generation record")
def get_generation(generation_id: str, db: Session = Depends(get_db)):
    """Retrieves a single generation record by its UUID."""
    gen = GenerationRepository.get_by_id(db, generation_id)
    if not gen:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Generation record {generation_id} not found."},
        )

    return {
        "id": gen.id,
        "seed": gen.seed,
        "num_samples": gen.num_samples,
        "truncation_psi": gen.truncation_psi,
        "mode": gen.mode,
        "noise_scale": gen.noise_scale,
        "use_ema": gen.use_ema,
        "clamp_adain": gen.clamp_adain,
        "output_format": gen.output_format,
        "latency_ms": gen.latency_ms,
        "status": gen.status,
        "artifacts": [f"/api/v1/artifacts/{p}" for p in gen.artifact_paths],
        "created_at": gen.created_at.isoformat(),
    }


@router.get("/artifacts/{filename}", summary="Download raw artifact image")
def download_artifact(filename: str):
    """Securely streams stored image binary."""
    data = storage_service.get_image(filename)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "FILE_NOT_FOUND", "message": "Artifact image not found."},
        )

    ext = filename.split(".")[-1].lower() if "." in filename else "png"
    mime = f"image/{ext}" if ext in ("png", "jpeg", "webp") else "application/octet-stream"

    return Response(
        content=data,
        media_type=mime,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
