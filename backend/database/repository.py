"""Data access and repository operations for database models."""

import hashlib
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database.models import Generation, GenerationMetadata, ModelVersion, User


class GenerationRepository:
    """Repository handling persistence and querying of Generation records."""

    @staticmethod
    def create_generation(
        db: Session,
        generation_id: str,
        num_samples: int,
        truncation_psi: float,
        mode: str,
        noise_scale: float,
        use_ema: bool,
        clamp_adain: bool,
        output_format: str,
        latency_ms: float,
        artifact_paths: List[str],
        seed: Optional[int] = None,
        user_id: Optional[int] = None,
        model_version_id: Optional[int] = None,
        device: str = "cpu",
        vram_allocated_mb: Optional[float] = None,
        request_id: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> Generation:
        gen = Generation(
            id=generation_id,
            user_id=user_id,
            model_version_id=model_version_id,
            seed=seed,
            num_samples=num_samples,
            truncation_psi=truncation_psi,
            mode=mode,
            noise_scale=noise_scale,
            use_ema=use_ema,
            clamp_adain=clamp_adain,
            output_format=output_format,
            latency_ms=latency_ms,
            status="completed",
            artifact_paths=artifact_paths,
        )
        db.add(gen)

        meta = GenerationMetadata(
            generation_id=generation_id,
            device=device,
            vram_allocated_mb=vram_allocated_mb,
            request_id=request_id,
            client_ip=client_ip,
        )
        db.add(meta)
        db.commit()
        db.refresh(gen)
        return gen

    @staticmethod
    def get_by_id(db: Session, generation_id: str) -> Optional[Generation]:
        return db.query(Generation).filter(Generation.id == generation_id).first()

    @staticmethod
    def list_generations(
        db: Session,
        limit: int = 20,
        offset: int = 0,
        mode: Optional[str] = None,
    ) -> Tuple[List[Generation], int]:
        query = db.query(Generation)
        if mode:
            query = query.filter(Generation.mode == mode)
        total = query.count()
        items = query.order_by(desc(Generation.created_at)).offset(offset).limit(limit).all()
        return items, total


class ModelRepository:
    """Repository handling model catalog entries."""

    @staticmethod
    def get_active(db: Session) -> Optional[ModelVersion]:
        return db.query(ModelVersion).filter(ModelVersion.is_active.is_(True)).first()

    @staticmethod
    def list_all(db: Session) -> List[ModelVersion]:
        return db.query(ModelVersion).order_by(desc(ModelVersion.created_at)).all()


class UserRepository:
    """Repository handling user lookup and authentication."""

    @staticmethod
    def verify_api_key(db: Session, raw_key: str) -> Optional[User]:
        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return db.query(User).filter(User.api_key_hash == key_hash, User.is_active.is_(True)).first()
