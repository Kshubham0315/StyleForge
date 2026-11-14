"""SQLAlchemy ORM models for StyleForge database persistence."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    """API User identity model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    api_key_hash = Column(String(128), nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    generations = relationship("Generation", back_populates="user", cascade="all, delete-orphan")


class ModelVersion(Base):
    """Model registry catalog entry."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), default="StyleForge", nullable=False)
    version_tag = Column(String(32), default="v1.0.0", nullable=False, index=True)
    checkpoint_sha256 = Column(String(64), nullable=False, index=True)
    resolution = Column(Integer, default=64, nullable=False)
    latent_dim = Column(Integer, default=512, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    generations = relationship("Generation", back_populates="model_version")


class Generation(Base):
    """Record of a generative synthesis job."""
    __tablename__ = "generations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True)

    seed = Column(BigInteger, nullable=True)
    num_samples = Column(Integer, default=1, nullable=False)
    truncation_psi = Column(Float, default=0.8, nullable=False)
    mode = Column(String(32), default="standard", nullable=False)
    noise_scale = Column(Float, default=1.0, nullable=False)
    use_ema = Column(Boolean, default=False, nullable=False)
    clamp_adain = Column(Boolean, default=False, nullable=False)
    output_format = Column(String(16), default="png", nullable=False)
    latency_ms = Column(Float, default=0.0, nullable=False)
    status = Column(String(32), default="completed", nullable=False)
    artifact_paths = Column(JSON, default=list, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    user = relationship("User", back_populates="generations")
    model_version = relationship("ModelVersion", back_populates="generations")
    metadata_rel = relationship("GenerationMetadata", back_populates="generation", uselist=False, cascade="all, delete-orphan")


class GenerationMetadata(Base):
    """Detailed hardware and request telemetry for a generation job."""
    __tablename__ = "generation_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_id = Column(String(36), ForeignKey("generations.id", ondelete="CASCADE"), nullable=False, index=True)
    device = Column(String(32), default="cpu", nullable=False)
    vram_allocated_mb = Column(Float, nullable=True)
    request_id = Column(String(64), nullable=True)
    client_ip = Column(String(64), nullable=True)

    generation = relationship("Generation", back_populates="metadata_rel")
