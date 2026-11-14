"""Database connection, engine configuration, and session management."""

import hashlib
import logging
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.config import settings
from backend.database.models import Base, ModelVersion, User
from core.engine.checkpoint_manager import PRODUCTION_CHECKPOINT_SHA256

logger = logging.getLogger(__name__)

# Resolve connection parameters
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False
    # Ensure directory exists for sqlite file
    sqlite_path = settings.DATABASE_URL.replace("sqlite:///", "")
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Creates database tables and seeds initial metadata records."""
    logger.info("Initializing database schemas on %s", settings.DATABASE_URL)
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        # 1. Seed official ModelVersion if not present
        existing_model = db.query(ModelVersion).filter(ModelVersion.version_tag == "v1.0.0").first()
        if not existing_model:
            logger.info("Seeding official ModelVersion v1.0.0 record")
            official_model = ModelVersion(
                name="StyleForge",
                version_tag="v1.0.0",
                checkpoint_sha256=PRODUCTION_CHECKPOINT_SHA256,
                resolution=64,
                latent_dim=512,
                is_active=True,
                description="Official StyleForge v1.0.0 production checkpoint trained on CelebA 64x64.",
            )
            db.add(official_model)

        # 2. Seed default admin user for API key auth
        existing_user = db.query(User).filter(User.username == "admin").first()
        if not existing_user:
            logger.info("Seeding default system user with master API key")
            key_hash = hashlib.sha256(settings.API_KEY.encode("utf-8")).hexdigest()
            default_user = User(
                username="admin",
                api_key_hash=key_hash,
                is_active=True,
            )
            db.add(default_user)

        db.commit()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for scoped database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
