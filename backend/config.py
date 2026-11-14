"""Application configuration and environment settings for StyleForge Backend."""

from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables or .env file."""

    APP_NAME: str = "StyleForge API"
    APP_VERSION: str = "2.0.0"
    ENV: str = Field(default="development", description="'development', 'production', or 'test'")
    DEBUG: bool = False

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./data/styleforge.db",
        description="SQLAlchemy DB connection string (PostgreSQL in production, SQLite fallback for dev)",
    )

    # Artifact Storage
    STORAGE_DIR: str = Field(default="./data/artifacts", description="Local directory for image artifact files")
    S3_BUCKET_NAME: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None

    # Security & Auth
    ENABLE_AUTH: bool = Field(default=True, description="Enforce API Key authentication on protected endpoints")
    API_KEY: str = Field(default="styleforge-dev-key-2026", description="Master API Key for backend access")
    RATE_LIMIT_PER_MINUTE: int = Field(default=120, description="Max allowed requests per client per minute")
    CORS_ORIGINS: List[str] = ["*"]

    # Model & Inference
    CHECKPOINT_PATH: str = "checkpoints/styleforge_final_model.pt"
    DEVICE: Optional[str] = None
    MAX_BATCH_SIZE: int = 16
    GPU_CONCURRENCY_LIMIT: int = 1  # Bounded concurrency to protect VRAM

    # MLOps & Observability
    MLFLOW_TRACKING_URI: Optional[str] = None
    ENABLE_PROMETHEUS: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
