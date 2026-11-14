"""Storage service factory."""

from backend.config import settings
from backend.storage.local import LocalStorageService

# Initialize default local storage backend
storage_service = LocalStorageService(settings.STORAGE_DIR)

__all__ = ["storage_service", "LocalStorageService"]
