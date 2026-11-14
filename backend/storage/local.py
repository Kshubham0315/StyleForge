"""Local disk implementation of artifact storage with path sanitization."""

import logging
from pathlib import Path
from typing import Optional
import re

from backend.storage.base import AbstractStorageService

logger = logging.getLogger(__name__)


class LocalStorageService(AbstractStorageService):
    """Stores artifact images safely on the local filesystem."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("LocalStorageService mounted at %s", self.base_dir)

    def _sanitize_filename(self, filename: str) -> str:
        """Removes directory traversal attempts and illegal characters."""
        clean = Path(filename).name
        clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", clean)
        return clean or "artifact.png"

    def save_image(self, data: bytes, filename: str, content_type: str = "image/png") -> str:
        safe_name = self._sanitize_filename(filename)
        dest = self.base_dir / safe_name
        dest.write_bytes(data)
        return safe_name

    def get_image(self, path_locator: str) -> Optional[bytes]:
        safe_name = self._sanitize_filename(path_locator)
        target = (self.base_dir / safe_name).resolve()
        # Prevent traversal outside base_dir
        if not str(target).startswith(str(self.base_dir)):
            logger.warning("Attempted path traversal detected: %s", path_locator)
            return None
        if target.exists() and target.is_file():
            return target.read_bytes()
        return None

    def delete_image(self, path_locator: str) -> bool:
        safe_name = self._sanitize_filename(path_locator)
        target = (self.base_dir / safe_name).resolve()
        if not str(target).startswith(str(self.base_dir)):
            return False
        if target.exists() and target.is_file():
            target.unlink()
            return True
        return False
