"""Abstract storage service contract."""

from abc import ABC, abstractmethod
from typing import Optional


class AbstractStorageService(ABC):
    """Storage interface defining operations for persistent artifact files."""

    @abstractmethod
    def save_image(self, data: bytes, filename: str, content_type: str = "image/png") -> str:
        """Stores image binary and returns unique relative locator."""

    @abstractmethod
    def get_image(self, path_locator: str) -> Optional[bytes]:
        """Retrieves image binary from storage locator."""

    @abstractmethod
    def delete_image(self, path_locator: str) -> bool:
        """Deletes artifact from storage."""
