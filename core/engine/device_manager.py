"""Device and GPU memory management for StyleForge."""

import logging
from typing import Any, Dict, Optional
import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages computation device placement, CUDA memory, and device telemetry."""

    @staticmethod
    def resolve_device(preferred: Optional[str] = None) -> torch.device:
        """Resolves target torch device with safe CPU fallback.

        Args:
            preferred: Desired device string ('cuda', 'cuda:0', 'cpu', or None).

        Returns:
            torch.device instance.
        """
        if preferred is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Auto-resolved computation device to %s", device)
            return device

        req = preferred.strip().lower()
        if "cuda" in req:
            if not torch.cuda.is_available():
                logger.warning("CUDA requested (%s) but not available. Falling back to CPU.", preferred)
                return torch.device("cpu")
            try:
                device = torch.device(req)
                # Test device availability
                _ = torch.empty(1, device=device)
                return device
            except Exception as e:
                logger.warning("Failed to initialize requested CUDA device (%s): %s. Falling back to CPU.", req, e)
                return torch.device("cpu")

        return torch.device("cpu")

    @staticmethod
    def get_vram_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Returns VRAM telemetry in megabytes (MB).

        Returns:
            Dict containing allocated, reserved, total, and free MB.
        """
        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "device_name": "CPU",
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "total_mb": 0.0,
                "free_mb": 0.0,
            }

        dev = device if (device is not None and device.type == "cuda") else torch.device("cuda:0")
        try:
            allocated = torch.cuda.memory_allocated(dev) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(dev) / (1024 * 1024)
            props = torch.cuda.get_device_properties(dev)
            total = props.total_memory / (1024 * 1024)
            free = max(0.0, total - allocated)
            return {
                "cuda_available": True,
                "device_name": props.name,
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "total_mb": round(total, 2),
                "free_mb": round(free, 2),
            }
        except Exception as e:
            logger.warning("Error querying CUDA VRAM info: %s", e)
            return {
                "cuda_available": True,
                "device_name": "CUDA (query failed)",
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "total_mb": 0.0,
                "free_mb": 0.0,
            }

    @staticmethod
    def empty_cache(device: Optional[torch.device] = None) -> None:
        """Safely release unallocated cached GPU memory."""
        if torch.cuda.is_available():
            if device is not None and device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()

    @staticmethod
    def synchronize(device: Optional[torch.device] = None) -> None:
        """Synchronizes device stream if CUDA is active."""
        if torch.cuda.is_available():
            if device is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            else:
                torch.cuda.synchronize()
