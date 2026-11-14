"""Image postprocessing, color-space mapping, base64 encoding, and grid layout."""

import base64
import io
from typing import List, Union
import numpy as np
from PIL import Image
import torch


class ImagePostProcessor:
    """Handles tensor denormalization, PIL conversion, and byte/base64 encoding."""

    @staticmethod
    def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Maps tensor in [-1.0, 1.0] to [0.0, 1.0]."""
        return (tensor * 0.5 + 0.5).clamp(0.0, 1.0)

    @classmethod
    def tensor_to_pil(cls, tensor: torch.Tensor) -> Image.Image:
        """Converts a (3, H, W) float tensor in [-1, 1] to a RGB PIL Image."""
        if tensor.dim() == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        np_img = (cls.denormalize_tensor(tensor) * 255.0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        return Image.fromarray(np_img, mode="RGB")

    @classmethod
    def batch_to_pil(cls, batch_tensor: torch.Tensor) -> List[Image.Image]:
        """Converts a (B, 3, H, W) tensor to a list of PIL Images."""
        return [cls.tensor_to_pil(batch_tensor[i]) for i in range(batch_tensor.size(0))]

    @staticmethod
    def pil_to_bytes(image: Image.Image, format: str = "PNG", quality: int = 95) -> bytes:
        """Encodes PIL Image into binary format."""
        buf = io.BytesIO()
        fmt = format.upper()
        if fmt in ("JPG", "JPEG"):
            image.convert("RGB").save(buf, format="JPEG", quality=quality)
        elif fmt == "WEBP":
            image.save(buf, format="WEBP", quality=quality)
        else:
            image.save(buf, format="PNG")
        return buf.getvalue()

    @classmethod
    def pil_to_base64(cls, image: Image.Image, format: str = "PNG") -> str:
        """Encodes PIL Image into base64 data URL string."""
        raw_bytes = cls.pil_to_bytes(image, format=format)
        encoded = base64.b64encode(raw_bytes).decode("utf-8")
        mime = f"image/{format.lower()}"
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def make_grid(images: List[Image.Image], cols: int = 4) -> Image.Image:
        """Arranges a list of PIL images into a tiled rectangular grid."""
        if not images:
            raise ValueError("Cannot create grid from empty image list.")
        n = len(images)
        cols = max(1, min(cols, n))
        rows = (n + cols - 1) // cols
        w, h = images[0].size
        grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
        for i, img in enumerate(images):
            r, c = divmod(i, cols)
            grid.paste(img, (c * w, r * h))
        return grid
