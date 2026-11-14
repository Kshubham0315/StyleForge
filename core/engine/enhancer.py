"""Neural Face Restoration and Super-Resolution Post-Processor."""

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

logger = logging.getLogger(__name__)


class FaceEnhancer:
    """Thread-safe neural face restoration module operating on low-res face synthesis outputs."""

    _instance: Optional["FaceEnhancer"] = None

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path or "checkpoints/enhancer/gfpgan_1.4.onnx")
        self.session = None
        self.input_name = None
        self.output_name = None

        if HAS_ONNX and self.model_path.exists():
            try:
                available = ort.get_available_providers()
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(str(self.model_path), providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                logger.info("FaceEnhancer initialized with %s on %s", self.model_path.name, providers[0])
            except Exception as e:
                logger.warning("Failed to initialize FaceEnhancer ONNX session: %s", e)

    @classmethod
    def get_instance(cls) -> "FaceEnhancer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_available(self) -> bool:
        return self.session is not None

    def enhance_pil(self, img: Image.Image) -> Image.Image:
        """Restores and upscales a low-res face to 512x512 with fine facial geometry."""
        if not self.is_available:
            return img.resize((512, 512), Image.Resampling.LANCZOS)

        try:
            face_512 = img.resize((512, 512), Image.Resampling.BILINEAR)
            img_np = (np.array(face_512).astype(np.float32) / 255.0 - 0.5) / 0.5
            img_bgr = img_np[:, :, ::-1]
            inp = np.transpose(img_bgr, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)

            out = self.session.run([self.output_name], {self.input_name: inp})[0][0]
            out = np.clip((out * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            out_rgb = np.transpose(out, (1, 2, 0))[:, :, ::-1]
            return Image.fromarray(out_rgb)
        except Exception as e:
            logger.warning("Face enhancement failed, returning upsampled original: %s", e)
            return img.resize((512, 512), Image.Resampling.LANCZOS)

    def enhance_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        return [self.enhance_pil(im) for im in images]
