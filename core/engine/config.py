"""Configuration models and enumeration types for StyleForge inference."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class InferenceMode(str, Enum):
    """Supported inference modes."""
    STANDARD = "standard"  # Noise scale 1.0 (v1.0.0 default)
    SMOOTH = "smooth"      # Noise scale 0.25 (Empirical clean face spectral parity mode)
    CUSTOM = "custom"      # Explicit noise scale


class OutputFormat(str, Enum):
    """Supported image output formats."""
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class GenerationConfig(BaseModel):
    """Parameters governing a face generation request."""
    seed: Optional[int] = Field(default=None, description="Deterministic integer seed (0 to 2^31 - 1)")
    num_samples: int = Field(default=1, ge=1, le=32, description="Number of images to synthesize (1 to 32)")
    truncation_psi: float = Field(default=0.8, ge=0.0, le=1.0, description="Truncation psi coefficient")
    mode: InferenceMode = Field(default=InferenceMode.STANDARD, description="Inference mode (standard, smooth, custom)")
    noise_scale: float = Field(default=1.0, ge=0.0, le=1.0, description="Per-channel stochastic noise multiplier")
    use_ema: bool = Field(default=False, description="Whether to use Exponential Moving Average generator weights")
    clamp_adain: bool = Field(default=False, description="Experimental AdaIN scale clamping (defaults to False)")
    enhance: bool = Field(default=False, description="Apply neural face restoration and super-resolution")
    output_format: OutputFormat = Field(default=OutputFormat.PNG, description="Output image encoding format")

    @model_validator(mode="after")
    def sync_noise_scale_with_mode(self) -> "GenerationConfig":
        if self.mode == InferenceMode.SMOOTH and self.noise_scale == 1.0:
            self.noise_scale = 0.25
        elif self.mode == InferenceMode.STANDARD:
            self.noise_scale = 1.0
        return self

    model_config = {
        "extra": "forbid",
        "use_enum_values": True,
    }


class InterpolationConfig(BaseModel):
    """Parameters for continuous latent space traversal."""
    seed_a: int = Field(..., description="Starting identity seed")
    seed_b: int = Field(..., description="Target identity seed")
    steps: int = Field(default=8, ge=2, le=32, description="Interpolation step count")
    space: str = Field(default="W", description="Latent manifold: 'W' (LERP) or 'Z' (SLERP)")
    truncation_psi: float = Field(default=0.8, ge=0.0, le=1.0)
    use_ema: bool = Field(default=False)
    noise_scale: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("space")
    @classmethod
    def validate_space(cls, v: str) -> str:
        v_upper = v.strip().upper()
        if v_upper not in ("W", "Z"):
            raise ValueError("space must be either 'W' or 'Z'")
        return v_upper


class StyleMixConfig(BaseModel):
    """Parameters for multi-scale style mixing."""
    seed_a: int = Field(..., description="Source A seed (geometry or appearance)")
    seed_b: int = Field(..., description="Source B seed")
    coarse_from_b: bool = Field(default=True, description="Take coarse pose/geometry from B and fine style from A")
    truncation_psi: float = Field(default=0.8, ge=0.0, le=1.0)
    use_ema: bool = Field(default=False)
    noise_scale: float = Field(default=1.0, ge=0.0, le=1.0)


# Custom Exception Hierarchy
class StyleForgeError(Exception):
    """Base exception for StyleForge system."""


class CheckpointError(StyleForgeError):
    """Raised when checkpoint fails integrity check or is missing."""


class ModelLoadError(StyleForgeError):
    """Raised when model instantiation or state-dict loading fails."""


class InferenceError(StyleForgeError):
    """Raised when an inference computation fails."""
