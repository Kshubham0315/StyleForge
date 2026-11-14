"""Pydantic API request models with strict boundary validation."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    """Specification for generating one or more faces."""
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1, description="Deterministic integer seed")
    num_samples: int = Field(default=1, ge=1, le=16, description="Number of images to generate (max 16)")
    truncation_psi: float = Field(default=0.8, ge=0.0, le=1.0, description="Truncation psi coefficient")
    mode: str = Field(default="standard", description="'standard' (noise 1.0), 'smooth' (noise 0.25), or 'custom'")
    noise_scale: float = Field(default=1.0, ge=0.0, le=1.0, description="Per-channel stochastic noise multiplier")
    use_ema: bool = Field(default=False, description="Generate with EMA weights rather than raw weights")
    clamp_adain: bool = Field(default=False, description="Experimental AdaIN scale clamping (defaults to False)")
    enhance: bool = Field(default=False, description="Apply neural face restoration and super-resolution")
    output_format: str = Field(default="png", description="Image encoding: 'png', 'jpeg', or 'webp'")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ("standard", "smooth", "custom"):
            raise ValueError("mode must be one of: 'standard', 'smooth', 'custom'")
        return clean

    @field_validator("output_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ("png", "jpeg", "jpg", "webp"):
            raise ValueError("output_format must be 'png', 'jpeg', or 'webp'")
        return "png" if clean == "png" else ("jpeg" if clean in ("jpg", "jpeg") else "webp")


class BatchGenerateRequest(BaseModel):
    """Specification for batch generation across multiple seeds."""
    seeds: Optional[List[int]] = Field(default=None, description="Explicit list of seeds")
    num_samples: int = Field(default=4, ge=1, le=16, description="Total images to generate")
    truncation_psi: float = Field(default=0.8, ge=0.0, le=1.0)
    mode: str = Field(default="standard")
    noise_scale: float = Field(default=1.0, ge=0.0, le=1.0)
    use_ema: bool = Field(default=False)
    output_format: str = Field(default="png")


class InterpolateRequest(BaseModel):
    """Specification for latent manifold traversal."""
    seed_a: int = Field(..., ge=0, le=2**31 - 1)
    seed_b: int = Field(..., ge=0, le=2**31 - 1)
    steps: int = Field(default=8, ge=2, le=24)
    space: str = Field(default="W", description="'W' (LERP) or 'Z' (SLERP)")
    truncation_psi: float = Field(default=0.8, ge=0.0, le=1.0)
    use_ema: bool = Field(default=False)
    noise_scale: float = Field(default=1.0, ge=0.0, le=1.0)


class StyleMixRequest(BaseModel):
    """Specification for multi-scale style mixing."""
    seed_a: int = Field(..., ge=0, le=2**31 - 1)
    seed_b: int = Field(..., ge=0, le=2**31 - 1)
    coarse_from_b: bool = Field(default=True)
    truncation_psi: float = Field(default=0.8, ge=0.0, le=1.0)
    use_ema: bool = Field(default=False)
    noise_scale: float = Field(default=1.0, ge=0.0, le=1.0)
