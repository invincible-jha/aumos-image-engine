"""Stable Diffusion 3.5 image generation adapter.

Uses the diffusers StableDiffusion3Pipeline with flow-matching for
higher-quality generation than DDPM-based SD 1.5/XL models.
"""

from __future__ import annotations

import asyncio
import io
import uuid
from typing import Any

import structlog
from aumos_common.logging import get_logger


class SD35Adapter:
    """Stable Diffusion 3.5 Large model adapter.

    Uses diffusers SD3StableDiffusionPipeline with flow-matching
    (not DDPM) for faster, higher-quality generation.

    Args:
        model_id: HuggingFace model ID.
        torch_dtype: Computation dtype ("float16" or "float32").
        device: Target device ("cuda" | "cpu" | "mps").
        cache_dir: Local model cache directory.
    """

    MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
    SUPPORTED_RESOLUTIONS: list[tuple[int, int]] = [
        (512, 512),
        (768, 768),
        (1024, 1024),
        (1280, 720),
        (720, 1280),
    ]

    def __init__(
        self,
        model_id: str = MODEL_ID,
        torch_dtype: str = "float16",
        device: str = "cuda",
        cache_dir: str = "/tmp/model-cache",
    ) -> None:
        """Initialize SD35Adapter.

        Args:
            model_id: HuggingFace model identifier.
            torch_dtype: Tensor dtype for inference.
            device: Device to run inference on.
            cache_dir: Directory to cache downloaded model weights.
        """
        self._model_id = model_id
        self._torch_dtype_str = torch_dtype
        self._device = device
        self._cache_dir = cache_dir
        self._pipe: Any = None
        self._log: structlog.BoundLogger = get_logger(__name__)

    @property
    def model_id(self) -> str:
        """HuggingFace model identifier."""
        return self._model_id

    @property
    def supported_resolutions(self) -> list[tuple[int, int]]:
        """List of (width, height) tuples this model supports."""
        return self.SUPPORTED_RESOLUTIONS.copy()

    async def warm_up(self) -> None:
        """Load SD3.5 model weights into memory/GPU.

        Downloads weights from HuggingFace Hub on first call (cached thereafter).
        This is a blocking operation that may take several minutes on first run.
        """
        self._log.info("warming up SD3.5 pipeline", model_id=self._model_id, device=self._device)
        await asyncio.to_thread(self._load_pipeline)
        self._log.info("SD3.5 pipeline ready")

    def _load_pipeline(self) -> None:
        """Load the diffusers pipeline synchronously (called via to_thread)."""
        import torch
        from diffusers import StableDiffusion3Pipeline

        torch_dtype = torch.float16 if self._torch_dtype_str == "float16" else torch.float32
        self._pipe = StableDiffusion3Pipeline.from_pretrained(
            self._model_id,
            torch_dtype=torch_dtype,
            cache_dir=self._cache_dir,
        )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

    async def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
    ) -> bytes:
        """Generate an image and return PNG bytes.

        Args:
            prompt: Text description of desired image.
            negative_prompt: Elements to exclude from generation.
            num_inference_steps: Number of diffusion steps (more = higher quality).
            guidance_scale: Classifier-free guidance scale.
            width: Output image width in pixels.
            height: Output image height in pixels.
            seed: Random seed for reproducibility.

        Returns:
            PNG-encoded image bytes.

        Raises:
            RuntimeError: If warm_up() has not been called.
        """
        if self._pipe is None:
            raise RuntimeError("SD35Adapter.warm_up() must be called before generate()")

        import torch

        generator = torch.Generator(device=self._device).manual_seed(seed) if seed is not None else None

        result = await asyncio.to_thread(
            self._run_pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return buf.getvalue()

    def _run_pipeline(
        self,
        prompt: str,
        negative_prompt: str | None,
        num_inference_steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        generator: Any,
    ) -> Any:
        """Run pipeline synchronously (called via to_thread)."""
        output = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        return output.images[0]
