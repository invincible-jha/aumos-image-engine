"""SDXL Turbo image generation adapter.

SDXL Turbo uses adversarial diffusion distillation (ADD) to generate images
in a single inference step, enabling near-real-time image generation.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

import structlog
from aumos_common.logging import get_logger


class SDXLTurboAdapter:
    """SDXL Turbo 1-step inference adapter.

    Uses diffusers AutoPipelineForText2Image with the SDXL Turbo scheduler
    for single-step generation. Produces images in under 2 seconds on GPU.

    Args:
        model_id: HuggingFace model ID (default: "stabilityai/sdxl-turbo").
        torch_dtype: Computation dtype ("float16" or "float32").
        device: Target device ("cuda" | "cpu" | "mps").
        cache_dir: Local model cache directory.
    """

    MODEL_ID = "stabilityai/sdxl-turbo"
    SUPPORTED_RESOLUTIONS: list[tuple[int, int]] = [
        (512, 512),
        (768, 768),
        (1024, 1024),
    ]

    def __init__(
        self,
        model_id: str = MODEL_ID,
        torch_dtype: str = "float16",
        device: str = "cuda",
        cache_dir: str = "/tmp/model-cache",
    ) -> None:
        """Initialize SDXLTurboAdapter.

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
        """Load SDXL Turbo model weights into memory/GPU.

        Downloads weights from HuggingFace Hub on first call (cached thereafter).
        """
        self._log.info("warming up SDXL Turbo pipeline", model_id=self._model_id, device=self._device)
        await asyncio.to_thread(self._load_pipeline)
        self._log.info("SDXL Turbo pipeline ready")

    def _load_pipeline(self) -> None:
        """Load the pipeline synchronously (called via to_thread)."""
        import torch
        from diffusers import AutoPipelineForText2Image

        torch_dtype = torch.float16 if self._torch_dtype_str == "float16" else torch.float32
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            self._model_id,
            torch_dtype=torch_dtype,
            variant="fp16" if self._torch_dtype_str == "float16" else None,
            cache_dir=self._cache_dir,
        )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

    async def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        num_inference_steps: int = 1,
        guidance_scale: float = 0.0,
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
    ) -> bytes:
        """Generate an image in a single step and return PNG bytes.

        SDXL Turbo uses ADD (adversarial diffusion distillation) which
        requires guidance_scale=0.0 for optimal single-step quality.

        Args:
            prompt: Text description of desired image.
            negative_prompt: Not used in Turbo mode (guidance_scale=0).
            num_inference_steps: Steps (default: 1 for Turbo).
            guidance_scale: CFG scale (use 0.0 for single-step Turbo).
            width: Output image width in pixels.
            height: Output image height in pixels.
            seed: Random seed for reproducibility.

        Returns:
            PNG-encoded image bytes.

        Raises:
            RuntimeError: If warm_up() has not been called.
        """
        if self._pipe is None:
            raise RuntimeError("SDXLTurboAdapter.warm_up() must be called before generate()")

        import torch

        generator = torch.Generator(device=self._device).manual_seed(seed) if seed is not None else None

        result = await asyncio.to_thread(
            self._run_pipeline,
            prompt=prompt,
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
        num_inference_steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        generator: Any,
    ) -> Any:
        """Run pipeline synchronously (called via to_thread)."""
        output = self._pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        return output.images[0]
