"""Stable Diffusion + ControlNet image generation adapter.

Implements ImageGeneratorProtocol using HuggingFace diffusers.
Supports:
- Stable Diffusion v1.x / v2.x / SDXL
- ControlNet conditioning (pose, depth, canny edge, etc.)
- Multiple samplers (DPM-Solver, DDIM, PNDM, etc.)
- Deterministic generation via seed control
- Half-precision (float16) for GPU memory efficiency
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Deferred imports — torch/diffusers are large; only load when adapter is instantiated
_DIFFUSERS_AVAILABLE = False

try:
    import torch
    from diffusers import (
        ControlNetModel,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        PNDMScheduler,
        StableDiffusionControlNetPipeline,
        StableDiffusionPipeline,
    )
    from PIL import Image as PILImage

    _DIFFUSERS_AVAILABLE = True
except ImportError:
    pass


SCHEDULER_MAP: dict[str, Any] = {
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler if _DIFFUSERS_AVAILABLE else None,
    "DDIMScheduler": DDIMScheduler if _DIFFUSERS_AVAILABLE else None,
    "PNDMScheduler": PNDMScheduler if _DIFFUSERS_AVAILABLE else None,
}


class StableDiffusionAdapter:
    """Generates synthetic images using Stable Diffusion pipelines.

    Supports both standard text-to-image and ControlNet-conditioned generation.
    Model is loaded once at startup and reused across requests.
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        cache_dir: str,
    ) -> None:
        """Initialize adapter configuration (does not load model yet).

        Args:
            model_id: HuggingFace model ID (e.g., "runwayml/stable-diffusion-v1-5").
            device: Torch device ("cuda", "cpu", "mps").
            dtype: Model dtype ("float16", "float32", "bfloat16").
            cache_dir: Local cache directory for downloaded models.
        """
        self._model_id = model_id
        self._device = device
        self._dtype = dtype
        self._cache_dir = cache_dir
        self._pipeline: Any = None
        self._controlnet: Any = None
        self._log = logger.bind(adapter="stable_diffusion", model_id=model_id)

    async def load_model(self) -> None:
        """Load the Stable Diffusion pipeline into memory.

        Downloads model weights on first call (cached for subsequent calls).
        Runs in a thread pool to avoid blocking the event loop.
        """
        if not _DIFFUSERS_AVAILABLE:
            self._log.warning("stable_diffusion.load_skipped", reason="diffusers_not_installed")
            return

        self._log.info("stable_diffusion.load_start", device=self._device, dtype=self._dtype)

        # Load in thread pool — model loading is synchronous and CPU-intensive
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

        self._log.info("stable_diffusion.load_complete")

    def _load_model_sync(self) -> None:
        """Synchronous model loading — called from thread pool."""
        import torch

        torch_dtype = getattr(torch, self._dtype, torch.float16)

        self._pipeline = StableDiffusionPipeline.from_pretrained(
            self._model_id,
            torch_dtype=torch_dtype,
            cache_dir=self._cache_dir,
            safety_checker=None,  # We handle content policy at the service layer
        )
        self._pipeline = self._pipeline.to(self._device)

        # Enable memory-efficient attention if available
        if hasattr(self._pipeline, "enable_attention_slicing"):
            self._pipeline.enable_attention_slicing()

        if self._device == "cuda" and hasattr(self._pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                self._pipeline.enable_xformers_memory_efficient_attention()
                self._log.info("stable_diffusion.xformers_enabled")
            except Exception:
                self._log.info("stable_diffusion.xformers_unavailable")

    @property
    def is_ready(self) -> bool:
        """True if the pipeline is loaded and ready for inference."""
        return self._pipeline is not None

    async def generate(
        self,
        prompt: str,
        negative_prompt: str | None,
        num_images: int,
        width: int,
        height: int,
        model_config: dict[str, Any],
    ) -> list[Any]:
        """Generate synthetic images from a text prompt.

        Args:
            prompt: Positive description of image content.
            negative_prompt: Elements to exclude.
            num_images: Number of images to generate.
            width: Output width in pixels (must be multiple of 64).
            height: Output height in pixels (must be multiple of 64).
            model_config: Generation parameters:
                - guidance_scale: float (default 7.5)
                - num_inference_steps: int (default 50)
                - seed: int | None
                - scheduler: str

        Returns:
            List of generated PIL images.
        """
        if not self.is_ready:
            raise RuntimeError("Stable Diffusion model is not loaded. Call load_model() first.")

        self._log.info(
            "stable_diffusion.generate_start",
            num_images=num_images,
            width=width,
            height=height,
        )

        guidance_scale: float = model_config.get("guidance_scale", 7.5)
        num_inference_steps: int = model_config.get("num_inference_steps", 50)
        seed: int | None = model_config.get("seed")
        scheduler_name: str = model_config.get("scheduler", "DPMSolverMultistepScheduler")

        # Switch scheduler if requested
        if scheduler_name in SCHEDULER_MAP and SCHEDULER_MAP[scheduler_name] is not None:
            self._pipeline.scheduler = SCHEDULER_MAP[scheduler_name].from_config(
                self._pipeline.scheduler.config
            )

        # Create generator for reproducibility
        generator = None
        if seed is not None:
            import torch

            generator = torch.Generator(device=self._device).manual_seed(seed)

        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        generate_fn = partial(
            self._generate_sync,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        images = await loop.run_in_executor(None, generate_fn)

        self._log.info("stable_diffusion.generate_complete", count=len(images))
        return images

    def _generate_sync(
        self,
        prompt: str,
        negative_prompt: str | None,
        num_images: int,
        width: int,
        height: int,
        guidance_scale: float,
        num_inference_steps: int,
        generator: Any,
    ) -> list[Any]:
        """Synchronous generation — called from thread pool."""
        output = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        return output.images  # type: ignore[no-any-return]

    async def generate_with_controlnet(
        self,
        prompt: str,
        negative_prompt: str | None,
        control_image: Any,
        controlnet_model_id: str,
        num_images: int,
        width: int,
        height: int,
        model_config: dict[str, Any],
    ) -> list[Any]:
        """Generate images conditioned on a ControlNet guide image.

        ControlNet allows structured generation using pose maps, depth maps,
        or edge detection as spatial constraints.

        Args:
            prompt: Positive description.
            negative_prompt: Elements to exclude.
            control_image: PIL image to use as ControlNet conditioning.
            controlnet_model_id: ControlNet model (e.g., "lllyasviel/sd-controlnet-openpose").
            num_images: Number of images to generate.
            width: Output width.
            height: Output height.
            model_config: Generation parameters.

        Returns:
            List of generated PIL images.
        """
        if not _DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers not installed")

        self._log.info(
            "stable_diffusion.controlnet_generate",
            controlnet_model=controlnet_model_id,
        )

        loop = asyncio.get_event_loop()
        generate_fn = partial(
            self._generate_controlnet_sync,
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image=control_image,
            controlnet_model_id=controlnet_model_id,
            num_images=num_images,
            width=width,
            height=height,
            guidance_scale=model_config.get("guidance_scale", 7.5),
            num_inference_steps=model_config.get("num_inference_steps", 50),
        )
        return await loop.run_in_executor(None, generate_fn)

    def _generate_controlnet_sync(
        self,
        prompt: str,
        negative_prompt: str | None,
        control_image: Any,
        controlnet_model_id: str,
        num_images: int,
        width: int,
        height: int,
        guidance_scale: float,
        num_inference_steps: int,
    ) -> list[Any]:
        """Synchronous ControlNet generation."""
        import torch

        torch_dtype = getattr(torch, self._dtype, torch.float16)

        # Load ControlNet model
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=torch_dtype,
            cache_dir=self._cache_dir,
        )

        # Create ControlNet pipeline from existing base
        cn_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self._model_id,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            cache_dir=self._cache_dir,
        ).to(self._device)

        output = cn_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_images_per_prompt=num_images,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        return output.images  # type: ignore[no-any-return]
