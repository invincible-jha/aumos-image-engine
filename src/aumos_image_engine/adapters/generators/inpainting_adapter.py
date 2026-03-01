"""Inpainting and outpainting adapter for image editing.

Uses Stable Diffusion inpainting pipeline to fill masked regions
with generated content. Primary use case: face de-identification via
targeted inpainting (replace face regions, preserve background).
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

import structlog
from aumos_common.logging import get_logger


class InpaintingAdapter:
    """SD Inpainting pipeline for region-based image editing.

    Accepts an image and a binary mask; fills the masked region using
    a Stable Diffusion inpainting pipeline guided by a text prompt.
    The unmasked region is preserved exactly.

    Primary use cases:
    - Face de-identification: mask face regions, fill with synthetic faces
    - Object removal: mask unwanted objects, fill with background
    - Scene editing: replace specific image regions with generated content

    Args:
        model_id: HuggingFace inpainting model ID.
        torch_dtype: Computation dtype ("float16" or "float32").
        device: Target device ("cuda" | "cpu" | "mps").
        cache_dir: Local model cache directory.
    """

    MODEL_ID = "runwayml/stable-diffusion-inpainting"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        torch_dtype: str = "float16",
        device: str = "cuda",
        cache_dir: str = "/tmp/model-cache",
    ) -> None:
        """Initialize InpaintingAdapter.

        Args:
            model_id: HuggingFace inpainting model identifier.
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

    async def warm_up(self) -> None:
        """Load inpainting model weights into memory/GPU.

        Downloads weights from HuggingFace Hub on first call (cached thereafter).
        """
        self._log.info("warming_up_inpainting_pipeline", model_id=self._model_id)
        await asyncio.to_thread(self._load_pipeline)
        self._log.info("inpainting_pipeline_ready")

    def _load_pipeline(self) -> None:
        """Load the inpainting pipeline synchronously (called via to_thread)."""
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        torch_dtype = torch.float16 if self._torch_dtype_str == "float16" else torch.float32
        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self._model_id,
            torch_dtype=torch_dtype,
            cache_dir=self._cache_dir,
        )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

    async def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str,
        negative_prompt: str | None = None,
        strength: float = 0.75,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> bytes:
        """Fill masked image regions with prompt-guided generated content.

        The mask is a binary image where white pixels (255) indicate regions
        to inpaint and black pixels (0) indicate regions to preserve.

        Args:
            image_bytes: Input PNG/JPEG image bytes.
            mask_bytes: Binary mask PNG bytes (white = inpaint, black = preserve).
            prompt: Text description of content to generate in masked region.
            negative_prompt: Elements to avoid in the inpainted region.
            strength: Denoising strength for inpainting (0.0-1.0).
                Higher values allow more deviation from the input image.
            num_inference_steps: Number of diffusion denoising steps.
            guidance_scale: CFG scale for prompt adherence.
            seed: Random seed for reproducibility.

        Returns:
            PNG-encoded inpainted image bytes.

        Raises:
            RuntimeError: If warm_up() has not been called.
            ValueError: If strength is outside [0.0, 1.0].
        """
        if self._pipe is None:
            raise RuntimeError("InpaintingAdapter.warm_up() must be called before inpaint()")
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"strength must be in [0.0, 1.0], got {strength}")

        import torch

        generator = torch.Generator(device=self._device).manual_seed(seed) if seed is not None else None

        result = await asyncio.to_thread(
            self._run_inpainting_pipeline,
            image_bytes=image_bytes,
            mask_bytes=mask_bytes,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        return buffer.getvalue()

    def _run_inpainting_pipeline(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str,
        negative_prompt: str | None,
        strength: float,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Any,
    ) -> Any:
        """Run inpainting pipeline synchronously (called via to_thread).

        Args:
            image_bytes: Raw input image bytes.
            mask_bytes: Raw binary mask bytes.
            prompt: Inpainting guidance prompt.
            negative_prompt: Negative guidance prompt.
            strength: Inpainting strength coefficient.
            num_inference_steps: Denoising steps.
            guidance_scale: CFG scale.
            generator: Seeded torch Generator.

        Returns:
            PIL Image with inpainted content.
        """
        from PIL import Image

        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask_image = Image.open(io.BytesIO(mask_bytes)).convert("RGB")

        # Resize mask to match image dimensions
        if mask_image.size != input_image.size:
            mask_image = mask_image.resize(input_image.size)

        output = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            mask_image=mask_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return output.images[0]

    async def outpaint(
        self,
        image_bytes: bytes,
        prompt: str,
        direction: str = "right",
        expand_pixels: int = 256,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> bytes:
        """Extend the image canvas in a specified direction via outpainting.

        Pads the image in the specified direction and creates an outpainting
        mask to fill the new canvas area with generated content that
        seamlessly extends the original image.

        Args:
            image_bytes: Input PNG/JPEG image bytes.
            prompt: Text description of content to generate in the new area.
            direction: Direction to expand ("right" | "left" | "top" | "bottom").
            expand_pixels: Number of pixels to add in the specified direction.
            num_inference_steps: Number of diffusion denoising steps.
            guidance_scale: CFG scale for prompt adherence.
            seed: Random seed for reproducibility.

        Returns:
            PNG-encoded outpainted image bytes (larger than input).

        Raises:
            RuntimeError: If warm_up() has not been called.
            ValueError: If direction is not one of the supported values.
        """
        if self._pipe is None:
            raise RuntimeError("InpaintingAdapter.warm_up() must be called before outpaint()")

        valid_directions = {"right", "left", "top", "bottom"}
        if direction not in valid_directions:
            raise ValueError(f"direction must be one of {valid_directions}, got '{direction}'")

        from PIL import Image, ImageDraw

        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = original.size

        # Create expanded canvas with black padding
        if direction == "right":
            new_size = (orig_w + expand_pixels, orig_h)
            paste_pos = (0, 0)
            mask_box = (orig_w, 0, new_size[0], orig_h)
        elif direction == "left":
            new_size = (orig_w + expand_pixels, orig_h)
            paste_pos = (expand_pixels, 0)
            mask_box = (0, 0, expand_pixels, orig_h)
        elif direction == "bottom":
            new_size = (orig_w, orig_h + expand_pixels)
            paste_pos = (0, 0)
            mask_box = (0, orig_h, orig_w, new_size[1])
        else:  # top
            new_size = (orig_w, orig_h + expand_pixels)
            paste_pos = (0, expand_pixels)
            mask_box = (0, 0, orig_w, expand_pixels)

        canvas = Image.new("RGB", new_size, color=(0, 0, 0))
        canvas.paste(original, paste_pos)

        # Create mask: white where expansion occurred
        mask = Image.new("RGB", new_size, color=(0, 0, 0))
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(mask_box, fill=(255, 255, 255))

        canvas_bytes = io.BytesIO()
        canvas.save(canvas_bytes, format="PNG")

        mask_bytes = io.BytesIO()
        mask.save(mask_bytes, format="PNG")

        return await self.inpaint(
            image_bytes=canvas_bytes.getvalue(),
            mask_bytes=mask_bytes.getvalue(),
            prompt=prompt,
            strength=1.0,  # Full strength for outpainting new regions
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
