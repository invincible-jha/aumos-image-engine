"""LoRA fine-tuning adapter for image generation models.

Implements DreamBooth-style LoRA training via the diffusers library.
Tenants provide 20-100 reference images; a LoRA adapter is trained and
stored in MinIO for subsequent generation jobs.
"""

from __future__ import annotations

import asyncio
import io
import uuid
from typing import Any

import structlog
from aumos_common.logging import get_logger


class LoRATrainer:
    """Trains a LoRA adapter on tenant-provided reference images.

    Uses DreamBooth-style training via Hugging Face diffusers trainer.
    The trained adapter is stored in MinIO and referenced by adapter URI
    in subsequent generation jobs.

    Args:
        base_model: Base model to fine-tune (default: "sdxl").
        learning_rate: LoRA training learning rate.
        max_train_steps: Maximum training steps.
        rank: LoRA rank — higher rank means more capacity but more VRAM.
        device: Target device ("cuda" | "cpu" | "mps").
        cache_dir: Directory for cached base model weights.
    """

    SUPPORTED_BASE_MODELS: list[str] = ["sd15", "sdxl", "sd35"]

    def __init__(
        self,
        base_model: str = "sdxl",
        learning_rate: float = 1e-4,
        max_train_steps: int = 500,
        rank: int = 16,
        device: str = "cuda",
        cache_dir: str = "/tmp/model-cache",
    ) -> None:
        """Initialize LoRATrainer.

        Args:
            base_model: Registered model name to fine-tune.
            learning_rate: Optimizer learning rate.
            max_train_steps: Cap on training iterations.
            rank: LoRA decomposition rank.
            device: Torch device for training.
            cache_dir: Local cache for base model weights.
        """
        if base_model not in self.SUPPORTED_BASE_MODELS:
            raise ValueError(
                f"base_model '{base_model}' not supported. "
                f"Supported: {self.SUPPORTED_BASE_MODELS}"
            )
        self._base_model = base_model
        self._learning_rate = learning_rate
        self._max_train_steps = max_train_steps
        self._rank = rank
        self._device = device
        self._cache_dir = cache_dir
        self._log: structlog.BoundLogger = get_logger(__name__)

    @property
    def base_model(self) -> str:
        """Base model name being fine-tuned."""
        return self._base_model

    async def train(
        self,
        reference_image_uris: list[str],
        concept_prompt: str,
        job_id: uuid.UUID,
        storage: Any,
    ) -> str:
        """Train a LoRA adapter on reference images and return its MinIO URI.

        Downloads reference images from MinIO, trains a LoRA adapter using
        DreamBooth-style training, uploads the adapter weights to MinIO,
        and returns the adapter URI for use in generation jobs.

        Args:
            reference_image_uris: MinIO URIs of reference images (20-100 recommended).
            concept_prompt: Unique token prompt describing the concept
                (e.g., "a photo of sks industrial widget").
            job_id: Job identifier used for output path namespacing.
            storage: Storage adapter for downloading inputs and uploading adapter.

        Returns:
            MinIO URI of the trained LoRA adapter (.safetensors file).

        Raises:
            ValueError: If fewer than 5 reference images are provided.
            RuntimeError: If training fails due to VRAM or other hardware issues.
        """
        if len(reference_image_uris) < 5:
            raise ValueError(
                f"Minimum 5 reference images required, got {len(reference_image_uris)}"
            )

        self._log.info(
            "lora_training_started",
            job_id=str(job_id),
            base_model=self._base_model,
            num_images=len(reference_image_uris),
            max_steps=self._max_train_steps,
            rank=self._rank,
        )

        # Download reference images
        image_bytes_list: list[bytes] = []
        for uri in reference_image_uris:
            image_data = await storage.download(uri)
            image_bytes_list.append(image_data)

        adapter_bytes = await asyncio.to_thread(
            self._run_training,
            image_bytes_list=image_bytes_list,
            concept_prompt=concept_prompt,
        )

        adapter_key = f"img-lora-adapters/{job_id}/lora_adapter.safetensors"
        adapter_uri = await storage.upload(adapter_bytes, adapter_key)

        self._log.info(
            "lora_training_complete",
            job_id=str(job_id),
            adapter_uri=adapter_uri,
        )

        return adapter_uri

    def _run_training(
        self,
        image_bytes_list: list[bytes],
        concept_prompt: str,
    ) -> bytes:
        """Run LoRA training synchronously (called via to_thread).

        Args:
            image_bytes_list: List of raw image bytes for reference images.
            concept_prompt: DreamBooth concept prompt string.

        Returns:
            Serialized LoRA adapter weights as bytes (.safetensors format).
        """
        import tempfile
        from pathlib import Path

        import torch
        from diffusers import DiffusionPipeline
        from peft import LoraConfig, get_peft_model
        from PIL import Image
        from safetensors.torch import save_file

        # Resolve base model ID
        model_id_map = {
            "sd15": "runwayml/stable-diffusion-v1-5",
            "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
            "sd35": "stabilityai/stable-diffusion-3.5-large",
        }
        model_id = model_id_map[self._base_model]

        torch_dtype = torch.float16 if self._device != "cpu" else torch.float32

        # Load pipeline for UNet extraction
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            cache_dir=self._cache_dir,
        )
        unet = pipeline.unet

        # Configure LoRA on attention projection layers
        lora_config = LoraConfig(
            r=self._rank,
            lora_alpha=self._rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.1,
            bias="none",
        )
        unet = get_peft_model(unet, lora_config)

        if self._device != "cpu":
            unet = unet.to(self._device)

        # Prepare reference images as PIL images
        pil_images: list[Any] = []
        for raw_bytes in image_bytes_list:
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize((512, 512))
            pil_images.append(img)

        # Simple DreamBooth training loop
        optimizer = torch.optim.AdamW(unet.parameters(), lr=self._learning_rate)

        with tempfile.TemporaryDirectory() as tmp_dir:
            from torchvision import transforms

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

            image_tensors = torch.stack([transform(img) for img in pil_images])
            if self._device != "cpu":
                image_tensors = image_tensors.to(self._device)

            text_encoder = pipeline.text_encoder
            tokenizer = pipeline.tokenizer
            if self._device != "cpu":
                text_encoder = text_encoder.to(self._device)

            # Tokenize concept prompt
            tokens = tokenizer(
                [concept_prompt] * len(pil_images),
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            if self._device != "cpu":
                tokens = {k: v.to(self._device) for k, v in tokens.items()}

            vae = pipeline.vae
            if self._device != "cpu":
                vae = vae.to(self._device)

            scheduler = pipeline.scheduler

            unet.train()
            steps_done = 0
            while steps_done < self._max_train_steps:
                optimizer.zero_grad()

                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(image_tensors).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Add noise to latents
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0,
                        scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=latents.device,
                    ).long()
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                    # Encode text
                    encoder_hidden_states = text_encoder(**tokens)[0]

                # Predict noise
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()

                steps_done += 1

            # Save LoRA weights
            adapter_path = Path(tmp_dir) / "lora_adapter.safetensors"
            lora_state_dict = {
                k: v for k, v in unet.state_dict().items() if "lora" in k
            }
            save_file(lora_state_dict, str(adapter_path))
            return adapter_path.read_bytes()
