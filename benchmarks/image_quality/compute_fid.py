"""Fréchet Inception Distance (FID) computation for image quality benchmarks.

Compares real and generated image distributions using InceptionV3 features.
Lower FID scores indicate better generative quality.
"""

from __future__ import annotations

import asyncio
from pathlib import Path


async def compute_fid(
    real_images_dir: str,
    generated_images_dir: str,
    batch_size: int = 32,
    feature_dim: int = 2048,
) -> float:
    """Compute FID score between real and generated image distributions.

    Args:
        real_images_dir: Path to directory containing real reference images.
        generated_images_dir: Path to directory containing generated images.
        batch_size: Batch size for feature extraction.
        feature_dim: InceptionV3 feature dimension (default: 2048).

    Returns:
        FID score (lower is better; typical good range: < 50).
    """
    return await asyncio.to_thread(
        _compute_fid_sync,
        real_images_dir=real_images_dir,
        generated_images_dir=generated_images_dir,
        batch_size=batch_size,
        feature_dim=feature_dim,
    )


def _compute_fid_sync(
    real_images_dir: str,
    generated_images_dir: str,
    batch_size: int,
    feature_dim: int,
) -> float:
    """Synchronous FID computation (called via to_thread).

    Args:
        real_images_dir: Path to real images directory.
        generated_images_dir: Path to generated images directory.
        batch_size: Feature extraction batch size.
        feature_dim: InceptionV3 feature layer dimension.

    Returns:
        FID score as float.
    """
    import torch
    from PIL import Image
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    fid = FrechetInceptionDistance(feature=feature_dim)
    fid.eval()

    def _load_images_as_uint8(image_dir: str) -> list[torch.Tensor]:
        """Load all PNG/JPG images from directory as uint8 tensors."""
        directory = Path(image_dir)
        image_paths = list(directory.glob("*.png")) + list(directory.glob("*.jpg"))
        tensors: list[torch.Tensor] = []
        for img_path in sorted(image_paths):
            img = Image.open(img_path).convert("RGB")
            # FID expects uint8 [0, 255] tensors in shape (N, 3, H, W)
            tensor = transforms.functional.to_tensor(
                transforms.functional.resize(img, [299, 299])
            )
            # Convert to uint8 range [0, 255]
            uint8_tensor = (tensor * 255).to(torch.uint8)
            tensors.append(uint8_tensor)
        return tensors

    real_tensors = _load_images_as_uint8(real_images_dir)
    generated_tensors = _load_images_as_uint8(generated_images_dir)

    # Process in batches
    for i in range(0, len(real_tensors), batch_size):
        batch = torch.stack(real_tensors[i:i + batch_size])
        fid.update(batch, real=True)

    for i in range(0, len(generated_tensors), batch_size):
        batch = torch.stack(generated_tensors[i:i + batch_size])
        fid.update(batch, real=False)

    return float(fid.compute())


async def compute_inception_score(
    generated_images_dir: str,
    batch_size: int = 32,
    splits: int = 10,
) -> tuple[float, float]:
    """Compute Inception Score (IS) for generated images.

    Args:
        generated_images_dir: Path to directory containing generated images.
        batch_size: Batch size for feature extraction.
        splits: Number of splits for IS computation (more = more stable estimate).

    Returns:
        Tuple of (mean IS, std IS). Higher is better (typical range: 1-300).
    """
    return await asyncio.to_thread(
        _compute_is_sync,
        generated_images_dir=generated_images_dir,
        batch_size=batch_size,
        splits=splits,
    )


def _compute_is_sync(
    generated_images_dir: str,
    batch_size: int,
    splits: int,
) -> tuple[float, float]:
    """Synchronous IS computation (called via to_thread).

    Args:
        generated_images_dir: Path to generated images directory.
        batch_size: Feature extraction batch size.
        splits: Number of IS splits for variance estimation.

    Returns:
        Tuple of (mean, std) IS scores.
    """
    import torch
    from PIL import Image
    from torchmetrics.image.inception import InceptionScore
    from torchvision import transforms

    is_metric = InceptionScore(splits=splits)
    is_metric.eval()

    directory = Path(generated_images_dir)
    image_paths = list(directory.glob("*.png")) + list(directory.glob("*.jpg"))
    tensors: list[torch.Tensor] = []

    for img_path in sorted(image_paths):
        img = Image.open(img_path).convert("RGB")
        tensor = transforms.functional.to_tensor(
            transforms.functional.resize(img, [299, 299])
        )
        uint8_tensor = (tensor * 255).to(torch.uint8)
        tensors.append(uint8_tensor)

    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i + batch_size])
        is_metric.update(batch)

    mean, std = is_metric.compute()
    return float(mean), float(std)
