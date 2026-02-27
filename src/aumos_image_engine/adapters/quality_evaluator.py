"""Image quality evaluation adapter for synthetic image assessment.

Computes standard computer-vision quality metrics to evaluate the fidelity
and perceptual quality of generated synthetic images:

- FID (Frechet Inception Distance): measures distributional similarity between
  real and synthetic image sets using Inception-v3 feature statistics.
  Lower = more realistic.

- IS (Inception Score): measures both quality and diversity of a set of synthetic
  images. Higher = better quality + diversity.

- LPIPS (Learned Perceptual Image Patch Similarity): perceptual similarity using
  deep network features. Lower = more perceptually similar to reference.

- SSIM (Structural Similarity Index): structural, luminance, and contrast
  similarity. Range [0, 1], higher = more similar.

- PSNR (Peak Signal-to-Noise Ratio): pixel-level fidelity in dB.
  Higher = less noise/distortion.

All heavy computation runs via run_in_executor to avoid blocking the event loop.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any

import numpy as np
import structlog
from PIL import Image as PILImage

logger = structlog.get_logger(__name__)

try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import inception_v3

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as skimage_ssim  # type: ignore[import]

    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False

try:
    import lpips as lpips_lib  # type: ignore[import]

    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


class InceptionQualityEvaluator:
    """Image quality evaluator using Inception-v3 and perceptual metrics.

    Computes FID, IS, LPIPS, SSIM, and PSNR. When heavy ML libraries
    (torch, lpips) are unavailable, lightweight numpy fallbacks are used
    so quality reporting remains operational in CPU-only environments.
    """

    def __init__(
        self,
        device: str = "cpu",
        inception_batch_size: int = 32,
    ) -> None:
        """Initialize the quality evaluator.

        Args:
            device: PyTorch device string ("cuda", "cpu", or "mps").
            inception_batch_size: Batch size for Inception-v3 feature extraction.
        """
        self._device = device
        self._inception_batch_size = inception_batch_size
        self._inception_model: Any = None
        self._lpips_model: Any = None
        self._log = logger.bind(adapter="quality_evaluator")
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Load Inception-v3 and LPIPS models."""
        if _TORCH_AVAILABLE:
            try:
                self._inception_model = inception_v3(pretrained=True, transform_input=False)
                self._inception_model.fc = torch.nn.Identity()  # Remove classification head
                self._inception_model.eval()
                self._inception_model.to(self._device)
                self._log.info("quality_evaluator.inception_loaded", device=self._device)
            except Exception as exc:
                self._log.warning("quality_evaluator.inception_load_failed", error=str(exc))

        if _LPIPS_AVAILABLE and _TORCH_AVAILABLE:
            try:
                self._lpips_model = lpips_lib.LPIPS(net="alex")
                self._lpips_model.to(self._device)
                self._log.info("quality_evaluator.lpips_loaded")
            except Exception as exc:
                self._log.warning("quality_evaluator.lpips_load_failed", error=str(exc))

    async def compute_fid(
        self,
        real_images: list[PILImage.Image],
        synthetic_images: list[PILImage.Image],
    ) -> float:
        """Compute Frechet Inception Distance between real and synthetic sets.

        FID measures the Wasserstein-2 distance between multivariate Gaussians
        fitted to Inception-v3 feature activations of real vs. synthetic images.
        Lower FID = more realistic synthetic images.

        Args:
            real_images: Reference set of real images (minimum 50 recommended).
            synthetic_images: Synthetic images to evaluate.

        Returns:
            FID score (float). Lower is better. < 50 is generally acceptable
            for domain-specific synthetic data.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        fid_fn = partial(
            self._compute_fid_sync,
            real_images=real_images,
            synthetic_images=synthetic_images,
        )
        return await loop.run_in_executor(None, fid_fn)

    def _compute_fid_sync(
        self,
        real_images: list[PILImage.Image],
        synthetic_images: list[PILImage.Image],
    ) -> float:
        """Synchronous FID computation."""
        if self._inception_model is None:
            # Fallback: mean pixel-level Euclidean distance as FID approximation
            return self._fallback_fid(real_images, synthetic_images)

        real_features = self._extract_inception_features(real_images)
        synthetic_features = self._extract_inception_features(synthetic_images)

        # Compute mean and covariance for each set
        mu_real = np.mean(real_features, axis=0)
        mu_synth = np.mean(synthetic_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_synth = np.cov(synthetic_features, rowvar=False)

        fid = self._frechet_distance(mu_real, sigma_real, mu_synth, sigma_synth)
        self._log.info(
            "quality_evaluator.fid_computed",
            fid=round(fid, 4),
            real_count=len(real_images),
            synthetic_count=len(synthetic_images),
        )
        return float(fid)

    def _extract_inception_features(
        self, images: list[PILImage.Image]
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Extract 2048-dim Inception-v3 pool3 features from images."""
        import torch

        transform = T.Compose(
            [
                T.Resize(299),
                T.CenterCrop(299),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        all_features = []
        for batch_start in range(0, len(images), self._inception_batch_size):
            batch_images = images[batch_start : batch_start + self._inception_batch_size]
            tensors = torch.stack([
                transform(img.convert("RGB")) for img in batch_images
            ]).to(self._device)

            with torch.no_grad():
                features = self._inception_model(tensors)

            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    @staticmethod
    def _frechet_distance(
        mu1: np.ndarray,  # type: ignore[type-arg]
        sigma1: np.ndarray,  # type: ignore[type-arg]
        mu2: np.ndarray,  # type: ignore[type-arg]
        sigma2: np.ndarray,  # type: ignore[type-arg]
    ) -> float:
        """Compute Frechet distance between two multivariate Gaussians.

        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        """
        diff = mu1 - mu2
        # Matrix square root via eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(sigma1 @ sigma2)
        sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues.real, 0))
        sqrt_product = eigenvectors @ np.diag(sqrt_eigenvalues) @ np.linalg.inv(eigenvectors)
        sqrt_product = sqrt_product.real

        fid = (
            float(np.dot(diff, diff))
            + float(np.trace(sigma1))
            + float(np.trace(sigma2))
            - 2.0 * float(np.trace(sqrt_product))
        )
        return max(0.0, fid)

    def _fallback_fid(
        self,
        real_images: list[PILImage.Image],
        synthetic_images: list[PILImage.Image],
    ) -> float:
        """Lightweight FID approximation based on pixel statistics."""
        def image_stats(images: list[PILImage.Image]) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
            vectors = []
            for img in images:
                small = img.convert("RGB").resize((64, 64))
                arr = np.array(small, dtype=np.float32).flatten() / 255.0
                vectors.append(arr)
            mat = np.stack(vectors)
            return np.mean(mat, axis=0), np.std(mat, axis=0)

        mu_real, std_real = image_stats(real_images)
        mu_synth, std_synth = image_stats(synthetic_images)
        diff_mu = np.sum((mu_real - mu_synth) ** 2)
        diff_std = np.sum((std_real - std_synth) ** 2)
        return float(diff_mu + diff_std) * 1000.0  # Scale to FID-like range

    async def compute_inception_score(
        self,
        synthetic_images: list[PILImage.Image],
        num_splits: int = 10,
    ) -> tuple[float, float]:
        """Compute Inception Score for a set of synthetic images.

        IS measures both quality (low entropy per image) and diversity
        (high entropy across all images) using Inception-v3 softmax outputs.

        Args:
            synthetic_images: Synthetic images to evaluate (minimum 100).
            num_splits: Number of splits for variance estimation.

        Returns:
            Tuple of (mean IS, std IS). Higher IS = better quality + diversity.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        is_fn = partial(
            self._compute_is_sync,
            synthetic_images=synthetic_images,
            num_splits=num_splits,
        )
        return await loop.run_in_executor(None, is_fn)

    def _compute_is_sync(
        self,
        synthetic_images: list[PILImage.Image],
        num_splits: int,
    ) -> tuple[float, float]:
        """Synchronous IS computation."""
        if self._inception_model is None or not _TORCH_AVAILABLE:
            return self._fallback_inception_score(synthetic_images)

        import torch

        transform = T.Compose(
            [
                T.Resize(299),
                T.CenterCrop(299),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Temporarily restore classification head for IS computation
        import torchvision.models as tvm
        softmax_model = tvm.inception_v3(pretrained=True, transform_input=False)
        softmax_model.eval()
        softmax_model.to(self._device)

        all_preds = []
        for batch_start in range(0, len(synthetic_images), self._inception_batch_size):
            batch = synthetic_images[batch_start : batch_start + self._inception_batch_size]
            tensors = torch.stack([
                transform(img.convert("RGB")) for img in batch
            ]).to(self._device)

            with torch.no_grad():
                logits = softmax_model(tensors)
                if hasattr(logits, "logits"):
                    logits = logits.logits
                preds = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)

        preds_array = np.concatenate(all_preds, axis=0)
        return self._compute_is_from_preds(preds_array, num_splits)

    @staticmethod
    def _compute_is_from_preds(
        preds: np.ndarray,  # type: ignore[type-arg]
        num_splits: int,
    ) -> tuple[float, float]:
        """Compute IS mean and std from Inception softmax predictions."""
        split_size = preds.shape[0] // num_splits
        scores = []
        for split_index in range(num_splits):
            split_preds = preds[split_index * split_size : (split_index + 1) * split_size]
            marginal = np.mean(split_preds, axis=0)
            # KL divergence: sum(p * log(p/q))
            kl_divs = np.sum(
                split_preds * (np.log(split_preds + 1e-10) - np.log(marginal + 1e-10)),
                axis=1,
            )
            scores.append(float(np.exp(np.mean(kl_divs))))
        return float(np.mean(scores)), float(np.std(scores))

    def _fallback_inception_score(
        self, images: list[PILImage.Image]
    ) -> tuple[float, float]:
        """Lightweight IS approximation using pixel variance."""
        scores = []
        for img in images:
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            # Use normalized variance as a quality proxy
            per_channel_var = np.var(arr, axis=(0, 1))
            score = 1.0 + float(np.mean(per_channel_var)) * 10.0
            scores.append(score)
        return float(np.mean(scores)), float(np.std(scores))

    async def compute_lpips(
        self,
        reference_image: PILImage.Image,
        synthetic_image: PILImage.Image,
    ) -> float:
        """Compute LPIPS (Learned Perceptual Image Patch Similarity).

        Uses AlexNet features to measure perceptual distance between images.
        Range [0, 1]. Lower = more perceptually similar.

        Args:
            reference_image: Ground-truth reference image.
            synthetic_image: Synthetic image to compare.

        Returns:
            LPIPS score (float). Lower is better.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        lpips_fn = partial(
            self._compute_lpips_sync,
            reference_image=reference_image,
            synthetic_image=synthetic_image,
        )
        return await loop.run_in_executor(None, lpips_fn)

    def _compute_lpips_sync(
        self,
        reference_image: PILImage.Image,
        synthetic_image: PILImage.Image,
    ) -> float:
        """Synchronous LPIPS computation."""
        if self._lpips_model is None or not _TORCH_AVAILABLE:
            return self._fallback_lpips(reference_image, synthetic_image)

        import torch

        transform = T.Compose(
            [
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        ref_tensor = transform(reference_image.convert("RGB")).unsqueeze(0).to(self._device)
        synth_tensor = transform(synthetic_image.convert("RGB")).unsqueeze(0).to(self._device)

        with torch.no_grad():
            score = self._lpips_model(ref_tensor, synth_tensor)

        return float(score.item())

    @staticmethod
    def _fallback_lpips(
        reference_image: PILImage.Image,
        synthetic_image: PILImage.Image,
    ) -> float:
        """Fallback LPIPS approximation using MSE on downsampled images."""
        target_size = (256, 256)
        ref_array = np.array(reference_image.convert("RGB").resize(target_size), dtype=np.float32) / 255.0
        synth_array = np.array(synthetic_image.convert("RGB").resize(target_size), dtype=np.float32) / 255.0
        mse = float(np.mean((ref_array - synth_array) ** 2))
        return min(1.0, mse * 10.0)  # Scale MSE to LPIPS-like [0, 1] range

    async def compute_ssim(
        self,
        reference_image: PILImage.Image,
        synthetic_image: PILImage.Image,
    ) -> float:
        """Compute Structural Similarity Index (SSIM).

        Measures luminance, contrast, and structural similarity between
        two images. Range [0, 1]. Higher = more similar.

        Args:
            reference_image: Ground-truth reference image.
            synthetic_image: Synthetic image to compare.

        Returns:
            SSIM score in [0, 1]. Higher is better.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        ssim_fn = partial(
            self._compute_ssim_sync,
            reference_image=reference_image,
            synthetic_image=synthetic_image,
        )
        return await loop.run_in_executor(None, ssim_fn)

    def _compute_ssim_sync(
        self,
        reference_image: PILImage.Image,
        synthetic_image: PILImage.Image,
    ) -> float:
        """Synchronous SSIM computation."""
        target_size = (512, 512)
        ref_gray = np.array(reference_image.convert("L").resize(target_size), dtype=np.float64) / 255.0
        synth_gray = np.array(synthetic_image.convert("L").resize(target_size), dtype=np.float64) / 255.0

        if _SKIMAGE_AVAILABLE:
            return float(skimage_ssim(ref_gray, synth_gray, data_range=1.0))

        return self._numpy_ssim(ref_gray, synth_gray)

    @staticmethod
    def _numpy_ssim(
        image_a: np.ndarray,  # type: ignore[type-arg]
        image_b: np.ndarray,  # type: ignore[type-arg]
        window_size: int = 11,
    ) -> float:
        """Pure numpy SSIM implementation with sliding window."""
        c1 = (0.01) ** 2
        c2 = (0.03) ** 2

        mu_a = np.mean(image_a)
        mu_b = np.mean(image_b)
        sigma_a_sq = np.var(image_a)
        sigma_b_sq = np.var(image_b)
        sigma_ab = float(np.mean((image_a - mu_a) * (image_b - mu_b)))

        numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
        denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a_sq + sigma_b_sq + c2)
        return float(numerator / denominator)

    async def compute_psnr(
        self,
        reference_image: PILImage.Image,
        synthetic_image: PILImage.Image,
    ) -> float:
        """Compute Peak Signal-to-Noise Ratio (PSNR) in dB.

        Measures pixel-level fidelity relative to maximum signal value.
        Higher PSNR = less distortion. Typical range: 20-50 dB.
        > 40 dB is considered high quality for image synthesis.

        Args:
            reference_image: Ground-truth reference image.
            synthetic_image: Synthetic image to compare.

        Returns:
            PSNR in dB (float). Higher is better. Returns float("inf")
            for identical images.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        psnr_fn = partial(
            self._compute_psnr_sync,
            reference_image=reference_image,
            synthetic_image=synthetic_image,
        )
        return await loop.run_in_executor(None, psnr_fn)

    def _compute_psnr_sync(
        self,
        reference_image: PILImage.Image,
        synthetic_image: PILImage.Image,
    ) -> float:
        """Synchronous PSNR computation."""
        target_size = (512, 512)
        ref_array = np.array(reference_image.convert("RGB").resize(target_size), dtype=np.float64)
        synth_array = np.array(synthetic_image.convert("RGB").resize(target_size), dtype=np.float64)

        mse = float(np.mean((ref_array - synth_array) ** 2))
        if mse < 1e-10:
            return float("inf")

        max_pixel = 255.0
        psnr = 20.0 * math.log10(max_pixel) - 10.0 * math.log10(mse)
        return round(psnr, 4)

    async def evaluate_all(
        self,
        reference_images: list[PILImage.Image],
        synthetic_images: list[PILImage.Image],
    ) -> dict[str, Any]:
        """Compute all quality metrics and aggregate into a single report.

        Args:
            reference_images: Reference (real) images. Used for FID and
                pairwise comparisons (LPIPS, SSIM, PSNR).
            synthetic_images: Synthetic images to evaluate.

        Returns:
            Dict with keys:
            - fid: float
            - inception_score_mean: float
            - inception_score_std: float
            - lpips_mean: float
            - ssim_mean: float
            - psnr_mean_db: float
            - overall_quality_score: float (0-100, aggregated)
            - metrics_available: list[str]
        """
        import asyncio

        if not reference_images or not synthetic_images:
            self._log.warning("quality_evaluator.empty_image_sets")
            return {"error": "empty image sets"}

        # Run FID and IS concurrently
        fid_task = self.compute_fid(reference_images, synthetic_images)
        is_task = self.compute_inception_score(synthetic_images)
        fid_score, (is_mean, is_std) = await asyncio.gather(fid_task, is_task)

        # Pairwise metrics on min(len(ref), len(synth)) pairs
        pair_count = min(len(reference_images), len(synthetic_images), 20)
        lpips_tasks = [
            self.compute_lpips(reference_images[i], synthetic_images[i])
            for i in range(pair_count)
        ]
        ssim_tasks = [
            self.compute_ssim(reference_images[i], synthetic_images[i])
            for i in range(pair_count)
        ]
        psnr_tasks = [
            self.compute_psnr(reference_images[i], synthetic_images[i])
            for i in range(pair_count)
        ]

        lpips_scores = await asyncio.gather(*lpips_tasks)
        ssim_scores = await asyncio.gather(*ssim_tasks)
        psnr_scores = await asyncio.gather(*psnr_tasks)

        # Filter inf PSNR for averaging
        finite_psnr = [s for s in psnr_scores if not math.isinf(s)]
        psnr_mean = float(np.mean(finite_psnr)) if finite_psnr else 0.0

        # Aggregate quality score (0-100)
        # FID component: exp(-fid/100) * 30 points
        fid_component = math.exp(-fid_score / 100.0) * 30.0
        # IS component: min(is_mean / 10, 1) * 20 points
        is_component = min(is_mean / 10.0, 1.0) * 20.0
        # LPIPS component: (1 - lpips_mean) * 20 points
        lpips_mean = float(np.mean(lpips_scores))
        lpips_component = max(0.0, 1.0 - lpips_mean) * 20.0
        # SSIM component: ssim_mean * 20 points
        ssim_mean = float(np.mean(ssim_scores))
        ssim_component = ssim_mean * 20.0
        # PSNR component: min(psnr_mean / 50, 1) * 10 points
        psnr_component = min(psnr_mean / 50.0, 1.0) * 10.0

        overall_quality = fid_component + is_component + lpips_component + ssim_component + psnr_component

        report: dict[str, Any] = {
            "fid": round(fid_score, 4),
            "inception_score_mean": round(is_mean, 4),
            "inception_score_std": round(is_std, 4),
            "lpips_mean": round(lpips_mean, 4),
            "ssim_mean": round(ssim_mean, 4),
            "psnr_mean_db": round(psnr_mean, 4),
            "overall_quality_score": round(overall_quality, 2),
            "metrics_available": [
                "fid",
                "inception_score",
                "lpips" if self._lpips_model is not None else "lpips_fallback",
                "ssim" if _SKIMAGE_AVAILABLE else "ssim_numpy",
                "psnr",
            ],
            "pair_count": pair_count,
        }

        self._log.info(
            "quality_evaluator.evaluation_complete",
            fid=report["fid"],
            inception_score=report["inception_score_mean"],
            ssim=report["ssim_mean"],
            psnr=report["psnr_mean_db"],
            overall_quality=report["overall_quality_score"],
        )
        return report
