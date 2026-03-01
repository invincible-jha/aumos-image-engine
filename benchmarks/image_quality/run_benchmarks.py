"""Image quality benchmark runner for aumos-image-engine.

Generates 1000 images with each registered model, computes FID vs COCO
validation set and Inception Score, and writes results to
benchmarks/results/image_quality.json.

Usage:
    python -m benchmarks.image_quality.run_benchmarks \\
        --real-images /data/coco/val2017 \\
        --output benchmarks/results/image_quality.json \\
        --num-images 1000 \\
        --models sd35 sdxl_turbo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from aumos_common.logging import get_logger

from benchmarks.image_quality.compute_fid import compute_fid, compute_inception_score

_BENCHMARK_PROMPTS: list[str] = [
    "a doctor examining a patient in a modern hospital room",
    "industrial machinery on a factory floor with safety equipment",
    "a business meeting in a corporate conference room",
    "medical imaging equipment in a radiology department",
    "laboratory scientist working with microscope and samples",
    "surgical team performing an operation in an operating room",
    "warehouse workers using forklifts in a logistics facility",
    "data center server racks with blinking indicators",
    "construction workers on a building site wearing hard hats",
    "pharmaceutical production line with cleanroom workers",
]


@dataclass
class ModelBenchmarkResult:
    """Results for a single model's benchmark run."""

    model_name: str
    num_images_generated: int
    fid_score: float
    inception_score_mean: float
    inception_score_std: float
    generation_time_seconds: float
    timestamp_utc: str
    prompts_used: int
    error: str | None = None


@dataclass
class BenchmarkSuite:
    """Aggregated results across all models."""

    run_id: str
    timestamp_utc: str
    real_images_dir: str
    num_images_per_model: int
    results: list[ModelBenchmarkResult]


async def generate_images_for_model(
    model_name: str,
    num_images: int,
    output_dir: Path,
    settings: Any,
    log: structlog.BoundLogger,
) -> tuple[float, int]:
    """Generate images using a specific model adapter.

    Args:
        model_name: Registered model name (e.g., "sd35", "sdxl_turbo").
        num_images: Number of images to generate.
        output_dir: Directory to save generated images.
        settings: ImageEngineSettings instance.
        log: Structured logger.

    Returns:
        Tuple of (elapsed_seconds, images_generated).
    """
    from aumos_image_engine.adapters.generators.model_registry import ModelAdapterRegistry

    import time

    output_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelAdapterRegistry(settings=settings)
    adapter = await registry.get_warmed(model_name)

    start_time = time.monotonic()
    generated_count = 0

    for i in range(num_images):
        prompt = _BENCHMARK_PROMPTS[i % len(_BENCHMARK_PROMPTS)]
        try:
            image_bytes = await adapter.generate(
                prompt=prompt,
                seed=i,  # Reproducible generation
            )
            image_path = output_dir / f"generated_{i:04d}.png"
            image_path.write_bytes(image_bytes)
            generated_count += 1
            if (i + 1) % 100 == 0:
                log.info(
                    "benchmark_progress",
                    model=model_name,
                    generated=generated_count,
                    total=num_images,
                )
        except Exception:
            log.exception("image_generation_failed", model=model_name, index=i)

    elapsed = time.monotonic() - start_time
    return elapsed, generated_count


async def run_model_benchmark(
    model_name: str,
    real_images_dir: str,
    num_images: int,
    output_base_dir: Path,
    settings: Any,
    log: structlog.BoundLogger,
) -> ModelBenchmarkResult:
    """Run a complete benchmark for one model: generate + FID + IS.

    Args:
        model_name: Registered model name.
        real_images_dir: Directory of real reference images for FID.
        num_images: Number of images to generate.
        output_base_dir: Base directory for generated image subdirectories.
        settings: ImageEngineSettings instance.
        log: Structured logger.

    Returns:
        ModelBenchmarkResult with all computed metrics.
    """
    generated_dir = output_base_dir / model_name
    log.info("starting_model_benchmark", model=model_name, num_images=num_images)

    try:
        elapsed_seconds, images_generated = await generate_images_for_model(
            model_name=model_name,
            num_images=num_images,
            output_dir=generated_dir,
            settings=settings,
            log=log,
        )

        log.info("computing_fid", model=model_name)
        fid_score = await compute_fid(
            real_images_dir=real_images_dir,
            generated_images_dir=str(generated_dir),
        )

        log.info("computing_inception_score", model=model_name)
        is_mean, is_std = await compute_inception_score(
            generated_images_dir=str(generated_dir),
        )

        log.info(
            "model_benchmark_complete",
            model=model_name,
            fid=fid_score,
            is_mean=is_mean,
            is_std=is_std,
            elapsed_s=elapsed_seconds,
        )

        return ModelBenchmarkResult(
            model_name=model_name,
            num_images_generated=images_generated,
            fid_score=fid_score,
            inception_score_mean=is_mean,
            inception_score_std=is_std,
            generation_time_seconds=elapsed_seconds,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            prompts_used=len(_BENCHMARK_PROMPTS),
        )

    except Exception as exc:
        log.exception("model_benchmark_failed", model=model_name, error=str(exc))
        return ModelBenchmarkResult(
            model_name=model_name,
            num_images_generated=0,
            fid_score=-1.0,
            inception_score_mean=-1.0,
            inception_score_std=-1.0,
            generation_time_seconds=0.0,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            prompts_used=0,
            error=str(exc),
        )


async def main() -> None:
    """Run image quality benchmarks for specified models."""
    parser = argparse.ArgumentParser(description="aumos-image-engine quality benchmarks")
    parser.add_argument(
        "--real-images",
        required=True,
        help="Directory containing real reference images (e.g., COCO val2017)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/image_quality.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1000,
        help="Number of images to generate per model",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sd35", "sdxl_turbo"],
        help="Model names to benchmark",
    )
    parser.add_argument(
        "--temp-dir",
        default="/tmp/image_benchmarks",
        help="Temporary directory for generated images",
    )

    args = parser.parse_args()
    log = get_logger(__name__)

    from aumos_image_engine.settings import ImageEngineSettings
    settings = ImageEngineSettings()

    output_base = Path(args.temp_dir)
    results: list[ModelBenchmarkResult] = []

    for model_name in args.models:
        result = await run_model_benchmark(
            model_name=model_name,
            real_images_dir=args.real_images,
            num_images=args.num_images,
            output_base_dir=output_base,
            settings=settings,
            log=log,
        )
        results.append(result)

    suite = BenchmarkSuite(
        run_id=str(uuid.uuid4()),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        real_images_dir=args.real_images,
        num_images_per_model=args.num_images,
        results=results,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(suite), indent=2))

    log.info(
        "benchmarks_complete",
        output=args.output,
        models_benchmarked=len(results),
        results=[
            {"model": r.model_name, "fid": r.fid_score, "is_mean": r.inception_score_mean}
            for r in results
        ],
    )


if __name__ == "__main__":
    asyncio.run(main())
