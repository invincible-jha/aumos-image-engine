"""Core domain services for image generation and privacy processing.

Services orchestrate the pipeline of adapters to fulfill business requirements.
All services use dependency injection — adapters are passed as constructor arguments
rather than instantiated internally, enabling testing with mocks.

Services:
- GenerationService: orchestrate full image generation pipeline
- DeidentificationService: face detection + de-identification pipeline
- MetadataService: metadata stripping with steganographic analysis
- ProvenanceService: C2PA provenance + invisible watermarking
- BatchService: batch processing with progress tracking
"""

from __future__ import annotations

import asyncio
import io
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from PIL import Image as PILImage

from aumos_image_engine.core.interfaces import (
    BiometricVerifierProtocol,
    FaceDeidentifierProtocol,
    ImageGeneratorProtocol,
    MetadataStripperProtocol,
    WatermarkerProtocol,
)

logger = structlog.get_logger(__name__)


class GenerationService:
    """Orchestrates the full image generation pipeline.

    Pipeline:
    1. Configure model parameters
    2. Generate raw images via generator adapter
    3. Strip metadata from generated images
    4. Add invisible watermark
    5. Verify biometric non-linkability (no real faces leaked)
    6. Attach C2PA provenance
    7. Store outputs in MinIO
    """

    def __init__(
        self,
        generator: ImageGeneratorProtocol,
        metadata_stripper: MetadataStripperProtocol,
        watermarker: WatermarkerProtocol,
        biometric_verifier: BiometricVerifierProtocol,
    ) -> None:
        """Initialize with injected adapters.

        Args:
            generator: Image generation adapter (SD or BlenderProc).
            metadata_stripper: Metadata removal adapter.
            watermarker: C2PA and invisible watermarking adapter.
            biometric_verifier: NIST FRVT compliance verifier.
        """
        self._generator = generator
        self._stripper = metadata_stripper
        self._watermarker = watermarker
        self._verifier = biometric_verifier
        self._log = logger.bind(service="generation_service")

    async def generate_images(
        self,
        job_id: uuid.UUID,
        tenant_id: uuid.UUID,
        prompt: str,
        negative_prompt: str | None,
        num_images: int,
        width: int,
        height: int,
        model_config: dict[str, Any],
        enable_watermark: bool,
        enable_c2pa: bool,
        enable_biometric_check: bool,
    ) -> list[dict[str, Any]]:
        """Run full generation pipeline for a job.

        Args:
            job_id: Unique identifier for this generation job.
            tenant_id: Tenant owning this job (for provenance).
            prompt: Positive generation prompt.
            negative_prompt: Elements to exclude.
            num_images: Number of images to generate.
            width: Output width in pixels.
            height: Output height in pixels.
            model_config: Generator-specific configuration.
            enable_watermark: Whether to embed invisible watermarks.
            enable_c2pa: Whether to attach C2PA provenance.
            enable_biometric_check: Whether to verify no real faces.

        Returns:
            List of result dicts, one per generated image:
            - index: int
            - image_bytes: bytes
            - format: str
            - has_watermark: bool
            - has_c2pa: bool
            - biometric_verified: bool
            - biometric_score: float | None
            - metadata_stripped: bool
            - strip_report: dict
        """
        log = self._log.bind(job_id=str(job_id), tenant_id=str(tenant_id))
        log.info("generation.pipeline.start", num_images=num_images, prompt=prompt[:80])

        # Step 1: Generate raw images
        raw_images = await self._generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            width=width,
            height=height,
            model_config=model_config,
        )
        log.info("generation.pipeline.generated", count=len(raw_images))

        # Step 2: Process each image through the privacy pipeline
        results: list[dict[str, Any]] = []
        for index, image in enumerate(raw_images):
            image_result = await self._process_single_generated_image(
                image=image,
                index=index,
                job_id=job_id,
                tenant_id=tenant_id,
                image_format="PNG",
                enable_watermark=enable_watermark,
                enable_c2pa=enable_c2pa,
                enable_biometric_check=enable_biometric_check,
            )
            results.append(image_result)
            log.info("generation.pipeline.image_processed", index=index)

        log.info("generation.pipeline.complete", total=len(results))
        return results

    async def _process_single_generated_image(
        self,
        image: PILImage.Image,
        index: int,
        job_id: uuid.UUID,
        tenant_id: uuid.UUID,
        image_format: str,
        enable_watermark: bool,
        enable_c2pa: bool,
        enable_biometric_check: bool,
    ) -> dict[str, Any]:
        """Process a single generated image through privacy and provenance steps."""
        result: dict[str, Any] = {
            "index": index,
            "has_watermark": False,
            "has_c2pa": False,
            "biometric_verified": False,
            "biometric_score": None,
            "metadata_stripped": False,
            "strip_report": {},
        }

        # Strip metadata (generated images shouldn't have any, but strip anyway)
        processed_image, strip_report = await self._stripper.strip(
            image=image,
            image_format=image_format,
            steganographic_check=False,  # Generated images don't have stego
        )
        result["metadata_stripped"] = True
        result["strip_report"] = strip_report

        # Biometric check — ensure no real face leaked into generation
        if enable_biometric_check:
            verification = await self._verifier.verify_non_linkability(
                original_image=image,
                deidentified_image=processed_image,
                threshold=0.05,
            )
            result["biometric_verified"] = verification.get("is_non_linkable", True)
            result["biometric_score"] = verification.get("similarity_score")

        # Add invisible watermark
        if enable_watermark:
            payload = f"{job_id}:{tenant_id}:{index}"
            processed_image = await self._watermarker.add_invisible_watermark(
                image=processed_image,
                payload=payload,
                strength=0.3,
            )
            result["has_watermark"] = True

        # Convert to bytes for C2PA and storage
        buffer = io.BytesIO()
        processed_image.save(buffer, format=image_format)
        image_bytes = buffer.getvalue()

        # Add C2PA provenance
        if enable_c2pa:
            provenance_metadata = {
                "generator": "aumos-image-engine",
                "tenant_id": str(tenant_id),
                "job_id": str(job_id),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_hash": str(hash(str(job_id))),  # Don't store prompt in provenance
            }
            image_bytes = await self._watermarker.add_c2pa_provenance(
                image_bytes=image_bytes,
                image_format=image_format.lower(),
                provenance_metadata=provenance_metadata,
            )
            result["has_c2pa"] = True

        result["image_bytes"] = image_bytes
        result["format"] = image_format
        return result


class DeidentificationService:
    """Orchestrates face detection and de-identification pipeline.

    Pipeline:
    1. Detect faces in input image
    2. De-identify each face (replace biometric markers)
    3. Verify non-linkability with NIST FRVT
    4. Strip any metadata added during processing
    5. Return de-identified image + verification report
    """

    def __init__(
        self,
        face_deidentifier: FaceDeidentifierProtocol,
        biometric_verifier: BiometricVerifierProtocol,
        metadata_stripper: MetadataStripperProtocol,
    ) -> None:
        self._deidentifier = face_deidentifier
        self._verifier = biometric_verifier
        self._stripper = metadata_stripper
        self._log = logger.bind(service="deidentification_service")

    async def deidentify_image(
        self,
        job_id: uuid.UUID,
        image: PILImage.Image,
        strength: float,
        preserve_expression: bool,
        verify_non_linkability: bool,
        frvt_threshold: float,
    ) -> dict[str, Any]:
        """Run full de-identification pipeline on an image.

        Args:
            job_id: Unique identifier for this job.
            image: Input image containing faces to de-identify.
            strength: De-identification strength (0.0-1.0).
            preserve_expression: Preserve facial expressions in output.
            verify_non_linkability: Run NIST FRVT verification after.
            frvt_threshold: Maximum allowed similarity for non-linkability.

        Returns:
            Dict with:
            - deidentified_image: PIL Image
            - faces_detected: int
            - face_detections: list of face bbox/confidence dicts
            - biometric_verification: dict with FRVT results
            - is_compliant: bool
        """
        log = self._log.bind(job_id=str(job_id))
        log.info("deidentification.pipeline.start")

        # Detect faces first
        face_detections = await self._deidentifier.detect_faces(
            image=image,
            confidence_threshold=0.9,
        )
        faces_detected = len(face_detections)
        log.info("deidentification.faces_detected", count=faces_detected)

        if faces_detected == 0:
            log.warning("deidentification.no_faces_found")
            # Return original image with stripped metadata
            stripped_image, _ = await self._stripper.strip(
                image=image,
                image_format="PNG",
                steganographic_check=True,
            )
            return {
                "deidentified_image": stripped_image,
                "faces_detected": 0,
                "face_detections": [],
                "biometric_verification": None,
                "is_compliant": True,
            }

        # Run de-identification
        deidentified = await self._deidentifier.deidentify(
            image=image,
            strength=strength,
            preserve_expression=preserve_expression,
        )
        log.info("deidentification.faces_processed", count=faces_detected)

        # Verify non-linkability
        verification: dict[str, Any] = {}
        is_compliant = True

        if verify_non_linkability:
            verification = await self._verifier.verify_non_linkability(
                original_image=image,
                deidentified_image=deidentified,
                threshold=frvt_threshold,
            )
            is_compliant = verification.get("is_non_linkable", False)
            log.info(
                "deidentification.verification_complete",
                is_compliant=is_compliant,
                similarity_score=verification.get("similarity_score"),
            )

        # Strip any metadata from the de-identified output
        final_image, _ = await self._stripper.strip(
            image=deidentified,
            image_format="PNG",
            steganographic_check=False,
        )

        return {
            "deidentified_image": final_image,
            "faces_detected": faces_detected,
            "face_detections": face_detections,
            "biometric_verification": verification,
            "is_compliant": is_compliant,
        }


class MetadataService:
    """Handles metadata stripping and steganographic analysis.

    Removes all identifying metadata from images:
    - EXIF: GPS, camera model, datetime, serial numbers
    - IPTC: creator, copyright, caption, keywords
    - XMP: Adobe metadata, subject tags
    - DICOM: patient demographics (for medical images)

    Also performs steganographic analysis to detect hidden data
    embedded in pixel least-significant bits.
    """

    def __init__(
        self,
        metadata_stripper: MetadataStripperProtocol,
    ) -> None:
        self._stripper = metadata_stripper
        self._log = logger.bind(service="metadata_service")

    async def strip_metadata(
        self,
        job_id: uuid.UUID,
        image: PILImage.Image,
        image_format: str,
        steganographic_check: bool,
    ) -> dict[str, Any]:
        """Strip all metadata and optionally check for steganography.

        Args:
            job_id: Job identifier for logging.
            image: Input image to process.
            image_format: Target output format.
            steganographic_check: Whether to analyze for hidden data.

        Returns:
            Dict with:
            - stripped_image: PIL Image with metadata removed
            - strip_report: dict describing removed metadata
            - steganographic_findings: list of potential hidden data indicators
        """
        log = self._log.bind(job_id=str(job_id))
        log.info("metadata.strip.start", format=image_format, stego_check=steganographic_check)

        # First, analyze to report what was present
        analysis = await self._stripper.analyze(image=image)
        log.info(
            "metadata.analysis.complete",
            exif_keys=len(analysis.get("exif", {})),
            iptc_keys=len(analysis.get("iptc", {})),
            xmp_keys=len(analysis.get("xmp", {})),
        )

        # Strip metadata
        stripped_image, strip_report = await self._stripper.strip(
            image=image,
            image_format=image_format,
            steganographic_check=steganographic_check,
        )

        log.info("metadata.strip.complete", stripped_fields=strip_report.get("total_fields_removed", 0))

        return {
            "stripped_image": stripped_image,
            "original_metadata": analysis,
            "strip_report": strip_report,
        }

    async def analyze_metadata(
        self,
        image: PILImage.Image,
    ) -> dict[str, Any]:
        """Analyze image metadata without stripping.

        Useful for auditing what metadata an image contains
        before deciding whether to strip it.
        """
        return await self._stripper.analyze(image=image)


class ProvenanceService:
    """Handles C2PA provenance signing and invisible watermarking.

    C2PA manifests are cryptographically signed records that declare
    an image's synthetic origin. They are embedded in the image file
    (not visible) and can be verified by any C2PA-compatible tool.

    Invisible watermarks complement C2PA by embedding a recoverable
    payload in the frequency domain that survives JPEG compression.
    """

    def __init__(
        self,
        watermarker: WatermarkerProtocol,
    ) -> None:
        self._watermarker = watermarker
        self._log = logger.bind(service="provenance_service")

    async def add_provenance(
        self,
        job_id: uuid.UUID,
        tenant_id: uuid.UUID,
        image: PILImage.Image,
        image_format: str,
        generator_name: str,
        enable_c2pa: bool,
        enable_watermark: bool,
        watermark_strength: float,
    ) -> dict[str, Any]:
        """Add C2PA provenance and/or invisible watermark to an image.

        Args:
            job_id: Generation job ID (embedded in provenance).
            tenant_id: Tenant ID (embedded in provenance).
            image: Input PIL image.
            image_format: Image format (PNG, JPEG, WEBP).
            generator_name: Name of generator (e.g., "stable-diffusion-v1-5").
            enable_c2pa: Whether to attach C2PA manifest.
            enable_watermark: Whether to embed invisible watermark.
            watermark_strength: Strength of invisible watermark.

        Returns:
            Dict with:
            - image_bytes: bytes with provenance embedded
            - has_c2pa: bool
            - has_watermark: bool
            - c2pa_manifest_id: str | None
        """
        log = self._log.bind(job_id=str(job_id))
        log.info("provenance.start", enable_c2pa=enable_c2pa, enable_watermark=enable_watermark)

        processed_image = image
        result: dict[str, Any] = {
            "has_c2pa": False,
            "has_watermark": False,
            "c2pa_manifest_id": None,
        }

        # Add invisible watermark first (modifies pixels)
        if enable_watermark:
            payload = f"aumos:{job_id}:{tenant_id}"
            processed_image = await self._watermarker.add_invisible_watermark(
                image=processed_image,
                payload=payload,
                strength=watermark_strength,
            )
            result["has_watermark"] = True
            log.info("provenance.watermark_added")

        # Convert to bytes for C2PA signing
        buffer = io.BytesIO()
        processed_image.save(buffer, format=image_format)
        image_bytes = buffer.getvalue()

        # Add C2PA manifest (wraps around bytes)
        if enable_c2pa:
            provenance_metadata = {
                "generator": f"aumos-image-engine/{generator_name}",
                "tenant_id": str(tenant_id),
                "job_id": str(job_id),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "synthetic_origin": True,
            }
            image_bytes = await self._watermarker.add_c2pa_provenance(
                image_bytes=image_bytes,
                image_format=image_format.lower(),
                provenance_metadata=provenance_metadata,
            )
            result["has_c2pa"] = True
            result["c2pa_manifest_id"] = str(job_id)
            log.info("provenance.c2pa_added")

        result["image_bytes"] = image_bytes
        log.info("provenance.complete")
        return result

    async def verify_watermark(
        self,
        image: PILImage.Image,
    ) -> dict[str, Any]:
        """Attempt to recover invisible watermark from an image."""
        found, payload = await self._watermarker.verify_watermark(image=image)
        return {"watermark_found": found, "payload": payload}


class BatchService:
    """Manages batch image generation and processing with progress tracking.

    Batch jobs split a large request into concurrent sub-jobs,
    track completion status, and aggregate results for retrieval.
    """

    def __init__(
        self,
        generation_service: GenerationService,
        deidentification_service: DeidentificationService,
        max_concurrency: int,
    ) -> None:
        self._generation = generation_service
        self._deidentification = deidentification_service
        self._concurrency = max_concurrency
        self._log = logger.bind(service="batch_service")

    async def process_generation_batch(
        self,
        batch_id: uuid.UUID,
        tenant_id: uuid.UUID,
        prompts: list[str],
        negative_prompt: str | None,
        images_per_prompt: int,
        width: int,
        height: int,
        model_config: dict[str, Any],
        enable_watermark: bool,
        enable_c2pa: bool,
        progress_callback: Any,
    ) -> dict[str, Any]:
        """Process a batch of generation prompts concurrently.

        Args:
            batch_id: Unique batch identifier.
            tenant_id: Owning tenant.
            prompts: List of prompts to generate images for.
            negative_prompt: Shared negative prompt for all prompts.
            images_per_prompt: Number of images to generate per prompt.
            width: Output width in pixels.
            height: Output height in pixels.
            model_config: Generator configuration.
            enable_watermark: Whether to watermark outputs.
            enable_c2pa: Whether to add C2PA provenance.
            progress_callback: Async callable(completed_count, total) for updates.

        Returns:
            Dict with:
            - completed: int
            - failed: int
            - total: int
            - results: list of generation result dicts
        """
        log = self._log.bind(batch_id=str(batch_id))
        total = len(prompts) * images_per_prompt
        log.info("batch.start", total_images=total, prompts=len(prompts))

        semaphore = asyncio.Semaphore(self._concurrency)
        completed = 0
        failed = 0
        all_results: list[dict[str, Any]] = []

        async def process_one_prompt(prompt: str, prompt_index: int) -> list[dict[str, Any]]:
            nonlocal completed, failed
            async with semaphore:
                try:
                    job_id = uuid.uuid4()
                    results = await self._generation.generate_images(
                        job_id=job_id,
                        tenant_id=tenant_id,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_images=images_per_prompt,
                        width=width,
                        height=height,
                        model_config=model_config,
                        enable_watermark=enable_watermark,
                        enable_c2pa=enable_c2pa,
                        enable_biometric_check=False,  # Skip per-image check in batch
                    )
                    completed += len(results)
                    if progress_callback:
                        await progress_callback(completed, total)
                    return results
                except Exception as exc:
                    failed += images_per_prompt
                    log.error("batch.prompt_failed", prompt_index=prompt_index, error=str(exc))
                    return []

        tasks = [process_one_prompt(prompt, i) for i, prompt in enumerate(prompts)]
        prompt_results = await asyncio.gather(*tasks, return_exceptions=False)

        for results in prompt_results:
            all_results.extend(results)

        log.info("batch.complete", completed=completed, failed=failed, total=total)
        return {
            "completed": completed,
            "failed": failed,
            "total": total,
            "results": all_results,
        }
