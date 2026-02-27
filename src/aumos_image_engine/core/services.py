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
- QualityService: image quality metric evaluation (FID, IS, LPIPS, SSIM, PSNR)
- MedicalImagingService: DICOM creation, validation, and anonymization pipeline
- ExportService: multi-format export with storage upload and thumbnails
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
    ImageExportProtocol,
    ImageGeneratorProtocol,
    ImageQualityEvaluatorProtocol,
    MedicalImagingProtocol,
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


class QualityService:
    """Evaluates quality of synthetic image sets using standard metrics.

    Wraps the ImageQualityEvaluatorProtocol to provide a service-level
    interface that logs progress, validates inputs, and returns structured
    quality reports consumable by the API layer.

    Metrics computed:
    - FID: distributional realism vs. real image reference set
    - IS: combined quality and diversity score
    - LPIPS: perceptual similarity per image pair
    - SSIM: structural similarity per image pair
    - PSNR: pixel-level signal quality per image pair
    """

    def __init__(
        self,
        quality_evaluator: ImageQualityEvaluatorProtocol,
        minimum_reference_images: int = 10,
    ) -> None:
        """Initialize the quality service.

        Args:
            quality_evaluator: Quality metric computation adapter.
            minimum_reference_images: Minimum reference images required
                for FID/IS computation. Requests below this threshold
                will get a warning in the report.
        """
        self._evaluator = quality_evaluator
        self._min_reference = minimum_reference_images
        self._log = logger.bind(service="quality_service")

    async def evaluate_batch_quality(
        self,
        job_id: uuid.UUID,
        reference_images: list[PILImage.Image],
        synthetic_images: list[PILImage.Image],
        compute_pairwise: bool = True,
    ) -> dict[str, Any]:
        """Evaluate quality of a synthetic image batch.

        Args:
            job_id: Generation job identifier for logging.
            reference_images: Real images for distributional comparison.
            synthetic_images: Synthetic images to assess.
            compute_pairwise: If True, compute LPIPS/SSIM/PSNR pairwise.
                Disable for large batches when only FID/IS is needed.

        Returns:
            Dict with all quality metrics plus a warnings list and
            an overall_quality_score (0-100).
        """
        log = self._log.bind(job_id=str(job_id))
        log.info(
            "quality.evaluation.start",
            reference_count=len(reference_images),
            synthetic_count=len(synthetic_images),
        )

        warnings: list[str] = []
        if len(reference_images) < self._min_reference:
            warnings.append(
                f"Reference image count ({len(reference_images)}) is below the "
                f"recommended minimum of {self._min_reference}. "
                "FID and IS scores may be unreliable."
            )

        if not synthetic_images:
            log.warning("quality.evaluation.empty_synthetic_set")
            return {"error": "No synthetic images provided", "warnings": warnings}

        report = await self._evaluator.evaluate_all(
            reference_images=reference_images,
            synthetic_images=synthetic_images,
        )
        report["warnings"] = warnings
        report["job_id"] = str(job_id)
        report["reference_image_count"] = len(reference_images)
        report["synthetic_image_count"] = len(synthetic_images)

        log.info(
            "quality.evaluation.complete",
            fid=report.get("fid"),
            overall_quality=report.get("overall_quality_score"),
        )
        return report

    async def compute_single_pair_metrics(
        self,
        reference_image: PILImage.Image,
        synthetic_image: PILImage.Image,
    ) -> dict[str, Any]:
        """Compute pairwise quality metrics for a single image pair.

        Args:
            reference_image: Ground-truth reference.
            synthetic_image: Synthetic image to compare against reference.

        Returns:
            Dict with lpips, ssim, psnr metrics.
        """
        lpips_task = self._evaluator.compute_lpips(reference_image, synthetic_image)
        ssim_task = self._evaluator.compute_ssim(reference_image, synthetic_image)
        psnr_task = self._evaluator.compute_psnr(reference_image, synthetic_image)

        lpips_score, ssim_score, psnr_score = await asyncio.gather(
            lpips_task, ssim_task, psnr_task
        )
        return {
            "lpips": round(lpips_score, 6),
            "ssim": round(ssim_score, 6),
            "psnr_db": round(psnr_score, 4),
        }


class MedicalImagingService:
    """Orchestrates DICOM medical image creation and validation pipeline.

    Integrates with aumos-healthcare-synth for anatomy-specific generation
    parameters. Handles the full lifecycle: PIL -> DICOM -> validate ->
    anonymize -> return bytes.

    The service enforces that all generated DICOM files pass validation
    before being returned, and that patient data is fully anonymized.
    """

    def __init__(
        self,
        medical_imaging_adapter: MedicalImagingProtocol,
        enforce_validation: bool = True,
        enforce_anonymization: bool = True,
    ) -> None:
        """Initialize the medical imaging service.

        Args:
            medical_imaging_adapter: DICOM creation and validation adapter.
            enforce_validation: If True, raise ValueError on invalid DICOM.
            enforce_anonymization: If True, always anonymize before returning.
        """
        self._adapter = medical_imaging_adapter
        self._enforce_validation = enforce_validation
        self._enforce_anonymization = enforce_anonymization
        self._log = logger.bind(service="medical_imaging_service")

    async def synthesize_dicom(
        self,
        job_id: uuid.UUID,
        tenant_id: uuid.UUID,
        image: PILImage.Image,
        modality: str,
        anatomy: str,
        study_uid: str | None = None,
        series_uid: str | None = None,
        acquisition_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create, validate, and optionally anonymize a synthetic DICOM file.

        Args:
            job_id: Generation job identifier.
            tenant_id: Owning tenant (embedded in DICOM patient ID).
            image: Synthetic source image.
            modality: DICOM modality code (CT, MR, DX, US, PT).
            anatomy: Anatomy profile name (e.g., "chest_xray", "brain_mri").
            study_uid: Optional Study Instance UID.
            series_uid: Optional Series Instance UID.
            acquisition_params: Acquisition parameter overrides.

        Returns:
            Dict with:
            - dicom_bytes: bytes — validated and anonymized DICOM file
            - modality: str
            - anatomy: str
            - validation_result: dict — IOD compliance report
            - size_bytes: int
        """
        log = self._log.bind(job_id=str(job_id), modality=modality, anatomy=anatomy)
        log.info("medical_imaging.synthesize.start")

        synthetic_patient_id = f"SYN-{job_id.hex[:12].upper()}-{tenant_id.hex[:4].upper()}"

        # Create DICOM
        dicom_bytes = await self._adapter.create_dicom_from_pil(
            image=image,
            modality=modality,
            anatomy=anatomy,
            synthetic_patient_id=synthetic_patient_id,
            study_uid=study_uid,
            series_uid=series_uid,
            acquisition_params=acquisition_params,
        )
        log.info("medical_imaging.synthesize.created", size_bytes=len(dicom_bytes))

        # Validate
        validation_result = await self._adapter.validate_dicom(dicom_bytes)
        if self._enforce_validation and not validation_result.get("valid", False):
            errors = validation_result.get("errors", [])
            log.error("medical_imaging.synthesize.validation_failed", errors=errors)
            raise ValueError(f"DICOM validation failed: {'; '.join(errors)}")

        # Anonymize
        if self._enforce_anonymization:
            dicom_bytes = await self._adapter.anonymize_dicom(dicom_bytes)
            log.info("medical_imaging.synthesize.anonymized", size_bytes=len(dicom_bytes))

        log.info("medical_imaging.synthesize.complete")
        return {
            "dicom_bytes": dicom_bytes,
            "modality": modality,
            "anatomy": anatomy,
            "validation_result": validation_result,
            "size_bytes": len(dicom_bytes),
            "synthetic_patient_id": synthetic_patient_id,
        }

    async def validate_external_dicom(
        self,
        dicom_bytes: bytes,
    ) -> dict[str, Any]:
        """Validate an externally provided DICOM file.

        Args:
            dicom_bytes: Raw DICOM bytes to validate.

        Returns:
            Validation result dict from the adapter.
        """
        result = await self._adapter.validate_dicom(dicom_bytes)
        self._log.info(
            "medical_imaging.external_validation",
            valid=result.get("valid"),
            modality=result.get("modality"),
        )
        return result


class ExportService:
    """Orchestrates image export, format conversion, and storage upload.

    Provides a unified interface for exporting generated or de-identified
    images to various formats and uploading to MinIO object storage with
    automatic thumbnail generation and presigned URL support.
    """

    def __init__(
        self,
        export_handler: ImageExportProtocol,
        default_format: str = "PNG",
        always_generate_thumbnail: bool = True,
    ) -> None:
        """Initialize the export service.

        Args:
            export_handler: Format encoding and storage upload adapter.
            default_format: Default output format when not specified.
            always_generate_thumbnail: Always generate a thumbnail on export.
        """
        self._handler = export_handler
        self._default_format = default_format
        self._always_generate_thumbnail = always_generate_thumbnail
        self._log = logger.bind(service="export_service")

    async def export_image(
        self,
        job_id: uuid.UUID,
        tenant_id: uuid.UUID,
        image: PILImage.Image,
        output_format: str | None = None,
        export_options: dict[str, Any] | None = None,
        bucket: str | None = None,
    ) -> dict[str, Any]:
        """Export and upload a single image.

        Args:
            job_id: Generation job identifier.
            tenant_id: Owning tenant.
            image: Image to export.
            output_format: Target format (PNG, JPEG, WEBP, TIFF).
                Defaults to service default_format.
            export_options: Format-specific options (quality, compression, etc.).
            bucket: Override default storage bucket.

        Returns:
            Dict with object_name, bucket, thumbnail_object_name,
            size_bytes, format, etag, presigned_url.
        """
        effective_format = (output_format or self._default_format).upper()
        log = self._log.bind(job_id=str(job_id), format=effective_format)
        log.info("export.start")

        result = await self._handler.export_and_upload(
            image=image,
            output_format=effective_format,
            job_id=job_id,
            tenant_id=tenant_id,
            export_options=export_options or {},
            generate_thumbnail=self._always_generate_thumbnail,
            bucket=bucket,
        )

        log.info(
            "export.complete",
            object_name=result.get("object_name"),
            size_bytes=result.get("size_bytes"),
        )
        return result

    async def export_batch(
        self,
        job_id: uuid.UUID,
        tenant_id: uuid.UUID,
        images: list[PILImage.Image],
        output_format: str | None = None,
        export_options: dict[str, Any] | None = None,
        max_concurrency: int = 4,
        bucket: str | None = None,
    ) -> list[dict[str, Any]]:
        """Export and upload a batch of images concurrently.

        Args:
            job_id: Batch job identifier (shared across all images).
            tenant_id: Owning tenant.
            images: List of images to export.
            output_format: Target format for all images.
            export_options: Format-specific options.
            max_concurrency: Maximum concurrent uploads.
            bucket: Override default storage bucket.

        Returns:
            List of export result dicts in the same order as input images.
        """
        log = self._log.bind(job_id=str(job_id), total_images=len(images))
        log.info("export.batch.start")

        semaphore = asyncio.Semaphore(max_concurrency)

        async def export_one(image_index: int, image: PILImage.Image) -> dict[str, Any]:
            async with semaphore:
                # Give each image a unique sub-job ID for storage path differentiation
                sub_job_id = uuid.UUID(
                    hex=f"{job_id.hex[:24]}{image_index:08x}"[:32]
                )
                try:
                    return await self.export_image(
                        job_id=sub_job_id,
                        tenant_id=tenant_id,
                        image=image,
                        output_format=output_format,
                        export_options=export_options,
                        bucket=bucket,
                    )
                except Exception as exc:
                    log.error("export.batch.image_failed", index=image_index, error=str(exc))
                    return {
                        "index": image_index,
                        "error": str(exc),
                        "object_name": None,
                    }

        tasks = [export_one(idx, img) for idx, img in enumerate(images)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        successful = sum(1 for r in results if r.get("object_name") is not None)
        log.info("export.batch.complete", successful=successful, total=len(images))
        return list(results)
