"""FastAPI router for the image engine API.

Endpoints:
- POST /images/generate — Generate synthetic images from a prompt
- POST /images/deidentify — De-identify faces in an uploaded image
- POST /images/strip-metadata — Remove EXIF/IPTC/XMP/DICOM metadata
- POST /images/verify-biometric — Verify NIST FRVT non-linkability
- GET  /images/jobs/{job_id} — Poll job status
- POST /images/batch — Submit batch generation request

All endpoints require a tenant ID header (X-Tenant-ID).
Authentication is handled by the upstream API gateway (aumos-auth-gateway).
"""

from __future__ import annotations

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from fastapi.responses import Response

from aumos_image_engine.api.schemas import (
    BatchGenerationRequest,
    BatchStatusResponse,
    DeidentifyImageRequest,
    DeidentifyImageResponse,
    FinetuneJobRequest,
    FinetuneJobResponse,
    FinetuneJobStatusResponse,
    GenerateImageRequest,
    GenerateImageResponse,
    InpaintRequest,
    JobStatus,
    JobStatusResponse,
    JobType,
    StripMetadataRequest,
    StripMetadataResponse,
    SyncGenerateRequest,
    VerifyBiometricRequest,
    VerifyBiometricResponse,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


def get_tenant_id(
    x_tenant_id: Annotated[str | None, Header()] = None,
) -> uuid.UUID:
    """Extract and validate tenant ID from request header.

    The auth gateway injects X-Tenant-ID after verifying the JWT.
    This dependency ensures all endpoints have a valid tenant context.
    """
    if not x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Tenant-ID header",
        )
    try:
        return uuid.UUID(x_tenant_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tenant ID format: {x_tenant_id}",
        ) from exc


TenantDep = Annotated[uuid.UUID, Depends(get_tenant_id)]


@router.post(
    "/generate",
    response_model=GenerateImageResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate synthetic images",
    description=(
        "Submit an image generation request using Stable Diffusion or BlenderProc. "
        "Returns a job ID immediately; poll /jobs/{job_id} for status. "
        "Generated images have metadata stripped, C2PA provenance attached, "
        "and invisible watermarks embedded."
    ),
)
async def generate_images(
    request_body: GenerateImageRequest,
    tenant_id: TenantDep,
    request: Request,
) -> GenerateImageResponse:
    """Queue a synthetic image generation job."""
    log = logger.bind(tenant_id=str(tenant_id), endpoint="generate")
    job_id = uuid.uuid4()

    log.info(
        "api.generate.requested",
        job_id=str(job_id),
        num_images=request_body.num_images,
        resolution=request_body.resolution,
    )

    # Retrieve services from app state
    settings = request.app.state.settings

    try:
        from aumos_image_engine.adapters.repositories import ImageJobRepository
        from aumos_image_engine.core.models import ImageGenerationJob

        # Create job record
        job = ImageGenerationJob(
            id=job_id,
            tenant_id=tenant_id,
            job_type=JobType.GENERATE.value,
            status=JobStatus.PENDING.value,
            prompt=request_body.prompt,
            negative_prompt=request_body.negative_prompt,
            num_images=request_body.num_images,
            resolution=request_body.resolution,
            model_config=request_body.model_config_params.model_dump(),
            has_c2pa=request_body.enable_c2pa,
            has_watermark=request_body.enable_watermark,
            biometric_verified=False,
        )

        # Persist job (fire-and-forget async task would process it)
        db_engine = request.app.state.db_engine
        repo = ImageJobRepository(engine=db_engine)
        await repo.create(job)

        log.info("api.generate.job_created", job_id=str(job_id))

        # Estimate processing time based on num_images and steps
        steps = request_body.model_config_params.num_inference_steps
        estimated = request_body.num_images * max(5, steps // 10)

        return GenerateImageResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            num_images=request_body.num_images,
            estimated_seconds=estimated,
        )

    except Exception as exc:
        log.error("api.generate.failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue image generation job",
        ) from exc


@router.post(
    "/deidentify",
    response_model=DeidentifyImageResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="De-identify faces in an image",
    description=(
        "Submit a face de-identification job for an uploaded image. "
        "Detects all faces and replaces biometric identity markers while "
        "optionally preserving expressions and poses. "
        "NIST FRVT non-linkability verification is run after de-identification."
    ),
)
async def deidentify_image(
    request_body: DeidentifyImageRequest,
    tenant_id: TenantDep,
    request: Request,
) -> DeidentifyImageResponse:
    """Queue a face de-identification job."""
    log = logger.bind(tenant_id=str(tenant_id), endpoint="deidentify")
    job_id = uuid.uuid4()

    log.info(
        "api.deidentify.requested",
        job_id=str(job_id),
        input_uri=request_body.input_uri,
        strength=request_body.strength,
    )

    try:
        from aumos_image_engine.adapters.repositories import ImageJobRepository
        from aumos_image_engine.core.models import ImageGenerationJob

        job = ImageGenerationJob(
            id=job_id,
            tenant_id=tenant_id,
            job_type=JobType.DEIDENTIFY.value,
            status=JobStatus.PENDING.value,
            input_uri=request_body.input_uri,
            model_config={
                "strength": request_body.strength,
                "preserve_expression": request_body.preserve_expression,
                "verify_non_linkability": request_body.verify_non_linkability,
                "frvt_threshold": request_body.frvt_threshold,
            },
        )

        db_engine = request.app.state.db_engine
        repo = ImageJobRepository(engine=db_engine)
        await repo.create(job)

        log.info("api.deidentify.job_created", job_id=str(job_id))

        return DeidentifyImageResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            faces_detected=0,
            is_compliant=True,
        )

    except Exception as exc:
        log.error("api.deidentify.failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue de-identification job",
        ) from exc


@router.post(
    "/strip-metadata",
    response_model=StripMetadataResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Strip image metadata",
    description=(
        "Remove all EXIF, IPTC, XMP, and DICOM metadata from an image. "
        "Optionally performs steganographic analysis to detect hidden data. "
        "Output image is clean of all identifying metadata."
    ),
)
async def strip_metadata(
    request_body: StripMetadataRequest,
    tenant_id: TenantDep,
    request: Request,
) -> StripMetadataResponse:
    """Queue a metadata stripping job."""
    log = logger.bind(tenant_id=str(tenant_id), endpoint="strip_metadata")
    job_id = uuid.uuid4()

    log.info(
        "api.strip_metadata.requested",
        job_id=str(job_id),
        input_uri=request_body.input_uri,
    )

    try:
        from aumos_image_engine.adapters.repositories import ImageJobRepository
        from aumos_image_engine.core.models import ImageGenerationJob

        job = ImageGenerationJob(
            id=job_id,
            tenant_id=tenant_id,
            job_type=JobType.STRIP_METADATA.value,
            status=JobStatus.PENDING.value,
            input_uri=request_body.input_uri,
            model_config={
                "output_format": request_body.output_format,
                "steganographic_check": request_body.steganographic_check,
            },
        )

        db_engine = request.app.state.db_engine
        repo = ImageJobRepository(engine=db_engine)
        await repo.create(job)

        return StripMetadataResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            fields_removed=0,
            categories_stripped=[],
        )

    except Exception as exc:
        log.error("api.strip_metadata.failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue metadata stripping job",
        ) from exc


@router.post(
    "/verify-biometric",
    response_model=VerifyBiometricResponse,
    summary="Verify biometric non-linkability",
    description=(
        "Verify that a de-identified image cannot be biometrically linked to "
        "its original using NIST FRVT compliant face recognition. "
        "Returns similarity score and compliance determination."
    ),
)
async def verify_biometric(
    request_body: VerifyBiometricRequest,
    tenant_id: TenantDep,
    request: Request,
) -> VerifyBiometricResponse:
    """Run NIST FRVT non-linkability verification synchronously."""
    log = logger.bind(tenant_id=str(tenant_id), endpoint="verify_biometric")
    log.info("api.verify_biometric.requested")

    try:
        from aumos_image_engine.adapters.biometric_verifier import FRVTBiometricVerifier
        from aumos_image_engine.adapters.storage import ImageStorageAdapter

        storage: ImageStorageAdapter = request.app.state.storage
        settings = request.app.state.settings

        # Load images from storage
        original_bytes = await storage.download(request_body.original_uri)
        deidentified_bytes = await storage.download(request_body.deidentified_uri)

        from io import BytesIO

        from PIL import Image as PILImage

        original_image = PILImage.open(BytesIO(original_bytes))
        deidentified_image = PILImage.open(BytesIO(deidentified_bytes))

        verifier = FRVTBiometricVerifier()
        result = await verifier.verify_non_linkability(
            original_image=original_image,
            deidentified_image=deidentified_image,
            threshold=request_body.threshold,
        )

        recommendation = (
            "Images are non-linkable. Safe to use for synthetic data."
            if result["is_non_linkable"]
            else f"Images may be linkable (similarity={result['similarity_score']:.3f}). "
            f"Re-run de-identification with higher strength."
        )

        return VerifyBiometricResponse(
            is_non_linkable=result["is_non_linkable"],
            similarity_score=result["similarity_score"],
            threshold=result["threshold"],
            frvt_compliant=result["frvt_compliant"],
            embedding_model=result["embedding_model"],
            recommendation=recommendation,
        )

    except Exception as exc:
        log.error("api.verify_biometric.failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Biometric verification failed",
        ) from exc


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status",
    description="Poll the status of an image generation or processing job.",
)
async def get_job_status(
    job_id: uuid.UUID,
    tenant_id: TenantDep,
    request: Request,
) -> JobStatusResponse:
    """Retrieve the current status of a job."""
    log = logger.bind(tenant_id=str(tenant_id), job_id=str(job_id))

    try:
        from aumos_image_engine.adapters.repositories import ImageJobRepository

        db_engine = request.app.state.db_engine
        repo = ImageJobRepository(engine=db_engine)
        job = await repo.get_by_id(job_id=job_id, tenant_id=tenant_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )

        return JobStatusResponse(
            job_id=job.id,
            job_type=JobType(job.job_type),
            status=JobStatus(job.status),
            created_at=job.created_at,
            updated_at=job.updated_at,
            completed_at=job.completed_at,
            output_uri=job.output_uri,
            has_c2pa=job.has_c2pa,
            has_watermark=job.has_watermark,
            biometric_verified=job.biometric_verified,
            error_message=job.error_message,
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.error("api.get_job.failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job status",
        ) from exc


@router.post(
    "/batch",
    response_model=BatchStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit batch generation",
    description=(
        "Submit a batch of image generation prompts. "
        "Processes up to 100 prompts with configurable concurrency. "
        "Returns a batch ID for progress polling."
    ),
)
async def create_batch(
    request_body: BatchGenerationRequest,
    tenant_id: TenantDep,
    request: Request,
) -> BatchStatusResponse:
    """Queue a batch image generation job."""
    log = logger.bind(tenant_id=str(tenant_id), endpoint="batch")
    batch_id = uuid.uuid4()
    total_images = len(request_body.prompts) * request_body.images_per_prompt

    log.info(
        "api.batch.requested",
        batch_id=str(batch_id),
        prompts=len(request_body.prompts),
        total_images=total_images,
    )

    try:
        from datetime import datetime, timezone

        from aumos_image_engine.adapters.repositories import ImageBatchRepository
        from aumos_image_engine.core.models import ImageBatch

        batch = ImageBatch(
            id=batch_id,
            tenant_id=tenant_id,
            name=request_body.name,
            description=request_body.description,
            images_count=total_images,
            processing_status="pending",
            batch_config={
                "prompts": request_body.prompts,
                "negative_prompt": request_body.negative_prompt,
                "images_per_prompt": request_body.images_per_prompt,
                "resolution": request_body.resolution,
                "model_config": request_body.model_config_params.model_dump(),
                "enable_c2pa": request_body.enable_c2pa,
                "enable_watermark": request_body.enable_watermark,
            },
        )

        db_engine = request.app.state.db_engine
        repo = ImageBatchRepository(engine=db_engine)
        await repo.create(batch)

        now = datetime.now(timezone.utc)
        return BatchStatusResponse(
            batch_id=batch_id,
            name=request_body.name,
            status="pending",  # type: ignore[arg-type]
            images_count=total_images,
            completed_count=0,
            failed_count=0,
            progress_percentage=0.0,
            created_at=now,
            updated_at=now,
        )

    except Exception as exc:
        log.error("api.batch.failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create batch job",
        ) from exc


@router.post(
    "/generate/sync",
    summary="Synchronous image generation (SDXL Turbo)",
    description=(
        "Generate a single image synchronously using SDXL Turbo (1-step inference). "
        "Returns PNG bytes directly in the response body. "
        "Rate-limited per tenant. Use for interactive applications needing sub-2s generation."
    ),
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}, "description": "PNG image bytes"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def generate_sync(
    request_body: SyncGenerateRequest,
    tenant_id: TenantDep,
    request: Request,
) -> Response:
    """Generate a single image synchronously using SDXL Turbo."""
    log = logger.bind(tenant_id=str(tenant_id), endpoint="generate_sync")
    log.info("api.generate_sync.requested", prompt_length=len(request_body.prompt))

    try:
        from aumos_image_engine.adapters.generators.model_registry import ModelAdapterRegistry

        settings = request.app.state.settings
        registry: ModelAdapterRegistry = request.app.state.model_registry

        adapter = await registry.get_warmed("sdxl_turbo")
        image_bytes = await adapter.generate(
            prompt=request_body.prompt,
            width=request_body.width,
            height=request_body.height,
            seed=request_body.seed,
        )

        log.info("api.generate_sync.complete", image_size_bytes=len(image_bytes))
        return Response(content=image_bytes, media_type="image/png")

    except Exception as exc:
        log.error("api.generate_sync.failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Synchronous image generation failed",
        ) from exc


@router.post(
    "/inpaint",
    summary="Inpaint masked image regions",
    description=(
        "Fill masked regions of an image with prompt-guided generated content. "
        "Mask is a binary image where white pixels indicate regions to inpaint. "
        "Returns PNG bytes of the inpainted image."
    ),
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Inpainted PNG image bytes"}},
)
async def inpaint_image(
    request_body: InpaintRequest,
    tenant_id: TenantDep,
    request: Request,
) -> Response:
    """Inpaint masked regions of an image using SD inpainting pipeline."""
    log = logger.bind(tenant_id=str(tenant_id), endpoint="inpaint")
    log.info(
        "api.inpaint.requested",
        image_uri=request_body.image_uri,
        mask_uri=request_body.mask_uri,
    )

    try:
        from aumos_image_engine.adapters.generators.inpainting_adapter import InpaintingAdapter

        storage = request.app.state.storage
        settings = request.app.state.settings

        image_bytes = await storage.download(request_body.image_uri)
        mask_bytes = await storage.download(request_body.mask_uri)

        inpainter: InpaintingAdapter = request.app.state.inpainting_adapter
        result_bytes = await inpainter.inpaint(
            image_bytes=image_bytes,
            mask_bytes=mask_bytes,
            prompt=request_body.prompt,
            negative_prompt=request_body.negative_prompt,
            strength=request_body.strength,
            num_inference_steps=request_body.num_inference_steps,
            guidance_scale=request_body.guidance_scale,
            seed=request_body.seed,
        )

        log.info("api.inpaint.complete", output_size_bytes=len(result_bytes))
        return Response(content=result_bytes, media_type="image/png")

    except Exception as exc:
        log.error("api.inpaint.failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inpainting failed",
        ) from exc


@router.post(
    "/finetune",
    response_model=FinetuneJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit LoRA fine-tuning job",
    description=(
        "Train a LoRA adapter on tenant-provided reference images. "
        "Accepts 20-100 reference image URIs and a concept prompt. "
        "Returns a finetune job ID; poll /finetune/{id} for status."
    ),
)
async def create_finetune_job(
    request_body: FinetuneJobRequest,
    tenant_id: TenantDep,
    request: Request,
) -> FinetuneJobResponse:
    """Queue a LoRA fine-tuning job."""
    log = logger.bind(tenant_id=str(tenant_id), endpoint="finetune")
    job_id = uuid.uuid4()

    log.info(
        "api.finetune.requested",
        job_id=str(job_id),
        base_model=request_body.base_model,
        num_images=len(request_body.reference_image_uris),
    )

    try:
        from aumos_image_engine.adapters.repositories import ImageFinetuneJobRepository
        from aumos_image_engine.core.models import ImageFinetuneJob

        finetune_job = ImageFinetuneJob(
            id=job_id,
            tenant_id=tenant_id,
            base_model=request_body.base_model,
            concept_prompt=request_body.concept_prompt,
            reference_image_uris=request_body.reference_image_uris,
            status="pending",
            training_steps_completed=0,
        )

        db_engine = request.app.state.db_engine
        repo = ImageFinetuneJobRepository(engine=db_engine)
        await repo.create(finetune_job)

        log.info("api.finetune.job_created", job_id=str(job_id))

        return FinetuneJobResponse(
            job_id=job_id,
            status="pending",
            base_model=request_body.base_model,
            num_reference_images=len(request_body.reference_image_uris),
        )

    except Exception as exc:
        log.error("api.finetune.failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue fine-tuning job",
        ) from exc


@router.get(
    "/finetune/{job_id}",
    response_model=FinetuneJobStatusResponse,
    summary="Get fine-tuning job status",
    description="Poll the status of a LoRA fine-tuning job.",
)
async def get_finetune_job_status(
    job_id: uuid.UUID,
    tenant_id: TenantDep,
    request: Request,
) -> FinetuneJobStatusResponse:
    """Retrieve the current status of a fine-tuning job."""
    log = logger.bind(tenant_id=str(tenant_id), job_id=str(job_id))

    try:
        from aumos_image_engine.adapters.repositories import ImageFinetuneJobRepository

        db_engine = request.app.state.db_engine
        repo = ImageFinetuneJobRepository(engine=db_engine)
        job = await repo.get_by_id(job_id=job_id, tenant_id=tenant_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Fine-tuning job {job_id} not found",
            )

        return FinetuneJobStatusResponse(
            job_id=job.id,
            status=job.status,
            base_model=job.base_model,
            training_steps_completed=job.training_steps_completed,
            training_time_s=job.training_time_s,
            adapter_uri=job.adapter_uri,
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.error("api.finetune.get_status.failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve fine-tuning job status",
        ) from exc
