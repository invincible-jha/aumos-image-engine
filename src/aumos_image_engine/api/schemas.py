"""Pydantic request and response schemas for the image engine API.

All schemas use strict validation at the API boundary.
UUIDs are used for all identifiers.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class JobType(str, Enum):
    """Supported image processing job types."""

    GENERATE = "generate"
    DEIDENTIFY = "deidentify"
    STRIP_METADATA = "strip_metadata"
    BATCH = "batch"


class JobStatus(str, Enum):
    """Possible job lifecycle states."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchStatus(str, Enum):
    """Batch-level processing states."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


# ---------------------------------------------------------------------------
# Image Generation
# ---------------------------------------------------------------------------


class GenerationModelConfig(BaseModel):
    """Configuration parameters for the generation model."""

    guidance_scale: float = Field(
        default=7.5,
        ge=1.0,
        le=30.0,
        description="How closely to follow the prompt (classifier-free guidance scale)",
    )
    num_inference_steps: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of denoising steps — more steps = higher quality but slower",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility. None = random.",
    )
    scheduler: str = Field(
        default="DPMSolverMultistepScheduler",
        description="Noise scheduler: DPMSolverMultistepScheduler, DDIMScheduler, etc.",
    )
    use_controlnet: bool = Field(
        default=False,
        description="Whether to use ControlNet for structured generation",
    )
    controlnet_model: str | None = Field(
        default=None,
        description="ControlNet model ID if use_controlnet=True",
    )


class GenerateImageRequest(BaseModel):
    """Request to generate synthetic images."""

    prompt: str = Field(
        min_length=1,
        max_length=2000,
        description="Positive description of the image to generate",
        examples=["a diverse group of people in a modern office, professional lighting"],
    )
    negative_prompt: str | None = Field(
        default=None,
        max_length=1000,
        description="Elements to exclude from generation",
        examples=["blurry, low quality, watermark, text"],
    )
    num_images: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of images to generate (max 20 per request)",
    )
    resolution: str = Field(
        default="512x512",
        description="Output resolution as WxH",
        examples=["512x512", "768x768", "512x768"],
    )
    model_config_params: GenerationModelConfig = Field(
        default_factory=GenerationModelConfig,
        description="Model generation parameters",
    )
    enable_c2pa: bool = Field(
        default=True,
        description="Attach C2PA provenance manifest declaring synthetic origin",
    )
    enable_watermark: bool = Field(
        default=True,
        description="Embed invisible watermark with job ID",
    )
    enable_biometric_check: bool = Field(
        default=True,
        description="Verify no real biometric data leaked into output",
    )

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, value: str) -> str:
        """Ensure resolution is in WxH format with valid dimensions."""
        parts = value.split("x")
        if len(parts) != 2:
            raise ValueError("Resolution must be in WxH format, e.g. 512x512")
        try:
            width, height = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError("Resolution dimensions must be integers") from exc
        if width % 64 != 0 or height % 64 != 0:
            raise ValueError("Width and height must be multiples of 64")
        if width > 1024 or height > 1024:
            raise ValueError("Max resolution is 1024x1024")
        return value

    def parsed_resolution(self) -> tuple[int, int]:
        """Return (width, height) as integers."""
        parts = self.resolution.split("x")
        return int(parts[0]), int(parts[1])


class GenerateImageResponse(BaseModel):
    """Response containing generation job details."""

    job_id: uuid.UUID = Field(description="Unique job identifier for status polling")
    status: JobStatus = Field(description="Initial job status (always pending)")
    num_images: int = Field(description="Number of images requested")
    estimated_seconds: int | None = Field(
        default=None,
        description="Estimated seconds to completion",
    )
    message: str = Field(default="Image generation job queued successfully")


# ---------------------------------------------------------------------------
# Face De-identification
# ---------------------------------------------------------------------------


class DeidentifyImageRequest(BaseModel):
    """Request to de-identify faces in an uploaded image."""

    input_uri: str = Field(
        description="MinIO URI of the input image to de-identify",
        examples=["s3://aumos-images/uploads/tenant-uuid/image.jpg"],
    )
    strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="De-identification strength (0=minimal, 1=maximum divergence)",
    )
    preserve_expression: bool = Field(
        default=True,
        description="Whether to preserve facial expressions for dataset utility",
    )
    verify_non_linkability: bool = Field(
        default=True,
        description="Run NIST FRVT verification after de-identification",
    )
    frvt_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum allowed biometric similarity (NIST FRVT threshold)",
    )


class FaceDetection(BaseModel):
    """Detected face in an image."""

    bbox: list[float] = Field(description="[x, y, width, height] in pixels")
    confidence: float = Field(description="Detection confidence score")


class BiometricVerificationResult(BaseModel):
    """Result of NIST FRVT non-linkability verification."""

    is_non_linkable: bool = Field(description="True if faces are not biometrically linkable")
    similarity_score: float = Field(description="Cosine similarity between face embeddings")
    threshold: float = Field(description="Maximum allowed similarity threshold used")
    frvt_compliant: bool = Field(description="Whether result meets NIST FRVT standards")
    embedding_model: str = Field(description="Face embedding model used for verification")


class DeidentifyImageResponse(BaseModel):
    """Response containing de-identification job results."""

    job_id: uuid.UUID
    status: JobStatus
    faces_detected: int = Field(description="Number of faces found and processed")
    output_uri: str | None = Field(
        default=None,
        description="MinIO URI of the de-identified output image",
    )
    biometric_verification: BiometricVerificationResult | None = Field(
        default=None,
        description="NIST FRVT verification result (if requested)",
    )
    is_compliant: bool = Field(
        description="Whether the output meets non-linkability requirements",
    )


# ---------------------------------------------------------------------------
# Metadata Stripping
# ---------------------------------------------------------------------------


class StripMetadataRequest(BaseModel):
    """Request to strip metadata from an image."""

    input_uri: str = Field(description="MinIO URI of the image to process")
    output_format: str = Field(
        default="PNG",
        description="Output image format",
        examples=["PNG", "JPEG", "WEBP"],
    )
    steganographic_check: bool = Field(
        default=True,
        description="Analyze for hidden steganographic data in pixel LSBs",
    )

    @field_validator("output_format")
    @classmethod
    def validate_format(cls, value: str) -> str:
        """Ensure output format is supported."""
        supported = {"PNG", "JPEG", "WEBP", "TIFF"}
        upper = value.upper()
        if upper not in supported:
            raise ValueError(f"Unsupported format: {value}. Supported: {supported}")
        return upper


class MetadataField(BaseModel):
    """A single metadata field found in an image."""

    tag: str = Field(description="Metadata tag name (e.g., GPS.GPSLatitude)")
    value: str = Field(description="Metadata value (may be truncated for long values)")
    category: str = Field(description="Metadata category: exif | iptc | xmp | dicom")


class StripMetadataResponse(BaseModel):
    """Response containing metadata stripping results."""

    job_id: uuid.UUID
    status: JobStatus
    output_uri: str | None = Field(
        default=None,
        description="MinIO URI of the metadata-stripped output image",
    )
    fields_removed: int = Field(description="Total number of metadata fields removed")
    categories_stripped: list[str] = Field(
        description="Categories of metadata that were present and stripped",
    )
    steganographic_findings: list[str] = Field(
        default_factory=list,
        description="Potential steganographic anomalies found (empty = none)",
    )


# ---------------------------------------------------------------------------
# Biometric Verification
# ---------------------------------------------------------------------------


class VerifyBiometricRequest(BaseModel):
    """Request to verify biometric non-linkability between two images."""

    original_uri: str = Field(description="MinIO URI of the original image")
    deidentified_uri: str = Field(description="MinIO URI of the de-identified image")
    threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum allowed similarity score for non-linkability",
    )


class VerifyBiometricResponse(BaseModel):
    """Response from biometric non-linkability verification."""

    is_non_linkable: bool
    similarity_score: float
    threshold: float
    frvt_compliant: bool
    embedding_model: str
    recommendation: str = Field(
        description="Human-readable recommendation based on verification result",
    )


# ---------------------------------------------------------------------------
# Job Status
# ---------------------------------------------------------------------------


class JobStatusResponse(BaseModel):
    """Generic job status response for polling."""

    job_id: uuid.UUID
    job_type: JobType
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    output_uri: str | None = None
    has_c2pa: bool = False
    has_watermark: bool = False
    biometric_verified: bool = False
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------


class BatchGenerationRequest(BaseModel):
    """Request to generate images for multiple prompts in a batch."""

    name: str = Field(
        min_length=1,
        max_length=255,
        description="Human-readable batch name",
    )
    description: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional batch description",
    )
    prompts: list[str] = Field(
        min_length=1,
        description="List of prompts to generate images for",
    )
    negative_prompt: str | None = Field(
        default=None,
        description="Shared negative prompt for all prompts in the batch",
    )
    images_per_prompt: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of images to generate per prompt",
    )
    resolution: str = Field(default="512x512")
    model_config_params: GenerationModelConfig = Field(default_factory=GenerationModelConfig)
    enable_c2pa: bool = Field(default=True)
    enable_watermark: bool = Field(default=True)

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, value: list[str]) -> list[str]:
        """Validate prompt list size."""
        if len(value) > 100:
            raise ValueError("Maximum 100 prompts per batch")
        if any(not p.strip() for p in value):
            raise ValueError("All prompts must be non-empty")
        return value


class BatchStatusResponse(BaseModel):
    """Batch processing status and progress."""

    batch_id: uuid.UUID
    name: str
    status: BatchStatus
    images_count: int = Field(description="Total images requested")
    completed_count: int = Field(description="Successfully generated images")
    failed_count: int = Field(description="Failed generation attempts")
    progress_percentage: float = Field(description="Completion percentage (0-100)")
    output_prefix: str | None = Field(
        default=None,
        description="MinIO URI prefix where batch outputs are stored",
    )
    created_at: datetime
    updated_at: datetime
