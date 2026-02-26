"""SQLAlchemy ORM models for aumos-image-engine.

Table prefix: img_
Tenant isolation via RLS (row-level security) using tenant_id column.

Models:
- ImageGenerationJob: tracks individual generation/de-identification jobs
- ImageBatch: groups multiple images into a named batch
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class with common columns for all models."""

    pass


class TenantMixin:
    """Mixin providing tenant isolation column.

    All tables with this mixin participate in PostgreSQL RLS.
    The application sets `SET app.current_tenant = '<tenant_id>'`
    on each connection, and RLS policies filter rows automatically.
    """

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="Tenant identifier for RLS isolation",
    )


class ImageGenerationJob(TenantMixin, Base):
    """Tracks a single image generation or processing job.

    Job types:
    - generate: Stable Diffusion or BlenderProc generation
    - deidentify: Face de-identification on uploaded image
    - strip_metadata: EXIF/IPTC/XMP/DICOM removal
    - batch: Batch processing parent job

    Status flow: pending → processing → completed | failed
    """

    __tablename__ = "img_generation_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Primary key",
    )
    job_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Job type: generate | deidentify | strip_metadata | batch",
    )
    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="pending",
        index=True,
        comment="Job status: pending | processing | completed | failed",
    )

    # Generation parameters (nullable for non-generation jobs)
    model_config: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Model configuration: model_id, scheduler, guidance_scale, etc.",
    )
    prompt: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Positive prompt for image generation",
    )
    negative_prompt: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Negative prompt to avoid specific features",
    )
    num_images: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Number of images to generate",
    )
    resolution: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="512x512",
        comment="Output resolution in WxH format",
    )

    # Privacy and provenance flags
    has_c2pa: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether C2PA provenance manifest is attached",
    )
    has_watermark: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether invisible watermark is embedded",
    )
    biometric_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether NIST FRVT non-linkability has been verified",
    )

    # Input / output
    input_uri: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="MinIO URI of input image (for deidentify/strip_metadata jobs)",
    )
    output_uri: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="MinIO URI of processed output image(s)",
    )

    # Batch relationship
    batch_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("img_batches.id"),
        nullable=True,
        comment="Parent batch ID if this job is part of a batch",
    )
    batch: Mapped[ImageBatch | None] = relationship("ImageBatch", back_populates="jobs")

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if job failed",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when job reached completed or failed status",
    )

    def __repr__(self) -> str:
        return f"<ImageGenerationJob id={self.id} type={self.job_type} status={self.status}>"


class ImageBatch(TenantMixin, Base):
    """Groups multiple image generation jobs into a named batch.

    A batch tracks overall progress across many individual jobs.
    Useful for generating training datasets with hundreds of images.
    """

    __tablename__ = "img_batches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable batch name",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Optional description of this batch's purpose",
    )
    images_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of images requested in this batch",
    )
    processing_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="pending",
        index=True,
        comment="Batch status: pending | processing | completed | failed | partial",
    )
    completed_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of successfully completed jobs in this batch",
    )
    failed_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of failed jobs in this batch",
    )

    # Batch configuration
    batch_config: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Shared configuration for all jobs in this batch",
    )
    output_prefix: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="MinIO URI prefix for batch outputs",
    )

    # Relationships
    jobs: Mapped[list[ImageGenerationJob]] = relationship(
        "ImageGenerationJob",
        back_populates="batch",
        lazy="select",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    @property
    def progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.images_count == 0:
            return 0.0
        return round((self.completed_count / self.images_count) * 100, 1)

    def __repr__(self) -> str:
        return f"<ImageBatch id={self.id} name={self.name!r} status={self.processing_status}>"
