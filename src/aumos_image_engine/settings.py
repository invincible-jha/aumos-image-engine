"""Settings for aumos-image-engine, loaded from environment variables.

All settings are validated at startup using Pydantic v2 BaseSettings.
Image-engine-specific settings are nested under AUMOS_IMG__ prefix.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection settings."""

    model_config = SettingsConfigDict(env_prefix="AUMOS_DATABASE__")

    url: str = Field(
        default="postgresql+asyncpg://aumos:aumos_dev@localhost:5432/aumos",
        description="Async SQLAlchemy database URL",
    )
    pool_size: int = Field(default=10, ge=1, le=100)
    echo: bool = Field(default=False)


class KafkaSettings(BaseSettings):
    """Kafka producer/consumer settings."""

    model_config = SettingsConfigDict(env_prefix="AUMOS_KAFKA__")

    brokers: str = Field(default="localhost:9092")
    schema_registry_url: str = Field(default="http://localhost:8081")

    @property
    def broker_list(self) -> list[str]:
        """Return brokers as a list."""
        return [b.strip() for b in self.brokers.split(",")]


class MinioSettings(BaseSettings):
    """MinIO object storage settings."""

    model_config = SettingsConfigDict(env_prefix="AUMOS_MINIO__")

    url: str = Field(default="http://localhost:9000")
    access_key: str = Field(default="minioadmin")
    secret_key: str = Field(default="minioadmin")
    default_bucket: str = Field(default="aumos-images")


class RedisSettings(BaseSettings):
    """Redis cache/queue settings."""

    model_config = SettingsConfigDict(env_prefix="AUMOS_REDIS__")

    url: str = Field(default="redis://localhost:6379/0")


class ImageEngineSettings(BaseSettings):
    """Image-engine-specific configuration."""

    model_config = SettingsConfigDict(env_prefix="AUMOS_IMG__")

    # Stable Diffusion
    sd_model_id: str = Field(
        default="runwayml/stable-diffusion-v1-5",
        description="HuggingFace model ID for Stable Diffusion",
    )
    sd_device: str = Field(default="cuda", description="Torch device: cuda or cpu")
    sd_dtype: str = Field(default="float16", description="Torch dtype: float16 or float32")
    model_cache_dir: str = Field(default="/app/model_cache")

    # BlenderProc
    blenderproc_enabled: bool = Field(default=False)
    blenderproc_output_dir: str = Field(default="/tmp/blenderproc_output")

    # Face de-identification
    face_detection_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    face_deidentification_strength: float = Field(default=0.8, ge=0.0, le=1.0)
    preserve_expression: bool = Field(default=True)

    # C2PA
    c2pa_enabled: bool = Field(default=True)
    c2pa_signing_key_path: str = Field(default="/run/secrets/c2pa_signing_key")
    c2pa_cert_path: str = Field(default="/run/secrets/c2pa_cert")

    # Watermarking
    watermark_enabled: bool = Field(default=True)
    watermark_strength: float = Field(default=0.3, ge=0.0, le=1.0)

    # Biometric verification
    biometric_verification_enabled: bool = Field(default=True)
    frvt_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    # Batch processing
    batch_max_size: int = Field(default=100, ge=1, le=1000)
    batch_concurrency: int = Field(default=4, ge=1, le=32)

    @field_validator("sd_device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        """Ensure device is valid."""
        if value not in {"cuda", "cpu", "mps"}:
            raise ValueError(f"Invalid device: {value}. Must be cuda, cpu, or mps")
        return value

    @field_validator("sd_dtype")
    @classmethod
    def validate_dtype(cls, value: str) -> str:
        """Ensure dtype is valid."""
        if value not in {"float16", "float32", "bfloat16"}:
            raise ValueError(f"Invalid dtype: {value}")
        return value


class Settings(BaseSettings):
    """Root settings aggregating all sub-configurations."""

    model_config = SettingsConfigDict(env_prefix="AUMOS_")

    service_name: str = Field(default="aumos-image-engine")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    otel_endpoint: str = Field(default="http://localhost:4317")
    otel_enabled: bool = Field(default=False)

    # External service URLs
    privacy_engine_url: str = Field(
        default="http://localhost:8010",
        validation_alias="PRIVACY_ENGINE_URL",
    )
    gpu_enabled: bool = Field(default=True, validation_alias="GPU_ENABLED")

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    minio: MinioSettings = Field(default_factory=MinioSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    img: ImageEngineSettings = Field(default_factory=ImageEngineSettings)

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
