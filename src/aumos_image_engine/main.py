"""FastAPI application factory for aumos-image-engine.

Creates the ASGI application with lifespan management for:
- Database connection pool
- Kafka producer
- MinIO client
- ML model warm-up (Stable Diffusion)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aumos_image_engine.api.router import router as image_router
from aumos_image_engine.settings import Settings, get_settings

logger = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override for testing.

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="AumOS Image Engine",
        description=(
            "Synthetic image generation with biometric non-linkability, "
            "metadata stripping, C2PA provenance, and face de-identification."
        ),
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=_lifespan,
    )

    # Store settings on app state for access in routes
    app.state.settings = settings

    # CORS — restrict in production
    allowed_origins = ["*"] if settings.environment == "development" else []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Tenant-ID"],
    )

    # Mount routers
    app.include_router(image_router, prefix="/images", tags=["images"])

    # Health endpoints (not behind auth)
    @app.get("/live", tags=["health"])
    async def liveness() -> JSONResponse:
        """Kubernetes liveness probe — service is alive."""
        return JSONResponse({"status": "live", "service": settings.service_name})

    @app.get("/ready", tags=["health"])
    async def readiness() -> JSONResponse:
        """Kubernetes readiness probe — service can accept traffic."""
        checks: dict[str, str] = {}
        all_ready = True

        # Check if model is loaded (stored on app state during lifespan)
        model_ready: bool = getattr(app.state, "model_ready", False)
        checks["model"] = "ready" if model_ready else "loading"
        if not model_ready:
            all_ready = False

        status_code = 200 if all_ready else 503
        return JSONResponse(
            {
                "status": "ready" if all_ready else "not_ready",
                "service": settings.service_name,
                "checks": checks,
            },
            status_code=status_code,
        )

    @app.get("/metrics", tags=["health"])
    async def metrics() -> JSONResponse:
        """Basic metrics endpoint (Prometheus scrape target)."""
        return JSONResponse(
            {
                "service": settings.service_name,
                "environment": settings.environment,
                "gpu_enabled": settings.gpu_enabled,
            }
        )

    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan: startup and shutdown hooks.

    On startup:
    - Initializes database connection pool
    - Connects Kafka producer
    - Initializes MinIO client + ensures bucket exists
    - Warms up ML model (lazy on first request if GPU not available)

    On shutdown:
    - Flushes Kafka producer
    - Closes database pool
    """
    settings: Settings = app.state.settings
    log = logger.bind(service=settings.service_name, environment=settings.environment)

    log.info("image_engine.startup.begin")
    startup_start = time.monotonic()

    # Mark model as not yet ready
    app.state.model_ready = False

    try:
        # Initialize database
        from aumos_image_engine.adapters.repositories import init_db

        app.state.db_engine = await init_db(settings.database.url)
        log.info("image_engine.startup.database_ready")

        # Initialize Kafka producer
        from aumos_image_engine.adapters.kafka import ImageEventPublisher

        publisher = ImageEventPublisher(
            brokers=settings.kafka.broker_list,
            service_name=settings.service_name,
        )
        await publisher.start()
        app.state.event_publisher = publisher
        log.info("image_engine.startup.kafka_ready")

        # Initialize MinIO client
        from aumos_image_engine.adapters.storage import ImageStorageAdapter

        storage = ImageStorageAdapter(
            endpoint=settings.minio.url,
            access_key=settings.minio.access_key,
            secret_key=settings.minio.secret_key,
            bucket=settings.minio.default_bucket,
        )
        await storage.ensure_bucket()
        app.state.storage = storage
        log.info("image_engine.startup.storage_ready")

        # Warm up model only if GPU is enabled (skip in CPU-only CI)
        if settings.gpu_enabled:
            from aumos_image_engine.adapters.generators.stable_diffusion import StableDiffusionAdapter

            sd_adapter = StableDiffusionAdapter(
                model_id=settings.img.sd_model_id,
                device=settings.img.sd_device,
                dtype=settings.img.sd_dtype,
                cache_dir=settings.img.model_cache_dir,
            )
            await sd_adapter.load_model()
            app.state.sd_adapter = sd_adapter
            log.info("image_engine.startup.model_ready", model_id=settings.img.sd_model_id)
        else:
            app.state.sd_adapter = None
            log.info("image_engine.startup.model_skipped", reason="gpu_disabled")

        app.state.model_ready = True
        elapsed = time.monotonic() - startup_start
        log.info("image_engine.startup.complete", elapsed_seconds=round(elapsed, 2))

    except Exception as exc:
        log.error("image_engine.startup.failed", error=str(exc))
        raise

    yield

    # Shutdown
    log.info("image_engine.shutdown.begin")

    try:
        if hasattr(app.state, "event_publisher"):
            await app.state.event_publisher.stop()
            log.info("image_engine.shutdown.kafka_closed")

        if hasattr(app.state, "db_engine"):
            await app.state.db_engine.dispose()
            log.info("image_engine.shutdown.database_closed")

    except Exception as exc:
        log.error("image_engine.shutdown.error", error=str(exc))

    log.info("image_engine.shutdown.complete")


# Module-level app instance for uvicorn
app = create_app()
