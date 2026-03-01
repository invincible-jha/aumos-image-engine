"""Model adapter registry for image generation.

Provides a factory pattern for instantiating image generation adapters.
Adapters are loaded lazily and cached in-process to avoid reloading weights.
"""

from __future__ import annotations

from typing import Any

import structlog
from aumos_common.errors import NotFoundError
from aumos_common.logging import get_logger


class ModelAdapterRegistry:
    """Factory for image generation model adapters.

    Adapters are loaded lazily (on first use) and cached in-process.
    Adding a new model requires only registering the adapter class here.

    Registered models:
        - "sd15": Stable Diffusion 1.5 (legacy, ControlNet compatible)
        - "sdxl": Stable Diffusion XL Base 1.0
        - "sdxl_turbo": SDXL Turbo (1-step inference)
        - "sd35": Stable Diffusion 3.5 Large
        - "sd35_medium": Stable Diffusion 3.5 Medium (faster, lower VRAM)

    Args:
        settings: ImageEngineSettings for model configuration.
    """

    def __init__(self, settings: Any) -> None:
        """Initialize the registry with engine settings.

        Args:
            settings: ImageEngineSettings containing model config.
        """
        self._settings = settings
        self._instances: dict[str, Any] = {}
        self._log: structlog.BoundLogger = get_logger(__name__)

    def _get_adapter_class(self, model_name: str) -> type:
        """Return the adapter class for a given model name.

        Args:
            model_name: Registered model name.

        Returns:
            Adapter class (not yet instantiated).

        Raises:
            NotFoundError: If model_name is not registered.
        """
        from aumos_image_engine.adapters.generators.sd35_adapter import SD35Adapter
        from aumos_image_engine.adapters.generators.sdxl_turbo_adapter import SDXLTurboAdapter
        from aumos_image_engine.adapters.generators.stable_diffusion import StableDiffusionAdapter

        registry: dict[str, type] = {
            "sd15": StableDiffusionAdapter,
            "sdxl": StableDiffusionAdapter,
            "sdxl_turbo": SDXLTurboAdapter,
            "sd35": SD35Adapter,
            "sd35_medium": SD35Adapter,
        }

        allowed = getattr(self._settings, "allowed_models", list(registry.keys()))
        if model_name not in allowed:
            raise NotFoundError(
                f"Model '{model_name}' is not in the allowed models list: {allowed}"
            )
        if model_name not in registry:
            raise NotFoundError(
                f"Unknown model '{model_name}'. Supported: {list(registry.keys())}"
            )
        return registry[model_name]

    def get(self, model_name: str) -> Any:
        """Get or create a model adapter instance.

        Adapters are cached after first instantiation. The adapter's
        warm_up() method must be called separately before use.

        Args:
            model_name: Registered model name (e.g., "sd35", "sdxl_turbo").

        Returns:
            Instantiated adapter (may not be warmed up yet).

        Raises:
            NotFoundError: If model_name is not registered or not allowed.
        """
        if model_name not in self._instances:
            adapter_class = self._get_adapter_class(model_name)
            cache_dir = getattr(self._settings, "model_cache_dir", "/tmp/model-cache")
            device = getattr(self._settings, "sd_device", "cuda")
            dtype = getattr(self._settings, "sd_dtype", "float16")

            # SD3.5 Medium uses a different model ID
            if model_name == "sd35_medium":
                self._instances[model_name] = adapter_class(
                    model_id="stabilityai/stable-diffusion-3.5-medium",
                    torch_dtype=dtype,
                    device=device,
                    cache_dir=cache_dir,
                )
            else:
                self._instances[model_name] = adapter_class(
                    torch_dtype=dtype,
                    device=device,
                    cache_dir=cache_dir,
                )

            self._log.info("model adapter instantiated", model_name=model_name)

        return self._instances[model_name]

    async def get_warmed(self, model_name: str) -> Any:
        """Get a warmed-up model adapter, loading weights if necessary.

        Args:
            model_name: Registered model name.

        Returns:
            Warmed adapter ready for inference.
        """
        adapter = self.get(model_name)
        if not getattr(adapter, "_pipe", None):
            await adapter.warm_up()
        return adapter

    @property
    def registered_models(self) -> list[str]:
        """Return list of all registered model names.

        Returns:
            List of model name strings.
        """
        return ["sd15", "sdxl", "sdxl_turbo", "sd35", "sd35_medium"]
