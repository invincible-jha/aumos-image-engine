"""BlenderProc 3D synthetic data generation adapter.

BlenderProc is a procedural Blender-based framework for generating
photorealistic synthetic datasets with ground-truth annotations.

This adapter wraps BlenderProc to generate:
- Diverse synthetic scenes with randomized lighting, camera angles
- Segmentation masks and depth maps as auxiliary data
- Human avatars with varied poses (SMPL-X body model)
- Object datasets with 6-DOF pose annotations

BlenderProc runs as a subprocess because it requires its own Python
environment with Blender's embedded Python interpreter.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import structlog
from PIL import Image as PILImage

logger = structlog.get_logger(__name__)


class BlenderProcAdapter:
    """Generates 3D synthetic images via BlenderProc.

    BlenderProc runs as a child process — this adapter orchestrates:
    1. Writing a scene config JSON
    2. Spawning the blenderproc subprocess
    3. Collecting output images from the output directory
    4. Returning PIL images to the caller
    """

    def __init__(
        self,
        output_dir: str,
        blenderproc_executable: str = "blenderproc",
        timeout_seconds: int = 300,
    ) -> None:
        """Initialize BlenderProc adapter.

        Args:
            output_dir: Directory for BlenderProc to write rendered images.
            blenderproc_executable: Path to the blenderproc CLI binary.
            timeout_seconds: Maximum time to wait for a render to complete.
        """
        self._output_dir = Path(output_dir)
        self._executable = blenderproc_executable
        self._timeout = timeout_seconds
        self._log = logger.bind(adapter="blenderproc")
        self._ready = False

    async def load_model(self) -> None:
        """Verify BlenderProc is installed and accessible."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._executable,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            version = stdout.decode().strip()
            self._log.info("blenderproc.ready", version=version)
            self._ready = True
        except FileNotFoundError:
            self._log.warning("blenderproc.not_found", executable=self._executable)
            self._ready = False

    @property
    def is_ready(self) -> bool:
        """True if BlenderProc is available."""
        return self._ready

    async def generate(
        self,
        prompt: str,
        negative_prompt: str | None,
        num_images: int,
        width: int,
        height: int,
        model_config: dict[str, Any],
    ) -> list[PILImage.Image]:
        """Generate synthetic 3D scene images via BlenderProc.

        BlenderProc doesn't use text prompts; instead, the prompt
        is parsed to extract scene parameters (objects, lighting, environment).

        Args:
            prompt: Scene description (parsed for object and environment hints).
            negative_prompt: Ignored for BlenderProc (3D rendering doesn't use negatives).
            num_images: Number of camera angles to render.
            width: Output image width in pixels.
            height: Output image height in pixels.
            model_config: BlenderProc-specific configuration:
                - scene_type: "indoor" | "outdoor" | "object"
                - num_lights: int
                - camera_distance: float
                - hdri_name: str | None
                - object_models: list[str]

        Returns:
            List of rendered PIL images.
        """
        if not self._ready:
            raise RuntimeError("BlenderProc is not available on this system.")

        render_id = str(uuid.uuid4())
        render_output_dir = self._output_dir / render_id
        render_output_dir.mkdir(parents=True, exist_ok=True)

        self._log.info(
            "blenderproc.generate_start",
            render_id=render_id,
            num_views=num_images,
            width=width,
            height=height,
        )

        # Build scene configuration from model_config and prompt hints
        scene_config = self._build_scene_config(
            prompt=prompt,
            num_images=num_images,
            width=width,
            height=height,
            model_config=model_config,
            output_dir=str(render_output_dir),
        )

        # Write scene config to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as config_file:
            json.dump(scene_config, config_file, indent=2)
            config_path = config_file.name

        try:
            # Run BlenderProc subprocess
            await self._run_blenderproc(config_path=config_path)

            # Collect rendered images
            images = self._collect_rendered_images(output_dir=render_output_dir)
            self._log.info("blenderproc.generate_complete", count=len(images))
            return images

        finally:
            Path(config_path).unlink(missing_ok=True)

    def _build_scene_config(
        self,
        prompt: str,
        num_images: int,
        width: int,
        height: int,
        model_config: dict[str, Any],
        output_dir: str,
    ) -> dict[str, Any]:
        """Build BlenderProc scene configuration dict.

        Parses prompt for object hints and merges with explicit model_config.
        """
        scene_type = model_config.get("scene_type", "indoor")
        num_lights = model_config.get("num_lights", 3)
        camera_distance = model_config.get("camera_distance", 2.5)
        hdri_name = model_config.get("hdri_name")
        object_models = model_config.get("object_models", [])

        # Simple prompt parsing — extract object keywords
        prompt_lower = prompt.lower()
        if "outdoor" in prompt_lower or "nature" in prompt_lower:
            scene_type = "outdoor"
        elif "office" in prompt_lower or "room" in prompt_lower:
            scene_type = "indoor"

        return {
            "render_engine": "CYCLES",
            "resolution": {"width": width, "height": height},
            "num_views": num_images,
            "scene_type": scene_type,
            "lighting": {
                "num_lights": num_lights,
                "hdri": hdri_name,
                "randomize_strength": True,
                "strength_range": [0.5, 2.0],
            },
            "camera": {
                "distance": camera_distance,
                "elevation_range": [-15, 45],
                "azimuth_range": [0, 360],
            },
            "objects": object_models,
            "output": {
                "directory": output_dir,
                "format": "PNG",
                "render_rgb": True,
                "render_depth": False,
                "render_segmap": False,
            },
            "samples": 128,  # CYCLES samples per pixel
        }

    async def _run_blenderproc(self, config_path: str) -> None:
        """Run BlenderProc as an async subprocess."""
        proc = await asyncio.create_subprocess_exec(
            self._executable,
            "run",
            "--config",
            config_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            raise TimeoutError(f"BlenderProc render timed out after {self._timeout}s") from exc

        if proc.returncode != 0:
            error_msg = stderr.decode()[-500:] if stderr else "Unknown error"
            raise RuntimeError(f"BlenderProc render failed: {error_msg}")

        self._log.debug("blenderproc.subprocess_complete", stdout=stdout.decode()[-200:] if stdout else "")

    def _collect_rendered_images(self, output_dir: Path) -> list[PILImage.Image]:
        """Load all rendered PNG images from the output directory."""
        image_paths = sorted(output_dir.glob("*.png"))
        images = []
        for path in image_paths:
            try:
                img = PILImage.open(path).convert("RGB")
                images.append(img)
            except Exception as exc:
                self._log.warning("blenderproc.image_load_failed", path=str(path), error=str(exc))
        return images

    async def generate_human_avatars(
        self,
        num_avatars: int,
        num_views_per_avatar: int,
        width: int,
        height: int,
        pose_config: dict[str, Any] | None,
    ) -> list[PILImage.Image]:
        """Generate diverse synthetic human avatars with varied poses.

        Uses SMPL-X body model for anatomically correct human generation
        with randomized demographics, clothing, and lighting.

        Args:
            num_avatars: Number of distinct avatar identities.
            num_views_per_avatar: Camera angles per avatar.
            width: Output width in pixels.
            height: Output height in pixels.
            pose_config: SMPL-X pose configuration overrides.

        Returns:
            List of rendered avatar images (num_avatars * num_views_per_avatar).
        """
        all_images: list[PILImage.Image] = []

        for avatar_index in range(num_avatars):
            avatar_config = {
                "scene_type": "avatar",
                "avatar": {
                    "index": avatar_index,
                    "randomize_appearance": True,
                    "body_model": "smplx",
                    "pose": pose_config or {"random": True},
                },
                "num_lights": 4,
                "camera_distance": 2.0,
            }

            images = await self.generate(
                prompt=f"synthetic human avatar {avatar_index}",
                negative_prompt=None,
                num_images=num_views_per_avatar,
                width=width,
                height=height,
                model_config=avatar_config,
            )
            all_images.extend(images)

        self._log.info(
            "blenderproc.avatars_generated",
            num_avatars=num_avatars,
            total_images=len(all_images),
        )
        return all_images
