"""Protocol interfaces for the image engine's hexagonal architecture.

These protocols define the contracts between the core domain and adapters.
Concrete implementations live in aumos_image_engine/adapters/.

Protocols:
- ImageGeneratorProtocol: Generates synthetic images from prompts or configs
- FaceDeidentifierProtocol: Detects and de-identifies faces in images
- MetadataStripperProtocol: Removes EXIF/IPTC/XMP/DICOM metadata
- WatermarkerProtocol: Embeds C2PA provenance and invisible watermarks
- BiometricVerifierProtocol: Verifies NIST FRVT non-linkability compliance
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from PIL import Image as PILImage


@runtime_checkable
class ImageGeneratorProtocol(Protocol):
    """Contract for synthetic image generation adapters.

    Implementations: StableDiffusionAdapter, BlenderProcAdapter
    """

    async def generate(
        self,
        prompt: str,
        negative_prompt: str | None,
        num_images: int,
        width: int,
        height: int,
        model_config: dict[str, Any],
    ) -> list[PILImage.Image]:
        """Generate synthetic images from a text prompt.

        Args:
            prompt: Positive description of desired image content.
            negative_prompt: Elements to exclude from generation.
            num_images: Number of images to generate.
            width: Output image width in pixels.
            height: Output image height in pixels.
            model_config: Generator-specific configuration
                (e.g., guidance_scale, num_inference_steps, seed).

        Returns:
            List of generated PIL images.
        """
        ...

    async def load_model(self) -> None:
        """Load/warm up the model. Called once at startup."""
        ...

    @property
    def is_ready(self) -> bool:
        """True if the model is loaded and ready for inference."""
        ...


@runtime_checkable
class FaceDeidentifierProtocol(Protocol):
    """Contract for face detection and de-identification adapters.

    Implementations preserve facial expressions and poses while replacing
    biometric identity markers (face geometry, texture).
    """

    async def detect_faces(
        self,
        image: PILImage.Image,
        confidence_threshold: float,
    ) -> list[dict[str, Any]]:
        """Detect all faces in an image.

        Args:
            image: Input PIL image.
            confidence_threshold: Minimum detection confidence (0.0-1.0).

        Returns:
            List of face detection dicts with keys:
            - bbox: [x, y, width, height]
            - confidence: float
            - landmarks: dict of facial landmark coordinates
        """
        ...

    async def deidentify(
        self,
        image: PILImage.Image,
        strength: float,
        preserve_expression: bool,
    ) -> PILImage.Image:
        """De-identify all faces in an image.

        Replaces face identity markers (geometry, texture) while
        optionally preserving expression and pose to maintain utility
        for downstream ML tasks.

        Args:
            image: Input PIL image containing faces.
            strength: De-identification strength (0.0-1.0).
                Higher values produce more divergent synthetic faces.
            preserve_expression: If True, maps expressions from original
                to synthetic face to preserve emotional annotations.

        Returns:
            PIL image with all faces de-identified.
        """
        ...


@runtime_checkable
class MetadataStripperProtocol(Protocol):
    """Contract for metadata removal adapters.

    Removes all identifying metadata from images including:
    - EXIF: camera model, GPS, datetime, serial numbers
    - IPTC: copyright, creator, caption
    - XMP: Adobe metadata, tags
    - DICOM: patient data (for medical images)
    """

    async def strip(
        self,
        image: PILImage.Image,
        image_format: str,
        steganographic_check: bool,
    ) -> tuple[PILImage.Image, dict[str, Any]]:
        """Strip all metadata from an image.

        Args:
            image: Input PIL image.
            image_format: Target format (JPEG, PNG, TIFF, etc.).
            steganographic_check: If True, also check for hidden
                steganographic data in pixel LSBs.

        Returns:
            Tuple of (stripped image, report dict) where report
            describes what metadata was found and removed.
        """
        ...

    async def analyze(
        self,
        image: PILImage.Image,
    ) -> dict[str, Any]:
        """Analyze an image for metadata without stripping.

        Returns a report of all metadata found in the image.
        """
        ...


@runtime_checkable
class WatermarkerProtocol(Protocol):
    """Contract for C2PA provenance and invisible watermarking adapters.

    C2PA (Coalition for Content Provenance and Authenticity) attaches
    a signed manifest to images declaring their synthetic origin.
    Invisible watermarks embed a recoverable payload in the image
    frequency domain without visible artifacts.
    """

    async def add_c2pa_provenance(
        self,
        image_bytes: bytes,
        image_format: str,
        provenance_metadata: dict[str, Any],
    ) -> bytes:
        """Attach a C2PA provenance manifest to an image.

        Args:
            image_bytes: Raw image bytes to sign.
            image_format: Image format (jpeg, png, webp).
            provenance_metadata: Metadata to include in manifest:
                - generator: model name and version
                - tenant_id: tenant identifier
                - job_id: generation job ID
                - timestamp: ISO 8601 creation timestamp

        Returns:
            Image bytes with embedded C2PA manifest.
        """
        ...

    async def add_invisible_watermark(
        self,
        image: PILImage.Image,
        payload: str,
        strength: float,
    ) -> PILImage.Image:
        """Embed an invisible watermark in an image.

        Uses DCT frequency domain embedding — robust against
        JPEG compression and minor image transformations.

        Args:
            image: Input PIL image.
            payload: String payload to embed (e.g., job ID + tenant hash).
            strength: Watermark strength (0.0-1.0).

        Returns:
            Watermarked PIL image.
        """
        ...

    async def verify_watermark(
        self,
        image: PILImage.Image,
    ) -> tuple[bool, str | None]:
        """Attempt to recover an invisible watermark from an image.

        Returns:
            Tuple of (found: bool, payload: str | None).
        """
        ...


@runtime_checkable
class BiometricVerifierProtocol(Protocol):
    """Contract for NIST FRVT biometric non-linkability verification.

    Verifies that de-identified images cannot be linked back to
    original subjects using face recognition algorithms. Compliance
    with NIST FRVT (Face Recognition Vendor Test) standards.
    """

    async def verify_non_linkability(
        self,
        original_image: PILImage.Image,
        deidentified_image: PILImage.Image,
        threshold: float,
    ) -> dict[str, Any]:
        """Verify that two images are not biometrically linkable.

        Extracts face embeddings from both images using a NIST FRVT
        compliant algorithm and verifies that the cosine similarity
        is below the non-linkability threshold.

        Args:
            original_image: Original image with real face.
            deidentified_image: De-identified image to test.
            threshold: Maximum allowed similarity score (e.g., 0.05).
                Images with similarity > threshold are considered linkable.

        Returns:
            Dict with keys:
            - is_non_linkable: bool
            - similarity_score: float
            - threshold: float
            - embedding_model: str
            - frvt_compliant: bool
        """
        ...

    async def extract_embedding(
        self,
        image: PILImage.Image,
    ) -> list[float]:
        """Extract a face embedding vector from an image.

        Returns:
            512-dimensional face embedding vector, or empty list
            if no face is detected.
        """
        ...
