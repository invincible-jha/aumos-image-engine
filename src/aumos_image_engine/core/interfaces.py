"""Protocol interfaces for the image engine's hexagonal architecture.

These protocols define the contracts between the core domain and adapters.
Concrete implementations live in aumos_image_engine/adapters/.

Protocols:
- ImageGeneratorProtocol: Generates synthetic images from prompts or configs
- FaceDeidentifierProtocol: Detects and de-identifies faces in images
- MetadataStripperProtocol: Removes EXIF/IPTC/XMP/DICOM metadata
- WatermarkerProtocol: Embeds C2PA provenance and invisible watermarks
- BiometricVerifierProtocol: Verifies NIST FRVT non-linkability compliance
- ImageQualityEvaluatorProtocol: Computes FID, IS, LPIPS, SSIM, PSNR
- MedicalImagingProtocol: DICOM creation, validation, and anonymization
- ImageExportProtocol: Multi-format export and MinIO/S3 storage upload
"""

from __future__ import annotations

import uuid
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


@runtime_checkable
class ImageQualityEvaluatorProtocol(Protocol):
    """Contract for image quality evaluation adapters.

    Computes standard quality metrics to assess synthetic image fidelity:
    FID (distributional realism), IS (quality + diversity), LPIPS
    (perceptual similarity), SSIM (structural similarity), and PSNR
    (pixel-level signal-to-noise ratio).

    Implementations: InceptionQualityEvaluator
    """

    async def compute_fid(
        self,
        real_images: list[PILImage.Image],
        synthetic_images: list[PILImage.Image],
    ) -> float:
        """Compute Frechet Inception Distance between real and synthetic sets.

        Args:
            real_images: Reference set of real images.
            synthetic_images: Synthetic images to evaluate.

        Returns:
            FID score (float). Lower is better. < 50 is generally acceptable.
        """
        ...

    async def compute_inception_score(
        self,
        synthetic_images: list[PILImage.Image],
        num_splits: int,
    ) -> tuple[float, float]:
        """Compute Inception Score mean and std for a set of synthetic images.

        Returns:
            Tuple of (mean IS, std IS). Higher is better.
        """
        ...

    async def compute_lpips(
        self,
        reference_image: PILImage.Image,
        synthetic_image: PILImage.Image,
    ) -> float:
        """Compute Learned Perceptual Image Patch Similarity.

        Returns:
            LPIPS score in [0, 1]. Lower = more perceptually similar.
        """
        ...

    async def compute_ssim(
        self,
        reference_image: PILImage.Image,
        synthetic_image: PILImage.Image,
    ) -> float:
        """Compute Structural Similarity Index.

        Returns:
            SSIM score in [0, 1]. Higher = more similar.
        """
        ...

    async def compute_psnr(
        self,
        reference_image: PILImage.Image,
        synthetic_image: PILImage.Image,
    ) -> float:
        """Compute Peak Signal-to-Noise Ratio in dB.

        Returns:
            PSNR in dB. Higher is better. float("inf") for identical images.
        """
        ...

    async def evaluate_all(
        self,
        reference_images: list[PILImage.Image],
        synthetic_images: list[PILImage.Image],
    ) -> dict[str, Any]:
        """Compute all quality metrics and return aggregated report.

        Returns:
            Dict with fid, inception_score_mean, inception_score_std,
            lpips_mean, ssim_mean, psnr_mean_db, overall_quality_score.
        """
        ...


@runtime_checkable
class MedicalImagingProtocol(Protocol):
    """Contract for DICOM medical image creation and validation adapters.

    Creates properly formatted DICOM files with synthetic patient data,
    modality-appropriate acquisition parameters, and anonymization-aware
    metadata. Validates DICOM IOD compliance.

    Implementations: DicomMedicalImagingAdapter
    """

    async def create_dicom_from_pil(
        self,
        image: PILImage.Image,
        modality: str,
        anatomy: str,
        synthetic_patient_id: str,
        study_uid: str | None,
        series_uid: str | None,
        acquisition_params: dict[str, Any] | None,
    ) -> bytes:
        """Create a DICOM file from a PIL image.

        Args:
            image: Source synthetic image.
            modality: DICOM modality code (CT, MR, DX, US, PT).
            anatomy: Anatomy keyword selecting the parameter profile
                (e.g., "chest_xray", "brain_mri", "abdominal_ct").
            synthetic_patient_id: Non-real patient identifier for DICOM headers.
            study_uid: DICOM Study Instance UID (generated if None).
            series_uid: DICOM Series Instance UID (generated if None).
            acquisition_params: Override specific acquisition parameters.

        Returns:
            DICOM file bytes.
        """
        ...

    async def validate_dicom(self, dicom_bytes: bytes) -> dict[str, Any]:
        """Validate a DICOM file for IOD compliance.

        Returns:
            Dict with: valid (bool), modality (str), errors (list[str]),
            warnings (list[str]), sop_class (str).
        """
        ...

    async def anonymize_dicom(self, dicom_bytes: bytes) -> bytes:
        """Remove all patient-identifying attributes from a DICOM file.

        Returns:
            Anonymized DICOM bytes.
        """
        ...


@runtime_checkable
class ImageExportProtocol(Protocol):
    """Contract for image export and object storage adapters.

    Handles format encoding (PNG, JPEG, WebP, TIFF), color space
    conversion, resolution control, thumbnail generation, and upload
    to MinIO/S3-compatible object storage.

    Implementations: ImageExportHandler
    """

    async def export_png(
        self,
        image: PILImage.Image,
        compression_level: int,
        optimize: bool,
        include_alpha: bool,
    ) -> bytes:
        """Export image as PNG bytes.

        Args:
            image: Input PIL image.
            compression_level: zlib compression level (0-9).
            optimize: Search for smaller file size (slower).
            include_alpha: Preserve alpha channel if present.

        Returns:
            PNG-encoded bytes.
        """
        ...

    async def export_jpeg(
        self,
        image: PILImage.Image,
        quality: int,
        progressive: bool,
        optimize: bool,
        subsampling: int,
    ) -> bytes:
        """Export image as JPEG bytes.

        Args:
            image: Input PIL image.
            quality: JPEG quality (1-95).
            progressive: Encode as progressive JPEG.
            optimize: Optimal Huffman encoding.
            subsampling: Chroma subsampling (0=4:4:4, 1=4:2:2, 2=4:2:0).

        Returns:
            JPEG-encoded bytes.
        """
        ...

    async def export_webp(
        self,
        image: PILImage.Image,
        quality: int,
        lossless: bool,
        method: int,
    ) -> bytes:
        """Export image as WebP bytes.

        Args:
            image: Input PIL image.
            quality: WebP quality for lossy encoding (1-100).
            lossless: Use lossless encoding.
            method: Encoding method (0=fastest, 6=smallest).

        Returns:
            WebP-encoded bytes.
        """
        ...

    async def export_tiff(
        self,
        image: PILImage.Image,
        compression: str,
        dpi: tuple[int, int],
        bit_depth_16: bool,
    ) -> bytes:
        """Export image as TIFF bytes.

        Args:
            image: Input PIL image.
            compression: TIFF compression ("none", "lzw", "deflate").
            dpi: Resolution as (horizontal_dpi, vertical_dpi).
            bit_depth_16: Promote to 16-bit precision.

        Returns:
            TIFF-encoded bytes.
        """
        ...

    async def upload_to_storage(
        self,
        image_bytes: bytes,
        object_name: str | None,
        bucket: str | None,
        content_type: str,
        metadata: dict[str, str] | None,
        generate_presigned_url: bool,
        presigned_expiry_seconds: int,
    ) -> dict[str, Any]:
        """Upload image bytes to MinIO/S3 object storage.

        Returns:
            Dict with: object_name, bucket, size_bytes, etag, presigned_url.
        """
        ...

    async def generate_thumbnail(
        self,
        image: PILImage.Image,
        max_side: int | None,
        output_format: str,
        jpeg_quality: int,
    ) -> bytes:
        """Generate a thumbnail image.

        Returns:
            Thumbnail image bytes.
        """
        ...

    async def export_and_upload(
        self,
        image: PILImage.Image,
        output_format: str,
        job_id: uuid.UUID,
        tenant_id: uuid.UUID,
        export_options: dict[str, Any] | None,
        generate_thumbnail: bool,
        bucket: str | None,
    ) -> dict[str, Any]:
        """Export and upload image in one call.

        Returns:
            Dict with: object_name, bucket, thumbnail_object_name,
            size_bytes, format, presigned_url.
        """
        ...
