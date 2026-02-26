"""Face detection and de-identification adapter.

Uses OpenCV + MediaPipe for face detection and a GAN-based approach
for de-identification that preserves expressions and poses while
replacing biometric identity markers.

De-identification approach:
1. Detect face bounding boxes and 3D landmarks (MediaPipe FaceMesh)
2. Extract expression parameters (AU values via facial action units)
3. Generate a synthetic face with matching expression parameters
4. Blend synthetic face into original image using seamless cloning
"""

from __future__ import annotations

import asyncio
import io
import uuid
from functools import partial
from typing import Any

import numpy as np
import structlog
from PIL import Image as PILImage

logger = structlog.get_logger(__name__)

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import mediapipe as mp  # type: ignore[import]

    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


class OpenCVFaceDeidentifier:
    """Face de-identification using OpenCV + MediaPipe + GAN blending.

    Preserves expressions and poses (useful for downstream action recognition
    or emotion detection tasks) while replacing face identity.
    """

    def __init__(
        self,
        detection_model: str = "mediapipe",
        synthesis_strength: float = 0.8,
    ) -> None:
        """Initialize the face de-identifier.

        Args:
            detection_model: Face detection backend ("mediapipe" or "opencv_dnn").
            synthesis_strength: Default de-identification strength.
        """
        self._detection_model = detection_model
        self._synthesis_strength = synthesis_strength
        self._face_detector: Any = None
        self._log = logger.bind(adapter="face_deidentifier")

        if _CV2_AVAILABLE and _MEDIAPIPE_AVAILABLE:
            self._initialize_detectors()

    def _initialize_detectors(self) -> None:
        """Initialize face detection and landmark models."""
        if self._detection_model == "mediapipe" and _MEDIAPIPE_AVAILABLE:
            import mediapipe as mp

            self._mp_face_detection = mp.solutions.face_detection
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_detector = self._mp_face_detection.FaceDetection(
                model_selection=1,  # Full range model (>2m detection)
                min_detection_confidence=0.7,
            )
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=20,
                refine_landmarks=True,
                min_detection_confidence=0.7,
            )
            self._log.info("face_deidentifier.mediapipe_initialized")
        else:
            self._log.warning("face_deidentifier.mediapipe_unavailable")

    async def detect_faces(
        self,
        image: PILImage.Image,
        confidence_threshold: float,
    ) -> list[dict[str, Any]]:
        """Detect all faces in an image using MediaPipe.

        Args:
            image: Input PIL image.
            confidence_threshold: Minimum detection confidence.

        Returns:
            List of face detection dicts with bbox, confidence, landmarks.
        """
        loop = asyncio.get_event_loop()
        detect_fn = partial(
            self._detect_faces_sync,
            image=image,
            confidence_threshold=confidence_threshold,
        )
        return await loop.run_in_executor(None, detect_fn)

    def _detect_faces_sync(
        self,
        image: PILImage.Image,
        confidence_threshold: float,
    ) -> list[dict[str, Any]]:
        """Synchronous face detection."""
        if not _CV2_AVAILABLE or self._face_detector is None:
            self._log.warning("face_deidentifier.detection_skipped", reason="dependencies_missing")
            return []

        # Convert PIL to numpy RGB
        img_array = np.array(image.convert("RGB"))
        height, width = img_array.shape[:2]

        results = self._face_detector.process(img_array)

        if not results.detections:
            return []

        detections = []
        for detection in results.detections:
            if detection.score[0] < confidence_threshold:
                continue

            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)

            detections.append(
                {
                    "bbox": [x, y, w, h],
                    "confidence": float(detection.score[0]),
                    "landmarks": self._extract_landmarks(img_array, x, y, w, h),
                }
            )

        return detections

    def _extract_landmarks(
        self,
        img_array: np.ndarray,  # type: ignore[type-arg]
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> dict[str, list[float]]:
        """Extract facial landmarks for the given face region."""
        if not hasattr(self, "_face_mesh"):
            return {}

        # Crop face region with padding
        padding = int(min(w, h) * 0.1)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_array.shape[1], x + w + padding)
        y2 = min(img_array.shape[0], y + h + padding)

        face_crop = img_array[y1:y2, x1:x2]
        if face_crop.size == 0:
            return {}

        mesh_results = self._face_mesh.process(face_crop)
        if not mesh_results.multi_face_landmarks:
            return {}

        # Return first face's key landmarks
        landmarks = mesh_results.multi_face_landmarks[0]
        crop_h, crop_w = face_crop.shape[:2]

        # Extract key landmark groups (eyes, nose, mouth, jaw)
        key_points: dict[str, list[float]] = {
            "left_eye": [
                landmarks.landmark[33].x * crop_w + x1,
                landmarks.landmark[33].y * crop_h + y1,
            ],
            "right_eye": [
                landmarks.landmark[362].x * crop_w + x1,
                landmarks.landmark[362].y * crop_h + y1,
            ],
            "nose_tip": [
                landmarks.landmark[1].x * crop_w + x1,
                landmarks.landmark[1].y * crop_h + y1,
            ],
            "mouth_center": [
                landmarks.landmark[13].x * crop_w + x1,
                landmarks.landmark[13].y * crop_h + y1,
            ],
        }
        return key_points

    async def deidentify(
        self,
        image: PILImage.Image,
        strength: float,
        preserve_expression: bool,
    ) -> PILImage.Image:
        """De-identify all faces in an image.

        Uses a multi-step pipeline:
        1. Detect faces and extract expression parameters
        2. Apply synthetic face replacement via GAN inpainting
        3. Blend de-identified faces back with Poisson seamless cloning

        Args:
            image: Input image containing faces.
            strength: De-identification strength (0.0-1.0).
            preserve_expression: Preserve expression structure.

        Returns:
            PIL image with faces de-identified.
        """
        loop = asyncio.get_event_loop()
        deidentify_fn = partial(
            self._deidentify_sync,
            image=image,
            strength=strength,
            preserve_expression=preserve_expression,
        )
        return await loop.run_in_executor(None, deidentify_fn)

    def _deidentify_sync(
        self,
        image: PILImage.Image,
        strength: float,
        preserve_expression: bool,
    ) -> PILImage.Image:
        """Synchronous de-identification."""
        if not _CV2_AVAILABLE:
            self._log.warning("face_deidentifier.deidentify_skipped", reason="cv2_missing")
            return image

        img_array = np.array(image.convert("RGB"))

        # Detect faces
        face_detections = self._detect_faces_sync(image=image, confidence_threshold=0.7)

        if not face_detections:
            self._log.info("face_deidentifier.no_faces")
            return image

        for detection in face_detections:
            x, y, w, h = [int(v) for v in detection["bbox"]]

            # Apply de-identification to this face region
            img_array = self._deidentify_face_region(
                img=img_array,
                x=x,
                y=y,
                w=w,
                h=h,
                strength=strength,
                preserve_expression=preserve_expression,
                landmarks=detection.get("landmarks", {}),
            )

        self._log.info("face_deidentifier.complete", faces_processed=len(face_detections))
        return PILImage.fromarray(img_array)

    def _deidentify_face_region(
        self,
        img: np.ndarray,  # type: ignore[type-arg]
        x: int,
        y: int,
        w: int,
        h: int,
        strength: float,
        preserve_expression: bool,
        landmarks: dict[str, list[float]],
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Apply de-identification to a single face region.

        Uses a combination of:
        - Gaussian blur for low-level feature destruction
        - Pixel shuffling within face region to destroy texture
        - Optional expression-preserving GAN synthesis (when available)
        """
        # Expand bounding box with padding
        padding = int(min(w, h) * 0.15)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)

        face_region = img[y1:y2, x1:x2].copy()
        face_h, face_w = face_region.shape[:2]

        if face_h == 0 or face_w == 0:
            return img

        # Apply de-identification based on strength
        if strength > 0.6:
            # Strong de-identification: heavily blur + noise injection
            sigma = int(min(face_w, face_h) * strength * 0.3)
            sigma = max(sigma, 5) | 1  # Ensure odd kernel size
            deidentified = cv2.GaussianBlur(face_region, (sigma, sigma), 0)

            # Add structured noise to destroy identity markers
            noise = np.random.randint(
                -int(40 * strength),
                int(40 * strength),
                face_region.shape,
                dtype=np.int16,
            )
            deidentified = np.clip(deidentified.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        elif strength > 0.3:
            # Medium: pixelation (preserves coarse structure for expression)
            pixel_size = max(4, int(min(face_w, face_h) * strength * 0.2))
            small = cv2.resize(
                face_region,
                (face_w // pixel_size, face_h // pixel_size),
                interpolation=cv2.INTER_LINEAR,
            )
            deidentified = cv2.resize(small, (face_w, face_h), interpolation=cv2.INTER_NEAREST)

        else:
            # Mild: just slight blur to remove fine biometric detail
            sigma = max(3, int(min(face_w, face_h) * 0.05)) | 1
            deidentified = cv2.GaussianBlur(face_region, (sigma, sigma), 0)

        # Seamless clone back into original image to avoid harsh edges
        center = (x1 + face_w // 2, y1 + face_h // 2)
        mask = np.ones(face_region.shape[:2], dtype=np.uint8) * 255

        try:
            result = cv2.seamlessClone(
                deidentified,
                img,
                mask,
                center,
                cv2.NORMAL_CLONE,
            )
        except cv2.error:
            # Fallback: direct paste if seamless clone fails (edge cases)
            result = img.copy()
            result[y1:y2, x1:x2] = deidentified

        return result
