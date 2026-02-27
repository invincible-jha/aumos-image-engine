"""DICOM medical imaging adapter for synthetic medical image synthesis.

Handles creation, validation, and anonymization of DICOM files for
synthetic medical imaging datasets. Integrates with aumos-healthcare-synth
for anatomy-specific generation parameters.

Supported modalities:
- CR/DX: Computed Radiography / Digital X-ray
- CT: Computed Tomography (Hounsfield unit range -1000 to 3000)
- MR: Magnetic Resonance Imaging (multi-sequence: T1, T2, FLAIR, DWI)
- US: Ultrasound (grayscale, sector/linear probe simulation)
- PT: Positron Emission Tomography (SUV-normalized)

DICOM compliance:
- Proper UIDs generated per DICOM PS3.5 standard
- Required Type 1/2/3 attributes populated per IOD
- Patient data generated as synthetic (GDPR-safe)
- SOP Class UIDs for each modality
"""

from __future__ import annotations

import io
import struct
import uuid
from datetime import datetime, timezone
from functools import partial
from typing import Any

import numpy as np
import structlog
from PIL import Image as PILImage

logger = structlog.get_logger(__name__)

try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import generate_uid, UID

    _PYDICOM_AVAILABLE = True
except ImportError:
    _PYDICOM_AVAILABLE = False

# DICOM SOP Class UIDs per modality
_SOP_CLASS_UIDS: dict[str, str] = {
    "CR": "1.2.840.10008.5.1.4.1.1.1",       # Computed Radiography Image
    "DX": "1.2.840.10008.5.1.4.1.1.1.1",     # Digital X-Ray Image
    "CT": "1.2.840.10008.5.1.4.1.1.2",       # CT Image
    "MR": "1.2.840.10008.5.1.4.1.1.4",       # MR Image
    "US": "1.2.840.10008.5.1.4.1.1.6.1",     # Ultrasound Image
    "PT": "1.2.840.10008.5.1.4.1.1.128",     # PET Image
    "NM": "1.2.840.10008.5.1.4.1.1.20",      # Nuclear Medicine Image
    "XA": "1.2.840.10008.5.1.4.1.1.12.1",   # X-Ray Angiographic Image
}

# DICOM Transfer Syntax UIDs
_EXPLICIT_VR_LITTLE_ENDIAN = "1.2.840.10008.1.2.1"
_JPEG_BASELINE = "1.2.840.10008.1.2.4.50"

# Anatomy-specific parameter profiles for generation
_ANATOMY_PROFILES: dict[str, dict[str, Any]] = {
    "chest_xray": {
        "rows": 2048,
        "columns": 2048,
        "bits_allocated": 16,
        "bits_stored": 12,
        "photometric": "MONOCHROME2",
        "modality": "DX",
        "body_part": "CHEST",
        "pixel_spacing": [0.143, 0.143],
        "kvp": 120.0,
        "exposure_time": 5,
    },
    "brain_mri": {
        "rows": 256,
        "columns": 256,
        "bits_allocated": 16,
        "bits_stored": 12,
        "photometric": "MONOCHROME2",
        "modality": "MR",
        "body_part": "BRAIN",
        "pixel_spacing": [0.9375, 0.9375],
        "slice_thickness": 5.0,
        "repetition_time": 2500.0,
        "echo_time": 30.0,
        "mr_acquisition_type": "2D",
    },
    "abdominal_ct": {
        "rows": 512,
        "columns": 512,
        "bits_allocated": 16,
        "bits_stored": 16,
        "photometric": "MONOCHROME2",
        "modality": "CT",
        "body_part": "ABDOMEN",
        "pixel_spacing": [0.703, 0.703],
        "slice_thickness": 5.0,
        "rescale_intercept": -1024,
        "rescale_slope": 1,
        "window_center": 40,
        "window_width": 400,
    },
    "cardiac_us": {
        "rows": 600,
        "columns": 800,
        "bits_allocated": 8,
        "bits_stored": 8,
        "photometric": "YBR_FULL_422",
        "modality": "US",
        "body_part": "HEART",
        "pixel_spacing": None,  # US typically doesn't have pixel spacing
        "ultrasound_color_data_present": 0,
    },
}


class DicomMedicalImagingAdapter:
    """DICOM medical imaging adapter for synthetic dataset generation.

    Creates properly formatted DICOM files with synthetic patient data,
    anatomy-specific acquisition parameters, and anonymization-aware
    metadata. Validates generated DICOM against IOD compliance rules.
    """

    def __init__(
        self,
        implementation_uid_prefix: str = "1.2.826.0.1.3680043.10.519",
        default_institution: str = "AumOS Synthetic Health Institute",
    ) -> None:
        """Initialize the DICOM adapter.

        Args:
            implementation_uid_prefix: DICOM implementation UID prefix for
                generated Study/Series/SOP Instance UIDs.
            default_institution: Institution name embedded in DICOM headers.
        """
        self._uid_prefix = implementation_uid_prefix
        self._default_institution = default_institution
        self._log = logger.bind(adapter="medical_imaging")

        if not _PYDICOM_AVAILABLE:
            self._log.warning(
                "medical_imaging.pydicom_unavailable",
                note="DICOM creation will use minimal fallback format",
            )

    async def create_dicom_from_pil(
        self,
        image: PILImage.Image,
        modality: str,
        anatomy: str,
        synthetic_patient_id: str,
        study_uid: str | None = None,
        series_uid: str | None = None,
        acquisition_params: dict[str, Any] | None = None,
    ) -> bytes:
        """Create a DICOM file from a PIL image.

        Args:
            image: Source synthetic image.
            modality: DICOM modality code (CT, MR, DX, US, PT, etc.).
            anatomy: Anatomy keyword (e.g., "chest_xray", "brain_mri").
                Used to select the appropriate parameter profile.
            synthetic_patient_id: Synthetic (non-real) patient identifier.
            study_uid: DICOM Study Instance UID. Generated if None.
            series_uid: DICOM Series Instance UID. Generated if None.
            acquisition_params: Override specific acquisition parameters.
                Merged over anatomy profile defaults.

        Returns:
            DICOM file as raw bytes.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        create_fn = partial(
            self._create_dicom_sync,
            image=image,
            modality=modality.upper(),
            anatomy=anatomy,
            synthetic_patient_id=synthetic_patient_id,
            study_uid=study_uid,
            series_uid=series_uid,
            acquisition_params=acquisition_params or {},
        )
        return await loop.run_in_executor(None, create_fn)

    def _create_dicom_sync(
        self,
        image: PILImage.Image,
        modality: str,
        anatomy: str,
        synthetic_patient_id: str,
        study_uid: str | None,
        series_uid: str | None,
        acquisition_params: dict[str, Any],
    ) -> bytes:
        """Synchronous DICOM creation."""
        if not _PYDICOM_AVAILABLE:
            return self._create_minimal_dicom(image, modality, synthetic_patient_id)

        # Merge anatomy profile with override params
        profile = dict(_ANATOMY_PROFILES.get(anatomy, _ANATOMY_PROFILES.get("chest_xray", {})))
        profile.update(acquisition_params)

        # Override modality from acquisition_params if provided
        effective_modality = acquisition_params.get("modality", profile.get("modality", modality))

        # Generate UIDs
        sop_instance_uid = generate_uid(prefix=self._uid_prefix + ".")
        effective_study_uid = study_uid or generate_uid(prefix=self._uid_prefix + ".")
        effective_series_uid = series_uid or generate_uid(prefix=self._uid_prefix + ".")

        # Build file meta information dataset
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = _SOP_CLASS_UIDS.get(
            effective_modality, _SOP_CLASS_UIDS["CT"]
        )
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.TransferSyntaxUID = _EXPLICIT_VR_LITTLE_ENDIAN
        file_meta.ImplementationClassUID = generate_uid(prefix=self._uid_prefix + ".")
        file_meta.ImplementationVersionName = "AUMOS_IMG_1.0"

        # Build the main dataset
        dataset = Dataset()
        dataset.file_meta = file_meta
        dataset.is_implicit_VR = False
        dataset.is_little_endian = True

        # --- Patient Module (all synthetic) ---
        dataset.PatientName = f"SYNTHETIC^{synthetic_patient_id[:8].upper()}"
        dataset.PatientID = synthetic_patient_id
        dataset.PatientBirthDate = ""  # Stripped for anonymization
        dataset.PatientSex = ""        # Not assigned to avoid bias
        dataset.PatientAge = ""
        dataset.PatientWeight = None

        # --- General Study Module ---
        now = datetime.now(timezone.utc)
        dataset.StudyInstanceUID = effective_study_uid
        dataset.StudyDate = now.strftime("%Y%m%d")
        dataset.StudyTime = now.strftime("%H%M%S.%f")[:14]
        dataset.StudyID = f"AUMOS_{synthetic_patient_id[:6]}"
        dataset.AccessionNumber = f"SYN{uuid.uuid4().hex[:8].upper()}"
        dataset.StudyDescription = f"Synthetic {anatomy.replace('_', ' ').title()}"
        dataset.ReferringPhysicianName = ""

        # --- General Series Module ---
        dataset.SeriesInstanceUID = effective_series_uid
        dataset.SeriesNumber = 1
        dataset.SeriesDate = dataset.StudyDate
        dataset.SeriesTime = dataset.StudyTime
        dataset.Modality = effective_modality
        dataset.SeriesDescription = f"Synthetic {effective_modality} {anatomy}"
        dataset.BodyPartExamined = profile.get("body_part", "UNKNOWN")

        # --- General Equipment Module ---
        dataset.Manufacturer = "AumOS Synthetic Imaging"
        dataset.InstitutionName = self._default_institution
        dataset.ManufacturerModelName = "AumOS Image Engine"
        dataset.SoftwareVersions = "1.0.0"
        dataset.DeviceSerialNumber = "SYN-" + uuid.uuid4().hex[:8].upper()

        # --- SOP Common Module ---
        dataset.SOPClassUID = _SOP_CLASS_UIDS.get(effective_modality, _SOP_CLASS_UIDS["CT"])
        dataset.SOPInstanceUID = sop_instance_uid
        dataset.InstanceCreationDate = dataset.StudyDate
        dataset.InstanceCreationTime = dataset.StudyTime
        dataset.SpecificCharacterSet = "ISO_IR 192"  # UTF-8

        # --- Image Pixel Module ---
        pixel_array = self._prepare_pixel_array(image, profile, effective_modality)
        rows, columns = pixel_array.shape[:2] if pixel_array.ndim == 2 else pixel_array.shape[:2]
        bits_allocated = profile.get("bits_allocated", 16)
        bits_stored = profile.get("bits_stored", 12)

        dataset.Rows = rows
        dataset.Columns = columns
        dataset.BitsAllocated = bits_allocated
        dataset.BitsStored = bits_stored
        dataset.HighBit = bits_stored - 1
        dataset.PixelRepresentation = 0  # Unsigned
        dataset.SamplesPerPixel = 1 if pixel_array.ndim == 2 else 3
        dataset.PhotometricInterpretation = profile.get("photometric", "MONOCHROME2")
        dataset.PixelSpacing = profile.get("pixel_spacing", [1.0, 1.0])

        # --- Modality-specific attributes ---
        self._apply_modality_attributes(dataset, effective_modality, profile)

        # Set pixel data
        if bits_allocated == 16:
            pixel_bytes = pixel_array.astype(np.uint16).tobytes()
        else:
            pixel_bytes = pixel_array.astype(np.uint8).tobytes()
        dataset.PixelData = pixel_bytes

        # Serialize to bytes
        output_buffer = io.BytesIO()
        pydicom.dcmwrite(output_buffer, dataset, write_like_original=False)
        dicom_bytes = output_buffer.getvalue()

        self._log.info(
            "medical_imaging.dicom_created",
            modality=effective_modality,
            anatomy=anatomy,
            size_bytes=len(dicom_bytes),
            sop_uid=sop_instance_uid[:20] + "...",
        )
        return dicom_bytes

    def _prepare_pixel_array(
        self,
        image: PILImage.Image,
        profile: dict[str, Any],
        modality: str,
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Convert PIL image to DICOM-appropriate pixel array.

        Applies modality-specific Hounsfield unit mapping for CT,
        signal intensity mapping for MR, and grayscale conversion for others.
        """
        target_rows = profile.get("rows", 512)
        target_cols = profile.get("columns", 512)
        bits_allocated = profile.get("bits_allocated", 16)

        # Resize to match DICOM dimensions
        resized = image.resize((target_cols, target_rows), PILImage.LANCZOS)

        if modality == "CT":
            # Map [0, 255] pixel values to Hounsfield Unit range [-1024, 3072]
            gray = np.array(resized.convert("L"), dtype=np.float32)
            # CT HU range: air=-1024, water=0, bone=+1000
            hu_array = ((gray / 255.0) * 4096 - 1024).clip(-1024, 3071)
            rescale_intercept = profile.get("rescale_intercept", -1024)
            pixel_array = (hu_array - rescale_intercept).clip(0, 65535).astype(np.uint16)

        elif modality == "MR":
            # MR: [0, 4095] signal intensity range (12-bit)
            gray = np.array(resized.convert("L"), dtype=np.float32)
            pixel_array = (gray / 255.0 * 4095).clip(0, 4095).astype(np.uint16)

        elif modality in ("DX", "CR"):
            # Digital X-ray: inverted 12-bit (high pixel = bone/dense tissue)
            gray = np.array(resized.convert("L"), dtype=np.float32)
            inverted = 255.0 - gray  # X-ray convention: bone appears bright
            pixel_array = (inverted / 255.0 * 4095).clip(0, 4095).astype(np.uint16)

        elif modality == "US":
            # Ultrasound: 8-bit grayscale or YBR
            pixel_array = np.array(resized.convert("L"), dtype=np.uint8)

        else:
            # Generic: 16-bit grayscale
            gray = np.array(resized.convert("L"), dtype=np.float32)
            pixel_array = (gray / 255.0 * 65535).clip(0, 65535).astype(np.uint16)

        return pixel_array

    def _apply_modality_attributes(
        self,
        dataset: Any,
        modality: str,
        profile: dict[str, Any],
    ) -> None:
        """Apply modality-specific DICOM attributes."""
        if modality == "CT":
            dataset.RescaleIntercept = profile.get("rescale_intercept", -1024)
            dataset.RescaleSlope = profile.get("rescale_slope", 1)
            dataset.RescaleType = "HU"
            dataset.WindowCenter = profile.get("window_center", 40)
            dataset.WindowWidth = profile.get("window_width", 400)
            dataset.SliceThickness = profile.get("slice_thickness", 5.0)
            dataset.KVP = profile.get("kvp", 120.0)
            dataset.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]

        elif modality == "MR":
            dataset.RepetitionTime = profile.get("repetition_time", 2500.0)
            dataset.EchoTime = profile.get("echo_time", 30.0)
            dataset.MRAcquisitionType = profile.get("mr_acquisition_type", "2D")
            dataset.SliceThickness = profile.get("slice_thickness", 5.0)
            dataset.ImageType = ["ORIGINAL", "PRIMARY", "M"]
            dataset.ScanningSequence = "SE"
            dataset.SequenceVariant = "NONE"
            dataset.ScanOptions = ""

        elif modality in ("DX", "CR"):
            dataset.KVP = profile.get("kvp", 120.0)
            dataset.ExposureTime = profile.get("exposure_time", 5)
            dataset.ImageType = ["ORIGINAL", "PRIMARY"]
            dataset.PresentationIntentType = "FOR PRESENTATION"
            dataset.ImagerPixelSpacing = profile.get("pixel_spacing", [0.143, 0.143])

        elif modality == "US":
            dataset.UltrasoundColorDataPresent = profile.get("ultrasound_color_data_present", 0)
            dataset.ImageType = ["ORIGINAL", "PRIMARY"]

        elif modality == "PT":
            dataset.CorrectedImage = ["ATTN", "DCYD", "SCAT", "DT", "RAN"]
            dataset.DecayCorrection = "START"
            dataset.Units = "BQML"
            dataset.ImageType = ["ORIGINAL", "PRIMARY"]

    def _create_minimal_dicom(
        self,
        image: PILImage.Image,
        modality: str,
        synthetic_patient_id: str,
    ) -> bytes:
        """Create a minimal DICOM-like file without pydicom dependency.

        This is a fallback implementation that creates a file with
        the DICOM preamble and a subset of required tags sufficient
        for basic viewing. Not fully IOD-compliant.
        """
        # DICOM preamble: 128 bytes of zeros + "DICM" magic
        preamble = b"\x00" * 128 + b"DICM"

        # Minimal tag set as raw bytes (explicit VR little endian)
        def encode_tag(group: int, element: int, vr: str, value: bytes) -> bytes:
            tag_bytes = struct.pack("<HH", group, element)
            vr_bytes = vr.encode("ascii")
            if vr in ("OB", "OW", "SQ", "UC", "UN", "UR", "UT"):
                # Long VR: 2 reserved bytes + 4-byte length
                length_bytes = struct.pack("<xxI", len(value))
            else:
                length_bytes = struct.pack("<H", len(value))
            return tag_bytes + vr_bytes + length_bytes + value

        gray_image = image.convert("L").resize((256, 256))
        pixel_data = np.array(gray_image, dtype=np.uint8).tobytes()

        tags = [
            encode_tag(0x0008, 0x0060, "CS", modality.encode("ascii")),
            encode_tag(0x0010, 0x0020, "LO", synthetic_patient_id.encode("ascii")),
            encode_tag(0x0028, 0x0010, "US", struct.pack("<H", 256)),
            encode_tag(0x0028, 0x0011, "US", struct.pack("<H", 256)),
            encode_tag(0x0028, 0x0100, "US", struct.pack("<H", 8)),
            encode_tag(0x0028, 0x0103, "US", struct.pack("<H", 0)),
            encode_tag(0x7FE0, 0x0010, "OW", pixel_data),
        ]

        return preamble + b"".join(tags)

    async def validate_dicom(self, dicom_bytes: bytes) -> dict[str, Any]:
        """Validate a DICOM file for IOD compliance.

        Args:
            dicom_bytes: Raw DICOM file bytes.

        Returns:
            Dict with:
            - valid: bool
            - modality: str | None
            - errors: list[str] — IOD violations
            - warnings: list[str] — non-critical issues
            - sop_class: str | None
        """
        import asyncio

        loop = asyncio.get_running_loop()
        validate_fn = partial(self._validate_dicom_sync, dicom_bytes=dicom_bytes)
        return await loop.run_in_executor(None, validate_fn)

    def _validate_dicom_sync(self, dicom_bytes: bytes) -> dict[str, Any]:
        """Synchronous DICOM validation."""
        errors: list[str] = []
        warnings: list[str] = []

        if not _PYDICOM_AVAILABLE:
            # Basic magic byte check
            valid_magic = len(dicom_bytes) > 132 and dicom_bytes[128:132] == b"DICM"
            return {
                "valid": valid_magic,
                "modality": None,
                "errors": ["pydicom not available for full validation"],
                "warnings": [],
                "sop_class": None,
            }

        try:
            dataset = pydicom.dcmread(io.BytesIO(dicom_bytes), force=True)

            # Check required Type 1 attributes
            required_tags = [
                (0x0028, 0x0010, "Rows"),
                (0x0028, 0x0011, "Columns"),
                (0x0028, 0x0100, "BitsAllocated"),
                (0x7FE0, 0x0010, "PixelData"),
                (0x0008, 0x0060, "Modality"),
            ]
            for group, element, name in required_tags:
                if not hasattr(dataset, name) or getattr(dataset, name) is None:
                    errors.append(f"Missing required Type 1 attribute: {name}")

            modality = getattr(dataset, "Modality", None)
            sop_class = getattr(dataset, "SOPClassUID", None)

            # Check patient data is anonymized
            patient_name = str(getattr(dataset, "PatientName", ""))
            if patient_name and "SYNTHETIC" not in patient_name.upper():
                warnings.append("PatientName may contain real data — ensure it is synthetic")

            if getattr(dataset, "PatientBirthDate", ""):
                warnings.append("PatientBirthDate is set — consider clearing for anonymization")

            self._log.info(
                "medical_imaging.validation_complete",
                valid=len(errors) == 0,
                error_count=len(errors),
                modality=modality,
            )

            return {
                "valid": len(errors) == 0,
                "modality": str(modality) if modality else None,
                "errors": errors,
                "warnings": warnings,
                "sop_class": str(sop_class) if sop_class else None,
            }

        except Exception as exc:
            return {
                "valid": False,
                "modality": None,
                "errors": [f"Parse error: {str(exc)}"],
                "warnings": [],
                "sop_class": None,
            }

    async def anonymize_dicom(self, dicom_bytes: bytes) -> bytes:
        """Remove all patient-identifying attributes from a DICOM file.

        Args:
            dicom_bytes: Raw DICOM bytes to anonymize.

        Returns:
            Anonymized DICOM bytes with patient attributes cleared.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        anon_fn = partial(self._anonymize_dicom_sync, dicom_bytes=dicom_bytes)
        return await loop.run_in_executor(None, anon_fn)

    def _anonymize_dicom_sync(self, dicom_bytes: bytes) -> bytes:
        """Synchronous DICOM anonymization."""
        if not _PYDICOM_AVAILABLE:
            self._log.warning("medical_imaging.anonymization_skipped", reason="pydicom_missing")
            return dicom_bytes

        try:
            dataset = pydicom.dcmread(io.BytesIO(dicom_bytes), force=True)

            # Attributes to clear per DICOM PS3.15 Annex E (De-identification profile)
            tags_to_clear = [
                "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                "PatientAge", "PatientWeight", "PatientAddress", "PatientTelephone",
                "OtherPatientIDs", "AccessionNumber", "ReferringPhysicianName",
                "PerformingPhysicianName", "InstitutionName", "InstitutionAddress",
                "PhysiciansOfRecord", "OperatorsName",
                "StudyID", "RequestingPhysician", "RequestedProcedureDescription",
                "ScheduledProcedureStepDescription", "StudyDescription",
            ]

            for tag_name in tags_to_clear:
                if hasattr(dataset, tag_name):
                    try:
                        delattr(dataset, tag_name)
                    except AttributeError:
                        pass

            # Replace UIDs with new synthetic ones
            if hasattr(dataset, "StudyInstanceUID"):
                dataset.StudyInstanceUID = generate_uid(prefix=self._uid_prefix + ".")
            if hasattr(dataset, "SeriesInstanceUID"):
                dataset.SeriesInstanceUID = generate_uid(prefix=self._uid_prefix + ".")

            output_buffer = io.BytesIO()
            pydicom.dcmwrite(output_buffer, dataset, write_like_original=False)
            result = output_buffer.getvalue()
            self._log.info("medical_imaging.anonymization_complete", size_bytes=len(result))
            return result

        except Exception as exc:
            self._log.error("medical_imaging.anonymization_error", error=str(exc))
            return dicom_bytes

    def get_anatomy_profile(self, anatomy: str) -> dict[str, Any]:
        """Return the parameter profile for a named anatomy.

        Args:
            anatomy: Anatomy identifier (e.g., "chest_xray", "brain_mri").

        Returns:
            Copy of the anatomy profile dict, or empty dict if not found.
        """
        return dict(_ANATOMY_PROFILES.get(anatomy, {}))

    def list_supported_anatomies(self) -> list[str]:
        """Return list of supported anatomy profile names."""
        return list(_ANATOMY_PROFILES.keys())

    def list_supported_modalities(self) -> list[str]:
        """Return list of supported DICOM modality codes."""
        return list(_SOP_CLASS_UIDS.keys())
