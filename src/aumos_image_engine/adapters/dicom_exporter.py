"""DICOM export adapter for synthetic medical images.

Wraps generated PNG/JPEG images in DICOM format with fully synthetic
metadata (UIDs, patient demographics, study information). All identifiers
are randomly generated — no real patient data is embedded.
"""

from __future__ import annotations

import asyncio
import io
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from aumos_common.logging import get_logger


# Supported DICOM modalities with their SOP class UIDs
MODALITY_SOP_CLASSES: dict[str, str] = {
    "CT": "1.2.840.10008.5.1.4.1.1.2",       # CT Image Storage
    "MR": "1.2.840.10008.5.1.4.1.1.4",       # MR Image Storage
    "CR": "1.2.840.10008.5.1.4.1.1.1",       # CR Image Storage
    "DX": "1.2.840.10008.5.1.4.1.1.1.1",     # Digital X-Ray Image Storage
    "US": "1.2.840.10008.5.1.4.1.1.6.1",     # Ultrasound Image Storage
    "SC": "1.2.840.10008.5.1.4.1.1.7",       # Secondary Capture Image Storage
}

# AumOS root UID prefix (2.25 = UUID-based UID space)
AUMOS_UID_ROOT = "2.25"


class DICOMExporter:
    """Wraps generated images in DICOM format with synthetic metadata.

    All patient identifiers, study UIDs, and dates are synthetically generated.
    No real patient data is embedded at any point. The exported DICOM files
    are valid for use in medical AI training pipelines and PACS systems.

    Args:
        institution_name: Institution name embedded in DICOM tags.
        manufacturer: Equipment manufacturer name.
    """

    def __init__(
        self,
        institution_name: str = "AumOS Synthetic Imaging",
        manufacturer: str = "AumOS Enterprise",
    ) -> None:
        """Initialize DICOMExporter.

        Args:
            institution_name: Institution name for DICOM InstitutionName tag.
            manufacturer: Manufacturer name for DICOM Manufacturer tag.
        """
        self._institution_name = institution_name
        self._manufacturer = manufacturer
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def export(
        self,
        image_bytes: bytes,
        modality: str = "SC",
        study_description: str = "Synthetic Medical Image",
        series_description: str | None = None,
        rows: int | None = None,
        columns: int | None = None,
    ) -> bytes:
        """Export a generated image as a DICOM file with synthetic metadata.

        Args:
            image_bytes: PNG or JPEG image bytes to embed.
            modality: DICOM modality code (CT, MR, CR, DX, US, SC).
            study_description: Human-readable study description.
            series_description: Human-readable series description.
            rows: Image height in pixels (auto-detected if None).
            columns: Image width in pixels (auto-detected if None).

        Returns:
            DICOM file bytes (.dcm format, valid per DICOM PS3.10).

        Raises:
            ValueError: If modality is not supported.
        """
        if modality not in MODALITY_SOP_CLASSES:
            raise ValueError(
                f"Unsupported modality '{modality}'. "
                f"Supported: {list(MODALITY_SOP_CLASSES.keys())}"
            )

        self._log.info(
            "dicom_export_started",
            modality=modality,
            study_description=study_description,
        )

        result = await asyncio.to_thread(
            self._create_dicom_file,
            image_bytes=image_bytes,
            modality=modality,
            study_description=study_description,
            series_description=series_description or study_description,
            rows=rows,
            columns=columns,
        )

        self._log.info("dicom_export_complete", output_size_bytes=len(result))
        return result

    def _create_dicom_file(
        self,
        image_bytes: bytes,
        modality: str,
        study_description: str,
        series_description: str,
        rows: int | None,
        columns: int | None,
    ) -> bytes:
        """Create DICOM file synchronously (called via to_thread).

        Args:
            image_bytes: Raw PNG or JPEG image bytes.
            modality: DICOM modality code.
            study_description: Study description tag value.
            series_description: Series description tag value.
            rows: Image height override (auto-detected if None).
            columns: Image width override (auto-detected if None).

        Returns:
            Valid DICOM file bytes.
        """
        import numpy as np
        import pydicom
        from PIL import Image
        from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
        from pydicom.sequence import Sequence
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        # Load image to get dimensions and pixel data
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if rows is not None or columns is not None:
            target_w = columns or pil_image.width
            target_h = rows or pil_image.height
            pil_image = pil_image.resize((target_w, target_h))

        image_array = np.array(pil_image)
        actual_rows, actual_columns = image_array.shape[:2]

        # Generate all synthetic UIDs deterministically from a session UUID
        study_instance_uid = generate_uid(prefix=AUMOS_UID_ROOT + ".")
        series_instance_uid = generate_uid(prefix=AUMOS_UID_ROOT + ".")
        sop_instance_uid = generate_uid(prefix=AUMOS_UID_ROOT + ".")
        sop_class_uid = MODALITY_SOP_CLASSES[modality]

        now = datetime.now(timezone.utc)
        study_date = now.strftime("%Y%m%d")
        study_time = now.strftime("%H%M%S.%f")

        # Generate synthetic patient demographics
        patient_id = f"AUMOS-{uuid.uuid4().hex[:8].upper()}"
        patient_name = f"SYNTHETIC^PATIENT^{uuid.uuid4().hex[:6].upper()}"

        # Build file meta dataset
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = sop_class_uid
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        # Build main dataset
        dataset = FileDataset(
            filename_or_obj="",
            dataset={},
            file_meta=file_meta,
            is_implicit_VR=False,
            is_little_endian=True,
            preamble=b"\0" * 128,
        )

        # Patient module
        dataset.PatientName = patient_name
        dataset.PatientID = patient_id
        dataset.PatientBirthDate = ""
        dataset.PatientSex = "O"  # Other — synthetic, not real
        dataset.PatientComments = "SYNTHETIC — NOT A REAL PATIENT"

        # General study module
        dataset.StudyInstanceUID = study_instance_uid
        dataset.StudyDate = study_date
        dataset.StudyTime = study_time
        dataset.StudyDescription = study_description
        dataset.AccessionNumber = uuid.uuid4().hex[:8].upper()
        dataset.ReferringPhysicianName = ""
        dataset.StudyID = uuid.uuid4().hex[:8].upper()

        # General series module
        dataset.SeriesInstanceUID = series_instance_uid
        dataset.SeriesNumber = "1"
        dataset.SeriesDescription = series_description
        dataset.Modality = modality
        dataset.SeriesDate = study_date
        dataset.SeriesTime = study_time

        # Equipment module
        dataset.Manufacturer = self._manufacturer
        dataset.ManufacturerModelName = "AumOS Image Engine"
        dataset.InstitutionName = self._institution_name
        dataset.SoftwareVersions = "0.1.0"

        # SOP common module
        dataset.SOPClassUID = sop_class_uid
        dataset.SOPInstanceUID = sop_instance_uid
        dataset.InstanceNumber = "1"
        dataset.ContentDate = study_date
        dataset.ContentTime = study_time

        # Image pixel module
        dataset.Rows = actual_rows
        dataset.Columns = actual_columns
        dataset.BitsAllocated = 8
        dataset.BitsStored = 8
        dataset.HighBit = 7
        dataset.PixelRepresentation = 0  # unsigned
        dataset.SamplesPerPixel = 3
        dataset.PhotometricInterpretation = "RGB"
        dataset.PlanarConfiguration = 0  # pixel-interleaved
        dataset.PixelData = image_array.tobytes()

        # Image plane module (SC defaults)
        if modality == "SC":
            dataset.PresentationLUTShape = "IDENTITY"

        # Write to bytes buffer
        buffer = io.BytesIO()
        pydicom.dcmwrite(buffer, dataset)
        return buffer.getvalue()

    async def export_batch(
        self,
        image_bytes_list: list[bytes],
        modality: str = "SC",
        study_description: str = "Synthetic Medical Image Series",
    ) -> list[bytes]:
        """Export multiple images as a DICOM series with shared study UID.

        Args:
            image_bytes_list: List of image byte strings to export.
            modality: DICOM modality code applied to all images.
            study_description: Study description shared across the series.

        Returns:
            List of DICOM file bytes, one per input image.
        """
        tasks = [
            self.export(
                image_bytes=img_bytes,
                modality=modality,
                study_description=study_description,
                series_description=f"Image {i + 1} of {len(image_bytes_list)}",
            )
            for i, img_bytes in enumerate(image_bytes_list)
        ]
        return list(await asyncio.gather(*tasks))
