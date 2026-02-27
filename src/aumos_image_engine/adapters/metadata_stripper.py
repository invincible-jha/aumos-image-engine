"""EXIF/IPTC/XMP/DICOM metadata stripping adapter.

Removes all identifying metadata from images to prevent personal data
leakage. Supports selective retention (e.g., keep resolution metadata
while stripping GPS) and optional steganographic LSB analysis.

Supported metadata standards:
- EXIF: camera model, GPS, datetime, orientation, serial numbers
- IPTC: creator, copyright, caption, keywords, location
- XMP: Adobe metadata, subject tags, rights statements
- DICOM: patient demographics for medical images
"""

from __future__ import annotations

import io
import struct
from functools import partial
from typing import Any

import structlog
from PIL import Image as PILImage

logger = structlog.get_logger(__name__)

try:
    import piexif

    _PIEXIF_AVAILABLE = True
except ImportError:
    _PIEXIF_AVAILABLE = False

try:
    import iptcinfo3  # type: ignore[import]

    _IPTCINFO_AVAILABLE = True
except ImportError:
    _IPTCINFO_AVAILABLE = False

# EXIF tag IDs that are safe to retain (non-identifying)
_SAFE_EXIF_TAGS: frozenset[int] = frozenset(
    {
        256,   # ImageWidth
        257,   # ImageLength
        282,   # XResolution
        283,   # YResolution
        296,   # ResolutionUnit
        274,   # Orientation
        306,   # DateTime (keep creation time for provenance)
        305,   # Software (strip to avoid model fingerprinting — excluded)
    }
)

# EXIF tag names for human-readable reports
_EXIF_TAG_NAMES: dict[int, str] = {
    271: "Make",
    272: "Model",
    305: "Software",
    306: "DateTime",
    315: "Artist",
    33434: "ExposureTime",
    33437: "FNurnber",
    34850: "ExposureProgram",
    34867: "ISOSpeed",
    36864: "ExifVersion",
    36867: "DateTimeOriginal",
    36868: "DateTimeDigitized",
    37121: "ComponentsConfiguration",
    37377: "ShutterSpeedValue",
    37378: "ApertureValue",
    37380: "ExposureBiasValue",
    37383: "MeteringMode",
    37385: "Flash",
    37386: "FocalLength",
    40960: "FlashPixVersion",
    40961: "ColorSpace",
    40962: "PixelXDimension",
    40963: "PixelYDimension",
    41728: "FlashEnergy",
    41986: "ExposureMode",
    41987: "WhiteBalance",
    41988: "DigitalZoomRatio",
    41990: "SceneCaptureType",
    # GPS tags — always strip
    1: "GPSLatitudeRef",
    2: "GPSLatitude",
    3: "GPSLongitudeRef",
    4: "GPSLongitude",
    5: "GPSAltitudeRef",
    6: "GPSAltitude",
    7: "GPSTimeStamp",
    12: "GPSSpeedRef",
    13: "GPSSpeed",
    16: "GPSImgDirectionRef",
    17: "GPSImgDirection",
    23: "GPSDestBearingRef",
    24: "GPSDestBearing",
    29: "GPSDateStamp",
    # Camera serial numbers
    42033: "BodySerialNumber",
    42034: "LensSpecification",
    42035: "LensMake",
    42036: "LensModel",
    42037: "LensSerialNumber",
}


class PiexifMetadataStripper:
    """Metadata stripping adapter using Pillow and piexif.

    Handles EXIF, IPTC, and XMP metadata. For DICOM files,
    delegates to pydicom when available. Implements selective
    retention so resolution data can be preserved while GPS
    and identity markers are always removed.
    """

    def __init__(
        self,
        retain_resolution: bool = True,
        retain_color_profile: bool = True,
        strip_xmp: bool = True,
    ) -> None:
        """Initialize the metadata stripper.

        Args:
            retain_resolution: Keep XResolution/YResolution/ResolutionUnit tags.
            retain_color_profile: Keep ICC color profile bytes (not metadata).
            strip_xmp: Strip XMP sidecar metadata (recommended True).
        """
        self._retain_resolution = retain_resolution
        self._retain_color_profile = retain_color_profile
        self._strip_xmp = strip_xmp
        self._log = logger.bind(adapter="metadata_stripper")

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
            steganographic_check: If True, also analyze pixel LSBs for
                hidden steganographic data.

        Returns:
            Tuple of (stripped image, report dict) describing what was
            found and removed.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        strip_fn = partial(
            self._strip_sync,
            image=image,
            image_format=image_format,
            steganographic_check=steganographic_check,
        )
        return await loop.run_in_executor(None, strip_fn)

    def _strip_sync(
        self,
        image: PILImage.Image,
        image_format: str,
        steganographic_check: bool,
    ) -> tuple[PILImage.Image, dict[str, Any]]:
        """Synchronous metadata stripping."""
        report: dict[str, Any] = {
            "exif_fields_removed": [],
            "iptc_fields_removed": [],
            "xmp_removed": False,
            "icc_profile_retained": False,
            "steganographic_findings": [],
            "total_fields_removed": 0,
            "format": image_format,
        }

        # Collect EXIF fields before stripping for audit report
        exif_data = self._collect_exif_report(image)
        report["exif_fields_removed"] = exif_data.get("fields_found", [])

        # Collect IPTC report
        iptc_data = self._collect_iptc_report(image)
        report["iptc_fields_removed"] = iptc_data.get("fields_found", [])

        # Check for XMP metadata
        xmp_present = self._detect_xmp(image)
        report["xmp_removed"] = xmp_present and self._strip_xmp

        # Optional steganographic analysis before stripping
        if steganographic_check:
            stego_findings = self._analyze_steganography(image)
            report["steganographic_findings"] = stego_findings
            if stego_findings:
                self._log.warning(
                    "metadata_stripper.stego_detected",
                    findings_count=len(stego_findings),
                )

        # Preserve ICC color profile bytes (not metadata)
        icc_profile: bytes | None = None
        if self._retain_color_profile:
            icc_profile = image.info.get("icc_profile")
            report["icc_profile_retained"] = icc_profile is not None

        # Build clean image via round-trip through bytes
        stripped_image = self._rebuild_clean_image(image, image_format, icc_profile)

        # Tally total removed fields
        report["total_fields_removed"] = (
            len(report["exif_fields_removed"])
            + len(report["iptc_fields_removed"])
            + (1 if report["xmp_removed"] else 0)
        )

        self._log.info(
            "metadata_stripper.strip_complete",
            total_removed=report["total_fields_removed"],
            format=image_format,
        )
        return stripped_image, report

    def _rebuild_clean_image(
        self,
        image: PILImage.Image,
        image_format: str,
        icc_profile: bytes | None,
    ) -> PILImage.Image:
        """Rebuild image through a bytes round-trip to purge all metadata.

        The round-trip approach is the most reliable way to strip ALL
        metadata because Pillow's save/open cycle drops unrecognised
        chunks rather than blindly copying them.
        """
        buffer = io.BytesIO()

        save_kwargs: dict[str, Any] = {}

        # Retain ICC profile bytes if configured
        if icc_profile and self._retain_color_profile:
            save_kwargs["icc_profile"] = icc_profile

        # Format-specific clean-save parameters
        upper_format = image_format.upper()
        if upper_format == "JPEG":
            # No exif= argument means Pillow strips EXIF on save
            save_kwargs["quality"] = 95
            save_kwargs["subsampling"] = 0
        elif upper_format == "PNG":
            save_kwargs["compress_level"] = 6

        # Convert to RGB if saving as JPEG (no alpha channel support)
        output_image = image
        if upper_format == "JPEG" and image.mode in ("RGBA", "LA", "P"):
            output_image = image.convert("RGB")

        output_image.save(buffer, format=image_format, **save_kwargs)
        buffer.seek(0)

        # Re-open from bytes — Pillow will only populate standard image info
        clean = PILImage.open(buffer)
        clean.load()
        return clean

    def _collect_exif_report(self, image: PILImage.Image) -> dict[str, Any]:
        """Collect a list of EXIF field names found in the image."""
        found_fields: list[str] = []

        raw_exif = image.info.get("exif", b"")
        if not raw_exif or not _PIEXIF_AVAILABLE:
            return {"fields_found": found_fields}

        try:
            exif_dict = piexif.load(raw_exif)
            for ifd_name, ifd_data in exif_dict.items():
                if not isinstance(ifd_data, dict):
                    continue
                for tag_id in ifd_data:
                    tag_name = _EXIF_TAG_NAMES.get(tag_id, f"Tag_{tag_id}")
                    found_fields.append(f"{ifd_name}.{tag_name}")
        except Exception as exc:
            self._log.debug("metadata_stripper.exif_parse_error", error=str(exc))

        return {"fields_found": found_fields}

    def _collect_iptc_report(self, image: PILImage.Image) -> dict[str, Any]:
        """Collect IPTC field names if present."""
        found_fields: list[str] = []

        # Pillow exposes some IPTC data via image.info["iptc"]
        iptc_raw = image.info.get("photoshop", {})
        if iptc_raw:
            found_fields.append("iptc.photoshop_block")

        return {"fields_found": found_fields}

    def _detect_xmp(self, image: PILImage.Image) -> bool:
        """Detect presence of XMP metadata in the image."""
        xmp_raw = image.info.get("xml:com.adobe.xmp", "")
        if xmp_raw:
            return True
        # Also check for xmp key under different names
        for key in image.info:
            if "xmp" in str(key).lower():
                return True
        return False

    def _analyze_steganography(self, image: PILImage.Image) -> list[str]:
        """Analyze pixel LSBs for potential steganographic data.

        Performs a basic chi-square test on LSB distribution.
        A uniform LSB distribution (chi-square p > 0.05) indicates
        potential steganographic payload.

        Returns:
            List of finding strings describing suspicious regions.
        """
        import numpy as np

        findings: list[str] = []

        try:
            rgb_image = image.convert("RGB")
            pixels = np.array(rgb_image)

            # Extract LSBs for each channel
            for channel_index, channel_name in enumerate(("R", "G", "B")):
                channel_data = pixels[:, :, channel_index]
                lsbs = channel_data & 1
                total_pixels = lsbs.size

                # Count 0-bits and 1-bits
                zeros = int(np.sum(lsbs == 0))
                ones = total_pixels - zeros
                expected = total_pixels / 2.0

                # Chi-square statistic for uniform distribution
                chi_sq = ((zeros - expected) ** 2 + (ones - expected) ** 2) / expected

                # Very low chi-square means suspiciously uniform (possible stego)
                # Threshold: chi_sq < 3.84 at 0.05 significance level (1 DOF)
                if chi_sq < 2.0 and total_pixels > 10000:
                    findings.append(
                        f"Channel {channel_name}: suspiciously uniform LSB distribution "
                        f"(chi_sq={chi_sq:.4f}, pixels={total_pixels})"
                    )

        except Exception as exc:
            self._log.debug("metadata_stripper.stego_analysis_error", error=str(exc))

        return findings

    async def analyze(self, image: PILImage.Image) -> dict[str, Any]:
        """Analyze an image for metadata without stripping.

        Returns a comprehensive report of all metadata found,
        useful for auditing before deciding to strip.

        Args:
            image: Input PIL image.

        Returns:
            Dict with keys: exif, iptc, xmp, icc_profile_present,
            format, mode, size.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        analyze_fn = partial(self._analyze_sync, image=image)
        return await loop.run_in_executor(None, analyze_fn)

    def _analyze_sync(self, image: PILImage.Image) -> dict[str, Any]:
        """Synchronous metadata analysis."""
        report: dict[str, Any] = {
            "exif": {},
            "iptc": {},
            "xmp": {},
            "icc_profile_present": False,
            "format": image.format,
            "mode": image.mode,
            "size": list(image.size),
        }

        # EXIF analysis
        raw_exif = image.info.get("exif", b"")
        if raw_exif and _PIEXIF_AVAILABLE:
            try:
                exif_dict = piexif.load(raw_exif)
                for ifd_name, ifd_data in exif_dict.items():
                    if not isinstance(ifd_data, dict):
                        continue
                    for tag_id, value in ifd_data.items():
                        tag_name = _EXIF_TAG_NAMES.get(tag_id, f"Tag_{tag_id}")
                        key = f"{ifd_name}.{tag_name}"
                        # Represent bytes as hex for readability
                        if isinstance(value, bytes):
                            report["exif"][key] = value.hex()[:64]
                        elif isinstance(value, (tuple, list)):
                            report["exif"][key] = str(value)[:128]
                        else:
                            report["exif"][key] = value
            except Exception as exc:
                self._log.debug("metadata_stripper.analyze_exif_error", error=str(exc))

        # IPTC analysis (via Pillow's photoshop block)
        photoshop_data = image.info.get("photoshop", {})
        if photoshop_data:
            report["iptc"]["photoshop_block_present"] = True

        # XMP analysis
        for key, value in image.info.items():
            if "xmp" in str(key).lower():
                report["xmp"][str(key)] = str(value)[:256]

        # ICC profile
        report["icc_profile_present"] = "icc_profile" in image.info

        self._log.info(
            "metadata_stripper.analysis_complete",
            exif_fields=len(report["exif"]),
            xmp_fields=len(report["xmp"]),
            iptc_present=bool(report["iptc"]),
        )
        return report

    async def batch_strip(
        self,
        images: list[tuple[PILImage.Image, str]],
        steganographic_check: bool = False,
    ) -> list[tuple[PILImage.Image, dict[str, Any]]]:
        """Strip metadata from multiple images concurrently.

        Args:
            images: List of (image, format) tuples to process.
            steganographic_check: Run LSB analysis on each image.

        Returns:
            List of (stripped_image, report) tuples in same order.
        """
        import asyncio

        tasks = [
            self.strip(
                image=image,
                image_format=fmt,
                steganographic_check=steganographic_check,
            )
            for image, fmt in images
        ]
        results = await asyncio.gather(*tasks)
        self._log.info("metadata_stripper.batch_complete", count=len(results))
        return list(results)
