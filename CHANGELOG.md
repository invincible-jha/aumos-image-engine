# Changelog

All notable changes to aumos-image-engine will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added
- Initial scaffolding for aumos-image-engine
- Stable Diffusion + ControlNet image generation adapter
- BlenderProc 3D synthetic data generation adapter
- Face de-identification preserving expressions and poses
- EXIF/IPTC/XMP/DICOM metadata stripping
- C2PA provenance signing + invisible watermarking
- NIST FRVT biometric non-linkability verification
- FastAPI service with hexagonal architecture (api/core/adapters)
- SQLAlchemy models: ImageGenerationJob, ImageBatch
- Kafka event publishing for job lifecycle events
- MinIO object storage adapter for image persistence
- Privacy engine HTTP client integration
- Batch processing with progress tracking
- Docker support with NVIDIA CUDA base image
- Multi-tenant support with RLS isolation
