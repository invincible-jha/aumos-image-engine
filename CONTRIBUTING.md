# Contributing to aumos-image-engine

## Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.1+ (for local development with GPU)
- Docker with NVIDIA Container Toolkit
- Access to internal AumOS package registry

## Development Setup

```bash
# Clone the repository
git clone <repo-url>
cd aumos-image-engine

# Install in development mode
pip install -e ".[dev]"

# Copy environment variables
cp .env.example .env
# Edit .env with your local settings

# Start infrastructure
docker compose -f docker-compose.dev.yml up -d

# Run tests
make test
```

## Coding Standards

- All functions must have type hints
- Docstrings required for all public functions and classes
- Use `structlog` for logging — never `print()`
- Async I/O for all database and network operations
- 80% test coverage minimum

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Write implementation + tests
3. Run `make all` (lint + typecheck + test)
4. Open PR with description of changes

## Privacy & Safety

This service handles biometric data processing and synthetic image generation.
All contributors must:
- Never commit real biometric data or face images
- Follow privacy-by-design principles
- Ensure de-identification is complete before storing/returning images
- Review the SECURITY.md for responsible disclosure

## Questions

Open an issue or contact the AumOS platform team.
