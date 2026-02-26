# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Report security vulnerabilities to: security@aumos.ai

**Do not open public GitHub issues for security vulnerabilities.**

Include in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Privacy Considerations

aumos-image-engine processes sensitive biometric data:

1. **Face Detection Data**: Detected face coordinates and landmarks are never stored permanently
2. **De-identification**: All face de-identification is irreversible — original biometrics cannot be recovered
3. **Metadata Stripping**: EXIF/IPTC/XMP data is completely removed before storage
4. **C2PA Provenance**: Provenance records are synthetic-image-only markers, not biometric identifiers
5. **NIST FRVT Compliance**: Biometric non-linkability is verified against FRVT standards

## Responsible AI Use

This service is designed for:
- Synthetic training data generation
- Privacy-preserving data augmentation
- Research with proper IRB approval

Prohibited uses:
- Generating non-consensual synthetic images of real individuals
- Bypassing biometric authentication systems
- Creating deepfakes for deceptive purposes

## Response Timeline

We aim to respond within 48 hours and provide a fix within 7 days for critical issues.
