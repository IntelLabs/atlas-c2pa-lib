# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[0.1.2] - 2025-07-11
Added

Unit tests for all assertion types (#16)
Support for custom assertion validation (#15)

Security

Intel security scan workflow integration

[0.1.1] - 2025-06-11
Added

PartialEq trait implementation for HashAlgorithm struct (#13)
FromStr trait implementation for HashAlgorithm for string parsing
GitHub Actions workflows for CI/CD (#4)

Changed

Upgraded to Rust edition 2024 (#14)
Updated dependencies:

mockito to 1.7.0
base64 to use modern API (#12)


Improved code formatting and clippy compliance (#3, #5)

Fixed

Test compatibility issues
Code formatting inconsistencies

Security

Added OSSF Scorecard workflow


## [0.1.0] - 2025-05-01

### Added
- Initial release of Atlas C2PA Library
- Core functionality for C2PA manifest creation and validation
- Support for ML model and dataset manifests
- Implementation of various assertion types:
  - Action assertions
  - Ingredient assertions
  - Hash assertions
  - Metadata assertions
  - Creative work assertions
  - Custom assertions
  - DoNotTrain assertions
- Comprehensive ML framework support:
  - TensorFlow
  - PyTorch
  - ONNX
  - JAX
  - Keras
  - MxNet
  - ML.NET
  - OpenVINO
- Cross-reference validation between manifests
- CBOR encoding and decoding for claims
- Cryptographic signing and verification using OpenSSL
- Validation functions for all assertion types
- Support for linked ingredients
- Timestamp validation and verification
- Builder pattern for ML model manifests