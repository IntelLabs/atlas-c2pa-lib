# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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