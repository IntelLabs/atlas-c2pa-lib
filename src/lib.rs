//! # Atlas C2PA Library
//!
//! `atlas-c2pa-lib` is a Rust library for creating, signing, and verifying machine learning
//! assets (models and datasets) with C2PA (Content Provenance and Authenticity) specifications.
//!
//! The library provides tools to generate cryptographic claims about ML asset provenance,
//! track ML asset lineage, and create C2PA-compliant manifests.
//!
//! ## Key Components
//!
//! - **Assertions**: Claims about ML models and datasets
//! - **Ingredients**: Tracking datasets and components used to create models
//! - **Manifests**: Complete C2PA manifests for ML assets
//! - **Asset Types**: Support for various ML frameworks (TensorFlow, PyTorch, ONNX, etc.)
//!
//! ## Example: Creating a Model Manifest
//!
//! ```rust
//! use atlas_c2pa_lib::ml::types::{ModelInfo, MLFramework, ModelFormat};
//! use atlas_c2pa_lib::ml::manifest::MLManifestBuilder;
//! use time::OffsetDateTime;
//!
//! // Define model information
//! let model_info = ModelInfo {
//!     name: "bert-base".to_string(),
//!     version: "1.0.0".to_string(),
//!     framework: MLFramework::PyTorch,
//!     format: ModelFormat::TorchScript,
//! };
//!
//! // Create a manifest builder
//! let builder = MLManifestBuilder::new(
//!     model_info.clone(),
//!     "0123456789abcdef0123456789abcdef".to_string(), // Model hash
//! );
//!
//! // Build the manifest
//! let manifest = builder.build().unwrap();
//! ```
pub mod assertion;
pub mod asset_type;
pub mod cbor;
pub mod claim;
pub mod cose;
pub mod cross_reference;
pub mod datetime_wrapper;
pub mod ingredient;
pub mod manifest;
pub mod ml;
