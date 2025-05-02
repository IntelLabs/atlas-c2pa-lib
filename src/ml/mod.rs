//! # ML Module
//!
//! This module provides specialized types and utilities for working with machine learning
//! assets in the C2PA framework. It includes submodules for asset types, assertions, and
//! manifest creation specifically tailored to ML use cases.
//!
//! ## Submodules
//!
//! - **types**: Defines ML-specific data types for models, frameworks, and training metadata
//! - **assertions**: Provides ML-specific assertion types and creation functions
//! - **manifest**: Contains utilities for building ML-specific manifests
//!
//! ## Example: Creating an ML Model Manifest
//!
//! ```rust
//! use atlas_c2pa_lib::ml::types::{ModelInfo, MLFramework, ModelFormat};
//! use atlas_c2pa_lib::ml::manifest::MLManifestBuilder;
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
pub mod assertions;
pub mod ingredient;
pub mod manifest;
pub mod types;
