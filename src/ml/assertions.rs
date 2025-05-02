//! # ML Assertions Module
//!
//! This module provides specialized assertion types for machine learning assets in the C2PA
//! framework. It includes structures and functions for creating assertions specific to ML models,
//! their training information, and verification.
//!
//! ## Key Structures
//!
//! - **MLModelAssertion**: A specialized assertion for ML models
//!
//! ## Example: Creating an ML Model Assertion
//!
//! ```rust
//! use atlas_c2pa_lib::ml::assertions::{MLModelAssertion, create_ml_model_assertion};
//! use atlas_c2pa_lib::ml::types::{ModelInfo, MLFramework, ModelFormat};
//! use atlas_c2pa_lib::assertion::Assertion;
//!
//! // Create model information
//! let model_info = ModelInfo {
//!     name: "resnet50".to_string(),
//!     version: "1.0.0".to_string(),
//!     framework: MLFramework::PyTorch,
//!     format: ModelFormat::TorchScript,
//! };
//!
//! // Create an ML model assertion
//! let ml_assertion = MLModelAssertion::new(
//!     model_info.clone(),
//!     None, // No training info
//!     "0123456789abcdef0123456789abcdef".to_string(), // Model hash
//! );
//!
//! //
//! // Create the assertion as an enum variant
//! let assertion = create_ml_model_assertion(
//!     model_info,
//!     None, // No training info
//!     "0123456789abcdef0123456789abcdef".to_string(), // Model hash
//! );
//! ```
//!
//! ## Verification
//!
//! ML model assertions include a verification method:
//!
//! ```rust
//! use atlas_c2pa_lib::ml::assertions::MLModelAssertion;
//! use atlas_c2pa_lib::ml::types::{ModelInfo, MLFramework, ModelFormat};
//!
//! // Create an ML model assertion
//! let ml_assertion = MLModelAssertion::new(
//!     ModelInfo {
//!         name: "resnet50".to_string(),
//!         version: "1.0.0".to_string(),
//!         framework: MLFramework::PyTorch,
//!         format: ModelFormat::TorchScript,
//!     },
//!     None, // No training info
//!     "0123456789abcdef0123456789abcdef".to_string(), // Model hash
//! );
//!
//! // Verify the assertion
//! let verification_result = ml_assertion.verify();
//! assert!(verification_result.is_ok());
//! ```
use super::types::{ModelInfo, TrainingInfo};
use crate::assertion::Assertion;
use crate::assertion::CustomAssertion;
use chrono::Utc;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MLModelAssertion {
    pub model_info: ModelInfo,
    pub training_info: Option<TrainingInfo>,
    pub hash: String,
    pub timestamp: String,
}

impl MLModelAssertion {
    pub fn new(model_info: ModelInfo, training_info: Option<TrainingInfo>, hash: String) -> Self {
        Self {
            model_info,
            training_info,
            hash,
            timestamp: Utc::now().to_rfc3339(),
        }
    }

    pub fn verify(&self) -> Result<(), String> {
        // Basic verification logic
        if self.model_info.name.is_empty() {
            return Err("Model name cannot be empty".to_string());
        }
        if self.hash.is_empty() {
            return Err("Model hash cannot be empty".to_string());
        }
        Ok(())
    }
}

pub fn create_ml_model_assertion(
    model_info: ModelInfo,
    training_info: Option<TrainingInfo>,
    hash: String,
) -> Assertion {
    let ml_assertion = MLModelAssertion::new(model_info, training_info, hash);
    Assertion::CustomAssertion(CustomAssertion {
        label: "c2pa.ml.model".to_string(),
        data: serde_json::to_value(ml_assertion).unwrap(),
    })
}
