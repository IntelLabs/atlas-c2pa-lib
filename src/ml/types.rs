//! # ML Types Module
//!
//! This module defines data structures for representing machine learning-specific information
//! in C2PA manifests. It includes types for model metadata, training information, and framework details.
//!
//! ## Key Structures
//!
//! - **ModelInfo**: Metadata about an ML model
//! - **MLFramework**: Enumeration of supported ML frameworks
//! - **ModelFormat**: Enumeration of supported model formats
//! - **TrainingInfo**: Information about the training process
//! - **DatasetInfo**: Metadata about datasets used in training
//! - **Parameter**: Key-value representation of hyperparameters
//! - **Metric**: Key-value representation of evaluation metrics
//!
//! ## Example: Creating Model Information
//!
//! ```rust
//! use atlas_c2pa_lib::ml::types::{ModelInfo, MLFramework, ModelFormat};
//!
//! // Create model information
//! let model_info = ModelInfo {
//!     name: "resnet50".to_string(),
//!     version: "1.0.0".to_string(),
//!     framework: MLFramework::PyTorch,
//!     format: ModelFormat::TorchScript,
//! };
//! ```
//!
//! ## Example: Creating Training Information
//!
//! ```rust
//! use atlas_c2pa_lib::ml::types::{TrainingInfo, DatasetInfo, Parameter, Metric};
//!
//! // Create dataset information
//! let dataset_info = DatasetInfo {
//!     name: "imagenet".to_string(),
//!     version: "2012".to_string(),
//!     size: 1_281_167,
//!     format: "jpeg".to_string(),
//! };
//!
//! // Create hyperparameters
//! let hyperparameters = vec![
//!     Parameter {
//!         name: "learning_rate".to_string(),
//!         value: "0.001".to_string(),
//!     },
//!     Parameter {
//!         name: "batch_size".to_string(),
//!         value: "32".to_string(),
//!     },
//! ];
//!
//! // Create metrics
//! let metrics = vec![
//!     Metric {
//!         name: "accuracy".to_string(),
//!         value: 0.76,
//!     },
//!     Metric {
//!         name: "top_5_accuracy".to_string(),
//!         value: 0.93,
//!     },
//! ];
//!
//! // Create training information
//! let training_info = TrainingInfo {
//!     dataset_info,
//!     hyperparameters,
//!     metrics,
//! };
//! ```
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub framework: MLFramework,
    pub format: ModelFormat,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MLFramework {
    TensorFlow,
    PyTorch,
    ONNX,
    OpenVINO,
    Custom(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ModelFormat {
    SavedModel,
    ONNX,
    TorchScript,
    OpenVINO,
    Custom(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingInfo {
    pub dataset_info: DatasetInfo,
    pub hyperparameters: Vec<Parameter>,
    pub metrics: Vec<Metric>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DatasetInfo {
    pub name: String,
    pub version: String,
    pub size: usize,
    pub format: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub value: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Metric {
    pub name: String,
    pub value: f64,
}
