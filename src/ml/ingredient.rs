//! # ML Ingredient Module
//!
//! This module provides specialized utilities for creating ingredients specific to machine learning
//! assets such as models and datasets. It includes builder patterns to simplify creation of
//! ML-specific ingredients with appropriate asset types.
//!
//! ## Key Components
//!
//! - **MLIngredientBuilder**: A builder for ML-specific ingredients
//! - **Helper functions**: For creating model and dataset ingredients
//!
//! ## Example: Creating a Model Ingredient
//!
//! ```rust
//! use atlas_c2pa_lib::ml::types::{ModelInfo, MLFramework, ModelFormat};
//! use atlas_c2pa_lib::ml::ingredient::{MLIngredientBuilder, create_model_ingredient};
//!
//! // Create model information
//! let model_info = ModelInfo {
//!     name: "resnet50".to_string(),
//!     version: "1.0.0".to_string(),
//!     framework: MLFramework::PyTorch,
//!     format: ModelFormat::TorchScript,
//! };
//!
//! // Create a model ingredient using the builder
//! let builder = MLIngredientBuilder::new(
//!     model_info.clone(),
//!     "https://example.com/models/resnet50.pt".to_string(),
//!     "0123456789abcdef0123456789abcdef".to_string(),
//! );
//! let ingredient = builder.build();
//!
//! // Alternatively, use the helper function
//! let model_ingredient = create_model_ingredient(
//!     model_info,
//!     "https://example.com/models/resnet50.pt",
//!     "0123456789abcdef0123456789abcdef",
//! );
//! ```
//!
//! ## Example: Creating a Dataset Ingredient
//!
//! ```rust
//! use atlas_c2pa_lib::ml::ingredient::create_dataset_ingredient;
//!
//! // Create a dataset ingredient
//! let dataset_ingredient = create_dataset_ingredient(
//!     "ImageNet",
//!     "https://example.com/datasets/imagenet.zip",
//!     "0123456789abcdef0123456789abcdef",
//! );
//! ```
//!
//! ## Usage in Manifests
//!
//! These ingredients are typically used in ML-specific manifests:
//!
//! ```rust
//! use atlas_c2pa_lib::ml::manifest::MLManifestBuilder;
//! use atlas_c2pa_lib::ml::ingredient::create_dataset_ingredient;
//!
//! // Create a dataset ingredient
//! let dataset_ingredient = create_dataset_ingredient(
//!     "ImageNet",
//!     "https://example.com/datasets/imagenet.zip",
//!     "0123456789abcdef0123456789abcdef",
//! );
//!
//! // Add the ingredient to a manifest builder
//! // let builder = MLManifestBuilder::new(/* ... */)
//! //    .with_dataset(dataset_ingredient);
//! ```
use super::types::{MLFramework, ModelInfo};
use crate::asset_type::AssetType;
use crate::ingredient::{Ingredient, IngredientData};

pub struct MLIngredientBuilder {
    model_info: ModelInfo,
    url: String,
    hash: String,
}

impl MLIngredientBuilder {
    pub fn new(model_info: ModelInfo, url: String, hash: String) -> Self {
        Self {
            model_info,
            url,
            hash,
        }
    }

    pub fn build(&self) -> Ingredient {
        let data_types = vec![match self.model_info.framework {
            MLFramework::TensorFlow => AssetType::ModelTensorFlow,
            MLFramework::PyTorch => AssetType::ModelPytorch,
            MLFramework::ONNX => AssetType::ModelOnnx,
            MLFramework::OpenVINO => AssetType::ModelOpenVino,
            MLFramework::Custom(_) => AssetType::Model,
        }];

        Ingredient {
            title: self.model_info.name.clone(),
            format: "application/octet-stream".to_string(),
            relationship: "inputTo".to_string(),
            document_id: format!("model-{}", uuid::Uuid::new_v4()),
            instance_id: format!("instance-{}", uuid::Uuid::new_v4()),
            data: IngredientData {
                url: self.url.clone(),
                alg: "sha256".to_string(),
                hash: self.hash.clone(),
                data_types,
                linked_ingredient_url: None,
                linked_ingredient_hash: None,
            },
            linked_ingredient: None,
            public_key: None,
        }
    }
}

// Helper functions for creating ML ingredients
pub fn create_model_ingredient(model_info: ModelInfo, model_path: &str, hash: &str) -> Ingredient {
    MLIngredientBuilder::new(model_info, model_path.to_string(), hash.to_string()).build()
}

pub fn create_dataset_ingredient(dataset_name: &str, dataset_path: &str, hash: &str) -> Ingredient {
    Ingredient {
        title: dataset_name.to_string(),
        format: "multipart/mixed".to_string(),
        relationship: "inputTo".to_string(),
        document_id: format!("dataset-{}", uuid::Uuid::new_v4()),
        instance_id: format!("instance-{}", uuid::Uuid::new_v4()),
        data: IngredientData {
            url: dataset_path.to_string(),
            alg: "sha256".to_string(),
            hash: hash.to_string(),
            data_types: vec![AssetType::Dataset],
            linked_ingredient_url: None,
            linked_ingredient_hash: None,
        },
        linked_ingredient: None,
        public_key: None,
    }
}
