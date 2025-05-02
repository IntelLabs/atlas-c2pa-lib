//! # ML Manifest Module
//!
//! This module provides specialized utilities for creating C2PA manifests for machine learning
//! assets. It includes a builder pattern for constructing ML-specific manifests with appropriate
//! assertions, ingredients, and metadata.
//!
//! ## Key Components
//!
//! - **MLManifestBuilder**: A builder for ML-specific manifests
//! - **Helper functions**: For creating model manifests
//!
//! ## Example: Creating an ML Manifest with the Builder
//!
//! ```rust
//! use atlas_c2pa_lib::ml::types::{ModelInfo, MLFramework, ModelFormat, TrainingInfo, DatasetInfo};
//! use atlas_c2pa_lib::ml::manifest::MLManifestBuilder;
//! use atlas_c2pa_lib::ml::ingredient::create_dataset_ingredient;
//!
//! // Create model information
//! let model_info = ModelInfo {
//!     name: "bert-base".to_string(),
//!     version: "1.0.0".to_string(),
//!     framework: MLFramework::PyTorch,
//!     format: ModelFormat::TorchScript,
//! };
//!
//! // Create training information
//! let training_info = TrainingInfo {
//!     dataset_info: DatasetInfo {
//!         name: "wikitext".to_string(),
//!         version: "2".to_string(),
//!         size: 1000000,
//!         format: "text".to_string(),
//!     },
//!     hyperparameters: vec![],
//!     metrics: vec![],
//! };
//!
//! // Create a dataset ingredient
//! let dataset = create_dataset_ingredient(
//!     "wikitext",
//!     "https://example.com/datasets/wikitext.zip",
//!     "0123456789abcdef0123456789abcdef",
//! );
//!
//! // Create a manifest builder
//! let builder = MLManifestBuilder::new(
//!     model_info,
//!     "0123456789abcdef0123456789abcdef".to_string(), // Model hash
//! )
//! .with_training_info(training_info)
//! .with_dataset(dataset)
//! .with_claim_generator("MyMLTrainer/1.0".to_string())
//! .with_title("BERT Base Model Manifest".to_string());
//!
//! // Build the manifest
//! let manifest = builder.build().unwrap();
//! ```
//!
//! ## Example: Using the Helper Function
//!
//! ```rust
//! use atlas_c2pa_lib::ml::types::{ModelInfo, MLFramework, ModelFormat, TrainingInfo, DatasetInfo};
//! use atlas_c2pa_lib::ml::manifest::{create_model_manifest};
//! use atlas_c2pa_lib::manifest::Manifest;
//! use atlas_c2pa_lib::ml::ingredient::create_dataset_ingredient;
//!
//! // Create model information
//! let model_info = ModelInfo {
//!     name: "gpt2".to_string(),
//!     version: "1.0.0".to_string(),
//!     framework: MLFramework::PyTorch,
//!     format: ModelFormat::TorchScript,
//! };
//!
//! // Create training information
//! let training_info = TrainingInfo {
//!     dataset_info: DatasetInfo {
//!         name: "wikitext".to_string(),
//!         version: "2".to_string(),
//!         size: 1000000,
//!         format: "text".to_string(),
//!     },
//!     hyperparameters: vec![],
//!     metrics: vec![],
//! };
//!
//! // Create a dataset ingredient
//! let dataset = create_dataset_ingredient(
//!     "wikitext",
//!     "https://example.com/datasets/wikitext.zip",
//!     "0123456789abcdef0123456789abcdef",
//! );
//!
//! // Create a model manifest
//! let manifest = create_model_manifest(
//!     model_info,
//!     "0123456789abcdef0123456789abcdef".to_string(), // Model hash
//!     Some(training_info),
//!     vec![dataset],
//! ).unwrap();
//! ```
use super::assertions::MLModelAssertion;
use super::types::{ModelInfo, TrainingInfo};
use crate::assertion::{Assertion, CustomAssertion};
use crate::claim::ClaimV2;
use crate::datetime_wrapper::OffsetDateTimeWrapper;
use crate::ingredient::Ingredient;
use crate::manifest::Manifest;
use time::OffsetDateTime;
use uuid::Uuid;
pub struct MLManifestBuilder {
    model_info: ModelInfo,
    training_info: Option<TrainingInfo>,
    datasets: Vec<Ingredient>,
    model_hash: String,
    claim_generator: String,
    title: String,
}

impl MLManifestBuilder {
    pub fn new(model_info: ModelInfo, model_hash: String) -> Self {
        Self {
            model_info,
            training_info: None,
            datasets: Vec::new(),
            model_hash,
            claim_generator: "c2pa-ml".to_string(),
            title: "ML Model Manifest".to_string(),
        }
    }

    pub fn with_training_info(mut self, training_info: TrainingInfo) -> Self {
        self.training_info = Some(training_info);
        self
    }

    pub fn with_dataset(mut self, dataset: Ingredient) -> Self {
        self.datasets.push(dataset);
        self
    }

    pub fn with_claim_generator(mut self, claim_generator: String) -> Self {
        self.claim_generator = claim_generator;
        self
    }

    pub fn with_title(mut self, title: String) -> Self {
        self.title = title;
        self
    }

    pub fn build(&self) -> Result<Manifest, String> {
        // Create ML model assertion
        let ml_assertion = MLModelAssertion::new(
            self.model_info.clone(),
            self.training_info.clone(),
            self.model_hash.clone(),
        );

        // Create claim
        let assertions = vec![Assertion::CustomAssertion(CustomAssertion {
            label: "c2pa.ml.model".to_string(),
            data: serde_json::to_value(ml_assertion).map_err(|e| e.to_string())?,
        })];

        // Add all ingredients to the claim
        let mut ingredients = Vec::new();

        // Add datasets as ingredients
        for dataset in &self.datasets {
            ingredients.push(dataset.clone());
        }

        let claim = ClaimV2 {
            instance_id: format!("xmp:iid:ml-model-{}", Uuid::new_v4()),
            created_assertions: assertions,
            ingredients,
            signature: None,
            claim_generator_info: self.claim_generator.clone(),
            created_at: OffsetDateTimeWrapper(OffsetDateTime::now_utc()),
        };

        // Create manifest
        Ok(Manifest {
            claim_generator: self.claim_generator.clone(),
            title: self.title.clone(),
            instance_id: format!("xmp:iid:ml-manifest-{}", uuid::Uuid::new_v4()),
            ingredients: Vec::new(),
            claim: claim.clone(),
            created_at: OffsetDateTimeWrapper(OffsetDateTime::now_utc()),
            cross_references: Vec::new(),
            claim_v2: Some(claim),
            is_active: true,
        })
    }
}
pub fn create_model_manifest(
    model_info: ModelInfo,
    model_hash: String,
    training_info: Option<TrainingInfo>,
    datasets: Vec<Ingredient>,
) -> Result<Manifest, String> {
    let mut builder = MLManifestBuilder::new(model_info, model_hash);

    if let Some(training_info) = training_info {
        builder = builder.with_training_info(training_info);
    }

    for dataset in datasets {
        builder = builder.with_dataset(dataset);
    }

    builder.build()
}
