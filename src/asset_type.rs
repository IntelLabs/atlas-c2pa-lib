//! # Asset Type Module
//!
//! This module defines the various types of ML assets that can be represented in a C2PA manifest.
//! It provides enumerations for datasets, models, and formats across different ML frameworks.
//!
//! ## Asset Types
//!
//! The module includes:
//!
//! - **Dataset types**: Different formats of datasets (JAX, Keras, PyTorch, etc.)
//! - **Model types**: Different model frameworks (TensorFlow, ONNX, PyTorch, etc.)
//! - **Format types**: Different data formats (NumPy, Protobuf, Pickle)
//! - **Generator types**: For generative models, prompts, and seeds
//! - **ML task types**: Classifier, Regressor, Cluster
//!
//! ## Example: Specifying an Asset Type
//!
//! ```rust
//! use atlas_c2pa_lib::asset_type::{AssetType, AssetTypeMap};
//!
//! // Create an asset type for a PyTorch model
//! let asset_type = AssetType::ModelPytorch;
//!
//! // Create an asset type map with version information
//! let asset_type_map = AssetTypeMap {
//!     asset_type: AssetType::ModelPytorch,
//!     version: Some("1.13.1".to_string()),
//! };
//!
//! // Check if two asset type maps are equivalent
//! let another_map = AssetTypeMap {
//!     asset_type: AssetType::ModelPytorch,
//!     version: Some("1.13.1".to_string()),
//! };
//!
//! assert!(asset_type_map == another_map);
//! ```
//!
//! ## Usage in Ingredients
//!
//! Asset types are typically used in ingredient data to specify the format of ML assets:
//!
//! ```rust
//! use atlas_c2pa_lib::asset_type::AssetType;
//! use atlas_c2pa_lib::ingredient::IngredientData;
//!
//! let ingredient_data = IngredientData {
//!     url: "https://example.com/model.pt".to_string(),
//!     alg: "sha256".to_string(),
//!     hash: "0123456789abcdef0123456789abcdef".to_string(),
//!     data_types: vec![AssetType::ModelPytorch],
//!     linked_ingredient_url: None,
//!     linked_ingredient_hash: None,
//! };
//! ```
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum AssetType {
    #[serde(rename = "c2pa.types.dataset")]
    Dataset,
    #[serde(rename = "c2pa.types.dataset.jax")]
    DatasetJax,
    #[serde(rename = "c2pa.types.dataset.keras")]
    DatasetKeras,
    #[serde(rename = "c2pa.types.dataset.ml_net")]
    DatasetMlNet,
    #[serde(rename = "c2pa.types.dataset.mxnet")]
    DatasetMxNet,
    #[serde(rename = "c2pa.types.dataset.onnx")]
    DatasetOnnx,
    #[serde(rename = "c2pa.types.dataset.openvino")]
    DatasetOpenVino,
    #[serde(rename = "c2pa.types.dataset.pytorch")]
    DatasetPytorch,
    #[serde(rename = "c2pa.types.dataset.tensorflow")]
    DatasetTensorFlow,
    #[serde(rename = "c2pa.types.model")]
    Model,
    #[serde(rename = "c2pa.types.model.jax")]
    ModelJax,
    #[serde(rename = "c2pa.types.model.keras")]
    ModelKeras,
    #[serde(rename = "c2pa.types.model.ml_net")]
    ModelMlNet,
    #[serde(rename = "c2pa.types.model.mxnet")]
    ModelMxNet,
    #[serde(rename = "c2pa.types.model.onnx")]
    ModelOnnx,
    #[serde(rename = "c2pa.types.model.openvino")]
    ModelOpenVino,
    #[serde(rename = "c2pa.types.model.pytorch")]
    ModelPytorch,
    #[serde(rename = "c2pa.types.model.tensorflow")]
    ModelTensorFlow,
    #[serde(rename = "c2pa.types.format.numpy")]
    FormatNumpy,
    #[serde(rename = "c2pa.types.format.protobuf")]
    FormatProtobuf,
    #[serde(rename = "c2pa.types.format.pickle")]
    FormatPickle,
    #[serde(rename = "c2pa.types.generator")]
    Generator,
    #[serde(rename = "c2pa.types.generator.prompt")]
    GeneratorPrompt,
    #[serde(rename = "c2pa.types.generator.seed")]
    GeneratorSeed,
    #[serde(rename = "c2pa.types.classifier")]
    Classifier,
    #[serde(rename = "c2pa.types.cluster")]
    Cluster,
    #[serde(rename = "c2pa.types.regressor")]
    Regressor,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AssetTypeMap {
    #[serde(rename = "type")]
    pub asset_type: AssetType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

impl PartialEq for AssetTypeMap {
    fn eq(&self, other: &Self) -> bool {
        self.asset_type == other.asset_type && self.version == other.version
    }
}
