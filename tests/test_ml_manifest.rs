use atlas_c2pa_lib::asset_type::AssetType;
use atlas_c2pa_lib::ingredient::{Ingredient, IngredientData};
use atlas_c2pa_lib::ml::manifest::{MLManifestBuilder, create_model_manifest};
use atlas_c2pa_lib::ml::types::{DatasetInfo, MLFramework, ModelFormat, ModelInfo, TrainingInfo};

#[test]
fn test_basic_manifest_creation() {
    let model_info = ModelInfo {
        name: "bert-base".to_string(),
        version: "1.0.0".to_string(),
        framework: MLFramework::PyTorch,
        format: ModelFormat::TorchScript,
    };

    let builder = MLManifestBuilder::new(model_info.clone(), "model_hash_123".to_string());

    let manifest = builder.build().unwrap();

    assert_eq!(manifest.claim_generator, "c2pa-ml");
    assert!(manifest.is_active);
    assert!(manifest.claim_v2.is_some());

    // Check claim assertions
    let claim = manifest.claim_v2.unwrap();
    assert_eq!(claim.created_assertions.len(), 1);
    assert_eq!(claim.ingredients.len(), 0);
}

#[test]
fn test_manifest_with_dataset() {
    let model_info = ModelInfo {
        name: "resnet50".to_string(),
        version: "1.0.0".to_string(),
        framework: MLFramework::TensorFlow,
        format: ModelFormat::SavedModel,
    };

    let dataset = Ingredient {
        title: "imagenet".to_string(),
        format: "application/zip".to_string(),
        relationship: "inputTo".to_string(),
        document_id: "dataset_doc_id".to_string(),
        instance_id: "dataset_instance_id".to_string(),
        data: IngredientData {
            url: "path/to/dataset".to_string(),
            alg: "sha256".to_string(),
            hash: "dataset_hash_123".to_string(),
            data_types: vec![AssetType::Dataset],
            linked_ingredient_url: None,
            linked_ingredient_hash: None,
        },
        linked_ingredient: None,
        public_key: None,
    };

    let manifest = MLManifestBuilder::new(model_info.clone(), "model_hash_123".to_string())
        .with_dataset(dataset)
        .build()
        .unwrap();

    // Verify dataset was added as ingredient
    let claim = manifest.claim_v2.unwrap();
    assert_eq!(claim.ingredients.len(), 1);
    assert_eq!(claim.ingredients[0].title, "imagenet");
}

#[test]
fn test_complete_model_manifest() {
    let model_info = ModelInfo {
        name: "gpt2".to_string(),
        version: "1.0.0".to_string(),
        framework: MLFramework::PyTorch,
        format: ModelFormat::TorchScript,
    };

    let training_info = TrainingInfo {
        dataset_info: DatasetInfo {
            name: "wikitext".to_string(),
            version: "2".to_string(),
            size: 1000000,
            format: "text".to_string(),
        },
        hyperparameters: vec![],
        metrics: vec![],
    };

    let dataset = Ingredient {
        title: "wikitext".to_string(),
        format: "application/zip".to_string(),
        relationship: "inputTo".to_string(),
        document_id: "dataset_doc_id".to_string(),
        instance_id: "dataset_instance_id".to_string(),
        data: IngredientData {
            url: "path/to/dataset".to_string(),
            alg: "sha256".to_string(),
            hash: "dataset_hash_123".to_string(),
            data_types: vec![AssetType::Dataset],
            linked_ingredient_url: None,
            linked_ingredient_hash: None,
        },
        linked_ingredient: None,
        public_key: None,
    };

    let manifest = create_model_manifest(
        model_info,
        "model_hash_123".to_string(),
        Some(training_info),
        vec![dataset],
    )
    .unwrap();

    let claim = manifest.claim_v2.unwrap();
    assert_eq!(claim.created_assertions.len(), 1);
    assert_eq!(claim.ingredients.len(), 1);
}
