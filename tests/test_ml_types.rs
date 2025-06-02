use atlas_c2pa_lib::ml::assertions::MLModelAssertion;
use atlas_c2pa_lib::ml::types::{
    DatasetInfo, MLFramework, Metric, ModelFormat, ModelInfo, Parameter, TrainingInfo,
};

#[test]
fn test_model_info_creation() {
    let model_info = ModelInfo {
        name: "bert-base".to_string(),
        version: "1.0.0".to_string(),
        framework: MLFramework::PyTorch,
        format: ModelFormat::TorchScript,
    };

    assert_eq!(model_info.name, "bert-base");
    assert_eq!(model_info.version, "1.0.0");

    match model_info.framework {
        MLFramework::PyTorch => (),
        _ => panic!("Expected PyTorch framework"),
    }

    match model_info.format {
        ModelFormat::TorchScript => (),
        _ => panic!("Expected TorchScript format"),
    }
}

#[test]
fn test_custom_framework() {
    let model_info = ModelInfo {
        name: "custom-model".to_string(),
        version: "1.0.0".to_string(),
        framework: MLFramework::Custom("MyFramework".to_string()),
        format: ModelFormat::Custom("CustomFormat".to_string()),
    };

    match model_info.framework {
        MLFramework::Custom(name) => assert_eq!(name, "MyFramework"),
        _ => panic!("Expected custom framework"),
    }

    match model_info.format {
        ModelFormat::Custom(format) => assert_eq!(format, "CustomFormat"),
        _ => panic!("Expected custom format"),
    }
}

#[test]
fn test_training_info() {
    let training_info = TrainingInfo {
        dataset_info: DatasetInfo {
            name: "imagenet".to_string(),
            version: "2012".to_string(),
            size: 1000000,
            format: "tfrecord".to_string(),
        },
        hyperparameters: vec![
            Parameter {
                name: "learning_rate".to_string(),
                value: "0.001".to_string(),
            },
            Parameter {
                name: "batch_size".to_string(),
                value: "32".to_string(),
            },
        ],
        metrics: vec![Metric {
            name: "accuracy".to_string(),
            value: 0.95,
        }],
    };

    assert_eq!(training_info.dataset_info.name, "imagenet");
    assert_eq!(training_info.dataset_info.version, "2012");
    assert_eq!(training_info.hyperparameters.len(), 2);
    assert_eq!(training_info.metrics[0].value, 0.95);
}

#[test]
fn test_ml_model_assertion() {
    let model_info = ModelInfo {
        name: "resnet50".to_string(),
        version: "1.0.0".to_string(),
        framework: MLFramework::TensorFlow,
        format: ModelFormat::SavedModel,
    };

    let ml_assertion = MLModelAssertion::new(model_info, None, "hash123".to_string());

    assert!(ml_assertion.verify().is_ok());

    // Test invalid case
    let invalid_model_info = ModelInfo {
        name: "".to_string(), // Empty name should fail verification
        version: "1.0.0".to_string(),
        framework: MLFramework::TensorFlow,
        format: ModelFormat::SavedModel,
    };

    let invalid_assertion = MLModelAssertion::new(invalid_model_info, None, "hash123".to_string());

    assert!(invalid_assertion.verify().is_err());
}

#[test]
fn test_serialization() {
    let model_info = ModelInfo {
        name: "bert-base".to_string(),
        version: "1.0.0".to_string(),
        framework: MLFramework::PyTorch,
        format: ModelFormat::TorchScript,
    };

    let serialized = serde_json::to_string(&model_info).unwrap();
    let deserialized: ModelInfo = serde_json::from_str(&serialized).unwrap();

    assert_eq!(model_info.name, deserialized.name);
    assert_eq!(model_info.version, deserialized.version);

    match deserialized.framework {
        MLFramework::PyTorch => (),
        _ => panic!("Framework serialization failed"),
    }

    match deserialized.format {
        ModelFormat::TorchScript => (),
        _ => panic!("Format serialization failed"),
    }
}

#[test]
fn test_training_metrics() {
    let metrics = [Metric {
            name: "accuracy".to_string(),
            value: 0.95,
        },
        Metric {
            name: "loss".to_string(),
            value: 0.23,
        },
        Metric {
            name: "f1_score".to_string(),
            value: 0.89,
        }];

    assert_eq!(metrics.len(), 3);
    assert!(metrics[0].value >= 0.0 && metrics[0].value <= 1.0);
    assert!(metrics[1].value >= 0.0);
    assert!(metrics[2].value >= 0.0 && metrics[2].value <= 1.0);
}
