# Atlas C2PA Library

![GitHub License](https://img.shields.io/github/license/IntelLabs/atlas-c2pa-lib)
[![Crates.io](https://img.shields.io/crates/v/atlas-c2pa-lib.svg)](https://crates.io/crates/atlas-c2pa-lib)
[![Documentation](https://docs.rs/atlas-c2pa-lib/badge.svg)](https://docs.rs/atlas-c2pa-lib)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/IntelLabs/atlas-c2pa-lib/badge)](https://scorecard.dev/viewer/?uri=github.com/IntelLabs/atlas-c2pa-lib)

**Atlas C2PA Library** is a Rust implementation for integrating [C2PA](https://c2pa.org/) (Coalition for Content Provenance and Authenticity) standards into machine learning workflows. This library enables cryptographically verifiable ML asset provenance, helping to establish trust in the AI ecosystem.

## 🌟 Features

- **Complete C2PA Compliance**: Create, sign, and verify manifests according to C2PA specifications
- **ML-Specific Extensions**: Specialized types for models, datasets, and training processes
- **Framework Support**: Works with TensorFlow, PyTorch, ONNX, and other major ML frameworks
- **Secure Provenance**: Cryptographic verification of asset origin and lineage
- **Transparent Lineage**: Track datasets and components used to create models
- **Governance Controls**: DoNotTrain assertions to control asset usage permissions

## 📦 Installation

Add Atlas C2PA Library to your `Cargo.toml`:

```toml
[dependencies]
atlas-c2pa-lib = "0.1"
```

## 🚀 Quick Start

### Creating a Model Manifest

```rust
use atlas_c2pa_lib::ml::types::{ModelInfo, MLFramework, ModelFormat};
use atlas_c2pa_lib::ml::manifest::MLManifestBuilder;

// Define model information
let model_info = ModelInfo {
    name: "bert-base".to_string(),
    version: "1.0.0".to_string(),
    framework: MLFramework::PyTorch,
    format: ModelFormat::TorchScript,
};

// Create and build a manifest
let manifest = MLManifestBuilder::new(
    model_info,
    "0123456789abcdef0123456789abcdef".to_string(), // Model hash
)
.with_title("BERT Language Model".to_string())
.build()
.unwrap();
```

### Adding DoNotTrain Assertions

```rust
use atlas_c2pa_lib::assertion::{Assertion, DoNotTrainAssertion};

// Create a DoNotTrain assertion
let do_not_train = DoNotTrainAssertion {
    reason: "This dataset contains licensed content".to_string(),
    enforced: true,
};

// Convert to assertion type
let assertion = Assertion::DoNotTrain(do_not_train);
```

### Including Dataset Ingredients

```rust
use atlas_c2pa_lib::ml::ingredient::create_dataset_ingredient;
use atlas_c2pa_lib::ml::manifest::MLManifestBuilder;
use atlas_c2pa_lib::ml::types::{ModelInfo, MLFramework, ModelFormat};

// Create model info
let model_info = ModelInfo {
    name: "text-classifier".to_string(),
    version: "2.0".to_string(),
    framework: MLFramework::TensorFlow,
    format: ModelFormat::SavedModel,
};

// Create dataset ingredient
let dataset = create_dataset_ingredient(
    "Wikipedia-EN-2023",
    "https://example.com/datasets/wiki2023.zip",
    "0123456789abcdef0123456789abcdef", // Dataset hash
);

// Build model manifest with dataset
let manifest = MLManifestBuilder::new(
    model_info,
    "fedcba9876543210fedcba9876543210".to_string(), // Model hash
)
.with_dataset(dataset)
.build()
.unwrap();
```

## 📋 Use Cases

- **Model Cards**: Enhance model documentation with verifiable provenance information
- **Dataset Verification**: Ensure datasets come from trustworthy sources and are unmodified
- **Training Transparency**: Document training parameters, metrics, and dataset usage
- **Supply Chain Security**: Verify the origin and integrity of model components

## 📚 Documentation

For comprehensive documentation, visit [docs.rs/atlas-c2pa-lib](https://docs.rs/atlas-c2pa-lib).

## 🧪 Examples

The repository includes several examples demonstrating common usage patterns:

- Creating manifests for models and datasets
- Adding assertions about model capabilities and limitations
- Signing and validating manifests
- Linking related assets with cross-references

## 🔧 Requirements

- Rust 1.70.0 or higher

## 👥 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the [Apache License, Version 2.0](LICENSE).

## 🔗 Related Projects

- [C2PA Specification](https://c2pa.org/specifications/specifications/2.1/index.html) - The official C2PA specification
- [Content Authenticity Initiative](https://contentauthenticity.org/) - Industry initiative for content provenance
- [Atlas CLI](https://github.com/IntelLabs/atlas-cli) - Related tools for responsible AI
## Disclaimer

This code in this repo is not stable yet, and should not be used in production environments.

---

*Developed by Intel Labs to advance transparency and trust in AI systems.*