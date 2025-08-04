//! # Assertion Module
//!
//! This module defines the various types of assertions that can be included in a C2PA claim.
//! Assertions represent statements about the asset, its processing history, ingredients, and restrictions.
//!
//! ## Assertion Types
//!
//! - **Action**: Records operations performed on the asset
//! - **Ingredient**: Documents components used in creating the asset
//! - **Hash**: Provides cryptographic verification of asset integrity
//! - **Metadata**: Contains arbitrary metadata about the asset
//! - **CreativeWork**: Specifies authorship and creative work information
//! - **CustomAssertion**: Allows for domain-specific assertions
//! - **DoNotTrain**: Indicates the asset should not be used for training ML models
//!
//! ## Example: Creating a DoNotTrain Assertion
//!
//! ```rust
//! use atlas_c2pa_lib::assertion::{Assertion, DoNotTrainAssertion};
//!
//! // Create a DoNotTrain assertion
//! let do_not_train = DoNotTrainAssertion::new(
//!     "Contains copyrighted content".to_string(),
//!     true
//! );
//!
//! // Verify the assertion is valid
//! assert!(do_not_train.verify().is_ok());
//!
//! // Convert to the enum type
//! let assertion = Assertion::DoNotTrain(do_not_train);
//! ```
//!
//! ## Validation
//!
//! The module provides validation functions to ensure assertions are properly formed:
//!
//! ```rust
//! use atlas_c2pa_lib::assertion::{Assertion, validate_assertion, Action, ActionAssertion};
//!
//! // Create an action assertion
//! let action_assertion = ActionAssertion {
//!     actions: vec![
//!         Action {
//!             action: "c2pa.created".to_string(),
//!             software_agent: Some("Model Trainer v1.0".to_string()),
//!             parameters: None,
//!             digital_source_type: None,
//!             instance_id: None,
//!         }
//!     ]
//! };
//!
//! // Validate the assertion
//! let result = validate_assertion(&Assertion::Action(action_assertion));
//! ```
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Enum to represent different types of assertions as defined in the C2PA specification.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Assertion {
    Action(ActionAssertion),
    Ingredient(IngredientAssertion),
    Hash(HashAssertion),
    Metadata(MetadataAssertion),
    CreativeWork(CreativeWorkAssertion),
    CustomAssertion(CustomAssertion),
    DoNotTrain(DoNotTrainAssertion),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActionAssertion {
    pub actions: Vec<Action>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Action {
    pub action: String,
    pub software_agent: Option<String>,
    pub parameters: Option<Value>, // It can take any data
    pub digital_source_type: Option<String>,
    pub instance_id: Option<String>,
}

/// Struct to represent the Ingredient assertion as described in C2PA.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IngredientAssertion {
    pub title: String,
    pub relationship: String,
    #[serde(rename = "dc:format")]
    pub format: String,
    pub document_id: String,
    pub instance_id: String,
    pub data: IngredientData,
}

/// Struct for handling data related to an ingredient.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IngredientData {
    pub url: String,
    pub alg: String,
    pub hash: String,
    #[serde(rename = "data_types")]
    pub data_types: Vec<String>,
}

/// Struct to represent Hash assertions.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HashAssertion {
    pub algorithm: String,
    pub hash_value: Vec<u8>,
}

/// Struct to represent Metadata assertions.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MetadataAssertion {
    #[serde(rename = "@type")]
    pub metadata_type: String,
    pub fields: Value,
}

/// Struct to represent CreativeWork assertions.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreativeWorkAssertion {
    #[serde(rename = "@context")]
    pub context: String,
    #[serde(rename = "@type")]
    pub creative_type: String,
    pub author: Vec<Author>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Author {
    #[serde(rename = "@type")]
    pub author_type: String,
    pub name: String,
}

/// Struct to represent a custom assertion, allowing for custom key-value pairs.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CustomAssertion {
    pub label: String,
    pub data: Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DoNotTrainAssertion {
    pub reason: String, // A field to describe the reason why the asset should not be used for training
    pub enforced: bool, // Whether this assertion is enforced
}

impl DoNotTrainAssertion {
    pub fn new(reason: String, enforced: bool) -> Self {
        DoNotTrainAssertion { reason, enforced }
    }

    pub fn verify(&self) -> Result<(), String> {
        if self.reason.trim().is_empty() {
            return Err("[DoNotTrainAssertion] Missing required reason field".to_string());
        }

        if !self.enforced {
            return Err(
                "[DoNotTrainAssertion] Assertion must have enforced=true to be valid".to_string(),
            );
        }

        Ok(())
    }
}

/// Comparison think if it can be simpler
impl PartialEq for Assertion {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Assertion::Hash(hash1), Assertion::Hash(hash2)) => {
                hash1.algorithm == hash2.algorithm && hash1.hash_value == hash2.hash_value
            }
            (Assertion::Ingredient(ing1), Assertion::Ingredient(ing2)) => {
                ing1.document_id == ing2.document_id && ing1.instance_id == ing2.instance_id
            }
            _ => false,
        }
    }
}

/// Helper function to generate assertion labels with optional indexing (for multiple instances of the same type).
pub fn generate_assertion_label(base_label: &str, index: Option<u32>) -> String {
    if let Some(idx) = index {
        format!("{base_label}__{idx}")
    } else {
        base_label.to_string()
    }
}

/// Function to validate an assertion (placeholder, can be extended with actual logic).
pub fn validate_assertion(assertion: &Assertion) -> Result<(), String> {
    match assertion {
        Assertion::Action(action) => validate_action_assertion(action),
        Assertion::Ingredient(ingredient) => validate_ingredient_assertion(ingredient),
        Assertion::Hash(hash) => validate_hash_assertion(hash),
        Assertion::DoNotTrain(do_not_train) => do_not_train.verify(),
        Assertion::CreativeWork(creative_work) => validate_creative_work_assertion(creative_work),
        Assertion::CustomAssertion(custom) => validate_custom_assertion(custom),
        _ => Err("Unsupported assertion type".to_string()),
    }
}

fn validate_action_assertion(action_assertion: &ActionAssertion) -> Result<(), String> {
    // Check if the actions vector is empty
    if action_assertion.actions.is_empty() {
        return Err("Action assertion must contain at least one action.".to_string());
    }

    // Loop through all actions and validate each one
    for action in &action_assertion.actions {
        // Validate that the action name is present
        if action.action.trim().is_empty() {
            return Err("Each action must have a valid action name.".to_string());
        }

        // Validate the software_agent if present
        if let Some(software_agent) = &action.software_agent {
            if software_agent.trim().is_empty() {
                return Err("If provided, software_agent must be a non-empty string.".to_string());
            }
        }

        // Optionally validate the parameters field
        if let Some(parameters) = &action.parameters {
            if parameters.is_null() {
                return Err("If provided, parameters must not be null.".to_string());
            }
        }

        // Optionally validate the instance_id if present
        if let Some(instance_id) = &action.instance_id {
            if instance_id.trim().is_empty() {
                return Err("If provided, instance_id must be a non-empty string.".to_string());
            }
        }

        // Optionally validate the digital_source_type if present
        if let Some(digital_source_type) = &action.digital_source_type {
            if digital_source_type.trim().is_empty() {
                return Err(
                    "If provided, digital_source_type must be a non-empty string.".to_string(),
                );
            }
        }
    }

    Ok(())
}

fn validate_creative_work_assertion(creative_work: &CreativeWorkAssertion) -> Result<(), String> {
    // Validate that the context is non-empty
    if creative_work.context.trim().is_empty() {
        return Err("CreativeWork assertion must have a valid context.".to_string());
    }

    // Validate that the creative_type is non-empty
    if creative_work.creative_type.trim().is_empty() {
        return Err("CreativeWork assertion must have a valid type.".to_string());
    }

    // Validate that there's at least one author
    if creative_work.author.is_empty() {
        return Err("CreativeWork assertion must have at least one author.".to_string());
    }

    // Validate each author
    for author in &creative_work.author {
        if author.author_type.trim().is_empty() {
            return Err("Author must have a valid type.".to_string());
        }
        if author.name.trim().is_empty() {
            return Err("Author must have a valid name.".to_string());
        }
    }

    Ok(())
}

pub fn validate_ingredient_assertion(ingredient: &IngredientAssertion) -> Result<(), String> {
    // Validate that the title is non-empty
    if ingredient.title.trim().is_empty() {
        return Err("Ingredient assertion must have a non-empty title.".to_string());
    }

    // Validate that the relationship is non-empty
    if ingredient.relationship.trim().is_empty() {
        return Err("Ingredient assertion must have a non-empty relationship.".to_string());
    }

    // Validate that the format is non-empty
    if ingredient.format.trim().is_empty() {
        return Err("Ingredient assertion must have a non-empty format.".to_string());
    }

    // Validate that the document_id is non-empty
    if ingredient.document_id.trim().is_empty() {
        return Err("Ingredient assertion must have a valid document_id.".to_string());
    }

    // Validate that the instance_id is non-empty
    if ingredient.instance_id.trim().is_empty() {
        return Err("Ingredient assertion must have a valid instance_id.".to_string());
    }

    // Validate that the hash is non-empty
    if ingredient.data.hash.trim().is_empty() {
        return Err("Ingredient assertion must have a valid hash.".to_string());
    }

    // Validate the IngredientData part
    validate_ingredient_data(&ingredient.data)?;

    Ok(())
}

fn validate_ingredient_data(data: &IngredientData) -> Result<(), String> {
    // Validate that the URL is non-empty
    if data.url.trim().is_empty() {
        return Err("IngredientData must have a valid URL.".to_string());
    }

    // Validate that the algorithm is specified
    if data.alg.trim().is_empty() {
        return Err("IngredientData must specify a valid hashing algorithm.".to_string());
    }

    // Validate that the hash is non-empty
    if data.hash.trim().is_empty() {
        return Err("IngredientData must have a valid hash.".to_string());
    }

    // Validate that the data_types array is not empty
    if data.data_types.is_empty() {
        return Err("IngredientData must contain at least one data type.".to_string());
    }

    Ok(())
}

fn validate_hash_assertion(hash_assertion: &HashAssertion) -> Result<(), String> {
    // Validate that the algorithm is specified and non-empty
    if hash_assertion.algorithm.trim().is_empty() {
        return Err("[HashAssertion] Must specify a valid hashing algorithm".to_string());
    }

    // Validate that the hash algorithm is one of the supported ones
    match hash_assertion.algorithm.as_str() {
        "sha256" | "sha384" | "sha512" => {} // Supported algorithms per C2PA spec
        _ => {
            return Err(format!(
                "[HashAssertion] Unsupported hashing algorithm: '{}'. Expected one of: sha256, sha384, sha512",
                hash_assertion.algorithm
            ));
        }
    }

    // Validate that the hash value is non-empty
    if hash_assertion.hash_value.is_empty() {
        return Err("[HashAssertion] Hash value cannot be empty".to_string());
    }

    // Validate the length of the hash based on the algorithm
    let expected_length = match hash_assertion.algorithm.as_str() {
        "sha256" => 32, // 256 bits = 32 bytes
        "sha384" => 48, // 384 bits = 48 bytes
        "sha512" => 64, // 512 bits = 64 bytes
        _ => return Err("[HashAssertion] Unsupported algorithm length validation".to_string()),
    };

    if hash_assertion.hash_value.len() != expected_length {
        return Err(format!(
            "[HashAssertion] Hash length mismatch for algorithm '{}': expected {} bytes, got {} bytes",
            hash_assertion.algorithm,
            expected_length,
            hash_assertion.hash_value.len()
        ));
    }

    Ok(())
}

fn validate_custom_assertion(custom: &CustomAssertion) -> Result<(), String> {
    // Validate that the label is non-empty
    if custom.label.trim().is_empty() {
        return Err("Custom assertion must have a valid label.".to_string());
    }

    // Validate that the data is non-null
    if custom.data.is_null() {
        return Err("Custom assertion must have a data field.".to_string());
    }

    Ok(())
}
#[cfg(test)]
mod assertion_tests {
    use super::*;
    use serde_json::json;

    // Tests for DoNotTrainAssertion
    #[test]
    fn test_do_not_train_assertion_creation_and_validation() {
        // Valid DoNotTrainAssertion
        let valid_assertion =
            DoNotTrainAssertion::new("Contains copyrighted material".to_string(), true);
        assert!(valid_assertion.verify().is_ok());

        // Invalid - empty reason
        let invalid_empty = DoNotTrainAssertion::new("".to_string(), true);
        assert!(invalid_empty.verify().is_err());
        assert_eq!(
            invalid_empty.verify().unwrap_err(),
            "[DoNotTrainAssertion] Missing required reason field"
        );

        // Invalid - whitespace only reason
        let invalid_whitespace = DoNotTrainAssertion::new("   ".to_string(), true);
        assert!(invalid_whitespace.verify().is_err());

        // Invalid - not enforced
        let invalid_not_enforced = DoNotTrainAssertion::new("Valid reason".to_string(), false);
        assert!(invalid_not_enforced.verify().is_err());
        assert_eq!(
            invalid_not_enforced.verify().unwrap_err(),
            "[DoNotTrainAssertion] Assertion must have enforced=true to be valid"
        );
    }

    // Tests for ActionAssertion
    #[test]
    fn test_action_assertion_validation() {
        // Valid ActionAssertion
        let valid_action = ActionAssertion {
            actions: vec![Action {
                action: "c2pa.created".to_string(),
                software_agent: Some("TestAgent/1.0".to_string()),
                parameters: Some(json!({"key": "value"})),
                digital_source_type: Some("trained".to_string()),
                instance_id: Some("instance_123".to_string()),
            }],
        };
        let assertion = Assertion::Action(valid_action);
        assert!(validate_assertion(&assertion).is_ok());

        // Invalid - empty actions vector
        let empty_actions = ActionAssertion { actions: vec![] };
        let assertion = Assertion::Action(empty_actions);
        assert!(validate_assertion(&assertion).is_err());

        // Invalid - empty action name
        let invalid_action_name = ActionAssertion {
            actions: vec![Action {
                action: "".to_string(),
                software_agent: None,
                parameters: None,
                digital_source_type: None,
                instance_id: None,
            }],
        };
        let assertion = Assertion::Action(invalid_action_name);
        assert!(validate_assertion(&assertion).is_err());

        // Invalid - empty software agent
        let invalid_software_agent = ActionAssertion {
            actions: vec![Action {
                action: "c2pa.created".to_string(),
                software_agent: Some("".to_string()),
                parameters: None,
                digital_source_type: None,
                instance_id: None,
            }],
        };
        let assertion = Assertion::Action(invalid_software_agent);
        assert!(validate_assertion(&assertion).is_err());

        // Invalid - null parameters
        let invalid_null_params = ActionAssertion {
            actions: vec![Action {
                action: "c2pa.created".to_string(),
                software_agent: None,
                parameters: Some(serde_json::Value::Null),
                digital_source_type: None,
                instance_id: None,
            }],
        };
        let assertion = Assertion::Action(invalid_null_params);
        assert!(validate_assertion(&assertion).is_err());
    }

    // Tests for HashAssertion
    #[test]
    fn test_hash_assertion_validation() {
        // Valid SHA-256 hash
        let valid_sha256 = HashAssertion {
            algorithm: "sha256".to_string(),
            hash_value: vec![0u8; 32], // 32 bytes for SHA-256
        };
        let assertion = Assertion::Hash(valid_sha256);
        assert!(validate_assertion(&assertion).is_ok());

        // Valid SHA-384 hash
        let valid_sha384 = HashAssertion {
            algorithm: "sha384".to_string(),
            hash_value: vec![0u8; 48], // 48 bytes for SHA-384
        };
        let assertion = Assertion::Hash(valid_sha384);
        assert!(validate_assertion(&assertion).is_ok());

        // Valid SHA-512 hash
        let valid_sha512 = HashAssertion {
            algorithm: "sha512".to_string(),
            hash_value: vec![0u8; 64], // 64 bytes for SHA-512
        };
        let assertion = Assertion::Hash(valid_sha512);
        assert!(validate_assertion(&assertion).is_ok());

        // Invalid - empty algorithm
        let invalid_empty_alg = HashAssertion {
            algorithm: "".to_string(),
            hash_value: vec![0u8; 32],
        };
        let assertion = Assertion::Hash(invalid_empty_alg);
        assert!(validate_assertion(&assertion).is_err());

        // Invalid - unsupported algorithm
        let invalid_alg = HashAssertion {
            algorithm: "md5".to_string(),
            hash_value: vec![0u8; 16],
        };
        let assertion = Assertion::Hash(invalid_alg);
        assert!(validate_assertion(&assertion).is_err());

        // Invalid - wrong hash length for SHA-256
        let invalid_length = HashAssertion {
            algorithm: "sha256".to_string(),
            hash_value: vec![0u8; 16], // Wrong length
        };
        let assertion = Assertion::Hash(invalid_length);
        assert!(validate_assertion(&assertion).is_err());

        // Invalid - empty hash value
        let invalid_empty_hash = HashAssertion {
            algorithm: "sha256".to_string(),
            hash_value: vec![],
        };
        let assertion = Assertion::Hash(invalid_empty_hash);
        assert!(validate_assertion(&assertion).is_err());
    }

    // Tests for IngredientAssertion
    #[test]
    fn test_ingredient_assertion_validation() {
        // Valid IngredientAssertion
        let valid_ingredient = IngredientAssertion {
            title: "Training Dataset".to_string(),
            relationship: "inputTo".to_string(),
            format: "application/zip".to_string(),
            document_id: "doc_123".to_string(),
            instance_id: "instance_123".to_string(),
            data: IngredientData {
                url: "https://example.com/dataset.zip".to_string(),
                alg: "sha256".to_string(),
                hash: "abc123".to_string(),
                data_types: vec!["dataset".to_string()],
            },
        };
        let assertion = Assertion::Ingredient(valid_ingredient.clone());
        assert!(validate_assertion(&assertion).is_ok());

        // Invalid - empty title
        let mut invalid = valid_ingredient.clone();
        invalid.title = "".to_string();
        let assertion = Assertion::Ingredient(invalid);
        assert!(validate_assertion(&assertion).is_err());

        // Invalid - empty data types
        let mut invalid = valid_ingredient.clone();
        invalid.data.data_types = vec![];
        let assertion = Assertion::Ingredient(invalid);
        assert!(validate_assertion(&assertion).is_err());
    }

    // Tests for CreativeWorkAssertion
    #[test]
    fn test_creative_work_assertion_validation() {
        // Valid CreativeWorkAssertion
        let valid_creative = CreativeWorkAssertion {
            context: "http://schema.org/".to_string(),
            creative_type: "Dataset".to_string(),
            author: vec![Author {
                author_type: "Person".to_string(),
                name: "Jane Doe".to_string(),
            }],
        };
        let assertion = Assertion::CreativeWork(valid_creative.clone());
        assert!(validate_assertion(&assertion).is_ok());

        // Invalid - empty context
        let mut invalid = valid_creative.clone();
        invalid.context = "".to_string();
        let assertion = Assertion::CreativeWork(invalid);
        assert!(validate_assertion(&assertion).is_err());

        // Invalid - empty authors
        let mut invalid = valid_creative.clone();
        invalid.author = vec![];
        let assertion = Assertion::CreativeWork(invalid);
        assert!(validate_assertion(&assertion).is_err());

        // Invalid - author with empty name
        let invalid_author = CreativeWorkAssertion {
            context: "http://schema.org/".to_string(),
            creative_type: "Dataset".to_string(),
            author: vec![Author {
                author_type: "Person".to_string(),
                name: "".to_string(),
            }],
        };
        let assertion = Assertion::CreativeWork(invalid_author);
        assert!(validate_assertion(&assertion).is_err());
    }

    // Tests for CustomAssertion
    #[test]
    fn test_custom_assertion_validation() {
        // Valid CustomAssertion
        let valid_custom = CustomAssertion {
            label: "c2pa.ml.custom".to_string(),
            data: json!({"key": "value"}),
        };
        let assertion = Assertion::CustomAssertion(valid_custom);
        assert!(validate_assertion(&assertion).is_ok());

        // Invalid - empty label
        let invalid_label = CustomAssertion {
            label: "".to_string(),
            data: json!({"key": "value"}),
        };
        let assertion = Assertion::CustomAssertion(invalid_label);
        assert!(validate_assertion(&assertion).is_err());

        // Invalid - null data
        let invalid_data = CustomAssertion {
            label: "c2pa.ml.custom".to_string(),
            data: serde_json::Value::Null,
        };
        let assertion = Assertion::CustomAssertion(invalid_data);
        assert!(validate_assertion(&assertion).is_err());
    }

    // Tests for assertion label generation
    #[test]
    fn test_generate_assertion_label() {
        assert_eq!(generate_assertion_label("c2pa.action", None), "c2pa.action");
        assert_eq!(
            generate_assertion_label("c2pa.action", Some(0)),
            "c2pa.action__0"
        );
        assert_eq!(
            generate_assertion_label("c2pa.action", Some(42)),
            "c2pa.action__42"
        );
    }

    // Tests for Assertion PartialEq implementation
    #[test]
    fn test_assertion_partial_eq() {
        // Hash assertions equality
        let hash1 = Assertion::Hash(HashAssertion {
            algorithm: "sha256".to_string(),
            hash_value: vec![1, 2, 3],
        });
        let hash2 = Assertion::Hash(HashAssertion {
            algorithm: "sha256".to_string(),
            hash_value: vec![1, 2, 3],
        });
        let hash3 = Assertion::Hash(HashAssertion {
            algorithm: "sha256".to_string(),
            hash_value: vec![4, 5, 6],
        });
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);

        // Ingredient assertions equality
        let ing1 = Assertion::Ingredient(IngredientAssertion {
            title: "Dataset".to_string(),
            relationship: "inputTo".to_string(),
            format: "application/zip".to_string(),
            document_id: "doc_123".to_string(),
            instance_id: "instance_123".to_string(),
            data: IngredientData {
                url: "https://example.com/data.zip".to_string(),
                alg: "sha256".to_string(),
                hash: "abc123".to_string(),
                data_types: vec!["dataset".to_string()],
            },
        });
        let ing2 = ing1.clone();
        let mut ing3 = ing1.clone();
        if let Assertion::Ingredient(ref mut ing) = ing3 {
            ing.document_id = "doc_456".to_string();
        }
        assert_eq!(ing1, ing2);
        assert_ne!(ing1, ing3);

        // Different types are not equal
        assert_ne!(hash1, ing1);
    }

    // Test serialization/deserialization of assertions
    #[test]
    fn test_assertion_serialization() {
        let original =
            Assertion::DoNotTrain(DoNotTrainAssertion::new("Test reason".to_string(), true));

        // Serialize to JSON
        let serialized = serde_json::to_string(&original).unwrap();

        // Deserialize back
        let deserialized: Assertion = serde_json::from_str(&serialized).unwrap();

        // Verify they match
        if let (Assertion::DoNotTrain(orig), Assertion::DoNotTrain(deser)) =
            (&original, &deserialized)
        {
            assert_eq!(orig.reason, deser.reason);
            assert_eq!(orig.enforced, deser.enforced);
        } else {
            panic!("Deserialization produced wrong assertion type");
        }
    }
}
