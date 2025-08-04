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
