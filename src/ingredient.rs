//! # Ingredient Module
//!
//! This module defines structures and functions for representing and validating ingredients
//! in a C2PA manifest. Ingredients are the components that were used to create an ML asset,
//! such as training datasets, pre-trained models, or other resources.
//!
//! ## Key Structures
//!
//! - **Ingredient**: Represents a single component used in creating an asset
//! - **IngredientData**: Contains additional information about the ingredient, including
//!   its URL, hash, and asset types
//! - **LinkedIngredient**: Represents a connection to another ingredient
//!
//! ## Example: Creating an Ingredient
//!
//! ```rust
//! use atlas_c2pa_lib::ingredient::{Ingredient, IngredientData};
//! use atlas_c2pa_lib::asset_type::AssetType;
//!
//! // Create ingredient data
//! let ingredient_data = IngredientData {
//!     url: "https://example.com/datasets/mnist.zip".to_string(),
//!     alg: "sha256".to_string(),
//!     hash: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
//!     data_types: vec![AssetType::Dataset],
//!     linked_ingredient_url: None,
//!     linked_ingredient_hash: None,
//! };
//!
//! // Create an ingredient
//! let ingredient = Ingredient {
//!     title: "MNIST Dataset".to_string(),
//!     relationship: "inputTo".to_string(),
//!     format: "application/zip".to_string(),
//!     document_id: "uuid:dataset-12345".to_string(),
//!     instance_id: "uuid:dataset-instance-12345".to_string(),
//!     data: ingredient_data,
//!     linked_ingredient: None,
//!     public_key: None,
//! };
//! ```
//!
//! ## Validation
//!
//! The module provides validation functions to ensure ingredients are well-formed:
//!
//! ```rust,ignore
//! use atlas_c2pa_lib::ingredient::validate_ingredient;
//!
//! // Validate an ingredient
//! let validation_result = validate_ingredient(&ingredient);
//! assert!(validation_result.is_ok());
//! ```
//!
//! ## Verification
//!
//! Ingredients can be verified with a public key to ensure their authenticity:
//!
//! ```rust
//! use atlas_c2pa_lib::ingredient::Verifiable;
//! use openssl::pkey::PKey;
//!
//! // Verify an ingredient with a public key
//! // let public_key_pem = std::fs::read("public_key.pem").unwrap();
//! // let public_key = PKey::public_key_from_pem(&public_key_pem).unwrap();
//! // let verification_result = ingredient.verify(Some(&public_key));
//! // assert!(verification_result.is_ok());
//! ```
//!
//! ## Usage in Claims
//!
//! Ingredients are typically included in a claim to document asset lineage:
//!
//! ```rust
//! use atlas_c2pa_lib::claim::ClaimV2;
//!
//! // Include ingredient in a claim
//! // let claim = ClaimV2 {
//! //     ingredients: vec![ingredient],
//! //     // ... other fields
//! // };
//! ```
use crate::asset_type::AssetType;
use crate::manifest::Manifest;
use openssl::pkey::{PKey, Public};
use openssl::sign::Verifier;
use serde::{Deserialize, Serialize};

/// Struct representing an individual ingredient in a C2PA manifest.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Ingredient {
    pub title: String,        // Title of the ingredient
    pub relationship: String, // Relationship to the final asset (e.g., "derivedFrom")
    #[serde(rename = "dc:format")]
    pub format: String, // MIME type of the ingredient (e.g., "image/jpeg")
    pub document_id: String,  // Unique identifier for the document
    pub instance_id: String,  // Instance ID for this particular version of the ingredient
    pub data: IngredientData, // Additional data like URL, hash, and asset type
    pub linked_ingredient: Option<LinkedIngredient>,
    #[serde(skip)]
    pub public_key: Option<PKey<Public>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LinkedIngredient {
    pub url: String,
    pub hash: String,
    pub media_type: String,
}

/// Struct representing additional data for an ingredient.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IngredientData {
    pub url: String,  // URL where the ingredient can be accessed or validated
    pub alg: String,  // Hash algorithm used (e.g., "sha256")
    pub hash: String, // Cryptographic hash of the ingredient
    pub data_types: Vec<AssetType>, // Types of data the ingredient represents (e.g., "model.onnx")
    pub linked_ingredient_url: Option<String>, // URL of the linked ingredient
    pub linked_ingredient_hash: Option<String>, // Hash of the linked ingredient
}

impl Ingredient {
    // Updated verification method to accept the public key
    pub fn verify(&self, public_key: Option<&PKey<Public>>) -> Result<(), String> {
        if let Some(key) = public_key {
            // Perform verification with the provided public key (dummy implementation)
            let mut verifier = Verifier::new(openssl::hash::MessageDigest::sha256(), key)
                .map_err(|e| format!("Failed to create verifier: {}", e))?;

            // This is a placeholder for the actual data that would be verified
            let data_to_verify = b"example data";

            verifier
                .update(data_to_verify)
                .map_err(|e| format!("Failed to update verifier: {}", e))?;
            if verifier
                .verify(&[])
                .map_err(|e| format!("Failed to verify: {}", e))?
            {
                Ok(())
            } else {
                Err("Signature verification failed.".to_string())
            }
        } else {
            Err("No public key provided.".to_string())
        }
    }
}

pub trait Verifiable {
    fn verify(&self, public_key: Option<&PKey<Public>>) -> Result<(), String>;
}

impl Verifiable for Ingredient {
    fn verify(&self, public_key: Option<&PKey<Public>>) -> Result<(), String> {
        if self.title.trim().is_empty() {
            return Err("Ingredient must have a non-empty title.".to_string());
        }
        if self.relationship.trim().is_empty() {
            return Err("Ingredient must have a non-empty relationship.".to_string());
        }
        if self.format.trim().is_empty() {
            return Err("Ingredient must have a non-empty format.".to_string());
        }
        if self.document_id.trim().is_empty() {
            return Err("Ingredient must have a valid document_id.".to_string());
        }
        if self.instance_id.trim().is_empty() {
            return Err("Ingredient must have a valid instance_id.".to_string());
        }
        validate_ingredient_data(&self.data)?;
        if let Some(key) = public_key {
            // Perform signature verification using the public key
            let mut verifier = Verifier::new(openssl::hash::MessageDigest::sha256(), key)
                .map_err(|e| format!("Failed to create verifier: {}", e))?;

            let data_to_verify = b"example data"; // Replace with actual data to verify
            verifier
                .update(data_to_verify)
                .map_err(|e| format!("Failed to update verifier: {}", e))?;

            if verifier
                .verify(&[])
                .map_err(|e| format!("Failed to verify: {}", e))?
            {
                Ok(())
            } else {
                Err("Signature verification failed.".to_string())
            }
        } else {
            Err("No public key provided.".to_string())
        }
    }
}

impl Verifiable for IngredientData {
    fn verify(&self, _public_key: Option<&PKey<Public>>) -> Result<(), String> {
        // Validate the URL
        if self.url.trim().is_empty() {
            return Err("Ingredient data must have a non-empty URL.".to_string());
        }

        // Validate the algorithm
        if self.alg.trim().is_empty() {
            return Err("Ingredient data must specify a hash algorithm.".to_string());
        }

        // Validate the hash
        if self.hash.trim().is_empty() {
            return Err("Ingredient data must have a valid hash.".to_string());
        }

        // Validate data types
        if self.data_types.is_empty() {
            return Err("Ingredient data must specify at least one data type.".to_string());
        }

        Ok(())
    }
}

impl Verifiable for Manifest {
    fn verify(&self, public_key: Option<&PKey<Public>>) -> Result<(), String> {
        // Verify each ingredient
        for ingredient in &self.ingredients {
            ingredient.verify(public_key)?;
        }

        Ok(())
    }
}

/// Function to validate an ingredient.
pub fn validate_ingredient(ingredient: &Ingredient) -> Result<(), String> {
    // Validate that the title is non-empty
    if ingredient.title.trim().is_empty() {
        return Err("Ingredient must have a non-empty title.".to_string());
    }

    // Validate that the relationship is non-empty
    if ingredient.relationship.trim().is_empty() {
        return Err("Ingredient must have a non-empty relationship.".to_string());
    }

    // Validate that the format is non-empty
    if ingredient.format.trim().is_empty() {
        return Err("Ingredient must have a non-empty format.".to_string());
    }

    // Validate that the document ID is non-empty
    if ingredient.document_id.trim().is_empty() {
        return Err("Ingredient must have a valid document_id.".to_string());
    }

    // Validate that the instance ID is non-empty
    if ingredient.instance_id.trim().is_empty() {
        return Err("Ingredient must have a valid instance_id.".to_string());
    }

    // Validate the IngredientData
    validate_ingredient_data(&ingredient.data)?;

    Ok(())
}

/// Function to validate the data within an ingredient.
/// Checks that the URL, algorithm, hash, and data types are valid.
pub fn validate_ingredient_data(data: &IngredientData) -> Result<(), String> {
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

    // Validate that the data types array is not empty
    if data.data_types.is_empty() {
        return Err("IngredientData must contain at least one data type.".to_string());
    }

    Ok(())
}

impl PartialEq for Ingredient {
    fn eq(&self, other: &Self) -> bool {
        self.title == other.title
            && self.format == other.format
            && self.relationship == other.relationship
            && self.document_id == other.document_id
            && self.instance_id == other.instance_id
            && self.data == other.data
    }
}

impl PartialEq for IngredientData {
    fn eq(&self, other: &Self) -> bool {
        self.url == other.url
            && self.alg == other.alg
            && self.hash == other.hash
            && self.data_types == other.data_types
    }
}
