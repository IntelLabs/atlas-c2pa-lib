//! # Claim Module
//!
//! This module defines the core structures for C2PA claims, which are the statements of provenance
//! and authenticity about an ML asset. A claim contains assertions, ingredients, signatures,
//! and other metadata necessary to verify the asset's origin and processing history.
//!
//! ## Claim Versions
//!
//! The module supports two claim structures:
//!
//! - **Claim**: The original claim structure (for backward compatibility)
//! - **ClaimV2**: The updated claim structure with improved fields and semantics
//!
//! ## Example: Creating a Claim
//!
//! ```rust
//! use atlas_c2pa_lib::claim::ClaimV2;
//! use atlas_c2pa_lib::datetime_wrapper::OffsetDateTimeWrapper;
//! use atlas_c2pa_lib::assertion::{Assertion, DoNotTrainAssertion};
//! use time::OffsetDateTime;
//!
//! // Create a DoNotTrain assertion
//! let do_not_train = DoNotTrainAssertion {
//!     reason: "This model contains proprietary information".to_string(),
//!     enforced: true,
//! };
//!
//! // Create a claim with the assertion
//! let claim = ClaimV2 {
//!     instance_id: "xmp:iid:model-123456".to_string(),
//!     created_assertions: vec![Assertion::DoNotTrain(do_not_train)],
//!     ingredients: vec![],
//!     signature: None,
//!     claim_generator_info: "atlas_c2pa_lib/0.1.4".to_string(),
//!     created_at: OffsetDateTimeWrapper(OffsetDateTime::now_utc()),
//! };
//! ```
//!
//! ## Validation
//!
//! The module provides validation functions to ensure claims are well-formed:
//!
//! ```rust,ignore
//! use atlas_c2pa_lib::claim::{validate_claim_v2, ClaimV2};
//!
//! // Validate a claim
//! let validation_result = validate_claim_v2(&claim);
//! if let Err(error) = validation_result {
//!     println!("Claim validation failed: {}", error);
//! }
//! ```
//!
//! ## Usage in C2PA Workflow
//!
//! Claims are typically part of a manifest and may be serialized, signed, and verified:
//!
//! ```rust,ignore
//! use atlas_c2pa_lib::manifest::Manifest;
//! use atlas_c2pa_lib::claim::ClaimV2;
//!
//! // Create a manifest with the claim
//! let manifest = Manifest {
//!     claim_generator: "atlas_c2pa_lib/0.1.4".to_string(),
//!     title: "ML Model Manifest".to_string(),
//!     instance_id: "xmp:iid:manifest-123456".to_string(),
//!     ingredients: vec![],
//!     claim: claim.clone(),
//!     created_at: claim.created_at.clone(),
//!     cross_references: vec![],
//!     claim_v2: Some(claim),
//!     is_active: true,
//! };
//! ```
use crate::assertion::validate_assertion;
use crate::assertion::Assertion;
use crate::ingredient::validate_ingredient;

use crate::datetime_wrapper::{validate_datetime, OffsetDateTimeWrapper};
use crate::ingredient::Ingredient;
use serde::{Deserialize, Serialize};

/// Struct to represent a Claim in the C2PA Manifest.
#[derive(Serialize, Deserialize, Debug)]
pub struct Claim {
    pub id: String,                        // Unique identifier for the claim
    pub assertions: Vec<Assertion>,        // Assertions made in the claim
    pub ingredients: Vec<Ingredient>,      // Ingredients used in the asset
    pub creator: String,                   // Who created the claim (e.g., software or user)
    pub created_at: OffsetDateTimeWrapper, // Timestamp when the claim was created
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>, // Optional digital signature for the claim
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClaimV2 {
    pub instance_id: String, // Unique identifier for this instance of the claim
    pub created_assertions: Vec<Assertion>, // Assertions created in this claim
    pub ingredients: Vec<Ingredient>,
    pub signature: Option<String>, // Optional signature for the claim
    pub claim_generator_info: String, // Information about the generator that created this claim
    pub created_at: OffsetDateTimeWrapper, // Creation timestamp for this claim
}

/// Helper function to validate a Claim.
pub fn validate_claim(claim: &Claim) -> Result<(), String> {
    // Validate that the ID is non-empty
    if claim.id.trim().is_empty() {
        return Err("Claim must have a valid ID.".to_string());
    }

    // Validate that the creator field is non-empty
    if claim.creator.trim().is_empty() {
        return Err("Claim must have a valid creator.".to_string());
    }

    // Validate that the created_at timestamp is valid (this assumes valid parsing in datetime_wrapper.rs)
    if claim.created_at.0.year() < 1970 {
        return Err("Claim must have a valid creation date.".to_string());
    }

    // Validate assertions
    if claim.assertions.is_empty() {
        return Err("Claim must contain at least one assertion.".to_string());
    }
    for assertion in &claim.assertions {
        validate_assertion(assertion)?; // Calls the validate function from assertion.rs
    }

    // Validate ingredients
    for ingredient in &claim.ingredients {
        validate_ingredient(ingredient)?; // Calls the validate function from ingredient.rs
    }

    Ok(())
}

pub fn validate_claim_v2(claim: &ClaimV2) -> Result<(), String> {
    // Validate that the instance_id is non-empty
    if claim.instance_id.trim().is_empty() {
        return Err("ClaimV2 must have a valid instance_id.".to_string());
    }

    // Validate that the claim_generator_info field is non-empty
    if claim.claim_generator_info.trim().is_empty() {
        return Err("ClaimV2 must have valid claim generator info.".to_string());
    }

    // Validate that the created_at timestamp is valid
    validate_datetime(&claim.created_at)?;

    // Validate created assertions
    if claim.created_assertions.is_empty() {
        return Err("ClaimV2 must contain at least one assertion.".to_string());
    }
    for assertion in &claim.created_assertions {
        validate_assertion(assertion)?;
    }

    Ok(())
}
