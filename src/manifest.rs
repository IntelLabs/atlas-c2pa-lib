//! # Manifest Module
//!
//! This module defines the structure and validation for C2PA manifests, which are the
//! top-level containers for claims, ingredients, and cross-references that document
//! the provenance and authenticity of ML assets.
//!
//! ## Key Structures
//!
//! - **Manifest**: The main container for C2PA information
//! - **ClaimGeneratorInfo**: Information about the software that generated the claim
//! - **SignatureInfo**: Details about the digital signature applied to the manifest
//!
//! ## Example: Creating a Manifest
//!
//! ```rust,ignore
//! use atlas_c2pa_lib::manifest::Manifest;
//! use atlas_c2pa_lib::claim::ClaimV2;
//! use atlas_c2pa_lib::datetime_wrapper::OffsetDateTimeWrapper;
//! use time::OffsetDateTime;
//!
//! // Create a manifest (simplified example)
//! let manifest = Manifest {
//!     claim_generator: "atlas_c2pa_lib/0.1.4".to_string(),
//!     title: "ML Model Manifest".to_string(),
//!     instance_id: "xmp:iid:model-12345".to_string(),
//!     ingredients: vec![],
//!     claim: ClaimV2 {
//!         instance_id: "xmp:iid:claim-12345".to_string(),
//!         created_assertions: vec![],
//!         ingredients: vec![],
//!         signature: None,
//!         claim_generator_info: "atlas_c2pa_lib/0.1.4".to_string(),
//!         created_at: OffsetDateTimeWrapper(OffsetDateTime::now_utc()),
//!     },
//!     created_at: OffsetDateTimeWrapper(OffsetDateTime::now_utc()),
//!     cross_references: vec![],
//!     claim_v2: None,
//!     is_active: true,
//! };
//! ```
//!
//! ## Validation
//!
//! The module provides validation functions to ensure manifests are well-formed:
//!
//! ```rust,ignore
//! use atlas_c2pa_lib::manifest::validate_manifest;
//!
//! // Validate a manifest
//! let validation_result = validate_manifest(&manifest);
//! ```
//!
//! ## Manifest Management
//!
//! Manifests can be activated or deactivated:
//!
//! ```rust,ignore
//! // Activate a manifest
//! use atlas_c2pa_lib::manifest::Manifest;
//! let mut manifest = Manifest {
//!     // ... fields
//!     is_active: false,
//! };
//!
//! manifest.activate();
//! assert!(manifest.is_active());
//!
//! // Deactivate a manifest
//! manifest.deactivate();
//! assert!(!manifest.is_active());
//! ```
//!
//! ## Cross-Reference Validation
//!
//! The module provides functions to validate linked manifests via cross-references:
//!
//! ```rust
//! use atlas_c2pa_lib::manifest::validate_linked_manifest;
//! use atlas_c2pa_lib::cross_reference::CrossReference;
//! use openssl::pkey::PKey;
//!
//! // Create a cross reference
//! let cross_ref = CrossReference::new(
//!     "https://example.com/manifests/dataset123.json".to_string(),
//!     "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string()
//! );
//!
//! // Validate the linked manifest
//! // let public_key_pem = std::fs::read("public_key.pem").unwrap();
//! // let public_key = PKey::public_key_from_pem(&public_key_pem).unwrap();
//! // let validation_result = validate_linked_manifest(&cross_ref, &public_key);
//! ```
use crate::assertion::validate_assertion;
use crate::claim::{ClaimV2, validate_claim_v2};
pub use crate::cross_reference::CrossReference;
use crate::datetime_wrapper::OffsetDateTimeWrapper;
use crate::ingredient::{Ingredient, validate_ingredient};
use base64::{Engine as _, engine::general_purpose};
use openssl::pkey::PKey;
use openssl::sign::Verifier;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Struct representing the full manifest in a C2PA-compliant workflow.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Manifest {
    pub claim_generator: String, // Software or entity that generated this claim
    pub title: String,           // Title of the manifest or asset
    pub instance_id: String,     // Unique identifier for the manifest instance
    pub ingredients: Vec<Ingredient>, // List of ingredients that make up the asset
    pub claim: ClaimV2,          // Use the updated claim struct for c2pa.claim.v2
    pub created_at: OffsetDateTimeWrapper, // Timestamp indicating when the manifest was created
    pub cross_references: Vec<CrossReference>, // Cross-references to other manifests
    #[serde(rename = "c2pa.claim.v2")]
    pub claim_v2: Option<ClaimV2>, // Updated claim format for v2
    pub is_active: bool,
}

impl Manifest {
    // Method to activate a manifest
    pub fn activate(&mut self) {
        self.is_active = true;
    }

    // Method to deactivate a manifest
    pub fn deactivate(&mut self) {
        self.is_active = false;
    }

    // Method to check if a manifest is active
    pub fn is_active(&self) -> bool {
        self.is_active
    }
}

/// Function to validate the manifest.
/// Ensures that the manifest contains valid data, including the claim and ingredients.
pub fn validate_manifest(manifest: &Manifest) -> Result<(), String> {
    // Validate that the claim_generator is non-empty
    if manifest.claim_generator.trim().is_empty() {
        return Err("Manifest must have a valid claim_generator.".to_string());
    }

    // Validate that the title is non-empty
    if manifest.title.trim().is_empty() {
        return Err("Manifest must have a valid title.".to_string());
    }

    // Validate that the instance_id is non-empty
    if manifest.instance_id.trim().is_empty() {
        return Err("Manifest must have a valid instance_id.".to_string());
    }

    if let Some(ref claim_v2) = manifest.claim_v2 {
        validate_claim_v2(claim_v2)?; // Pass the reference to ClaimV2 if it exists
    } else {
        return Err("Manifest must contain a valid c2pa.claim.v2.".to_string());
    }

    // Validate all ingredients
    for ingredient in &manifest.ingredients {
        validate_ingredient(ingredient)?; // Validates each ingredient
    }

    // Validate the created_at timestamp
    validate_manifest_timestamp(&manifest.created_at)?; // Ensure the datetime is valid

    if !manifest.is_active() {
        return Err("Manifest is not active.".to_string());
    }
    for assertion in &manifest.claim.created_assertions {
        validate_assertion(assertion)?;
    }

    Ok(())
}

/// Function to validate the timestamp of the manifest, ensuring it's not in the future or past.
pub fn validate_manifest_timestamp(datetime: &OffsetDateTimeWrapper) -> Result<(), String> {
    // Validate using the existing datetime validation function
    crate::datetime_wrapper::validate_datetime(datetime)
}

pub fn validate_manifest_structure(manifest_bytes: &[u8]) -> Result<(), String> {
    // Parse the manifest JSON
    let manifest: Value = serde_json::from_slice(manifest_bytes)
        .map_err(|e| format!("Failed to parse manifest JSON: {e}"))?;

    // Ensure it has required fields (e.g., claim, instance_id, ingredients)
    if manifest.get("claim").is_none() {
        return Err("Manifest is missing the 'claim' field.".to_string());
    }

    if manifest.get("instance_id").is_none() {
        return Err("Manifest is missing the 'instance_id' field.".to_string());
    }

    Ok(())
}

pub fn validate_linked_manifest(
    cross_ref: &CrossReference,
    public_key: &PKey<openssl::pkey::Public>,
) -> Result<(), String> {
    // Validate cross_ref fields
    if cross_ref.manifest_url.trim().is_empty() {
        return Err("[LinkedManifest] Cross-reference has empty URL".to_string());
    }
    if cross_ref.manifest_hash.trim().is_empty() {
        return Err("[LinkedManifest] Cross-reference has empty hash".to_string());
    }

    // Fetch the external manifest
    let response = reqwest::blocking::get(&cross_ref.manifest_url).map_err(|e| {
        format!(
            "[LinkedManifest] Failed to fetch manifest from URL '{}': {}",
            cross_ref.manifest_url, e
        )
    })?;

    if !response.status().is_success() {
        return Err(format!(
            "[LinkedManifest] HTTP error when fetching manifest: status code {} for URL '{}'",
            response.status(),
            cross_ref.manifest_url
        ));
    }

    let manifest_bytes = response
        .bytes()
        .map_err(|e| format!("[LinkedManifest] Failed to read manifest response: {e}"))?;

    if manifest_bytes.is_empty() {
        return Err("[LinkedManifest] Received empty manifest data".to_string());
    }

    // Compute hash of the fetched manifest
    let computed_hash = hex::encode(openssl::sha::sha256(&manifest_bytes));

    // Validate the hash
    if computed_hash != cross_ref.manifest_hash {
        return Err(format!(
            "[LinkedManifest] Hash mismatch: expected '{}', computed '{}'",
            cross_ref.manifest_hash, computed_hash
        ));
    }

    // Validate the manifest structure
    validate_manifest_structure(&manifest_bytes)
        .map_err(|e| format!("[LinkedManifest] Invalid manifest structure: {e}"))?;

    // Validate the manifest signature
    let signature = extract_signature_from_manifest(&manifest_bytes)
        .map_err(|e| format!("[LinkedManifest] Failed to extract signature: {e}"))?;

    validate_manifest_signature(&manifest_bytes, &signature, public_key)
        .map_err(|e| format!("[LinkedManifest] Signature validation failed: {e}"))?;

    Ok(())
}

fn extract_signature_from_manifest(manifest_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let manifest: Manifest = serde_json::from_slice(manifest_bytes)
        .map_err(|e| format!("Failed to parse manifest JSON: {e}"))?;

    // Check if claim_v2 is present
    if let Some(claim_v2) = manifest.claim_v2 {
        if let Some(signature) = claim_v2.signature {
            let signature_bytes = general_purpose::STANDARD
                .decode(signature)
                .map_err(|e| format!("Failed to decode signature: {e}"))?;
            return Ok(signature_bytes);
        } else {
            return Err("Signature field is missing in claim_v2.".to_string());
        }
    }

    if let Some(signature) = manifest.claim.signature {
        let signature_bytes = general_purpose::STANDARD
            .decode(signature)
            .map_err(|e| format!("Failed to decode signature: {e}"))?;
        Ok(signature_bytes)
    } else {
        Err("Signature field is missing in claim.".to_string())
    }
}

pub fn validate_manifest_signature(
    manifest_bytes: &[u8],
    signature: &[u8],
    public_key: &PKey<openssl::pkey::Public>,
) -> Result<(), String> {
    // Create a verifier with the public key
    let mut verifier = Verifier::new(openssl::hash::MessageDigest::sha256(), public_key)
        .map_err(|e| format!("Failed to create verifier: {e}"))?;

    // Update the verifier with the manifest data
    verifier
        .update(manifest_bytes)
        .map_err(|e| format!("Failed to update verifier: {e}"))?;

    // Verify the signature
    if verifier
        .verify(signature)
        .map_err(|e| format!("Failed to verify signature: {e}"))?
    {
        Ok(())
    } else {
        Err("Signature verification failed.".to_string())
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct ClaimGeneratorInfo {
    pub name: String,
    pub version: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct SignatureInfo {
    pub alg: String,
    pub issuer: String,
    pub cert_serial_number: String,
    pub time: OffsetDateTimeWrapper,
}
