//! # Cross Reference Module
//!
//! This module defines structures for linking between different C2PA manifests. Cross-references
//! enable connections between related assets, such as different versions of a model or datasets
//! used in training, while maintaining cryptographic verification of these relationships.
//!
//! ## Structure
//!
//! The `CrossReference` structure contains:
//!
//! - **manifest_url**: URL or URN identifying the referenced manifest
//! - **manifest_hash**: SHA-256 hash of the referenced manifest (hex format)
//! - **media_type**: Optional media type of the referenced manifest
//!
//! ## Example: Creating a Cross Reference
//!
//! ```rust
//! use atlas_c2pa_lib::cross_reference::CrossReference;
//!
//! // Create a simple cross reference
//! let cross_ref = CrossReference::new(
//!     "https://example.com/manifests/dataset123.json".to_string(),
//!     "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string()
//! );
//!
//! // Create a cross reference with media type
//! let cross_ref_with_type = CrossReference::new_with_media_type(
//!     "https://example.com/manifests/model456.json".to_string(),
//!     "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
//!     "application/json".to_string()
//! );
//!
//! // Validate the hash format
//! assert!(cross_ref.validate_hash().is_ok());
//! ```
//!
//! ## Usage in Manifests
//!
//! Cross references are typically included in a manifest to link related assets:
//!
//! ```rust
//! use atlas_c2pa_lib::manifest::Manifest;
//! use atlas_c2pa_lib::cross_reference::CrossReference;
//!
//! // Create a cross reference
//! let cross_ref = CrossReference::new(
//!     "https://example.com/manifests/dataset123.json".to_string(),
//!     "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string()
//! );
//!
//! // Include the cross reference in a manifest
//! // manifest.cross_references.push(cross_ref);
//! ```
//!
//! ## Validation
//!
//! The module provides validation for hash formats to ensure they meet requirements:
//!
//! ```rust
//! use atlas_c2pa_lib::cross_reference::CrossReference;
//!
//! // Create a cross reference with an invalid hash
//! let invalid_cross_ref = CrossReference::new(
//!     "https://example.com/manifests/dataset123.json".to_string(),
//!     "invalid-hash".to_string()
//! );
//!
//! // Validation should fail
//! assert!(invalid_cross_ref.validate_hash().is_err());
//! ```
use serde::{Deserialize, Serialize};

/// CrossReference structure for linking manifests
/// Each CrossReference contains a URL to another manifest and its hash
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CrossReference {
    /// URL or URN identifying the referenced manifest
    pub manifest_url: String,

    /// SHA-256 hash of the referenced manifest in hex format
    pub manifest_hash: String,

    /// Optional media type of the referenced manifest
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

impl CrossReference {
    /// Creates a new CrossReference
    pub fn new(manifest_url: String, manifest_hash: String) -> Self {
        CrossReference {
            manifest_url,
            manifest_hash,
            media_type: None,
        }
    }

    /// Creates a new CrossReference with a specified media type
    pub fn new_with_media_type(
        manifest_url: String,
        manifest_hash: String,
        media_type: String,
    ) -> Self {
        CrossReference {
            manifest_url,
            manifest_hash,
            media_type: Some(media_type),
        }
    }

    /// Validates the hash format (must be a valid hex string)
    pub fn validate_hash(&self) -> Result<(), String> {
        // Check if the hash is a valid hex string
        if !self.manifest_hash.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err("Invalid hash format: must be a hex string".to_string());
        }

        // Check if the hash has the expected length for SHA-256 (64 hex chars)
        if self.manifest_hash.len() != 64 {
            return Err(format!(
                "Invalid hash length: expected 64 characters, got {}",
                self.manifest_hash.len()
            ));
        }

        Ok(())
    }
}
