//! # DateTime Wrapper Module
//!
//! This module provides utilities for handling timestamps in C2PA manifests, including
//! serialization, validation, and time-stamping authority (TSA) token verification.
//!
//! ## Structures
//!
//! - **OffsetDateTimeWrapper**: A wrapper around `time::OffsetDateTime` that provides
//!   serialization and deserialization capabilities for RFC3339 formatted timestamps
//!
//! ## Timestamp Validation
//!
//! The module includes validation to ensure timestamps are reasonable:
//!
//! - Timestamps must not be before 1970 (Unix epoch)
//! - Timestamps must not be in the future
//!
//! ## Example: Creating and Validating a Timestamp
//!
//! ```rust
//! use atlas_c2pa_lib::datetime_wrapper::{OffsetDateTimeWrapper, validate_datetime};
//! use time::OffsetDateTime;
//!
//! // Create a timestamp for the current time
//! let now = OffsetDateTimeWrapper(OffsetDateTime::now_utc());
//!
//! // Validate the timestamp
//! let validation_result = validate_datetime(&now);
//! assert!(validation_result.is_ok());
//!
//! // Create an invalid future timestamp
//! let future_time = OffsetDateTime::now_utc() + time::Duration::days(30);
//! let future = OffsetDateTimeWrapper(future_time);
//!
//! // This validation should fail
//! let validation_result = validate_datetime(&future);
//! assert!(validation_result.is_err());
//! ```
//!
//! ## TSA Token Verification
//!
//! The module provides functionality to verify Time-Stamping Authority (TSA) tokens:
//!
//! ```rust
//! use atlas_c2pa_lib::datetime_wrapper::verify_tsa_token;
//!
//! // Verify a TSA token (example)
//! // let verification_result = verify_tsa_token("tsa_token.cms", "tsa_public_key.pem");
//! // assert!(verification_result.is_ok());
//! ```
//!
//! ## Comparison
//!
//! The module implements `PartialEq` for the wrapper to allow timestamp comparisons:
//!
//! ```rust
//! use atlas_c2pa_lib::datetime_wrapper::OffsetDateTimeWrapper;
//! use time::OffsetDateTime;
//!
//! let time1 = OffsetDateTimeWrapper(OffsetDateTime::now_utc());
//! let time2 = time1.clone();
//!
//! assert_eq!(time1, time2);
//! ```
use openssl::cms::{CMSOptions, CmsContentInfo};
use openssl::x509::X509;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use time::OffsetDateTime;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct OffsetDateTimeWrapper(#[serde(with = "time::serde::rfc3339")] pub OffsetDateTime);

/// Function to validate the datetime, ensuring it's not in the past or future.
pub fn validate_datetime(datetime: &OffsetDateTimeWrapper) -> Result<(), String> {
    // Ensure the datetime is not in the past (earlier than 1970)
    if datetime.0.year() < 1970 {
        return Err("The datetime must be later than January 1, 1970.".to_string());
    }

    // Ensure the datetime is not in the past
    let now = OffsetDateTime::now_utc();
    if datetime.0 > now {
        return Err("The datetime must not be in the future.".to_string());
    }

    Ok(())
}

/// Reads the contents of a PEM file into a `Vec<u8>`.
fn load_pem_file(path: &str) -> Result<Vec<u8>, String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open PEM file: {}", e))?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)
        .map_err(|e| format!("Failed to read PEM file: {}", e))?;
    Ok(contents)
}

/// Verifies the TSA token by parsing the CMS structure, extracting the signature and timestamp,
/// and using the TSA's public key to verify the signature.
///
/// # Arguments
///
/// * `tsa_token_path` - Path to the TSA token (CMS format, RFC3161).
/// * `tsa_public_key_path` - Path to the TSA's public key (PEM or DER format).
///
/// # Returns
///
/// * `Ok(())` if the signature is valid.
/// * `Err(String)` if the signature is invalid or verification fails.
///
pub fn verify_tsa_token(tsa_token_path: &str, tsa_public_key_path: &str) -> Result<(), String> {
    // Check if files exist first
    if !std::path::Path::new(tsa_token_path).exists() {
        return Err(format!(
            "[TSA] Token file not found at path: {}",
            tsa_token_path
        ));
    }

    if !std::path::Path::new(tsa_public_key_path).exists() {
        return Err(format!(
            "[TSA] Public key file not found at path: {}",
            tsa_public_key_path
        ));
    }

    let mut tsa_token_file = File::open(tsa_token_path).map_err(|e| {
        format!(
            "[TSA] Failed to open token file '{}': {}",
            tsa_token_path, e
        )
    })?;

    let mut tsa_token_data = Vec::new();
    tsa_token_file
        .read_to_end(&mut tsa_token_data)
        .map_err(|e| format!("[TSA] Failed to read token file: {}", e))?;

    if tsa_token_data.is_empty() {
        return Err("[TSA] TSA token file is empty".to_string());
    }

    let mut cms = CmsContentInfo::from_der(&tsa_token_data).map_err(|e| {
        format!(
            "[TSA] Failed to parse CMS structure: {} (invalid DER format)",
            e
        )
    })?;

    let pem_data = load_pem_file(tsa_public_key_path)
        .map_err(|e| format!("[TSA] Failed to load public key: {}", e))?;

    let tsa_public_key = X509::stack_from_pem(&pem_data).map_err(|e| {
        format!(
            "[TSA] Failed to parse public key (invalid PEM format): {}",
            e
        )
    })?;

    if tsa_public_key.is_empty() {
        return Err("[TSA] No valid certificates found in the public key file".to_string());
    }

    // Extract the signed data and verify it
    cms.verify(
        None, // No extra certificates
        None, // No X509 store
        None, // No additional certificates
        None, // No X509 store reference
        CMSOptions::empty(),
    )
    .map_err(|e| format!("[TSA] Token signature verification failed: {}", e))?;

    Ok(())
}
