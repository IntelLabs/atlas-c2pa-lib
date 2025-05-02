//! # CBOR Module
//!
//! This module provides functions for encoding and decoding C2PA claims using CBOR
//! (Concise Binary Object Representation) format. CBOR is a binary data serialization format
//! that is more compact and efficient than JSON, making it suitable for cryptographic operations.
//!
//! ## Functionality
//!
//! The module provides two main functions:
//!
//! - **encode_claim_to_cbor**: Converts a ClaimV2 object to CBOR binary format
//! - **decode_claim_from_cbor**: Converts CBOR binary data back to a ClaimV2 object
//!
//! ## Example: Encoding and Decoding Claims
//!
//! ```rust
//! use atlas_c2pa_lib::claim::ClaimV2;
//! use atlas_c2pa_lib::cbor::{encode_claim_to_cbor, decode_claim_from_cbor};
//! use atlas_c2pa_lib::datetime_wrapper::OffsetDateTimeWrapper;
//! use time::OffsetDateTime;
//!
//! // Create a minimal claim
//! let claim = ClaimV2 {
//!     instance_id: "xmp:iid:123456".to_string(),
//!     created_assertions: vec![],
//!     ingredients: vec![],
//!     signature: None,
//!     claim_generator_info: "atlas_c2pa_lib/0.1.4".to_string(),
//!     created_at: OffsetDateTimeWrapper(OffsetDateTime::now_utc()),
//! };
//!
//! // Encode the claim to CBOR
//! let cbor_data = encode_claim_to_cbor(&claim).unwrap();
//!
//! // Decode the CBOR data back to a claim
//! let decoded_claim = decode_claim_from_cbor(&cbor_data).unwrap();
//!
//! // Verify the round-trip worked
//! assert_eq!(claim.instance_id, decoded_claim.instance_id);
//! ```
//!
//! ## Usage in C2PA Workflow
//!
//! CBOR encoding is typically used before signing a claim, as it provides a compact binary
//! representation that can be efficiently signed and verified:
//!
//! ```rust,ignore
//! use atlas_c2pa_lib::cbor::encode_claim_to_cbor;
//! use atlas_c2pa_lib::cose::sign_claim;
//! let claim = ClaimV2 {
//!     instance_id: "xmp:iid:123456".to_string(),
//!     created_assertions: vec![],
//!     ingredients: vec![],
//!     signature: None,
//!     claim_generator_info: "atlas_c2pa_lib/0.1.4".to_string(),
//!     created_at: OffsetDateTimeWrapper(OffsetDateTime::now_utc()),
//! };
//! // Encode claim to CBOR
//! let cbor_data = encode_claim_to_cbor(&claim).unwrap();
//!
//! // Sign the CBOR-encoded claim
//! // let signed_data = sign_claim(&cbor_data, &private_key).unwrap();
//! ```
use crate::claim::ClaimV2;
use serde_cbor::{from_slice, to_vec};

/// Encodes a ClaimV2 into CBOR format.
///
/// # Arguments
///
/// * `claim` - The claim to encode.
///
/// # Returns
///
/// * `Vec<u8>` - The encoded claim as a byte array.
/// * `Err` - Returns an error if encoding fails.
pub fn encode_claim_to_cbor(claim: &ClaimV2) -> Result<Vec<u8>, serde_cbor::Error> {
    to_vec(claim)
}

/// Decodes a CBOR byte array into a ClaimV2 object.
///
/// # Arguments
///
/// * `data` - The CBOR byte array to decode.
///
/// # Returns
///
/// * `ClaimV2` - The decoded claim.
/// * `Err` - Returns an error if decoding fails.
pub fn decode_claim_from_cbor(data: &[u8]) -> Result<ClaimV2, serde_cbor::Error> {
    from_slice(data)
}
