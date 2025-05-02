//! # COSE Module
//!
//! This module provides functions for cryptographic operations on C2PA claims using the COSE
//! (CBOR Object Signing and Encryption) standard. It enables signing and verification of claims
//! to ensure their authenticity and integrity.
//!
//! ## Functionality
//!
//! The module provides two main functions:
//!
//! - **sign_claim**: Signs a CBOR-encoded claim using a private key
//! - **verify_signed_claim**: Verifies a signed claim using a public key
//!
//! ## Example: Signing a Claim
//!
//! ```rust,ignore
//! use atlas_c2pa_lib::cbor::encode_claim_to_cbor;
//! use atlas_c2pa_lib::cose::sign_claim;
//! use openssl::pkey::PKey;
//!
//! // Encode claim to CBOR
//! let cbor_data = encode_claim_to_cbor(&claim).unwrap();
//!
//! // Load a private key (example)
//! // let private_key_pem = std::fs::read("private_key.pem").unwrap();
//! // let private_key = PKey::private_key_from_pem(&private_key_pem).unwrap();
//!
//! // Sign the claim
//! // let signed_data = sign_claim(&cbor_data, &private_key).unwrap();
//! ```
//!
//! ## Example: Verifying a Signed Claim
//!
//! ```rust
//! use atlas_c2pa_lib::cose::verify_signed_claim;
//! use openssl::pkey::PKey;
//!
//! // Load a public key (example)
//! // let public_key_pem = std::fs::read("public_key.pem").unwrap();
//! // let public_key = PKey::public_key_from_pem(&public_key_pem).unwrap();
//!
//! // Verify the signed claim
//! // let verification_result = verify_signed_claim(&signed_data, &public_key);
//! // assert!(verification_result.is_ok());
//! ```
//!
//! ## Security Considerations
//!
//! - Private keys should be securely stored and never exposed
//! - Public keys should be distributed through secure channels
//! - The signature verification process ensures the claim hasn't been tampered with
//! - COSE provides a standardized way to represent signed data in CBOR format
use coset::{CborSerializable, CoseSign1, CoseSign1Builder, Header};
use openssl::pkey::PKey;
use openssl::sign::{Signer, Verifier};

pub fn sign_claim(
    claim_cbor: &[u8],
    private_key: &PKey<openssl::pkey::Private>,
) -> Result<Vec<u8>, String> {
    // Create a COSE Sign1 builder with the payload (claim_cbor)
    let sign1_builder = CoseSign1Builder::new()
        .payload(claim_cbor.to_vec()) // Set the payload to be signed
        .protected(Header::default()); // Add a protected header

    // Sign the payload using the provided private key
    let mut signer =
        Signer::new(openssl::hash::MessageDigest::sha256(), private_key).map_err(|e| {
            format!(
                "[COSE] Failed to create signer: {} (check if private key is valid)",
                e
            )
        })?;

    // Feed the payload into the signer
    signer.update(claim_cbor).map_err(|e| {
        format!(
            "[COSE] Failed to update signer with payload: {} (payload size: {} bytes)",
            e,
            claim_cbor.len()
        )
    })?;

    // Generate the signature
    let signature = signer
        .sign_to_vec()
        .map_err(|e| format!("[COSE] Failed to generate signature: {}", e))?;

    // Add the signature to the COSE Sign1 structure
    let sign1 = sign1_builder.signature(signature).build();

    // Serialize the signed COSE structure to a byte array (CBOR format)
    sign1
        .to_vec()
        .map_err(|e| format!("[COSE] Failed to serialize signed claim: {}", e))
}

pub fn verify_signed_claim(
    signed_claim: &[u8],
    _public_key: &PKey<openssl::pkey::Public>,
) -> Result<(), String> {
    // 1: Parse the COSE-encoded signed claim
    let _sign1 = CoseSign1::from_slice(signed_claim)
        .map_err(|e| format!("Failed to parse signed claim: {}", e))?;

    // S2: Get the payload (the signed data)
    let payload = _sign1
        .payload
        .as_ref()
        .ok_or("No payload found in signed claim")?;

    // 3: Extract the signature from the COSE structure (assumed to be Vec<u8> or <u16>)
    let signature: &[u8] = &_sign1.signature;

    //  4: Initialize the verifier with the public key and the payload
    let mut verifier = Verifier::new(openssl::hash::MessageDigest::sha256(), _public_key)
        .map_err(|e| format!("Failed to create verifier: {}", e))?;

    // Feed the payload (the data that was signed) into the verifier
    verifier
        .update(payload)
        .map_err(|e| format!("Failed to update verifier with payload: {}", e))?;

    // Verify the signature using the public key
    if verifier
        .verify(signature)
        .map_err(|e| format!("Verification failed: {}", e))?
    {
        Ok(()) // Signature is valid
    } else {
        Err("Invalid signature".to_string()) // Signature is invalid
    }
}
