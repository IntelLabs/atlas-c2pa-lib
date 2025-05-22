//! # COSE Module
//!
//! This module provides functions for cryptographic operations on C2PA claims using the COSE
//! (CBOR Object Signing and Encryption) standard. It enables signing and verification of claims
//! to ensure their authenticity and integrity, with support for multiple hash algorithms.
//!
//! ## Supported Hash Algorithms
//!
//! The module supports the following hash algorithms as per C2PA specification:
//! - SHA-256 (`sha256`)
//! - SHA-384 (`sha384`)
//! - SHA-512 (`sha512`)
//!
//! ## Functionality
//!
//! The module provides the following functions:
//!
//! - **sign_claim**: Signs a CBOR-encoded claim using SHA-384 (default)
//! - **sign_claim_with_algorithm**: Signs a CBOR-encoded claim using a specified hash algorithm
//! - **verify_signed_claim**: Verifies a signed claim using SHA-384 (default)
//! - **verify_signed_claim_with_algorithm**: Verifies a signed claim using a specified hash algorithm
//!
//! ## Example: Signing a Claim with Default SHA-384
//!
//! ```rust,ignore
//! use atlas_c2pa_lib::cbor::encode_claim_to_cbor;
//! use atlas_c2pa_lib::cose::sign_claim;
//! use openssl::pkey::PKey;
//!
//! // Create and encode a claim to CBOR
//! let claim = create_claim(); // Your claim creation logic
//! let cbor_data = encode_claim_to_cbor(&claim).unwrap();
//!
//! // Load a private key (key generation is the responsibility of the end user)
//! let private_key_pem = std::fs::read("private_key.pem").unwrap();
//! let private_key = PKey::private_key_from_pem(&private_key_pem).unwrap();
//!
//! // Sign the claim using default SHA-384
//! let signed_data = sign_claim(&cbor_data, &private_key).unwrap();
//! ```
//!
//! ## Example: Signing with Different Hash Algorithms
//!
//! ```rust,ignore
//! use atlas_c2pa_lib::cbor::encode_claim_to_cbor;
//! use atlas_c2pa_lib::cose::{sign_claim_with_algorithm, HashAlgorithm};
//! use openssl::pkey::PKey;
//!
//! // Create and encode a claim to CBOR
//! let claim = create_claim(); // Your claim creation logic
//! let cbor_data = encode_claim_to_cbor(&claim).unwrap();
//!
//! // Load a private key (key generation is the responsibility of the end user)
//! let private_key_pem = std::fs::read("private_key.pem").unwrap();
//! let private_key = PKey::private_key_from_pem(&private_key_pem).unwrap();
//!
//! // Sign with SHA-384
//! let signed_sha384 = sign_claim_with_algorithm(
//!     &cbor_data,
//!     &private_key,
//!     HashAlgorithm::Sha384
//! ).unwrap();
//!
//! // Sign with SHA-512
//! let signed_sha512 = sign_claim_with_algorithm(
//!     &cbor_data,
//!     &private_key,
//!     HashAlgorithm::Sha512
//! ).unwrap();
//! ```
//!
//! ## Example: Verifying a Signed Claim
//!
//! ```rust,ignore
//! use atlas_c2pa_lib::cose::{verify_signed_claim, verify_signed_claim_with_algorithm, HashAlgorithm};
//! use openssl::pkey::PKey;
//!
//! // Load a public key
//! let public_key_pem = std::fs::read("public_key.pem").unwrap();
//! let public_key = PKey::public_key_from_pem(&public_key_pem).unwrap();
//!
//! // Verify a claim signed with SHA-256 (default)
//! let verification_result = verify_signed_claim(&signed_data, &public_key);
//! assert!(verification_result.is_ok());
//!
//! // Verify a claim signed with SHA-384
//! let verification_sha384 = verify_signed_claim_with_algorithm(
//!     &signed_sha384,
//!     &public_key,
//!     HashAlgorithm::Sha384
//! );
//! assert!(verification_sha384.is_ok());
//! ```
//!
//! ## Example: Complete Signing and Verification Workflow
//!
//! ```rust,ignore
//! use atlas_c2pa_lib::claim::ClaimV2;
//! use atlas_c2pa_lib::cbor::encode_claim_to_cbor;
//! use atlas_c2pa_lib::cose::{sign_claim_with_algorithm, verify_signed_claim_with_algorithm, HashAlgorithm};
//! use atlas_c2pa_lib::datetime_wrapper::OffsetDateTimeWrapper;
//! use openssl::pkey::PKey;
//! use time::OffsetDateTime;
//!
//! // Create a claim
//! let claim = ClaimV2 {
//!     instance_id: "xmp:iid:123456".to_string(),
//!     created_assertions: vec![],
//!     ingredients: vec![],
//!     signature: None,
//!     claim_generator_info: "atlas_c2pa_lib/0.1.4".to_string(),
//!     created_at: OffsetDateTimeWrapper(OffsetDateTime::now_utc()),
//! };
//!
//! // Encode to CBOR
//! let cbor_data = encode_claim_to_cbor(&claim).unwrap();
//!
//! // Load pre-generated keys (key generation is the responsibility of the end user)
//! let private_key_pem = std::fs::read("private_key.pem").unwrap();
//! let private_key = PKey::private_key_from_pem(&private_key_pem).unwrap();
//!
//! let public_key_pem = std::fs::read("public_key.pem").unwrap();
//! let public_key = PKey::public_key_from_pem(&public_key_pem).unwrap();
//!
//! // Try all supported algorithms
//! let algorithms = vec![
//!     HashAlgorithm::Sha256,
//!     HashAlgorithm::Sha384,
//!     HashAlgorithm::Sha512,
//! ];
//!
//! for algo in algorithms {
//!     // Sign the claim
//!     let signed = sign_claim_with_algorithm(&cbor_data, &private_key, algo.clone()).unwrap();
//!
//!     // Verify the signature
//!     let result = verify_signed_claim_with_algorithm(&signed, &public_key, algo);
//!     assert!(result.is_ok(), "Verification failed for {:?}", algo);
//! }
//! ```
//!
//! ## Security Considerations
//!
//! - **Key Generation**: The generation and management of cryptographic key pairs is the
//!   responsibility of the end user. This library does not provide key generation functionality.
//!   Users should follow cryptographic best practices for key generation, storage, and rotation.
//! - Private keys should be securely stored and never exposed
//! - Public keys should be distributed through secure channels
//! - The signature verification process ensures the claim hasn't been tampered with
//! - COSE provides a standardized way to represent signed data in CBOR format
//! - Choose the appropriate hash algorithm based on your security requirements:
//!   - SHA-256: Good balance of security and performance
//!   - SHA-384: Higher security with moderate performance impact
//!   - SHA-512: Highest security but larger signatures

use coset::{CborSerializable, CoseSign1, CoseSign1Builder, Header};
use openssl::hash::MessageDigest;
use openssl::pkey::PKey;
use openssl::sign::{Signer, Verifier};

/// Supported hash algorithms for COSE signing and verification
#[derive(Debug, Clone)]
pub enum HashAlgorithm {
    /// SHA-256 hash algorithm (256-bit)
    Sha256,
    /// SHA-384 hash algorithm (384-bit)
    Sha384,
    /// SHA-512 hash algorithm (512-bit)
    Sha512,
}

impl HashAlgorithm {
    /// Convert the hash algorithm to OpenSSL MessageDigest
    fn to_message_digest(&self) -> MessageDigest {
        match self {
            HashAlgorithm::Sha256 => MessageDigest::sha256(),
            HashAlgorithm::Sha384 => MessageDigest::sha384(),
            HashAlgorithm::Sha512 => MessageDigest::sha512(),
        }
    }

    /// Get the string representation of the algorithm
    pub fn as_str(&self) -> &'static str {
        match self {
            HashAlgorithm::Sha256 => "sha256",
            HashAlgorithm::Sha384 => "sha384",
            HashAlgorithm::Sha512 => "sha512",
        }
    }

    /// Parse algorithm from string representation
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "sha256" => Ok(HashAlgorithm::Sha256),
            "sha384" => Ok(HashAlgorithm::Sha384),
            "sha512" => Ok(HashAlgorithm::Sha512),
            _ => Err(format!(
                "Unsupported hash algorithm: {}. Supported algorithms are: sha256, sha384, sha512",
                s
            )),
        }
    }
}

/// Signs a CBOR-encoded claim using the specified hash algorithm.
///
/// # Arguments
///
/// * `claim_cbor` - The CBOR-encoded claim data to sign
/// * `private_key` - The private key used for signing
/// * `algorithm` - The hash algorithm to use for signing
///
/// # Returns
///
/// * `Ok(Vec<u8>)` - The signed COSE structure as a byte array
/// * `Err(String)` - Error message if signing fails
///
/// # Example
///
/// ```rust,ignore
/// use atlas_c2pa_lib::cose::{sign_claim_with_algorithm, HashAlgorithm};
/// use openssl::pkey::PKey;
///
/// let private_key = PKey::private_key_from_pem(&key_pem).unwrap();
/// let signed = sign_claim_with_algorithm(
///     &cbor_data,
///     &private_key,
///     HashAlgorithm::Sha384
/// ).unwrap();
/// ```
pub fn sign_claim_with_algorithm(
    claim_cbor: &[u8],
    private_key: &PKey<openssl::pkey::Private>,
    algorithm: HashAlgorithm,
) -> Result<Vec<u8>, String> {
    // Create a COSE Sign1 builder with the payload (claim_cbor)
    let sign1_builder = CoseSign1Builder::new()
        .payload(claim_cbor.to_vec()) // Set the payload to be signed
        .protected(Header::default()); // Add a protected header

    // Sign the payload using the provided private key and algorithm
    let mut signer = Signer::new(algorithm.to_message_digest(), private_key).map_err(|e| {
        format!(
            "[COSE] Failed to create signer with {} algorithm: {} (check if private key is valid)",
            algorithm.as_str(),
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
    let signature = signer.sign_to_vec().map_err(|e| {
        format!(
            "[COSE] Failed to generate signature with {}: {}",
            algorithm.as_str(),
            e
        )
    })?;

    // Add the signature to the COSE Sign1 structure
    let sign1 = sign1_builder.signature(signature).build();

    // Serialize the signed COSE structure to a byte array (CBOR format)
    sign1
        .to_vec()
        .map_err(|e| format!("[COSE] Failed to serialize signed claim: {}", e))
}

/// Signs a CBOR-encoded claim using the default SHA-384 algorithm.
///
/// This function is provided for backward compatibility and convenience.
/// It internally calls `sign_claim_with_algorithm` with SHA-384.
///
/// # Arguments
///
/// * `claim_cbor` - The CBOR-encoded claim data to sign
/// * `private_key` - The private key used for signing
///
/// # Returns
///
/// * `Ok(Vec<u8>)` - The signed COSE structure as a byte array
/// * `Err(String)` - Error message if signing fails
///
/// # Example
///
/// ```rust,ignore
/// use atlas_c2pa_lib::cose::sign_claim;
/// use openssl::pkey::PKey;
///
/// let private_key = PKey::private_key_from_pem(&key_pem).unwrap();
/// let signed = sign_claim(&cbor_data, &private_key).unwrap();
/// ```
pub fn sign_claim(
    claim_cbor: &[u8],
    private_key: &PKey<openssl::pkey::Private>,
) -> Result<Vec<u8>, String> {
    sign_claim_with_algorithm(claim_cbor, private_key, HashAlgorithm::Sha384)
}

/// Verifies a signed claim using the specified hash algorithm.
///
/// # Arguments
///
/// * `signed_claim` - The signed COSE structure to verify
/// * `public_key` - The public key used for verification
/// * `algorithm` - The hash algorithm that was used for signing
///
/// # Returns
///
/// * `Ok(())` - If the signature is valid
/// * `Err(String)` - Error message if verification fails
///
/// # Example
///
/// ```rust,ignore
/// use atlas_c2pa_lib::cose::{verify_signed_claim_with_algorithm, HashAlgorithm};
/// use openssl::pkey::PKey;
///
/// let public_key = PKey::public_key_from_pem(&key_pem).unwrap();
/// let result = verify_signed_claim_with_algorithm(
///     &signed_data,
///     &public_key,
///     HashAlgorithm::Sha384
/// );
/// assert!(result.is_ok());
/// ```
pub fn verify_signed_claim_with_algorithm(
    signed_claim: &[u8],
    public_key: &PKey<openssl::pkey::Public>,
    algorithm: HashAlgorithm,
) -> Result<(), String> {
    // Parse the COSE-encoded signed claim
    let sign1 = CoseSign1::from_slice(signed_claim)
        .map_err(|e| format!("Failed to parse signed claim: {}", e))?;

    // Get the payload (the signed data)
    let payload = sign1
        .payload
        .as_ref()
        .ok_or("No payload found in signed claim")?;

    // Extract the signature from the COSE structure
    let signature: &[u8] = &sign1.signature;

    // Initialize the verifier with the public key and the specified algorithm
    let mut verifier = Verifier::new(algorithm.to_message_digest(), public_key).map_err(|e| {
        format!(
            "Failed to create verifier with {} algorithm: {}",
            algorithm.as_str(),
            e
        )
    })?;

    // Feed the payload (the data that was signed) into the verifier
    verifier
        .update(payload)
        .map_err(|e| format!("Failed to update verifier with payload: {}", e))?;

    // Verify the signature using the public key
    if verifier
        .verify(signature)
        .map_err(|e| format!("Verification failed with {}: {}", algorithm.as_str(), e))?
    {
        Ok(()) // Signature is valid
    } else {
        Err(format!(
            "Invalid signature for {} algorithm",
            algorithm.as_str()
        )) // Signature is invalid
    }
}

/// Verifies a signed claim using the default SHA-384 algorithm.
///
/// This function is provided for backward compatibility and convenience.
/// It internally calls `verify_signed_claim_with_algorithm` with SHA-384.
///
/// # Arguments
///
/// * `signed_claim` - The signed COSE structure to verify
/// * `public_key` - The public key used for verification
///
/// # Returns
///
/// * `Ok(())` - If the signature is valid
/// * `Err(String)` - Error message if verification fails
///
/// # Example
///
/// ```rust,ignore
/// use atlas_c2pa_lib::cose::verify_signed_claim;
/// use openssl::pkey::PKey;
///
/// let public_key = PKey::public_key_from_pem(&key_pem).unwrap();
/// let result = verify_signed_claim(&signed_data, &public_key);
/// assert!(result.is_ok());
/// ```
pub fn verify_signed_claim(
    signed_claim: &[u8],
    public_key: &PKey<openssl::pkey::Public>,
) -> Result<(), String> {
    verify_signed_claim_with_algorithm(signed_claim, public_key, HashAlgorithm::Sha384)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openssl::pkey::PKey;
    use openssl::rsa::Rsa;

    // Helper function to generate a test key pair
    fn generate_test_keypair() -> (PKey<openssl::pkey::Private>, PKey<openssl::pkey::Public>) {
        let rsa = Rsa::generate(2048).expect("Failed to generate RSA key");
        let private_key = PKey::from_rsa(rsa.clone()).expect("Failed to create private key");
        let public_key_rsa = rsa
            .public_key_to_pem()
            .expect("Failed to export public key");
        let public_key =
            PKey::public_key_from_pem(&public_key_rsa).expect("Failed to create public key");
        (private_key, public_key)
    }

    #[test]
    fn test_hash_algorithm_conversions() {
        // Test as_str()
        assert_eq!(HashAlgorithm::Sha256.as_str(), "sha256");
        assert_eq!(HashAlgorithm::Sha384.as_str(), "sha384");
        assert_eq!(HashAlgorithm::Sha512.as_str(), "sha512");

        // Test from_str() - valid cases
        assert!(matches!(
            HashAlgorithm::from_str("sha256").unwrap(),
            HashAlgorithm::Sha256
        ));
        assert!(matches!(
            HashAlgorithm::from_str("sha384").unwrap(),
            HashAlgorithm::Sha384
        ));
        assert!(matches!(
            HashAlgorithm::from_str("sha512").unwrap(),
            HashAlgorithm::Sha512
        ));

        // Test from_str() - invalid cases
        assert!(HashAlgorithm::from_str("sha1").is_err());
        assert!(HashAlgorithm::from_str("md5").is_err());
        assert!(HashAlgorithm::from_str("SHA256").is_err()); // Case sensitive
        assert!(HashAlgorithm::from_str("").is_err());
    }

    #[test]
    fn test_sign_and_verify_with_sha256() {
        let (private_key, public_key) = generate_test_keypair();
        let test_data = b"Test claim data for SHA-256";

        // Sign with SHA-256
        let signed = sign_claim_with_algorithm(test_data, &private_key, HashAlgorithm::Sha256)
            .expect("Failed to sign with SHA-256");

        // Verify with SHA-256
        assert!(
            verify_signed_claim_with_algorithm(&signed, &public_key, HashAlgorithm::Sha256).is_ok(),
            "Failed to verify SHA-256 signature"
        );

        // Verify with wrong algorithm should fail
        assert!(
            verify_signed_claim_with_algorithm(&signed, &public_key, HashAlgorithm::Sha384)
                .is_err(),
            "Verification should fail with wrong algorithm"
        );
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        use openssl::pkey::PKey;
        use openssl::rsa::Rsa;

        // Helper function to generate a test key pair
        fn generate_test_keypair() -> (PKey<openssl::pkey::Private>, PKey<openssl::pkey::Public>) {
            let rsa = Rsa::generate(2048).expect("Failed to generate RSA key");
            let private_key = PKey::from_rsa(rsa.clone()).expect("Failed to create private key");
            let public_key_rsa = rsa
                .public_key_to_pem()
                .expect("Failed to export public key");
            let public_key =
                PKey::public_key_from_pem(&public_key_rsa).expect("Failed to create public key");
            (private_key, public_key)
        }

        #[test]
        fn test_hash_algorithm_conversions() {
            // Test as_str()
            assert_eq!(HashAlgorithm::Sha256.as_str(), "sha256");
            assert_eq!(HashAlgorithm::Sha384.as_str(), "sha384");
            assert_eq!(HashAlgorithm::Sha512.as_str(), "sha512");

            // Test from_str() - valid cases
            assert!(matches!(
                HashAlgorithm::from_str("sha256").unwrap(),
                HashAlgorithm::Sha256
            ));
            assert!(matches!(
                HashAlgorithm::from_str("sha384").unwrap(),
                HashAlgorithm::Sha384
            ));
            assert!(matches!(
                HashAlgorithm::from_str("sha512").unwrap(),
                HashAlgorithm::Sha512
            ));

            // Test from_str() - invalid cases
            assert!(HashAlgorithm::from_str("sha1").is_err());
            assert!(HashAlgorithm::from_str("md5").is_err());
            assert!(HashAlgorithm::from_str("SHA256").is_err()); // Case sensitive
            assert!(HashAlgorithm::from_str("").is_err());
        }

        #[test]
        fn test_sign_and_verify_with_sha256() {
            let (private_key, public_key) = generate_test_keypair();
            let test_data = b"Test claim data for SHA-256";

            // Sign with SHA-256
            let signed = sign_claim_with_algorithm(test_data, &private_key, HashAlgorithm::Sha256)
                .expect("Failed to sign with SHA-256");

            // Verify with SHA-256
            assert!(
                verify_signed_claim_with_algorithm(&signed, &public_key, HashAlgorithm::Sha256)
                    .is_ok(),
                "Failed to verify SHA-256 signature"
            );

            // Verify with wrong algorithm should fail
            assert!(
                verify_signed_claim_with_algorithm(&signed, &public_key, HashAlgorithm::Sha384)
                    .is_err(),
                "Verification should fail with wrong algorithm"
            );
        }

        #[test]
        fn test_sign_and_verify_with_sha384() {
            let (private_key, public_key) = generate_test_keypair();
            let test_data = b"Test claim data for SHA-384";

            // Sign with SHA-384
            let signed = sign_claim_with_algorithm(test_data, &private_key, HashAlgorithm::Sha384)
                .expect("Failed to sign with SHA-384");

            // Verify with SHA-384
            assert!(
                verify_signed_claim_with_algorithm(&signed, &public_key, HashAlgorithm::Sha384)
                    .is_ok(),
                "Failed to verify SHA-384 signature"
            );

            // Verify with wrong algorithm should fail
            assert!(
                verify_signed_claim_with_algorithm(&signed, &public_key, HashAlgorithm::Sha512)
                    .is_err(),
                "Verification should fail with wrong algorithm"
            );
        }

        #[test]
        fn test_sign_and_verify_with_sha512() {
            let (private_key, public_key) = generate_test_keypair();
            let test_data = b"Test claim data for SHA-512";

            // Sign with SHA-512
            let signed = sign_claim_with_algorithm(test_data, &private_key, HashAlgorithm::Sha512)
                .expect("Failed to sign with SHA-512");

            // Verify with SHA-512
            assert!(
                verify_signed_claim_with_algorithm(&signed, &public_key, HashAlgorithm::Sha512)
                    .is_ok(),
                "Failed to verify SHA-512 signature"
            );

            // Verify with wrong algorithm should fail
            assert!(
                verify_signed_claim_with_algorithm(&signed, &public_key, HashAlgorithm::Sha256)
                    .is_err(),
                "Verification should fail with wrong algorithm"
            );
        }

        #[test]
        fn test_default_functions_use_sha384() {
            let (private_key, public_key) = generate_test_keypair();
            let test_data = b"Test claim data for default algorithm";

            // Sign with default function (should use SHA-384)
            let signed_default =
                sign_claim(test_data, &private_key).expect("Failed to sign with default algorithm");

            // Sign explicitly with SHA-384
            let signed_explicit =
                sign_claim_with_algorithm(test_data, &private_key, HashAlgorithm::Sha384)
                    .expect("Failed to sign with explicit SHA-384");

            // Verify default signed data with SHA-384
            assert!(
                verify_signed_claim_with_algorithm(
                    &signed_default,
                    &public_key,
                    HashAlgorithm::Sha384
                )
                .is_ok(),
                "Default sign_claim should use SHA-384"
            );

            // Verify default function can verify SHA-384 signatures
            assert!(
                verify_signed_claim(&signed_explicit, &public_key).is_ok(),
                "Default verify_signed_claim should use SHA-384"
            );

            // Default verify should work with default sign
            assert!(
                verify_signed_claim(&signed_default, &public_key).is_ok(),
                "Default functions should be compatible"
            );
        }

        #[test]
        fn test_signature_uniqueness() {
            let (private_key, public_key) = generate_test_keypair();
            let test_data = b"Test data for signature uniqueness";

            // Sign the same data multiple times with the same algorithm
            let sig1 = sign_claim_with_algorithm(test_data, &private_key, HashAlgorithm::Sha384)
                .expect("Failed to sign 1");
            let sig2 = sign_claim_with_algorithm(test_data, &private_key, HashAlgorithm::Sha384)
                .expect("Failed to sign 2");

            // Signatures might be different due to randomness in signing process
            // But both should b valid
            assert!(
                verify_signed_claim_with_algorithm(&sig1, &public_key, HashAlgorithm::Sha384)
                    .is_ok(),
                "First signature should be valid"
            );
            assert!(
                verify_signed_claim_with_algorithm(&sig2, &public_key, HashAlgorithm::Sha384)
                    .is_ok(),
                "Second signature should be valid"
            );

            // Sign with different algorithms
            let sig_256 = sign_claim_with_algorithm(test_data, &private_key, HashAlgorithm::Sha256)
                .expect("Failed to sign with SHA-256");
            let sig_384 = sign_claim_with_algorithm(test_data, &private_key, HashAlgorithm::Sha384)
                .expect("Failed to sign with SHA-384");
            let sig_512 = sign_claim_with_algorithm(test_data, &private_key, HashAlgorithm::Sha512)
                .expect("Failed to sign with SHA-512");

            // Signatures with different algorithms should produce different results
            assert_ne!(
                sig_256, sig_384,
                "SHA-256 and SHA-384 signatures should differ"
            );
            assert_ne!(
                sig_384, sig_512,
                "SHA-384 and SHA-512 signatures should differ"
            );
            assert_ne!(
                sig_256, sig_512,
                "SHA-256 and SHA-512 signatures should differ"
            );
        }

        #[test]
        fn test_empty_data_signing() {
            let (private_key, public_key) = generate_test_keypair();
            let empty_data = b"";

            // Test all algorithms with empty data
            for algo in vec![
                HashAlgorithm::Sha256,
                HashAlgorithm::Sha384,
                HashAlgorithm::Sha512,
            ] {
                let signed = sign_claim_with_algorithm(empty_data, &private_key, algo.clone())
                    .expect(&format!("Failed to sign empty data with {:?}", algo));

                assert!(
                    verify_signed_claim_with_algorithm(&signed, &public_key, algo.clone()).is_ok(),
                    "Failed to verify empty data signature with {:?}",
                    algo
                );
            }
        }

        #[test]
        fn test_large_data_signing() {
            let (private_key, public_key) = generate_test_keypair();
            // Create 1MB of test data
            let large_data = vec![0x42u8; 1024 * 1024];

            // Test with SHA-384 (default)
            let signed = sign_claim(&large_data, &private_key).expect("Failed to sign large data");

            assert!(
                verify_signed_claim(&signed, &public_key).is_ok(),
                "Failed to verify large data signature"
            );
        }

        #[test]
        fn test_verification_with_wrong_key() {
            let (private_key1, _) = generate_test_keypair();
            let (_, public_key2) = generate_test_keypair();
            let test_data = b"Test data for wrong key verification";

            // Sign with key1
            let signed = sign_claim_with_algorithm(test_data, &private_key1, HashAlgorithm::Sha384)
                .expect("Failed to sign");

            // Verify with key2 should fail
            assert!(
                verify_signed_claim_with_algorithm(&signed, &public_key2, HashAlgorithm::Sha384)
                    .is_err(),
                "Verification should fail with wrong public key"
            );
        }
    }
}
