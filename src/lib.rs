//! # Atlas C2PA Library
//!
//! `atlas-c2pa-lib` is a Rust library for creating, signing, and verifying machine learning
//! assets (models and datasets) with C2PA (Content Provenance and Authenticity) specifications.
//!
//! The library provides tools to generate cryptographic claims about ML asset provenance,
//! track ML asset lineage, and create C2PA-compliant manifests.
//!
//! ## Key Components
//!
//! - **Assertions**: Claims about ML models and datasets
//! - **Ingredients**: Tracking datasets and components used to create models
//! - **Manifests**: Complete C2PA manifests for ML assets
//! - **Asset Types**: Support for various ML frameworks (TensorFlow, PyTorch, ONNX, etc.)
//!
//! ## Optional Features
//!
//! - **gpu-hashing**: GPU-accelerated hashing using Intel Xe GPUs via SYCL/oneAPI.
//!   Enable with: `atlas-c2pa-lib = { version = "0.1", features = ["gpu-hashing"] }`
//!
//! ## Example: Creating a Model Manifest
//!
//! ```rust
//! use atlas_c2pa_lib::ml::types::{ModelInfo, MLFramework, ModelFormat};
//! use atlas_c2pa_lib::ml::manifest::MLManifestBuilder;
//! use time::OffsetDateTime;
//!
//! // Define model information
//! let model_info = ModelInfo {
//!     name: "bert-base".to_string(),
//!     version: "1.0.0".to_string(),
//!     framework: MLFramework::PyTorch,
//!     format: ModelFormat::TorchScript,
//! };
//!
//! // Create a manifest builder
//! let builder = MLManifestBuilder::new(
//!     model_info.clone(),
//!     "0123456789abcdef0123456789abcdef".to_string(), // Model hash
//! );
//!
//! // Build the manifest
//! let manifest = builder.build().unwrap();
//! ```
//!
//! ## Example: GPU-Accelerated Hashing (requires `gpu-hashing` feature)
//!
//! ```rust,ignore
//! use atlas_c2pa_lib::gpu::{GpuHasher, GpuHashAlgorithm, hash_auto};
//!
//! // Hash using GPU if available, CPU otherwise
//! let hash = hash_auto(b"Hello, World!", GpuHashAlgorithm::Sha256)?;
//! println!("Hash: {}", hex::encode(&hash));
//!
//! // Or create a hasher for multiple operations
//! let hasher = GpuHasher::new(GpuHashAlgorithm::Sha256)?;
//! let hash1 = hasher.hash(b"Message 1")?;
//! let hash2 = hasher.hash(b"Message 2")?;
//!
//! // Batch hash multiple messages efficiently on GPU
//! let messages = vec![b"Msg1".to_vec(), b"Msg2".to_vec(), b"Msg3".to_vec()];
//! let hashes = hasher.hash_batch(&messages)?;
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod assertion;
pub mod asset_type;
pub mod cbor;
pub mod claim;
pub mod cose;
pub mod cross_reference;
pub mod datetime_wrapper;
pub mod ingredient;
pub mod manifest;
pub mod ml;

/// GPU-accelerated hashing module (requires `gpu-hashing` feature)
///
/// This module provides GPU-accelerated hashing using Intel Xe GPUs via SYCL/oneAPI.
/// It supports SHA-256, SHA-384, and SHA-512 algorithms.
///
/// ## Prerequisites
///
/// - Intel GPU (Arc, Iris Xe, UHD Graphics, or other Xe-based GPU)
/// - Intel oneAPI Base Toolkit (for DPC++ compiler during build)
/// - Intel GPU drivers with Level Zero support
///
/// ## Build Requirements
///
/// ```bash
/// # Install Intel oneAPI compiler
/// sudo apt install intel-oneapi-compiler-dpcpp-cpp
///
/// # Source the environment before building
/// source /opt/intel/oneapi/setvars.sh
///
/// # Build with GPU support
/// cargo build --features gpu-hashing
/// ```
///
/// ## Example
///
/// ```rust,ignore
/// use atlas_c2pa_lib::gpu::{GpuHasher, GpuHashAlgorithm, is_gpu_available};
///
/// if is_gpu_available() {
///     let hasher = GpuHasher::new(GpuHashAlgorithm::Sha256)?;
///     let hash = hasher.hash(b"Hello, World!")?;
///     println!("GPU hash: {}", hex::encode(&hash));
/// } else {
///     println!("GPU not available, using CPU");
/// }
/// ```
#[cfg(feature = "gpu-hashing")]
#[cfg_attr(docsrs, doc(cfg(feature = "gpu-hashing")))]
pub mod gpu;

/// Hash utilities module
///
/// Provides convenient hashing functions that work regardless of GPU availability.
pub mod hash {
    use crate::cose::HashAlgorithm;

    /// Compute a hash of the given data using the specified algorithm.
    ///
    /// This function uses CPU-based hashing via the ring crate.
    /// For GPU-accelerated hashing, use the `gpu` module when the
    /// `gpu-hashing` feature is enabled.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to hash
    /// * `algorithm` - The hash algorithm to use
    ///
    /// # Returns
    ///
    /// The hash as a byte vector
    ///
    /// # Example
    ///
    /// ```rust
    /// use atlas_c2pa_lib::hash::compute_hash;
    /// use atlas_c2pa_lib::cose::HashAlgorithm;
    ///
    /// let hash = compute_hash(b"Hello, World!", HashAlgorithm::Sha256);
    /// assert_eq!(hash.len(), 32); // SHA-256 produces 32 bytes
    /// ```
    pub fn compute_hash(data: &[u8], algorithm: HashAlgorithm) -> Vec<u8> {
        use ring::digest;

        let algorithm = match algorithm {
            HashAlgorithm::Sha256 => &digest::SHA256,
            HashAlgorithm::Sha384 => &digest::SHA384,
            HashAlgorithm::Sha512 => &digest::SHA512,
        };

        digest::digest(algorithm, data).as_ref().to_vec()
    }

    /// Compute a SHA-256 hash of the given data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use atlas_c2pa_lib::hash::sha256;
    ///
    /// let hash = sha256(b"Hello, World!");
    /// assert_eq!(hash.len(), 32);
    /// ```
    pub fn sha256(data: &[u8]) -> Vec<u8> {
        compute_hash(data, HashAlgorithm::Sha256)
    }

    /// Compute a SHA-384 hash of the given data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use atlas_c2pa_lib::hash::sha384;
    ///
    /// let hash = sha384(b"Hello, World!");
    /// assert_eq!(hash.len(), 48);
    /// ```
    pub fn sha384(data: &[u8]) -> Vec<u8> {
        compute_hash(data, HashAlgorithm::Sha384)
    }

    /// Compute a SHA-512 hash of the given data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use atlas_c2pa_lib::hash::sha512;
    ///
    /// let hash = sha512(b"Hello, World!");
    /// assert_eq!(hash.len(), 64);
    /// ```
    pub fn sha512(data: &[u8]) -> Vec<u8> {
        compute_hash(data, HashAlgorithm::Sha512)
    }

    /// Compute a hash and return it as a hex string.
    ///
    /// # Example
    ///
    /// ```rust
    /// use atlas_c2pa_lib::hash::compute_hash_hex;
    /// use atlas_c2pa_lib::cose::HashAlgorithm;
    ///
    /// let hash_hex = compute_hash_hex(b"Hello, World!", HashAlgorithm::Sha256);
    /// assert_eq!(hash_hex.len(), 64); // 32 bytes = 64 hex chars
    /// ```
    pub fn compute_hash_hex(data: &[u8], algorithm: HashAlgorithm) -> String {
        hex::encode(compute_hash(data, algorithm))
    }

    /// Compute a hash of a file.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atlas_c2pa_lib::hash::hash_file;
    /// use atlas_c2pa_lib::cose::HashAlgorithm;
    /// use std::path::Path;
    ///
    /// let hash = hash_file(Path::new("model.onnx"), HashAlgorithm::Sha256)?;
    /// println!("Model hash: {}", hex::encode(&hash));
    /// ```
    pub fn hash_file(
        path: &std::path::Path,
        algorithm: HashAlgorithm,
    ) -> Result<Vec<u8>, std::io::Error> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        Ok(compute_hash(&data, algorithm))
    }

    /// Verify that data matches an expected hash.
    ///
    /// # Example
    ///
    /// ```rust
    /// use atlas_c2pa_lib::hash::{sha256, verify_hash};
    /// use atlas_c2pa_lib::cose::HashAlgorithm;
    ///
    /// let data = b"Hello, World!";
    /// let expected_hash = sha256(data);
    ///
    /// assert!(verify_hash(data, &expected_hash, HashAlgorithm::Sha256));
    /// ```
    pub fn verify_hash(data: &[u8], expected: &[u8], algorithm: HashAlgorithm) -> bool {
        let computed = compute_hash(data, algorithm);
        computed == expected
    }

    /// GPU-accelerated hash computation (when available).
    ///
    /// This function automatically uses GPU hashing if available and the
    /// `gpu-hashing` feature is enabled. Otherwise, it falls back to CPU.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atlas_c2pa_lib::hash::compute_hash_auto;
    /// use atlas_c2pa_lib::cose::HashAlgorithm;
    ///
    /// let hash = compute_hash_auto(b"Hello, World!", HashAlgorithm::Sha256)?;
    /// ```
    #[cfg(feature = "gpu-hashing")]
    #[cfg_attr(docsrs, doc(cfg(feature = "gpu-hashing")))]
    pub fn compute_hash_auto(
        data: &[u8],
        algorithm: HashAlgorithm,
    ) -> Result<Vec<u8>, crate::gpu::GpuError> {
        use crate::gpu::GpuHashAlgorithm;

        let gpu_alg = match algorithm {
            HashAlgorithm::Sha256 => GpuHashAlgorithm::Sha256,
            HashAlgorithm::Sha384 => GpuHashAlgorithm::Sha384,
            HashAlgorithm::Sha512 => GpuHashAlgorithm::Sha512,
        };

        crate::gpu::hash_auto(data, gpu_alg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_functions() {
        let data = b"Hello, World!";

        let sha256_hash = hash::sha256(data);
        assert_eq!(sha256_hash.len(), 32);

        let sha384_hash = hash::sha384(data);
        assert_eq!(sha384_hash.len(), 48);

        let sha512_hash = hash::sha512(data);
        assert_eq!(sha512_hash.len(), 64);
    }

    #[test]
    fn test_hash_verification() {
        let data = b"Test data for verification";
        let hash = hash::sha256(data);

        assert!(hash::verify_hash(data, &hash, cose::HashAlgorithm::Sha256));
        assert!(!hash::verify_hash(
            b"Wrong data",
            &hash,
            cose::HashAlgorithm::Sha256
        ));
    }

    #[test]
    fn test_hash_hex() {
        let data = b"";
        let hash_hex = hash::compute_hash_hex(data, cose::HashAlgorithm::Sha256);

        // SHA-256 of empty string
        assert_eq!(
            hash_hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
