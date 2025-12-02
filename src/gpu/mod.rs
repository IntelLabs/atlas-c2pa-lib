//! # GPU Hashing Module
//!
//! This module provides GPU-accelerated hashing using Intel Xe GPUs via SYCL/oneAPI.
//! It supports SHA-256, SHA-384, and SHA-512 algorithms optimized for Intel discrete
//! and integrated GPUs.
//!
//! ## Features
//!
//! - **Intel Xe GPU Support**: Optimized SYCL kernels for Intel Arc, Iris Xe, and UHD Graphics
//! - **Multiple Algorithms**: SHA-256, SHA-384, SHA-512
//! - **Batch Processing**: Efficient hashing of multiple messages in parallel
//! - **Automatic Fallback**: Falls back to CPU hashing when GPU is unavailable
//!
//! ## Usage
//!
//! This module is only available when the `gpu-hashing` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! atlas-c2pa-lib = { version = "0.1", features = ["gpu-hashing"] }
//! ```
//!
//! ## Prerequisites
//!
//! - Intel oneAPI Base Toolkit (for DPC++/SYCL compiler)
//! - Intel GPU drivers with Level Zero support
//!
//! ## Example
//!
//! ```rust,ignore
//! use atlas_c2pa_lib::gpu::{GpuHasher, GpuHashAlgorithm};
//!
//! // Create a GPU hasher for SHA-256
//! let hasher = GpuHasher::new(GpuHashAlgorithm::Sha256)?;
//!
//! // Hash a single message
//! let hash = hasher.hash(b"Hello, World!")?;
//!
//! // Hash multiple messages in parallel
//! let messages = vec![
//!     b"Message 1".to_vec(),
//!     b"Message 2".to_vec(),
//!     b"Message 3".to_vec(),
//! ];
//! let hashes = hasher.hash_batch(&messages)?;
//! ```

mod context;
mod error;
mod ffi;
mod hasher;

pub use context::{GpuDevice, GpuDeviceType};
pub use error::{GpuError, GpuResult};
pub use hasher::{GpuHashAlgorithm, GpuHasher, HashOutput};

/// Check if GPU hashing is available on this system
///
/// Returns `true` if an Intel GPU with SYCL support is detected.
///
/// # Example
///
/// ```rust,ignore
/// use atlas_c2pa_lib::gpu::is_gpu_available;
///
/// if is_gpu_available() {
///     println!("GPU hashing is available!");
/// } else {
///     println!("Falling back to CPU hashing");
/// }
/// ```
pub fn is_gpu_available() -> bool {
    unsafe { ffi::gpu_hash_is_available() != 0 }
}

/// Get information about available GPU devices
///
/// Returns a list of available Intel GPU devices that support SYCL.
///
/// # Example
///
/// ```rust,ignore
/// use atlas_c2pa_lib::gpu::get_available_devices;
///
/// for device in get_available_devices()? {
///     println!("Found device: {} ({})", device.name(), device.vendor());
/// }
/// ```
pub fn get_available_devices() -> GpuResult<Vec<GpuDevice>> {
    context::get_available_devices()
}

/// Convenience function to hash data using GPU if available, CPU otherwise
///
/// This function automatically selects the best available method for hashing.
/// If an Intel GPU is available, it uses GPU acceleration. Otherwise, it falls
/// back to CPU-based hashing using the ring crate.
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
/// ```rust,ignore
/// use atlas_c2pa_lib::gpu::{hash_auto, GpuHashAlgorithm};
///
/// let hash = hash_auto(b"Hello, World!", GpuHashAlgorithm::Sha256)?;
/// println!("Hash: {}", hex::encode(&hash));
/// ```
pub fn hash_auto(data: &[u8], algorithm: GpuHashAlgorithm) -> GpuResult<Vec<u8>> {
    if is_gpu_available() {
        let hasher = GpuHasher::new(algorithm)?;
        hasher.hash(data)
    } else {
        // Fallback to CPU hashing
        Ok(hash_cpu(data, algorithm))
    }
}

/// Hash data using CPU (fallback implementation)
fn hash_cpu(data: &[u8], algorithm: GpuHashAlgorithm) -> Vec<u8> {
    use ring::digest;

    let algorithm = match algorithm {
        GpuHashAlgorithm::Sha256 => &digest::SHA256,
        GpuHashAlgorithm::Sha384 => &digest::SHA384,
        GpuHashAlgorithm::Sha512 => &digest::SHA512,
    };

    digest::digest(algorithm, data).as_ref().to_vec()
}

/// Batch hash multiple messages using GPU if available
///
/// This is more efficient than hashing messages one by one when you have
/// many messages to process.
///
/// # Arguments
///
/// * `messages` - Slice of messages to hash
/// * `algorithm` - The hash algorithm to use
///
/// # Returns
///
/// Vector of hashes, one per input message
///
/// # Example
///
/// ```rust,ignore
/// use atlas_c2pa_lib::gpu::{hash_batch_auto, GpuHashAlgorithm};
///
/// let messages = vec![
///     b"Message 1".to_vec(),
///     b"Message 2".to_vec(),
///     b"Message 3".to_vec(),
/// ];
///
/// let hashes = hash_batch_auto(&messages, GpuHashAlgorithm::Sha256)?;
/// for (i, hash) in hashes.iter().enumerate() {
///     println!("Message {} hash: {}", i, hex::encode(hash));
/// }
/// ```
pub fn hash_batch_auto(
    messages: &[Vec<u8>],
    algorithm: GpuHashAlgorithm,
) -> GpuResult<Vec<Vec<u8>>> {
    if is_gpu_available() && messages.len() > 1 {
        let hasher = GpuHasher::new(algorithm)?;
        hasher.hash_batch(messages)
    } else {
        // Fallback to CPU hashing
        Ok(messages
            .iter()
            .map(|msg| hash_cpu(msg, algorithm))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_availability_check() {
        // This test just ensures the availability check doesn't panic
        let _ = is_gpu_available();
    }

    #[test]
    fn test_hash_auto_sha256() {
        let data = b"Hello, World!";
        let result = hash_auto(data, GpuHashAlgorithm::Sha256);
        assert!(result.is_ok());
        let hash = result.unwrap();
        assert_eq!(hash.len(), 32); // SHA-256 produces 32 bytes
    }

    #[test]
    fn test_hash_auto_sha384() {
        let data = b"Hello, World!";
        let result = hash_auto(data, GpuHashAlgorithm::Sha384);
        assert!(result.is_ok());
        let hash = result.unwrap();
        assert_eq!(hash.len(), 48); // SHA-384 produces 48 bytes
    }

    #[test]
    fn test_hash_auto_sha512() {
        let data = b"Hello, World!";
        let result = hash_auto(data, GpuHashAlgorithm::Sha512);
        assert!(result.is_ok());
        let hash = result.unwrap();
        assert_eq!(hash.len(), 64); // SHA-512 produces 64 bytes
    }

    #[test]
    fn test_batch_hash() {
        let messages = vec![
            b"Message 1".to_vec(),
            b"Message 2".to_vec(),
            b"Message 3".to_vec(),
        ];

        let result = hash_batch_auto(&messages, GpuHashAlgorithm::Sha256);
        assert!(result.is_ok());
        let hashes = result.unwrap();
        assert_eq!(hashes.len(), 3);

        // Verify each hash has correct length
        for hash in hashes {
            assert_eq!(hash.len(), 32);
        }
    }

    #[test]
    fn test_known_sha256_vector() {
        // Test vector: SHA-256 of empty string
        let data = b"";
        let result = hash_auto(data, GpuHashAlgorithm::Sha256).unwrap();
        let expected =
            hex::decode("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
                .unwrap();
        assert_eq!(result, expected);
    }
}
