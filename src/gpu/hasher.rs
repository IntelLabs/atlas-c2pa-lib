//! GPU Hasher Implementation
//!
//! This module provides the main GpuHasher type for GPU-accelerated hashing.

use super::context::GpuDevice;
use super::error::{GpuError, GpuResult, check_error};
use super::ffi;

/// Supported hash algorithms for GPU hashing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuHashAlgorithm {
    /// SHA-256 (256-bit hash)
    Sha256,
    /// SHA-384 (384-bit hash)  
    Sha384,
    /// SHA-512 (512-bit hash)
    Sha512,
}

impl GpuHashAlgorithm {
    /// Get the output size in bytes for this algorithm
    pub fn output_size(&self) -> usize {
        match self {
            GpuHashAlgorithm::Sha256 => 32,
            GpuHashAlgorithm::Sha384 => 48,
            GpuHashAlgorithm::Sha512 => 64,
        }
    }

    /// Get the block size in bytes for this algorithm
    pub fn block_size(&self) -> usize {
        match self {
            GpuHashAlgorithm::Sha256 => 64,
            GpuHashAlgorithm::Sha384 | GpuHashAlgorithm::Sha512 => 128,
        }
    }

    /// Convert to the cose::HashAlgorithm type
    pub fn to_cose_algorithm(&self) -> crate::cose::HashAlgorithm {
        match self {
            GpuHashAlgorithm::Sha256 => crate::cose::HashAlgorithm::Sha256,
            GpuHashAlgorithm::Sha384 => crate::cose::HashAlgorithm::Sha384,
            GpuHashAlgorithm::Sha512 => crate::cose::HashAlgorithm::Sha512,
        }
    }

    fn to_ffi(&self) -> ffi::GpuHashAlgorithm {
        match self {
            GpuHashAlgorithm::Sha256 => ffi::GpuHashAlgorithm::Sha256,
            GpuHashAlgorithm::Sha384 => ffi::GpuHashAlgorithm::Sha384,
            GpuHashAlgorithm::Sha512 => ffi::GpuHashAlgorithm::Sha512,
        }
    }
}

impl From<crate::cose::HashAlgorithm> for GpuHashAlgorithm {
    fn from(alg: crate::cose::HashAlgorithm) -> Self {
        match alg {
            crate::cose::HashAlgorithm::Sha256 => GpuHashAlgorithm::Sha256,
            crate::cose::HashAlgorithm::Sha384 => GpuHashAlgorithm::Sha384,
            crate::cose::HashAlgorithm::Sha512 => GpuHashAlgorithm::Sha512,
        }
    }
}

impl std::str::FromStr for GpuHashAlgorithm {
    type Err = GpuError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sha256" | "sha-256" => Ok(GpuHashAlgorithm::Sha256),
            "sha384" | "sha-384" => Ok(GpuHashAlgorithm::Sha384),
            "sha512" | "sha-512" => Ok(GpuHashAlgorithm::Sha512),
            _ => Err(GpuError::InvalidInput(format!(
                "Unknown algorithm: {}. Supported: sha256, sha384, sha512",
                s
            ))),
        }
    }
}

/// Output from a hash operation
#[derive(Debug, Clone)]
pub struct HashOutput {
    /// The hash bytes
    bytes: Vec<u8>,
    /// The algorithm used
    algorithm: GpuHashAlgorithm,
}

impl HashOutput {
    /// Create a new hash output
    pub fn new(bytes: Vec<u8>, algorithm: GpuHashAlgorithm) -> Self {
        Self { bytes, algorithm }
    }

    /// Get the hash bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Convert to a hex string
    pub fn to_hex(&self) -> String {
        hex::encode(&self.bytes)
    }

    /// Get the algorithm used
    pub fn algorithm(&self) -> GpuHashAlgorithm {
        self.algorithm
    }

    /// Consume and return the bytes
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

impl AsRef<[u8]> for HashOutput {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl std::fmt::Display for HashOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// GPU-accelerated hasher for Intel Xe GPUs
pub struct GpuHasher {
    algorithm: GpuHashAlgorithm,
    handle: ffi::GpuHashContextHandle,
    output_size: usize,
}

impl GpuHasher {
    /// Create a new GPU hasher for the specified algorithm
    ///
    /// This will automatically select the best available Intel GPU.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - The hash algorithm to use
    ///
    /// # Returns
    ///
    /// A new GpuHasher instance or an error if GPU is unavailable
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atlas_c2pa_lib::gpu::{GpuHasher, GpuHashAlgorithm};
    ///
    /// let hasher = GpuHasher::new(GpuHashAlgorithm::Sha256)?;
    /// ```
    pub fn new(algorithm: GpuHashAlgorithm) -> GpuResult<Self> {
        Self::new_with_device(algorithm, None)
    }

    /// Create a new GPU hasher with a specific device
    ///
    /// # Arguments
    ///
    /// * `algorithm` - The hash algorithm to use
    /// * `device` - Optional device to use (None for auto-select)
    pub fn new_with_device(
        algorithm: GpuHashAlgorithm,
        device: Option<&GpuDevice>,
    ) -> GpuResult<Self> {
        // Initialize the library
        let err = unsafe { ffi::gpu_hash_init() };
        check_error(err)?;

        let device_index = device.map(|d| d.device_index()).unwrap_or(-1);
        let mut handle: ffi::GpuHashContextHandle = std::ptr::null_mut();

        let err =
            unsafe { ffi::gpu_hash_create_context(algorithm.to_ffi(), device_index, &mut handle) };
        check_error(err)?;

        Ok(Self {
            algorithm,
            handle,
            output_size: algorithm.output_size(),
        })
    }

    /// Get the algorithm being used
    pub fn algorithm(&self) -> GpuHashAlgorithm {
        self.algorithm
    }

    /// Get the output size in bytes
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Hash a single message
    ///
    /// # Arguments
    ///
    /// * `data` - The data to hash
    ///
    /// # Returns
    ///
    /// The hash as a byte vector
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let hasher = GpuHasher::new(GpuHashAlgorithm::Sha256)?;
    /// let hash = hasher.hash(b"Hello, World!")?;
    /// println!("Hash: {}", hex::encode(&hash));
    /// ```
    pub fn hash(&self, data: &[u8]) -> GpuResult<Vec<u8>> {
        let mut output = vec![0u8; self.output_size];
        let mut output_len = 0usize;

        let err = unsafe {
            ffi::gpu_hash_single(
                self.handle,
                data.as_ptr(),
                data.len(),
                output.as_mut_ptr(),
                &mut output_len,
            )
        };
        check_error(err)?;

        Ok(output)
    }

    /// Hash a single message and return a HashOutput
    pub fn hash_to_output(&self, data: &[u8]) -> GpuResult<HashOutput> {
        let bytes = self.hash(data)?;
        Ok(HashOutput::new(bytes, self.algorithm))
    }

    /// Hash multiple messages in parallel
    ///
    /// This is more efficient than calling hash() multiple times
    /// when you have many messages to process.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of messages to hash
    ///
    /// # Returns
    ///
    /// Vector of hashes, one per input message
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let hasher = GpuHasher::new(GpuHashAlgorithm::Sha256)?;
    /// let messages = vec![
    ///     b"Message 1".to_vec(),
    ///     b"Message 2".to_vec(),
    /// ];
    /// let hashes = hasher.hash_batch(&messages)?;
    /// ```
    pub fn hash_batch(&self, messages: &[Vec<u8>]) -> GpuResult<Vec<Vec<u8>>> {
        if messages.is_empty() {
            return Ok(Vec::new());
        }

        let num_messages = messages.len();

        // Prepare input pointers and lengths
        let input_ptrs: Vec<*const u8> = messages.iter().map(|m| m.as_ptr()).collect();

        let input_lens: Vec<usize> = messages.iter().map(|m| m.len()).collect();

        // Prepare output buffers
        let mut output_buffers: Vec<Vec<u8>> = (0..num_messages)
            .map(|_| vec![0u8; self.output_size])
            .collect();

        let mut output_ptrs: Vec<*mut u8> =
            output_buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();

        let err = unsafe {
            ffi::gpu_hash_batch(
                self.handle,
                input_ptrs.as_ptr(),
                input_lens.as_ptr(),
                num_messages,
                output_ptrs.as_mut_ptr(),
                self.output_size,
            )
        };
        check_error(err)?;

        Ok(output_buffers)
    }

    /// Hash contiguous fixed-size messages
    ///
    /// This is the most efficient method when all messages have the same size.
    ///
    /// # Arguments
    ///
    /// * `data` - Contiguous buffer containing all messages
    /// * `message_size` - Size of each individual message
    ///
    /// # Returns
    ///
    /// Vector of hashes, one per message
    pub fn hash_batch_fixed(&self, data: &[u8], message_size: usize) -> GpuResult<Vec<Vec<u8>>> {
        if message_size == 0 {
            return Err(GpuError::InvalidInput(
                "Message size cannot be zero".to_string(),
            ));
        }

        if data.len() % message_size != 0 {
            return Err(GpuError::InvalidInput(format!(
                "Data length {} is not a multiple of message size {}",
                data.len(),
                message_size
            )));
        }

        let num_messages = data.len() / message_size;
        if num_messages == 0 {
            return Ok(Vec::new());
        }

        let mut output = vec![0u8; self.output_size * num_messages];

        let err = unsafe {
            ffi::gpu_hash_batch_fixed(
                self.handle,
                data.as_ptr(),
                message_size,
                num_messages,
                output.as_mut_ptr(),
            )
        };
        check_error(err)?;

        // Split output into individual hashes
        let hashes: Vec<Vec<u8>> = output
            .chunks(self.output_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(hashes)
    }

    /// Hash a file from disk
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to hash
    ///
    /// # Returns
    ///
    /// The file hash as a byte vector
    pub fn hash_file(&self, path: &std::path::Path) -> GpuResult<Vec<u8>> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)
            .map_err(|e| GpuError::InvalidInput(format!("Failed to open file: {}", e)))?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| GpuError::InvalidInput(format!("Failed to read file: {}", e)))?;

        self.hash(&data)
    }

    /// Hash multiple files in parallel
    ///
    /// # Arguments
    ///
    /// * `paths` - Slice of file paths to hash
    ///
    /// # Returns
    ///
    /// Vector of hashes, one per file
    pub fn hash_files(&self, paths: &[&std::path::Path]) -> GpuResult<Vec<Vec<u8>>> {
        use std::fs::File;
        use std::io::Read;

        let mut messages = Vec::with_capacity(paths.len());

        for path in paths {
            let mut file = File::open(path).map_err(|e| {
                GpuError::InvalidInput(format!("Failed to open file {:?}: {}", path, e))
            })?;

            let mut data = Vec::new();
            file.read_to_end(&mut data).map_err(|e| {
                GpuError::InvalidInput(format!("Failed to read file {:?}: {}", path, e))
            })?;

            messages.push(data);
        }

        self.hash_batch(&messages)
    }
}

impl Drop for GpuHasher {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::gpu_hash_destroy_context(self.handle);
            }
        }
    }
}

// GpuHasher is safe to send between threads as the underlying SYCL queue is thread-safe
unsafe impl Send for GpuHasher {}

impl std::fmt::Debug for GpuHasher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuHasher")
            .field("algorithm", &self.algorithm)
            .field("output_size", &self.output_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_properties() {
        assert_eq!(GpuHashAlgorithm::Sha256.output_size(), 32);
        assert_eq!(GpuHashAlgorithm::Sha384.output_size(), 48);
        assert_eq!(GpuHashAlgorithm::Sha512.output_size(), 64);

        assert_eq!(GpuHashAlgorithm::Sha256.block_size(), 64);
        assert_eq!(GpuHashAlgorithm::Sha384.block_size(), 128);
        assert_eq!(GpuHashAlgorithm::Sha512.block_size(), 128);
    }

    #[test]
    fn test_algorithm_from_str() {
        assert_eq!(
            "sha256".parse::<GpuHashAlgorithm>().unwrap(),
            GpuHashAlgorithm::Sha256
        );
        assert_eq!(
            "SHA256".parse::<GpuHashAlgorithm>().unwrap(),
            GpuHashAlgorithm::Sha256
        );
        assert_eq!(
            "sha-384".parse::<GpuHashAlgorithm>().unwrap(),
            GpuHashAlgorithm::Sha384
        );
        assert!("invalid".parse::<GpuHashAlgorithm>().is_err());
    }

    #[test]
    fn test_hash_output() {
        let bytes = vec![0x01, 0x02, 0x03, 0x04];
        let output = HashOutput::new(bytes.clone(), GpuHashAlgorithm::Sha256);

        assert_eq!(output.as_bytes(), &bytes);
        assert_eq!(output.to_hex(), "01020304");
        assert_eq!(output.algorithm(), GpuHashAlgorithm::Sha256);
    }
}
