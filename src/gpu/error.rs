//! GPU Error Types
//!
//! This module defines error types for GPU hashing operations.

use super::ffi;
use std::fmt;

/// Result type for GPU operations
pub type GpuResult<T> = Result<T, GpuError>;

/// Errors that can occur during GPU hashing operations
#[derive(Debug)]
pub enum GpuError {
    /// No compatible GPU device was found
    NoDeviceFound,

    /// Invalid hash algorithm specified
    InvalidAlgorithm,

    /// GPU memory allocation failed
    MemoryAllocationFailed(String),

    /// Kernel execution failed
    KernelExecutionFailed(String),

    /// Invalid input data
    InvalidInput(String),

    /// Library not initialized
    NotInitialized,

    /// Generic error with message
    Other(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoDeviceFound => {
                write!(
                    f,
                    "No compatible Intel GPU device found. Ensure Intel GPU drivers and oneAPI runtime are installed."
                )
            }
            GpuError::InvalidAlgorithm => {
                write!(f, "Invalid hash algorithm specified")
            }
            GpuError::MemoryAllocationFailed(msg) => {
                write!(f, "GPU memory allocation failed: {msg}")
            }
            GpuError::KernelExecutionFailed(msg) => {
                write!(f, "Kernel execution failed: {msg}")
            }
            GpuError::InvalidInput(msg) => {
                write!(f, "Invalid input: {msg}")
            }
            GpuError::NotInitialized => {
                write!(f, "GPU hashing library not initialized")
            }
            GpuError::Other(msg) => {
                write!(f, "GPU error: {msg}")
            }
        }
    }
}

impl std::error::Error for GpuError {}

impl From<String> for GpuError {
    fn from(s: String) -> Self {
        GpuError::Other(s)
    }
}

impl From<&str> for GpuError {
    fn from(s: &str) -> Self {
        GpuError::Other(s.to_string())
    }
}

impl From<ffi::GpuHashError> for GpuError {
    fn from(err: ffi::GpuHashError) -> Self {
        let msg = ffi::get_last_error_message();
        match err {
            ffi::GpuHashError::Success => GpuError::Other("Unexpected success error".to_string()),
            ffi::GpuHashError::NoDevice => GpuError::NoDeviceFound,
            ffi::GpuHashError::InvalidAlgorithm => GpuError::InvalidAlgorithm,
            ffi::GpuHashError::MemoryAllocation => GpuError::MemoryAllocationFailed(msg),
            ffi::GpuHashError::KernelExecution => GpuError::KernelExecutionFailed(msg),
            ffi::GpuHashError::InvalidInput => GpuError::InvalidInput(msg),
            ffi::GpuHashError::NotInitialized => GpuError::NotInitialized,
            ffi::GpuHashError::Unknown => GpuError::Other(msg),
        }
    }
}

/// Check a GPU hash error result and convert to Result
pub fn check_error(err: ffi::GpuHashError) -> GpuResult<()> {
    if err == ffi::GpuHashError::Success {
        Ok(())
    } else {
        Err(err.into())
    }
}
