//! FFI bindings to the SYCL GPU hashing library
//!
//! This module provides low-level bindings to the C API defined in gpu_hash.h

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::os::raw::{c_char, c_int, c_void};

/// Hash algorithm identifiers
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuHashAlgorithm {
    Sha256 = 0,
    Sha384 = 1,
    Sha512 = 2,
}

/// Error codes from the C library
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuHashError {
    Success = 0,
    NoDevice = 1,
    InvalidAlgorithm = 2,
    MemoryAllocation = 3,
    KernelExecution = 4,
    InvalidInput = 5,
    NotInitialized = 6,
    Unknown = 99,
}

/// Device type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeviceType {
    Gpu = 0,
    Cpu = 1,
    Accelerator = 2,
}

/// Device information structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub name: [c_char; 256],
    pub vendor: [c_char; 256],
    pub driver_version: [c_char; 64],
    pub device_type: GpuDeviceType,
    pub max_compute_units: u32,
    pub global_memory_size: u64,
    pub local_memory_size: u64,
    pub max_work_group_size: usize,
    pub is_intel: c_int,
    pub is_intel_xe: c_int,
}

impl Default for GpuDeviceInfo {
    fn default() -> Self {
        Self {
            name: [0; 256],
            vendor: [0; 256],
            driver_version: [0; 64],
            device_type: GpuDeviceType::Gpu,
            max_compute_units: 0,
            global_memory_size: 0,
            local_memory_size: 0,
            max_work_group_size: 0,
            is_intel: 0,
            is_intel_xe: 0,
        }
    }
}

/// Opaque handle to GPU hasher context
pub type GpuHashContextHandle = *mut c_void;

// Link to the native library
#[link(name = "gpu_hash")]
unsafe extern "C" {
    /// Initialize the GPU hashing library
    pub fn gpu_hash_init() -> GpuHashError;

    /// Cleanup the GPU hashing library
    pub fn gpu_hash_cleanup();

    /// Check if GPU hashing is available
    pub fn gpu_hash_is_available() -> c_int;

    /// Get the number of available GPU devices
    pub fn gpu_hash_get_device_count() -> c_int;

    /// Get information about a specific device
    pub fn gpu_hash_get_device_info(device_index: c_int, info: *mut GpuDeviceInfo) -> GpuHashError;

    /// Create a new hash context
    pub fn gpu_hash_create_context(
        algorithm: GpuHashAlgorithm,
        device_index: c_int,
        handle: *mut GpuHashContextHandle,
    ) -> GpuHashError;

    /// Destroy a hash context
    pub fn gpu_hash_destroy_context(handle: GpuHashContextHandle);

    /// Hash a single message
    pub fn gpu_hash_single(
        handle: GpuHashContextHandle,
        input: *const u8,
        input_len: usize,
        output: *mut u8,
        output_len: *mut usize,
    ) -> GpuHashError;

    /// Hash multiple messages in parallel
    pub fn gpu_hash_batch(
        handle: GpuHashContextHandle,
        inputs: *const *const u8,
        input_lens: *const usize,
        num_inputs: usize,
        outputs: *mut *mut u8,
        output_size: usize,
    ) -> GpuHashError;

    /// Hash contiguous fixed-size messages
    pub fn gpu_hash_batch_fixed(
        handle: GpuHashContextHandle,
        input: *const u8,
        message_size: usize,
        num_messages: usize,
        output: *mut u8,
    ) -> GpuHashError;

    /// Get the output size for a hash algorithm
    pub fn gpu_hash_output_size(algorithm: GpuHashAlgorithm) -> usize;

    /// Get the last error message
    pub fn gpu_hash_get_last_error() -> *const c_char;
}

/// Safe wrapper to get the last error message
pub fn get_last_error_message() -> String {
    unsafe {
        let ptr = gpu_hash_get_last_error();
        if ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

/// Convert GpuDeviceInfo name field to String
pub fn device_info_name(info: &GpuDeviceInfo) -> String {
    unsafe {
        std::ffi::CStr::from_ptr(info.name.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}

/// Convert GpuDeviceInfo vendor field to String
pub fn device_info_vendor(info: &GpuDeviceInfo) -> String {
    unsafe {
        std::ffi::CStr::from_ptr(info.vendor.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}

/// Convert GpuDeviceInfo driver_version field to String
pub fn device_info_driver_version(info: &GpuDeviceInfo) -> String {
    unsafe {
        std::ffi::CStr::from_ptr(info.driver_version.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}
