//! GPU Device Management
//!
//! This module provides device discovery and information for Intel Xe GPUs.

use super::error::{GpuError, GpuResult, check_error};
use super::ffi;

/// Type of GPU device
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeviceType {
    /// GPU device
    Gpu,
    /// CPU device (can run SYCL code)
    Cpu,
    /// Accelerator device
    Accelerator,
}

impl From<ffi::GpuDeviceType> for GpuDeviceType {
    fn from(dt: ffi::GpuDeviceType) -> Self {
        match dt {
            ffi::GpuDeviceType::Gpu => GpuDeviceType::Gpu,
            ffi::GpuDeviceType::Cpu => GpuDeviceType::Cpu,
            ffi::GpuDeviceType::Accelerator => GpuDeviceType::Accelerator,
        }
    }
}

/// Information about a GPU device
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Device name
    name: String,
    /// Vendor name
    vendor: String,
    /// Driver version
    driver_version: String,
    /// Device type
    device_type: GpuDeviceType,
    /// Maximum compute units
    max_compute_units: u32,
    /// Global memory size in bytes
    global_memory_size: u64,
    /// Local memory size in bytes
    local_memory_size: u64,
    /// Maximum work group size
    max_work_group_size: usize,
    /// Whether this is an Intel device
    is_intel: bool,
    /// Whether this is an Intel Xe GPU
    is_intel_xe: bool,
    /// Device index for selection
    device_index: i32,
}

impl GpuDevice {
    /// Get the device name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the vendor name
    pub fn vendor(&self) -> &str {
        &self.vendor
    }

    /// Get the driver version
    pub fn driver_version(&self) -> &str {
        &self.driver_version
    }

    /// Get the device type
    pub fn device_type(&self) -> GpuDeviceType {
        self.device_type
    }

    /// Get the maximum compute units
    pub fn max_compute_units(&self) -> u32 {
        self.max_compute_units
    }

    /// Get the maximum work group size
    pub fn max_work_group_size(&self) -> usize {
        self.max_work_group_size
    }

    /// Get global memory size in bytes
    pub fn global_memory_size(&self) -> u64 {
        self.global_memory_size
    }

    /// Get local memory size in bytes
    pub fn local_memory_size(&self) -> u64 {
        self.local_memory_size
    }

    /// Check if this is an Intel device
    pub fn is_intel(&self) -> bool {
        self.is_intel
    }

    /// Check if this is an Intel Xe GPU
    pub fn is_intel_xe(&self) -> bool {
        self.is_intel_xe
    }

    /// Get the device index (for internal use)
    pub(crate) fn device_index(&self) -> i32 {
        self.device_index
    }
}

/// Get all available GPU devices
pub fn get_available_devices() -> GpuResult<Vec<GpuDevice>> {
    // Initialize the library
    unsafe {
        let err = ffi::gpu_hash_init();
        check_error(err)?;
    }

    let count = unsafe { ffi::gpu_hash_get_device_count() };
    let mut devices = Vec::with_capacity(count as usize);

    for i in 0..count {
        let mut info = ffi::GpuDeviceInfo::default();
        let err = unsafe { ffi::gpu_hash_get_device_info(i, &mut info) };
        check_error(err)?;

        devices.push(GpuDevice {
            name: ffi::device_info_name(&info),
            vendor: ffi::device_info_vendor(&info),
            driver_version: ffi::device_info_driver_version(&info),
            device_type: info.device_type.into(),
            max_compute_units: info.max_compute_units,
            global_memory_size: info.global_memory_size,
            local_memory_size: info.local_memory_size,
            max_work_group_size: info.max_work_group_size,
            is_intel: info.is_intel != 0,
            is_intel_xe: info.is_intel_xe != 0,
            device_index: i,
        });
    }

    Ok(devices)
}

/// Get the best available Intel Xe GPU device
pub fn get_best_intel_device() -> GpuResult<GpuDevice> {
    let devices = get_available_devices()?;

    // First try to find an Intel Xe GPU
    if let Some(device) = devices.iter().find(|d| d.is_intel_xe) {
        return Ok(device.clone());
    }

    // Fall back to any Intel device
    if let Some(device) = devices.iter().find(|d| d.is_intel) {
        return Ok(device.clone());
    }

    // Fall back to any GPU
    if let Some(device) = devices.into_iter().next() {
        return Ok(device);
    }

    Err(GpuError::NoDeviceFound)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_discovery() {
        match get_available_devices() {
            Ok(devices) => {
                println!("Found {} devices", devices.len());
                for device in &devices {
                    println!(
                        "  {} ({}) - Intel Xe: {}",
                        device.name(),
                        device.vendor(),
                        device.is_intel_xe()
                    );
                }
            }
            Err(e) => {
                println!("Device discovery failed (expected if no GPU): {}", e);
            }
        }
    }
}
