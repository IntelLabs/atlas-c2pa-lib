//! Build script for atlas-c2pa-lib
//!
//! This script handles compilation of the SYCL GPU hashing library when
//! the `gpu-hashing` feature is enabled.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only compile SYCL code if the gpu-hashing feature is enabled
    #[cfg(feature = "gpu-hashing")]
    compile_sycl();

    // Tell cargo to rerun if these files change
    println!("cargo:rerun-if-changed=src/gpu/sycl/gpu_hash.h");
    println!("cargo:rerun-if-changed=src/gpu/sycl/gpu_hash.cpp");
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "gpu-hashing")]
fn compile_sycl() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let sycl_src = manifest_dir.join("src/gpu/sycl");
    let cpp_file = sycl_src.join("gpu_hash.cpp");
    let lib_file = out_dir.join("libgpu_hash.a");
    let obj_file = out_dir.join("gpu_hash.o");

    // Find the SYCL compiler (icpx from Intel oneAPI)
    let compiler = find_sycl_compiler();

    println!("cargo:warning=Using SYCL compiler: {}", compiler);

    // Compile the SYCL code to object file
    let compile_status = Command::new(&compiler)
        .args(&[
            "-fsycl",
            "-c",
            "-fPIC",
            "-O3",
            "-std=c++17",
            "-I",
            sycl_src.to_str().unwrap(),
            "-o",
            obj_file.to_str().unwrap(),
            cpp_file.to_str().unwrap(),
        ])
        .status();

    match compile_status {
        Ok(status) if status.success() => {
            println!("cargo:warning=SYCL compilation successful");
        }
        Ok(status) => {
            // Compilation failed, try to provide helpful error message
            eprintln!("SYCL compilation failed with status: {}", status);
            eprintln!("Make sure Intel oneAPI is installed and sourced:");
            eprintln!("  source /opt/intel/oneapi/setvars.sh");

            // Fall back to stub implementation
            create_stub_library(&out_dir);
            return;
        }
        Err(e) => {
            eprintln!("Failed to run SYCL compiler '{}': {}", compiler, e);
            eprintln!("Make sure Intel oneAPI is installed.");
            eprintln!("On Ubuntu: apt install intel-oneapi-compiler-dpcpp-cpp");
            eprintln!("Then source the environment: source /opt/intel/oneapi/setvars.sh");

            // Fall back to stub implementation
            create_stub_library(&out_dir);
            return;
        }
    }

    // Create static library from object file
    let ar_status = Command::new("ar")
        .args(&[
            "rcs",
            lib_file.to_str().unwrap(),
            obj_file.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run ar");

    if !ar_status.success() {
        panic!("Failed to create static library");
    }

    // Tell cargo where to find the library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=gpu_hash");

    // Link SYCL runtime
    println!("cargo:rustc-link-lib=sycl");
    println!("cargo:rustc-link-lib=stdc++");

    // Link Level Zero if available (for Intel GPUs)
    if has_level_zero() {
        println!("cargo:rustc-link-lib=ze_loader");
    }
}

#[cfg(feature = "gpu-hashing")]
fn find_sycl_compiler() -> String {
    // Check for icpx (Intel oneAPI DPC++ compiler)
    if which_exists("icpx") {
        return "icpx".to_string();
    }

    // Check for dpcpp (older name)
    if which_exists("dpcpp") {
        return "dpcpp".to_string();
    }

    // Check common installation paths
    let common_paths = [
        "/opt/intel/oneapi/compiler/latest/bin/icpx",
        "/opt/intel/oneapi/compiler/latest/linux/bin/icpx",
    ];

    for path in &common_paths {
        if std::path::Path::new(path).exists() {
            return path.to_string();
        }
    }

    // Default to icpx, will fail later with helpful message
    "icpx".to_string()
}

#[cfg(feature = "gpu-hashing")]
fn which_exists(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(feature = "gpu-hashing")]
fn has_level_zero() -> bool {
    // Check if Level Zero is available
    Command::new("pkg-config")
        .args(&["--exists", "level-zero"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(feature = "gpu-hashing")]
fn create_stub_library(out_dir: &PathBuf) {
    // Create a stub C library that returns "not available"
    let stub_code = r#"
#include <stdint.h>
#include <stddef.h>

typedef enum { GPU_HASH_SHA256 = 0, GPU_HASH_SHA384 = 1, GPU_HASH_SHA512 = 2 } GpuHashAlgorithm;
typedef enum { 
    GPU_HASH_SUCCESS = 0, GPU_HASH_ERROR_NO_DEVICE = 1, GPU_HASH_ERROR_INVALID_ALGORITHM = 2,
    GPU_HASH_ERROR_MEMORY_ALLOCATION = 3, GPU_HASH_ERROR_KERNEL_EXECUTION = 4,
    GPU_HASH_ERROR_INVALID_INPUT = 5, GPU_HASH_ERROR_NOT_INITIALIZED = 6, GPU_HASH_ERROR_UNKNOWN = 99
} GpuHashError;
typedef enum { GPU_DEVICE_TYPE_GPU = 0, GPU_DEVICE_TYPE_CPU = 1, GPU_DEVICE_TYPE_ACCELERATOR = 2 } GpuDeviceType;

typedef struct {
    char name[256]; char vendor[256]; char driver_version[64];
    GpuDeviceType device_type; uint32_t max_compute_units;
    uint64_t global_memory_size; uint64_t local_memory_size;
    size_t max_work_group_size; int is_intel; int is_intel_xe;
} GpuDeviceInfo;

typedef void* GpuHashContextHandle;

static const char* last_error = "SYCL not available - Intel oneAPI not installed";

GpuHashError gpu_hash_init(void) { return GPU_HASH_SUCCESS; }
void gpu_hash_cleanup(void) {}
int gpu_hash_is_available(void) { return 0; }
int gpu_hash_get_device_count(void) { return 0; }
GpuHashError gpu_hash_get_device_info(int idx, GpuDeviceInfo* info) { return GPU_HASH_ERROR_NO_DEVICE; }
GpuHashError gpu_hash_create_context(GpuHashAlgorithm alg, int idx, GpuHashContextHandle* h) { return GPU_HASH_ERROR_NO_DEVICE; }
void gpu_hash_destroy_context(GpuHashContextHandle h) {}
GpuHashError gpu_hash_single(GpuHashContextHandle h, const uint8_t* in, size_t len, uint8_t* out, size_t* olen) { return GPU_HASH_ERROR_NO_DEVICE; }
GpuHashError gpu_hash_batch(GpuHashContextHandle h, const uint8_t** ins, const size_t* lens, size_t n, uint8_t** outs, size_t osize) { return GPU_HASH_ERROR_NO_DEVICE; }
GpuHashError gpu_hash_batch_fixed(GpuHashContextHandle h, const uint8_t* in, size_t msize, size_t n, uint8_t* out) { return GPU_HASH_ERROR_NO_DEVICE; }
size_t gpu_hash_output_size(GpuHashAlgorithm alg) { return alg == GPU_HASH_SHA256 ? 32 : alg == GPU_HASH_SHA384 ? 48 : 64; }
const char* gpu_hash_get_last_error(void) { return last_error; }
"#;

    let stub_c = out_dir.join("gpu_hash_stub.c");
    std::fs::write(&stub_c, stub_code).expect("Failed to write stub");

    let obj_file = out_dir.join("gpu_hash_stub.o");
    let lib_file = out_dir.join("libgpu_hash.a");

    // Compile stub
    let status = Command::new("cc")
        .args(&[
            "-c",
            "-fPIC",
            "-o",
            obj_file.to_str().unwrap(),
            stub_c.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to compile stub");

    if !status.success() {
        panic!("Failed to compile stub library");
    }

    // Create library
    Command::new("ar")
        .args(&[
            "rcs",
            lib_file.to_str().unwrap(),
            obj_file.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to create stub library");

    println!("cargo:warning=Created stub GPU library (SYCL not available)");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=gpu_hash");
}
