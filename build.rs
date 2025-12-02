//! Build script for atlas-c2pa-lib

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/gpu/sycl/gpu_hash.h");
    println!("cargo:rerun-if-changed=src/gpu/sycl/gpu_hash.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    // IMPORTANT: build script *does* see crate features
    #[cfg(feature = "gpu-hashing")]
    compile_sycl();

    // Without gpu-hashing, we build nothing and the pure-CPU
    // code in Rust should handle things.
}

#[cfg(feature = "gpu-hashing")]
fn compile_sycl() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let sycl_src = manifest_dir.join("src/gpu/sycl");
    let cpp_file = sycl_src.join("gpu_hash.cpp");
    let lib_file = out_dir.join("libgpu_hash.so");

    let compiler = find_sycl_compiler();
    println!("cargo:warning=Using SYCL compiler: {}", compiler);

    // One step: compile + link as shared library with device image
    let output = Command::new(&compiler)
        .args(&[
            "-fsycl",
            "-fPIC",
            "-O3",
            "-std=c++17",
            "-fno-builtin",
            "-shared",
            "-I",
            sycl_src.to_str().unwrap(),
            cpp_file.to_str().unwrap(),
            "-o",
            lib_file.to_str().unwrap(),
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            println!("cargo:warning=SYCL shared library built successfully");
        }
        Ok(out) => {
            println!("cargo:warning=SYCL compilation/link failed:");
            println!(
                "cargo:warning=stderr: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            create_stub_library(&out_dir);
            return;
        }
        Err(e) => {
            println!("cargo:warning=Failed to run SYCL compiler: {}", e);
            create_stub_library(&out_dir);
            return;
        }
    }

    // Tell Rust to link against the shared lib
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=gpu_hash");

    // These are usually resolved transitively, but leaving them explicit is fine
    println!("cargo:rustc-link-lib=sycl");
    println!("cargo:rustc-link-lib=stdc++");
}

#[cfg(feature = "gpu-hashing")]
fn find_sycl_compiler() -> String {
    if Command::new("icpx")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return "icpx".to_string();
    }
    for path in &["/opt/intel/oneapi/compiler/latest/bin/icpx"] {
        if std::path::Path::new(path).exists() {
            return path.to_string();
        }
    }
    "icpx".to_string()
}

#[cfg(feature = "gpu-hashing")]
fn create_stub_library(out_dir: &PathBuf) {
    println!("cargo:warning=Creating stub library (no SYCL)");

    let stub_code = r#"
#include <stdint.h>
#include <stddef.h>
typedef enum { GPU_HASH_SHA256 = 0, GPU_HASH_SHA384 = 1, GPU_HASH_SHA512 = 2 } GpuHashAlgorithm;
typedef enum { GPU_HASH_SUCCESS = 0, GPU_HASH_ERROR_NO_DEVICE = 1, GPU_HASH_ERROR_INVALID_ALGORITHM = 2,
    GPU_HASH_ERROR_MEMORY_ALLOCATION = 3, GPU_HASH_ERROR_KERNEL_EXECUTION = 4,
    GPU_HASH_ERROR_INVALID_INPUT = 5, GPU_HASH_ERROR_NOT_INITIALIZED = 6, GPU_HASH_ERROR_UNKNOWN = 99 } GpuHashError;
typedef enum { GPU_DEVICE_TYPE_GPU = 0, GPU_DEVICE_TYPE_CPU = 1, GPU_DEVICE_TYPE_ACCELERATOR = 2 } GpuDeviceType;
typedef struct { char name[256]; char vendor[256]; char driver_version[64]; GpuDeviceType device_type;
    uint32_t max_compute_units; uint64_t global_memory_size; uint64_t local_memory_size;
    size_t max_work_group_size; int is_intel; int is_intel_xe; } GpuDeviceInfo;
typedef void* GpuHashContextHandle;
static const char* last_error = "SYCL not available";
GpuHashError gpu_hash_init(void) { return GPU_HASH_SUCCESS; }
void gpu_hash_cleanup(void) {}
int gpu_hash_is_available(void) { return 0; }
int gpu_hash_get_device_count(void) { return 0; }
GpuHashError gpu_hash_get_device_info(int idx, GpuDeviceInfo* info) { (void)idx; (void)info; return GPU_HASH_ERROR_NO_DEVICE; }
GpuHashError gpu_hash_create_context(GpuHashAlgorithm alg, int idx, GpuHashContextHandle* h) { (void)alg; (void)idx; (void)h; return GPU_HASH_ERROR_NO_DEVICE; }
void gpu_hash_destroy_context(GpuHashContextHandle h) { (void)h; }
GpuHashError gpu_hash_single(GpuHashContextHandle h, const uint8_t* in, size_t len, uint8_t* out, size_t* olen) { (void)h; (void)in; (void)len; (void)out; (void)olen; return GPU_HASH_ERROR_NO_DEVICE; }
GpuHashError gpu_hash_batch(GpuHashContextHandle h, const uint8_t** ins, const size_t* lens, size_t n, uint8_t** outs, size_t osize) { (void)h; (void)ins; (void)lens; (void)n; (void)outs; (void)osize; return GPU_HASH_ERROR_NO_DEVICE; }
GpuHashError gpu_hash_batch_fixed(GpuHashContextHandle h, const uint8_t* in, size_t msize, size_t n, uint8_t* out) { (void)h; (void)in; (void)msize; (void)n; (void)out; return GPU_HASH_ERROR_NO_DEVICE; }
size_t gpu_hash_output_size(GpuHashAlgorithm alg) { return alg == GPU_HASH_SHA256 ? 32 : alg == GPU_HASH_SHA384 ? 48 : 64; }
const char* gpu_hash_get_last_error(void) { return last_error; }
"#;

    let stub_c = out_dir.join("gpu_hash_stub.c");
    std::fs::write(&stub_c, stub_code).expect("Failed to write stub");
    let obj_file = out_dir.join("gpu_hash_stub.o");
    let lib_file = out_dir.join("libgpu_hash.so");

    Command::new("cc")
        .args(&[
            "-c",
            "-fPIC",
            "-o",
            obj_file.to_str().unwrap(),
            stub_c.to_str().unwrap(),
        ])
        .status()
        .expect("cc failed");

    Command::new("cc")
        .args(&[
            "-shared",
            "-o",
            lib_file.to_str().unwrap(),
            obj_file.to_str().unwrap(),
        ])
        .status()
        .expect("cc failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=gpu_hash");
}
