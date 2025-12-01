# GPU-Accelerated Hashing for Intel Xe GPUs (SYCL/oneAPI)

This module provides GPU-accelerated hashing support for Intel Xe GPUs using SYCL/oneAPI - Intel's modern GPU programming framework.

## Features

- **SHA-256, SHA-384, SHA-512** hash algorithms optimized for Intel GPUs
- **Automatic fallback** to CPU when GPU is unavailable
- **Batch processing** for efficiently hashing multiple messages in parallel
- **File hashing** support for ML model files

## Prerequisites

### Hardware
- Intel GPU with Level Zero support:
  - Intel Arc (discrete GPUs)
  - Intel Iris Xe Graphics
  - Intel UHD Graphics 
  - Other Intel Xe-based GPUs

### Software

#### Linux (Ubuntu/Debian)

1. **Install Intel oneAPI Base Toolkit:**
```bash
# Add Intel repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
    sudo gpg --dearmor -o /usr/share/keyrings/intel-oneapi-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] \
    https://apt.repos.intel.com/oneapi all main" | \
    sudo tee /etc/apt/sources.list.d/intel-oneapi.list

sudo apt update

# Install DPC++ compiler
sudo apt install intel-oneapi-compiler-dpcpp-cpp

# Install Level Zero loader
sudo apt install level-zero
```

2. **Install Intel GPU drivers:**
```bash
# Add Intel graphics repository
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | \
    sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg
    
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
    https://repositories.intel.com/graphics/ubuntu jammy main" | \
    sudo tee /etc/apt/sources.list.d/intel-graphics.list

sudo apt update
sudo apt install intel-opencl-icd intel-level-zero-gpu level-zero
```

3. **Source oneAPI environment before building:**
```bash
source /opt/intel/oneapi/setvars.sh
```

#### Fedora/RHEL

```bash
# Install Intel oneAPI
sudo dnf install intel-oneapi-compiler-dpcpp-cpp

# Install Intel compute runtime  
sudo dnf install intel-compute-runtime level-zero

# Source environment
source /opt/intel/oneapi/setvars.sh
```

## Building

The library must be built with the oneAPI environment sourced:

```bash
# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Build with GPU support
cargo build --features gpu-hashing

# Run tests
cargo test --features gpu-hashing

# Run benchmarks
cargo bench --features gpu-hashing
```

If oneAPI is not available, the build will create a stub library that returns "GPU not available" - the library will still compile and fall back to CPU hashing.

## Usage

### Basic Hashing

```rust
use atlas_c2pa_lib::gpu::{GpuHasher, GpuHashAlgorithm, hash_auto};

// Automatic selection (GPU if available, CPU otherwise)
let hash = hash_auto(b"Hello, World!", GpuHashAlgorithm::Sha256)?;
println!("Hash: {}", hex::encode(&hash));
```

### Using the Hasher Directly

```rust
use atlas_c2pa_lib::gpu::{GpuHasher, GpuHashAlgorithm};

// Create a hasher for SHA-256
let hasher = GpuHasher::new(GpuHashAlgorithm::Sha256)?;

// Hash single messages
let hash1 = hasher.hash(b"Message 1")?;
let hash2 = hasher.hash(b"Message 2")?;
```

### Batch Processing

Batch processing is significantly faster when you have many messages to hash:

```rust
use atlas_c2pa_lib::gpu::{GpuHasher, GpuHashAlgorithm};

let hasher = GpuHasher::new(GpuHashAlgorithm::Sha256)?;

let messages = vec![
    b"Model weights chunk 1".to_vec(),
    b"Model weights chunk 2".to_vec(),
    b"Model weights chunk 3".to_vec(),
    // ... more chunks
];

// Hash all messages in parallel on GPU
let hashes = hasher.hash_batch(&messages)?;
```

### Hashing ML Model Files

```rust
use atlas_c2pa_lib::gpu::{GpuHasher, GpuHashAlgorithm};
use std::path::Path;

let hasher = GpuHasher::new(GpuHashAlgorithm::Sha256)?;

// Hash a single file
let hash = hasher.hash_file(Path::new("model.onnx"))?;

// Hash multiple files in parallel
let hashes = hasher.hash_files(&[
    Path::new("model.onnx"),
    Path::new("weights.bin"),
    Path::new("config.json"),
])?;
```

### Checking GPU Availability

```rust
use atlas_c2pa_lib::gpu::{is_gpu_available, get_available_devices};

if is_gpu_available() {
    println!("GPU hashing is available!");
    
    // List available devices
    for device in get_available_devices()? {
        println!("Device: {} ({})", device.name(), device.vendor());
        println!("  Type: {:?}", device.device_type());
        println!("  Compute Units: {}", device.max_compute_units());
        println!("  Is Intel Xe: {}", device.is_intel_xe());
    }
} else {
    println!("GPU not available, using CPU fallback");
}
```

## Performance

GPU hashing provides the best performance for:

1. **Large files** (> 4KB) - GPU initialization overhead is amortized
2. **Batch processing** - Multiple messages processed in parallel
3. **ML model hashing** - Large model files benefit significantly

For small messages (< 4KB), the library automatically uses CPU hashing.

### Benchmarks

Run benchmarks to measure performance on your hardware:

```bash
source /opt/intel/oneapi/setvars.sh
cargo bench --features gpu-hashing
```

## Supported Algorithms

| Algorithm | Output Size | Block Size | C2PA Compliant |
|-----------|-------------|------------|----------------|
| SHA-256   | 32 bytes    | 64 bytes   | ✓              |
| SHA-384   | 48 bytes    | 128 bytes  | ✓              |
| SHA-512   | 64 bytes    | 128 bytes  | ✓              |

## Troubleshooting

### GPU Not Detected

1. Verify Intel GPU is present:
   ```bash
   lspci | grep -i "vga\|3d\|display"
   ```

2. Check Level Zero is available:
   ```bash
   # List Level Zero devices
   ls /dev/dri/
   
   # Check Level Zero loader
   ldconfig -p | grep ze_loader
   ```

3. Verify oneAPI environment is sourced:
   ```bash
   which icpx
   echo $SYCL_DEVICE_FILTER
   ```

4. List SYCL devices:
   ```bash
   sycl-ls
   ```

### Compilation Errors

If you see "SYCL compiler not found":

1. Install Intel oneAPI:
   ```bash
   sudo apt install intel-oneapi-compiler-dpcpp-cpp
   ```

2. Source the environment:
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```

3. Verify compiler is available:
   ```bash
   which icpx
   icpx --version
   ```

### Performance Issues

1. Ensure release builds:
   ```bash
   cargo build --release --features gpu-hashing
   ```

2. For batch processing, use batch sizes > 4 messages

3. For large files, ensure data fits in GPU memory

## Architecture

The GPU hashing module consists of:

1. **SYCL Kernels** (`src/gpu/sycl/`):
   - `gpu_hash.cpp` - SHA-256/384/512 SYCL implementation
   - `gpu_hash.h` - C API header

2. **Rust FFI** (`src/gpu/`):
   - `mod.rs` - Public API
   - `ffi.rs` - C bindings
   - `hasher.rs` - Hash computation
   - `context.rs` - Device management
   - `error.rs` - Error types

3. **Build System** (`build.rs`):
   - Compiles SYCL code using icpx
   - Creates stub library if oneAPI not available

## License

MIT OR Apache-2.0
