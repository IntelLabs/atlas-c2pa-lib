/**
 * @file gpu_hash.h
 * @brief C API for SYCL-based GPU hashing on Intel Xe GPUs
 *
 * This header provides a C-compatible interface for GPU-accelerated
 * SHA-256, SHA-384, and SHA-512 hashing using Intel SYCL/oneAPI.
 *
 * Compile with: icpx -fsycl -shared -fPIC -o libgpu_hash.so gpu_hash.cpp
 */

#ifndef GPU_HASH_H
#define GPU_HASH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Hash algorithm identifiers
 */
typedef enum {
  GPU_HASH_SHA256 = 0,
  GPU_HASH_SHA384 = 1,
  GPU_HASH_SHA512 = 2
} GpuHashAlgorithm;

/**
 * Error codes
 */
typedef enum {
  GPU_HASH_SUCCESS = 0,
  GPU_HASH_ERROR_NO_DEVICE = 1,
  GPU_HASH_ERROR_INVALID_ALGORITHM = 2,
  GPU_HASH_ERROR_MEMORY_ALLOCATION = 3,
  GPU_HASH_ERROR_KERNEL_EXECUTION = 4,
  GPU_HASH_ERROR_INVALID_INPUT = 5,
  GPU_HASH_ERROR_NOT_INITIALIZED = 6,
  GPU_HASH_ERROR_UNKNOWN = 99
} GpuHashError;

/**
 * Device type
 */
typedef enum {
  GPU_DEVICE_TYPE_GPU = 0,
  GPU_DEVICE_TYPE_CPU = 1,
  GPU_DEVICE_TYPE_ACCELERATOR = 2
} GpuDeviceType;

/**
 * Device information structure
 */
typedef struct {
  char name[256];
  char vendor[256];
  char driver_version[64];
  GpuDeviceType device_type;
  uint32_t max_compute_units;
  uint64_t global_memory_size;
  uint64_t local_memory_size;
  size_t max_work_group_size;
  int is_intel;
  int is_intel_xe;
} GpuDeviceInfo;

/**
 * Opaque handle to GPU hasher context
 */
typedef struct GpuHashContext *GpuHashContextHandle;

/**
 * Initialize the GPU hashing library
 * Must be called before any other functions
 *
 * @return GPU_HASH_SUCCESS on success, error code otherwise
 */
GpuHashError gpu_hash_init(void);

/**
 * Cleanup the GPU hashing library
 * Should be called when done using the library
 */
void gpu_hash_cleanup(void);

/**
 * Check if GPU hashing is available
 *
 * @return 1 if available, 0 otherwise
 */
int gpu_hash_is_available(void);

/**
 * Get the number of available GPU devices
 *
 * @return Number of devices
 */
int gpu_hash_get_device_count(void);

/**
 * Get information about a specific device
 *
 * @param device_index Index of the device (0-based)
 * @param info Pointer to device info structure to fill
 * @return GPU_HASH_SUCCESS on success, error code otherwise
 */
GpuHashError gpu_hash_get_device_info(int device_index, GpuDeviceInfo *info);

/**
 * Create a new hash context for a specific algorithm
 *
 * @param algorithm The hash algorithm to use
 * @param device_index Device to use (-1 for auto-select best Intel GPU)
 * @param handle Pointer to receive the context handle
 * @return GPU_HASH_SUCCESS on success, error code otherwise
 */
GpuHashError gpu_hash_create_context(GpuHashAlgorithm algorithm,
                                     int device_index,
                                     GpuHashContextHandle *handle);

/**
 * Destroy a hash context
 *
 * @param handle The context handle to destroy
 */
void gpu_hash_destroy_context(GpuHashContextHandle handle);

/**
 * Hash a single message
 *
 * @param handle The hash context
 * @param input Input data
 * @param input_len Length of input data
 * @param output Output buffer (must be large enough for the hash)
 * @param output_len Pointer to receive actual output length
 * @return GPU_HASH_SUCCESS on success, error code otherwise
 */
GpuHashError gpu_hash_single(GpuHashContextHandle handle, const uint8_t *input,
                             size_t input_len, uint8_t *output,
                             size_t *output_len);

/**
 * Hash multiple messages in parallel (batch processing)
 *
 * @param handle The hash context
 * @param inputs Array of input data pointers
 * @param input_lens Array of input lengths
 * @param num_inputs Number of inputs
 * @param outputs Array of output buffers
 * @param output_size Size of each output buffer
 * @return GPU_HASH_SUCCESS on success, error code otherwise
 */
GpuHashError gpu_hash_batch(GpuHashContextHandle handle, const uint8_t **inputs,
                            const size_t *input_lens, size_t num_inputs,
                            uint8_t **outputs, size_t output_size);

/**
 * Hash contiguous data with multiple messages of same size
 * More efficient than gpu_hash_batch for fixed-size messages
 *
 * @param handle The hash context
 * @param input Contiguous input data
 * @param message_size Size of each message
 * @param num_messages Number of messages
 * @param output Output buffer (num_messages * hash_size bytes)
 * @return GPU_HASH_SUCCESS on success, error code otherwise
 */
GpuHashError gpu_hash_batch_fixed(GpuHashContextHandle handle,
                                  const uint8_t *input, size_t message_size,
                                  size_t num_messages, uint8_t *output);

/**
 * Get the output size for a hash algorithm
 *
 * @param algorithm The hash algorithm
 * @return Output size in bytes (32 for SHA-256, 48 for SHA-384, 64 for SHA-512)
 */
size_t gpu_hash_output_size(GpuHashAlgorithm algorithm);

/**
 * Get the last error message
 *
 * @return Pointer to error message string (do not free)
 */
const char *gpu_hash_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* GPU_HASH_H */
