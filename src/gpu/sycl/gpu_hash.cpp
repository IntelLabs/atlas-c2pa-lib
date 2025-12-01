/**
 * @file gpu_hash.cpp
 * @brief SYCL implementation of GPU-accelerated hashing for Intel Xe GPUs
 * 
 * This file implements SHA-256, SHA-384, and SHA-512 hashing using Intel SYCL.
 * Optimized for Intel Arc, Iris Xe, and UHD Graphics.
 * 
 * Compile with:
 *   icpx -fsycl -shared -fPIC -O3 -o libgpu_hash.so gpu_hash.cpp
 * 
 * Or for specific Intel GPU targets:
 *   icpx -fsycl -fsycl-targets=intel_gpu_pvc,intel_gpu_dg2 -shared -fPIC -O3 -o libgpu_hash.so gpu_hash.cpp
 */

#include "gpu_hash.h"
#include <sycl/sycl.hpp>
#include <vector>
#include <memory>
#include <mutex>
#include <cstring>
#include <algorithm>

// ============================================================================
// SHA-256 Constants and Functions
// ============================================================================

namespace {

// SHA-256 round constants
constexpr uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 initial hash values
constexpr uint32_t H256_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// SHA-512 round constants
constexpr uint64_t K512[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

// SHA-512 initial values
constexpr uint64_t H512_INIT[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

// SHA-384 initial values
constexpr uint64_t H384_INIT[8] = {
    0xcbbb9d5dc1059ed8ULL, 0x629a292a367cd507ULL, 0x9159015a3070dd17ULL, 0x152fecd8f70e5939ULL,
    0x67332667ffc00b31ULL, 0x8eb44a8768581511ULL, 0xdb0c2e0d64f98fa7ULL, 0x47b5481dbefa4fa4ULL
};

// Bitwise operations
inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
inline uint64_t rotr64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

// SHA-256 functions
inline uint32_t ch32(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
inline uint32_t maj32(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint32_t sigma0_256(uint32_t x) { return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22); }
inline uint32_t sigma1_256(uint32_t x) { return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25); }
inline uint32_t gamma0_256(uint32_t x) { return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3); }
inline uint32_t gamma1_256(uint32_t x) { return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10); }

// SHA-512 functions
inline uint64_t ch64(uint64_t x, uint64_t y, uint64_t z) { return (x & y) ^ (~x & z); }
inline uint64_t maj64(uint64_t x, uint64_t y, uint64_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint64_t sigma0_512(uint64_t x) { return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39); }
inline uint64_t sigma1_512(uint64_t x) { return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41); }
inline uint64_t gamma0_512(uint64_t x) { return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7); }
inline uint64_t gamma1_512(uint64_t x) { return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6); }

// Thread-local error message
thread_local std::string g_last_error;

// Global state
std::mutex g_init_mutex;
bool g_initialized = false;
std::vector<sycl::device> g_devices;

} // anonymous namespace

// ============================================================================
// Hash Context Structure
// ============================================================================

struct GpuHashContext {
    GpuHashAlgorithm algorithm;
    sycl::queue queue;
    size_t output_size;
    
    GpuHashContext(GpuHashAlgorithm alg, sycl::queue q)
        : algorithm(alg), queue(std::move(q)) {
        switch (alg) {
            case GPU_HASH_SHA256: output_size = 32; break;
            case GPU_HASH_SHA384: output_size = 48; break;
            case GPU_HASH_SHA512: output_size = 64; break;
            default: output_size = 32;
        }
    }
};

// ============================================================================
// SYCL Kernels
// ============================================================================

// SHA-256 kernel for batch processing
class Sha256BatchKernel;

void sha256_hash_batch(
    sycl::queue& q,
    const uint8_t* input,
    const uint64_t* offsets,
    const uint64_t* lengths,
    uint8_t* output,
    size_t num_messages
) {
    // Copy constants to device
    sycl::buffer<uint32_t, 1> k_buf(K256, sycl::range<1>(64));
    sycl::buffer<uint32_t, 1> h_buf(H256_INIT, sycl::range<1>(8));
    
    q.submit([&](sycl::handler& h) {
        auto k_acc = k_buf.get_access<sycl::access::mode::read>(h);
        auto h_acc = h_buf.get_access<sycl::access::mode::read>(h);
        
        h.parallel_for<Sha256BatchKernel>(
            sycl::range<1>(num_messages),
            [=](sycl::id<1> idx) {
                size_t msg_idx = idx[0];
                uint64_t msg_offset = offsets[msg_idx];
                uint64_t msg_len = lengths[msg_idx];
                
                // Initialize state
                uint32_t state[8];
                for (int i = 0; i < 8; i++) {
                    state[i] = h_acc[i];
                }
                
                // Process full blocks
                uint64_t num_blocks = msg_len / 64;
                for (uint64_t b = 0; b < num_blocks; b++) {
                    uint32_t W[64];
                    
                    // Load block
                    for (int i = 0; i < 16; i++) {
                        uint64_t pos = msg_offset + b * 64 + i * 4;
                        W[i] = ((uint32_t)input[pos] << 24) |
                               ((uint32_t)input[pos + 1] << 16) |
                               ((uint32_t)input[pos + 2] << 8) |
                               (uint32_t)input[pos + 3];
                    }
                    
                    // Extend
                    for (int i = 16; i < 64; i++) {
                        W[i] = gamma1_256(W[i-2]) + W[i-7] + gamma0_256(W[i-15]) + W[i-16];
                    }
                    
                    // Compress
                    uint32_t a = state[0], bb = state[1], c = state[2], d = state[3];
                    uint32_t e = state[4], f = state[5], g = state[6], hh = state[7];
                    
                    for (int i = 0; i < 64; i++) {
                        uint32_t T1 = hh + sigma1_256(e) + ch32(e, f, g) + k_acc[i] + W[i];
                        uint32_t T2 = sigma0_256(a) + maj32(a, bb, c);
                        hh = g; g = f; f = e; e = d + T1;
                        d = c; c = bb; bb = a; a = T1 + T2;
                    }
                    
                    state[0] += a; state[1] += bb; state[2] += c; state[3] += d;
                    state[4] += e; state[5] += f; state[6] += g; state[7] += hh;
                }
                
                // Handle padding
                uint8_t padded[128] = {0};
                uint32_t remaining = msg_len % 64;
                uint32_t pad_blocks = (remaining < 56) ? 1 : 2;
                
                // Copy remaining bytes
                for (uint32_t i = 0; i < remaining; i++) {
                    padded[i] = input[msg_offset + num_blocks * 64 + i];
                }
                padded[remaining] = 0x80;
                
                // Add length
                uint64_t bit_len = msg_len * 8;
                uint32_t len_offset = (pad_blocks == 1) ? 56 : 120;
                padded[len_offset] = (bit_len >> 56) & 0xff;
                padded[len_offset + 1] = (bit_len >> 48) & 0xff;
                padded[len_offset + 2] = (bit_len >> 40) & 0xff;
                padded[len_offset + 3] = (bit_len >> 32) & 0xff;
                padded[len_offset + 4] = (bit_len >> 24) & 0xff;
                padded[len_offset + 5] = (bit_len >> 16) & 0xff;
                padded[len_offset + 6] = (bit_len >> 8) & 0xff;
                padded[len_offset + 7] = bit_len & 0xff;
                
                // Process padding blocks
                for (uint32_t p = 0; p < pad_blocks; p++) {
                    uint32_t W[64];
                    for (int i = 0; i < 16; i++) {
                        W[i] = ((uint32_t)padded[p * 64 + i * 4] << 24) |
                               ((uint32_t)padded[p * 64 + i * 4 + 1] << 16) |
                               ((uint32_t)padded[p * 64 + i * 4 + 2] << 8) |
                               (uint32_t)padded[p * 64 + i * 4 + 3];
                    }
                    
                    for (int i = 16; i < 64; i++) {
                        W[i] = gamma1_256(W[i-2]) + W[i-7] + gamma0_256(W[i-15]) + W[i-16];
                    }
                    
                    uint32_t a = state[0], bb = state[1], c = state[2], d = state[3];
                    uint32_t e = state[4], f = state[5], g = state[6], hh = state[7];
                    
                    for (int i = 0; i < 64; i++) {
                        uint32_t T1 = hh + sigma1_256(e) + ch32(e, f, g) + k_acc[i] + W[i];
                        uint32_t T2 = sigma0_256(a) + maj32(a, bb, c);
                        hh = g; g = f; f = e; e = d + T1;
                        d = c; c = bb; bb = a; a = T1 + T2;
                    }
                    
                    state[0] += a; state[1] += bb; state[2] += c; state[3] += d;
                    state[4] += e; state[5] += f; state[6] += g; state[7] += hh;
                }
                
                // Write output
                size_t out_offset = msg_idx * 32;
                for (int i = 0; i < 8; i++) {
                    output[out_offset + i * 4] = (state[i] >> 24) & 0xff;
                    output[out_offset + i * 4 + 1] = (state[i] >> 16) & 0xff;
                    output[out_offset + i * 4 + 2] = (state[i] >> 8) & 0xff;
                    output[out_offset + i * 4 + 3] = state[i] & 0xff;
                }
            }
        );
    }).wait();
}

// SHA-512/384 kernel
class Sha512BatchKernel;

void sha512_hash_batch(
    sycl::queue& q,
    const uint8_t* input,
    const uint64_t* offsets,
    const uint64_t* lengths,
    uint8_t* output,
    size_t num_messages,
    bool is_sha384
) {
    sycl::buffer<uint64_t, 1> k_buf(K512, sycl::range<1>(80));
    sycl::buffer<uint64_t, 1> h_buf(is_sha384 ? H384_INIT : H512_INIT, sycl::range<1>(8));
    
    size_t output_words = is_sha384 ? 6 : 8;
    size_t output_bytes = is_sha384 ? 48 : 64;
    
    q.submit([&](sycl::handler& h) {
        auto k_acc = k_buf.get_access<sycl::access::mode::read>(h);
        auto h_acc = h_buf.get_access<sycl::access::mode::read>(h);
        
        h.parallel_for<Sha512BatchKernel>(
            sycl::range<1>(num_messages),
            [=](sycl::id<1> idx) {
                size_t msg_idx = idx[0];
                uint64_t msg_offset = offsets[msg_idx];
                uint64_t msg_len = lengths[msg_idx];
                
                // Initialize state
                uint64_t state[8];
                for (int i = 0; i < 8; i++) {
                    state[i] = h_acc[i];
                }
                
                // Process full 128-byte blocks
                uint64_t num_blocks = msg_len / 128;
                for (uint64_t b = 0; b < num_blocks; b++) {
                    uint64_t W[80];
                    
                    for (int i = 0; i < 16; i++) {
                        uint64_t pos = msg_offset + b * 128 + i * 8;
                        W[i] = ((uint64_t)input[pos] << 56) |
                               ((uint64_t)input[pos + 1] << 48) |
                               ((uint64_t)input[pos + 2] << 40) |
                               ((uint64_t)input[pos + 3] << 32) |
                               ((uint64_t)input[pos + 4] << 24) |
                               ((uint64_t)input[pos + 5] << 16) |
                               ((uint64_t)input[pos + 6] << 8) |
                               (uint64_t)input[pos + 7];
                    }
                    
                    for (int i = 16; i < 80; i++) {
                        W[i] = gamma1_512(W[i-2]) + W[i-7] + gamma0_512(W[i-15]) + W[i-16];
                    }
                    
                    uint64_t a = state[0], bb = state[1], c = state[2], d = state[3];
                    uint64_t e = state[4], f = state[5], g = state[6], hh = state[7];
                    
                    for (int i = 0; i < 80; i++) {
                        uint64_t T1 = hh + sigma1_512(e) + ch64(e, f, g) + k_acc[i] + W[i];
                        uint64_t T2 = sigma0_512(a) + maj64(a, bb, c);
                        hh = g; g = f; f = e; e = d + T1;
                        d = c; c = bb; bb = a; a = T1 + T2;
                    }
                    
                    state[0] += a; state[1] += bb; state[2] += c; state[3] += d;
                    state[4] += e; state[5] += f; state[6] += g; state[7] += hh;
                }
                
                // Handle padding
                uint8_t padded[256] = {0};
                uint64_t remaining = msg_len % 128;
                uint32_t pad_blocks = (remaining < 112) ? 1 : 2;
                
                for (uint64_t i = 0; i < remaining; i++) {
                    padded[i] = input[msg_offset + num_blocks * 128 + i];
                }
                padded[remaining] = 0x80;
                
                // Add 128-bit length (we only use lower 64 bits)
                uint64_t bit_len = msg_len * 8;
                uint32_t len_offset = (pad_blocks == 1) ? 112 : 240;
                // Upper 64 bits = 0
                for (int i = 0; i < 8; i++) padded[len_offset + i] = 0;
                // Lower 64 bits
                padded[len_offset + 8] = (bit_len >> 56) & 0xff;
                padded[len_offset + 9] = (bit_len >> 48) & 0xff;
                padded[len_offset + 10] = (bit_len >> 40) & 0xff;
                padded[len_offset + 11] = (bit_len >> 32) & 0xff;
                padded[len_offset + 12] = (bit_len >> 24) & 0xff;
                padded[len_offset + 13] = (bit_len >> 16) & 0xff;
                padded[len_offset + 14] = (bit_len >> 8) & 0xff;
                padded[len_offset + 15] = bit_len & 0xff;
                
                for (uint32_t p = 0; p < pad_blocks; p++) {
                    uint64_t W[80];
                    for (int i = 0; i < 16; i++) {
                        W[i] = ((uint64_t)padded[p * 128 + i * 8] << 56) |
                               ((uint64_t)padded[p * 128 + i * 8 + 1] << 48) |
                               ((uint64_t)padded[p * 128 + i * 8 + 2] << 40) |
                               ((uint64_t)padded[p * 128 + i * 8 + 3] << 32) |
                               ((uint64_t)padded[p * 128 + i * 8 + 4] << 24) |
                               ((uint64_t)padded[p * 128 + i * 8 + 5] << 16) |
                               ((uint64_t)padded[p * 128 + i * 8 + 6] << 8) |
                               (uint64_t)padded[p * 128 + i * 8 + 7];
                    }
                    
                    for (int i = 16; i < 80; i++) {
                        W[i] = gamma1_512(W[i-2]) + W[i-7] + gamma0_512(W[i-15]) + W[i-16];
                    }
                    
                    uint64_t a = state[0], bb = state[1], c = state[2], d = state[3];
                    uint64_t e = state[4], f = state[5], g = state[6], hh = state[7];
                    
                    for (int i = 0; i < 80; i++) {
                        uint64_t T1 = hh + sigma1_512(e) + ch64(e, f, g) + k_acc[i] + W[i];
                        uint64_t T2 = sigma0_512(a) + maj64(a, bb, c);
                        hh = g; g = f; f = e; e = d + T1;
                        d = c; c = bb; bb = a; a = T1 + T2;
                    }
                    
                    state[0] += a; state[1] += bb; state[2] += c; state[3] += d;
                    state[4] += e; state[5] += f; state[6] += g; state[7] += hh;
                }
                
                // Write output
                size_t out_offset = msg_idx * output_bytes;
                for (size_t i = 0; i < output_words; i++) {
                    output[out_offset + i * 8] = (state[i] >> 56) & 0xff;
                    output[out_offset + i * 8 + 1] = (state[i] >> 48) & 0xff;
                    output[out_offset + i * 8 + 2] = (state[i] >> 40) & 0xff;
                    output[out_offset + i * 8 + 3] = (state[i] >> 32) & 0xff;
                    output[out_offset + i * 8 + 4] = (state[i] >> 24) & 0xff;
                    output[out_offset + i * 8 + 5] = (state[i] >> 16) & 0xff;
                    output[out_offset + i * 8 + 6] = (state[i] >> 8) & 0xff;
                    output[out_offset + i * 8 + 7] = state[i] & 0xff;
                }
            }
        );
    }).wait();
}

// ============================================================================
// C API Implementation
// ============================================================================

unsafe extern "C" {

GpuHashError gpu_hash_init(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    
    if (g_initialized) {
        return GPU_HASH_SUCCESS;
    }
    
    try {
        // Enumerate all GPU devices
        auto platforms = sycl::platform::get_platforms();
        for (const auto& platform : platforms) {
            auto devices = platform.get_devices(sycl::info::device_type::gpu);
            for (const auto& device : devices) {
                g_devices.push_back(device);
            }
        }
        
        g_initialized = true;
        return GPU_HASH_SUCCESS;
    } catch (const sycl::exception& e) {
        g_last_error = std::string("SYCL initialization failed: ") + e.what();
        return GPU_HASH_ERROR_NO_DEVICE;
    } catch (const std::exception& e) {
        g_last_error = std::string("Initialization failed: ") + e.what();
        return GPU_HASH_ERROR_UNKNOWN;
    }
}

void gpu_hash_cleanup(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    g_devices.clear();
    g_initialized = false;
}

int gpu_hash_is_available(void) {
    if (!g_initialized) {
        if (gpu_hash_init() != GPU_HASH_SUCCESS) {
            return 0;
        }
    }
    
    // Check for Intel GPUs
    for (const auto& device : g_devices) {
        std::string vendor = device.get_info<sycl::info::device::vendor>();
        std::transform(vendor.begin(), vendor.end(), vendor.begin(), ::tolower);
        if (vendor.find("intel") != std::string::npos) {
            return 1;
        }
    }
    
    return 0;
}

int gpu_hash_get_device_count(void) {
    if (!g_initialized) {
        gpu_hash_init();
    }
    return static_cast<int>(g_devices.size());
}

GpuHashError gpu_hash_get_device_info(int device_index, GpuDeviceInfo* info) {
    if (!g_initialized) {
        gpu_hash_init();
    }
    
    if (device_index < 0 || device_index >= static_cast<int>(g_devices.size())) {
        g_last_error = "Invalid device index";
        return GPU_HASH_ERROR_INVALID_INPUT;
    }
    
    if (!info) {
        g_last_error = "Null info pointer";
        return GPU_HASH_ERROR_INVALID_INPUT;
    }
    
    try {
        const auto& device = g_devices[device_index];
        
        std::string name = device.get_info<sycl::info::device::name>();
        std::string vendor = device.get_info<sycl::info::device::vendor>();
        std::string driver = device.get_info<sycl::info::device::driver_version>();
        
        strncpy(info->name, name.c_str(), sizeof(info->name) - 1);
        strncpy(info->vendor, vendor.c_str(), sizeof(info->vendor) - 1);
        strncpy(info->driver_version, driver.c_str(), sizeof(info->driver_version) - 1);
        
        info->device_type = GPU_DEVICE_TYPE_GPU;
        info->max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
        info->global_memory_size = device.get_info<sycl::info::device::global_mem_size>();
        info->local_memory_size = device.get_info<sycl::info::device::local_mem_size>();
        info->max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
        
        std::string vendor_lower = vendor;
        std::transform(vendor_lower.begin(), vendor_lower.end(), vendor_lower.begin(), ::tolower);
        info->is_intel = (vendor_lower.find("intel") != std::string::npos) ? 1 : 0;
        
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        info->is_intel_xe = (info->is_intel && 
            (name_lower.find("arc") != std::string::npos ||
             name_lower.find("xe") != std::string::npos ||
             name_lower.find("iris") != std::string::npos ||
             name_lower.find("uhd") != std::string::npos)) ? 1 : 0;
        
        return GPU_HASH_SUCCESS;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return GPU_HASH_ERROR_UNKNOWN;
    }
}

GpuHashError gpu_hash_create_context(
    GpuHashAlgorithm algorithm,
    int device_index,
    GpuHashContextHandle* handle
) {
    if (!g_initialized) {
        GpuHashError err = gpu_hash_init();
        if (err != GPU_HASH_SUCCESS) return err;
    }
    
    if (!handle) {
        g_last_error = "Null handle pointer";
        return GPU_HASH_ERROR_INVALID_INPUT;
    }
    
    if (algorithm < GPU_HASH_SHA256 || algorithm > GPU_HASH_SHA512) {
        g_last_error = "Invalid algorithm";
        return GPU_HASH_ERROR_INVALID_ALGORITHM;
    }
    
    try {
        sycl::device selected_device;
        
        if (device_index >= 0) {
            if (device_index >= static_cast<int>(g_devices.size())) {
                g_last_error = "Invalid device index";
                return GPU_HASH_ERROR_INVALID_INPUT;
            }
            selected_device = g_devices[device_index];
        } else {
            // Auto-select best Intel GPU
            bool found = false;
            for (const auto& device : g_devices) {
                std::string vendor = device.get_info<sycl::info::device::vendor>();
                std::transform(vendor.begin(), vendor.end(), vendor.begin(), ::tolower);
                if (vendor.find("intel") != std::string::npos) {
                    selected_device = device;
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                if (g_devices.empty()) {
                    g_last_error = "No GPU devices found";
                    return GPU_HASH_ERROR_NO_DEVICE;
                }
                selected_device = g_devices[0];
            }
        }
        
        sycl::queue q(selected_device, sycl::property::queue::in_order());
        *handle = new GpuHashContext(algorithm, std::move(q));
        
        return GPU_HASH_SUCCESS;
    } catch (const sycl::exception& e) {
        g_last_error = std::string("SYCL error: ") + e.what();
        return GPU_HASH_ERROR_KERNEL_EXECUTION;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return GPU_HASH_ERROR_UNKNOWN;
    }
}

void gpu_hash_destroy_context(GpuHashContextHandle handle) {
    delete handle;
}

GpuHashError gpu_hash_single(
    GpuHashContextHandle handle,
    const uint8_t* input,
    size_t input_len,
    uint8_t* output,
    size_t* output_len
) {
    if (!handle) {
        g_last_error = "Invalid context handle";
        return GPU_HASH_ERROR_NOT_INITIALIZED;
    }
    
    if (!output) {
        g_last_error = "Null output buffer";
        return GPU_HASH_ERROR_INVALID_INPUT;
    }
    
    try {
        // Allocate device memory
        uint8_t* d_input = sycl::malloc_device<uint8_t>(std::max(input_len, (size_t)1), handle->queue);
        uint64_t* d_offsets = sycl::malloc_device<uint64_t>(1, handle->queue);
        uint64_t* d_lengths = sycl::malloc_device<uint64_t>(1, handle->queue);
        uint8_t* d_output = sycl::malloc_device<uint8_t>(handle->output_size, handle->queue);
        
        if (!d_input || !d_offsets || !d_lengths || !d_output) {
            sycl::free(d_input, handle->queue);
            sycl::free(d_offsets, handle->queue);
            sycl::free(d_lengths, handle->queue);
            sycl::free(d_output, handle->queue);
            g_last_error = "Failed to allocate device memory";
            return GPU_HASH_ERROR_MEMORY_ALLOCATION;
        }
        
        // Copy data to device
        uint64_t offset = 0;
        uint64_t length = input_len;
        
        if (input_len > 0) {
            handle->queue.memcpy(d_input, input, input_len);
        }
        handle->queue.memcpy(d_offsets, &offset, sizeof(uint64_t));
        handle->queue.memcpy(d_lengths, &length, sizeof(uint64_t));
        handle->queue.wait();
        
        // Execute kernel
        switch (handle->algorithm) {
            case GPU_HASH_SHA256:
                sha256_hash_batch(handle->queue, d_input, d_offsets, d_lengths, d_output, 1);
                break;
            case GPU_HASH_SHA384:
                sha512_hash_batch(handle->queue, d_input, d_offsets, d_lengths, d_output, 1, true);
                break;
            case GPU_HASH_SHA512:
                sha512_hash_batch(handle->queue, d_input, d_offsets, d_lengths, d_output, 1, false);
                break;
        }
        
        // Copy result back
        handle->queue.memcpy(output, d_output, handle->output_size).wait();
        
        if (output_len) {
            *output_len = handle->output_size;
        }
        
        // Cleanup
        sycl::free(d_input, handle->queue);
        sycl::free(d_offsets, handle->queue);
        sycl::free(d_lengths, handle->queue);
        sycl::free(d_output, handle->queue);
        
        return GPU_HASH_SUCCESS;
    } catch (const sycl::exception& e) {
        g_last_error = std::string("SYCL error: ") + e.what();
        return GPU_HASH_ERROR_KERNEL_EXECUTION;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return GPU_HASH_ERROR_UNKNOWN;
    }
}

GpuHashError gpu_hash_batch(
    GpuHashContextHandle handle,
    const uint8_t** inputs,
    const size_t* input_lens,
    size_t num_inputs,
    uint8_t** outputs,
    size_t output_size
) {
    if (!handle || !inputs || !input_lens || !outputs) {
        g_last_error = "Invalid parameters";
        return GPU_HASH_ERROR_INVALID_INPUT;
    }
    
    if (num_inputs == 0) {
        return GPU_HASH_SUCCESS;
    }
    
    try {
        // Calculate total size and build offset array
        std::vector<uint64_t> offsets(num_inputs);
        std::vector<uint64_t> lengths(num_inputs);
        size_t total_size = 0;
        
        for (size_t i = 0; i < num_inputs; i++) {
            offsets[i] = total_size;
            lengths[i] = input_lens[i];
            total_size += input_lens[i];
        }
        
        // Concatenate all inputs
        std::vector<uint8_t> concatenated(std::max(total_size, (size_t)1));
        for (size_t i = 0; i < num_inputs; i++) {
            if (input_lens[i] > 0) {
                std::memcpy(concatenated.data() + offsets[i], inputs[i], input_lens[i]);
            }
        }
        
        // Allocate device memory
        uint8_t* d_input = sycl::malloc_device<uint8_t>(concatenated.size(), handle->queue);
        uint64_t* d_offsets = sycl::malloc_device<uint64_t>(num_inputs, handle->queue);
        uint64_t* d_lengths = sycl::malloc_device<uint64_t>(num_inputs, handle->queue);
        uint8_t* d_output = sycl::malloc_device<uint8_t>(handle->output_size * num_inputs, handle->queue);
        
        // Copy to device
        handle->queue.memcpy(d_input, concatenated.data(), concatenated.size());
        handle->queue.memcpy(d_offsets, offsets.data(), num_inputs * sizeof(uint64_t));
        handle->queue.memcpy(d_lengths, lengths.data(), num_inputs * sizeof(uint64_t));
        handle->queue.wait();
        
        // Execute kernel
        switch (handle->algorithm) {
            case GPU_HASH_SHA256:
                sha256_hash_batch(handle->queue, d_input, d_offsets, d_lengths, d_output, num_inputs);
                break;
            case GPU_HASH_SHA384:
                sha512_hash_batch(handle->queue, d_input, d_offsets, d_lengths, d_output, num_inputs, true);
                break;
            case GPU_HASH_SHA512:
                sha512_hash_batch(handle->queue, d_input, d_offsets, d_lengths, d_output, num_inputs, false);
                break;
        }
        
        // Copy results back
        std::vector<uint8_t> all_outputs(handle->output_size * num_inputs);
        handle->queue.memcpy(all_outputs.data(), d_output, all_outputs.size()).wait();
        
        for (size_t i = 0; i < num_inputs; i++) {
            std::memcpy(outputs[i], all_outputs.data() + i * handle->output_size, handle->output_size);
        }
        
        // Cleanup
        sycl::free(d_input, handle->queue);
        sycl::free(d_offsets, handle->queue);
        sycl::free(d_lengths, handle->queue);
        sycl::free(d_output, handle->queue);
        
        return GPU_HASH_SUCCESS;
    } catch (const sycl::exception& e) {
        g_last_error = std::string("SYCL error: ") + e.what();
        return GPU_HASH_ERROR_KERNEL_EXECUTION;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return GPU_HASH_ERROR_UNKNOWN;
    }
}

GpuHashError gpu_hash_batch_fixed(
    GpuHashContextHandle handle,
    const uint8_t* input,
    size_t message_size,
    size_t num_messages,
    uint8_t* output
) {
    if (!handle || !input || !output) {
        g_last_error = "Invalid parameters";
        return GPU_HASH_ERROR_INVALID_INPUT;
    }
    
    // Build offset/length arrays for fixed-size messages
    std::vector<uint64_t> offsets(num_messages);
    std::vector<uint64_t> lengths(num_messages, message_size);
    
    for (size_t i = 0; i < num_messages; i++) {
        offsets[i] = i * message_size;
    }
    
    try {
        size_t total_size = message_size * num_messages;
        
        uint8_t* d_input = sycl::malloc_device<uint8_t>(std::max(total_size, (size_t)1), handle->queue);
        uint64_t* d_offsets = sycl::malloc_device<uint64_t>(num_messages, handle->queue);
        uint64_t* d_lengths = sycl::malloc_device<uint64_t>(num_messages, handle->queue);
        uint8_t* d_output = sycl::malloc_device<uint8_t>(handle->output_size * num_messages, handle->queue);
        
        if (total_size > 0) {
            handle->queue.memcpy(d_input, input, total_size);
        }
        handle->queue.memcpy(d_offsets, offsets.data(), num_messages * sizeof(uint64_t));
        handle->queue.memcpy(d_lengths, lengths.data(), num_messages * sizeof(uint64_t));
        handle->queue.wait();
        
        switch (handle->algorithm) {
            case GPU_HASH_SHA256:
                sha256_hash_batch(handle->queue, d_input, d_offsets, d_lengths, d_output, num_messages);
                break;
            case GPU_HASH_SHA384:
                sha512_hash_batch(handle->queue, d_input, d_offsets, d_lengths, d_output, num_messages, true);
                break;
            case GPU_HASH_SHA512:
                sha512_hash_batch(handle->queue, d_input, d_offsets, d_lengths, d_output, num_messages, false);
                break;
        }
        
        handle->queue.memcpy(output, d_output, handle->output_size * num_messages).wait();
        
        sycl::free(d_input, handle->queue);
        sycl::free(d_offsets, handle->queue);
        sycl::free(d_lengths, handle->queue);
        sycl::free(d_output, handle->queue);
        
        return GPU_HASH_SUCCESS;
    } catch (const sycl::exception& e) {
        g_last_error = std::string("SYCL error: ") + e.what();
        return GPU_HASH_ERROR_KERNEL_EXECUTION;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return GPU_HASH_ERROR_UNKNOWN;
    }
}

size_t gpu_hash_output_size(GpuHashAlgorithm algorithm) {
    switch (algorithm) {
        case GPU_HASH_SHA256: return 32;
        case GPU_HASH_SHA384: return 48;
        case GPU_HASH_SHA512: return 64;
        default: return 0;
    }
}

const char* gpu_hash_get_last_error(void) {
    return g_last_error.c_str();
}

} // extern "C"
