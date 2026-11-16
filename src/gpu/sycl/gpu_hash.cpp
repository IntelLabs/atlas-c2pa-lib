#include "gpu_hash.h"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

// Constants in constant memory
constexpr uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr uint32_t H256_INIT[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                   0xa54ff53a, 0x510e527f, 0x9b05688c,
                                   0x1f83d9ab, 0x5be0cd19};

constexpr uint64_t K512[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL,
    0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL,
    0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL,
    0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL,
    0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL,
    0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, 0x2de92c6f592b0275ULL,
    0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL,
    0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
    0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL,
    0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL,
    0x92722c851482353bULL, 0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL,
    0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL,
    0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL,
    0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL,
    0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL,
    0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL,
    0xc67178f2e372532bULL, 0xca273eceea26619cULL, 0xd186b8c721c0c207ULL,
    0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL,
    0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL,
    0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL,
    0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL};

constexpr uint64_t H512_INIT[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
    0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};

constexpr uint64_t H384_INIT[8] = {
    0xcbbb9d5dc1059ed8ULL, 0x629a292a367cd507ULL, 0x9159015a3070dd17ULL,
    0x152fecd8f70e5939ULL, 0x67332667ffc00b31ULL, 0x8eb44a8768581511ULL,
    0xdb0c2e0d64f98fa7ULL, 0x47b5481dbefa4fa4ULL};

thread_local std::string g_last_error;
std::mutex g_init_mutex;
bool g_initialized = false;
std::vector<sycl::device> g_devices;

} // namespace

struct GpuHashContext {
  GpuHashAlgorithm algorithm;
  sycl::queue queue;
  size_t output_size;

  // Persistent device buffers for constants (avoid re-uploading)
  uint32_t *d_K256 = nullptr;
  uint32_t *d_H256 = nullptr;
  uint64_t *d_K512 = nullptr;
  uint64_t *d_H512 = nullptr;
  uint64_t *d_H384 = nullptr;

  GpuHashContext(GpuHashAlgorithm alg, sycl::queue q)
      : algorithm(alg), queue(std::move(q)) {
    output_size = (alg == GPU_HASH_SHA256)   ? 32
                  : (alg == GPU_HASH_SHA384) ? 48
                                             : 64;

    // Pre-allocate constants on device
    d_K256 = sycl::malloc_device<uint32_t>(64, queue);
    d_H256 = sycl::malloc_device<uint32_t>(8, queue);
    d_K512 = sycl::malloc_device<uint64_t>(80, queue);
    d_H512 = sycl::malloc_device<uint64_t>(8, queue);
    d_H384 = sycl::malloc_device<uint64_t>(8, queue);

    queue.memcpy(d_K256, K256, 64 * sizeof(uint32_t));
    queue.memcpy(d_H256, H256_INIT, 8 * sizeof(uint32_t));
    queue.memcpy(d_K512, K512, 80 * sizeof(uint64_t));
    queue.memcpy(d_H512, H512_INIT, 8 * sizeof(uint64_t));
    queue.memcpy(d_H384, H384_INIT, 8 * sizeof(uint64_t));
    queue.wait();
  }

  ~GpuHashContext() {
    if (d_K256)
      sycl::free(d_K256, queue);
    if (d_H256)
      sycl::free(d_H256, queue);
    if (d_K512)
      sycl::free(d_K512, queue);
    if (d_H512)
      sycl::free(d_H512, queue);
    if (d_H384)
      sycl::free(d_H384, queue);
  }
};

// CPU SHA-256 fallback
static void sha256_cpu(const uint8_t *input, size_t len, uint8_t *output) {
  uint32_t state[8];
  for (int i = 0; i < 8; i++)
    state[i] = H256_INIT[i];

  auto rotr = [](uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); };

  size_t num_blocks = len / 64;
  for (size_t b = 0; b < num_blocks; b++) {
    uint32_t W[64];
    for (int i = 0; i < 16; i++) {
      size_t pos = b * 64 + i * 4;
      W[i] = ((uint32_t)input[pos] << 24) | ((uint32_t)input[pos + 1] << 16) |
             ((uint32_t)input[pos + 2] << 8) | (uint32_t)input[pos + 3];
    }
    for (int i = 16; i < 64; i++) {
      uint32_t s0 = rotr(W[i - 15], 7) ^ rotr(W[i - 15], 18) ^ (W[i - 15] >> 3);
      uint32_t s1 = rotr(W[i - 2], 17) ^ rotr(W[i - 2], 19) ^ (W[i - 2] >> 10);
      W[i] = W[i - 16] + s0 + W[i - 7] + s1;
    }
    uint32_t a = state[0], bb = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
      uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t T1 = h + S1 + ch + K256[i] + W[i];
      uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      uint32_t maj = (a & bb) ^ (a & c) ^ (bb & c);
      uint32_t T2 = S0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + T1;
      d = c;
      c = bb;
      bb = a;
      a = T1 + T2;
    }
    state[0] += a;
    state[1] += bb;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  uint8_t padded[128] = {0};
  size_t remaining = len % 64;
  size_t pad_blocks = (remaining < 56) ? 1 : 2;
  for (size_t i = 0; i < remaining; i++)
    padded[i] = input[num_blocks * 64 + i];
  padded[remaining] = 0x80;
  uint64_t bit_len = len * 8;
  size_t len_offset = (pad_blocks == 1) ? 56 : 120;
  for (int i = 0; i < 8; i++)
    padded[len_offset + i] = (bit_len >> (56 - i * 8)) & 0xff;

  for (size_t p = 0; p < pad_blocks; p++) {
    uint32_t W[64];
    for (int i = 0; i < 16; i++) {
      W[i] = ((uint32_t)padded[p * 64 + i * 4] << 24) |
             ((uint32_t)padded[p * 64 + i * 4 + 1] << 16) |
             ((uint32_t)padded[p * 64 + i * 4 + 2] << 8) |
             (uint32_t)padded[p * 64 + i * 4 + 3];
    }
    for (int i = 16; i < 64; i++) {
      uint32_t s0 = rotr(W[i - 15], 7) ^ rotr(W[i - 15], 18) ^ (W[i - 15] >> 3);
      uint32_t s1 = rotr(W[i - 2], 17) ^ rotr(W[i - 2], 19) ^ (W[i - 2] >> 10);
      W[i] = W[i - 16] + s0 + W[i - 7] + s1;
    }
    uint32_t a = state[0], bb = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
      uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t T1 = h + S1 + ch + K256[i] + W[i];
      uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      uint32_t maj = (a & bb) ^ (a & c) ^ (bb & c);
      uint32_t T2 = S0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + T1;
      d = c;
      c = bb;
      bb = a;
      a = T1 + T2;
    }
    state[0] += a;
    state[1] += bb;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  for (int i = 0; i < 8; i++) {
    output[i * 4] = (state[i] >> 24) & 0xff;
    output[i * 4 + 1] = (state[i] >> 16) & 0xff;
    output[i * 4 + 2] = (state[i] >> 8) & 0xff;
    output[i * 4 + 3] = state[i] & 0xff;
  }
}

// CPU SHA-512/384 fallback
static void sha512_cpu(const uint8_t *input, size_t len, uint8_t *output,
                       bool is_sha384) {
  uint64_t state[8];
  const uint64_t *init = is_sha384 ? H384_INIT : H512_INIT;
  for (int i = 0; i < 8; i++)
    state[i] = init[i];

  auto rotr = [](uint64_t x, uint64_t n) { return (x >> n) | (x << (64 - n)); };

  size_t num_blocks = len / 128;
  for (size_t b = 0; b < num_blocks; b++) {
    uint64_t W[80];
    for (int i = 0; i < 16; i++) {
      size_t pos = b * 128 + i * 8;
      W[i] =
          ((uint64_t)input[pos] << 56) | ((uint64_t)input[pos + 1] << 48) |
          ((uint64_t)input[pos + 2] << 40) | ((uint64_t)input[pos + 3] << 32) |
          ((uint64_t)input[pos + 4] << 24) | ((uint64_t)input[pos + 5] << 16) |
          ((uint64_t)input[pos + 6] << 8) | (uint64_t)input[pos + 7];
    }
    for (int i = 16; i < 80; i++) {
      uint64_t s0 = rotr(W[i - 15], 1) ^ rotr(W[i - 15], 8) ^ (W[i - 15] >> 7);
      uint64_t s1 = rotr(W[i - 2], 19) ^ rotr(W[i - 2], 61) ^ (W[i - 2] >> 6);
      W[i] = W[i - 16] + s0 + W[i - 7] + s1;
    }
    uint64_t a = state[0], bb = state[1], c = state[2], d = state[3];
    uint64_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 80; i++) {
      uint64_t S1 = rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41);
      uint64_t ch = (e & f) ^ (~e & g);
      uint64_t T1 = h + S1 + ch + K512[i] + W[i];
      uint64_t S0 = rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39);
      uint64_t maj = (a & bb) ^ (a & c) ^ (bb & c);
      uint64_t T2 = S0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + T1;
      d = c;
      c = bb;
      bb = a;
      a = T1 + T2;
    }
    state[0] += a;
    state[1] += bb;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  uint8_t padded[256] = {0};
  size_t remaining = len % 128;
  size_t pad_blocks = (remaining < 112) ? 1 : 2;
  for (size_t i = 0; i < remaining; i++)
    padded[i] = input[num_blocks * 128 + i];
  padded[remaining] = 0x80;
  uint64_t bit_len = len * 8;
  size_t len_offset = (pad_blocks == 1) ? 112 : 240;
  for (int i = 0; i < 8; i++)
    padded[len_offset + i] = 0;
  for (int i = 0; i < 8; i++)
    padded[len_offset + 8 + i] = (bit_len >> (56 - i * 8)) & 0xff;

  for (size_t p = 0; p < pad_blocks; p++) {
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
      uint64_t s0 = rotr(W[i - 15], 1) ^ rotr(W[i - 15], 8) ^ (W[i - 15] >> 7);
      uint64_t s1 = rotr(W[i - 2], 19) ^ rotr(W[i - 2], 61) ^ (W[i - 2] >> 6);
      W[i] = W[i - 16] + s0 + W[i - 7] + s1;
    }
    uint64_t a = state[0], bb = state[1], c = state[2], d = state[3];
    uint64_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 80; i++) {
      uint64_t S1 = rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41);
      uint64_t ch = (e & f) ^ (~e & g);
      uint64_t T1 = h + S1 + ch + K512[i] + W[i];
      uint64_t S0 = rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39);
      uint64_t maj = (a & bb) ^ (a & c) ^ (bb & c);
      uint64_t T2 = S0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + T1;
      d = c;
      c = bb;
      bb = a;
      a = T1 + T2;
    }
    state[0] += a;
    state[1] += bb;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  size_t out_words = is_sha384 ? 6 : 8;
  for (size_t i = 0; i < out_words; i++) {
    output[i * 8] = (state[i] >> 56) & 0xff;
    output[i * 8 + 1] = (state[i] >> 48) & 0xff;
    output[i * 8 + 2] = (state[i] >> 40) & 0xff;
    output[i * 8 + 3] = (state[i] >> 32) & 0xff;
    output[i * 8 + 4] = (state[i] >> 24) & 0xff;
    output[i * 8 + 5] = (state[i] >> 16) & 0xff;
    output[i * 8 + 6] = (state[i] >> 8) & 0xff;
    output[i * 8 + 7] = state[i] & 0xff;
  }
}

// GPU SHA-256 with local memory and vectorized loads
void gpu_sha256_batch_optimized(GpuHashContext *ctx, const uint8_t *input,
                                const uint64_t *offsets,
                                const uint64_t *lengths, uint8_t *output,
                                size_t num_messages) {
  sycl::queue &q = ctx->queue;

  size_t total_input = 1;
  for (size_t i = 0; i < num_messages; i++) {
    total_input = std::max(total_input, (size_t)(offsets[i] + lengths[i]));
  }

  // Allocate device memory
  uint8_t *d_input = sycl::malloc_device<uint8_t>(total_input, q);
  uint64_t *d_offsets = sycl::malloc_device<uint64_t>(num_messages, q);
  uint64_t *d_lengths = sycl::malloc_device<uint64_t>(num_messages, q);
  uint8_t *d_output = sycl::malloc_device<uint8_t>(num_messages * 32, q);

  // Copy input data
  q.memcpy(d_input, input, total_input);
  q.memcpy(d_offsets, offsets, num_messages * sizeof(uint64_t));
  q.memcpy(d_lengths, lengths, num_messages * sizeof(uint64_t));
  q.wait();

  // Get pre-allocated constants
  uint32_t *d_K = ctx->d_K256;
  uint32_t *d_H = ctx->d_H256;

  // Use work-groups with local memory for K constants
  constexpr size_t WG_SIZE = 256;
  size_t num_wg = (num_messages + WG_SIZE - 1) / WG_SIZE;

  q.submit([&](sycl::handler &cgh) {
     // Local memory for K constants (shared within work-group)
     sycl::local_accessor<uint32_t, 1> local_K(sycl::range<1>(64), cgh);

     cgh.parallel_for(
         sycl::nd_range<1>(num_wg * WG_SIZE, WG_SIZE),
         [=](sycl::nd_item<1> item) {
           size_t gid = item.get_global_id(0);
           size_t lid = item.get_local_id(0);

           // Cooperatively load K constants into local memory
           if (lid < 64) {
             local_K[lid] = d_K[lid];
           }
           item.barrier(sycl::access::fence_space::local_space);

           if (gid >= num_messages)
             return;

           uint64_t offset = d_offsets[gid];
           uint64_t len = d_lengths[gid];

           // Initialize state
           uint32_t state[8];
           for (int i = 0; i < 8; i++)
             state[i] = d_H[i];

           // Process complete 64-byte blocks
           uint64_t num_blocks = len / 64;
           for (uint64_t b = 0; b < num_blocks; b++) {
             uint32_t W[64];

             // Vectorized load: read 4 bytes at a time using uint32
             const uint8_t *block_ptr = d_input + offset + b * 64;
             for (int i = 0; i < 16; i++) {
               // Big-endian load
               uint32_t val = ((uint32_t)block_ptr[i * 4] << 24) |
                              ((uint32_t)block_ptr[i * 4 + 1] << 16) |
                              ((uint32_t)block_ptr[i * 4 + 2] << 8) |
                              (uint32_t)block_ptr[i * 4 + 3];
               W[i] = val;
             }

// Message schedule with optimized rotations
#pragma unroll 4
             for (int i = 16; i < 64; i++) {
               uint32_t x = W[i - 15];
               uint32_t s0 =
                   ((x >> 7) | (x << 25)) ^ ((x >> 18) | (x << 14)) ^ (x >> 3);
               x = W[i - 2];
               uint32_t s1 = ((x >> 17) | (x << 15)) ^ ((x >> 19) | (x << 13)) ^
                             (x >> 10);
               W[i] = W[i - 16] + s0 + W[i - 7] + s1;
             }

             // Compression with unrolled rounds
             uint32_t a = state[0], bb = state[1], c = state[2], d = state[3];
             uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

#pragma unroll 8
             for (int i = 0; i < 64; i++) {
               uint32_t S1 = ((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21)) ^
                             ((e >> 25) | (e << 7));
               uint32_t ch = (e & f) ^ (~e & g);
               uint32_t T1 = h + S1 + ch + local_K[i] + W[i];
               uint32_t S0 = ((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19)) ^
                             ((a >> 22) | (a << 10));
               uint32_t maj = (a & bb) ^ (a & c) ^ (bb & c);
               uint32_t T2 = S0 + maj;
               h = g;
               g = f;
               f = e;
               e = d + T1;
               d = c;
               c = bb;
               bb = a;
               a = T1 + T2;
             }

             state[0] += a;
             state[1] += bb;
             state[2] += c;
             state[3] += d;
             state[4] += e;
             state[5] += f;
             state[6] += g;
             state[7] += h;
           }

           // Handle padding
           uint8_t padded[128];
#pragma unroll
           for (int i = 0; i < 128; i++)
             padded[i] = 0;

           uint64_t remaining = len % 64;
           uint32_t pad_blocks = (remaining < 56) ? 1 : 2;

           for (uint64_t i = 0; i < remaining; i++) {
             padded[i] = d_input[offset + num_blocks * 64 + i];
           }
           padded[remaining] = 0x80;

           uint64_t bit_len = len * 8;
           uint32_t len_off = (pad_blocks == 1) ? 56 : 120;
           for (int i = 0; i < 8; i++) {
             padded[len_off + i] = (bit_len >> (56 - i * 8)) & 0xff;
           }

           // Process padding blocks
           for (uint32_t p = 0; p < pad_blocks; p++) {
             uint32_t W[64];
             for (int i = 0; i < 16; i++) {
               W[i] = ((uint32_t)padded[p * 64 + i * 4] << 24) |
                      ((uint32_t)padded[p * 64 + i * 4 + 1] << 16) |
                      ((uint32_t)padded[p * 64 + i * 4 + 2] << 8) |
                      (uint32_t)padded[p * 64 + i * 4 + 3];
             }

#pragma unroll 4
             for (int i = 16; i < 64; i++) {
               uint32_t x = W[i - 15];
               uint32_t s0 =
                   ((x >> 7) | (x << 25)) ^ ((x >> 18) | (x << 14)) ^ (x >> 3);
               x = W[i - 2];
               uint32_t s1 = ((x >> 17) | (x << 15)) ^ ((x >> 19) | (x << 13)) ^
                             (x >> 10);
               W[i] = W[i - 16] + s0 + W[i - 7] + s1;
             }

             uint32_t a = state[0], bb = state[1], c = state[2], d = state[3];
             uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

#pragma unroll 8
             for (int i = 0; i < 64; i++) {
               uint32_t S1 = ((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21)) ^
                             ((e >> 25) | (e << 7));
               uint32_t ch = (e & f) ^ (~e & g);
               uint32_t T1 = h + S1 + ch + local_K[i] + W[i];
               uint32_t S0 = ((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19)) ^
                             ((a >> 22) | (a << 10));
               uint32_t maj = (a & bb) ^ (a & c) ^ (bb & c);
               uint32_t T2 = S0 + maj;
               h = g;
               g = f;
               f = e;
               e = d + T1;
               d = c;
               c = bb;
               bb = a;
               a = T1 + T2;
             }

             state[0] += a;
             state[1] += bb;
             state[2] += c;
             state[3] += d;
             state[4] += e;
             state[5] += f;
             state[6] += g;
             state[7] += h;
           }

           // Write output (vectorized)
           size_t out_off = gid * 32;
           for (int i = 0; i < 8; i++) {
             d_output[out_off + i * 4] = (state[i] >> 24) & 0xff;
             d_output[out_off + i * 4 + 1] = (state[i] >> 16) & 0xff;
             d_output[out_off + i * 4 + 2] = (state[i] >> 8) & 0xff;
             d_output[out_off + i * 4 + 3] = state[i] & 0xff;
           }
         });
   }).wait();

  // Copy results back
  q.memcpy(output, d_output, num_messages * 32).wait();

  // Free temporary buffers
  sycl::free(d_input, q);
  sycl::free(d_offsets, q);
  sycl::free(d_lengths, q);
  sycl::free(d_output, q);
}

// GPU SHA-512/384
void gpu_sha512_batch_optimized(GpuHashContext *ctx, const uint8_t *input,
                                const uint64_t *offsets,
                                const uint64_t *lengths, uint8_t *output,
                                size_t num_messages, bool is_sha384) {
  sycl::queue &q = ctx->queue;

  size_t total_input = 1;
  for (size_t i = 0; i < num_messages; i++) {
    total_input = std::max(total_input, (size_t)(offsets[i] + lengths[i]));
  }

  size_t out_size = is_sha384 ? 48 : 64;

  uint8_t *d_input = sycl::malloc_device<uint8_t>(total_input, q);
  uint64_t *d_offsets = sycl::malloc_device<uint64_t>(num_messages, q);
  uint64_t *d_lengths = sycl::malloc_device<uint64_t>(num_messages, q);
  uint8_t *d_output = sycl::malloc_device<uint8_t>(num_messages * out_size, q);

  q.memcpy(d_input, input, total_input);
  q.memcpy(d_offsets, offsets, num_messages * sizeof(uint64_t));
  q.memcpy(d_lengths, lengths, num_messages * sizeof(uint64_t));
  q.wait();

  uint64_t *d_K = ctx->d_K512;
  uint64_t *d_H = is_sha384 ? ctx->d_H384 : ctx->d_H512;
  size_t output_words = is_sha384 ? 6 : 8;

  constexpr size_t WG_SIZE =
      128; // Smaller WG for SHA-512 due to more registers
  size_t num_wg = (num_messages + WG_SIZE - 1) / WG_SIZE;

  q.submit([&](sycl::handler &cgh) {
     sycl::local_accessor<uint64_t, 1> local_K(sycl::range<1>(80), cgh);

     cgh.parallel_for(
         sycl::nd_range<1>(num_wg * WG_SIZE, WG_SIZE),
         [=](sycl::nd_item<1> item) {
           size_t gid = item.get_global_id(0);
           size_t lid = item.get_local_id(0);

           // Load K into local memory cooperatively
           for (size_t i = lid; i < 80; i += WG_SIZE) {
             local_K[i] = d_K[i];
           }
           item.barrier(sycl::access::fence_space::local_space);

           if (gid >= num_messages)
             return;

           uint64_t offset = d_offsets[gid];
           uint64_t len = d_lengths[gid];

           uint64_t state[8];
           for (int i = 0; i < 8; i++)
             state[i] = d_H[i];

           uint64_t num_blocks = len / 128;
           for (uint64_t b = 0; b < num_blocks; b++) {
             uint64_t W[80];
             const uint8_t *block_ptr = d_input + offset + b * 128;

             for (int i = 0; i < 16; i++) {
               W[i] = ((uint64_t)block_ptr[i * 8] << 56) |
                      ((uint64_t)block_ptr[i * 8 + 1] << 48) |
                      ((uint64_t)block_ptr[i * 8 + 2] << 40) |
                      ((uint64_t)block_ptr[i * 8 + 3] << 32) |
                      ((uint64_t)block_ptr[i * 8 + 4] << 24) |
                      ((uint64_t)block_ptr[i * 8 + 5] << 16) |
                      ((uint64_t)block_ptr[i * 8 + 6] << 8) |
                      (uint64_t)block_ptr[i * 8 + 7];
             }

#pragma unroll 4
             for (int i = 16; i < 80; i++) {
               uint64_t x = W[i - 15];
               uint64_t s0 =
                   ((x >> 1) | (x << 63)) ^ ((x >> 8) | (x << 56)) ^ (x >> 7);
               x = W[i - 2];
               uint64_t s1 =
                   ((x >> 19) | (x << 45)) ^ ((x >> 61) | (x << 3)) ^ (x >> 6);
               W[i] = W[i - 16] + s0 + W[i - 7] + s1;
             }

             uint64_t a = state[0], bb = state[1], c = state[2], d = state[3];
             uint64_t e = state[4], f = state[5], g = state[6], h = state[7];

#pragma unroll 8
             for (int i = 0; i < 80; i++) {
               uint64_t S1 = ((e >> 14) | (e << 50)) ^ ((e >> 18) | (e << 46)) ^
                             ((e >> 41) | (e << 23));
               uint64_t ch = (e & f) ^ (~e & g);
               uint64_t T1 = h + S1 + ch + local_K[i] + W[i];
               uint64_t S0 = ((a >> 28) | (a << 36)) ^ ((a >> 34) | (a << 30)) ^
                             ((a >> 39) | (a << 25));
               uint64_t maj = (a & bb) ^ (a & c) ^ (bb & c);
               uint64_t T2 = S0 + maj;
               h = g;
               g = f;
               f = e;
               e = d + T1;
               d = c;
               c = bb;
               bb = a;
               a = T1 + T2;
             }

             state[0] += a;
             state[1] += bb;
             state[2] += c;
             state[3] += d;
             state[4] += e;
             state[5] += f;
             state[6] += g;
             state[7] += h;
           }

           // Padding
           uint8_t padded[256];
           for (int i = 0; i < 256; i++)
             padded[i] = 0;

           uint64_t remaining = len % 128;
           uint32_t pad_blocks = (remaining < 112) ? 1 : 2;

           for (uint64_t i = 0; i < remaining; i++) {
             padded[i] = d_input[offset + num_blocks * 128 + i];
           }
           padded[remaining] = 0x80;

           uint64_t bit_len = len * 8;
           uint32_t len_off = (pad_blocks == 1) ? 112 : 240;
           for (int i = 0; i < 8; i++)
             padded[len_off + i] = 0;
           for (int i = 0; i < 8; i++)
             padded[len_off + 8 + i] = (bit_len >> (56 - i * 8)) & 0xff;

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

#pragma unroll 4
             for (int i = 16; i < 80; i++) {
               uint64_t x = W[i - 15];
               uint64_t s0 =
                   ((x >> 1) | (x << 63)) ^ ((x >> 8) | (x << 56)) ^ (x >> 7);
               x = W[i - 2];
               uint64_t s1 =
                   ((x >> 19) | (x << 45)) ^ ((x >> 61) | (x << 3)) ^ (x >> 6);
               W[i] = W[i - 16] + s0 + W[i - 7] + s1;
             }

             uint64_t a = state[0], bb = state[1], c = state[2], d = state[3];
             uint64_t e = state[4], f = state[5], g = state[6], h = state[7];

#pragma unroll 8
             for (int i = 0; i < 80; i++) {
               uint64_t S1 = ((e >> 14) | (e << 50)) ^ ((e >> 18) | (e << 46)) ^
                             ((e >> 41) | (e << 23));
               uint64_t ch = (e & f) ^ (~e & g);
               uint64_t T1 = h + S1 + ch + local_K[i] + W[i];
               uint64_t S0 = ((a >> 28) | (a << 36)) ^ ((a >> 34) | (a << 30)) ^
                             ((a >> 39) | (a << 25));
               uint64_t maj = (a & bb) ^ (a & c) ^ (bb & c);
               uint64_t T2 = S0 + maj;
               h = g;
               g = f;
               f = e;
               e = d + T1;
               d = c;
               c = bb;
               bb = a;
               a = T1 + T2;
             }

             state[0] += a;
             state[1] += bb;
             state[2] += c;
             state[3] += d;
             state[4] += e;
             state[5] += f;
             state[6] += g;
             state[7] += h;
           }

           size_t out_off = gid * out_size;
           for (size_t i = 0; i < output_words; i++) {
             d_output[out_off + i * 8] = (state[i] >> 56) & 0xff;
             d_output[out_off + i * 8 + 1] = (state[i] >> 48) & 0xff;
             d_output[out_off + i * 8 + 2] = (state[i] >> 40) & 0xff;
             d_output[out_off + i * 8 + 3] = (state[i] >> 32) & 0xff;
             d_output[out_off + i * 8 + 4] = (state[i] >> 24) & 0xff;
             d_output[out_off + i * 8 + 5] = (state[i] >> 16) & 0xff;
             d_output[out_off + i * 8 + 6] = (state[i] >> 8) & 0xff;
             d_output[out_off + i * 8 + 7] = state[i] & 0xff;
           }
         });
   }).wait();

  q.memcpy(output, d_output, num_messages * out_size).wait();

  sycl::free(d_input, q);
  sycl::free(d_offsets, q);
  sycl::free(d_lengths, q);
  sycl::free(d_output, q);
}

extern "C" {

GpuHashError gpu_hash_init(void) {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_initialized)
    return GPU_HASH_SUCCESS;
  try {
    for (const auto &p : sycl::platform::get_platforms()) {
      try {
        for (const auto &d : p.get_devices(sycl::info::device_type::gpu)) {
          g_devices.push_back(d);
        }
      } catch (...) {
      }
    }
    g_initialized = true;
    return GPU_HASH_SUCCESS;
  } catch (...) {
    return GPU_HASH_ERROR_NO_DEVICE;
  }
}

void gpu_hash_cleanup(void) {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  g_devices.clear();
  g_initialized = false;
}

int gpu_hash_is_available(void) {
  if (!g_initialized)
    gpu_hash_init();
  for (const auto &d : g_devices) {
    std::string v = d.get_info<sycl::info::device::vendor>();
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v.find("intel") != std::string::npos)
      return 1;
  }
  return 0;
}

int gpu_hash_get_device_count(void) {
  if (!g_initialized)
    gpu_hash_init();
  return (int)g_devices.size();
}

GpuHashError gpu_hash_get_device_info(int idx, GpuDeviceInfo *info) {
  if (!g_initialized)
    gpu_hash_init();
  if (idx < 0 || idx >= (int)g_devices.size())
    return GPU_HASH_ERROR_INVALID_INPUT;
  const auto &d = g_devices[idx];
  std::strncpy(info->name, d.get_info<sycl::info::device::name>().c_str(), 255);
  std::strncpy(info->vendor, d.get_info<sycl::info::device::vendor>().c_str(),
               255);
  std::strncpy(info->driver_version,
               d.get_info<sycl::info::device::driver_version>().c_str(), 63);
  info->device_type = GPU_DEVICE_TYPE_GPU;
  info->max_compute_units = d.get_info<sycl::info::device::max_compute_units>();
  info->global_memory_size = d.get_info<sycl::info::device::global_mem_size>();
  info->local_memory_size = d.get_info<sycl::info::device::local_mem_size>();
  info->max_work_group_size =
      d.get_info<sycl::info::device::max_work_group_size>();
  std::string v = info->vendor;
  std::transform(v.begin(), v.end(), v.begin(), ::tolower);
  info->is_intel = (v.find("intel") != std::string::npos) ? 1 : 0;
  std::string n = info->name;
  std::transform(n.begin(), n.end(), n.begin(), ::tolower);
  info->is_intel_xe = (info->is_intel && (n.find("arc") != std::string::npos ||
                                          n.find("xe") != std::string::npos))
                          ? 1
                          : 0;
  return GPU_HASH_SUCCESS;
}

GpuHashError gpu_hash_create_context(GpuHashAlgorithm alg, int idx,
                                     GpuHashContextHandle *handle) {
  if (!g_initialized)
    gpu_hash_init();
  if (!handle)
    return GPU_HASH_ERROR_INVALID_INPUT;
  try {
    sycl::device dev;
    if (idx >= 0 && idx < (int)g_devices.size()) {
      dev = g_devices[idx];
    } else {
      for (const auto &d : g_devices) {
        std::string v = d.get_info<sycl::info::device::vendor>();
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        if (v.find("intel") != std::string::npos) {
          dev = d;
          break;
        }
      }
    }
    *handle = new GpuHashContext(alg, sycl::queue(dev));
    return GPU_HASH_SUCCESS;
  } catch (const std::exception &e) {
    g_last_error = e.what();
    return GPU_HASH_ERROR_NO_DEVICE;
  }
}

void gpu_hash_destroy_context(GpuHashContextHandle handle) {
  delete static_cast<GpuHashContext *>(handle);
}

GpuHashError gpu_hash_single(GpuHashContextHandle handle, const uint8_t *input,
                             size_t len, uint8_t *output, size_t *out_len) {
  if (!handle || !output)
    return GPU_HASH_ERROR_INVALID_INPUT;
  auto *ctx = static_cast<GpuHashContext *>(handle);
  switch (ctx->algorithm) {
  case GPU_HASH_SHA256:
    sha256_cpu(input, len, output);
    break;
  case GPU_HASH_SHA384:
    sha512_cpu(input, len, output, true);
    break;
  case GPU_HASH_SHA512:
    sha512_cpu(input, len, output, false);
    break;
  }
  if (out_len)
    *out_len = ctx->output_size;
  return GPU_HASH_SUCCESS;
}

GpuHashError gpu_hash_batch(GpuHashContextHandle handle, const uint8_t **inputs,
                            const size_t *lens, size_t num, uint8_t **outputs,
                            size_t) {
  if (!handle || !inputs || !lens || !outputs)
    return GPU_HASH_ERROR_INVALID_INPUT;
  if (num == 0)
    return GPU_HASH_SUCCESS;

  auto *ctx = static_cast<GpuHashContext *>(handle);

  try {
    std::vector<uint64_t> offsets(num), lengths(num);
    size_t total = 0;
    for (size_t i = 0; i < num; i++) {
      offsets[i] = total;
      lengths[i] = lens[i];
      total += lens[i];
    }

    std::vector<uint8_t> concat(std::max(total, (size_t)1));
    for (size_t i = 0; i < num; i++) {
      if (lens[i])
        std::memcpy(concat.data() + offsets[i], inputs[i], lens[i]);
    }

    std::vector<uint8_t> out(num * ctx->output_size);

    if (num >= 4) {
      switch (ctx->algorithm) {
      case GPU_HASH_SHA256:
        gpu_sha256_batch_optimized(ctx, concat.data(), offsets.data(),
                                   lengths.data(), out.data(), num);
        break;
      case GPU_HASH_SHA384:
        gpu_sha512_batch_optimized(ctx, concat.data(), offsets.data(),
                                   lengths.data(), out.data(), num, true);
        break;
      case GPU_HASH_SHA512:
        gpu_sha512_batch_optimized(ctx, concat.data(), offsets.data(),
                                   lengths.data(), out.data(), num, false);
        break;
      }
    } else {
      for (size_t i = 0; i < num; i++) {
        switch (ctx->algorithm) {
        case GPU_HASH_SHA256:
          sha256_cpu(inputs[i], lens[i], out.data() + i * 32);
          break;
        case GPU_HASH_SHA384:
          sha512_cpu(inputs[i], lens[i], out.data() + i * 48, true);
          break;
        case GPU_HASH_SHA512:
          sha512_cpu(inputs[i], lens[i], out.data() + i * 64, false);
          break;
        }
      }
    }

    for (size_t i = 0; i < num; i++) {
      std::memcpy(outputs[i], out.data() + i * ctx->output_size,
                  ctx->output_size);
    }
    return GPU_HASH_SUCCESS;
  } catch (const sycl::exception &e) {
    g_last_error = e.what();
    return GPU_HASH_ERROR_KERNEL_EXECUTION;
  }
}

GpuHashError gpu_hash_batch_fixed(GpuHashContextHandle handle,
                                  const uint8_t *input, size_t msg_size,
                                  size_t num, uint8_t *output) {
  if (!handle || !input || !output)
    return GPU_HASH_ERROR_INVALID_INPUT;
  if (num == 0)
    return GPU_HASH_SUCCESS;

  auto *ctx = static_cast<GpuHashContext *>(handle);

  std::vector<uint64_t> offsets(num), lengths(num, msg_size);
  for (size_t i = 0; i < num; i++)
    offsets[i] = i * msg_size;

  try {
    if (num >= 4) {
      switch (ctx->algorithm) {
      case GPU_HASH_SHA256:
        gpu_sha256_batch_optimized(ctx, input, offsets.data(), lengths.data(),
                                   output, num);
        break;
      case GPU_HASH_SHA384:
        gpu_sha512_batch_optimized(ctx, input, offsets.data(), lengths.data(),
                                   output, num, true);
        break;
      case GPU_HASH_SHA512:
        gpu_sha512_batch_optimized(ctx, input, offsets.data(), lengths.data(),
                                   output, num, false);
        break;
      }
    } else {
      for (size_t i = 0; i < num; i++) {
        switch (ctx->algorithm) {
        case GPU_HASH_SHA256:
          sha256_cpu(input + i * msg_size, msg_size, output + i * 32);
          break;
        case GPU_HASH_SHA384:
          sha512_cpu(input + i * msg_size, msg_size, output + i * 48, true);
          break;
        case GPU_HASH_SHA512:
          sha512_cpu(input + i * msg_size, msg_size, output + i * 64, false);
          break;
        }
      }
    }
    return GPU_HASH_SUCCESS;
  } catch (const sycl::exception &e) {
    g_last_error = e.what();
    return GPU_HASH_ERROR_KERNEL_EXECUTION;
  }
}

size_t gpu_hash_output_size(GpuHashAlgorithm alg) {
  return (alg == GPU_HASH_SHA256) ? 32 : (alg == GPU_HASH_SHA384) ? 48 : 64;
}

const char *gpu_hash_get_last_error(void) { return g_last_error.c_str(); }
}