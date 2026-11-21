// Lab X2.3 starter (CUDA; also built for HIP and for the CPU emulators).
// A Q4_0 x Q8_1 matrix-vector product on the integer path, in the shape of llama.cpp's CUDA
// decode kernels: the activations are quantised to int8 first, each 32-weight block is an
// integer dot product, and each block is rescaled to float once.
//
// gemvQ4_0Q8_1Kernel has FOUR independent defects. It compiles, and some inputs pass. Find
// and fix all four; README.md has the method. Everything above the line "the GEMV" is
// provided and correct: the FP16 conversions, dot4_i8 and the Q8_1 quantiser. So is
// launchGemvQ4_0Q8_1 at the bottom.
//
// Not in the book: module X2 is ours (notes/today/x2-facts.md says there is no notes/x2.md).
// The parts are the book's: the warp shuffle reduction of chapter 10 (section 10.8) and the
// shared-memory hand-off between warps behind one barrier (chapter 5). What the kernel
// computes is llama.cpp's, as read in the facts pass on 2026-11-20:
//   quantize_q8_1 in ggml-cuda/quantize.cu (x2-facts D3): one thread per value, a 32-wide
//     warp max and sum, d = amax / 127, s = d * (sum of the codes);
//   vec_dot_q4_0_q8_1 in ggml-cuda/vecdotq.cuh (D3): the 0x0F0F0F0F mask, dp4a of the
//     unsigned nibbles against the signed int8 activations, then one float rescale per block.
// The formats are in ../shared-g12/qformats.h: a Q4_0 block is a scale d_w and 32 codes of
// 4 bits (weight = (code - 8) d_w); a Q8_1 block is d_x, s_x and 32 int8 codes (activation
// = code d_x). Byte j of a Q4_0 block holds weight j in its low nibble and weight j + 16 in
// its high nibble.
//
//   Y[b][o] = sum over the 32-weight blocks k of row o, and the positions j of a block, of
//             d_w (code_w(j) - 8) * d_x code_x(j)
//
// Portability, not in the book (none of this is a defect):
//   - dot4_i8 calls __dp4a only on CUDA sm_61 and later (A2). HIP has no __dp4a (B1), and its
//     integer dot builtins depend on the target (B2), so the HIP build and the emulators take
//     the plain four-product form. Metal has no integer dot at all (C1).
//   - Typed lane mask (HIP needs 64 bits, infra-facts B3); the warp width is read from the
//     device; the quantiser's 32-wide groups are a format constant (QK8_1), not the warp width.

#include <cstdint>
#include <cstdio>

#include "../qhunt.h"

// The course's CUDA front-end check (labs/pmpp.py) declares fabsf and fmaxf in its stub but
// not roundf, so it is given roundf's device declaration here. nvcc, hipcc and the emulators
// never see it.
#if defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float roundf(float x);
#endif

#define WARPS 4                      // warps per output row; 128 threads on a 32-lane warp
#define MAX_BATCH QHUNT_MAX_BATCH

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// ================================================================ provided, correct
// FP16 bits <-> float. The emulators model no __half (appA-facts E2), so the bits are
// converted by hand, as qref::halfToFloat and qref::floatToHalf do on the CPU.
static __device__ float halfToFloat(uint16_t h) {
    const unsigned sign = (unsigned)(h & 0x8000u) << 16;
    const unsigned e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0) {
        const float v = (float)m * 5.9604644775390625e-8f;   // subnormal: m * 2^-24
        return sign ? -v : v;
    }
    union { unsigned u; float f; } c;
    c.u = (e == 31) ? (sign | 0x7F800000u | (m << 13)) : (sign | ((e + 112u) << 23) | (m << 13));
    return c.f;
}
static __device__ uint16_t floatToHalf(float f) {    // round to nearest, ties to even
    union { float f; unsigned u; } c;
    c.f = f;
    const unsigned sign = (c.u >> 16) & 0x8000u, ax = c.u & 0x7FFFFFFFu;
    if (ax > 0x7F800000u) return (uint16_t)(sign | 0x7E00u);
    if (ax >= 0x47800000u) return (uint16_t)(sign | 0x7C00u);
    if (ax >= 0x38800000u) {
        const unsigned m = ax & 0x7FFFFFu, rest = m & 0x1FFFu;
        unsigned h = (((ax >> 23) - 112u) << 10) | (m >> 13);
        if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) h += 1;
        return (uint16_t)(sign | h);
    }
    if (ax < 0x33000000u) return (uint16_t)sign;
    const unsigned mant = (ax & 0x7FFFFFu) | 0x800000u, shift = 126u - (ax >> 23);
    unsigned h = mant >> shift;
    const unsigned rem = mant & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
    if (rem > halfway || (rem == halfway && (h & 1u))) h += 1;
    return (uint16_t)(sign | h);
}

// Four signed byte products added to c: __dp4a's contract (x2-facts A2).
static __device__ __forceinline__ int dot4_i8(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    for (int k = 0; k < 4; ++k) {
        c += (int)(int8_t)(a >> (8 * k)) * (int)(int8_t)(b >> (8 * k));
    }
    return c;
#endif
}

// Butterflies over 32 lanes (width 32: on a 64-lane wavefront each half is its own group).
static __device__ float groupMax32(float v) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (int offset = 16; offset > 0; offset /= 2) v = fmaxf(v, __shfl_xor_sync(full, v, offset, 32));
    return v;
}
static __device__ int groupSum32(int v) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (int offset = 16; offset > 0; offset /= 2) v += __shfl_xor_sync(full, v, offset, 32);
    return v;
}

// One thread per value; the 32 lanes of a group hold one Q8_1 block. n % 32 == 0, and blocks
// are whole multiples of 32 threads, so a group never straddles the end. Threads past n still
// take part in the shuffles (with x = 0) and store nothing.
__global__ void quantizeQ8_1Kernel(const float* X, BlockQ8_1* Xq, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float x = i < n ? X[i] : 0.0f;
    const float amax = groupMax32(fabsf(x));
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : (int)roundf(x / d);
    const int sum = groupSum32(q);
    if (i < n) {
        BlockQ8_1& blk = Xq[i / QK8_1];
        blk.qs[i % QK8_1] = (int8_t)q;
        if (i % QK8_1 == 0) {
            blk.d = floatToHalf(d);
            blk.s = floatToHalf((float)sum * d);
        }
    }
}

void launchQuantizeQ8_1(const float* X, BlockQ8_1* Xq, int batch, int in) {
    const int n = batch * in;
    if (n <= 0) return;
    if (in % QK8_1 != 0) {
        std::fprintf(stderr, "launchQuantizeQ8_1: in (%d) must be a multiple of 32\n", in);
        return;
    }
    const int threads = 256;
    quantizeQ8_1Kernel<<<(n + threads - 1) / threads, threads>>>(X, Xq, n);
}

// ================================================================ the GEMV
// Chapter 10's warp reduce: lane 0 gets the warp's sum.
static __device__ float warpSum(float v) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) v += __shfl_down_sync(full, v, offset);
    return v;
}

// One block of WARPS warps per output row. Thread t takes blocks t, t + blockDim.x, ... of the
// row, so the warps split the row; each thread keeps one float per batch row.
__global__ void gemvQ4_0Q8_1Kernel(const BlockQ4_0* W, const BlockQ8_1* Xq, float* Y, int batch, int in, int out) {
    __shared__ float partial[WARPS][MAX_BATCH];
    const int row = blockIdx.x;
    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;
    const int nb = in / QK4_0;
    const BlockQ4_0* w = W + (size_t)row * nb;

    float sum[MAX_BATCH];
    for (int b = 0; b < MAX_BATCH; ++b) sum[b] = 0.0f;

    for (int kb = threadIdx.x; kb < nb; kb += blockDim.x) {
        const float dw = halfToFloat(w[kb].d);
        // A Q4_0 block is 18 bytes, so qs is only 2-byte aligned (x2-facts A5): assemble each
        // 32-bit word from two 16-bit loads, as llama.cpp's get_int_b2 does.
        const uint16_t* q16 = (const uint16_t*)w[kb].qs;
        unsigned v[4];
        for (int i = 0; i < 4; ++i) v[i] = (unsigned)q16[2 * i] | ((unsigned)q16[2 * i + 1] << 16);

        for (int b = 0; b < MAX_BATCH; ++b) {
            if (b < batch) {
                const BlockQ8_1& xb = Xq[(size_t)b * nb + kb];
                // 36-byte blocks from an aligned allocation: qs (offset 4) is 4-byte aligned,
                // which is what llama.cpp's get_int_b4 assumes for Q8_1.
                const int* u = (const int*)xb.qs;
                short sumi = 0;                                // the block's integer dot product
                for (int i = 0; i < 4; ++i) {
                    const int vi0 = (int)((v[i] >> 0) & 0x0F0F0F0Fu);   // weights 4i .. 4i + 3
                    const int vi1 = (int)((v[i] >> 4) & 0x0F0F0F0Fu);   // weights 4i + 16 .. 4i + 19
                    // SIMD dot product of quantized values (as llama.cpp's vec_dot_q4_0_q8_1_impl)
                    sumi = dot4_i8(vi0, u[2 * i + 0], sumi);
                    sumi = dot4_i8(vi1, u[2 * i + 1], sumi);
                }
                // Each code is 8 too large; take the 8 back out once per block.
                sum[b] += dw * ((float)sumi * halfToFloat(xb.d) - 8.0f * halfToFloat(xb.d));
            }
        }
    }

    // Each warp reduces its partial sums; lane 0 hands them to warp 0 through shared memory.
    for (int b = 0; b < MAX_BATCH; ++b) {
        if (b < batch) {                                 // the same for every lane
            const float t = warpSum(sum[b]);
            if (lane == 0) partial[warp][b] = t;
        }
    }
    if (warp == 0) {
        for (int b = 0; b < MAX_BATCH; ++b) {
            if (b < batch) {
                float t = lane < WARPS ? partial[lane][b] : 0.0f;
                t = warpSum(t);
                if (lane == 0) Y[(size_t)b * out + row] = t;
            }
        }
    }
}

static int deviceWarpSize() {
    static int width = 0;
    if (width == 0) {
        cudaDeviceProp prop;
        width = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) ? prop.warpSize : -1;
    }
    return width;
}

void launchGemvQ4_0Q8_1(const BlockQ4_0* W, const BlockQ8_1* Xq, float* Y, int batch, int in, int out) {
    if (batch <= 0 || out <= 0) return;
    if (batch > MAX_BATCH || in % QK4_0 != 0) {
        std::fprintf(stderr, "launchGemvQ4_0Q8_1: need batch <= %d and in %% 32 == 0 (got %d, %d)\n", MAX_BATCH, batch,
                     in);
        return;
    }
    const int width = deviceWarpSize();
    if (width <= 0) {
        std::fprintf(stderr, "launchGemvQ4_0Q8_1: could not read the device's warp width\n");
        return;
    }
    gemvQ4_0Q8_1Kernel<<<out, WARPS * width>>>(W, Xq, Y, batch, in, out);
}
