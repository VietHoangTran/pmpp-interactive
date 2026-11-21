// Lab X3.1 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Milestone M10's missing pieces: RMSNorm, RoPE and SwiGLU (part A), and a port review of an
// RMSNorm written in the structure of llama.cpp's kernels (part B).
//
// Not in the book: module X3 is ours. The block reduction is lab 10.3's (the book's two-stage
// reduction, section 10.8, Fig. 10.16, p. 243, with that lab's corrections: a typed lane mask,
// warpSize instead of 32, and a guarded second stage). What each kernel computes follows the
// projects read in notes/today/x3-facts.md: B2 (RMSNorm), B6 (the two RoPE pairings), B1 (SiLU
// and SwiGLU).
//
// Contract (README.md states it in full). Row-major float32, batch (token) first.
//   launchRmsNorm      y[r][c] = x[r][c] * (1 / sqrtf(mean over c' of x[r][c']^2 + eps)) * w[c]
//                      x, y: [rows x d]; w: [d]. y may be x (in place).
//   launchRope         in place on x: [rows x heads x headDim], headDim even. Row r is at
//                      position pos[r]; cosT, sinT: [maxPos x headDim/2], built by the host.
//                      Pair i of a head (0 <= i < headDim/2) is rotated by angle index i:
//                        mode 0 rotates elements (2i, 2i+1)            (llama.cpp NORM, llama2.c)
//                        mode 1 rotates elements (i, i + headDim/2)    (NeoX, HF rotate_half)
//                      (a, b) -> (a cos - b sin, a sin + b cos).
//   launchSwiGLU       out[i] = gate[i] / (1 + expf(-gate[i])) * up[i]; out may be gate.
#include <cstdint>

#define BLOCK_SIZE 256                          // part A: threads per block
#define MIN_WARP_SIZE 32                        // the narrowest warp any backend here has
#define MAX_WARPS (BLOCK_SIZE / MIN_WARP_SIZE)  // per-warp slots for a block of BLOCK_SIZE

// CUDA: unsigned. HIP: every _sync intrinsic requires a 64-bit mask (infra-facts B3).
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

__device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
__device__ unsigned int warpsInBlock() { return (blockDim.x + warpSize - 1) / warpSize; }

// Lab 10.3's warp sum (Fig. 10.13 with the typed mask and warpSize). Lane 0 gets the sum.
__device__ float warpReduceSum(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        val += __shfl_down_sync(full, val, stride);
    }
    return val;
}

// Lab 10.3's block sum: two stages, the second guarded and padded with 0, the result handed
// to every thread through scratch[0] behind a second barrier. scratch: MAX_WARPS floats.
__device__ float blockReduceSum(float val, float* scratch) {
    val = warpReduceSum(val);
    if (laneIdx() == 0) {
        scratch[warpIdx()] = val;
    }
    __syncthreads();
    if (warpIdx() == 0) {
        val = laneIdx() < warpsInBlock() ? scratch[laneIdx()] : 0.0f;
        val = warpReduceSum(val);
        if (laneIdx() == 0) {
            scratch[0] = val;
        }
    }
    __syncthreads();
    return scratch[0];
}

// ================================================================ part A
// ---------------------------------------------------------------- RMSNorm
// One block per row: a sum of squares (lab 10.3's row reduction with x*x for max), one scale
// per row, then one pass that scales and multiplies by the weight. In place is safe: every
// read of the first pass finishes before the reduction's barriers, and in the second pass each
// thread reads x[c] before it writes y[c], for its own c only.
__global__ void rmsNormRowsKernel(const float* x, const float* w, float* y, int d, float eps) {
    __shared__ float scratch[MAX_WARPS];
    const float* in = x + (size_t)blockIdx.x * d;
    float* out = y + (size_t)blockIdx.x * d;

    float s = 0.0f;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        const float v = in[c];
        s += v * v;
    }
    s = blockReduceSum(s, scratch);

    // llama2.c and llama.cpp's Metal kernel write 1 / sqrt; llama.cpp's CUDA kernel writes
    // rsqrtf, which NVIDIA documents as 2 ulp where sqrtf and the division are correctly
    // rounded by default (ch21-facts A10). Either passes the README's tolerance.
    const float scale = 1.0f / sqrtf(s / (float)d + eps);
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        out[c] = in[c] * scale * w[c];
    }
}

void launchRmsNorm(const float* x, const float* w, float* y, int rows, int d, float eps) {
    rmsNormRowsKernel<<<rows, BLOCK_SIZE>>>(x, w, y, d, eps);
}

// ---------------------------------------------------------------- RoPE
// One thread per rotated pair. Thread p owns pair i = p % half of head p / half; the angle
// comes from the host's table, so the kernel does two multiplies and an add per output and no
// trigonometry. Both elements of a pair belong to one thread, so in place needs no barrier.
__global__ void ropeKernel(float* x, const int* pos, const float* cosT, const float* sinT,
                           int rows, int heads, int headDim, int mode) {
    const int half = headDim / 2;
    const size_t pairs = (size_t)rows * heads * half;
    const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) {
        return;
    }
    const int i = (int)(p % half);           // pair within the head, and the angle index
    const size_t head = p / half;            // which (row, head) vector
    const int r = (int)(head / heads);
    float* v = x + head * headDim;
    const int a = mode == 0 ? 2 * i : i;
    const int b = mode == 0 ? 2 * i + 1 : i + half;
    const size_t t = (size_t)pos[r] * half + i;
    const float c = cosT[t];
    const float s = sinT[t];
    const float x0 = v[a];
    const float x1 = v[b];
    v[a] = x0 * c - x1 * s;
    v[b] = x0 * s + x1 * c;
}

void launchRope(float* x, const int* pos, const float* cosT, const float* sinT,
                int rows, int heads, int headDim, int mode) {
    const size_t pairs = (size_t)rows * heads * (headDim / 2);
    const unsigned int blocks = (unsigned int)((pairs + BLOCK_SIZE - 1) / BLOCK_SIZE);
    ropeKernel<<<blocks, BLOCK_SIZE>>>(x, pos, cosT, sinT, rows, heads, headDim, mode);
}

// ---------------------------------------------------------------- SwiGLU
// Chapter 2's elementwise kernel over two inputs: llama.cpp's op_silu times the up projection
// (x3-facts B1). With gate = -100, expf(100) overflows to +infinity and -100 / infinity is -0:
// a finite result under CUDA's default IEEE division. Metal's fast math leaves infinity
// undefined, so kernel.metal computes the same function without forming it.
__global__ void swigluKernel(const float* gate, const float* up, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];            // read before the write: out may be gate
        out[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

void launchSwiGLU(const float* gate, const float* up, float* out, int n) {
    swigluKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(gate, up, out, n);
}

// ================================================================ part B: the port, fixed
// An RMSNorm in the shape of llama.cpp's kernels (x3-facts B2): the launch picks the block
// size from the row length and a float4 path when it is safe; the reduction is two stages
// through one shared buffer. Unlike llama.cpp (one block per row), a fixed grid of PORT_GRID
// blocks walks the rows, so one block reuses the buffer row after row. The three FIX comments
// mark what the starter got wrong.
#define PORT_GRID 4      // blocks; block b normalises rows b, b + PORT_GRID, ...
#define PORT_SLOTS 64    // one slot per lane of the widest warp here (64 on AMD): stage 2 reads them all

__global__ void rmsNormPortKernel(const float* x, const float* w, float* y, int rows, int d, float eps,
                                  int vec) {
    __shared__ float s_sum[PORT_SLOTS];
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp = tid / warpSize;
    // The float4 view: the book's pointer cast, which no vendor documents (infra-facts A6). The
    // host takes this path only when d % 4 == 0 and every pointer is 16-byte aligned, so each
    // row starts on a float4 boundary and holds exactly n4 of them.
    const int n4 = (d + 3) / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* y4 = reinterpret_cast<float4*>(y);

    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        // FIX (a): stage 2 below reads one slot per lane of warp 0, but only the first
        // warpsInBlock() slots are written. Warp 0 zeroes them all first, as llama.cpp's Metal
        // kernel does; llama.cpp's CUDA kernel guards the read and pads with 0 instead
        // (x3-facts B2, ch10-facts D1). Either is right.
        if (warp == 0) {
            s_sum[lane] = 0.0f;
        }

        float sum = 0.0f;
        if (vec) {
            const float4* xr = x4 + (size_t)row * n4;
            for (int i = tid; i < n4; i += blockDim.x) {
                const float4 v = xr[i];
                sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        } else {
            const float* xr = x + (size_t)row * d;
            for (int i = tid; i < d; i += blockDim.x) {
                const float v = xr[i];
                sum += v * v;
            }
        }
        sum = warpReduceSum(sum);
        __syncthreads();                     // the zeros are in place before any slot is written
        if (lane == 0) {
            s_sum[warp] = sum;
        }
        __syncthreads();
        if (warp == 0) {
            sum = warpReduceSum(s_sum[lane]);
            if (lane == 0) {
                s_sum[0] = sum;
            }
        }
        __syncthreads();
        const float scale = 1.0f / sqrtf(s_sum[0] / (float)d + eps);

        if (vec) {
            const float4* xr = x4 + (size_t)row * n4;
            float4* yr = y4 + (size_t)row * n4;
            for (int i = tid; i < n4; i += blockDim.x) {
                const float4 v = xr[i];
                const float4 g = w4[i];
                yr[i] = make_float4(v.x * scale * g.x, v.y * scale * g.y, v.z * scale * g.z, v.w * scale * g.w);
            }
        } else {
            const float* xr = x + (size_t)row * d;
            float* yr = y + (size_t)row * d;
            for (int i = tid; i < d; i += blockDim.x) {
                yr[i] = xr[i] * scale * w[i];
            }
        }
        // FIX (b): every thread must have read s_sum[0] before the next row writes the buffer
        // again (the zeroing above, then the slots). llama.cpp's softmax puts the same barrier
        // between two reductions that share a buffer: "see #26385" (ch10-facts D1).
        __syncthreads();
    }
}

static bool aligned16(const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

void launchRmsNormPort(const float* x, const float* w, float* y, int rows, int d, float eps) {
    // llama.cpp's scalar-path rule (x3-facts B2): 256 threads for rows under 1024 elements,
    // else 1024.
    const int threads = d < 1024 ? 256 : 1024;
    const int grid = rows < PORT_GRID ? rows : PORT_GRID;
    // FIX (c): a row starts on a 16-byte boundary and is whole float4s only when d % 4 == 0.
    // llama.cpp checks ncols % 4 as well as the alignment (x3-facts B2).
    const bool vec = d % 4 == 0 && aligned16(x) && aligned16(w) && aligned16(y);
    rmsNormPortKernel<<<grid, threads>>>(x, w, y, rows, d, eps, vec ? 1 : 0);
}
