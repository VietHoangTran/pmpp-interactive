// Lab X3.1 starter (CUDA; also built for HIP and for the CPU emulators).
// Milestone M10's missing pieces: RMSNorm, RoPE and SwiGLU (part A), and a port review of an
// RMSNorm written in the structure of llama.cpp's kernels (part B).
//
// Not in the book: module X3 is ours. The block reduction is lab 10.3's (the book's two-stage
// reduction, section 10.8, Fig. 10.16, p. 243, with that lab's corrections). What each kernel
// computes follows the projects read in notes/today/x3-facts.md: B2 (RMSNorm), B6 (the two
// RoPE pairings), B1 (SiLU and SwiGLU). Read README.md first.
//
// Contract. Row-major float32, batch (token) first.
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

// The course's CUDA front-end check (labs/pmpp.py) declares no CUDA Math API functions, so it
// is given the device declarations used here. nvcc, hipcc and the emulators never see them.
#if defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float expf(float x);
__device__ float sqrtf(float x);
#endif

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
// Provided: part B uses it too.
__device__ float warpReduceSum(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        val += __shfl_down_sync(full, val, stride);
    }
    return val;
}

// ================================================================ part A
// TODO 1: lab 10.3's block sum (bring yours): stage 1 in every warp, one slot per warp in
//         shared memory, stage 2 in warp 0, and the result handed to EVERY thread, because
//         every thread needs the scale. Remember what lab 10.3 learnt about the lanes of the
//         stage-2 warp that have no warp result to read.

// ---------------------------------------------------------------- RMSNorm
__global__ void rmsNormRowsKernel(const float* x, const float* w, float* y, int d, float eps) {
    // TODO 2: block blockIdx.x owns row blockIdx.x. A loop strided by blockDim.x adds the
    //         squares of the row; the block sum; scale = 1 / sqrtf(sum / d + eps); a second
    //         strided loop writes x * scale * w. y may be x: check that no thread can write an
    //         element another thread has still to read.
}

void launchRmsNorm(const float* x, const float* w, float* y, int rows, int d, float eps) {
    // TODO 3: one block of BLOCK_SIZE threads per row.
}

// ---------------------------------------------------------------- RoPE
__global__ void ropeKernel(float* x, const int* pos, const float* cosT, const float* sinT,
                           int rows, int heads, int headDim, int mode) {
    // TODO 4: one thread per rotated pair (rows * heads * headDim / 2 of them). From the
    //         thread's index find the head vector it works in, its row (for pos[row]) and its
    //         pair i; then the two element offsets, which depend on `mode`; then rotate with
    //         cosT and sinT at [pos[row] * headDim / 2 + i]. Both elements of a pair belong to
    //         one thread, so in place is safe if each thread reads both before it writes.
}

void launchRope(float* x, const int* pos, const float* cosT, const float* sinT,
                int rows, int heads, int headDim, int mode) {
    // TODO 5: enough blocks of BLOCK_SIZE for every pair.
}

// ---------------------------------------------------------------- SwiGLU
__global__ void swigluKernel(const float* gate, const float* up, float* out, int n) {
    // TODO 6: chapter 2's elementwise kernel over two inputs. out may be gate. The test feeds
    //         gate = -100 and +100: work out what expf(100) is in float, and what your formula
    //         does with it.
}

void launchSwiGLU(const float* gate, const float* up, float* out, int n) {
    // TODO 7
}

// ================================================================ part B: the port
// An RMSNorm in the shape of llama.cpp's kernels (x3-facts B2), as a colleague ported it: the
// launch picks the block size from the row length and a float4 path when it judges it safe;
// the reduction is two stages through one shared buffer. Unlike llama.cpp (one block per row),
// a fixed grid of PORT_GRID blocks walks the rows, so one block reuses the buffer row after row.
//
// It compiles, and it passes some of the tests. It has THREE independent defects. Find and fix
// all three (README.md, part B). Everything above this line is not part of the hunt.
#define PORT_GRID 4      // blocks; block b normalises rows b, b + PORT_GRID, ...
#define PORT_SLOTS 64    // one slot per lane of the widest warp here (64 on AMD): stage 2 reads them all

__global__ void rmsNormPortKernel(const float* x, const float* w, float* y, int rows, int d, float eps,
                                  int vec) {
    __shared__ float s_sum[PORT_SLOTS];
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp = tid / warpSize;
    // The float4 view: the book's pointer cast, which no vendor documents (infra-facts A6).
    const int n4 = (d + 3) / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* y4 = reinterpret_cast<float4*>(y);

    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
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
    }
}

static bool aligned16(const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

void launchRmsNormPort(const float* x, const float* w, float* y, int rows, int d, float eps) {
    // llama.cpp's scalar-path rule (x3-facts B2): 256 threads for rows under 1024 elements,
    // else 1024.
    const int threads = d < 1024 ? 256 : 1024;
    const int grid = rows < PORT_GRID ? rows : PORT_GRID;
    // The float4 path reads and writes whole float4s, which must be 16-byte aligned.
    const bool vec = aligned16(x) && aligned16(w) && aligned16(y);
    rmsNormPortKernel<<<grid, threads>>>(x, w, y, rows, d, eps, vec ? 1 : 0);
}
