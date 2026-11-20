// Lab 5.1 starter (Metal Shading Language).
#include <metal_stdlib>
using namespace metal;

// ---------- provided probe 1: memory bandwidth ----------
kernel void vec_add(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    constant uint&      n [[buffer(3)]],
                    uint3 gid [[thread_position_in_grid]]) {
    if (gid.x < n) C[gid.x] = A[gid.x] + B[gid.x];
}

// ---------- provided probe 2: arithmetic throughput ----------
struct FmaParams { uint n; uint iters; float a; float b; };

kernel void fma_chain(device const float* in  [[buffer(0)]],
                      device float*       out [[buffer(1)]],
                      constant FmaParams&   p [[buffer(2)]],
                      uint3 gid [[thread_position_in_grid]]) {
    if (gid.x < p.n) {
        float x = in[gid.x];
        for (uint it = 0; it < p.iters; ++it) x = x * p.a + p.b;
        out[gid.x] = x;
    }
}

// ---------- your part: the naive matrix multiplication ----------
// Same kernel as in kernel.cu. `gid` is this thread's position in the whole
// grid, i.e. what CUDA code computes as blockIdx*blockDim + threadIdx.
kernel void matmul(device const float* M [[buffer(0)]],
                   device const float* N [[buffer(1)]],
                   device float*       P [[buffer(2)]],
                   constant uint&  Width [[buffer(3)]],
                   uint3 gid [[thread_position_in_grid]]) {
    // TODO: row/column from gid, bounds guard, dot product, one store.
}
