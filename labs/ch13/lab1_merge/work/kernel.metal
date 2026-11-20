// Lab 13.1 starter (Metal Shading Language).
// The same merge as kernel.cu: write co_rank, merge_sequential and the three kernels.
//
// Two things are different in Metal:
//
// - co_rank and merge_sequential must read device memory (merge_basic, merge_partition)
//   and threadgroup memory (merge_tiled). MSL address spaces are disjoint and every
//   pointer names one, so one `const KV*` helper as in CUDA is not possible. Here each
//   helper is a template on the pointer type, as PyTorch's MPS merge sort does; Ptr is
//   `const device KV*` or `const threadgroup KV*`. MLX writes two overloads instead
//   (notes/today/ch13-facts.md C3).
// - The MSL specification does not restate C++'s rule that && skips its right operand
//   when the left one is false (ch13-facts C2). Do not rely on it to keep a read in
//   bounds: test the bounds first, and read inside the `if`.
//
// CUDA                          Metal
// extern __shared__ KV tile[]   threadgroup KV* tile [[threadgroup(0)]]   (sized by the host)
// __syncthreads()               threadgroup_barrier(mem_flags::mem_threadgroup)
// threadIdx.x                   [[thread_position_in_threadgroup]]
// blockIdx.x                    [[threadgroup_position_in_grid]]
// blockDim.x                    [[threads_per_threadgroup]]
#include <metal_stdlib>
using namespace metal;

// Same layout as ../merge.h (a file compiled at run time cannot include it).
struct KV { int key; int val; };
#define MERGE_BLOCK 128
// Same layout as MergeParams in launch.mm.
struct MergeParams { int m; int n; int total; int perThread; };

// Fig. 13.5: as in kernel.cu.
template <typename Ptr>
int co_rank(int k, Ptr A, int m, Ptr B, int n) {
    // TODO 1
    return 0;
}

// Fig. 13.2: reads from device or threadgroup memory, writes to device memory.
template <typename Ptr>
void merge_sequential(Ptr A, int m, Ptr B, int n, device KV* C) {
    // TODO 2
}

// Part (a), Fig. 13.9. One thread per perThread outputs.
kernel void merge_basic(device const KV*       A [[buffer(0)]],
                        device const KV*       B [[buffer(1)]],
                        device KV*             C [[buffer(2)]],
                        constant MergeParams&  p [[buffer(3)]],
                        uint tid [[thread_position_in_grid]]) {
    // TODO 3: as mergeBasicKernel. The position is unsigned and p's fields are int:
    //         convert once, so every comparison is between ints.
}

// Part (b), step 1: bounds[b] for b = 0 .. numTiles, one thread each.
kernel void merge_partition(device const KV*       A [[buffer(0)]],
                            device const KV*       B [[buffer(1)]],
                            device int*       bounds [[buffer(2)]],
                            constant MergeParams&  p [[buffer(3)]],
                            uint b [[thread_position_in_grid]]) {
    // TODO 4: as partitionKernel, with a tile of MERGE_BLOCK * p.perThread outputs.
}

// Part (b), step 2: one tile per threadgroup, staged in `tile`.
kernel void merge_tiled(device const KV*       A [[buffer(0)]],
                        device const KV*       B [[buffer(1)]],
                        device KV*             C [[buffer(2)]],
                        constant MergeParams&  p [[buffer(3)]],
                        device const int* bounds [[buffer(4)]],
                        threadgroup KV*     tile [[threadgroup(0)]],
                        uint tx  [[thread_position_in_threadgroup]],
                        uint bx  [[threadgroup_position_in_grid]],
                        uint ntx [[threads_per_threadgroup]]) {
    // TODO 5: as mergeTiledKernel. Every thread reaches the barrier.
}

// ---------------------------------------------------------------- Provided
// Lets the tests call your co_rank directly: out[k] = co_rank(k, A, m, B, n) for
// k = 0 .. m + n. Do not edit.
kernel void co_rank_test(device const KV*       A [[buffer(0)]],
                         device const KV*       B [[buffer(1)]],
                         device int*          out [[buffer(2)]],
                         constant MergeParams&  p [[buffer(3)]],
                         uint k [[thread_position_in_grid]]) {
    if (int(k) <= p.m + p.n) out[k] = co_rank(int(k), A, p.m, B, p.n);
}
