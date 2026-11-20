// Lab 13.1 reference solution (Metal Shading Language).
// The same merge as kernel.cu, with the same differences from the page, and two that
// belong to Metal:
//
// - co_rank and merge_sequential run on device memory (merge_basic, merge_partition)
//   and on threadgroup memory (merge_tiled). MSL address spaces are disjoint and every
//   pointer names one, so a single `const KV*` helper as in CUDA is not possible. Here
//   each helper is a template on the pointer type, as PyTorch's MPS merge sort does;
//   MLX writes two overloads instead (notes/today/ch13-facts.md C3).
// - The MSL specification does not restate C++'s rule that && skips its right operand
//   (ch13-facts C2), so co_rank reads A and B only inside an `if` whose condition has
//   already checked the bounds, never on the right of an &&.
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

// Fig. 13.5. Ptr is `const device KV*` or `const threadgroup KV*`.
template <typename Ptr>
int co_rank(int k, Ptr A, int m, Ptr B, int n) {
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : k - n;
    int j_low = 0 > (k - m) ? 0 : k - m;
    int delta;
    bool active = true;
    while (active) {
        bool iTooHigh = false;
        bool jTooHigh = false;
        if (i > 0 && j < n) iTooHigh = A[i - 1].key > B[j].key;
        if (!iTooHigh && j > 0 && i < m) jTooHigh = B[j - 1].key >= A[i].key;
        if (iTooHigh) {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (jTooHigh) {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}

// Fig. 13.2. Reads from device or threadgroup memory, writes to device memory.
template <typename Ptr>
void merge_sequential(Ptr A, int m, Ptr B, int n, device KV* C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i].key <= B[j].key) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    if (i == m) {
        while (j < n) C[k++] = B[j++];
    } else {
        while (i < m) C[k++] = A[i++];
    }
}

// Part (a), Fig. 13.9.
kernel void merge_basic(device const KV*       A [[buffer(0)]],
                        device const KV*       B [[buffer(1)]],
                        device KV*             C [[buffer(2)]],
                        constant MergeParams&  p [[buffer(3)]],
                        uint tid [[thread_position_in_grid]]) {
    int k_curr = int(tid) * p.perThread;
    if (k_curr >= p.total) return;
    int k_next = k_curr + p.perThread < p.total ? k_curr + p.perThread : p.total;
    int i_curr = co_rank(k_curr, A, p.m, B, p.n);
    int i_next = co_rank(k_next, A, p.m, B, p.n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential(A + i_curr, i_next - i_curr, B + j_curr, j_next - j_curr, C + k_curr);
}

// Part (b), step 1: one co-rank per tile boundary.
kernel void merge_partition(device const KV*       A [[buffer(0)]],
                            device const KV*       B [[buffer(1)]],
                            device int*       bounds [[buffer(2)]],
                            constant MergeParams&  p [[buffer(3)]],
                            uint b [[thread_position_in_grid]]) {
    int tileSize = MERGE_BLOCK * p.perThread;
    int numTiles = (p.total + tileSize - 1) / tileSize;
    if (int(b) <= numTiles) {
        int k = int(b) * tileSize < p.total ? int(b) * tileSize : p.total;
        bounds[b] = co_rank(k, A, p.m, B, p.n);
    }
}

// Part (b), step 2: one tile per threadgroup, staged in threadgroup memory.
kernel void merge_tiled(device const KV*       A [[buffer(0)]],
                        device const KV*       B [[buffer(1)]],
                        device KV*             C [[buffer(2)]],
                        constant MergeParams&  p [[buffer(3)]],
                        device const int* bounds [[buffer(4)]],
                        threadgroup KV*     tile [[threadgroup(0)]],
                        uint tx  [[thread_position_in_threadgroup]],
                        uint bx  [[threadgroup_position_in_grid]],
                        uint ntx [[threads_per_threadgroup]]) {
    int tileSize = int(ntx) * p.perThread;
    int C_curr = int(bx) * tileSize;
    int C_next = C_curr + tileSize < p.total ? C_curr + tileSize : p.total;
    int A_curr = bounds[bx];
    int A_next = bounds[bx + 1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int C_length = C_next - C_curr;

    threadgroup KV* A_S = tile;
    threadgroup KV* B_S = tile + A_length;
    for (int i = 0; i < A_length; i += int(ntx)) {
        if (i + int(tx) < A_length) A_S[i + tx] = A[A_curr + i + int(tx)];
    }
    for (int i = 0; i < B_length; i += int(ntx)) {
        if (i + int(tx) < B_length) B_S[i + tx] = B[B_curr + i + int(tx)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int c_curr = int(tx) * p.perThread;
    int c_next = c_curr + p.perThread;
    c_curr = c_curr < C_length ? c_curr : C_length;
    c_next = c_next < C_length ? c_next : C_length;
    threadgroup const KV* As = A_S;
    threadgroup const KV* Bs = B_S;
    int a_curr = co_rank(c_curr, As, A_length, Bs, B_length);
    int a_next = co_rank(c_next, As, A_length, Bs, B_length);
    int b_curr = c_curr - a_curr;
    int b_next = c_next - a_next;
    merge_sequential(As + a_curr, a_next - a_curr, Bs + b_curr, b_next - b_curr, C + C_curr + c_curr);
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
