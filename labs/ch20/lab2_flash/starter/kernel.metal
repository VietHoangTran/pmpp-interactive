// Lab 20.2 starter (Metal Shading Language): the same kernel as kernel.cu,
// flash_attention_forward. One threadgroup of 512 threads (16 SIMD-groups of 32) per panel
// of B_r = 32 rows of Q (grid x), one per (sequence, head) (grid y). launch.mm chooses B_c
// for this device, compiles this file with TILE_WIDTH = B_c, and dispatches it.
//
// CUDA                                  Metal
// __shared__ float t[..] (in kernel)    threadgroup float t[..] (in the kernel; helpers get pointers, C2)
// __syncthreads()                       threadgroup_barrier(mem_flags::mem_threadgroup)
// __syncwarp()                          simdgroup_barrier(mem_flags::mem_threadgroup)   (C4)
// __shfl_sync(m, v, src, 32)            simd_broadcast(v, src)   (src the same in every lane, C3)
// xor butterfly of max / sum            simd_max(v) / simd_sum(v)   (result in every lane, C3)
// threadIdx.x % 32, threadIdx.x / 32    [[thread_index_in_simdgroup]], [[simdgroup_index_in_threadgroup]]
// expf, logf                            exp, log
//
// Notes (notes/today/ch20-facts.md):
//   - The book's B_c = 32 tiles take 37,376 bytes of threadgroup memory; Apple's limit is
//     32 KB (C1). Your launch.mm picks the largest B_c in {32, 16, 8} that fits this device.
//   - The lane layout needs SIMD-groups of exactly 32 threads: your launch.mm checks the
//     pipeline's threadExecutionWidth. Apple documents no fixed width (C3).
//   - No infinity anywhere: under fast math its behaviour is undefined (C5). Test masks by
//     index, use -FLT_MAX as the identity of max and start the running max at -FLT_MAX/2.
//   - Apple documents no threadgroup-memory banks; addr() is the book's padding, kept so that
//     the two sources match.
#include <metal_stdlib>
using namespace metal;

#ifndef TILE_WIDTH
#error "launch.mm defines TILE_WIDTH, the B_c it chose for this device"
#endif

#define WARP_SIZE 32                       // the book's warp; launch.mm checks the SIMD-group width
#define BLOCK_SIZE 512
#define N_WARPS (BLOCK_SIZE / WARP_SIZE)
#define B_r 32
#define B_r_warp (B_r / N_WARPS)
#define B_c TILE_WIDTH
#define HEAD_DIM 128
#define D_SIZE (HEAD_DIM / WARP_SIZE)
#define LOG_NUM_BANKS 5
#define KT_SIZE (B_c * HEAD_DIM + ((B_c * HEAD_DIM) >> LOG_NUM_BANKS))

#if B_c > WARP_SIZE || WARP_SIZE % B_c != 0
#error "each lane owns at most one column of a tile here: B_c (TILE_WIDTH) must be 8, 16 or 32"
#endif

struct FlashArgs { uint n; float scale; };   // must match launch.mm

inline int addr(int x) { return x + (x >> LOG_NUM_BANKS); }

// ---------------------------------------------------------------- provided helpers
inline void load_Q(device const float* Q, thread float Q_i[B_r_warp][D_SIZE], int i, int n, int warp, int lane) {
    for (int ii = 0; ii < B_r_warp; ii++) {
        const int row = B_r * i + B_r_warp * warp + ii;
        for (int dd = lane, ddd = 0; dd < HEAD_DIM; dd += WARP_SIZE, ddd++) {
            Q_i[ii][ddd] = row < n ? Q[ulong(row) * HEAD_DIM + dd] : 0.0f;
        }
    }
}

inline void load_KT_and_V(device const float* K, device const float* V, threadgroup float* KT_j,
                          threadgroup float V_j[][HEAD_DIM], int j, int n, int t, int threads) {
    for (int jj = 0; jj < B_c; jj++) {
        const int col = B_c * j + jj;
        for (int dd = t; dd < HEAD_DIM; dd += threads) {
            KT_j[addr(dd * B_c + jj)] = col < n ? K[ulong(col) * HEAD_DIM + dd] : 0.0f;
            V_j[jj][dd] = col < n ? V[ulong(col) * HEAD_DIM + dd] : 0.0f;
        }
    }
}

inline void store_O(device float* O, device float* L, thread const float O_i[B_r_warp][D_SIZE],
                    thread const float* D_i, thread const float* m_i, int i, int n, int warp, int lane) {
    for (int ii = 0; ii < B_r_warp; ii++) {
        const int row = B_r * i + B_r_warp * warp + ii;
        if (row >= n) continue;
        for (int ddd = 0; ddd < D_SIZE; ddd++) {
            const int col = ddd * WARP_SIZE + lane;
            O[ulong(row) * HEAD_DIM + col] = O_i[ii][ddd] / D_i[ii];
        }
        if (lane == 0) {
            L[row] = m_i[ii] + log(D_i[ii]);
        }
    }
}

// ---------------------------------------------------------------- your part
// The same nine steps as TODO 1 to 9 in kernel.cu, in Metal's vocabulary (table above).
inline void initialize(thread float O_i[B_r_warp][D_SIZE], thread float* D_i, thread float* m_i) {
    // TODO 1
}

inline float compute_S_and_max(thread const float Q_i[B_r_warp][D_SIZE], threadgroup const float* KT_j,
                               threadgroup float S_i[][B_c], int i, int j, int ii, int n, float scale, int warp,
                               int lane) {
    // TODO 2 to 4. simd_broadcast wants the source lane to be the same in every lane, and
    // every lane must reach it, a lane with no column too.
    return -FLT_MAX;
}

inline float update_m_and_D(thread float* m_i, thread float* D_i, float curr_max_warp, int ii) {
    // TODO 5
    return m_i[ii];
}

inline void compute_P_and_update_D(threadgroup float S_i[][B_c], thread const float* m_i, thread float* D_i, int i,
                                   int j, int ii, int n, int warp, int lane) {
    // TODO 6 to 8. Whether a SIMD-group runs in lockstep is not documented (C4).
}

inline void compute_O(threadgroup const float S_i[][B_c], threadgroup const float V_j[][HEAD_DIM],
                      thread float O_i[B_r_warp][D_SIZE], int ii, float last_m, thread const float* m_i, int warp,
                      int lane) {
    // TODO 9
}

kernel void flash_attention_forward(device const float* Q    [[buffer(0)]],
                                    device const float* K    [[buffer(1)]],
                                    device const float* V    [[buffer(2)]],
                                    device float*       O    [[buffer(3)]],
                                    device float*       L    [[buffer(4)]],
                                    constant FlashArgs& args [[buffer(5)]],
                                    uint2  tid   [[thread_position_in_threadgroup]],
                                    uint2  tsize [[threads_per_threadgroup]],
                                    uint2  tg    [[threadgroup_position_in_grid]],
                                    uint2  ntg   [[threadgroups_per_grid]],
                                    ushort lane  [[thread_index_in_simdgroup]],
                                    ushort warp  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float KT_j[KT_SIZE];
    threadgroup float S_i[B_r][B_c];
    threadgroup float V_j[B_c][HEAD_DIM];

    const int n = int(args.n);
    const float scale = args.scale;
    const ulong head = tg.y;
    device const float* Qh = Q + head * ulong(n) * HEAD_DIM;
    device const float* Kh = K + head * ulong(n) * HEAD_DIM;
    device const float* Vh = V + head * ulong(n) * HEAD_DIM;
    device float* Oh = O + head * ulong(n) * HEAD_DIM;
    device float* Lh = L + head * ulong(n);

    const int T_r = (n + B_r - 1) / B_r;
    const int T_c = (n + B_c - 1) / B_c;

    // TODO 10: the two loops, as in kernel.cu. The panel loop starts at tg.x and strides by
    //          ntg.x; load_KT_and_V takes tid.x and tsize.x as its thread index and count.
}
