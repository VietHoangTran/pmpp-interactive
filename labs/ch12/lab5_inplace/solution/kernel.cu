// Lab 12.5 reference solution (CUDA; also built for HIP and for the CPU emulators). Optional lab.
// A stable filter of unsigned int keys (keep a key when key < threshold) in ONE kernel that
// reads every key once and works in place: input == output is allowed.
//
// Contract: n >= 1. With no keys there is no last block to write *outputSize, and the launcher
// would launch zero blocks, which HIP rejects (notes/today/ch02-facts.md B4; CUDA's documentation
// does not state the case). The driver never calls with n = 0.
//
// Book: exercises 2, 3 and 4 (p. 302) and section 12.7 (p. 300, Fig. 12.11), built on chapter
// 11's single-pass scan: the dynamic block index and the single look-back (Figs. 11.17 and
// 11.18, pp. 278-281).
//
// One block per tile of BLOCK_DIM * coarse keys:
//   1. thread 0 takes the next block index from *blockCounter; the block uses it, not blockIdx.x
//   2. the block loads its whole tile into shared memory (coalesced rounds); barrier
//   3. it compacts its kept keys, in order, at the front of the same shared array: per round, a
//      warp vote gives each kept lane its rank, and a scan of the warp totals gives the rest
//   4. single look-back: thread 0 waits until block bid - 1 has published the count of all kept
//      keys before it (acquire), publishes its own running total (release), and shares the
//      start of its run with the block
//   5. the block writes its run, consecutive threads on consecutive keys; the last block writes
//      *outputSize
//
// Why it is safe in place (section 12.7, and notes/today/ch12-facts.md A9). A key never moves
// to a higher index, so a block's writes land in its own tile or earlier ones. Its own tile is
// read completely before the barrier in step 2. An earlier block's tile was read completely
// before that block's release in step 4, and this block's acquire in step 4 comes after it
// through the chain of flags, so every earlier block has finished reading before this block
// writes. That needs the release and the acquire: with relaxed flags the chain orders nothing.
//
// Why waiting is allowed at all: on CUDA, a block that has taken an index has started, and the
// execution model then promises its threads eventually make progress (ch12-facts A10). Waiting
// only on smaller dynamic indices therefore ends. AMD documents no such rule (B6, ch11-facts B3):
// HIP is unverified. Apple documents none either (C4): there is no Metal version (README.md).
//
// Where this file differs from the book:
//   - Fig. 11.17 reads the flag with fetch_add(0, acquire). This uses load(acquire), the read-only
//     form NVIDIA's own message-passing example uses (ch11-facts A5).
//   - Fig. 11.17's spin loop has an empty body. This one pauses: __nanosleep on CUDA (ch11-facts
//     A7), s_sleep on AMD, where rocPRIM found early MI100 parts hang without it (ch11-facts B3).
//   - Fig. 11.17's leader is the last thread of the block; here thread 0 already holds the count.
//   - Warp votes are __ballot_sync over the whole warp with a typed mask, as in lab 12.3.
#include <cuda/atomic>

// Threads per block. The race-detecting emulator runs blocks of more than 128 threads one at a
// time; it gets 64 so that several blocks, and their look-back, really run at once there.
#if defined(PMPP_EMU_RACE)
#define BLOCK_DIM 64
#else
#define BLOCK_DIM 256
#endif
#define MIN_WARP_SIZE 32
#define MAX_WARPS (BLOCK_DIM / MIN_WARP_SIZE)

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif
static __device__ __forceinline__ int maskPopc(lane_mask_t m) {
#if defined(__HIPCC__)
    return __popcll(m);
#else
    return __popc(m);
#endif
}
// Any other argument type (an unsigned int holding a HIP ballot, say) is a compile error.
template <typename T> static __device__ __forceinline__ int maskPopc(T m) = delete;

__device__ __forceinline__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
__device__ __forceinline__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ __forceinline__ unsigned int warpsInBlock() { return blockDim.x / warpSize; }

// One pause in the look-back's wait. Plain spinning in the emulators and the front-end check.
__device__ __forceinline__ void spinPause() {
#if defined(__HIPCC__)
    __builtin_amdgcn_s_sleep(1);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(PMPP_EMU) && !defined(PMPP_CUDA_FRONTEND_CHECK)
    __nanosleep(32);
#endif
}

__global__ void stableFilterSinglePassKernel(const unsigned int* input, unsigned int* output, unsigned int n,
                                             unsigned int threshold, unsigned int* outputSize,
                                             unsigned int* blockCounter, unsigned int* flags,
                                             unsigned int* inclusive, unsigned int coarse) {
    extern __shared__ unsigned int tile_s[];   // BLOCK_DIM * coarse: the tile, then its kept keys
    __shared__ unsigned int warpTotals_s[MAX_WARPS];
    __shared__ unsigned int bid_s;
    __shared__ unsigned int runStart_s;

    // 1. Dynamic block index (Fig. 11.18, lines 04-11).
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> counter(*blockCounter);
        bid_s = counter.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    const unsigned int bid = bid_s;
    const unsigned int tileStart = bid * BLOCK_DIM * coarse;

    // 2. Load the tile, round by round (coalesced).
    for (unsigned int r = 0; r < coarse; ++r) {
        const unsigned int i = tileStart + r * BLOCK_DIM + threadIdx.x;
        tile_s[r * BLOCK_DIM + threadIdx.x] = i < n ? input[i] : 0u;
    }
    __syncthreads();   // every read of this block's input is done

    // 3. Compact in shared memory. A kept key moves to a position at or below its own, and every
    //    thread reads its key of round r before the round's first barrier, so the writes after
    //    it never overwrite a key that has not been read.
    const lane_mask_t fullMask = ~(lane_mask_t)0;
    const lane_mask_t lanesBelow = ((lane_mask_t)1 << laneIdx()) - 1;
    unsigned int staged = 0;   // kept keys so far; the same in every thread
    for (unsigned int r = 0; r < coarse; ++r) {
        const unsigned int i = tileStart + r * BLOCK_DIM + threadIdx.x;
        const unsigned int key = tile_s[r * BLOCK_DIM + threadIdx.x];
        const bool keep = i < n && key < threshold;
        const lane_mask_t kept = __ballot_sync(fullMask, keep);
        if (laneIdx() == 0) {
            warpTotals_s[warpIdx()] = maskPopc(kept);
        }
        __syncthreads();
        unsigned int warpsBefore = 0, roundTotal = 0;
        for (unsigned int w = 0; w < warpsInBlock(); ++w) {
            const unsigned int t = warpTotals_s[w];
            warpsBefore += w < warpIdx() ? t : 0u;
            roundTotal += t;
        }
        if (keep) {
            tile_s[staged + warpsBefore + maskPopc(kept & lanesBelow)] = key;
        }
        staged += roundTotal;
        __syncthreads();
    }

    // 4. Single look-back (Fig. 11.17): inclusive[b] = kept keys in blocks 0 .. b.
    if (threadIdx.x == 0) {
        unsigned int before = 0;
        if (bid > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> prevFlag(flags[bid - 1]);
            while (prevFlag.load(cuda::memory_order_acquire) == 0) {
                spinPause();
            }
            before = inclusive[bid - 1];   // ordered after the acquire
        }
        inclusive[bid] = before + staged;   // ordered before the release
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> myFlag(flags[bid]);
        myFlag.store(1u, cuda::memory_order_release);
        runStart_s = before;
    }
    __syncthreads();   // the whole block writes only after thread 0's acquire

    // 5. Write the run (coalesced).
    const unsigned int runStart = runStart_s;
    for (unsigned int j = threadIdx.x; j < staged; j += BLOCK_DIM) {
        output[runStart + j] = tile_s[j];
    }
    if (bid == gridDim.x - 1 && threadIdx.x == 0) {
        *outputSize = runStart + staged;
    }
}

// The driver zeroes *blockCounter and flags[0 .. blocks) before every call, and sizes flags and
// inclusive for ceil(n / (BLOCK_DIM * coarse)) blocks.
void launchStableFilterSinglePass(const unsigned int* input, unsigned int* output, unsigned int n,
                                  unsigned int threshold, unsigned int* outputSize, unsigned int* blockCounter,
                                  unsigned int* flags, unsigned int* inclusive, int coarse) {
    const unsigned int tile = BLOCK_DIM * (unsigned int)coarse;
    const unsigned int blocks = (n + tile - 1) / tile;
    const size_t tileBytes = (size_t)tile * sizeof(unsigned int);
    stableFilterSinglePassKernel<<<blocks, BLOCK_DIM, tileBytes>>>(input, output, n, threshold, outputSize,
                                                                  blockCounter, flags, inclusive,
                                                                  (unsigned int)coarse);
}
