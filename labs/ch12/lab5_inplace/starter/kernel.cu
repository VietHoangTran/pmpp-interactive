// Lab 12.5 starter (CUDA; also built for HIP and for the CPU emulators). Optional lab.
// A stable filter of unsigned int keys (keep a key when key < threshold) in ONE kernel that
// reads every key once and works in place: input == output is allowed. README.md has the task.
//
// Book: exercises 2, 3 and 4 (p. 302) and section 12.7 (p. 300, Fig. 12.11), on chapter 11's
// dynamic block index and single look-back (Figs. 11.17 and 11.18, pp. 278-281).
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

// The lane mask and its popcount, as in lab 12.3 (64 bits on HIP, ch12-facts B1).
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

// Call this once per turn of a wait loop. It sleeps briefly on a GPU (CUDA: __nanosleep,
// ch11-facts A7; AMD: s_sleep, without which early MI100 parts hang, ch11-facts B3) and does
// nothing in the emulators.
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
    // TODO 1: take this block's index from *blockCounter (one thread, a device-scope atomic),
    //         share it with the block, and use it in place of blockIdx.x from here on.
    //         Block b owns keys [b * BLOCK_DIM * coarse, (b + 1) * BLOCK_DIM * coarse).

    // TODO 2: load the whole tile into dynamic shared memory (BLOCK_DIM * coarse keys, sized by
    //         the launch), in coalesced rounds. Then a barrier. In place, what does this barrier
    //         guarantee that the look-back below relies on?

    // TODO 3: compact the kept keys, in order, at the front of the same shared array, round by
    //         round, as lab 12.3's staged scatter does. Can a write in round r overwrite a key
    //         that nobody has read yet? Where must the barriers go so that it cannot?

    // TODO 4: the single look-back. One thread waits until flags[bid - 1] is set, reads
    //         inclusive[bid - 1] (the kept keys of all earlier blocks), writes this block's
    //         inclusive[bid], then sets flags[bid]. Choose the memory order of each flag access:
    //         what must the flag read keep from moving before it, and the flag write keep from
    //         moving after it? Call spinPause() in the wait. Block 0 waits for nobody.

    // TODO 5: share the start of this block's run, barrier, then write the run with
    //         consecutive threads on consecutive keys. The last block writes *outputSize.
}

// The driver zeroes *blockCounter and flags[0 .. blocks) before every call, and sizes flags and
// inclusive for ceil(n / (BLOCK_DIM * coarse)) blocks.
void launchStableFilterSinglePass(const unsigned int* input, unsigned int* output, unsigned int n,
                                  unsigned int threshold, unsigned int* outputSize, unsigned int* blockCounter,
                                  unsigned int* flags, unsigned int* inclusive, int coarse) {
    // TODO 6: one block per tile, BLOCK_DIM threads, and the tile's bytes of dynamic shared memory.
}
