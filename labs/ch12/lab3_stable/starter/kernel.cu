// Lab 12.3 starter (CUDA; also built for HIP and for the CPU emulators).
// Two stable, out-of-place filters of unsigned int keys: keep a key when key < threshold, and
// keep the kept keys in input order. Book: sections 12.5 and 12.6 (pp. 297-300), exercises 1,
// 3 and 4 (p. 302). README.md has the whole task.
//
// Part A, exercise 1: keepFlagsKernel, then the provided scan, then scatterKernel.
// Part B, exercises 3 and 4: countKeptKernel, then the provided scan of the per-block counts,
//         then stagedScatterKernel, which gathers each block's kept keys in shared memory and
//         writes them as one contiguous run.
//
// Neither part may make one block wait for another: that is what keeps them portable.
#define BLOCK_DIM 256                           // threads per block, every kernel in this file
#define MIN_WARP_SIZE 32                        // the narrowest warp of any backend here
#define MAX_WARPS (BLOCK_DIM / MIN_WARP_SIZE)   // per-warp slots in shared memory

// Provided in ../shared-g12/scan_u32.cu: out[0] = 0, out[i] = in[0] + ... + in[i-1].
// scratch holds scanScratchElems(n) elements; the driver allocates it for you.
void launchExclusiveScanU32(const unsigned int* in, unsigned int* out, unsigned int n, unsigned int* scratch);

// The lane mask and its popcount. CUDA: 32 bits and __popc. HIP: every _sync intrinsic takes a
// 64-bit mask and __ballot_sync returns one, which __popc would truncate (ch12-facts B1).
// Use lane_mask_t for every mask and maskPopc to count one; never write 0xffffffff.
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

// The book's helpers (chapter 10), with warpSize in place of the constant 32.
__device__ __forceinline__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
__device__ __forceinline__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ __forceinline__ unsigned int warpsInBlock() { return blockDim.x / warpSize; }

// ======================================================================== part A (exercise 1)

__global__ void keepFlagsKernel(const unsigned int* input, unsigned int* keep, unsigned int n,
                                unsigned int threshold) {
    // TODO A1: one thread per key: keep[i] is 1 if the key is kept, 0 if not.
    //          n is not a multiple of BLOCK_DIM in general.
}

__global__ void scatterKernel(const unsigned int* input, const unsigned int* keep, const unsigned int* offset,
                              unsigned int* output, unsigned int n, unsigned int* outputSize) {
    // TODO A2: one thread per key. Where does a kept key go?
    // TODO A3: exactly one thread writes *outputSize. Which thread, and what does it know that
    //          gives the count of kept keys? (Fig. 12.8's last lines, p. 298)
}

void launchStableFilterFlags(const unsigned int* input, unsigned int* output, unsigned int n,
                             unsigned int threshold, unsigned int* outputSize, unsigned int* keep,
                             unsigned int* offset, unsigned int* scanScratch) {
    // TODO A4: the three steps, in order. Launches on the same stream run one after another,
    //          and each sees the previous one's writes.
}

// ======================================================================== part B (exercises 3, 4)
// Block b owns the tile of keys [b * BLOCK_DIM * coarse, (b + 1) * BLOCK_DIM * coarse). It walks
// the tile in `coarse` rounds; in round r, thread t takes key tileStart + r * BLOCK_DIM + t, so
// every round's loads are coalesced.

__global__ void countKeptKernel(const unsigned int* input, unsigned int n, unsigned int threshold,
                                unsigned int* blockCounts, unsigned int coarse) {
    // TODO B1: count the kept keys of each round one warp at a time, with a single vote per
    //          warp per round. Have every lane vote, kept or not, past the end or not, with a
    //          full mask of type lane_mask_t: in uniform control flow the vote is exactly the
    //          kept lanes. (Fig. 12.3 calls __activemask() inside the branch; NVIDIA now warns
    //          that it does not tell you which lanes took a branch, ch12-facts A1.)
    // TODO B2: add the warps' counts together (shared memory, one barrier) and write this
    //          block's count to blockCounts[blockIdx.x].
}

__global__ void stagedScatterKernel(const unsigned int* input, unsigned int* output, unsigned int n,
                                    unsigned int threshold, const unsigned int* blockOffsets,
                                    unsigned int* outputSize, unsigned int coarse) {
    // TODO B3: the staging buffer is dynamic shared memory: BLOCK_DIM * coarse keys, sized by
    //          the launch. Declare it.
    // TODO B4: round by round, give every kept key its position in the staging buffer:
    //          (kept keys staged in earlier rounds) + (kept keys in earlier warps of this round)
    //          + (kept lanes below this one in its warp). The last term comes from the vote and a
    //          mask of the lanes below (the binary prefix sum of p. 293); the middle one needs
    //          every warp's total, so it goes through shared memory and a barrier.
    // TODO B5: one more barrier per round. What would the next round overwrite too early?
    // TODO B6: consecutive threads write consecutive staged keys to the block's run, which
    //          starts at output[blockOffsets[blockIdx.x]]. The last block writes *outputSize.
}

void launchStableFilterStaged(const unsigned int* input, unsigned int* output, unsigned int n,
                              unsigned int threshold, unsigned int* outputSize, unsigned int* blockCounts,
                              unsigned int* blockOffsets, unsigned int* scanScratch, int coarse) {
    // TODO B7: one block per tile of BLOCK_DIM * coarse keys (round up). Count, scan the counts
    //          into blockOffsets with the provided scan, then scatter, giving the scatter kernel
    //          its dynamic shared memory in bytes as the third launch parameter.
}
