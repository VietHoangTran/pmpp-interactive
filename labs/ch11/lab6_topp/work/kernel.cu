// Lab 11.6 starter (CUDA; also built for HIP and for the CPU emulators).
// Milestone M5, part 1: the top-p (nucleus) cumulative mass and keep mask.
//
// Book: the exclusive scan (section 11.1, p. 253) and lab 11.3's row scan, built from the
// coarsened, register-tiled scan of Figs. 11.12 and 11.13 (pp. 268 and 271) and the warp and
// block scans of Figs. 11.8 and 11.9 (p. 264). The top-p rule is not in the book. Read
// README.md first: it states the contract, the tie rule and the tolerance.
//
// Contract:
//   probs is [rows x cols] float32, row-major, batch first; each row sorted in descending order
//   and summing to 1 within rounding; rows >= 1, cols >= 1 (any value). For every row r and
//   token i:
//     massBefore[r][i] = probs[r][0] + ... + probs[r][i-1]        (massBefore[r][0] = 0)
//     keep[r][i]       = 1 exactly when i == 0, or p >= 1, or massBefore[r][i] < p; else 0
//     kept[r]          = the number of ones in keep[r]
//   One block of BLOCK_DIM threads per row.

#define BLOCK_DIM 256                           // threads per block; one block per row
#define COARSE_FACTOR 4                         // elements per thread per chunk (p. 269)
#define MIN_WARP_SIZE 32                        // the narrowest warp a CUDA or HIP device has
#define MAX_WARPS (BLOCK_DIM / MIN_WARP_SIZE)   // enough per-warp slots on every backend

// From lab 11.3: the shuffle mask has its own type (HIP requires 64 bits), and the warp
// width is warpSize, never a literal 32.
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

__device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }

// TODO 1: the warp-level pieces: your fixed warpScan from lab 11.3, and a warp sum for
//         integers (lab 10.1's warp_reduce, on int) for the counts.

__global__ void TopPMaskKernel(const float* probs, float* massBefore, unsigned char* keep, int* kept,
                               unsigned int cols, float p) {
    // TODO 2: block blockIdx.x handles row blockIdx.x. Start from your fixed lab 11.3 kernel:
    //         its chunk loop and carried prefix produce massBefore. Then, while each token's
    //         mass is still in a register, apply the contract's keep rule to it, write keep,
    //         and count what this thread kept. Read the probabilities once.
    //         Three cases to settle before you write the comparison: token 0, a token whose
    //         mass before it is exactly p, and p >= 1 (README.md says why it is special).
    // TODO 3: after the loop, add up the threads' counts (the two stages of chapter 10) and
    //         let one thread write kept[blockIdx.x].
}

void launchTopPMask(const float* probs, float* massBefore, unsigned char* keep, int* kept, int rows, int cols,
                    float p) {
    // TODO 4: one block of BLOCK_DIM threads per row.
}
