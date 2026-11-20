// Lab A.1 starter (CUDA; also built for HIP and for the CPU emulators).
// Summation error: order, accumulator width and compensation.
//
// Book: section A.6 (pp. 593-594): the order of a float sum changes its result, a reduction
// tree adds in another order than a loop, and Kahan's compensated summation (ref. [5]) is the
// sturdier option the book names without describing it. Appendix A shows no code. The tree is
// chapter 10's shared-memory reduction (Fig. 10.10, p. 237) behind a coarsening loop
// (Fig. 10.20, p. 247). The templated accumulator, the compensated loop and the 16-bit
// accumulators are ours.
//
// Contract (README.md states it in full): x holds n >= 1 float32 values; out[0] receives their
// sum, computed in the accumulator type AccT and converted to float. block is a power of two
// from 1 to MAX_BLOCK, coarse >= 1. partials holds at least ceil(n / block) floats.
//   (a) sumSerial  one thread adds x[0], x[1], ... in index order: the CPU loop's order.
//   (b) sumTree    thread t of block b adds x[b*B*coarse + t + j*B], j = 0 .. coarse-1, in that
//                  order (B = blockDim.x); the block combines its B sums with Fig. 10.10's tree;
//                  a second launch of the same kernel reduces the per-block sums.
//   (c) sumKahan   (b) with a compensated (Kahan) loop in each thread. The tree stays plain.
// The driver checks the order exactly: in FP16 and BF16 your result must have the same bits
// as a CPU model of these orders, so add in exactly the order above.
//
// Where this differs from the book's listings:
//   - Fig. 10.10's tree combines 2*blockDim.x inputs; here it combines blockDim.x per-thread
//     sums, and the coarsening loop takes the place of its add-on-load.
//   - Fig. 10.10 writes *output but has no output parameter: yours has one.
//   - Fig. 10.20 ends with an atomicAdd per block, which adds the block sums in whatever order
//     the blocks finish. Here they go to `partials` and a second launch adds them, so every run
//     adds in one order.
#include "../sumerror.h"

#define MAX_BLOCK 1024   // the largest block these kernels accept

// ---------------------------------------------------------------- provided: additions kept as written
// Use addRn and subRn for the compensated loop of (c). Its correction term is 0 in real
// arithmetic, so a compiler that may reassociate could delete it (notes/today/appA-facts.md
// A13). __fadd_rn and __fsub_rn are the float additions NVIDIA documents as never merged into
// a fused multiply-add (A9); whether nvcc ever reassociates is not documented, and the tests
// show it. HIP-Clang does not reassociate by default (B5). The 16-bit types use their
// documented operators (A16, B1).
__device__ __forceinline__ float addRn(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float subRn(float a, float b) { return __fsub_rn(a, b); }
#if PMPP_NARROW_ACC
__device__ __forceinline__ __half addRn(__half a, __half b) { return a + b; }
__device__ __forceinline__ __half subRn(__half a, __half b) { return a - b; }
__device__ __forceinline__ __nv_bfloat16 addRn(__nv_bfloat16 a, __nv_bfloat16 b) { return a + b; }
__device__ __forceinline__ __nv_bfloat16 subRn(__nv_bfloat16 a, __nv_bfloat16 b) { return a - b; }
#endif
// AccT(0.0f) and AccT(x[i]) convert a float to any of the three types, float(v) converts back,
// and + and - work on all three: write each kernel once, for any AccT.

// ---------------------------------------------------------------- (a)
template <typename AccT>
__global__ void sumSerial(const float* x, float* out, int n) {
    // TODO 1: add x[0], x[1], ..., x[n-1] in index order into an AccT, so that every
    //         addition is rounded to AccT, and store the sum in out[0] as a float.
}

// ---------------------------------------------------------------- (b)
template <typename AccT>
__global__ void sumTree(const float* x, float* out, int n, int coarse) {
    // TODO 2: one partial sum per block, in out[blockIdx.x].
    //   a. this thread adds its `coarse` elements x[b*B*coarse + t + j*B], j = 0, 1, ...,
    //      in that order, into an AccT; an element at or past n is skipped. Why do
    //      neighbouring threads read neighbouring elements at each step j, and not each a
    //      run of its own?
    //   b. the block's tree: a shared array of AccT sized for the largest block, each thread
    //      storing its sum; then stride = blockDim.x/2, ..., 2, 1, thread t < stride adding
    //      the value `stride` places to its right, one barrier per level. Where must the
    //      barrier go so that the first level sees every thread's store?
    //   c. thread 0 writes the block's sum. Every thread takes part in step b, even one whose
    //      elements were all out of range: why?
    //   (c) needs the same steps b and c, so a __device__ helper template saves writing
    //   them twice.
}

// ---------------------------------------------------------------- (c)
template <typename AccT>
__global__ void sumKahan(const float* x, float* out, int n, int coarse) {
    // TODO 3: as (b), but step a is a compensated sum. Keep, beside the sum, a correction c
    //         that holds minus what the last addition lost:
    //           y = element - c        (the element plus what was lost before)
    //           t = sum + y            (low bits of y fall off here)
    //           c = (t - sum) - y      (what was really added, minus what was meant)
    //           sum = t
    //         with addRn and subRn for every one of those additions. The tree stays plain.
}

// ---------------------------------------------------------------- launchers
// Each returns false, without launching, for a shape the kernels do not accept (n < 1,
// coarse < 1, block not a power of two from 1 to MAX_BLOCK) and for an accumulator this
// build has no kernel for. Return true once the launches are made.
//
// For the tree and Kahan sums, two launches of the same kernel:
//   pass 1: enough blocks of `block` threads that every element of x is covered, writing one
//           partial sum per block into `partials`;
//   pass 2: ONE block of `block` threads that reduces those partials into out[0]. How many
//           partials must each of its threads add first?
bool launchSumSerial(const float* x, float* out, int n, Acc acc) {
    switch (acc) {
    case AccFloat:
        // TODO 4: one thread runs sumSerial<float>.
        return false;
#if PMPP_NARROW_ACC
    // Real CUDA and HIP builds only: FP16 and BF16. The emulators and the front-end check
    // never compile this block (sumerror.h says why).
    case AccHalf:
        // TODO 5: sumSerial<__half>
        return false;
    case AccBfloat:
        // TODO 5: sumSerial<__nv_bfloat16> (on HIP this name is a typedef of __hip_bfloat16)
        return false;
#endif
    default:
        return false;
    }
}

bool launchSumTree(const float* x, float* partials, float* out, int n, int coarse, int block, Acc acc) {
    switch (acc) {
    case AccFloat:
        // TODO 6: the two passes of sumTree<float>.
        return false;
#if PMPP_NARROW_ACC
    case AccHalf:
        // TODO 7: the same for __half
        return false;
    case AccBfloat:
        // TODO 7: and for __nv_bfloat16
        return false;
#endif
    default:
        return false;
    }
}

bool launchSumKahan(const float* x, float* partials, float* out, int n, int coarse, int block, Acc acc) {
    switch (acc) {
    case AccFloat:
        // TODO 8: the two passes of sumKahan<float>.
        return false;
#if PMPP_NARROW_ACC
    case AccHalf:
        // TODO 9: the same for __half
        return false;
    case AccBfloat:
        // TODO 9: and for __nv_bfloat16
        return false;
#endif
    default:
        return false;
    }
}
