// Chapter 12, summary page: a kernel of our own for the first question of the mixed check.
// It collects the candidates of one row of logits as (column, logit) pairs, unstable: every
// column whose logit is at or above the threshold, in any order. It compiles and it runs,
// and it is wrong on purpose: the question is about why. Not a lab file.
// Shown by app/src/chapters/ch12/Wrap.tsx. app/src/chapters/ch12/Wrap.snippets.test.ts
// runs it on the course's CPU emulator through check/wrap_pairs_test.cu.
// For the CUDA front-end check and the emulators only: on HIP, <cuda/atomic> needs
// libhipcxx (notes/today/infra-facts.md B2), and lab 12.1 shows the portable spelling.
#include <cuda/atomic>

__global__ void candidatePairs(const float* logits, int cols, float threshold,
                               int* outIdx, float* outVal,
                               unsigned int* idxCount, unsigned int* valCount) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols) {
        float v = logits[c];
        if (v >= threshold) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> idxRef(*idxCount);
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> valRef(*valCount);
            outIdx[idxRef.fetch_add(1, cuda::memory_order_relaxed)] = c;
            outVal[valRef.fetch_add(1, cuda::memory_order_relaxed)] = v;
        }
    }
}
