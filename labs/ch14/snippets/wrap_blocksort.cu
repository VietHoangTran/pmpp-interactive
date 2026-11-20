// Chapter 14, "Summary and exercises", the question ch14/wrap/q-blocksort: one block sorts
// up to MAXN keys in shared memory with a fixed number of odd-even transposition steps, the
// shape of a sorting network inside a block (section 14.9), with no flag and no host loop.
// Not in the book. It is shown with one defect on purpose: there is no barrier between
// steps. The corrected copy, one added line, is in check/wrap_check.cu. Launch: one block of
// (n + 1) / 2 threads, n at most MAXN. Written to serve HIP too, through pmpp/vendors/hip.h;
// not built for HIP. Run on the course's CPU emulator by
// app/src/chapters/ch14/Wrap.snippets.test.ts (driver: check/wrap_check.cu).

#define MAXN 1024

__global__ void blockOddEvenSort(unsigned int* keys, unsigned int n) {
    __shared__ unsigned int s[MAXN];
    unsigned int t = threadIdx.x;
    for (unsigned int i = t; i < n; i += blockDim.x)
        s[i] = keys[i];
    __syncthreads();
    for (unsigned int step = 0; step < n; ++step) {
        unsigned int i = 2 * t + (step & 1);
        if (i + 1 < n && s[i] > s[i + 1]) {
            unsigned int tmp = s[i];
            s[i] = s[i + 1];
            s[i + 1] = tmp;
        }
    }
    __syncthreads();
    for (unsigned int i = t; i < n; i += blockDim.x)
        keys[i] = s[i];
}
