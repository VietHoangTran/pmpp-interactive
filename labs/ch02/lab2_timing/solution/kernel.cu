// Lab 2.2 reference solution (CUDA; also built for HIP and for the CPU emulator).
// The vecAdd of lab 2.1 with a clock read between its parts. It measures, on
// your machine, the book's remark on p. 28 that outsourcing a computation this
// way can be very inefficient because of the copying back and forth.
//
// Not in the book: the book gives no timings, and this chapter does not say
// whether a kernel call waits for the grid. Where the clock reads and the
// synchronise calls go follows the vendors' current documentation, collected
// in notes/today/ch02-facts.md (items A3, A5 and B1).
#include <chrono>
#include <cstdio>

// The sidebar on p. 32, corrected (the page tests `error`, a name it never
// declared), as a macro. It reports and makes the enclosing function return 1.
#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            printf("%s in %s at line %d\n", cudaGetErrorString(err),     \
                   __FILE__, __LINE__);                                  \
            return 1;                                                    \
        }                                                                \
    } while (0)

// The book's block size (p. 38). Both functions below use it.
static const int threadsPerBlock = 256;

// Provided: a clock that never runs backwards, and the seconds between two reads.
typedef std::chrono::steady_clock::time_point Instant;
static Instant now() { return std::chrono::steady_clock::now(); }
static double seconds(Instant from, Instant to) {
    return std::chrono::duration<double>(to - from).count();
}

// Compute vector sum C = A + B.
// Each thread performs one pair-wise addition.
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Provided. Launches the kernel on arrays that are already on the device, and
// returns as soon as the launch is queued. The driver calls it between its own
// clock reads, with its own synchronise: that is the "resident" measurement.
void launchVecAdd(const float* A_d, const float* B_d, float* C_d, int n) {
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vecAddKernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, n);
}

// phase_s[0..4]: seconds spent in allocation, copies in, kernel, copy back, free.
int vecAddTimed(const float* A_h, const float* B_h, float* C_h, int n,
                int syncAfterLaunch, double* phase_s) {
    float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    size_t size = (size_t)n * sizeof(float);

    // Whatever the caller left running is not ours to time.
    CHECK(cudaDeviceSynchronize());
    Instant t0 = now();

    // Phase 0: allocation.
    CHECK(cudaMalloc((void**)&A_d, size));
    CHECK(cudaMalloc((void**)&B_d, size));
    CHECK(cudaMalloc((void**)&C_d, size));
    Instant t1 = now();

    // Phase 1: copies in. cudaMemcpy is documented as synchronous "for most
    // use cases", and NVIDIA's older guide said a small host-to-device copy
    // may return before it has happened. The synchronise charges the copies
    // to this phase whichever is true, and returns at once when they are done.
    CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    Instant t2 = now();

    // Phase 2: the kernel. The launch statement returns once the launch is
    // queued. Without the synchronise, t3 is read while the grid is still
    // running, and this phase holds the cost of the launch alone.
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vecAddKernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, n);
    CHECK(cudaGetLastError());
    if (syncAfterLaunch) {
        CHECK(cudaDeviceSynchronize());
    }
    Instant t3 = now();

    // Phase 3: copy back. The launch and this copy are queued on the same
    // default stream, which runs its work in order, and the copy does not
    // return until it is done. So the result is right either way. When phase 2
    // did not wait, the wait is here, and this phase reads as the kernel plus
    // the copy.
    CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));
    Instant t4 = now();

    // Phase 4: free. Nothing is running by now, so this is the free alone.
    // (cudaFree may wait for outstanding work; hipFree always does.)
    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));
    Instant t5 = now();

    phase_s[0] = seconds(t0, t1);
    phase_s[1] = seconds(t1, t2);
    phase_s[2] = seconds(t2, t3);
    phase_s[3] = seconds(t3, t4);
    phase_s[4] = seconds(t4, t5);
    return 0;
}
