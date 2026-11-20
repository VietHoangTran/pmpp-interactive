// Lab 23.1 starter (CUDA; also built for HIP and for the CPU emulator).
// Hide the copies behind the kernel. Book: section 23.3 (pp. 553-563): streams
// and the fourth launch parameter (p. 556), cudaMemcpyAsync (p. 560), pinned
// memory (pp. 562-563).
//
// The kernel, its launcher and the serial baseline are provided and correct.
// runSerial does the chapter 2 round trip: copy all n floats in, run the kernel,
// copy all n back, one blocking step after another. Your runStaged does the same
// work in chunks, so that one chunk's copies can run while another chunk's
// kernel does. That staged pattern is NVIDIA's (Best Practices Guide 10.1.2,
// notes/today/ch23-facts.md A7), not the book's; the polynomial kernel and every
// measurement are ours.
//
// The contract (overlap.h has the chunk rule):
//   - create `streams` streams with cudaStreamCreateWithFlags and the flag that
//     keeps them from synchronising with the legacy default stream;
//   - chunk k is [k*n/chunks, (k+1)*n/chunks); skip empty chunks; the j-th chunk
//     you queue goes to stream j mod streams;
//   - each chunk gets three operations in its stream: copy hostIn into d_in,
//     launchPoly from d_in into d_out, copy d_out into hostOut. Order::Depth
//     queues them chunk by chunk; Order::Breadth queues every copy in, then every
//     kernel, then every copy out;
//   - before returning, every result must be in hostOut. Destroy every stream you
//     created, once, and return 0. On any CUDA error, print it and return 1: the
//     CHECK macro below does both.
#include <cstdio>
#include <vector>

#include "../overlap.h"

// The error-checking sidebar of ch. 2 (p. 32) as a macro: report, and make the
// enclosing function return 1. (On an error, streams created before it leak.)
#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            printf("%s in %s at line %d\n", cudaGetErrorString(err),     \
                   __FILE__, __LINE__);                                  \
            return 1;                                                    \
        }                                                                \
    } while (0)

// ---- provided -------------------------------------------------------------------
// One thread per element: out[i] = p(in[i]), the polynomial of overlap.h by
// Horner's rule. Its cost grows with `degree` (one multiply-add per step), which
// is how the driver makes the kernel take about as long as the copy in.
__global__ void polyKernel(const float* in, float* out, int count, int degree) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float x = in[i];
        float y = (degree & 1) ? -0.25f : 0.5f;          // a_degree
        for (int k = degree - 1; k >= 0; --k)
            y = y * x + ((k & 1) ? -0.25f : 0.5f);        // y = y*x + a_k
        out[i] = y;
    }
}

// What the launcher was given, for the driver's structural tests.
static PolyRecord g_record;

const PolyRecord& polyRecord() { return g_record; }
void polyResetRecord() {
    g_record.calls.clear();
    g_record.calls.reserve(256);   // so that recording allocates nothing during a test
}

int launchPoly(const float* d_in, float* d_out, int count, int degree, cudaStream_t stream) {
    PolyCall c{d_in, d_out, 0, 0, count, degree, (const void*)stream, -1, -1.0, nullptr};
    if (count < 1) {
        // A grid of zero blocks is a launch error on CUDA. Refuse it here, the same
        // way on every backend, instead of relying on what each one does with it.
        c.refused = "count below 1";
        g_record.calls.push_back(c);
        printf("launchPoly: count %d refused; skip empty chunks\n", count);
        return 1;
    }
    g_record.calls.push_back(c);
    const int threadsPerBlock = 256;
    const int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
    polyKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_in, d_out, count, degree);
    return 0;
}

int runSerial(const StagedJob& job) {
    const size_t bytes = (size_t)job.n * sizeof(float);
    CHECK(cudaMemcpy(job.d_in, job.hostIn, bytes, cudaMemcpyHostToDevice));
    if (launchPoly(job.d_in, job.d_out, job.n, job.degree, 0) != 0) return 1;
    CHECK(cudaGetLastError());
    // In the default stream after the kernel, and blocking: it returns when the
    // last byte is in hostOut.
    CHECK(cudaMemcpy(job.hostOut, job.d_out, bytes, cudaMemcpyDeviceToHost));
    return 0;
}

// ---- your code below ------------------------------------------------------------

int runStaged(const StagedJob& job, int streams, int chunks, Order order) {
    // TODO 1: the streams, with cudaStreamNonBlocking. What would a plain
    //         cudaStreamCreate stream do whenever something is queued in the
    //         legacy default stream, as runSerial's calls are (ch23-facts A3, A9)?
    // TODO 2: the chunks. Chunk k starts at k*n/chunks: at the benchmark's size,
    //         k*n does not fit in an int. Empty chunks are skipped, and the j-th
    //         chunk you queue goes to stream j mod streams.
    // TODO 3: per chunk, a copy in, a launch, a copy out, all in the chunk's
    //         stream. cudaMemcpy would make the host wait; which call does not
    //         (p. 560)? Offsets apply to the device pointers and the host pointers
    //         alike. launchPoly refuses a count of 0 and returns 1.
    // TODO 4: Order::Depth and Order::Breadth issue the same operations in a
    //         different order. Only the order of your loops changes.
    // TODO 5: when your last call returns, has any copy finished? Wait for each
    //         stream (p. 559), then destroy each stream exactly once.
    (void)job; (void)streams; (void)chunks; (void)order;
    printf("runStaged: not written yet\n");
    return -1;
}
