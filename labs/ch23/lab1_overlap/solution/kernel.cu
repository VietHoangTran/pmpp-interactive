// Lab 23.1 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Hide the copies behind the kernel. Book: section 23.3 (pp. 553-563): streams
// and the fourth launch parameter (p. 556), cudaMemcpyAsync (p. 560), pinned
// memory (pp. 562-563).
//
// The book overlaps a halo exchange over the network with the internal rows of a
// stencil. This lab overlaps the copies over the host-device link with an
// elementwise kernel, by splitting the array into chunks and giving each chunk
// its own copy in, kernel and copy out in a stream. That staged pattern is
// NVIDIA's (Best Practices Guide 10.1.2, notes/today/ch23-facts.md A7), not the
// book's; the polynomial kernel and every measurement are ours.
//
// Where this file differs from the book:
//  - The streams are created with cudaStreamCreateWithFlags(..., cudaStreamNonBlocking),
//    not cudaStreamCreate (p. 556). The book's streams are blocking streams: any
//    work that lands in the legacy default stream waits for them and makes them
//    wait (ch23-facts A3, A9). Non-blocking streams take that coupling out.
//  - No stream priorities (p. 558): there is one kind of work here, and priorities
//    are a hint for kernels only, with no effect on copies (A2), ignored on AMD by
//    default (B2).
//  - The book says cudaMemcpyAsync "requires" pinned memory (p. 563). NVIDIA's
//    reference says an async copy from pageable memory works but may become
//    synchronous with the host (A6). The tests run both kinds; the benchmark
//    measures the difference.
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

struct Chunk {
    int begin;            // first element
    int count;            // elements, at least 1
    cudaStream_t stream;
};

static int copyIn(const StagedJob& job, const Chunk& c) {
    CHECK(cudaMemcpyAsync(job.d_in + c.begin, job.hostIn + c.begin, (size_t)c.count * sizeof(float),
                          cudaMemcpyHostToDevice, c.stream));
    return 0;
}
static int compute(const StagedJob& job, const Chunk& c) {
    // The fourth launch parameter (p. 556), inside launchPoly.
    if (launchPoly(job.d_in + c.begin, job.d_out + c.begin, c.count, job.degree, c.stream) != 0) return 1;
    CHECK(cudaGetLastError());
    return 0;
}
static int copyOut(const StagedJob& job, const Chunk& c) {
    CHECK(cudaMemcpyAsync(job.hostOut + c.begin, job.d_out + c.begin, (size_t)c.count * sizeof(float),
                          cudaMemcpyDeviceToHost, c.stream));
    return 0;
}

int runStaged(const StagedJob& job, int streams, int chunks, Order order) {
    // Streams that do not synchronise with the legacy default stream (A9).
    std::vector<cudaStream_t> stream((size_t)streams);
    for (int s = 0; s < streams; ++s)
        CHECK(cudaStreamCreateWithFlags(&stream[(size_t)s], cudaStreamNonBlocking));

    // The non-empty chunks, in order. k * n can pass 2^31 at the benchmark's size,
    // hence the 64-bit product. The j-th chunk queued goes to stream j mod streams.
    std::vector<Chunk> list;
    for (int k = 0; k < chunks; ++k) {
        const int begin = (int)((long long)k * job.n / chunks);
        const int end = (int)((long long)(k + 1) * job.n / chunks);
        if (end > begin) list.push_back(Chunk{begin, end - begin, stream[list.size() % (size_t)streams]});
    }

    // Every call below returns as soon as its operation is queued. Within one
    // stream the three operations of a chunk run in order; operations in
    // different streams may run at the same time (p. 556).
    if (order == Order::Depth) {
        for (const Chunk& c : list) {
            if (copyIn(job, c) || compute(job, c) || copyOut(job, c)) return 1;
        }
    } else {
        for (const Chunk& c : list)
            if (copyIn(job, c)) return 1;
        for (const Chunk& c : list)
            if (compute(job, c)) return 1;
        for (const Chunk& c : list)
            if (copyOut(job, c)) return 1;
    }

    // Nothing has necessarily happened yet. Wait for each stream's last copy out
    // (p. 559), then hand the streams back.
    for (int s = 0; s < streams; ++s) CHECK(cudaStreamSynchronize(stream[(size_t)s]));
    for (int s = 0; s < streams; ++s) CHECK(cudaStreamDestroy(stream[(size_t)s]));
    return 0;
}
