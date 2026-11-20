// The staged loop of section 23.3 (ours, after the CUDA C++ Best Practices Guide,
// section 10.1.2): cut a copy-in, kernel, copy-out job into chunks and spread the
// chunks over streams. Both functions only queue work; the caller synchronises before
// it reads out_h. in_h and out_h must be pinned (cudaMallocHost), or no copy overlaps.
#include <cuda_runtime.h>
#include <vector>

// A stand-in for any kernel that works element by element.
__global__ void chunkKernel(const float* in, float* out, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) out[i] = 2.0f * in[i] + 1.0f;
}

struct Job {
    const float* in_h; float* out_h;   // n floats each, pinned host memory
    float* in_d; float* out_d;         // n floats each, device memory
    int n, nChunks, nStreams;
    cudaStream_t* streams;             // nStreams streams the caller created
};
struct Chunk { int off, len; cudaStream_t s; };

// Chunk c is [c*n/nChunks, (c+1)*n/nChunks): lengths differ by at most one, and some
// are empty when n < nChunks. The j-th non-empty chunk goes to stream j % nStreams.
static std::vector<Chunk> chunks(const Job& job) {
    std::vector<Chunk> list;
    for (int c = 0; c < job.nChunks; ++c) {
        int off = (int)((long long)c * job.n / job.nChunks);
        int len = (int)((long long)(c + 1) * job.n / job.nChunks) - off;
        if (len > 0) list.push_back({off, len, job.streams[list.size() % job.nStreams]});
    }
    return list;
}

static void copyIn(const Job& job, const Chunk& k) {
    cudaMemcpyAsync(job.in_d + k.off, job.in_h + k.off, k.len * sizeof(float), cudaMemcpyHostToDevice, k.s);
}
static void compute(const Job& job, const Chunk& k) {
    chunkKernel<<<(k.len + 255) / 256, 256, 0, k.s>>>(job.in_d + k.off, job.out_d + k.off, k.len);
}
static void copyOut(const Job& job, const Chunk& k) {
    cudaMemcpyAsync(job.out_h + k.off, job.out_d + k.off, k.len * sizeof(float), cudaMemcpyDeviceToHost, k.s);
}

// Depth first: chunk 0's copy-in, kernel and copy-out, then chunk 1's, and so on.
void stagedDepthFirst(const Job& job) {
    for (const Chunk& k : chunks(job)) {
        copyIn(job, k);
        compute(job, k);
        copyOut(job, k);
    }
}

// Breadth first: every copy-in, then every kernel, then every copy-out.
void stagedBreadthFirst(const Job& job) {
    std::vector<Chunk> list = chunks(job);
    for (const Chunk& k : list) copyIn(job, k);
    for (const Chunk& k : list) compute(job, k);
    for (const Chunk& k : list) copyOut(job, k);
}
