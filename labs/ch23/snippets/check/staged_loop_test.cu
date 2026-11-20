// Runs ../staged_loop.cu, the staged loop section 23.3 shows, against a CPU reference:
// both issue orders, chunks that are equal, uneven (lengths differing by one) and empty
// (more chunks than elements), more streams than chunks, one stream, and n = 0. Host
// buffers come from cudaMallocHost, as the snippet requires; guard elements around
// out_h must stay untouched. On the course's emulator the streams run synchronously,
// so this checks what each chunk computes and where it lands, not any overlap.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

struct Job {
    const float* in_h; float* out_h;
    float* in_d; float* out_d;
    int n, nChunks, nStreams;
    cudaStream_t* streams;
};
void stagedDepthFirst(const Job& job);
void stagedBreadthFirst(const Job& job);

static int g_failures = 0;
static void expect(bool ok, const char* what) {
    std::printf("  %s: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_failures;
}

static void runCase(int n, int nChunks, int nStreams, bool depth) {
    const int guard = 8;
    float *in_h = nullptr, *out_h = nullptr, *in_d = nullptr, *out_d = nullptr;
    cudaMallocHost((void**)&in_h, (n + 1) * sizeof(float));
    cudaMallocHost((void**)&out_h, (n + 2 * guard) * sizeof(float));
    cudaMalloc((void**)&in_d, (n + 1) * sizeof(float));
    cudaMalloc((void**)&out_d, (n + 1) * sizeof(float));
    for (int i = 0; i < n; ++i) in_h[i] = (float)((i * 37) % 101) - 50.0f;
    for (int i = 0; i < n + 2 * guard; ++i) out_h[i] = -7.0f;
    std::vector<cudaStream_t> streams(nStreams);
    for (int s = 0; s < nStreams; ++s) cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking);
    Job job{in_h, out_h + guard, in_d, out_d, n, nChunks, nStreams, streams.data()};
    if (depth) stagedDepthFirst(job); else stagedBreadthFirst(job);
    cudaDeviceSynchronize();
    bool right = true, guards = true;
    for (int i = 0; i < n; ++i) right = right && out_h[guard + i] == 2.0f * in_h[i] + 1.0f;
    for (int i = 0; i < guard; ++i) guards = guards && out_h[i] == -7.0f && out_h[guard + n + i] == -7.0f;
    char what[200];
    std::snprintf(what, sizeof what, "%s, n %d, %d chunks, %d streams: every element right, nothing written outside",
                  depth ? "depth first" : "breadth first", n, nChunks, nStreams);
    expect(right && guards, what);
    for (int s = 0; s < nStreams; ++s) cudaStreamDestroy(streams[s]);
    cudaFreeHost(in_h);
    cudaFreeHost(out_h);
    cudaFree(in_d);
    cudaFree(out_d);
}

int main() {
    std::printf("staged_loop.cu\n");
    const int cases[][3] = {
        {4096, 8, 8},    // equal chunks, a stream each
        {1000, 8, 3},    // equal chunks, streams shared
        {10, 4, 2},      // uneven: lengths 2, 3, 2, 3
        {1023, 7, 16},   // uneven, more streams than chunks
        {5, 8, 3},       // more chunks than elements: three are empty
        {1, 64, 4},      // one element, 63 empty chunks
        {777, 1, 1},     // one chunk, one stream
        {300, 12, 1},    // one stream: correct, but nothing could overlap
        {0, 4, 2},       // nothing to do
    };
    for (const auto& c : cases)
        for (int d = 0; d < 2; ++d) runCase(c[0], c[1], c[2], d == 0);
    std::printf("%s\n", g_failures ? "FAILED" : "PASSED");
    return g_failures ? 1 : 0;
}
