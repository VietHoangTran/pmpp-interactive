// The same job on Apple silicon (ours, after Apple's sample "Synchronizing CPU and GPU
// work"). A shared buffer is the GPU's memory, so nothing is copied or pinned: the
// copy-in becomes the CPU writing a chunk into one of K buffers, the copy-out the CPU
// reading a finished chunk back. A slot is touched only after the command buffer that
// last used it has completed. Same chunks as staged_loop.cu; chunk j goes to slot j % K.
#import <Metal/Metal.h>
#include <cstring>
#include <vector>

void stagedMetal(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso, const float* in,
                 float* out, int n, int nChunks, int K) {
    if (n <= 0) return;
    size_t most = ((size_t)n + nChunks - 1) / nChunks * sizeof(float);   // the longest chunk
    std::vector<id<MTLBuffer>> bufIn(K), bufOut(K);
    std::vector<id<MTLCommandBuffer>> last(K);    // the last command buffer to use slot k
    std::vector<int> off(K, 0), len(K, 0);        // the chunk slot k holds; len 0: none
    for (int k = 0; k < K; ++k) {
        bufIn[k] = [queue.device newBufferWithLength:most options:MTLResourceStorageModeShared];
        bufOut[k] = [queue.device newBufferWithLength:most options:MTLResourceStorageModeShared];
    }
    auto drain = [&](int k) {                     // the copy-out: read a finished chunk
        [last[k] waitUntilCompleted];
        std::memcpy(out + off[k], bufOut[k].contents, len[k] * sizeof(float));
        len[k] = 0;
    };
    int j = 0;
    for (int c = 0; c < nChunks; ++c) {
        int o = (int)((long long)c * n / nChunks), l = (int)((long long)(c + 1) * n / nChunks) - o;
        if (l == 0) continue;
        int k = j++ % K;
        if (len[k] > 0) drain(k);                 // slot k must be free before it is refilled
        std::memcpy(bufIn[k].contents, in + o, l * sizeof(float));   // the copy-in: fill
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufIn[k] offset:0 atIndex:0];
        [enc setBuffer:bufOut[k] offset:0 atIndex:1];
        [enc setBytes:&l length:sizeof(l) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(l, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];                              // the kernel runs while the CPU moves on
        last[k] = cb; off[k] = o; len[k] = l;
    }
    for (int i = 0; i < K; ++i)                   // what is still in flight, oldest first
        if (len[(j + i) % K] > 0) drain((j + i) % K);
}
