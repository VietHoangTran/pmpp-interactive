// Lab 11.3: Metal host side (provided; the defects are all in the kernel).
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>

static const NSUInteger kMaxBlock = 1024;   // MAX_BLOCK_DIM in kernel.metal

// out[r][j] = in[r][0] + ... + in[r][j-1] for every row r, one threadgroup of `block`
// threads per row. Returns false, without dispatching, when this pipeline cannot run the
// kernel as written.
bool encodeExclusiveRowScan(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> in, id<MTLBuffer> out, uint32_t rows, uint32_t cols,
                            uint32_t block) {
    // The kernel needs whole SIMD-groups (every lane takes part in the shuffles), and
    // SIMD-group 0 gives one lane to every SIMD-group's total, so no more SIMD-groups than a
    // SIMD-group has lanes. Apple documents no width; threadExecutionWidth is this
    // pipeline's SIMD-group width (notes/today/ch03-facts.md C9).
    const NSUInteger width = pipeline.threadExecutionWidth;
    if (block == 0 || block > kMaxBlock || block > pipeline.maxTotalThreadsPerThreadgroup || width == 0 ||
        block % width != 0 || block / width > width) {
        std::fprintf(stderr, "encodeExclusiveRowScan: %u threads per threadgroup do not fit this pipeline "
                     "(SIMD-group width %lu, at most %lu threads)\n", block, (unsigned long)width,
                     (unsigned long)pipeline.maxTotalThreadsPerThreadgroup);
        return false;
    }
    if (rows == 0 || cols == 0) return true;

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&cols length:sizeof(cols) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
    return true;
}
