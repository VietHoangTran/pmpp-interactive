// Lab 10.3 starter: the Metal counterparts of launchRowMax, launchSoftmaxRows and
// launchArgmaxRows. Each binds its arguments, sizes the threadgroup memory the kernel
// declares with [[threadgroup(n)]], and dispatches one threadgroup of kBlock threads per row.
// Return false, without dispatching, when the pipeline cannot run the kernel as written.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;   // threads per threadgroup, as BLOCK_SIZE in kernel.cu

// TODO 1: a check, used by all three functions, that this pipeline can run kBlock threads
//         per threadgroup in whole SIMD-groups, with no more SIMD-groups than a SIMD-group
//         has lanes. The SIMD-group width is p.threadExecutionWidth: it belongs to the
//         compiled pipeline, so read it, never assume it. The pipeline's own thread limit
//         is p.maxTotalThreadsPerThreadgroup.

bool encodeRowMax(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                  id<MTLBuffer> x, id<MTLBuffer> out, uint32_t rows, uint32_t cols) {
    [enc setComputePipelineState:p];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&cols length:sizeof(cols) atIndex:2];
    // TODO 2: size scratch (one float per SIMD-group) with
    //         [enc setThreadgroupMemoryLength:bytes atIndex:0]; the length must be a
    //         multiple of 16 bytes. Then dispatch one threadgroup per row.
    return true;
}

bool encodeSoftmaxRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                       id<MTLBuffer> x, id<MTLBuffer> y, uint32_t rows, uint32_t cols, float scale) {
    [enc setComputePipelineState:p];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:y offset:0 atIndex:1];
    [enc setBytes:&cols length:sizeof(cols) atIndex:2];
    [enc setBytes:&scale length:sizeof(scale) atIndex:3];
    // TODO 3: as TODO 2.
    return true;
}

bool encodeArgmaxRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                      id<MTLBuffer> x, id<MTLBuffer> idx, uint32_t rows, uint32_t cols) {
    [enc setComputePipelineState:p];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:idx offset:0 atIndex:1];
    [enc setBytes:&cols length:sizeof(cols) atIndex:2];
    // TODO 4: two threadgroup buffers this time (vals at index 0, idxs at index 1), then
    //         the dispatch.
    return true;
}
