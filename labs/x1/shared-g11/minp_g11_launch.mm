// PROVIDED, do not edit. The Metal host side of minp_g11.metal: binds the arguments, computes
// ln(minP) once on the host in float (as llama.cpp does), sizes the per-SIMD-group counts and
// dispatches one threadgroup of 256 threads per row. Returns false, without dispatching, if the
// pipeline cannot run 256 threads in whole SIMD-groups.
#import <Metal/Metal.h>
#include <cmath>
#include <cstdint>

bool encodeMinPCombine(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                       id<MTLBuffer> sortedLogits, id<MTLBuffer> kept, id<MTLBuffer> n,
                       uint32_t rows, uint32_t cols, float minP) {
    static const NSUInteger block = 256;
    const NSUInteger width = p.threadExecutionWidth;
    if (p == nil || block > p.maxTotalThreadsPerThreadgroup || width == 0 || block % width != 0) return false;
    if (rows == 0 || cols == 0) return true;
    const uint32_t useMinP = minP > 0.0f ? 1u : 0u;
    const float lnMinP = useMinP ? std::log(minP) : 0.0f;   // std::log(float) is logf
    const NSUInteger groupBytes = ((block / width) * sizeof(int32_t) + 15) / 16 * 16;   // ch05-facts C8
    [enc setComputePipelineState:p];
    [enc setBuffer:sortedLogits offset:0 atIndex:0];
    [enc setBuffer:kept offset:0 atIndex:1];
    [enc setBuffer:n offset:0 atIndex:2];
    [enc setBytes:&cols length:sizeof(cols) atIndex:3];
    [enc setBytes:&lnMinP length:sizeof(lnMinP) atIndex:4];
    [enc setBytes:&useMinP length:sizeof(useMinP) atIndex:5];
    [enc setThreadgroupMemoryLength:groupBytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
    return true;
}
