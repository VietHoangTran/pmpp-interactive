// Lab A.1 reference solution: the Metal counterparts of launchSumSerial, launchSumTree and
// launchSumKahan. The driver picks the pipeline (sum_tree_half, ...), so these functions do
// not know the accumulator type. Each returns false, without dispatching, for a shape the
// kernels cannot run.
//
// The two passes of the tree go into the one compute encoder they are given. An encoder from
// plain computeCommandEncoder runs its dispatches in order, and each sees the previous one's
// writes (notes/today/ch02-facts.md C9): the counterpart of two launches on one CUDA stream.
#import <Metal/Metal.h>
#include <cstdint>

static const uint32_t kMaxBlock = 1024;   // MAX_BLOCK in kernel.metal

// The tree needs a power-of-two threadgroup that fits both the kernel's threadgroup array and
// this pipeline's own limit, which may be below the device's (ch05-facts C3).
static bool blockFits(id<MTLComputePipelineState> p, uint32_t block) {
    return block >= 1 && block <= kMaxBlock && (block & (block - 1)) == 0 &&
           block <= p.maxTotalThreadsPerThreadgroup;
}

bool encodeSumSerial(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                     id<MTLBuffer> x, id<MTLBuffer> out, uint32_t n) {
    if (n == 0) return false;
    [enc setComputePipelineState:p];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    // One threadgroup of one thread: the counterpart of <<<1, 1>>>.
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    return true;
}

// One pass: `groups` threadgroups of `block` threads, each thread adding `coarse` elements of
// `in` before the threadgroup's tree; threadgroup g writes out[g].
static void encodePass(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> in,
                       id<MTLBuffer> out, uint32_t n, uint32_t coarse, uint32_t block, uint32_t groups) {
    [enc setComputePipelineState:p];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc setBytes:&coarse length:sizeof(coarse) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
}

// Pass 1 leaves one partial sum per threadgroup; pass 2 is one threadgroup that reduces them,
// each thread first adding ceil(groups / block) of them. sum_tree and sum_kahan share it.
static bool encodeTwoPasses(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> x,
                            id<MTLBuffer> partials, id<MTLBuffer> out, uint32_t n, uint32_t coarse, uint32_t block) {
    if (n == 0 || coarse == 0 || !blockFits(p, block)) return false;
    const uint32_t perGroup = block * coarse;
    const uint32_t groups = (n + perGroup - 1) / perGroup;
    encodePass(enc, p, x, partials, n, coarse, block, groups);
    const uint32_t coarse2 = (groups + block - 1) / block;
    encodePass(enc, p, partials, out, groups, coarse2, block, 1);
    return true;
}

bool encodeSumTree(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> x,
                   id<MTLBuffer> partials, id<MTLBuffer> out, uint32_t n, uint32_t coarse, uint32_t block) {
    return encodeTwoPasses(enc, p, x, partials, out, n, coarse, block);
}

bool encodeSumKahan(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> x,
                    id<MTLBuffer> partials, id<MTLBuffer> out, uint32_t n, uint32_t coarse, uint32_t block) {
    return encodeTwoPasses(enc, p, x, partials, out, n, coarse, block);
}
