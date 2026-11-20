// Lab A.1 starter: the Metal counterparts of launchSumSerial, launchSumTree and launchSumKahan.
// The driver picks the pipeline (sum_tree_half, ...), so these functions do not know the
// accumulator type. Each returns false, without dispatching, for a shape the kernels cannot
// run, and true once its dispatches are encoded.
//
// The two passes of the tree go into the one compute encoder you are given. An encoder from
// plain computeCommandEncoder runs its dispatches in order, and each sees the previous one's
// writes (notes/today/ch02-facts.md C9): the counterpart of two launches on one CUDA stream.
#import <Metal/Metal.h>
#include <cstdint>

static const uint32_t kMaxBlock = 1024;   // MAX_BLOCK in kernel.metal

bool encodeSumSerial(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                     id<MTLBuffer> x, id<MTLBuffer> out, uint32_t n) {
    // TODO 1: bind the pipeline, the buffers and n at the indices kernel.metal declares
    //         (setBytes for a constant), and dispatch the counterpart of <<<1, 1>>>.
    return false;
}

bool encodeSumTree(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> x,
                   id<MTLBuffer> partials, id<MTLBuffer> out, uint32_t n, uint32_t coarse, uint32_t block) {
    // TODO 2: return false for n == 0, coarse == 0, or a block the kernel or this pipeline
    //         cannot run: not a power of two, above kMaxBlock, or above
    //         p.maxTotalThreadsPerThreadgroup (the pipeline's own limit, which can be lower
    //         than the device's).
    // TODO 3: pass 1, enough threadgroups of `block` threads to cover x, writing `partials`;
    //         then pass 2, one threadgroup reducing the partials into `out`. Each pass sets
    //         its own buffers and constants before its dispatch:
    //   [enc dispatchThreadgroups:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
    return false;
}

bool encodeSumKahan(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> x,
                    id<MTLBuffer> partials, id<MTLBuffer> out, uint32_t n, uint32_t coarse, uint32_t block) {
    // TODO 4: the same two passes with the Kahan pipeline the driver passes in.
    return false;
}
