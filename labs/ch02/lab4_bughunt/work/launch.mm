// Lab 2.4 starter: the Metal host side. Not in the book.
// kernel.metal and this file together have THREE independent defects.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// Returns 0 when everything worked and a non-zero value otherwise.
int scalePrefix(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                float* x_h, uint32_t n, uint32_t len, float s, uint32_t threadsPerGroup) {
    const NSUInteger size = (NSUInteger)n * sizeof(float);    // bytes of the whole array, padding included

    id<MTLBuffer> X = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (X == nil) {
        printf("buffer creation failed in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }
    memcpy(X.contents, x_h, n);

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (commandBuffer == nil) {
        printf("no command buffer in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        printf("no compute encoder in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:X offset:0 atIndex:0];
    [encoder setBytes:&s length:sizeof(s) atIndex:1];
    [encoder setBytes:&len length:sizeof(len) atIndex:2];

    NSUInteger groups = len / threadsPerGroup;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        NSError* error = commandBuffer.error;
        printf("command buffer failed (%s) in %s at line %d\n",
               error ? error.localizedDescription.UTF8String : "no error object", __FILE__, __LINE__);
        return 1;
    }

    memcpy(x_h, X.contents, size);
    return 0;
}
