// Two Metal host stubs for vector addition, for the stepper in section 2.4.
//
//   vecAddInBuffers  the caller's data is created in shared-storage buffers:
//                    no copy in, no copy back, and no free call.
//   vecAdd           the book's interface, ordinary host arrays in and out:
//                    one copy per input and one copy back return.
//
// Like the book's Fig. 2.13, neither function checks for errors (a nil buffer,
// the command buffer's status); section 2.4 says what to check on Metal.
//
// Not part of the lab build. To compile and run it by hand (prints PASS):
//   clang++ -std=c++17 -fobjc-arc -framework Metal -framework Foundation \
//       -framework CoreGraphics vecadd_stub_metal.mm -o vecadd_stub_metal
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>

// Apple's own introductory kernel ("Performing calculations on a GPU").
static NSString* const kKernelSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void add_arrays(device const float* inA,\n"
     "                       device const float* inB,\n"
     "                       device float* result,\n"
     "                       uint index [[thread_position_in_grid]])\n"
     "{\n"
     "    result[index] = inA[index] + inB[index];\n"
     "}\n";

// Stand-ins for wherever the caller's data comes from and goes to: a file, a
// tokenizer, the previous kernel, the next one.
static const float kA[8] = {3, 1, 4, 1, 5, 9, 2, 6};
static const float kB[8] = {2, 7, 1, 8, 2, 8, 1, 8};
static int gWrong = 0;
static void produceA(float* dst, int n) { for (int i = 0; i < n; ++i) dst[i] = kA[i % 8]; }
static void produceB(float* dst, int n) { for (int i = 0; i < n; ++i) dst[i] = kB[i % 8]; }
static void consume(const float* c, int n) {
    for (int i = 0; i < n; ++i) if (c[i] != kA[i % 8] + kB[i % 8]) ++gWrong;
}

void vecAddInBuffers(id<MTLDevice> device, id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> pipeline, int n, NSUInteger group) {
    NSUInteger size = n * sizeof(float);

    // Part 1: three zero-filled buffers; the inputs are produced straight into two of them
    id<MTLBuffer> A = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> B = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> C = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    produceA((float*)A.contents, n);
    produceB((float*)B.contents, n);

    // Part 2: describe the work, then hand it to the GPU
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:C offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    [enc endEncoding];
    [cb commit];

    // Part 3: wait, then read the result where it is
    [cb waitUntilCompleted];
    consume((const float*)C.contents, n);
}

void vecAdd(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
            NSUInteger group, const float* A_h, const float* B_h, float* C_h, int n) {
    NSUInteger size = n * sizeof(float);

    // Part 1: the caller's arrays are not buffers, so each input is copied into one
    id<MTLBuffer> A = [device newBufferWithBytes:A_h length:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> B = [device newBufferWithBytes:B_h length:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> C = [device newBufferWithLength:size options:MTLResourceStorageModeShared];

    // Part 2
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:C offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    [enc endEncoding];
    [cb commit];

    // Part 3: wait, then copy the result out to the caller's array
    [cb waitUntilCompleted];
    memcpy(C_h, C.contents, size);
}

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) { std::printf("no Metal device\n"); return 2; }
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kKernelSource options:nil error:&error];
        if (library == nil) { std::printf("compile failed: %s\n", error.localizedDescription.UTF8String); return 2; }
        id<MTLFunction> function = [library newFunctionWithName:@"add_arrays"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) { std::printf("pipeline failed: %s\n", error.localizedDescription.UTF8String); return 2; }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        const int sizes[] = {8, 1000, 100000};
        for (int n : sizes) {
            NSUInteger group = pipeline.maxTotalThreadsPerThreadgroup;
            if (group > (NSUInteger)n) group = n;

            vecAddInBuffers(device, queue, pipeline, n, group);

            float* A_h = new float[n];
            float* B_h = new float[n];
            float* C_h = new float[n];
            produceA(A_h, n);
            produceB(B_h, n);
            for (int i = 0; i < n; ++i) C_h[i] = -1;
            vecAdd(device, queue, pipeline, group, A_h, B_h, C_h, n);
            consume(C_h, n);
            delete[] A_h; delete[] B_h; delete[] C_h;
        }
        std::printf(gWrong == 0 ? "PASS\n" : "FAIL: %d wrong elements\n", gWrong);
        return gWrong == 0 ? 0 : 1;
    }
}
