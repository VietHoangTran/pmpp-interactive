// Compiles ../philox.metal at run time and runs it on this Mac's GPU. Prints the same lines
// as philox_kat.cu, for app/src/sim/philox.test.ts to compare with app/src/sim/philox.ts:
//   kat <name> <four words>     Random123's three 10-round vectors, one per thread, through a
//                               test kernel appended here that calls the snippet's philox4x32_10
//   u <seed> <stream> <step> <float bits>
//                               the snippet's uniformKernel on the 64 rows philox_kat.cu uses
// Usage: philox_metal_run <path to philox.metal>
// Build: clang++ -std=c++17 -fobjc-arc philox_metal_run.mm -framework Metal -framework Foundation
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>

static const char* kTestKernel = R"(
kernel void katKernel(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= 3) return;
    const uint4 r = philox4x32_10(uint4(in[6 * i], in[6 * i + 1], in[6 * i + 2], in[6 * i + 3]),
                                  uint2(in[6 * i + 4], in[6 * i + 5]));
    out[4 * i] = r.x; out[4 * i + 1] = r.y; out[4 * i + 2] = r.z; out[4 * i + 3] = r.w;
}
)";

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 2) { std::fprintf(stderr, "usage: %s philox.metal\n", argv[0]); return 2; }
        NSError* err = nil;
        NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                                  encoding:NSUTF8StringEncoding error:&err];
        if (!src) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
        src = [src stringByAppendingString:[NSString stringWithUTF8String:kTestKernel]];
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::fprintf(stderr, "no Metal device\n"); return 3; }
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
        if (!lib) { std::fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 4; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        auto pipe = [&](const char* name) {
            NSError* e = nil;
            id<MTLComputePipelineState> p =
                [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:name]] error:&e];
            if (!p) { std::fprintf(stderr, "pipeline %s failed\n", name); std::exit(5); }
            return p;
        };
        auto run = [&](id<MTLComputePipelineState> p, NSArray<id<MTLBuffer>>* bufs, NSUInteger threads) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:p];
            for (NSUInteger i = 0; i < bufs.count; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
            [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        };

        const char* names[3] = {"zero", "ones", "pi"};
        const uint32_t kat[18] = {0u, 0u, 0u, 0u, 0u, 0u,
                                  0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                                  0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u, 0xa4093822u, 0x299f31d0u};
        id<MTLBuffer> in = [dev newBufferWithBytes:kat length:sizeof kat options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [dev newBufferWithLength:12 * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        run(pipe("katKernel"), @[ in, out ], 3);
        const uint32_t* o = (const uint32_t*)out.contents;
        for (int i = 0; i < 3; ++i)
            std::printf("kat %s %08x %08x %08x %08x\n", names[i], o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);

        // The same 64 rows as philox_kat.cu.
        const uint32_t rows = 64;
        std::vector<uint64_t> seed(rows), stream(rows), step(rows);
        uint64_t s = 2026;
        for (uint32_t r = 0; r < rows; ++r) {
            uint64_t v[3];
            for (int k = 0; k < 3; ++k) {
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                v[k] = s ^ (s >> 29);
            }
            seed[r] = v[0];
            stream[r] = r < 32 ? (uint64_t)r : v[1];
            step[r] = r < 32 ? 7ull : v[2] >> (r % 40);
        }
        seed[0] = stream[0] = step[0] = 0;
        seed[1] = stream[1] = step[1] = ~0ull;
        id<MTLBuffer> b0 = [dev newBufferWithBytes:seed.data() length:rows * 8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b1 = [dev newBufferWithBytes:stream.data() length:rows * 8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b2 = [dev newBufferWithBytes:step.data() length:rows * 8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b3 = [dev newBufferWithLength:rows * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b4 = [dev newBufferWithBytes:&rows length:sizeof rows options:MTLResourceStorageModeShared];
        run(pipe("uniformKernel"), @[ b0, b1, b2, b3, b4 ], rows);
        const float* u = (const float*)b3.contents;
        for (uint32_t r = 0; r < rows; ++r) {
            uint32_t bits;
            std::memcpy(&bits, &u[r], 4);
            std::printf("u %llu %llu %llu %08x\n", (unsigned long long)seed[r], (unsigned long long)stream[r],
                        (unsigned long long)step[r], bits);
        }
        std::printf("device %s\nDONE\n", dev.name.UTF8String);
    }
    return 0;
}
