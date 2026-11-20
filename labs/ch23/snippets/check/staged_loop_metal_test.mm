// Runs ../staged_loop_metal.mm, the Metal version section 23.3 shows, on this Mac's GPU
// against a CPU reference: K = 1, 2, 3 and more buffers than chunks, with equal, uneven
// and empty chunks, and n = 0. The kernel is ../staged_loop.metal, compiled at run time.
// Guard elements around the output must stay untouched. Usage: staged_loop_metal_test <kernel.metal>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <vector>

void stagedMetal(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso, const float* in,
                 float* out, int n, int nChunks, int K);

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 2) { std::printf("usage: staged_loop_metal_test <kernel.metal>\n"); return 2; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("no Metal device\n"); return 2; }
        NSError* err = nil;
        NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                                  encoding:NSUTF8StringEncoding error:&err];
        if (!src) { std::printf("cannot read %s\n", argv[1]); return 1; }
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { std::printf("compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"chunkKernel"] error:&err];
        if (!pso) { std::printf("pipeline failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        std::printf("staged_loop_metal.mm on %s\n", dev.name.UTF8String);
        const int cases[][3] = {
            {4096, 8, 1}, {4096, 8, 2}, {4096, 8, 3},   // equal chunks
            {1000, 7, 2},                               // uneven
            {10, 4, 3},                                 // lengths 2, 3, 2, 3
            {5, 8, 2},                                  // three empty chunks
            {1, 64, 4},                                 // 63 empty chunks
            {300, 3, 8},                                // more buffers than chunks
            {0, 4, 2},                                  // nothing to do
        };
        int failures = 0;
        for (const auto& c : cases) {
            const int n = c[0], chunks = c[1], K = c[2], guard = 8;
            std::vector<float> in(n + 1), out(n + 2 * guard, -7.0f);
            for (int i = 0; i < n; ++i) in[i] = (float)((i * 37) % 101) - 50.0f;
            stagedMetal(queue, pso, in.data(), out.data() + guard, n, chunks, K);
            bool right = true, guards = true;
            // Metal does not promise strict IEEE arithmetic, so compare with a tolerance
            // (the inputs are small integers, so the results are exact in practice).
            for (int i = 0; i < n; ++i) {
                const float ref = 2.0f * in[i] + 1.0f;
                right = right && std::fabs(out[guard + i] - ref) <= 1e-6f * std::fmax(1.0f, std::fabs(ref));
            }
            for (int i = 0; i < guard; ++i) guards = guards && out[i] == -7.0f && out[guard + n + i] == -7.0f;
            std::printf("  %s: n %d, %d chunks, K = %d: every element right, nothing written outside\n",
                        right && guards ? "ok" : "FAIL", n, chunks, K);
            if (!(right && guards)) ++failures;
        }
        std::printf("%s\n", failures ? "FAILED" : "PASSED");
        return failures ? 1 : 0;
    }
}
