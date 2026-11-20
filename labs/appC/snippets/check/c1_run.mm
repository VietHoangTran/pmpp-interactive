// Runs section C.1's Metal code on this Mac's GPU and compares it with a CPU reference.
// Built and driven by labs/appC/snippets/check.py; not a lab, and not part of the lab runtime.
//
//   c1_run overloads <lib.metallib>   kernel row_max_twice from c1_helper_overloads.metal:
//                                     every row's maximum, read from device memory and from
//                                     the threadgroup copy
//   c1_run generic <lib.metallib>     kernel generic_probe from check/c1_generic_probe.metal:
//                                     one helper with an unqualified pointer, called on
//                                     device, threadgroup and thread memory
//
// Prints one line per case and exits 0 only if every case matches.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static const uint32_t TILE = 256;   // the snippets' TILE

static id<MTLComputePipelineState> pipelineFor(id<MTLDevice> dev, const char* lib, const char* fn) {
    NSError* err = nil;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:lib]];
    id<MTLLibrary> library = [dev newLibraryWithURL:url error:&err];
    if (!library) {
        fprintf(stderr, "cannot load %s: %s\n", lib, err.localizedDescription.UTF8String);
        return nil;
    }
    id<MTLFunction> f = [library newFunctionWithName:[NSString stringWithUTF8String:fn]];
    if (!f) {
        fprintf(stderr, "no function %s in %s\n", fn, lib);
        return nil;
    }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) fprintf(stderr, "pipeline for %s failed: %s\n", fn, err.localizedDescription.UTF8String);
    return p;
}

static id<MTLBuffer> shared(id<MTLDevice> dev, const void* src, size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (src) memcpy(b.contents, src, bytes);
    else memset(b.contents, 0, bytes);
    return b;
}

static std::vector<float> randomRows(uint32_t rows, uint32_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-100.0f, 100.0f);
    std::vector<float> x((size_t)rows * n);
    for (float& v : x) v = d(rng);
    return x;
}

static float cpuMax(const float* p, uint32_t n) {
    float m = p[0];
    for (uint32_t i = 1; i < n; ++i) m = p[i] > m ? p[i] : m;
    return m;
}

// row_max_twice(x, fromDevice, fromThreadgroup, n): one threadgroup of TILE threads per row.
static bool runOverloads(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLComputePipelineState> p) {
    const uint32_t shapes[][2] = {{1, 1}, {3, 7}, {64, 256}, {17, 255}, {5, 100}};
    bool ok = true;
    for (auto& s : shapes) {
        uint32_t rows = s[0], n = s[1];
        std::vector<float> x = randomRows(rows, n, 1000 * rows + n);
        id<MTLBuffer> bx = shared(dev, x.data(), x.size() * sizeof(float));
        id<MTLBuffer> bd = shared(dev, nullptr, rows * sizeof(float));
        id<MTLBuffer> bt = shared(dev, nullptr, rows * sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:bx offset:0 atIndex:0];
        [e setBuffer:bd offset:0 atIndex:1];
        [e setBuffer:bt offset:0 atIndex:2];
        [e setBytes:&n length:sizeof(n) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(TILE, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            printf("FAIL overloads rows=%u n=%u: %s\n", rows, n, cb.error.localizedDescription.UTF8String);
            ok = false;
            continue;
        }
        const float* fd = (const float*)bd.contents;
        const float* ft = (const float*)bt.contents;
        uint32_t bad = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            float want = cpuMax(&x[(size_t)r * n], n);
            if (fd[r] != want || ft[r] != want) ++bad;   // a maximum is one of the inputs: exact
        }
        printf("%s overloads rows=%u n=%u: %u of %u rows match from device and from threadgroup memory\n",
               bad ? "FAIL" : "ok  ", rows, n, rows - bad, rows);
        ok = ok && !bad;
    }
    return ok;
}

// generic_probe(x, out, n): out[0] from device memory (x), out[1] from a threadgroup copy
// holding x + 1, out[2] from a thread array holding x + 2. Each memory holds different values,
// so a load that reached the wrong memory would show.
static bool runGeneric(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLComputePipelineState> p) {
    const uint32_t sizes[] = {1, 9, 128, 256};
    bool ok = true;
    for (uint32_t n : sizes) {
        std::vector<float> x = randomRows(1, n, 77 + n);
        id<MTLBuffer> bx = shared(dev, x.data(), x.size() * sizeof(float));
        id<MTLBuffer> bo = shared(dev, nullptr, 3 * sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:bx offset:0 atIndex:0];
        [e setBuffer:bo offset:0 atIndex:1];
        [e setBytes:&n length:sizeof(n) atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(TILE, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            printf("FAIL generic n=%u: %s\n", n, cb.error.localizedDescription.UTF8String);
            ok = false;
            continue;
        }
        const float* o = (const float*)bo.contents;
        const float m = cpuMax(x.data(), n);
        const float want[3] = {m, m + 1.0f, m + 2.0f};
        const char* where[3] = {"device", "threadgroup", "thread"};
        for (int k = 0; k < 3; ++k) {
            bool good = o[k] == want[k];
            printf("%s generic n=%u, %s memory: got %g, CPU reference %g\n", good ? "ok  " : "FAIL", n, where[k], o[k], want[k]);
            ok = ok && good;
        }
    }
    return ok;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc != 3 || (strcmp(argv[1], "overloads") && strcmp(argv[1], "generic"))) {
            fprintf(stderr, "usage: %s overloads|generic <lib.metallib>\n", argv[0]);
            return 2;
        }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "no Metal device\n");
            return 3;
        }
        printf("device: %s\n", dev.name.UTF8String);
        id<MTLCommandQueue> q = [dev newCommandQueue];
        bool overloads = !strcmp(argv[1], "overloads");
        id<MTLComputePipelineState> p = pipelineFor(dev, argv[2], overloads ? "row_max_twice" : "generic_probe");
        if (!p) return 1;
        bool ok = overloads ? runOverloads(dev, q, p) : runGeneric(dev, q, p);
        return ok ? 0 : 1;
    }
}
