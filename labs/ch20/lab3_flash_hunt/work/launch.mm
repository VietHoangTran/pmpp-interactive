// Lab 20.3: lab 20.2's reference Metal host side of flash attention, unchanged (provided).
//
// buildFlashPipeline chooses B_c for this device, compiles kernel.metal with it, and checks
// that the pipeline can run the kernel as written. encodeFlashAttention binds the buffers
// and dispatches one threadgroup per panel of Q and per (sequence, head).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <string>

static const uint32_t kHeadDim = 128;     // HEAD_DIM in kernel.metal
static const uint32_t kBr = 32;           // B_r: rows of Q per threadgroup
static const NSUInteger kThreads = 512;   // BLOCK_SIZE: 16 SIMD-groups of 32
static const NSUInteger kWidth = 32;      // the lane layout needs SIMD-groups of exactly 32

struct FlashArgs { uint32_t n; float scale; };   // the same layout as in kernel.metal

// Threadgroup memory the kernel's three tiles take for a given B_c (KT_SIZE, S_i and V_j in
// kernel.metal): 37,376 bytes for the book's B_c = 32, 18,688 for 16 (ch20-facts F1).
static NSUInteger tileBytes(uint32_t bc) {
    const uint32_t kt = bc * kHeadDim + ((bc * kHeadDim) >> 5);
    return sizeof(float) * (kt + kBr * bc + bc * kHeadDim);
}

id<MTLComputePipelineState> buildFlashPipeline(id<MTLDevice> device, const char* sourcePath, uint32_t* bcOut,
                                               std::string* why) {
    NSError* err = nil;
    NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:sourcePath]
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (!src) {
        *why = std::string("cannot read ") + sourcePath;
        return nil;
    }
    // The largest B_c whose tiles fit. The book's 32 does not fit Apple's 32 KB (C1).
    const NSUInteger have = device.maxThreadgroupMemoryLength;
    uint32_t bc = 0;
    for (uint32_t candidate : {32u, 16u, 8u}) {
        if (tileBytes(candidate) <= have) {
            bc = candidate;
            break;
        }
    }
    if (bc == 0) {
        *why = "even B_c = 8 needs " + std::to_string(tileBytes(8)) + " bytes of threadgroup memory; this device has " +
               std::to_string((unsigned long)have);
        return nil;
    }
    // Compile with that B_c, as a preprocessor macro (ch05-facts C7, ch20-facts C6).
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.preprocessorMacros = @{@"TILE_WIDTH" : [NSString stringWithFormat:@"%u", bc]};
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        *why = std::string("kernel.metal did not compile: ") + (err ? err.localizedDescription.UTF8String : "unknown");
        return nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"flash_attention_forward"];
    if (!fn) {
        *why = "kernel.metal has no kernel named flash_attention_forward";
        return nil;
    }
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!p) {
        *why = std::string("pipeline creation failed: ") + (err ? err.localizedDescription.UTF8String : "unknown");
        return nil;
    }
    // The kernel's layout: 512 threads in SIMD-groups of exactly 32 (ch05-facts C3: the
    // width and the thread limit are properties of the pipeline, not of the device).
    if (p.threadExecutionWidth != kWidth) {
        *why = "this pipeline's SIMD-group width is " + std::to_string((unsigned long)p.threadExecutionWidth) +
               "; the kernel's lane layout needs 32";
        return nil;
    }
    if (p.maxTotalThreadsPerThreadgroup < kThreads) {
        *why = "this pipeline runs at most " + std::to_string((unsigned long)p.maxTotalThreadsPerThreadgroup) +
               " threads per threadgroup; the kernel needs 512";
        return nil;
    }
    *bcOut = bc;
    return p;
}

bool encodeFlashAttention(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> Q,
                          id<MTLBuffer> K, id<MTLBuffer> V, id<MTLBuffer> O, id<MTLBuffer> L, uint32_t batch,
                          uint32_t heads, uint32_t n, float scale) {
    if (!p) return false;
    if (batch == 0 || heads == 0 || n == 0) return true;   // nothing to do
    FlashArgs a = {n, scale};
    [enc setComputePipelineState:p];
    [enc setBuffer:Q offset:0 atIndex:0];
    [enc setBuffer:K offset:0 atIndex:1];
    [enc setBuffer:V offset:0 atIndex:2];
    [enc setBuffer:O offset:0 atIndex:3];
    [enc setBuffer:L offset:0 atIndex:4];
    [enc setBytes:&a length:sizeof(a) atIndex:5];
    // x: one threadgroup per panel of B_r rows; y: one per (sequence, head).
    [enc dispatchThreadgroups:MTLSizeMake((n + kBr - 1) / kBr, batch * heads, 1)
        threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
    return true;
}
