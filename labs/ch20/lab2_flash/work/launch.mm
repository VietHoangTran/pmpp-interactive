// Lab 20.2 starter: the Metal host side of flash attention (Exercise 1, on Metal).
//
// buildFlashPipeline chooses B_c for this device, compiles kernel.metal with it, and checks
// that the pipeline can run the kernel as written. It is called once; when it returns nil,
// *why is printed as the reason the kernel cannot run. encodeFlashAttention binds the
// buffers and dispatches one threadgroup per panel of Q and per (sequence, head).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <string>

static const uint32_t kHeadDim = 128;     // HEAD_DIM in kernel.metal
static const uint32_t kBr = 32;           // B_r: rows of Q per threadgroup
static const NSUInteger kThreads = 512;   // BLOCK_SIZE: 16 SIMD-groups of 32
static const NSUInteger kWidth = 32;      // the lane layout needs SIMD-groups of exactly 32

struct FlashArgs { uint32_t n; float scale; };   // the same layout as in kernel.metal

// Threadgroup memory the kernel's three tiles take for a given B_c: KT_j, S_i and V_j in
// kernel.metal (KT_SIZE includes the padding).
static NSUInteger tileBytes(uint32_t bc) {
    // TODO 1: the bytes of the three arrays for this B_c. Check yourself: the book's B_c = 32
    //         gives 37,376 bytes (ch20-facts F1).
    return 0;
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

    // TODO 2: the largest B_c in {32, 16, 8} whose tiles fit in the threadgroup memory this
    //         device offers (device.maxThreadgroupMemoryLength). The book's 32 may not fit:
    //         Apple documents 32 KB (ch20-facts C1). If none fits, set *why to a message that
    //         says how much was needed and how much there is, and return nil.
    uint32_t bc = 0;
    if (bc == 0) {
        *why = "TODO 2: launch.mm has not chosen B_c yet";
        return nil;
    }

    // TODO 3: compile kernel.metal with TILE_WIDTH defined as that B_c, through
    //         opts.preprocessorMacros (a dictionary of NSString names to values).
    MTLCompileOptions* opts = [MTLCompileOptions new];
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

    // TODO 4: the kernel's layout needs SIMD-groups of exactly kWidth threads and kThreads
    //         threads per threadgroup. Both are properties of this pipeline, not of the device
    //         (ch05-facts C3): check p.threadExecutionWidth and p.maxTotalThreadsPerThreadgroup,
    //         and return nil with a clear *why when either does not hold.

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
    // TODO 5: dispatch one threadgroup of kThreads threads per panel of kBr rows along x and
    //         one per (sequence, head) along y. Metal takes all three sizes (ch05-facts C5):
    //   [enc dispatchThreadgroups:MTLSizeMake(?, ?, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
    return true;
}
