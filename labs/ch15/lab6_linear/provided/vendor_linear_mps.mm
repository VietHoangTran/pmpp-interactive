// Lab 15.6, provided: Apple's GEMM as the baseline for milestone M3 on Metal,
// MPSMatrixMultiplication from Metal Performance Shaders. You do not edit or call this;
// the benchmark does.
//
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major.
//
// MPSMatrix is row-major (ch03-facts C7), so no layout trick is needed, only a flag for
// the operand stored the other way: C = alpha * op(A) * op(B) + beta * C with A = X,
// B = W and transposeRight YES (ch15-facts C6). The tests check this against the CPU
// reference.
//
// Precision: MPS computes with a higher internal precision unless the kernel's options
// include MPSKernelOptionsAllowReducedPrecision, which is opt-in (ch15-facts C6).
// vendorMath() reads the option from the object that runs.
//
// MPS encodes into a command buffer, not into a compute encoder, and a command buffer
// takes one encoder at a time. So this ends the pass's compute encoder, encodes the
// multiplication, and opens a fresh (empty) compute encoder for the harness to close.
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>

#include "pmpp/metal.h"

static MPSMatrixMultiplication* multiplication(id<MTLDevice> device, uint32_t batch, uint32_t in, uint32_t out) {
    static std::map<std::tuple<uint32_t, uint32_t, uint32_t>, MPSMatrixMultiplication*> cache;
    auto key = std::make_tuple(batch, in, out);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                    transposeLeft:NO
                                                                   transposeRight:YES
                                                                       resultRows:batch
                                                                    resultColumns:out
                                                                  interiorColumns:in
                                                                            alpha:1.0
                                                                             beta:0.0];
    cache[key] = mm;
    return mm;
}

static MPSMatrix* matrix(id<MTLBuffer> buffer, uint32_t rows, uint32_t columns) {
    MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                      columns:columns
                                                                     rowBytes:columns * sizeof(float)
                                                                     dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:desc];
}

// Returns false if MPS could not be set up.
bool encodeLinearVendor(pmpp::metal::Pass* pass, id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
                        uint32_t batch, uint32_t in, uint32_t out) {
    MPSMatrixMultiplication* mm = multiplication(pass->device, batch, in, out);
    if (!mm) return false;
    [pass->encoder endEncoding];
    [mm encodeToCommandBuffer:pass->commandBuffer
                   leftMatrix:matrix(X, batch, in)      // X: batch x in
                  rightMatrix:matrix(W, out, in)        // W: out x in, used transposed
                 resultMatrix:matrix(Y, batch, out)];   // Y: batch x out
    pass->encoder = [pass->commandBuffer computeCommandEncoder];
    return true;
}

const char* vendorName() { return "MPSMatrixMultiplication"; }

std::string vendorMath() {
    MPSMatrixMultiplication* mm = multiplication(pmpp::metal::device(), 1, 1, 1);
    if (!mm) return "MPS unavailable";
    return (mm.options & MPSKernelOptionsAllowReducedPrecision)
               ? "MPSKernelOptionsAllowReducedPrecision is SET: reduced precision allowed"
               : "MPSKernelOptionsAllowReducedPrecision not set (FP32 storage, higher internal precision)";
}
