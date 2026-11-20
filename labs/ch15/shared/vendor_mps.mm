// Labs 15.1 to 15.5, provided and driver-side: you do not edit or call this.
// Apple's single-precision GEMM, the Metal baseline milestone M3 is measured against:
// MPSMatrixMultiplication from Metal Performance Shaders (notes/today/ch15-facts.md C6).
// Not in the book, which names only NVIDIA's libraries.
//
// - MPSMatrix is row-major (ch03-facts C7), the course's layout, so no transposes.
// - It computes C = alpha * A * B + beta * C; here alpha = 1, beta = 0.
// - Precision: MPS keeps a higher precision internally unless a kernel is given
//   MPSKernelOptionsAllowReducedPrecision, which this file never sets (C6).
// - Signatures checked against the macOS 27.0 SDK headers (MPSMatrix.h,
//   MPSMatrixMultiplication.h).
//
// The lab runtime's pmpp::run opens a compute encoder before it calls the lab's enqueue
// function. MPS encodes onto the command buffer itself and needs no open encoder, so this
// file ends the pass's encoder, encodes the multiplication, and opens a fresh, empty one
// for pmpp::run to end. Everything stays in one command buffer, so the time reported is
// the GPU time of the MPS kernel(s).
//
// Built without MPS when the build is configured with -DPMPP_CH15_VENDOR=OFF.
#import <Metal/Metal.h>
#include <cstdint>
#include <string>

#include "pmpp/metal.h"

#if !defined(PMPP_CH15_NO_VENDOR)
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

std::string ch15VendorName() {
    return "MPSMatrixMultiplication (Metal Performance Shaders), float32, reduced precision not allowed";
}

std::string ch15VendorEncode(pmpp::metal::Pass* pass, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
                             uint32_t M, uint32_t K, uint32_t N) {
    MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
                                                                  rowBytes:(NSUInteger)K * sizeof(float)
                                                                  dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N
                                                                  rowBytes:(NSUInteger)N * sizeof(float)
                                                                  dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* dC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
                                                                  rowBytes:(NSUInteger)N * sizeof(float)
                                                                  dataType:MPSDataTypeFloat32];
    MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:A descriptor:dA];
    MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:B descriptor:dB];
    MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:C descriptor:dC];
    MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc] initWithDevice:pass->device
                                                                      transposeLeft:NO
                                                                     transposeRight:NO
                                                                         resultRows:M
                                                                      resultColumns:N
                                                                    interiorColumns:K
                                                                              alpha:1.0
                                                                               beta:0.0];
    if (!mA || !mB || !mC || !gemm) return "MPS objects could not be created";
    [pass->encoder endEncoding];
    [gemm encodeToCommandBuffer:pass->commandBuffer leftMatrix:mA rightMatrix:mB resultMatrix:mC];
    pass->encoder = [pass->commandBuffer computeCommandEncoder];
    return "";
}

#else

std::string ch15VendorName() { return "none (configured with -DPMPP_CH15_VENDOR=OFF)"; }

std::string ch15VendorEncode(pmpp::metal::Pass*, id<MTLBuffer>, id<MTLBuffer>, id<MTLBuffer>,
                             uint32_t, uint32_t, uint32_t) {
    return "switched off at configure time (-DPMPP_CH15_VENDOR=OFF)";
}

#endif
