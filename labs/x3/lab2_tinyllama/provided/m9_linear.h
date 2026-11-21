// Lab X3.2, provided: milestone M9's quantised linear layer behind the project's LinearFn, so
// that the step you wrote for M10 runs a quantised model with no change. Not in the book.
//
// The two milestones disagree on how a weight is passed. M10's step calls
//     m.linear(X, W, Y, batch, in, out)                  W: a float pointer (LinearFn)
// and M9's layer (lab X2.4's reference solution) is
//     launchQLinear(X, const QMatrix& W, Y, batch)       W: its format, shape and blocks
// The adapter keeps a table from the float pointer the step passes to the quantised copy of
// that matrix. The driver quantises every projection and the tied embedding, keeps the
// dequantised floats in the float buffers (the embedding reads them, as M9's dequantising
// embedding would give, and the CPU reference computes with them), and binds each float
// buffer to its quantised copy here. launchLinearM9 then looks W up and calls M9's
// launchQLinear with the copy; it never reads the float W. A W that was never bound, or a
// shape that differs from the bound one, launches nothing and leaves a problem the driver
// reports (m9TakeProblem): a step must pass the model's own weight pointers.
//
// M9's GEMVs are matrix-vector kernels: batch at most 8 (QDECODE_MAX_BATCH), and every
// quantised matrix needs `in` (its columns) a multiple of 32, the block size.
#pragma once
#include <cstdint>
#include <string>

// Lab X2.4's QFormat values.
enum M9Format { M9_F32 = 0, M9_Q8_0 = 1, M9_Q4_0 = 2 };
#define M9_MAX_BATCH 8

#if !defined(__OBJC__)
// ---------------------------------------------------------------- CUDA, HIP, emulators
// Device pointers. f32 for M9_F32 (usually the bound W itself); q8 and q8d for M9_Q8_0
// (int8 codes, FP16 scale bits); q4 for M9_Q4_0 (llama.cpp's 18-byte block_q4_0).
struct M9Weight {
    int fmt;
    int rows, cols;                // (out, in)
    const float* f32;
    const int8_t* q8;
    const uint16_t* q8d;
    const void* q4;
};
void m9Bind(const float* W, const M9Weight& q);
void m9Clear();
// LinearFn: Y[batch x out] = X[batch x in] W^T with W's quantised copy. Launches, does not wait.
void launchLinearM9(const float* X, const float* W, float* Y, int batch, int in, int out);

#else
// ---------------------------------------------------------------- Metal
#import <Metal/Metal.h>
struct M9Weight {
    int fmt;
    uint32_t rows, cols;
    id<MTLBuffer> f32, q8, q8d, q4;
};
void m9Bind(id<MTLBuffer> W, const M9Weight& q);
void m9Clear();
// Builds the three GEMV pipelines M9's encodeQLinear dispatches to (lab 22.1's gather, lab
// X2.1's Q8_0 and lab X2.2's Q4_0 GEMVs, or their stand-ins); "" or why one did not build.
std::string m9Pipelines();
// EncodeLinearFn. `pipeline` is ignored: M9 picks the GEMV for the bound matrix's format.
void encodeLinearM9(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, id<MTLBuffer> X,
                    id<MTLBuffer> W, id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
#endif

// The first problem since the last call ("" if none), and clears it.
std::string m9TakeProblem();
