// PROVIDED, do not edit. What the Metal side of lab X1.3 shares between main.cpp and your
// launch.mm: the pipelines of every stage, the workspace, and the earlier labs' encoders.
//
// Why the Metal workspace is a set of buffers, not one: the earlier labs' encoders (linked in
// from their solution/launch.mm files) bind every buffer at offset 0. Carving one MTLBuffer into
// regions would need an offset at every bind, so each intermediate gets its own buffer here.
// On CUDA and HIP the workspace is one allocation that your launchSampleRows carves itself.
#pragma once
#import <Metal/Metal.h>
#include <cstdint>

// Every pipeline the chain needs. main.cpp builds them: yours from your kernel.metal, the rest
// from the earlier labs' solution kernel.metal files and from labs/x1/shared-g11/.
struct SamplerPipelines {
  id<MTLComputePipelineState> copyRows, applyRepetition, drawRows, philoxKat;   // your kernel.metal
  id<MTLComputePipelineState> applyPenalties, appendTokens;                      // lab 9.3
  id<MTLComputePipelineState> argmax, softmax;                                   // lab 10.3
  id<MTLComputePipelineState> topkThreshold, topkGather;                         // lab X1.1 (topk_g11.metal)
  id<MTLComputePipelineState> descKeys, sortPass, sortGather;                    // lab 14.3
  id<MTLComputePipelineState> topP;                                              // lab 11.6
  id<MTLComputePipelineState> minP;                                              // minp_g11.metal
};

inline uint32_t effectiveTopK(uint32_t vocab, int topK) {
  return topK <= 0 || (uint32_t)topK >= vocab ? vocab : (uint32_t)topK;
}

// The intermediates of the chain for `rows` rows, K = effectiveTopK(vocab, topK) candidates.
struct SampleWorkspace {
  uint32_t rows = 0, vocab = 0, k = 0;
  id<MTLBuffer> logits;       // [rows x vocab] float: the penalised logits
  id<MTLBuffer> candIdx;      // [rows x K] int: lab X1.1's columns, ascending
  id<MTLBuffer> candVal;      // [rows x K] float: their logits
  id<MTLBuffer> selScratch;   // [rows x 4] uint: lab X1.1's scratch
  id<MTLBuffer> sortedVal;    // [rows x K] float: the candidates, largest first (lab 14.3)
  id<MTLBuffer> sortedPos;    // [rows x K] int: their positions in candIdx
  id<MTLBuffer> sortScratch;  // [4 x rows x K] uint: lab 14.3's scratch
  id<MTLBuffer> probs;        // [rows x K] float: softmax of sortedVal (lab 10.3)
  id<MTLBuffer> mass;         // [rows x K] float: lab 11.6's mass before each token
  id<MTLBuffer> keep;         // [rows x K] uchar: lab 11.6's keep flags
  id<MTLBuffer> kept;         // [rows] int: top-p's count
  id<MTLBuffer> n;            // [rows] int: survivors of top-p and min-p
};

inline SampleWorkspace makeSampleWorkspace(id<MTLDevice> device, uint32_t rows, uint32_t vocab, int topK) {
  SampleWorkspace w;
  w.rows = rows;
  w.vocab = vocab;
  w.k = effectiveTopK(vocab, topK);
  const NSUInteger R = rows, V = vocab, K = w.k;
  auto make = [&](NSUInteger bytes) { return [device newBufferWithLength:(bytes ? bytes : 16) options:MTLResourceStorageModeShared]; };
  w.logits = make(R * V * 4);
  w.candIdx = make(R * K * 4);
  w.candVal = make(R * K * 4);
  w.selScratch = make(R * 4 * 4);
  w.sortedVal = make(R * K * 4);
  w.sortedPos = make(R * K * 4);
  w.sortScratch = make(4 * R * K * 4);
  w.probs = make(R * K * 4);
  w.mass = make(R * K * 4);
  w.keep = make(R * K);
  w.kept = make(R * 4);
  w.n = make(R * 4);
  return w;
}

// ---- the earlier labs' encoders, linked in by CMakeLists.txt (SHARED_MM) ----
// Lab 9.3 (labs/ch09/lab3_tokencounts/solution/launch.mm)
void encodeAppendTokens(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                        id<MTLBuffer> newTokens, id<MTLBuffer> counts, uint32_t batch, uint32_t vocab);
void encodeApplyPenalties(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                          id<MTLBuffer> logits, id<MTLBuffer> counts, uint32_t batch, uint32_t vocab,
                          float frequencyPenalty, float presencePenalty);
// Lab 10.3 (labs/ch10/lab3_softmax_argmax/solution/launch.mm)
bool encodeSoftmaxRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                       id<MTLBuffer> x, id<MTLBuffer> y, uint32_t rows, uint32_t cols, float scale);
bool encodeArgmaxRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                      id<MTLBuffer> x, id<MTLBuffer> idx, uint32_t rows, uint32_t cols);
// Lab X1.1 (labs/x1/shared-g11/topk_g11_launch.mm)
bool encodeTopKRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> threshold,
                    id<MTLComputePipelineState> gather, id<MTLBuffer> logits, id<MTLBuffer> outIdx,
                    id<MTLBuffer> outVal, id<MTLBuffer> scratch, uint32_t rows, uint32_t cols, uint32_t k);
// Lab 14.3 (labs/ch14/lab3_sortrows/solution/launch.mm)
bool encodeSortRowsDesc(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> descKeys,
                        id<MTLComputePipelineState> sortPass, id<MTLComputePipelineState> gather,
                        id<MTLBuffer> probs, id<MTLBuffer> sortedProbs, id<MTLBuffer> sortedIdx,
                        id<MTLBuffer> scratch, uint32_t rows, uint32_t cols);
// Lab 11.6 (labs/ch11/lab6_topp/solution/launch.mm)
bool encodeTopPMask(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> probs, id<MTLBuffer> massBefore, id<MTLBuffer> keep, id<MTLBuffer> kept,
                    uint32_t rows, uint32_t cols, float p);
// Min-p and the combine (labs/x1/shared-g11/minp_g11_launch.mm)
bool encodeMinPCombine(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                       id<MTLBuffer> sortedLogits, id<MTLBuffer> kept, id<MTLBuffer> n,
                       uint32_t rows, uint32_t cols, float minP);

// ---- yours, in launch.mm ----
bool encodeCopyRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> in,
                    id<MTLBuffer> out, uint32_t count);
bool encodePhiloxKat(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> in,
                     id<MTLBuffer> out, uint32_t count);
bool encodeApplyRepetition(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> logits,
                           id<MTLBuffer> counts, uint32_t rows, uint32_t vocab, float penalty);
// uIn may be nil (u from Philox); uOut may be nil (not written).
bool encodeDrawRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> sortedLogits,
                    id<MTLBuffer> sortedPos, id<MTLBuffer> candIdx, id<MTLBuffer> n, id<MTLBuffer> seeds,
                    id<MTLBuffer> streams, id<MTLBuffer> steps, id<MTLBuffer> uIn, id<MTLBuffer> tokens,
                    id<MTLBuffer> uOut, uint32_t rows, uint32_t stride, float invT);
// uOut may be nil. ws must have been made for these rows, vocab and topK.
bool encodeSampleRows(id<MTLComputeCommandEncoder> enc, const SamplerPipelines& p, const SampleWorkspace& ws,
                      id<MTLBuffer> logits, id<MTLBuffer> counts, id<MTLBuffer> seeds, id<MTLBuffer> streams,
                      id<MTLBuffer> steps, id<MTLBuffer> tokens, id<MTLBuffer> uOut, uint32_t rows, uint32_t vocab,
                      int topK, float topP, float minP, float temperature, float repeatPenalty,
                      float frequencyPenalty, float presencePenalty);
