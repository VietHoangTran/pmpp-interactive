// Lab X3.2, provided: the M9-to-M10 adapter, Metal host side. provided/m9_linear.h says why
// it exists and what it does. Not in the book.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <map>
#include <string>

#include "pmpp/metal.h"
#include "m9_linear.h"
#include "../../../x2/lab4_m9/qdecode.h"   // lab X2.4's QMatrix, QPipelines and encodeQLinear

static std::map<const void*, M9Weight> gBound;
static std::string gProblem;
static QPipelines gPipelines;
static bool gBuilt = false;

static void note(const std::string& p) {
    if (gProblem.empty()) gProblem = p;
}

void m9Bind(id<MTLBuffer> W, const M9Weight& q) { gBound[(__bridge const void*)W] = q; }
void m9Clear() { gBound.clear(); }

std::string m9TakeProblem() {
    std::string p = gProblem;
    gProblem.clear();
    return p;
}

// The GEMVs as lab X2.4's driver compiles them: each from its own lab's kernel.metal, with no
// constants (pieces.cmake passes the paths and the kernel names).
std::string m9Pipelines() {
    static std::string problem;
    if (gBuilt) return problem;
    gBuilt = true;
    gPipelines.gemvF32 = pmpp::metal::pipeline(PMPP_GEMV_METAL, "gemv_gather");
    if (!gPipelines.gemvF32) return problem = "lab 22.1's gemv_gather did not build: " + pmpp::take_error();
    gPipelines.gemvQ8_0 = pmpp::metal::pipeline(PMPP_Q8_0_METAL, PMPP_Q8_0_KERNEL);
    if (!gPipelines.gemvQ8_0) return problem = "the Q8_0 GEMV (" PMPP_Q8_0_FROM ") did not build: " + pmpp::take_error();
    gPipelines.gemvQ4_0 = pmpp::metal::pipeline(PMPP_Q4_0_METAL, PMPP_Q4_0_KERNEL);
    if (!gPipelines.gemvQ4_0) return problem = "the Q4_0 GEMV (" PMPP_Q4_0_FROM ") did not build: " + pmpp::take_error();
    return problem;
}

void encodeLinearM9(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> /*pipeline*/, id<MTLBuffer> X,
                    id<MTLBuffer> W, id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out) {
    auto it = gBound.find((__bridge const void*)W);
    if (it == gBound.end()) {
        note("the step passed a weight buffer that is not one of the model's matrices");
        return;
    }
    const M9Weight& w = it->second;
    if (w.rows != out || w.cols != in) {
        char b[200];
        std::snprintf(b, sizeof b, "a weight of shape (out %u, in %u) was called with out %u, in %u", w.rows, w.cols, out,
                      in);
        note(b);
        return;
    }
    if (batch > M9_MAX_BATCH) {
        note("batch above 8: M9's GEMVs are matrix-vector kernels");
        return;
    }
    const std::string p = m9Pipelines();
    if (!p.empty()) {
        note(p);
        return;
    }
    QMatrix q{};
    q.fmt = (QFormat)w.fmt;
    q.rows = w.rows;
    q.cols = w.cols;
    q.f32 = w.f32;
    q.q8 = w.q8;
    q.q8d = w.q8d;
    q.q4 = w.q4;
    encodeQLinear(enc, gPipelines, X, q, Y, batch);   // lab X2.4's reference solution
}
