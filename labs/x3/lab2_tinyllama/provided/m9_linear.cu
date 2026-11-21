// Lab X3.2, provided: the M9-to-M10 adapter (CUDA; also built for HIP and for the CPU
// emulators). provided/m9_linear.h says why it exists and what it does. Not in the book.
#include <cstdio>
#include <map>
#include <string>

// Both paths resolve from this file's directory and, on the emulators (whose translated copy
// lives in the build directory), from the variant directory on the include path.
#include "../provided/m9_linear.h"
#include "../../../x2/lab4_m9/qdecode.h"   // lab X2.4's QMatrix and launchQLinear

static std::map<const float*, M9Weight> gBound;
static std::string gProblem;

static void note(const std::string& p) {
    if (gProblem.empty()) gProblem = p;
}

void m9Bind(const float* W, const M9Weight& q) { gBound[W] = q; }
void m9Clear() { gBound.clear(); }

std::string m9TakeProblem() {
    std::string p = gProblem;
    gProblem.clear();
    return p;
}

void launchLinearM9(const float* X, const float* W, float* Y, int batch, int in, int out) {
    auto it = gBound.find(W);
    if (it == gBound.end()) {
        note("the step passed a weight pointer that is not one of the model's matrices");
        return;
    }
    const M9Weight& w = it->second;
    if (w.rows != out || w.cols != in) {
        char b[200];
        std::snprintf(b, sizeof b, "a weight of shape (out %d, in %d) was called with out %d, in %d", w.rows, w.cols, out,
                      in);
        note(b);
        return;
    }
    if (batch > M9_MAX_BATCH) {
        note("batch above 8: M9's GEMVs are matrix-vector kernels");
        return;
    }
    QMatrix q{};
    q.fmt = (QFormat)w.fmt;
    q.rows = w.rows;
    q.cols = w.cols;
    q.f32 = w.f32;
    q.q8 = w.q8;
    q.q8d = w.q8d;
    q.q4 = (const BlockQ4_0*)w.q4;
    launchQLinear(X, q, Y, batch);   // lab X2.4's reference solution
}
