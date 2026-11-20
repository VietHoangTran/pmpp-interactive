// Provided baseline for lab 16.1 (Metal Shading Language): one dispatch per
// anti-diagonal, one thread per cell, every input read from device memory.
// The Metal counterpart of baseline.cu; book section 16.6, p. 384, Fig. 16.7(a).
// The driver compiles this file with SW_MATCH, SW_MISMATCH and SW_GAP defined
// from ../lab1_wavefront/sw.h.
#include <metal_stdlib>
using namespace metal;

#if !defined(SW_MATCH) || !defined(SW_MISMATCH) || !defined(SW_GAP)
#error "the driver defines SW_MATCH, SW_MISMATCH and SW_GAP from sw.h"
#endif

struct SwBaselineParams { int L; int s; int firstRow; int count; };

kernel void sw_baseline(device int*                  H [[buffer(0)]],
                        device const int*            a [[buffer(1)]],
                        device const int*            b [[buffer(2)]],
                        constant SwBaselineParams&   p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    const int t = int(gid);
    if (t < p.count) {
        const int L = p.L;
        const int r = p.firstRow + t;
        const int q = p.s - r;
        const int n  = H[(r - 1) * L + q];
        const int w  = H[r * L + (q - 1)];
        const int nw = H[(r - 1) * L + (q - 1)];
        const int score = (a[r - 1] == b[q - 1]) ? SW_MATCH : SW_MISMATCH;
        int h = 0;
        if (h < nw + score) h = nw + score;
        if (h < w + SW_GAP) h = w + SW_GAP;
        if (h < n + SW_GAP) h = n + SW_GAP;
        H[r * L + q] = h;
    }
}
