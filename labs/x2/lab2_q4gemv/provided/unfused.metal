// PROVIDED, do not edit. Lab X2.2's unfused baseline on Metal, first half: dequantise the
// Q4_0 weights into a float32 scratch matrix. encodeGemvQ4_0Unfused (unfused_launch.mm) then
// dispatches lab 22.1's gemv_gather on it, in the same serial encoder. For the benchmark
// only; not in the book; ours.
#include <metal_stdlib>
using namespace metal;

#define QK4_0 32

struct block_q4_0 {          // llama.cpp's declaration on Metal (notes/today/x2-facts.md C5, D1)
    half d;
    uchar qs[QK4_0 / 2];
};

// One thread per byte of qs: byte t of block i gives weights 32i + t and 32i + 16 + t.
kernel void dequant_q4_0(device const block_q4_0* W     [[buffer(0)]],
                         device float*            Wf    [[buffer(1)]],
                         constant uint&           bytes [[buffer(2)]],
                         uint g [[thread_position_in_grid]]) {
    if (g >= bytes) {
        return;
    }
    const uint i = g / (QK4_0 / 2);
    const uint t = g % (QK4_0 / 2);
    const float d = float(W[i].d);
    const uchar q = W[i].qs[t];
    Wf[i * QK4_0 + t] = float(int(q & 0x0F) - 8) * d;
    Wf[i * QK4_0 + QK4_0 / 2 + t] = float(int(q >> 4) - 8) * d;
}
