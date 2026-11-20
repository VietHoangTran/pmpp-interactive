// Provided instrument for lab 6.3 on Metal: the same probe as provided/bank_probe.cu.
// Read the comment there for what it does and why it is written this way. Do not edit.
//
// The driver dispatches ONE threadgroup of threadExecutionWidth threads, taken from
// this kernel's pipeline. The MSL specification ties that width to a SIMD-group, and
// SIMD-groups are linear, so the threadgroup is taken to be one SIMD-group
// (notes/today/ch05-facts.md C3, notes/today/infra-facts.md C5). Apple documents no
// bank structure for threadgroup memory (ch06 facts C2): this probe is how you find out
// whether the stride matters on your GPU.
#include <metal_stdlib>
using namespace metal;

kernel void bank_probe(device float*  out    [[buffer(0)]],
                       constant int&  stride [[buffer(1)]],
                       constant int&  iters  [[buffer(2)]],
                       constant float& zero  [[buffer(3)]],
                       uint lane  [[thread_position_in_threadgroup]],
                       uint lanes [[threads_per_threadgroup]]) {
    threadgroup float a[1024];
    for (uint k = lane; k < 1024; k += lanes) a[k] = float(k);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int idx = (int(lane) * stride) & 1023;
    float acc = 0.0f;
    for (int it = 0; it < iters; ++it) {
        float v = a[idx];
        a[idx] = v + zero;
        acc += v;
        idx = int(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[lane] = acc + a[idx];
}
