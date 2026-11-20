// Lab 17.1, provided (do not edit): the Metal version of provided/coo_spmv.cu, SpMV/COO
// (Fig. 17.5). Read that file's header for what it does.
//
// Metal differences (notes/today/ch17-facts.md C1; infra-facts C1; ch09-facts C1):
//   - MSL has no atomic_ref. The kernel declares y `device atomic_float*` and adds with
//     atomic_fetch_add_explicit. Float atomic add exists in device memory from Metal 3 and
//     Apple7, and relaxed is the order the spec allows for it.
//   - The buffer itself holds plain floats: the ELL kernel binds the same buffer as
//     `device float*`. The atomic type belongs to this kernel's declaration.
//   - dispatchThreads launches exactly numNonzeros threads (ch05-facts C5); the test on i
//     is kept from the CUDA kernel and never fails here.
#include <metal_stdlib>
using namespace metal;

kernel void spmv_coo(device const int*   rowIdx      [[buffer(0)]],
                     device const int*   colIdx      [[buffer(1)]],
                     device const float* values      [[buffer(2)]],
                     device const float* x           [[buffer(3)]],
                     device atomic_float* y          [[buffer(4)]],
                     constant uint&      numNonzeros [[buffer(5)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < numNonzeros) {
        int row = rowIdx[i];
        int col = colIdx[i];
        float value = values[i];
        atomic_fetch_add_explicit(&y[row], x[col] * value, memory_order_relaxed);
    }
}
