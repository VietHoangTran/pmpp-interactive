// Lab A.1 starter (Metal Shading Language). The same three sums as kernel.cu: sum_serial,
// sum_tree and sum_kahan, each a template on the accumulator type, instantiated at the bottom
// of this file for float, half and bfloat under the names the driver asks for
// ("sum_tree_half", ...). Same contract and same orders as kernel.cu; read its header.
//
// CUDA                               Metal
// __shared__ AccT a[..]              threadgroup AccT a[..]   (in the kernel body)
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
// threadIdx.x, blockIdx.x            [[thread_position_in_threadgroup]], [[threadgroup_position_in_grid]]
// blockDim.x                         [[threads_per_threadgroup]]
// __half, __nv_bfloat16              half, bfloat (Metal 3.1 and later; appA-facts C1)
// __fadd_rn, __fsub_rn               no counterpart: plain + and -
#include <metal_stdlib>
using namespace metal;

#define MAX_BLOCK 1024   // the largest threadgroup these kernels accept

// ---------------------------------------------------------------- (a)
template <typename AccT>
kernel void sum_serial(device const float* x   [[buffer(0)]],
                       device float*       out [[buffer(1)]],
                       constant uint&      n   [[buffer(2)]]) {
    // TODO 1: as kernel.cu. AccT(0.0f), AccT(x[i]) and float(v) convert, as there.
}

// ---------------------------------------------------------------- (b)
template <typename AccT>
kernel void sum_tree(device const float* x      [[buffer(0)]],
                     device float*       out    [[buffer(1)]],
                     constant uint&      n      [[buffer(2)]],
                     constant uint&      coarse [[buffer(3)]],
                     uint t    [[thread_position_in_threadgroup]],
                     uint b    [[threadgroup_position_in_grid]],
                     uint bdim [[threads_per_threadgroup]]) {
    // TODO 2: as kernel.cu: the coarsening loop, the threadgroup's tree, thread 0 writes out[b].
    //         A helper function that takes a `threadgroup AccT*` can hold the tree for (c) too.
}

// ---------------------------------------------------------------- (c)
template <typename AccT>
kernel void sum_kahan(device const float* x      [[buffer(0)]],
                      device float*       out    [[buffer(1)]],
                      constant uint&      n      [[buffer(2)]],
                      constant uint&      coarse [[buffer(3)]],
                      uint t    [[thread_position_in_threadgroup]],
                      uint b    [[threadgroup_position_in_grid]],
                      uint bdim [[threads_per_threadgroup]]) {
    // TODO 3: as kernel.cu, with plain + and - for the compensation. If the test
    //         "kahan, float, big-small" fails here and passes on emu, README.md has a hint.
}

// ---------------------------------------------------------------- the names the driver asks for
// Provided. Keep these in step with the signatures above if you change them.
template [[host_name("sum_serial_float")]]  kernel void sum_serial<float>(device const float*, device float*, constant uint&);
template [[host_name("sum_serial_half")]]   kernel void sum_serial<half>(device const float*, device float*, constant uint&);
template [[host_name("sum_serial_bfloat")]] kernel void sum_serial<bfloat>(device const float*, device float*, constant uint&);

template [[host_name("sum_tree_float")]]  kernel void sum_tree<float>(device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);
template [[host_name("sum_tree_half")]]   kernel void sum_tree<half>(device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);
template [[host_name("sum_tree_bfloat")]] kernel void sum_tree<bfloat>(device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);

template [[host_name("sum_kahan_float")]]  kernel void sum_kahan<float>(device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);
template [[host_name("sum_kahan_half")]]   kernel void sum_kahan<half>(device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);
template [[host_name("sum_kahan_bfloat")]] kernel void sum_kahan<bfloat>(device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);
