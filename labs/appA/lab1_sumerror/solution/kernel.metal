// Lab A.1 reference solution (Metal Shading Language). The same three sums as kernel.cu:
// sum_serial, sum_tree and sum_kahan, each a template on the accumulator type and
// instantiated for float, half and bfloat under the names the driver asks for
// ("sum_tree_half", ...).
//
// CUDA                               Metal
// __shared__ AccT a[..]              threadgroup AccT a[..]   (in the kernel body)
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
// threadIdx.x, blockIdx.x            [[thread_position_in_threadgroup]], [[threadgroup_position_in_grid]]
// blockDim.x                         [[threads_per_threadgroup]]
// __half, __nv_bfloat16              half, bfloat (Metal 3.1 and later; appA-facts C1)
// __fadd_rn, __fsub_rn               no counterpart: see the pragma below
#include <metal_stdlib>
using namespace metal;

// The compensated loop computes c = (t - sum) - y, which is 0 in real arithmetic. Metal's
// default math mode, which the lab runtime uses (default MTLCompileOptions), allows
// reassociation in both of its documented candidates, fast and relaxed; only `safe` forbids
// it (notes/today/appA-facts.md C3, ch05-facts C7). On the machine this lab was built on, the
// compensated kernel without this line returned the plain sum. It applies to the whole file.
#pragma METAL fp math_mode(safe)

#define MAX_BLOCK 1024   // the largest threadgroup these kernels accept

// ---------------------------------------------------------------- (a)
template <typename AccT>
kernel void sum_serial(device const float* x   [[buffer(0)]],
                       device float*       out [[buffer(1)]],
                       constant uint&      n   [[buffer(2)]]) {
    AccT sum = AccT(0.0f);
    for (uint i = 0; i < n; ++i) {
        sum = sum + AccT(x[i]);          // one rounding to AccT per addition
    }
    out[0] = float(sum);
}

// ---------------------------------------------------------------- the threadgroup's tree
// Fig. 10.10's tree on the per-thread sums; thread 0 writes the threadgroup's sum.
template <typename AccT>
void tree_to_output(threadgroup AccT* partial_s, AccT value, device float* out,
                    uint t, uint b, uint bdim) {
    partial_s[t] = value;
    for (uint stride = bdim / 2; stride >= 1; stride /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < stride) {
            partial_s[t] = partial_s[t] + partial_s[t + stride];
        }
    }
    if (t == 0) {
        out[b] = float(partial_s[0]);
    }
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
    threadgroup AccT partial_s[MAX_BLOCK];
    uint i = b * bdim * coarse + t;      // `coarse` elements, bdim apart
    AccT sum = AccT(0.0f);
    for (uint j = 0; j < coarse; ++j, i += bdim) {
        if (i < n) {
            sum = sum + AccT(x[i]);
        }
    }
    tree_to_output(partial_s, sum, out, t, b, bdim);
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
    threadgroup AccT partial_s[MAX_BLOCK];
    uint i = b * bdim * coarse + t;
    AccT sum = AccT(0.0f);
    AccT c = AccT(0.0f);                 // minus what the last addition lost
    for (uint j = 0; j < coarse; ++j, i += bdim) {
        if (i < n) {
            AccT y = AccT(x[i]) - c;     // the element plus what was lost before
            AccT next = sum + y;         // big + small: low bits of y fall off here
            c = (next - sum) - y;        // (what was really added) - (what we meant to add)
            sum = next;
        }
    }
    tree_to_output(partial_s, sum, out, t, b, bdim);
}

// ---------------------------------------------------------------- the names the driver asks for
template [[host_name("sum_serial_float")]]  kernel void sum_serial<float>(device const float*, device float*, constant uint&);
template [[host_name("sum_serial_half")]]   kernel void sum_serial<half>(device const float*, device float*, constant uint&);
template [[host_name("sum_serial_bfloat")]] kernel void sum_serial<bfloat>(device const float*, device float*, constant uint&);

template [[host_name("sum_tree_float")]]  kernel void sum_tree<float>(device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);
template [[host_name("sum_tree_half")]]   kernel void sum_tree<half>(device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);
template [[host_name("sum_tree_bfloat")]] kernel void sum_tree<bfloat>(device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);

template [[host_name("sum_kahan_float")]]  kernel void sum_kahan<float>(device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);
template [[host_name("sum_kahan_half")]]   kernel void sum_kahan<half>(device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);
template [[host_name("sum_kahan_bfloat")]] kernel void sum_kahan<bfloat>(device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);
