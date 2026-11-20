// PROVIDED, do not edit. The Metal counterpart of minp_g11.cu: min-p on a sorted row as a
// count, and n = max(1, min(kept, countMinP)). One threadgroup of 256 threads per row; the
// threadgroup memory for the per-SIMD-group counts is sized by minp_g11_launch.mm.
//
// The test l_i >= l_0 + ln(minP) is a float addition and a comparison of finite values; ln(minP)
// arrives from the host, so no log runs on the device. minP <= 0 arrives as use_min_p = 0: no
// minus infinity ever reaches the kernel (under fast math INF is undefined, ch10-facts C3).
#include <metal_stdlib>
using namespace metal;

kernel void minp_combine(device const float* sorted_logits [[buffer(0)]],
                         device const int*   kept          [[buffer(1)]],
                         device int*         n             [[buffer(2)]],
                         constant uint&      cols          [[buffer(3)]],
                         constant float&     ln_min_p      [[buffer(4)]],
                         constant uint&      use_min_p     [[buffer(5)]],
                         threadgroup int*    group_counts  [[threadgroup(0)]],
                         uint row    [[threadgroup_position_in_grid]],
                         uint t      [[thread_position_in_threadgroup]],
                         uint bdim   [[threads_per_threadgroup]],
                         uint lane   [[thread_index_in_simdgroup]],
                         uint sg     [[simdgroup_index_in_threadgroup]],
                         uint groups [[simdgroups_per_threadgroup]]) {
    device const float* r = sorted_logits + ulong(row) * cols;
    int count = 0;
    if (use_min_p != 0) {
        const float threshold = r[0] + ln_min_p;
        for (uint i = t; i < cols; i += bdim) {
            count += r[i] >= threshold ? 1 : 0;
        }
    }
    count = simd_sum(count);
    if (lane == 0) {
        group_counts[sg] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        int total = 0;
        for (uint g = 0; g < groups; ++g) {
            total += group_counts[g];
        }
        const int count_min_p = use_min_p != 0 ? total : int(cols);
        n[row] = max(1, min(kept[row], count_min_p));
    }
}
