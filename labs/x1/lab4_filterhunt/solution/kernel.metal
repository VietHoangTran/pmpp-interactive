// Lab X1.4 reference solution (Metal Shading Language): lab X1.2's filter_sorted_fused with the
// three defects of starter/ fixed. The contract is the one at the top of kernel.cu.
//
// Where Metal differs from the CUDA file:
//   - simd_sum gives every lane the SIMD-group's sum, for ints as for floats;
//     simd_prefix_exclusive_sum and simd_prefix_inclusive_sum scan in one call
//     (notes/today/infra-facts.md C2, ch11-facts C1).
//   - Lane and SIMD-group indices come from their attributes; every position in the chunk is
//     built from them (ch11-facts C2). The per-SIMD-group arrays have one slot per thread,
//     enough for any width; launch.mm checks that one SIMD-group has a lane for every group.
//   - min-p's "keep everything" threshold is -FLT_MAX, not -INFINITY, whose behaviour fast
//     math leaves undefined (ch10-facts C3).
#include <metal_stdlib>
using namespace metal;

#define BLOCK_DIM 256        // threads per threadgroup (kBlock in launch.mm); one threadgroup per row
#define COARSE_FACTOR 4      // tokens per thread per chunk

// ---------------------------------------------------------------- part B: fused
kernel void filter_sorted_fused(device const float* sortedLogits [[buffer(0)]],
                                device float*       sortedProbs  [[buffer(1)]],
                                device int*         n            [[buffer(2)]],
                                constant uint&      K            [[buffer(3)]],
                                constant float&     topP         [[buffer(4)]],
                                constant float&     minP         [[buffer(5)]],
                                constant float&     lnMinP       [[buffer(6)]],
                                uint t           [[thread_position_in_threadgroup]],
                                uint row         [[threadgroup_position_in_grid]],
                                uint bdim        [[threads_per_threadgroup]],
                                uint lane        [[thread_index_in_simdgroup]],
                                uint simdgroup   [[simdgroup_index_in_threadgroup]],
                                uint simd_size   [[threads_per_simdgroup]],
                                uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float stage_s[COARSE_FACTOR * BLOCK_DIM];   // each SIMD-group's slice of the chunk
    threadgroup float simdSums_s[BLOCK_DIM];                // the sum, then each chunk's group totals
    threadgroup int simdTopP_s[BLOCK_DIM];
    threadgroup int simdMinP_s[BLOCK_DIM];

    device const float* rowLogits = sortedLogits + ulong(row) * K;
    device float* rowProbs = sortedProbs + ulong(row) * K;

    const uint chunkSize = COARSE_FACTOR * simd_size * simd_groups;
    const uint groupSlice = simdgroup * COARSE_FACTOR * simd_size;
    threadgroup float* myStage = stage_s + groupSlice;

    const float l0 = rowLogits[0];                   // the row is sorted: its maximum
    const float threshold = minP > 0.0f ? l0 + lnMinP : -FLT_MAX;
    const bool keepAll = topP >= 1.0f;

    // Pass 1: the softmax denominator.
    float s = 0.0f;
    for (uint c = t; c < K; c += bdim) {
        s += exp(rowLogits[c] - l0);
    }
    s = simd_sum(s);
    if (lane == 0) {
        simdSums_s[simdgroup] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        float v = lane < simd_groups ? simdSums_s[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) {
            simdSums_s[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = simdSums_s[0];
    // Every thread has read the total; the first chunk's scan writes simdSums_s next.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / total;

    int keptTopP = 0;
    int keptMinP = 0;
    float carry = 0.0f;

    for (uint chunk = 0; chunk < K; chunk += chunkSize) {
        // (a) Coalesced: probabilities written and staged; min-p decided on the logit.
        for (uint c = 0; c < COARSE_FACTOR; ++c) {
            const uint j = chunk + groupSlice + c * simd_size + lane;
            float prob = 0.0f;
            if (j < K) {
                const float l = rowLogits[j];
                prob = exp(l - l0) * inv;
                rowProbs[j] = prob;
                keptMinP += l >= threshold ? 1 : 0;
            }
            myStage[c * simd_size + lane] = prob;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // (b) This lane's run: exclusive sums in registers.
        float run[COARSE_FACTOR];
        float threadSum = 0.0f;
        for (uint c = 0; c < COARSE_FACTOR; ++c) {
            run[c] = threadSum;
            threadSum += myStage[lane * COARSE_FACTOR + c];
        }

        // (c) Block scan of the run totals.
        const float exclusive = simd_prefix_exclusive_sum(threadSum);
        if (lane == simd_size - 1) {
            simdSums_s[simdgroup] = exclusive + threadSum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup == 0) {
            float groupSum = lane < simd_groups ? simdSums_s[lane] : 0.0f;
            groupSum = simd_prefix_inclusive_sum(groupSum);
            if (lane < simd_groups) {
                simdSums_s[lane] = groupSum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // (d) Mass before each token, earlier chunks included; the top-p decision.
        const float before = carry + (simdgroup > 0 ? simdSums_s[simdgroup - 1] : 0.0f) + exclusive;
        carry += simdSums_s[simd_groups - 1];
        const uint first = chunk + groupSlice + lane * COARSE_FACTOR;
        for (uint c = 0; c < COARSE_FACTOR; ++c) {
            const uint j = first + c;
            if (j < K) {
                const float mass = before + run[c];
                keptTopP += (j == 0 || keepAll || mass < topP) ? 1 : 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // simdSums_s and stage_s are reused next chunk
    }

    // n = max(1, min(top-p count, min-p count)).
    keptTopP = simd_sum(keptTopP);
    keptMinP = simd_sum(keptMinP);
    if (lane == 0) {
        simdTopP_s[simdgroup] = keptTopP;
        simdMinP_s[simdgroup] = keptMinP;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        int a = lane < simd_groups ? simdTopP_s[lane] : 0;
        int b = lane < simd_groups ? simdMinP_s[lane] : 0;
        a = simd_sum(a);
        b = simd_sum(b);
        if (lane == 0) {
            n[row] = max(1, min(a, b));
        }
    }
}
