// Lab 4.2, provided, used by the driver only: asks the device how it cut the
// threadgroups of a dispatch into SIMD-groups.
//
// On CUDA the cut is documented: threads in linear order, consecutive runs of
// warpSize, thread 0 in warp 0. The Metal Shading Language specification says
// instead that "threads are divided into SIMD-groups in an implementation-
// defined fashion" and that a thread "can query its SIMD lane ID and which
// SIMD-group it is a member of" (section 5.2.3.6). So the driver does not
// compute a thread's SIMD-group from its index. It dispatches this kernel in the
// shape it will use for yours and reads the answer.
//
// What this does and does not establish. The specification promises that the
// mapping is the same "across dispatches of a given kernel with the same launch
// parameters, and from one threadgroup to another within the dispatch" (section
// 4.4.1). This probe is a different kernel from yours, so that promise does not
// reach from one to the other. The patterns are built from the best evidence
// available: what this device did with the same threadgroup size.
#include <metal_stdlib>
using namespace metal;

kernel void simd_probe(device uint* group [[buffer(0)]],    // per thread: SIMD-group, numbered across the grid
                       device uint* lane  [[buffer(1)]],    // per thread: index within its SIMD-group
                       device uint* info  [[buffer(2)]],    // [0] threads per SIMD-group, [1] SIMD-groups per threadgroup
                       uint i   [[thread_position_in_grid]],
                       uint tg  [[threadgroup_position_in_grid]],
                       uint sg  [[simdgroup_index_in_threadgroup]],
                       uint l   [[thread_index_in_simdgroup]],
                       uint w   [[threads_per_simdgroup]],
                       uint per [[simdgroups_per_threadgroup]]) {
    group[i] = tg * per + sg;
    lane[i] = l;
    if (i == 0) {
        info[0] = w;
        info[1] = per;
    }
}
