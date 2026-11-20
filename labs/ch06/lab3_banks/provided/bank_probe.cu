// Provided instrument for lab 6.3: what does one warp-wide shared-memory access cost
// when neighbouring lanes are `stride` words apart? Book: section 6.4 (pp. 139-141) and
// Exercise 5 (p. 155), which uses the same array and index: a[1024], threadIdx.x * stride.
// Do not edit.
//
// The driver launches ONE block of exactly one warp: `lanes` is the warp width the
// device reports. Lane l works on word (l * stride) & 1023 of a 1024-float shared array.
// With a 32-wide warp and stride <= 33 the "& 1023" never changes anything (31 * 33 =
// 1023), so every lane has its own word. With a 64-wide wavefront at stride 32, lanes 32
// to 63 wrap round onto the words of lanes 0 to 31 (our arithmetic): read that row of a
// HIP measurement with care.
//
// Each of the `iters` steps is one read and one write of the lane's word. Two things
// stop the compiler from removing the accesses or keeping the word in a register:
// - a[k] holds k, and the next address is the value just read. The address is always
//   the same word, but the compiler cannot know that.
// - the write stores v + zero, where `zero` is 0 passed at run time: the compiler cannot
//   prove the write leaves the word unchanged.
// The sum of the values read goes to out[lane], so the reads are used. Lane l ends with
// (iters + 1) * ((l * stride) & 1023), which the driver's test checks at stride 1.
//
// What it measures is the time of a chain of dependent accesses by one warp, loop
// overhead included. The bank structure decides how much each access adds to it.

__global__ void bankProbeKernel(float* out, int stride, int iters, float zero) {
    __shared__ float a[1024];
    int lane = threadIdx.x;
    for (int k = lane; k < 1024; k += blockDim.x) a[k] = (float)k;   // a[k] holds its own index
    __syncthreads();

    int idx = (lane * stride) & 1023;
    float acc = 0.0f;
    for (int it = 0; it < iters; ++it) {
        float v = a[idx];       // the probed read
        a[idx] = v + zero;      // the probed write, same word
        acc += v;
        idx = (int)v;           // the next address comes from the value read
    }
    __syncthreads();
    out[lane] = acc + a[idx];
}

void launchBankProbe(float* out, int lanes, int stride, int iters, float zero) {
    bankProbeKernel<<<1, lanes>>>(out, stride, iters, zero);
}
