// Input-centric 1-D convolution (scatter): thread i owns in[i]
// and adds its contribution to every output that uses it.
// Wrong on purpose: several threads add into the same out[o],
// and the += is a plain read, add and write, not an atomic.
// out[o] = sum over t of filter[t] * in[o + t - r]
__global__ void convScatterPlain(const float* in, const float* filter,
                                 float* out, int n, int r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    for (int t = 0; t <= 2 * r; ++t) {
        int o = i - t + r;            // out[o] reads in[i] with tap t
        if (o >= 0 && o < n) out[o] += filter[t] * v;
    }
}

void launchConvScatterPlain(const float* in, const float* filter,
                            float* out, int n, int r) {
    cudaMemset(out, 0, n * sizeof(float));  // the threads add into it
    convScatterPlain<<<(n + 255) / 256, 256>>>(in, filter, out, n, r);
}
