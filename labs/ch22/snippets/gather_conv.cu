// Output-centric 1-D convolution (gather): thread o owns out[o],
// adds up the inputs it needs in a register, and writes once.
// No other thread writes out[o].
// out[o] = sum over t of filter[t] * in[o + t - r]
__global__ void convGather(const float* in, const float* filter,
                           float* out, int n, int r) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    float acc = 0.0f;
    for (int t = 0; t <= 2 * r; ++t) {
        int i = o + t - r;            // the input tap t reads
        if (i >= 0 && i < n) acc += filter[t] * in[i];
    }
    out[o] = acc;
}

void launchConvGather(const float* in, const float* filter,
                      float* out, int n, int r) {
    convGather<<<(n + 255) / 256, 256>>>(in, filter, out, n, r);
}
