#define TAPS 4

__constant__ float taps[TAPS];      // filter weights, filled in by the host before the launch

__global__ void smooth(const float* in, const float* gain, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float window[TAPS];
    __shared__ float blockGain;
    __shared__ float stage[256];

    stage[threadIdx.x] = (i < n) ? in[i] : 0.0f;
    if (threadIdx.x == 0) blockGain = gain[blockIdx.x];
    __syncthreads();

    for (int t = 0; t < TAPS; ++t) {
        unsigned j = threadIdx.x + t;
        window[t] = (j < blockDim.x) ? stage[j] : 0.0f;
    }
    float acc = 0.0f;
    for (int t = 0; t < TAPS; ++t) {
        acc += taps[t] * window[t];
    }
    if (i < n) out[i] = blockGain * acc;
}

void launchSmooth(const float* in, const float* gain, float* out, int n) {
    smooth<<<48, 256>>>(in, gain, out, n);      // 48 blocks of 256 threads
}
