// A causal 1D convolution: y[i] = w[0]*x[i-3] + w[1]*x[i-2] + w[2]*x[i-1] + w[3]*x[i],
// with x[j] = 0 for j < 0. Each output looks back K - 1 inputs, so the halo is one-sided.
// One thread per output; grid = ceil(n / BLOCK) blocks of BLOCK threads.
#define BLOCK 128
#define K 4

__constant__ float w[K];

__global__ void causalConv1d(const float* x, float* y, int n) {
    __shared__ float tile[BLOCK + K - 1];
    int t = threadIdx.x;
    int i = blockIdx.x * BLOCK + t;
    if (i >= n) return;                     // no output past the end

    tile[t + K - 1] = x[i];                 // this thread's own input
    if (t < K - 1) {                        // the halo: the K - 1 inputs before the block
        int j = i - (K - 1);
        tile[t] = (j >= 0) ? x[j] : 0.0f;   // before the start of x: a ghost zero
    }
    __syncthreads();

    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += w[k] * tile[t + k];
    y[i] = acc;
}
