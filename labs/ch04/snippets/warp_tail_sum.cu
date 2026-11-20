// Sum of each block's 256 values. The last six steps are written for one warp, with no barrier.
#define BLOCK 256

__global__ void blockSum(const float* in, float* out) {
    __shared__ float s[BLOCK];
    unsigned int t = threadIdx.x;
    s[t] = in[blockIdx.x * BLOCK + t];
    __syncthreads();
    for (unsigned int stride = BLOCK / 2; stride > 32; stride /= 2) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t < 32) {
        volatile float* v = s;
        v[t] += v[t + 32];
        v[t] += v[t + 16];
        v[t] += v[t + 8];
        v[t] += v[t + 4];
        v[t] += v[t + 2];
        v[t] += v[t + 1];
    }
    if (t == 0) out[blockIdx.x] = s[0];
}
