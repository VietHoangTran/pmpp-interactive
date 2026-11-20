// Runs ../s107_hybrid.cu (Fig. 10.14, corrected) on the course's CPU emulator for
// section 10.7. Compile both files with the same -DBLOCK_DIM=N. Under each of the three
// thread orders it prints one line per kernel, "<kernel> <order>: <result> <expected>":
//   printed     the snippet's kernel
//   noline11    the same kernel without the __syncthreads() after the loop (line 11);
//               each lane of warp 0 then reads only the slot it wrote itself
//   early2      a control: the loop stops one level sooner and lane t adds slots t and
//               t + warpSize, which other threads wrote; also without line 11
// app/src/chapters/ch10/S107.snippets.test.ts reads these lines. The two variants are
// copies made for this check, not lesson code.
#include <cstdio>

#ifndef BLOCK_DIM
#define BLOCK_DIM 1024
#endif

__global__ void WarpLevelSumReductionKernel(float* input, float* output);

#if defined(PMPP_EMU)
static const int kOrders = 3;
static const char* useOrder(int i) {
    using pmpp_emu::Schedule;
    pmpp_emu::set_schedule(i == 0 ? Schedule::Forward : i == 1 ? Schedule::Reverse : Schedule::Shuffle, 7);
    return i == 0 ? "in order" : i == 1 ? "reversed" : "shuffled";
}
#else
static const int kOrders = 1;
static const char* useOrder(int) { return "device"; }
#endif

__device__ __inline__ float warpSum(float v) {
    for (unsigned int s = warpSize / 2; s > 0; s /= 2) v += __shfl_down_sync(~0u, v, s);
    return v;
}

__global__ void noLine11(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (unsigned int stride = blockDim.x / 2; stride >= (unsigned int)warpSize; stride /= 2) {
        __syncthreads();
        if (t < stride) input_s[t] += input_s[t + stride];
    }
    if (t / warpSize == 0) {
        float partialSum = warpSum(input_s[t]);
        if (t == 0) *output = partialSum;
    }
}

__global__ void early2(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (unsigned int stride = blockDim.x / 2; stride > (unsigned int)warpSize; stride /= 2) {
        __syncthreads();
        if (t < stride) input_s[t] += input_s[t + stride];
    }
    if (t / warpSize == 0) {
        float partialSum = warpSum(input_s[t] + input_s[t + warpSize]);
        if (t == 0) *output = partialSum;
    }
}

int main() {
    const int n = 2 * BLOCK_DIM;
    static float in[2 * BLOCK_DIM];
    double expected = 0;
    for (int i = 0; i < n; ++i) {
        in[i] = (float)(((i * 7 + 3) % 9) + 1);  // small integers: every sum is exact
        expected += in[i];
    }
    float *in_d = nullptr, *out_d = nullptr;
    cudaMalloc((void**)&in_d, n * sizeof(float));
    cudaMalloc((void**)&out_d, sizeof(float));
    cudaMemcpy(in_d, in, n * sizeof(float), cudaMemcpyHostToDevice);
    const char* names[] = {"printed", "noline11", "early2"};
    std::printf("BLOCK_DIM %d\n", BLOCK_DIM);
    for (int k = 0; k < 3; ++k)
        for (int o = 0; o < kOrders; ++o) {
            const char* order = useOrder(o);
            float out = -1.0f;
            cudaMemcpy(out_d, &out, sizeof out, cudaMemcpyHostToDevice);
            if (k == 0) WarpLevelSumReductionKernel<<<1, BLOCK_DIM>>>(in_d, out_d);
            else if (k == 1) noLine11<<<1, BLOCK_DIM>>>(in_d, out_d);
            else early2<<<1, BLOCK_DIM>>>(in_d, out_d);
            cudaDeviceSynchronize();
            cudaMemcpy(&out, out_d, sizeof out, cudaMemcpyDeviceToHost);
            std::printf("%s %s: %g %g\n", names[k], order, out, expected);
        }
    cudaFree(in_d);
    cudaFree(out_d);
    std::printf("DONE\n");
    return 0;
}
