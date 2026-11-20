// Provided for labs 11.1 and 11.2, not edited: a plain copy, the bandwidth baseline.
// Not in the book as code: p. 273 says that even a copy kernel does not reach 100% of
// the published bandwidth, so the labs measure your copy rate and report every scan as a
// share of it. A scan has to move the same bytes: one read and one write per element.

__global__ void CopyKernel(const float* input, float* output, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = input[i];
    }
}

void launchCopy(const float* in, float* out, int n, int block) {
    unsigned int blocks = ((unsigned int)n + block - 1) / block;
    CopyKernel<<<blocks, block>>>(in, out, (unsigned int)n);
}
