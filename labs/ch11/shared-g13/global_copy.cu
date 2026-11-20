// Provided for labs 11.4 and 11.5, not edited: a plain copy, the bandwidth baseline.
// Not in the book as code: p. 273 says that even a copy kernel does not reach 100% of
// the published bandwidth, so the labs measure your copy rate and report every scan as a
// share of it. A copy moves 8 bytes per element (one 4-byte read, one 4-byte write),
// the least any scan can move (p. 272).

__global__ void CopyKernel(const float* input, float* output, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = input[i];
    }
}

void launchCopy(const float* in, float* out, unsigned int n, unsigned int block) {
    unsigned int blocks = (n + block - 1) / block;
    CopyKernel<<<blocks, block>>>(in, out, n);
}
