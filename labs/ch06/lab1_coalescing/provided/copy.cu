// Provided reference for lab 6.1: a plain copy, out[i] = in[i]. Neighbouring
// threads read neighbouring addresses and write neighbouring addresses, so both
// sides are coalesced. It moves the same bytes per element as your kernels, and
// its bandwidth is the figure the lab compares them with. Not a book listing.

__global__ void copyKernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i];
    }
}

void launchCopy(const float* in, float* out, int n) {
    const int block = 256;
    copyKernel<<<(n + block - 1) / block, block>>>(in, out, n);
}
