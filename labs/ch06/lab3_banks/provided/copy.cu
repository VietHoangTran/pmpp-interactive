// Provided instrument for lab 6.3: a plain copy, the yardstick for a transpose.
// Each element is read once and written once, both coalesced, with no shared memory and
// no turn. A transpose moves the same bytes, so one that runs about as fast as this copy
// has little left to gain from its shared-memory layout. Do not edit.

__global__ void copyKernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

void launchCopy(const float* in, float* out, int n) {
    copyKernel<<<(n + 255) / 256, 256>>>(in, out, n);
}
