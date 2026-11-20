// One definition, two compiled functions. __host__ __device__ makes the compiler
// build scaleShift once for the CPU and once for the GPU, so the kernel and the
// CPU reference it is tested against run the same arithmetic.
__host__ __device__ inline float scaleShift(float x, float scale, float shift) {
    return scale * x + shift;
}

// Device side: each thread calls the device version, and runs it itself.
__global__ void scaleShiftKernel(const float* in, float* out, float scale, float shift, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = scaleShift(in[i], scale, shift);
    }
}

// Host side: the reference loop calls the host version of the same function.
void scaleShiftReference(const float* in, float* out, float scale, float shift, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = scaleShift(in[i], scale, shift);
    }
}
