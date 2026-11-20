__global__ void foo_kernel(float* a, float* b, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        b[i] = 2.7f * a[i] - 4.3f;
    }
}
void foo(float* a_d, float* b_d) {
    unsigned int N = 200000;
    foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d, N);
}
