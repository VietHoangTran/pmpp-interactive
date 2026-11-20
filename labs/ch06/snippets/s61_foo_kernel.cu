__global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float a_s[256];
    __shared__ float bc_s[4*256];
    a_s[threadIdx.x] = a[i];
    for(unsigned int j = 0; j < 4; ++j) {
        bc_s[j*256 + threadIdx.x] = b[j*blockDim.x*gridDim.x + i] + c[i*4 + j];
    }
    __syncthreads();
    d[i + 8] = a_s[threadIdx.x];
    e[i*8] = bc_s[threadIdx.x*4];
}
