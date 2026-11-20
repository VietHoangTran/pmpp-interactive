// Book exercise 2 (p. 64): a = B c, with B an N x N row-major matrix of floats and one
// thread per element of a. Section 3.4's question offers the bodies of these four
// kernels. Only matvecOption2 is right; the other three are wrong on purpose (the
// question says why), so never base a lab on them.
__global__ void matvecOption1(float* a, const float* B, const float* c, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float s = 0;
    for (int j = 0; j < N; ++j) s += B[i*N + j]*c[j];
    a[i] = s;
}

__global__ void matvecOption2(float* a, const float* B, const float* c, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) {
        float s = 0;
        for (int j = 0; j < N; ++j) s += B[i*N + j]*c[j];
        a[i] = s;
    }
}

__global__ void matvecOption3(float* a, const float* B, const float* c, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) {
        float s = 0;
        for (int j = 0; j < N; ++j) s += B[j*N + i]*c[j];
        a[i] = s;
    }
}

__global__ void matvecOption4(float* a, const float* B, const float* c, int N) {
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N && j < N) a[i] += B[i*N + j]*c[j];
}
