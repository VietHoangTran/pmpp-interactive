// Section 6.7, pp. 146-147: the book's exchange loop, made compilable, once with one
// shared buffer and once with two. Each thread reads its right-hand neighbour's slot
// and writes that value plus 10 into its own slot, n times. Launch one block of
// BLOCK threads. The book does not show declarations, the computation or swap
// (notes item 15); they are ours.
#define BLOCK 256

// Exchanges the calling thread's own two pointers. Every thread must call it.
__device__ void swapPointers(float*& a, float*& b) {
    float* t = a;
    a = b;
    b = t;
}

__global__ void exchangeSingle(const float* in, float* out, unsigned int n) {
    __shared__ float buffer[BLOCK];
    unsigned int me = threadIdx.x;
    unsigned int other = (threadIdx.x + 1) % blockDim.x;
    buffer[me] = in[me];
    __syncthreads();
    for (unsigned int i = 0; i < n; ++i) {
        float v = buffer[other];
        __syncthreads();            // after the read
        buffer[me] = v + 10.0f;
        __syncthreads();            // after the write
    }
    out[me] = buffer[me];
}

__global__ void exchangeDouble(const float* in, float* out, unsigned int n) {
    __shared__ float bufferA[BLOCK];
    __shared__ float bufferB[BLOCK];
    float* inBuffer = bufferA;      // each thread holds its own copy of both pointers
    float* outBuffer = bufferB;
    unsigned int me = threadIdx.x;
    unsigned int other = (threadIdx.x + 1) % blockDim.x;
    inBuffer[me] = in[me];
    __syncthreads();
    for (unsigned int i = 0; i < n; ++i) {
        float v = inBuffer[other];
        outBuffer[me] = v + 10.0f;
        __syncthreads();            // after the write
        swapPointers(inBuffer, outBuffer);
    }
    out[me] = inBuffer[me];
}
