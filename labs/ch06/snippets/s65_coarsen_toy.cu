// Section 6.5, pp. 141-142: the book's thread-coarsening toy, made compilable.
// foo stands for one unit of work (the book leaves it abstract); here it doubles
// one element. The bounds checks are ours: the book's toy has none.
__device__ void foo(float* data, unsigned int i) { data[i] = 2.0f*data[i]; }

// Before: one unit of work per thread. Launch one thread per element.
__global__ void fine(float* data, unsigned int n) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) foo(data, i);
}

// After: coarsening factor 4. Launch a quarter as many threads.
#define COARSEN 4
__global__ void coarse(float* data, unsigned int n) {
    unsigned int iStart = COARSEN*(blockIdx.x*blockDim.x + threadIdx.x);
    for (unsigned int c = 0; c < COARSEN; ++c) {   // the coarsening loop
        unsigned int i = iStart + c;
        if (i < n) foo(data, i);
    }
}

// Not in the book: the same factor with a strided mapping. Thread t handles
// t, t + T, t + 2T and t + 3T, where T is the number of threads in the grid.
__global__ void coarseStrided(float* data, unsigned int n) {
    unsigned int T = gridDim.x*blockDim.x;
    unsigned int t = blockIdx.x*blockDim.x + threadIdx.x;
    for (unsigned int c = 0; c < COARSEN; ++c) {
        unsigned int i = t + c*T;
        if (i < n) foo(data, i);
    }
}
