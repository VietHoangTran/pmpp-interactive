// Section 6.6, pp. 145-146: where does x live? Five versions of one computation.
// foo is one unit of work. In the coarsened versions each thread handles COARSEN
// elements and keeps them in a local array x. Section 6.6's classification shows
// each kernel; app/src/sim/unroll.ts holds the copies and a unit test ties them here.
#define COARSEN 4

__device__ float foo(float v) { return v*v + 1.0f; }

__global__ void scalar(const float* in, float* out) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    float x = in[i];
    out[i] = foo(x);
}

// count must not exceed COARSEN.
__global__ void runtimeBound(const float* in, float* out, unsigned int count) {
    unsigned int iStart = COARSEN*(blockIdx.x*blockDim.x + threadIdx.x);
    float x[COARSEN];
    for (unsigned int c = 0; c < count; ++c) x[c] = in[iStart + c];
    for (unsigned int c = 0; c < count; ++c) out[iStart + c] = foo(x[c]);
}

__global__ void unrolledByHand(const float* in, float* out) {
    unsigned int iStart = COARSEN*(blockIdx.x*blockDim.x + threadIdx.x);
    float x[COARSEN];
    x[0] = in[iStart];     x[1] = in[iStart + 1];
    x[2] = in[iStart + 2]; x[3] = in[iStart + 3];
    out[iStart] = foo(x[0]);         out[iStart + 1] = foo(x[1]);
    out[iStart + 2] = foo(x[2]);     out[iStart + 3] = foo(x[3]);
}

__global__ void notUnrolled(const float* in, float* out) {
    unsigned int iStart = COARSEN*(blockIdx.x*blockDim.x + threadIdx.x);
    float x[COARSEN];
#pragma unroll 1
    for (unsigned int c = 0; c < COARSEN; ++c) x[c] = in[iStart + c];
#pragma unroll 1
    for (unsigned int c = 0; c < COARSEN; ++c) out[iStart + c] = foo(x[c]);
}

// pick[t] must be between 0 and COARSEN - 1.
__global__ void runtimeIndex(const float* in, float* out, const int* pick) {
    unsigned int t = blockIdx.x*blockDim.x + threadIdx.x;
    float x[COARSEN];
#pragma unroll
    for (unsigned int c = 0; c < COARSEN; ++c) x[c] = in[COARSEN*t + c];
    out[t] = foo(x[pick[t]]);       // which element: known only at run time
}
