// Lab 2.3 starter (CUDA; also built for HIP and for the CPU emulator).
// Project milestone M0: the elementwise operations a transformer block needs.
//
// The book shows none of these kernels. Each one is the vector addition kernel
// of Fig. 2.10 (p. 35) with a different body. scaleShift is the affine map of
// the book's Exercise 9 (Fig. 2.15, p. 42), written as the __host__ __device__
// function that p. 36 and Exercise 10 describe. Using them for bias, scale and
// residual is ours.
//
// Conventions that later milestones rely on:
//   - tensors are row-major floats, batch (rows) first: element (r, j) of a
//     rows x d tensor is at index r * d + j;
//   - kernels take raw device pointers and explicit sizes;
//   - each kernel has a launch<Name> host function beside it. It computes the
//     grid and launches. It does not wait and does not read the error state:
//     the caller does both, once, after as many launches as it likes.
//
// Most tested sizes are not a multiple of the block size: rows * d = 1, 257,
// 300 and 231 are all tested, and only 1024 is a multiple of 256.
#include <cstdio>

// Provided: the macro you wrote in lab 2.1. It reports and makes the enclosing
// function return 1. It frees nothing.
#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            printf("%s in %s at line %d\n", cudaGetErrorString(err),     \
                   __FILE__, __LINE__);                                  \
            return 1;                                                    \
        }                                                                \
    } while (0)

// Every launch in this file uses blocks of this many threads (the book's
// choice on p. 38).
const int threadsPerBlock = 256;

// One definition, two compilations: the compiler builds this function once for
// host callers and once for device callers (p. 36).
__host__ __device__ inline float scaleShift(float x, float alpha, float beta) {
    // TODO 1: alpha times x, plus beta.
    return 0.0f;
}

// Provided. A host function that calls the host compilation of scaleShift, so
// that the driver can compare it with what the kernel computed.
float scaleShiftOnHost(float x, float alpha, float beta) {
    return scaleShift(x, alpha, beta);
}

// Y[r][j] = X[r][j] + bias[j]. One thread per element of the rows x d tensor.
// X and Y hold rows * d floats, bias holds d.
__global__ void addBiasKernel(float* Y, const float* X, const float* bias, int rows, int d) {
    // TODO 2: the grid is flat and the tensor is not. Which element does this
    //         thread own, and which element of bias goes with it? Threads past
    //         the end of the tensor own nothing.
}

// Y[i] = alpha * X[i] + beta, for n elements.
__global__ void scaleShiftKernel(float* Y, const float* X, float alpha, float beta, int n) {
    // TODO 3: call scaleShift. Who executes it?
}

// X[i] += R[i], in place, for n elements.
__global__ void addResidualKernel(float* X, const float* R, int n) {
    // TODO 4
}

// TODO 5: the three launch functions. Each one works out how many blocks of
//         threadsPerBlock threads cover its tensor and launches its kernel.
//         Nothing else: no wait, no error check, no copies.
void launchAddBias(float* Y_d, const float* X_d, const float* bias_d, int rows, int d) {
}

void launchScaleShift(float* Y_d, const float* X_d, float alpha, float beta, int n) {
}

void launchAddResidual(float* X_d, const float* R_d, int n) {
}

// y = alpha * (x + bias) + beta + r, for host arrays, with one round trip.
// Returns 0 when everything worked and a non-zero value otherwise.
int blockTail(const float* x_h, const float* bias_h, const float* r_h, float alpha, float beta,
              float* y_h, int rows, int d) {
    // Every runtime call below goes through CHECK.

    // TODO 6: allocate what the three operations need on the device and copy
    //         the inputs in. Each host array crosses once. The bias is one row,
    //         not a whole tensor.

    // TODO 7: call your three launch functions, one after the other. Which
    //         device array does each one read, and which does it write? No
    //         intermediate result goes back to the host.

    // TODO 8: ask once whether the launches were accepted, and wait once.

    // TODO 9: copy the one result to y_h. Free the device arrays.

    return -1;   // replace with 0 once the function is complete
}
