// Lab 2.3 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Project milestone M0: the elementwise operations a transformer block needs.
//
// The book shows none of these kernels. Each one is the vector addition kernel
// of Fig. 2.10 (p. 35) with a different body: compute the global index, guard
// it, do one element. scaleShift is the affine map of the book's Exercise 9
// (Fig. 2.15, p. 42), written as the __host__ __device__ function that p. 36
// and Exercise 10 describe. Using them for bias, scale and residual is ours.
//
// Conventions that later milestones rely on:
//   - tensors are row-major floats, batch (rows) first: element (r, j) of a
//     rows x d tensor is at index r * d + j;
//   - kernels take raw device pointers and explicit sizes;
//   - each kernel has a launch<Name> host function beside it. It computes the
//     grid and launches. It does not wait and does not read the error state:
//     the caller does both, once, after as many launches as it likes.
//
// Where this file differs from the page:
//   - block counts use the integer ceiling of Fig. 2.15, not the ceil(n/256.0)
//     of Fig. 2.13;
//   - blockTail checks every runtime call with the macro of the sidebar on
//     p. 32 (the page declares `err` and then tests `error`), asks whether the
//     launches were accepted, and waits before it copies back. Fig. 2.13 does
//     none of the three.
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
    return alpha * x + beta;
}

// Provided. A host function that calls the host compilation of scaleShift, so
// that the driver can compare it with what the kernel computed.
float scaleShiftOnHost(float x, float alpha, float beta) {
    return scaleShift(x, alpha, beta);
}

// Y[r][j] = X[r][j] + bias[j]. One thread per element of the rows x d tensor.
__global__ void addBiasKernel(float* Y, const float* X, const float* bias, int rows, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * d) {
        // The grid is flat, the tensor is not. Row-major means i = r * d + j,
        // so the column is i % d (and the row would be i / d).
        Y[i] = X[i] + bias[i % d];
    }
}

// Y[i] = alpha * X[i] + beta. Each thread calls the device compilation of
// scaleShift and runs it itself: no new threads are launched.
__global__ void scaleShiftKernel(float* Y, const float* X, float alpha, float beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Y[i] = scaleShift(X[i], alpha, beta);
    }
}

// X[i] += R[i], in place.
__global__ void addResidualKernel(float* X, const float* R, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        X[i] = X[i] + R[i];
    }
}

void launchAddBias(float* Y_d, const float* X_d, const float* bias_d, int rows, int d) {
    int n = rows * d;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    addBiasKernel<<<numBlocks, threadsPerBlock>>>(Y_d, X_d, bias_d, rows, d);
}

void launchScaleShift(float* Y_d, const float* X_d, float alpha, float beta, int n) {
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    scaleShiftKernel<<<numBlocks, threadsPerBlock>>>(Y_d, X_d, alpha, beta, n);
}

void launchAddResidual(float* X_d, const float* R_d, int n) {
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    addResidualKernel<<<numBlocks, threadsPerBlock>>>(X_d, R_d, n);
}

// y = alpha * (x + bias) + beta + r, for host arrays, with one round trip.
int blockTail(const float* x_h, const float* bias_h, const float* r_h, float alpha, float beta,
              float* y_h, int rows, int d) {
    int n = rows * d;
    size_t size = (size_t)n * sizeof(float);         // bytes of one tensor
    size_t biasSize = (size_t)d * sizeof(float);     // bytes of one row
    float *x_d = nullptr, *bias_d = nullptr, *r_d = nullptr, *y_d = nullptr;

    // In: everything the three operations need, once.
    CHECK(cudaMalloc((void**)&x_d, size));
    CHECK(cudaMalloc((void**)&bias_d, biasSize));
    CHECK(cudaMalloc((void**)&r_d, size));
    CHECK(cudaMalloc((void**)&y_d, size));
    CHECK(cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(bias_d, bias_h, biasSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r_d, r_h, size, cudaMemcpyHostToDevice));

    // Three launches back to back. The intermediate results never leave the
    // device. A launch returns before its grid has run, but launches without a
    // stream go to the default stream, which runs its work in the order it was
    // issued (ch02-facts A5), so each kernel sees what the one before it wrote.
    launchAddBias(y_d, x_d, bias_d, rows, d);       // y = x + bias
    launchScaleShift(y_d, y_d, alpha, beta, n);     // y = alpha * y + beta (a thread reads and writes only its own element)
    launchAddResidual(y_d, r_d, n);                 // y = y + r

    // One check and one wait for all three launches. cudaGetLastError returns
    // the error state and resets it; nothing else is documented as resetting
    // it (ch02-facts A6). So a launch refused first should still be reported
    // here, after two more launches. The emulator behaves that way; on a CUDA
    // or HIP device it is unverified.
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Out: the one result, once.
    CHECK(cudaMemcpy(y_h, y_d, size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(x_d));
    CHECK(cudaFree(bias_d));
    CHECK(cudaFree(r_d));
    CHECK(cudaFree(y_d));
    return 0;
}
