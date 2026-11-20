// Lab 6.1 starter (CUDA; also built for HIP and for the CPU emulator).
// Coalescing on your device. Book: section 6.1, pp. 124-132.
//
// Three kernels whose only difference is which address each thread touches.
// None is a listing in the book: they are small probes of the rule section 6.1
// states. Row-major float arrays, raw device pointers, explicit sizes. A thread
// with no element to handle must do nothing.

// ---------- 1. strided read ----------
// Thread i copies in[i * stride] to out[i], for every i < n. `in` holds exactly
// n * stride floats, so on the emulator a read past it is reported with its line.
__global__ void stridedReadKernel(const float* in, float* out, int n, int stride) {
    // TODO: one global index per thread, a guard, one read and one write.
}

void launchStridedRead(const float* in, float* out, int n, int stride) {
    // TODO: a 1D grid of 1D blocks with at least n threads (round up).
}

// ---------- 2 and 3. naive transposes ----------
// `in` is rows x cols, `out` is cols x rows, and out(c, r) = in(r, c).
// Both kernels use 16 x 16 blocks and a grid rounded up to cover the matrix.

// Rows read: the thread's x index is a column of `in`, its y index a row of `in`.
// Neighbouring threadIdx.x then read neighbouring addresses of `in`.
// Before you write it: where in `out` do those same threads write?
__global__ void transposeReadRowsKernel(const float* in, float* out, int rows, int cols) {
    // TODO
}

void launchTransposeReadRows(const float* in, float* out, int rows, int cols) {
    // TODO
}

// Rows written: the opposite assignment. The thread's x index is a column of
// `out`, so neighbouring threadIdx.x write neighbouring addresses of `out`.
// Which shape must the grid cover now?
__global__ void transposeWriteRowsKernel(const float* in, float* out, int rows, int cols) {
    // TODO
}

void launchTransposeWriteRows(const float* in, float* out, int rows, int cols) {
    // TODO
}
