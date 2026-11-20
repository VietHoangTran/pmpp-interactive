// Lab 6.1 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Coalescing on your device. Book: section 6.1 (pp. 124-132), Figs. 6.2 and 6.3,
// and the remark on stores (p. 130).
//
// Three kernels whose only difference is which address each thread touches.
// None is a listing in the book: they are small probes of the rule section 6.1
// states. Row-major float arrays, raw device pointers, explicit sizes.

// ---------- 1. strided read ----------
// Thread i copies in[i * stride] to out[i]. The write is contiguous at every
// stride; the read is contiguous only at stride 1. `in` holds n * stride floats.
__global__ void stridedReadKernel(const float* in, float* out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i * stride];
    }
}

void launchStridedRead(const float* in, float* out, int n, int stride) {
    const int block = 256;
    stridedReadKernel<<<(n + block - 1) / block, block>>>(in, out, n, stride);
}

// ---------- 2 and 3. naive transposes ----------
// `in` is rows x cols, `out` is cols x rows, and out(c, r) = in(r, c):
// in(r, c) is at in[r * cols + c], out(c, r) is at out[c * rows + r].

// Rows read: x is the column of `in`, y its row. Neighbouring threadIdx.x read
// neighbouring floats of `in` (coalesced) and write floats `rows` apart in `out`.
__global__ void transposeReadRowsKernel(const float* in, float* out, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col];
    }
}

void launchTransposeReadRows(const float* in, float* out, int rows, int cols) {
    const int side = 16;
    dim3 dimBlock(side, side);
    dim3 dimGrid((cols + side - 1) / side,    // x covers the columns of `in`
                 (rows + side - 1) / side);   // y covers the rows of `in`
    transposeReadRowsKernel<<<dimGrid, dimBlock>>>(in, out, rows, cols);
}

// Rows written: the opposite assignment. x is the column of `out` (a row of `in`),
// y the row of `out` (a column of `in`). Neighbouring threadIdx.x write
// neighbouring floats of `out` (coalesced) and read floats `cols` apart in `in`.
__global__ void transposeWriteRowsKernel(const float* in, float* out, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;   // row of `in` = column of `out`
    int col = blockIdx.y * blockDim.y + threadIdx.y;   // column of `in` = row of `out`
    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col];
    }
}

void launchTransposeWriteRows(const float* in, float* out, int rows, int cols) {
    const int side = 16;
    dim3 dimBlock(side, side);
    dim3 dimGrid((rows + side - 1) / side,    // x covers the columns of `out`
                 (cols + side - 1) / side);   // y covers the rows of `out`
    transposeWriteRowsKernel<<<dimGrid, dimBlock>>>(in, out, rows, cols);
}
