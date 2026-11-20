// Lab 3.4 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Book exercises 1a and 1b (p. 64): the matrix multiplication of Fig. 3.11 with one
// thread per output row, and with one thread per output column.
// Square Width x Width matrices in row-major order, as in the exercise.
#define BLOCK_WIDTH 16    // side of the 2D blocks of the provided kernel
#define BLOCK_SIZE 256    // threads per block of the two 1D launches

// ---------- provided: one thread per element (book Fig. 3.11, p. 61) ----------
// The page's kernel, with the name changed (the book prints MatrixMulKernel in the
// listing and matrixMulKernel in the text) and const added to the inputs. The book
// prints no launch for it; launchElement is ours, consistent with Fig. 3.12.
__global__ void elementKernel(const float* M, const float* N, float* P, int Width) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row*Width + k]*N[k*Width + col];
        }
        P[row*Width + col] = Pvalue;
    }
}

void launchElement(float* M, float* N, float* P, int Width) {
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    dim3 dimGrid((Width + BLOCK_WIDTH - 1)/BLOCK_WIDTH, (Width + BLOCK_WIDTH - 1)/BLOCK_WIDTH, 1);
    elementKernel<<<dimGrid, dimBlock>>>(M, N, P, Width);
}

// ---------- exercise 1a: one thread per output row ----------
// The grid is 1D, so the thread's only coordinate is a row. The column, which the
// grid's x supplied in elementKernel, becomes a loop inside the thread.
__global__ void rowKernel(const float* M, const float* N, float* P, int Width) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < Width) {
        for (int col = 0; col < Width; ++col) {
            float Pvalue = 0;                          // a new sum for every element
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row*Width + k]*N[k*Width + col];
            }
            P[row*Width + col] = Pvalue;               // consecutive addresses as col advances
        }
    }
}

void launchRow(float* M, float* N, float* P, int Width) {
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid((Width + BLOCK_SIZE - 1)/BLOCK_SIZE, 1, 1);
    rowKernel<<<dimGrid, dimBlock>>>(M, N, P, Width);
}

// ---------- exercise 1b: one thread per output column ----------
__global__ void colKernel(const float* M, const float* N, float* P, int Width) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (col < Width) {
        for (int row = 0; row < Width; ++row) {
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row*Width + k]*N[k*Width + col];
            }
            P[row*Width + col] = Pvalue;               // Width apart as row advances
        }
    }
}

void launchCol(float* M, float* N, float* P, int Width) {
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid((Width + BLOCK_SIZE - 1)/BLOCK_SIZE, 1, 1);
    colKernel<<<dimGrid, dimBlock>>>(M, N, P, Width);
}
