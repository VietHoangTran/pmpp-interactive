// Lab 3.3 starter (CUDA; also built for HIP and for the CPU emulator).
// Naive GEMM, one thread per element of C:
//   C[j x l] = A[j x k] * B[k x l], row-major float, any sizes.
// Book: section 3.4, Fig. 3.11 (p. 61). First half of project milestone M1.
//
// As a linear layer: j is the batch, k the inputs, l the outputs, and B holds the
// weight as (in, out). The argument order and the storage convention are those of
// lab 5.3's gemmKernel, so the tiled kernel can replace this one later.
//
// Fig. 3.11 is square only: one Width stands for three different sizes. Your job
// is to decide, for every place Width appears in the figure, which of j, k and l
// it has become. (k is a size here, so use another name for the loop index.)
#define BLOCK_WIDTH 16

__global__ void naiveGemmKernel(const float* A, const float* B, float* C, int j, int k, int l) {
    // TODO 1: which row and which column of C does this thread own?

    // TODO 2: the grid is made of whole blocks, so some threads own no element.
    //         They must not read or write anything.

    // TODO 3: the dot product of one row of A and one column of B.
    //         How many terms does it have? Where does that row of A start?
    //         How far apart in memory are two consecutive elements of a column of B?

    // TODO 4: store the result in C.
}

void launchNaiveGemm(float* A, float* B, float* C, int j, int k, int l) {
    // TODO 5: BLOCK_WIDTH x BLOCK_WIDTH blocks, and enough of them to cover C for
    //         any j and l. Which size does the grid's x extent cover, and which its y?
}
