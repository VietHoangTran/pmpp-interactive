// Provided for lab 11.6 (do not edit): the benchmark baseline, one thread per row.
//
// Thread r walks all of row r on its own with Fig. 11.1's sequential loop, made exclusive
// (p. 254): it writes the mass before each token, decides keep, counts, and only then adds
// the token's probability. It is correct and as simple as a top-p mask can be, and it is
// what the lab's block-per-row kernel is measured against. Two things make it slow: a batch
// of R rows uses R threads, so at batch 1 one thread does the whole row; and neighbouring
// threads read addresses `cols` floats apart, so no access is coalesced (chapter 5).
//
// Contract as in the lab: probs is [rows x cols] float32, row-major, each row sorted in
// descending order. keep[r][i] = 1 exactly when i == 0, or p >= 1, or massBefore[r][i] < p.

#define NAIVE_BLOCK 64

__global__ void TopPMaskNaiveKernel(const float* probs, float* massBefore, unsigned char* keep, int* kept,
                                    int rows, int cols, float p) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* in = probs + (size_t)r * cols;
    float* mass = massBefore + (size_t)r * cols;
    unsigned char* k = keep + (size_t)r * cols;
    const bool keepAll = p >= 1.0f;
    float before = 0.0f;
    int count = 0;
    for (int i = 0; i < cols; ++i) {
        mass[i] = before;
        int keepIt = (i == 0 || keepAll || before < p) ? 1 : 0;
        k[i] = (unsigned char)keepIt;
        count += keepIt;
        before += in[i];
    }
    kept[r] = count;
}

void launchTopPMaskNaive(const float* probs, float* massBefore, unsigned char* keep, int* kept, int rows, int cols,
                         float p) {
    if (rows <= 0 || cols <= 0) return;
    TopPMaskNaiveKernel<<<(rows + NAIVE_BLOCK - 1) / NAIVE_BLOCK, NAIVE_BLOCK>>>(probs, massBefore, keep, kept, rows,
                                                                                 cols, p);
}
