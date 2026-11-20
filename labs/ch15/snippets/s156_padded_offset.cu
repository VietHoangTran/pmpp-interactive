// Section 15.6, pp. 362-363: the padding change exactly as the book describes it,
// applied to the tile loop of Fig. 15.3. A_s gets one padding column, and loadTile and
// mm get bK + 1 as A_s's leading dimension. One more line has to change and the book
// does not say so (notes/ch15.md, "Unclear in the PDF" item 4); it is left as printed
// here, because section 15.6's erratum hunt asks which line it is. Do not copy this
// loop: lab 15.4's solution has the corrected form.
// Shown by app/src/chapters/ch15/S156.tsx (constant PADDED); S156.test.ts ties the two.
// The helpers are declared only: this file is front-end-checked, not built or run.
constexpr unsigned int bM = 128, bN = 128, bK = 8, tM = 8, tN = 8;

__device__ void loadTile(float* T, unsigned int lda, unsigned int maxRow, unsigned int maxCol,
                         float* T_s, unsigned int ldas, unsigned int height, unsigned int width);
__device__ void mm(unsigned int m, unsigned int n, unsigned int k, float* a, unsigned int lda,
                   float* b, unsigned int ldb, float c[][tN]);
__device__ void writeTile(float* c, unsigned int ldc, unsigned int maxRow, unsigned int maxCol,
                          float C_r[][tN], unsigned int m, unsigned int n);

__global__ void mm_tiled_padded(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
    unsigned int bRow = blockIdx.y*bM;
    unsigned int bCol = blockIdx.x*bN;
    unsigned int tRow = (threadIdx.x/(bN/tN))*tM;
    unsigned int tCol = (threadIdx.x%(bN/tN))*tN;
    float C_r[tM][tN] = {};
    for (unsigned int tile = 0; tile < (K + bK - 1)/bK; ++tile) {
        __shared__ float A_s[bM*(bK + 1)];
        __shared__ float B_s[bK*bN];
        loadTile(&A[bRow*K + tile*bK], K, M - bRow, K - tile*bK, &A_s[0], bK + 1, bM, bK);
        loadTile(&B[tile*bK*N + bCol], N, K - tile*bK, N - bCol, &B_s[0], bN, bK, bN);
        __syncthreads();
        mm(tM, tN, bK, &A_s[tRow*bK], bK + 1, &B_s[tCol], bN, C_r);
        __syncthreads();
    }
    unsigned int maxRow = (bRow + tRow < M) ? (M - (bRow + tRow)) : 0;
    unsigned int maxCol = (bCol + tCol < N) ? (N - (bCol + tCol)) : 0;
    writeTile(&C[(bRow + tRow)*N + bCol + tCol], N, maxRow, maxCol, C_r, tM, tN);
}
