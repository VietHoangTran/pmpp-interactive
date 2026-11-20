// The register-tiled march of Fig. 8.12 with its z window written as a small array
// indexed at run time: w[z % 3] holds plane z, so no values are copied between
// registers. Same arithmetic as the book's kernel. But the indices change with i, and
// NVIDIA documents an array the compiler cannot see is indexed with constants as a
// candidate for local memory, which is device memory (notes/today/ch08-facts.md, A5).
// Ours, not the book's. It parses with the CUDA front end; it is never built or run.
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
__constant__ float c0, c1, c2, c3, c4, c5, c6;

__global__ void stencil_indexed_window(const float* in, float* out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    int ty = threadIdx.y, tx = threadIdx.x;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float w[3] = {0.0f, 0.0f, 0.0f};                       // w[z % 3] holds plane z
    bool jk = j >= 0 && j < N && k >= 0 && k < N;
    if (jk && iStart >= 1 && iStart - 1 < N) w[(iStart + 2) % 3] = in[(iStart - 1)*N*N + j*N + k];
    if (jk && iStart < N) { w[iStart % 3] = in[iStart*N*N + j*N + k]; inCurr_s[ty][tx] = w[iStart % 3]; }
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (jk && i + 1 < N) w[(i + 1) % 3] = in[(i + 1)*N*N + j*N + k];   // overwrites plane i - 2
        __syncthreads();
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1 &&
            ty >= 1 && ty < IN_TILE_DIM - 1 && tx >= 1 && tx < IN_TILE_DIM - 1)
            out[i*N*N + j*N + k] = c0*w[i % 3]
                + c1*inCurr_s[ty][tx - 1] + c2*inCurr_s[ty][tx + 1]
                + c3*inCurr_s[ty - 1][tx] + c4*inCurr_s[ty + 1][tx]
                + c5*w[(i + 2) % 3] + c6*w[(i + 1) % 3];
        __syncthreads();
        inCurr_s[ty][tx] = w[(i + 1) % 3];
    }
}
