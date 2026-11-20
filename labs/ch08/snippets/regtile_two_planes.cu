// Ours, not the book's: Fig. 8.12 (corrected) with its one shared plane
// doubled. Plane i is read from inCurr_s[b] and plane i + 1 is written
// to inCurr_s[1 - b], so the loop has one barrier where Fig. 8.12 has two.
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
__constant__ float c0, c1, c2, c3, c4, c5, c6;

__global__ void stencil_two_planes(const float* in, float* out,
                                   unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    int ty = threadIdx.y, tx = threadIdx.x;
    __shared__ float inCurr_s[2][IN_TILE_DIM][IN_TILE_DIM];
    float inPrev = 0.0f, inCurr = 0.0f, inNext = 0.0f;
    bool jk = j >= 0 && j < N && k >= 0 && k < N;
    if (jk && iStart >= 1 && iStart - 1 < N)
        inPrev = in[(iStart - 1)*N*N + j*N + k];
    if (jk && iStart < N)
        inCurr = in[iStart*N*N + j*N + k];
    inCurr_s[0][ty][tx] = inCurr;
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        int b = (i - iStart) % 2;   // the buffer that holds plane i
        if (jk && i + 1 < N)
            inNext = in[(i + 1)*N*N + j*N + k];
        __syncthreads();            // the only barrier
        bool inGrid = i >= 1 && i < N - 1 && j >= 1 && j < N - 1
                   && k >= 1 && k < N - 1;
        bool inner = ty >= 1 && ty < IN_TILE_DIM - 1
                  && tx >= 1 && tx < IN_TILE_DIM - 1;
        if (inGrid && inner)
            out[i*N*N + j*N + k] = c0*inCurr
              + c1*inCurr_s[b][ty][tx - 1] + c2*inCurr_s[b][ty][tx + 1]
              + c3*inCurr_s[b][ty - 1][tx] + c4*inCurr_s[b][ty + 1][tx]
              + c5*inPrev + c6*inNext;
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[1 - b][ty][tx] = inNext;
    }
}
