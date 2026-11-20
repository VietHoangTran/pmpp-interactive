// Fig. 8.12 (p. 197), corrected: the register inNext where the page
// prints inNext_s, c3 with y-1 and c4 with y+1, and registers that
// start at 0 (the page leaves them unset). ty and tx stand for
// threadIdx.y and threadIdx.x; the stepper uses tiles of 4 and 6.
#define IN_TILE_DIM 6
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
__constant__ float c0, c1, c2, c3, c4, c5, c6;

__global__ void stencil_registers(const float* in, float* out,
                                  unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    int ty = threadIdx.y, tx = threadIdx.x;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inPrev = 0.0f, inCurr = 0.0f, inNext = 0.0f;
    bool jk = j >= 0 && j < N && k >= 0 && k < N;
    if (jk && iStart >= 1 && iStart - 1 < N)
        inPrev = in[(iStart - 1)*N*N + j*N + k];
    if (jk && iStart < N) {
        inCurr = in[iStart*N*N + j*N + k];
        inCurr_s[ty][tx] = inCurr;
    }
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (jk && i + 1 < N)
            inNext = in[(i + 1)*N*N + j*N + k];
        __syncthreads();   // barrier 1
        bool inGrid = i >= 1 && i < N - 1 && j >= 1 && j < N - 1
                   && k >= 1 && k < N - 1;
        bool inner = ty >= 1 && ty < IN_TILE_DIM - 1
                  && tx >= 1 && tx < IN_TILE_DIM - 1;
        if (inGrid && inner)
            out[i*N*N + j*N + k] = c0*inCurr
              + c1*inCurr_s[ty][tx - 1] + c2*inCurr_s[ty][tx + 1]
              + c3*inCurr_s[ty - 1][tx] + c4*inCurr_s[ty + 1][tx]
              + c5*inPrev + c6*inNext;
        __syncthreads();   // barrier 2
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[ty][tx] = inNext;
    }
}
