// Fig. 8.10 (p. 194), corrected: c3 goes with y-1 and c4 with y+1,
// as in Figs. 8.6 and 8.8. ty and tx stand for threadIdx.y and
// threadIdx.x; the stepper uses tiles of 4 and 6.
#define IN_TILE_DIM 6
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
__constant__ float c0, c1, c2, c3, c4, c5, c6;

__global__ void stencil_planes(const float* in, float* out,
                               unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    int ty = threadIdx.y, tx = threadIdx.x;
    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
    bool jk = j >= 0 && j < N && k >= 0 && k < N;
    if (jk && iStart >= 1 && iStart - 1 < N)
        inPrev_s[ty][tx] = in[(iStart - 1)*N*N + j*N + k];
    if (jk && iStart < N)
        inCurr_s[ty][tx] = in[iStart*N*N + j*N + k];
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (jk && i + 1 < N)
            inNext_s[ty][tx] = in[(i + 1)*N*N + j*N + k];
        __syncthreads();   // barrier 1
        bool inGrid = i >= 1 && i < N - 1 && j >= 1 && j < N - 1
                   && k >= 1 && k < N - 1;
        bool inner = ty >= 1 && ty < IN_TILE_DIM - 1
                  && tx >= 1 && tx < IN_TILE_DIM - 1;
        if (inGrid && inner)
            out[i*N*N + j*N + k] = c0*inCurr_s[ty][tx]
              + c1*inCurr_s[ty][tx - 1] + c2*inCurr_s[ty][tx + 1]
              + c3*inCurr_s[ty - 1][tx] + c4*inCurr_s[ty + 1][tx]
              + c5*inPrev_s[ty][tx] + c6*inNext_s[ty][tx];
        __syncthreads();   // barrier 2
        inPrev_s[ty][tx] = inCurr_s[ty][tx];
        inCurr_s[ty][tx] = inNext_s[ty][tx];
    }
}
