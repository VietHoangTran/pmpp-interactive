#define FILTER_RADIUS 1
#define IN_TILE_DIM 8
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ void convolution_tiled_2D_const_mem_kernel(
        const float* N, float* P, int width, int height) {
    int tx = threadIdx.x, ty = threadIdx.y;   // signed from here on
    int col = blockIdx.x*OUT_TILE_DIM + tx - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + ty - FILTER_RADIUS;
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[ty][tx] = N[row*width + col];
    } else {
        N_s[ty][tx] = 0.0f;                   // ghost cell
    }
    __syncthreads();
    int tileCol = tx - FILTER_RADIUS;
    int tileRow = ty - FILTER_RADIUS;
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM &&
            tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                    Pvalue += F_c[fRow][fCol]
                              * N_s[tileRow+fRow][tileCol+fCol];
                }
            }
            P[row*width + col] = Pvalue;
        }
    }
}

// Not in the book, which shows no host code: one block per output tile.
void launchHaloToy(const float* N_d, float* P_d, int width, int height) {
    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM);   // the block is the input tile
    dim3 dimGrid((width + OUT_TILE_DIM - 1)/OUT_TILE_DIM,
                 (height + OUT_TILE_DIM - 1)/OUT_TILE_DIM);
    convolution_tiled_2D_const_mem_kernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);
}
