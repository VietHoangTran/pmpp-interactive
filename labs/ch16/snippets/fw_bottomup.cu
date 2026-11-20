// Fig. 16.4 (p. 380): the Floyd-Warshall kernel FW_bottomup and its host loop, corrected.
// Shown in section 16.3 of chapter 16. check.py runs it on the emulator against a CPU
// Floyd-Warshall; ./verify.sh parses it with the CUDA front-end check. Three changes from
// the page (notes/ch16.md, "Unclear in the PDF"; ch16-facts A3; ch04-facts A6, B3, C3;
// ch02-facts A5):
//  1. "No path" is the integer FW_INF = INT_MAX / 2. The page compares an int with
//     <cmath>'s INFINITY, a float that no int equals, so its early exits never fire.
//  2. No thread returns before the barrier. Out-of-range threads keep a flag and skip
//     the work; the return after the barrier is the same for the whole block.
//  3. The host loop does not call cudaDeviceSynchronize() after each launch: launches in
//     one stream already run in order, each seeing the previous one's writes.
#include <climits>

#define FW_INF (INT_MAX / 2)  // "no path"; FW_INF + FW_INF still fits in an int

__global__ void FW_bottomup(int k, int* dist, int n_vertices) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    bool inRange = col < n_vertices;   // no return before the barrier
    int row = blockIdx.y;
    int distIndex = n_vertices * row + col;

    __shared__ int dist_k_col;         // dist[row][k], the same for the whole block
    if (threadIdx.x == 0)
        dist_k_col = dist[n_vertices * row + k];
    __syncthreads();

    if (dist_k_col == FW_INF) return;  // no path from row to k: the block retires

    if (inRange) {
        int dist_k_row = dist[k * n_vertices + col];   // dist[k][col]
        if (dist_k_row != FW_INF) {
            int new_distance = dist_k_col + dist_k_row;
            if (new_distance < dist[distIndex])
                dist[distIndex] = new_distance;
        }
    }
}

void floydWarshall(int* dist_d, int n_vertices, int threads) {
    dim3 dimBlock(threads);
    dim3 dimGrid((n_vertices + threads - 1) / threads, n_vertices);
    for (int k = 0; k < n_vertices; k++)
        FW_bottomup<<<dimGrid, dimBlock>>>(k, dist_d, n_vertices);  // in order: one stream
}
