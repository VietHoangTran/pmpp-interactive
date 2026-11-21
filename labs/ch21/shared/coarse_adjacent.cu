// Lab 21.1, provided: the benchmark's baseline for your coarsened kernel. Do not edit.
// (CUDA; also built for HIP and for the CPU emulators.)
//
// Book Fig. 21.8 (p. 520), corrected: one thread per COARSEN = 4 ADJACENT points of a row,
// i, i+1, i+2, i+3. For any one of the four write statements, neighbouring threads of a
// warp write addresses four floats apart: the uncoalesced pattern of p. 521.
//
// Where this differs from the page:
//   - Line 05 as printed, i = blockIdx.x * blockDim.x*COARSEN_FACTOR + threadIdx.x, adds an
//     UNSCALED threadIdx.x. With adjacent ownership, thread t would then own points
//     base+t .. base+t+3: most points of a block's span written by four threads with a
//     plain +=, and its last 3 * blockDim.x - 3 points written by none (ch21-facts E2).
//     The text (p. 521) says neighbouring threads start four points apart, so threadIdx.x
//     is scaled by the factor here.
//   - A bound check per point, as in your kernels, and an int slice k (see work/kernel.cu).
//   - The constant array is atoms_adj, not atoms, so that it cannot clash with yours.
//     16 KB here plus 16 KB in your file (dcs.h).
#include "../lab1_dcs/dcs.h"

#include <cstdio>

#if COARSEN != 4
#error "dcsCoarseAdjacentKernel is written for four points per thread, as Fig. 21.8 is"
#endif

__constant__ float atoms_adj[CHUNK_MAX * 4];

__global__ void dcsCoarseAdjacentKernel(float* energygrid, int nx, int ny, float gridspacing, int k, int numatoms) {
    int i = blockIdx.x * blockDim.x * COARSEN + threadIdx.x * COARSEN;   // line 05, corrected
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny) return;
    int atomarrdim = numatoms * 4;
    float z = gridspacing * (float)k;
    float y = gridspacing * (float)j;
    float x = gridspacing * (float)i;
    float energy0 = 0.0f;
    float energy1 = 0.0f;
    float energy2 = 0.0f;
    float energy3 = 0.0f;
    for (int n = 0; n < atomarrdim; n += 4) {
        float dx0 = x - atoms_adj[n];
        float dx1 = dx0 + gridspacing;
        float dx2 = dx0 + 2 * gridspacing;
        float dx3 = dx0 + 3 * gridspacing;
        float dy = y - atoms_adj[n + 1];
        float dz = z - atoms_adj[n + 2];
        float dysqdzsq = dy * dy + dz * dz;
        float charge = atoms_adj[n + 3];
        energy0 += charge / sqrtf(dx0 * dx0 + dysqdzsq);
        energy1 += charge / sqrtf(dx1 * dx1 + dysqdzsq);
        energy2 += charge / sqrtf(dx2 * dx2 + dysqdzsq);
        energy3 += charge / sqrtf(dx3 * dx3 + dysqdzsq);
    }
    int row = nx * ny * k + nx * j;
    if (i < nx) energygrid[row + i] += energy0;
    if (i + 1 < nx) energygrid[row + i + 1] += energy1;
    if (i + 2 < nx) energygrid[row + i + 2] += energy2;
    if (i + 3 < nx) energygrid[row + i + 3] += energy3;
}

// The same grid as launchDcsCoarsened, so that the two differ only in which points a
// thread owns.
void launchDcsCoarseAdjacent(float* energygrid, int nx, int ny, int nz, float gridspacing, int k,
                             const float* hostAtoms, int numatoms, int chunk) {
    if (nx <= 0 || ny <= 0 || numatoms <= 0) return;
    if (k < 0 || k >= nz || chunk < 1 || chunk > CHUNK_MAX) {
        std::fprintf(stderr, "launchDcsCoarseAdjacent: need 0 <= k < nz and 1 <= chunk <= %d\n", CHUNK_MAX);
        return;
    }
    dim3 dimBlock(BLOCK_X, BLOCK_Y);
    dim3 dimGrid((nx + COARSEN * BLOCK_X - 1) / (COARSEN * BLOCK_X), (ny + BLOCK_Y - 1) / BLOCK_Y);
    for (int start = 0; start < numatoms; start += chunk) {
        int count = (numatoms - start < chunk) ? numatoms - start : chunk;
        cudaMemcpyToSymbol(atoms_adj, hostAtoms + 4 * start, count * 4 * sizeof(float));
        dcsCoarseAdjacentKernel<<<dimGrid, dimBlock>>>(energygrid, nx, ny, gridspacing, k, count);
    }
}
