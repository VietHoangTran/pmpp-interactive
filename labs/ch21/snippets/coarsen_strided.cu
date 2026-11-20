// Chapter 21, section 21.2: Fig. 21.10 (book p. 522), coarsening with coalescing-aware
// ownership, as the lesson shows it. Line 05 is the book's: a block covers
// COARSEN_FACTOR * blockDim.x points of a row and thread t owns t, t + blockDim.x,
// t + 2*blockDim.x, t + 3*blockDim.x of them, so in each write statement neighbouring
// lanes write neighbouring floats. Differences from the page:
//   - a bound check on j, and one per point written, so the slice needs no padding
//     (the book's kernel has none; Fig. 21.9 shows the padding instead).
//   - casts on the comparisons (dim3's fields are unsigned).
// The lesson's copy is the part below the marker, tied to this file by
// app/src/chapters/ch21/snippets.test.ts; check.py runs it on the CPU emulator against a
// float64 sum.
#ifndef CHUNK_SIZE
#define CHUNK_SIZE 256
#endif
// The course's CUDA front-end check declares no CUDA Math API functions, so it is given
// sqrtf's device declaration here. nvcc, hipcc and the emulator never see it.
#if defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float sqrtf(float x);
#endif
// ---- shown in the lesson ----
__constant__ float atoms[CHUNK_SIZE * 4];
#define COARSEN_FACTOR 4

__global__ void cenergy(float* energygrid, dim3 grid, float gridspacing, float z, int numatoms) {
    int i = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x;  // line 05, as printed
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= (int)grid.y) return;
    int atomarrdim = numatoms * 4;
    int k = z / gridspacing;
    float y = gridspacing * (float)j;
    float x = gridspacing * (float)i;
    float energy0 = 0.0f, energy1 = 0.0f, energy2 = 0.0f, energy3 = 0.0f;
    for (int n = 0; n < atomarrdim; n += 4) {
        float dx0 = x - atoms[n];
        // the thread's points are blockDim.x apart, and so are their distances
        float dx1 = dx0 + blockDim.x * gridspacing;
        float dx2 = dx0 + 2 * blockDim.x * gridspacing;
        float dx3 = dx0 + 3 * blockDim.x * gridspacing;
        float dy = y - atoms[n + 1];
        float dz = z - atoms[n + 2];
        float dysqdzsq = dy * dy + dz * dz;
        float charge = atoms[n + 3];
        energy0 += charge / sqrtf(dx0 * dx0 + dysqdzsq);
        energy1 += charge / sqrtf(dx1 * dx1 + dysqdzsq);
        energy2 += charge / sqrtf(dx2 * dx2 + dysqdzsq);
        energy3 += charge / sqrtf(dx3 * dx3 + dysqdzsq);
    }
    float* row = energygrid + grid.x * grid.y * k + grid.x * j;
    int B = blockDim.x;
    // four write statements; in each, neighbouring lanes write neighbouring floats
    if (i < (int)grid.x) row[i] += energy0;
    if (i + B < (int)grid.x) row[i + B] += energy1;
    if (i + 2 * B < (int)grid.x) row[i + 2 * B] += energy2;
    if (i + 3 * B < (int)grid.x) row[i + 3 * B] += energy3;
}
