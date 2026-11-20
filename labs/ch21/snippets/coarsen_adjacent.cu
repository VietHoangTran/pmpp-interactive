// Chapter 21, section 21.2: Fig. 21.8 (book p. 520) corrected, as the lesson shows it.
// Differences from the page:
//   - line 05 scales the thread's index by COARSEN_FACTOR, so thread t owns the four
//     adjacent points 4t to 4t + 3 of its block's stretch, which is what the text on
//     p. 521 describes. As printed, neighbouring threads overlap (coarsen_printed.cu).
//   - a bound check on j, and one per point written, so the slice needs no padding.
//   - casts on the comparisons (dim3's fields are unsigned).
// The writes are uncoalesced: in each of the four write statements neighbouring lanes are
// four floats apart (p. 521). The lesson's copy is the part below the marker, tied to
// this file by app/src/chapters/ch21/snippets.test.ts; check.py runs it on the CPU
// emulator against a float64 sum.
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
    // corrected: the thread index is scaled, so thread t owns points 4t .. 4t + 3
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * COARSEN_FACTOR;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= (int)grid.y) return;
    int atomarrdim = numatoms * 4;
    int k = z / gridspacing;
    float y = gridspacing * (float)j;
    float x = gridspacing * (float)i;
    float energy0 = 0.0f, energy1 = 0.0f, energy2 = 0.0f, energy3 = 0.0f;
    for (int n = 0; n < atomarrdim; n += 4) {
        float dx0 = x - atoms[n];  // one load of the atom serves four points
        float dx1 = dx0 + gridspacing;
        float dx2 = dx0 + 2 * gridspacing;
        float dx3 = dx0 + 3 * gridspacing;
        float dy = y - atoms[n + 1];
        float dz = z - atoms[n + 2];
        float dysqdzsq = dy * dy + dz * dz;  // shared by the four points of the row
        float charge = atoms[n + 3];
        energy0 += charge / sqrtf(dx0 * dx0 + dysqdzsq);
        energy1 += charge / sqrtf(dx1 * dx1 + dysqdzsq);
        energy2 += charge / sqrtf(dx2 * dx2 + dysqdzsq);
        energy3 += charge / sqrtf(dx3 * dx3 + dysqdzsq);
    }
    float* row = energygrid + grid.x * grid.y * k + grid.x * j;
    // four write statements; in each, neighbouring lanes are 4 floats apart
    if (i < (int)grid.x) row[i] += energy0;
    if (i + 1 < (int)grid.x) row[i + 1] += energy1;
    if (i + 2 < (int)grid.x) row[i + 2] += energy2;
    if (i + 3 < (int)grid.x) row[i + 3] += energy3;
}
