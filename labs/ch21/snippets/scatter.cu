// Chapter 21, section 21.2: the scatter kernel of Fig. 21.5 (book p. 517), as the lesson
// shows it. Differences from the page:
//   - an atom count, numatoms, and a bound on it. The printed kernel has neither, so a
//     launch with more threads than the chunk has atoms reads past the chunk's data.
//   - the slice index k = z / gridspacing, as Fig. 21.6 writes it, in place of
//     (grid.x*grid.y*z) / gridspacing; the row offset is then grid.x*grid.y*k + grid.x*j.
//     z / gridspacing is exact for the 0.5 spacing check.py uses (ours: a production
//     kernel would take k as an integer argument).
//   - casts on the loop bounds (dim3's fields are unsigned).
// The legacy atomicAdd is kept as printed; NVIDIA now recommends cuda::atomic_ref
// (notes/today/infra-facts.md A1, A2). The lesson's copy is the part below the marker,
// tied to this file by app/src/chapters/ch21/snippets.test.ts; check.py runs it on the
// CPU emulator against a float64 sum.
#ifndef CHUNK_SIZE
#define CHUNK_SIZE 256
#endif
// The course's CUDA front-end check declares no CUDA Math API functions, so it is given
// sqrtf's device declaration here. nvcc, hipcc and the emulator never see it.
#if defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float sqrtf(float x);
#endif
// ---- shown in the lesson ----
__constant__ float atoms[CHUNK_SIZE * 4];  // x, y, z, charge of each atom in the chunk

__global__ void cenergy(float* energygrid, dim3 grid, float gridspacing, float z, int numatoms) {
    int n = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // this thread's atom
    if (n >= numatoms * 4) return;                        // added: Fig. 21.5 has no bound
    int k = z / gridspacing;                              // which slice
    float dz = z - atoms[n + 2];  // the same for every point of the slice
    float dz2 = dz * dz;
    float charge = atoms[n + 3];
    for (int j = 0; j < (int)grid.y; j++) {
        float y = gridspacing * (float)j;
        float dy = y - atoms[n + 1];  // the same for every point of a row
        float dy2 = dy * dy;
        int row = grid.x * grid.y * k + grid.x * j;
        for (int i = 0; i < (int)grid.x; i++) {
            float x = gridspacing * (float)i;
            float dx = x - atoms[n];
            // every thread adds to every point: the update must be atomic
            atomicAdd(&energygrid[row + i], charge / sqrtf(dx * dx + dy2 + dz2));
        }
    }
}
