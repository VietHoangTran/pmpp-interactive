// Chapter 21, section 21.2: the gather kernel of Fig. 21.6 (book p. 518), as the lesson
// shows it. Differences from the page:
//   - a bound check on i and j. The printed kernel has none and relies on a slice padded
//     to a whole number of blocks (the "padding waste" of Fig. 21.9).
//   - casts on the comparisons (dim3's fields are unsigned).
// The lesson's copy is the part below the marker, tied to this file by
// app/src/chapters/ch21/snippets.test.ts; check.py runs it on the CPU emulator against a
// float64 sum.
#ifndef CHUNK_SIZE
#define CHUNK_SIZE 256
#endif
// ---- shown in the lesson ----
__constant__ float atoms[CHUNK_SIZE * 4];  // x, y, z, charge of each atom in the chunk

__global__ void cenergy(float* energygrid, dim3 grid, float gridspacing, float z, int numatoms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= (int)grid.x || j >= (int)grid.y) return;  // added: Fig. 21.6 relies on padding
    int atomarrdim = numatoms * 4;
    int k = z / gridspacing;
    float y = gridspacing * (float)j;  // recomputed by every thread
    float x = gridspacing * (float)i;
    float energy = 0.0f;
    for (int n = 0; n < atomarrdim; n += 4) {  // every lane reads the same atom
        float dx = x - atoms[n];
        float dy = y - atoms[n + 1];
        float dz = z - atoms[n + 2];
        energy += atoms[n + 3] / sqrtf(dx * dx + dy * dy + dz * dz);
    }
    // one plain write, to a point no other thread touches
    energygrid[grid.x * grid.y * k + grid.x * j + i] += energy;
}
