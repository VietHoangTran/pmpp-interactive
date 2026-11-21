// Lab 21.1 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Direct Coulomb summation for one slice of the potential map, three ways.
// Book: section 21.2 (pp. 515-519), Figs. 21.5 and 21.6; sections 21.3 and 21.4
// (pp. 519-522), Fig. 21.10; Exercise 1 (p. 528) is the host code.
//
// Where this file differs from the pages (notes/ch21.md, "Unclear in the PDF"):
//   - The slice is an int k, and its z coordinate is k * gridspacing. Fig. 21.6 takes a
//     float z and computes k = z / gridspacing, which truncates: a z that is not exactly a
//     multiple of the spacing in float lands on the slice below. Figs. 21.3 and 21.4 index
//     the slice three different ways; this file uses one.
//   - The sizes are two ints, nx and ny, where the book passes a dim3 grid.
//   - Every kernel tests its bounds; the listings test none. Fig. 21.5 has no atom count at
//     all, so the threads past the end of a partial chunk would add atoms left in constant
//     memory by the previous chunk. Figs. 21.6 and 21.10 do not test i and j: Fig. 21.9
//     draws a grid padded to whole blocks (the text never says so), and this one is not.
//   - Fig. 21.10's line 05 is kept as printed. It is Fig. 21.8's line 05 that is wrong
//     (an unscaled threadIdx.x with adjacent ownership): see ../shared/coarse_adjacent.cu.
//
// Portability, not in the book:
//   - atomicAdd on float is current CUDA, which NVIDIA now calls a "legacy" atomic and
//     for which it recommends cuda::atomic_ref (infra-facts A1, A2; chapter 9 used it).
//   - On HIP, float atomicAdd compiles to a safe operation, possibly a compare-and-swap
//     loop, unless the build passes -munsafe-fp-atomics; the course's HIP build does not
//     (infra-facts B1, ch21-facts B1).
#include "../dcs.h"

#include <cstdio>

#if COARSEN != 4
#error "dcsCoarsenedKernel is written for four points per thread, as Fig. 21.10 is"
#endif

// One chunk of atoms, four floats each: x, y, z, charge (p. 517). Written by the host with
// cudaMemcpyToSymbol before each launch; read-only in the kernels. Every thread of a
// gather warp reads the same element at the same time: the case NVIDIA's constant cache
// serves as fast as a register (ch07-facts A3; ch21-facts A5).
__constant__ float atoms[CHUNK_MAX * 4];

// ---------------------------------------------------------------- Fig. 21.5: scatter
// One thread per atom of the chunk. Each thread walks the whole slice and adds its atom's
// contribution to every point, so every point is updated by every thread: atomics.
__global__ void dcsScatterKernel(float* energygrid, int nx, int ny, float gridspacing, int k, int numatoms) {
    int n = (blockIdx.x * blockDim.x + threadIdx.x) * 4;    // this thread's atom: atoms[n .. n+3]
    if (n >= numatoms * 4) return;                          // not in the book: past the end of the chunk
    float z = gridspacing * (float)k;
    float dz = z - atoms[n + 2];
    float dz2 = dz * dz;                                    // the same for the whole slice
    int grid_slice_offset = nx * ny * k;
    float charge = atoms[n + 3];
    for (int j = 0; j < ny; j++) {
        float y = gridspacing * (float)j;
        float dy = y - atoms[n + 1];
        float dy2 = dy * dy;                                // the same for the whole row
        int grid_row_offset = grid_slice_offset + nx * j;
        for (int i = 0; i < nx; i++) {
            float x = gridspacing * (float)i;
            float dx = x - atoms[n];
            atomicAdd(&energygrid[grid_row_offset + i], charge / sqrtf(dx * dx + dy2 + dz2));
        }
    }
}

// ---------------------------------------------------------------- Fig. 21.6: gather
// One thread per grid point. Each thread walks the chunk's atoms, sums in a register, and
// writes its own point once. No other thread touches that point: no atomics.
__global__ void dcsGatherKernel(float* energygrid, int nx, int ny, float gridspacing, int k, int numatoms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;                         // not in the book: the last blocks overhang
    int atomarrdim = numatoms * 4;
    float z = gridspacing * (float)k;                       // the book: int k = z / gridspacing;
    float y = gridspacing * (float)j;
    float x = gridspacing * (float)i;
    float energy = 0.0f;
    for (int n = 0; n < atomarrdim; n += 4) {
        float dx = x - atoms[n];
        float dy = y - atoms[n + 1];
        float dz = z - atoms[n + 2];
        energy += atoms[n + 3] / sqrtf(dx * dx + dy * dy + dz * dz);
    }
    energygrid[nx * ny * k + nx * j + i] += energy;         // += : one launch per chunk, each adds its part
}

// ------------------------------------------------- Fig. 21.10: coarsened, coalesced writes
// One thread per COARSEN points of a row. A block covers COARSEN * blockDim.x consecutive
// points, and thread t owns points base + t + m * blockDim.x for m = 0 .. 3, so for each of
// the four write statements the threads of a warp write consecutive addresses. Each atom
// is read once per thread and its dy*dy + dz*dz and charge serve all four points.
__global__ void dcsCoarsenedKernel(float* energygrid, int nx, int ny, float gridspacing, int k, int numatoms) {
    int i = blockIdx.x * blockDim.x * COARSEN + threadIdx.x;   // line 05, as printed
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny) return;                                       // not in the book
    int atomarrdim = numatoms * 4;
    float z = gridspacing * (float)k;
    float y = gridspacing * (float)j;
    float x = gridspacing * (float)i;
    float energy0 = 0.0f;
    float energy1 = 0.0f;
    float energy2 = 0.0f;
    float energy3 = 0.0f;
    for (int n = 0; n < atomarrdim; n += 4) {
        float dx0 = x - atoms[n];
        float dx1 = dx0 + blockDim.x * gridspacing;            // point i + blockDim.x
        float dx2 = dx0 + 2 * blockDim.x * gridspacing;        // point i + 2 * blockDim.x
        float dx3 = dx0 + 3 * blockDim.x * gridspacing;        // point i + 3 * blockDim.x
        float dy = y - atoms[n + 1];
        float dz = z - atoms[n + 2];
        float dysqdzsq = dy * dy + dz * dz;                    // shared by the four points
        float charge = atoms[n + 3];
        energy0 += charge / sqrtf(dx0 * dx0 + dysqdzsq);
        energy1 += charge / sqrtf(dx1 * dx1 + dysqdzsq);
        energy2 += charge / sqrtf(dx2 * dx2 + dysqdzsq);
        energy3 += charge / sqrtf(dx3 * dx3 + dysqdzsq);
    }
    // One bound check per point (not in the book): the row need not be a multiple of
    // COARSEN * blockDim.x, and the points past its end belong to the next row.
    int row = nx * ny * k + nx * j;
    if (i < nx) energygrid[row + i] += energy0;
    if (i + blockDim.x < nx) energygrid[row + i + blockDim.x] += energy1;
    if (i + 2 * blockDim.x < nx) energygrid[row + i + 2 * blockDim.x] += energy2;
    if (i + 3 * blockDim.x < nx) energygrid[row + i + 3 * blockDim.x] += energy3;
}

// ---------------------------------------------------------------- the host side (Exercise 1)
// Shared argument check. Not in the book.
static bool argsOk(const char* who, int nx, int ny, int nz, int k, int numatoms, int chunk) {
    if (nx <= 0 || ny <= 0 || numatoms <= 0) return false;      // nothing to add
    if (k < 0 || k >= nz || chunk < 1 || chunk > CHUNK_MAX) {
        std::fprintf(stderr, "%s: need 0 <= k < nz and 1 <= chunk <= %d (got k %d, nz %d, chunk %d)\n", who,
                     CHUNK_MAX, k, nz, chunk);
        return false;
    }
    return true;
}

// The loop of p. 517 for one slice: copy at most `chunk` atoms into constant memory,
// launch, repeat. The copies and the launches go to the same default stream, which runs
// them in the order they were issued (ch02-facts A5), so a copy never
// overwrites atoms that the previous launch is still reading. The host pointer is called
// hostAtoms: a parameter named `atoms` would hide the __constant__ array.
void launchDcsScatter(float* energygrid, int nx, int ny, int nz, float gridspacing, int k,
                      const float* hostAtoms, int numatoms, int chunk) {
    if (!argsOk("launchDcsScatter", nx, ny, nz, k, numatoms, chunk)) return;
    for (int start = 0; start < numatoms; start += chunk) {
        int count = (numatoms - start < chunk) ? numatoms - start : chunk;
        cudaMemcpyToSymbol(atoms, hostAtoms + 4 * start, count * 4 * sizeof(float));   // a count of BYTES
        dim3 dimBlock(BLOCK_X);
        dim3 dimGrid((count + BLOCK_X - 1) / BLOCK_X);     // one thread per atom of this chunk
        dcsScatterKernel<<<dimGrid, dimBlock>>>(energygrid, nx, ny, gridspacing, k, count);
    }
}

void launchDcsGather(float* energygrid, int nx, int ny, int nz, float gridspacing, int k,
                     const float* hostAtoms, int numatoms, int chunk) {
    if (!argsOk("launchDcsGather", nx, ny, nz, k, numatoms, chunk)) return;
    dim3 dimBlock(BLOCK_X, BLOCK_Y);
    dim3 dimGrid((nx + BLOCK_X - 1) / BLOCK_X, (ny + BLOCK_Y - 1) / BLOCK_Y);   // one thread per point
    for (int start = 0; start < numatoms; start += chunk) {
        int count = (numatoms - start < chunk) ? numatoms - start : chunk;
        cudaMemcpyToSymbol(atoms, hostAtoms + 4 * start, count * 4 * sizeof(float));
        dcsGatherKernel<<<dimGrid, dimBlock>>>(energygrid, nx, ny, gridspacing, k, count);
    }
}

void launchDcsCoarsened(float* energygrid, int nx, int ny, int nz, float gridspacing, int k,
                        const float* hostAtoms, int numatoms, int chunk) {
    if (!argsOk("launchDcsCoarsened", nx, ny, nz, k, numatoms, chunk)) return;
    dim3 dimBlock(BLOCK_X, BLOCK_Y);
    // A block covers COARSEN * BLOCK_X points of a row.
    dim3 dimGrid((nx + COARSEN * BLOCK_X - 1) / (COARSEN * BLOCK_X), (ny + BLOCK_Y - 1) / BLOCK_Y);
    for (int start = 0; start < numatoms; start += chunk) {
        int count = (numatoms - start < chunk) ? numatoms - start : chunk;
        cudaMemcpyToSymbol(atoms, hostAtoms + 4 * start, count * 4 * sizeof(float));
        dcsCoarsenedKernel<<<dimGrid, dimBlock>>>(energygrid, nx, ny, gridspacing, k, count);
    }
}
