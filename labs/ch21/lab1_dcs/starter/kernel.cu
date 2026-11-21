// Lab 21.1 starter (CUDA; also built for HIP and for the CPU emulators).
// Direct Coulomb summation for one slice of the potential map, three ways.
// Book: section 21.2 (pp. 515-519), Figs. 21.5 and 21.6; sections 21.3 and 21.4
// (pp. 519-522), Fig. 21.10; Exercise 1 (p. 528) is the host code.
//
// dcs.h has the sizes (CHUNK_MAX, BLOCK_X, BLOCK_Y, COARSEN), the grid layout and what each
// launch must do. Do not hard-code their values.
//
// Where this lab differs from the pages:
//   - The kernels take the slice as an int k; its z coordinate is k * gridspacing. The
//     book passes a float z and truncates z / gridspacing back to k.
//   - The sizes are two ints, nx and ny, where the book passes a dim3 grid.
//   - This grid is not padded to whole blocks (Fig. 21.9 draws one that is), and a chunk
//     can hold fewer atoms than the launch has threads. The book's listings test no
//     bounds; yours must.
#include "../dcs.h"

#include <cstdio>

// TODO 0: declare one chunk of atoms in constant memory, at file scope: room for CHUNK_MAX
//         atoms of four floats each (x, y, z, charge). The book calls it `atoms`.

// ---------------------------------------------------------------- Fig. 21.5: scatter
// One thread per atom of the chunk; the thread adds its atom's contribution to every point
// of slice k.
__global__ void dcsScatterKernel(float* energygrid, int nx, int ny, float gridspacing, int k, int numatoms) {
    // TODO 1: which atom is this thread's? Leave if it is past the last of the `numatoms`
    //         atoms of this chunk (the book's listing has no such test: why does yours
    //         need one?).
    // TODO 2: walk every point (i, j) of slice k. Work out what depends only on the atom,
    //         what on the row, and what on the point, and compute each as rarely as you
    //         can (Fig. 21.4 shows the hoisting).
    // TODO 3: add this atom's charge / distance to the point. Every thread adds into every
    //         point at about the same time: the addition must be indivisible.
}

// ---------------------------------------------------------------- Fig. 21.6: gather
// One thread per point of slice k; the thread sums the chunk's atoms and writes once.
__global__ void dcsGatherKernel(float* energygrid, int nx, int ny, float gridspacing, int k, int numatoms) {
    // TODO 4: which point (i, j) is this thread's? Leave if it is outside the slice.
    // TODO 5: sum charge / distance over the chunk's atoms in a register.
    // TODO 6: add the sum to the point. The launch calls this kernel once per chunk, so
    //         the point already holds the earlier chunks' sum.
}

// ------------------------------------------------- Fig. 21.10: coarsened, coalesced writes
// One thread per COARSEN (four) points of a row. A block covers COARSEN * blockDim.x
// consecutive points of a row. Give thread t of the block the points t, t + blockDim.x,
// t + 2 * blockDim.x and t + 3 * blockDim.x of that span, so that for each of the four
// write statements the threads of a warp write consecutive addresses.
__global__ void dcsCoarsenedKernel(float* energygrid, int nx, int ny, float gridspacing, int k, int numatoms) {
    // TODO 7: this thread's first point i and its row j. Leave if the row is outside the slice.
    // TODO 8: four sums in registers. Read each atom once and use it for all four points:
    //         which parts of the distance do the four points share, and how far apart in x
    //         are they?
    // TODO 9: add each sum to its point, but only if that point is inside the row: the
    //         row need not be a multiple of COARSEN * blockDim.x points.
}

// ---------------------------------------------------------------- the host side (Exercise 1)
// The loop of p. 517, for one slice: copy at most `chunk` atoms into constant memory with
// cudaMemcpyToSymbol (its third argument is a count of BYTES), launch, repeat until every
// atom has been added. The last chunk may be partial. Name the host pointer something other
// than the constant array, or it will hide it.
// Blocks: 1D of BLOCK_X threads for scatter; BLOCK_X x BLOCK_Y for gather and coarsened.
void launchDcsScatter(float* energygrid, int nx, int ny, int nz, float gridspacing, int k,
                      const float* hostAtoms, int numatoms, int chunk) {
    // TODO 10: one thread per atom of the current chunk.
}

void launchDcsGather(float* energygrid, int nx, int ny, int nz, float gridspacing, int k,
                     const float* hostAtoms, int numatoms, int chunk) {
    // TODO 11: one thread per point of the slice.
}

void launchDcsCoarsened(float* energygrid, int nx, int ny, int nz, float gridspacing, int k,
                        const float* hostAtoms, int numatoms, int chunk) {
    // TODO 12: one thread per COARSEN points of a row.
}
