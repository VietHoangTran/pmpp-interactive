// Lab 4.1 starter (CUDA; also built for HIP and for the CPU emulator).
// Ask the device, then size the launch. Book: section 4.8 (pp. 87-89) for the
// queries, section 4.7 (pp. 85-86) for the slot arithmetic.
// Second half of project milestone M1: lab 3.3's kernel, launched with a
// configuration derived from what the device reports instead of a fixed 16 x 16.
//
// Your part is the three functions marked "part 1" to "part 3". The kernel, the
// launch statement and the functions at the end of the file are given.
#include <cstring>

#include "launch_config.h"

// ---------- given: the kernel ----------
// Lab 3.3's naive GEMM with a linear layer's names. One thread per output:
//   Y[b][o] = sum over k of X[b][k] * W[k][o]
// X is batch x in, W is in x out, Y is batch x out, all row-major float.
// The grid's x covers the outputs, its y covers the batch.
__global__ void linearKernel(const float* X, const float* W, float* Y, int batch, int in, int out) {
    int b = blockIdx.y*blockDim.y + threadIdx.y;   // row of Y, and the row of X to read
    int o = blockIdx.x*blockDim.x + threadIdx.x;   // column of Y, and the column of W to read
    if ((b < batch) && (o < out)) {                // the grid is whole blocks: some threads own nothing
        float y = 0;
        for (int k = 0; k < in; ++k) {
            y += X[b*in + k]*W[k*out + o];
        }
        Y[b*out + o] = y;
    }
}

// ---------- given: the launch itself ----------
// Launches linearKernel with the configuration it is handed and remembers that
// configuration, so the driver can check what was really used.
static LaunchShape lastShape;
static int launchCount = 0;

void launchLinearWith(dim3 dimGrid, dim3 dimBlock,
                      const float* X, const float* W, float* Y, int batch, int in, int out) {
    lastShape.grid[0] = dimGrid.x;   lastShape.grid[1] = dimGrid.y;   lastShape.grid[2] = dimGrid.z;
    lastShape.block[0] = dimBlock.x; lastShape.block[1] = dimBlock.y; lastShape.block[2] = dimBlock.z;
    ++launchCount;
    linearKernel<<<dimGrid, dimBlock>>>(X, W, Y, batch, in, out);
}

// ---------- part 1: ask ----------
// The loop of p. 88. Fill info[0 .. n-1], one entry per device, and return n
// (at most maxDevices). Return 0 if the count cannot be read.
int queryDevices(DeviceInfo* info, int maxDevices) {
    // TODO 1: ask how many devices there are.
    // TODO 2: for each one, ask for its properties and copy the fields that
    //         DeviceInfo names (launch_config.h). Watch the capital P: the
    //         vendors' headers are not consistent about it.
    // TODO 3: both calls return an error code. The page ignores it. Do not.
    //
    // Two things on p. 89 no longer hold. clockRate was removed from the struct
    // in CUDA 13.0, and reading it stops the file compiling. regsPerBlock is
    // the limit for one block; the registers of an SM are regsPerMultiprocessor.
    return 0;
}

// ---------- part 2: derive ----------
// Pick a block and a grid for a rows x cols output, one thread per element.
// This must stay a pure function of its arguments (no CUDA call), so that it
// can be tested on descriptions of devices you do not own.
//
// The tests check each of these separately and name the one that fails:
//   - the block is within maxThreadsPerBlock, and within maxThreadsDim on each axis;
//   - its thread count is a whole number of warps;
//   - blocks of that size can fill the thread slots of an SM: section 4.7 gives
//     two ways to fall short, one for blocks that are too small and one for
//     sizes that leave a remainder. A slot field of 0 means the device did not
//     report it: skip the rule that needs it;
//   - the block is no taller than `rows` (think of batch 1), and block.z is 1;
//   - the grid covers rows x cols with no block to spare, grid.z is 1, and the
//     grid is within maxGridSize. If no block you would accept gives such a
//     grid, return false.
// Several blocks satisfy all of this. Which one you prefer is your design
// decision; the benchmark tells you what it was worth.
bool chooseLaunch(const DeviceInfo& d, int rows, int cols, dim3* grid, dim3* block) {
    // TODO 4
    return false;
}

// ---------- part 3: launch ----------
// Y = X W for a batch x in input and an in x out weight: rows = batch, cols = out.
// The harness runs on device 0, so that is the description to use.
void launchLinear(const float* X, const float* W, float* Y, int batch, int in, int out) {
    // TODO 5: query, choose, then hand the result to launchLinearWith.
    //         The benchmark calls this many times in a row. What the device
    //         reports does not change between calls.
}

// ---------- given: how main.cpp reaches the code above ----------
bool chooseLaunchShape(const DeviceInfo& d, int rows, int cols, LaunchShape* shape) {
    dim3 dimGrid, dimBlock;
    if (!chooseLaunch(d, rows, cols, &dimGrid, &dimBlock)) return false;
    shape->grid[0] = dimGrid.x;   shape->grid[1] = dimGrid.y;   shape->grid[2] = dimGrid.z;
    shape->block[0] = dimBlock.x; shape->block[1] = dimBlock.y; shape->block[2] = dimBlock.z;
    return true;
}

// The benchmark's baseline: the same kernel with a block size chosen without
// looking at the device or the problem.
void launchLinearFixed(const float* X, const float* W, float* Y, int batch, int in, int out,
                       int blockX, int blockY) {
    dim3 dimBlock(blockX, blockY, 1);
    dim3 dimGrid((out + blockX - 1)/blockX, (batch + blockY - 1)/blockY, 1);
    launchLinearWith(dimGrid, dimBlock, X, W, Y, batch, in, out);
}

// The configuration of the most recent launch, and how many launches there have been.
int lastLaunch(LaunchShape* shape) {
    *shape = lastShape;
    return launchCount;
}
