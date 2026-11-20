// Lab 4.1 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Ask the device, then size the launch. Book: section 4.8 (pp. 87-89) for the
// queries, section 4.7 (pp. 85-86) for the slot arithmetic.
// Second half of project milestone M1: lab 3.3's kernel, launched with a
// configuration derived from what the device reports instead of a fixed 16 x 16.
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
// The loop of p. 88. Fills info[0 .. n-1] and returns n, the number of devices
// described (at most maxDevices). Returns 0 if the count cannot be read.
//
// Where this differs from the page:
//   - The page ignores both return values. Both calls return an error code
//     (cudaError_t), and NVIDIA's guide says to check every one.
//   - The page's loop index is unsigned int, compared with the int devCount.
//     NVIDIA's own version of this loop uses int.
//   - The page lists devProp.clockRate. CUDA 13.0 removed that member, so a file
//     that reads it no longer compiles. It is not read here.
//   - The page describes regsPerBlock as the registers of an SM. That figure is
//     regsPerMultiprocessor; regsPerBlock is the limit for one block.
//   - maxThreadsPerMultiProcessor and maxBlocksPerMultiProcessor are not on the
//     page. They are the thread slots and block slots of section 4.7.
int queryDevices(DeviceInfo* info, int maxDevices) {
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess) return 0;

    int n = 0;
    cudaDeviceProp devProp;
    for (int i = 0; i < devCount && n < maxDevices; i++) {
        if (cudaGetDeviceProperties(&devProp, i) != cudaSuccess) continue;   // skip a device that cannot be read
        DeviceInfo& d = info[n++];
        std::strncpy(d.name, devProp.name, sizeof(d.name) - 1);
        d.name[sizeof(d.name) - 1] = '\0';
        d.major = devProp.major;
        d.minor = devProp.minor;
        d.warpSize = devProp.warpSize;
        d.multiProcessorCount = devProp.multiProcessorCount;
        d.maxThreadsPerBlock = devProp.maxThreadsPerBlock;
        for (int a = 0; a < 3; a++) {
            d.maxThreadsDim[a] = devProp.maxThreadsDim[a];
            d.maxGridSize[a] = devProp.maxGridSize[a];
        }
        d.maxThreadsPerMultiProcessor = devProp.maxThreadsPerMultiProcessor;
        d.maxBlocksPerMultiProcessor = devProp.maxBlocksPerMultiProcessor;
        d.regsPerMultiprocessor = devProp.regsPerMultiprocessor;
        d.regsPerBlock = devProp.regsPerBlock;
        d.sharedMemPerBlock = devProp.sharedMemPerBlock;
    }
    return n;
}

// ---------- part 2: derive ----------
// Can blocks of `threads` threads fill the thread slots of an SM? Section 4.7
// names two ways to fall short (pp. 85-86):
//   - the block is so small that the block slots run out first (32-thread
//     blocks in the book's example: 32 slots x 32 threads fill half);
//   - the block size does not divide the thread slots (768 in the book's
//     example: two fit, a third does not, a quarter of the slots stay empty).
// A field of 0 means the device did not report it, and its rule is skipped.
static bool fillsThreadSlots(const DeviceInfo& d, int threads) {
    int threadSlots = d.maxThreadsPerMultiProcessor;
    int blockSlots = d.maxBlocksPerMultiProcessor;
    if (threadSlots <= 0) return true;
    if (threadSlots % threads != 0) return false;
    if (blockSlots > 0 && (long long)threads*blockSlots < threadSlots) return false;
    return true;
}

// Picks a block and a grid for a rows x cols output, one thread per element.
// A pure function of its arguments: no CUDA call, so it can be tested on
// descriptions of devices you do not own.
//
// The shape it looks for: a block whose rows are whole warps (width a multiple
// of warpSize), as tall as the limits and the problem allow. Threads are cut
// into warps along x first (p. 75), so every warp is then a run of consecutive
// outputs of one batch row. It widens the block only when a rule demands it.
// Returns false if no block within the device's limits satisfies every rule
// with a grid the device accepts.
bool chooseLaunch(const DeviceInfo& d, int rows, int cols, dim3* grid, dim3* block) {
    if (rows < 1 || cols < 1 || d.warpSize < 1) return false;

    int maxHeight = d.maxThreadsPerBlock/d.warpSize;       // w is at least one warp
    if (maxHeight > d.maxThreadsDim[1]) maxHeight = d.maxThreadsDim[1];
    if (maxHeight > rows) maxHeight = rows;                // a taller block would only add idle threads

    for (int h = maxHeight; h >= 1; --h) {
        int maxWidth = d.maxThreadsPerBlock/h;
        if (maxWidth > d.maxThreadsDim[0]) maxWidth = d.maxThreadsDim[0];
        for (int w = d.warpSize; w <= maxWidth; w += d.warpSize) {
            if (!fillsThreadSlots(d, w*h)) continue;
            int gridX = (cols + w - 1)/w;                  // x covers the columns
            int gridY = (rows + h - 1)/h;                  // y covers the rows
            if (gridX > d.maxGridSize[0] || gridY > d.maxGridSize[1]) continue;
            *block = dim3(w, h, 1);
            *grid = dim3(gridX, gridY, 1);
            return true;
        }
    }
    return false;
}

// ---------- part 3: launch ----------
// Y = X W for a batch x in input and an in x out weight. The harness runs on
// device 0, so its description is the one that counts. It is read once: the
// answer does not change between launches, and the benchmark calls this often.
void launchLinear(const float* X, const float* W, float* Y, int batch, int in, int out) {
    static DeviceInfo device;
    static bool known = false;
    if (!known) {
        known = (queryDevices(&device, 1) == 1);
        if (!known) return;
    }
    dim3 dimGrid, dimBlock;
    if (!chooseLaunch(device, batch, out, &dimGrid, &dimBlock)) return;
    launchLinearWith(dimGrid, dimBlock, X, W, Y, batch, in, out);
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
