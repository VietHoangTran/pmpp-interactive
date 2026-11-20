// Ask every CUDA device for the limits chapter 4 depends on, and print them.
// Host code only: nothing here runs on the GPU. Returns how many devices it described.
#include <cstdio>

int describeDevices() {
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess) return 0;

    int described = 0;
    cudaDeviceProp devProp;
    for (int i = 0; i < devCount; i++) {
        if (cudaGetDeviceProperties(&devProp, i) != cudaSuccess) continue;   // skip a device that cannot be read
        std::printf("device %d: %s, compute capability %d.%d\n", i, devProp.name, devProp.major, devProp.minor);
        std::printf("  a launch: at most %d threads per block, a block of %d x %d x %d, a grid of %d x %d x %d\n",
                    devProp.maxThreadsPerBlock,
                    devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2],
                    devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
        std::printf("  the device: %d SMs, warps of %d threads\n", devProp.multiProcessorCount, devProp.warpSize);
        std::printf("  one SM: %d thread slots, %d block slots, %d registers, %zu bytes of shared memory\n",
                    devProp.maxThreadsPerMultiProcessor, devProp.maxBlocksPerMultiProcessor,
                    devProp.regsPerMultiprocessor, devProp.sharedMemPerMultiprocessor);
        std::printf("  one block: at most %d registers\n", devProp.regsPerBlock);
        described++;
    }
    return described;
}
