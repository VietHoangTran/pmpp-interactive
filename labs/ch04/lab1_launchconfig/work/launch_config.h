// Lab 4.1: the records this lab fills. Given; do not edit.
// The same file sits in starter/, work/ and solution/. main.cpp includes it too.
#pragma once
#include <cstddef>

// ---------- CUDA and HIP ----------
// What queryDevices() reports for one device. Every field has the name of the
// cudaDeviceProp / hipDeviceProp_t member it is copied from, except major and
// minor, which the vendors' structs also call major and minor.
struct DeviceInfo {
    char   name[256];
    int    major, minor;                 // compute capability on NVIDIA; an architecture number on AMD
    int    warpSize;                     // threads in a warp
    int    multiProcessorCount;          // SMs on NVIDIA; compute units on AMD
    int    maxThreadsPerBlock;
    int    maxThreadsDim[3];             // largest block along x, y, z
    int    maxGridSize[3];               // largest grid along x, y, z
    int    maxThreadsPerMultiProcessor;  // thread slots of one SM. 0 = not reported
    int    maxBlocksPerMultiProcessor;   // block slots of one SM. 0 = not reported
    int    regsPerMultiprocessor;        // registers of one SM (lower-case p in the vendors' headers)
    int    regsPerBlock;                 // the most registers one block may hold
    size_t sharedMemPerBlock;            // bytes
};

// A launch configuration in plain integers. main.cpp is ordinary C++ and has no
// dim3, so the functions it calls hand configurations over in this form.
struct LaunchShape {
    unsigned grid[3];
    unsigned block[3];
};

// ---------- Metal ----------
// What queryMetal() reports. The first four fields come from the MTLDevice,
// the last two from the MTLComputePipelineState: on Metal the width and the
// thread limit belong to a compiled kernel, not to the device.
struct MetalInfo {
    char          name[256];
    unsigned long maxThreadsPerThreadgroup[3];     // device: width, height, depth
    unsigned long maxThreadgroupMemoryLength;      // device: bytes
    bool          hasUnifiedMemory;                // device
    unsigned long threadExecutionWidth;            // pipeline
    unsigned long maxTotalThreadsPerThreadgroup;   // pipeline
};
