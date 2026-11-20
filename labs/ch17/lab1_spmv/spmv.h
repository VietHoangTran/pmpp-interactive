// Lab 17.1, provided (do not edit): the block size of every SpMV launch in this lab.
// Included by kernel.cu and launch.mm (as "../spmv.h"), by provided/ and by main.cpp.
// The Metal kernels do not need it: the threadgroup size is chosen in launch.mm.
#pragma once

// Threads per block on CUDA and HIP; the largest threadgroup on Metal, where a pipeline
// may allow fewer (maxTotalThreadsPerThreadgroup).
#define SPMV_BLOCK 128
