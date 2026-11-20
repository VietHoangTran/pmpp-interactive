// Lab 2.1 starter (CUDA; also built for HIP and for the CPU emulator).
// The whole program of chapter 2: the kernel of Fig. 2.10 (p. 35) and the host
// function of Fig. 2.13 (p. 38), with the error check of the sidebar on p. 32.
//
// Contract (the driver calls vecAdd with ordinary host arrays):
//   - vecAdd returns 0 when everything worked and a non-zero value when any
//     runtime call or the launch failed. It never calls exit().
//   - threadsPerBlock is an argument. Do not hard-code 256.
//   - n is at least 1 and is rarely a multiple of threadsPerBlock.
#include <cstdio>

// TODO 1: write CHECK(call) from the sidebar on p. 32. As printed the snippet
//         does not compile: read it closely before you copy it. Your macro
//         prints the error string, the file and the line, and then makes the
//         enclosing function return a non-zero value instead of calling exit().
//         It frees nothing.

// Compute vector sum C = A + B.
// Each thread performs one pair-wise addition.
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    // TODO 2: which element does this thread own? Some threads own none.
}

int vecAdd(const float* A_h, const float* B_h, float* C_h, int n, int threadsPerBlock) {
    // Every runtime call below goes through CHECK.

    // TODO 3: Part 1. Allocate device memory for A, B and C. Copy A and B to
    //         the device. Sizes are in bytes.

    // TODO 4: Part 2. How many blocks of threadsPerBlock threads cover n
    //         elements? Launch them.

    // TODO 5: The launch statement has no value to test, and it returns before
    //         the grid has run. Ask the runtime whether the launch was accepted,
    //         then wait for the grid, and check both answers.

    // TODO 6: Part 3. Copy C back to the host. Free the three device arrays.

    return -1;   // replace with 0 once the function is complete
}
