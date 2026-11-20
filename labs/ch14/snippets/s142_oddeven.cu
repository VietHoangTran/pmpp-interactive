// Section 14.2: one step of parallel odd-even sort (Fig. 14.2, p. 332) and the host loop
// that the book describes but does not show. The kernel keeps the page's 13 lines and
// their numbers; it differs from the page on two lines:
//   line 05  i + 1 < N instead of i < N - 1: with N == 0 the unsigned N - 1 wraps to the
//            largest value and every thread would read past the end;
//   line 10  the flag is set with an atomic exchange, not a plain store: many threads
//            write it at once, and a data race is undefined behaviour in the CUDA memory
//            model even when every write stores the same value (p. 333).
// Written to serve HIP too, through pmpp/vendors/hip.h (atomicExch has the same name there);
// not built for HIP yet. Checked on the course's CPU emulator, and under its race-detection
// flavour, by app/src/chapters/ch14/S142.snippets.test.ts (driver: check/s142_check.cu).

__global__ void sort_kernel(unsigned int* data, unsigned int* hasChanged,
                            unsigned int N, unsigned int isOddStep) {
    unsigned int i = 2*(blockIdx.x*blockDim.x + threadIdx.x)
                     + (isOddStep ? 1 : 0);
    if(i + 1 < N) {
        if(data[i] > data[i + 1]) {
            unsigned int tmp = data[i];
            data[i] = data[i + 1];
            data[i + 1] = tmp;
            atomicExch(hasChanged, 1u);
        }
    }
}

// Not in the book. One launch per step, half as many threads as elements. The host stops
// only after two quiet steps in a row, one even and one odd: a single quiet step proves
// nothing ([1, 3, 2, 4] makes no swap on its first, even, step). Reading the flag back
// with cudaMemcpy waits for the step to finish. Returns the number of steps launched.
unsigned int oddEvenSort(unsigned int* data_d, unsigned int N) {
    const unsigned int threads = N / 2, block = 256;
    unsigned int* changed_d;
    cudaMalloc((void**)&changed_d, sizeof(unsigned int));
    unsigned int steps = 0, quiet = 0;
    while (threads > 0 && quiet < 2) {
        unsigned int changed = 0;
        cudaMemcpy(changed_d, &changed, sizeof(unsigned int), cudaMemcpyHostToDevice);
        sort_kernel<<<(threads + block - 1) / block, block>>>(data_d, changed_d, N, steps % 2);
        cudaMemcpy(&changed, changed_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        quiet = changed ? 0 : quiet + 1;
        ++steps;
    }
    cudaFree(changed_d);
    return steps;
}
