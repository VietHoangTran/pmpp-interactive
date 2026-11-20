// Lab X2.4, provided (CUDA; also built for HIP and for the CPU emulators). Not in the book.
// A copy of lab 22.2's provided/step_timer.cu, unchanged below this note.
// Times a stretch of launches with one pair of events, which is how the benchmark times one
// decode step on CUDA and HIP: an event before the step's first launch, one after its last,
// and the device's own count of the milliseconds between them (cudaEventElapsedTime,
// hipEventElapsedTime: infra-facts A8, B8). On the emulators the same calls return host
// wall-clock time, which means nothing; the benchmark does not run there.
//
// Returns the milliseconds, or -1 if an event call failed (the caller then falls back to
// the host's clock).
#include <cstdio>

float eventTimeMs(void (*enqueue)(void* ctx), void* ctx) {
    cudaEvent_t start, stop;
    if (cudaEventCreate(&start) != cudaSuccess) return -1.0f;
    if (cudaEventCreate(&stop) != cudaSuccess) {
        cudaEventDestroy(start);
        return -1.0f;
    }
    float ms = -1.0f;
    // Both events go to the default stream, the stream the step's launches go to.
    if (cudaEventRecord(start) == cudaSuccess) {
        enqueue(ctx);
        if (cudaEventRecord(stop) != cudaSuccess || cudaEventSynchronize(stop) != cudaSuccess ||
            cudaEventElapsedTime(&ms, start, stop) != cudaSuccess) {
            ms = -1.0f;
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}
