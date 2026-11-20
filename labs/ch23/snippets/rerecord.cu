// Chapter 23, summary page: a host sequence of our own, not the book's, shown for one question
// ("ch23/wrap/q-rerecord"). One event is recorded twice in stream s1, and stream s2 waits on it
// once, between the two records. The lesson's copy is the region between the markers, tied to
// this file by app/src/chapters/ch23/Wrap.snippets.test.ts.
//
// What it shows rests on NVIDIA's runtime reference (notes/today/ch23-facts.md, A5):
// cudaStreamWaitEvent uses the event's "most recent captured state at the time of the API
// call", and a later cudaEventRecord does not change a wait already issued. So consumeC waits
// for produceA only, and may read x while produceB is still adding to it.
//
// Status: front-end-checked (clang's CUDA parser against the course's stub headers). Never run.
// The course's emulator could not show the point: it runs every stream call synchronously.
#include <cuda_runtime.h>

__global__ void produceA(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f;
}
__global__ void produceB(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}
__global__ void consumeC(float* y, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 2.0f * x[i];
}

// s1 and s2 are two streams the caller created; done is an event made with
// cudaEventDisableTiming. x and y are device arrays of n floats.
void rerecord(float* x, float* y, int n, cudaStream_t s1, cudaStream_t s2, cudaEvent_t done) {
    int block = 256, grid = (n + block - 1) / block;
    // [shown]
    produceA<<<grid, block, 0, s1>>>(x, n);      // x = 1
    cudaEventRecord(done, s1);
    cudaStreamWaitEvent(s2, done, 0);
    produceB<<<grid, block, 0, s1>>>(x, n);      // x = x + 1
    cudaEventRecord(done, s1);
    consumeC<<<grid, block, 0, s2>>>(y, x, n);   // y = 2 * x
    // [/shown]
}

// The fix the question's explanation gives: issue the wait after the record it needs.
void rerecordFixed(float* x, float* y, int n, cudaStream_t s1, cudaStream_t s2, cudaEvent_t done) {
    int block = 256, grid = (n + block - 1) / block;
    produceA<<<grid, block, 0, s1>>>(x, n);
    produceB<<<grid, block, 0, s1>>>(x, n);
    cudaEventRecord(done, s1);
    cudaStreamWaitEvent(s2, done, 0);
    consumeC<<<grid, block, 0, s2>>>(y, x, n);
}
