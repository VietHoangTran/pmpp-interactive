// Section 12.4's stepper and outcome table: the privatised unstable filter of Fig. 12.6
// (p. 296), shrunk to the stepper's size (BLOCK_DIM = 4 threads per block, one key per
// thread), and four copies that are wrong on purpose. The stepper launches four blocks of
// four threads over sixteen keys.
//
// filterPrivatised is correct. filterNoBarrier1, 2 and 3 each drop one of its three
// barriers; filterPlainCounter bumps the private counter with ++ instead of an atomic.
// app/src/sim/privfilter.ts models every kernel; app/src/sim/privfilter.test.ts pins each
// kernel body to the model's text, runs this file on the course's emulator in its three
// thread orders and on the race-detection flavour, and compares both with the model.
//
// Differences from the book's listing: the book names two different things j (the slot in
// the private list and the start of the block's public range); here they are slot and
// blockStart, and blockStart is declared with the other shared variables. cond() keeps
// odd values: the driver stores key i as 2i + 1 when it is to be kept and as 2i when not.
// The four long statements are broken over two lines so that the stepper's listing fits its
// column without scrolling sideways.
#include <cuda/atomic>

#define BLOCK_DIM 4

__device__ bool cond(unsigned int val) { return (val & 1u) != 0u; }

__global__ void filterPrivatised(const unsigned int* input, unsigned int* output, unsigned int N, unsigned int* outputSize) {
    __shared__ unsigned int output_s[BLOCK_DIM];
    __shared__ unsigned int outputSize_s;
    __shared__ unsigned int blockStart;
    if (threadIdx.x == 0) outputSize_s = 0;
    __syncthreads();  // barrier 1
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int val = input[i];
        if (cond(val)) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                outputSize_s_ref(outputSize_s);
            unsigned int slot = outputSize_s_ref.fetch_add(1,
                cuda::memory_order_relaxed);
            output_s[slot] = val;
        }
    }
    __syncthreads();  // barrier 2
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            outputSize_ref(*outputSize);
        blockStart = outputSize_ref.fetch_add(outputSize_s,
            cuda::memory_order_relaxed);
    }
    __syncthreads();  // barrier 3
    if (threadIdx.x < outputSize_s)
        output[blockStart + threadIdx.x] = output_s[threadIdx.x];
}

__global__ void filterNoBarrier1(const unsigned int* input, unsigned int* output, unsigned int N, unsigned int* outputSize) {
    __shared__ unsigned int output_s[BLOCK_DIM];
    __shared__ unsigned int outputSize_s;
    __shared__ unsigned int blockStart;
    if (threadIdx.x == 0) outputSize_s = 0;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int val = input[i];
        if (cond(val)) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                outputSize_s_ref(outputSize_s);
            unsigned int slot = outputSize_s_ref.fetch_add(1,
                cuda::memory_order_relaxed);
            output_s[slot] = val;
        }
    }
    __syncthreads();  // barrier 2
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            outputSize_ref(*outputSize);
        blockStart = outputSize_ref.fetch_add(outputSize_s,
            cuda::memory_order_relaxed);
    }
    __syncthreads();  // barrier 3
    if (threadIdx.x < outputSize_s)
        output[blockStart + threadIdx.x] = output_s[threadIdx.x];
}

__global__ void filterNoBarrier2(const unsigned int* input, unsigned int* output, unsigned int N, unsigned int* outputSize) {
    __shared__ unsigned int output_s[BLOCK_DIM];
    __shared__ unsigned int outputSize_s;
    __shared__ unsigned int blockStart;
    if (threadIdx.x == 0) outputSize_s = 0;
    __syncthreads();  // barrier 1
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int val = input[i];
        if (cond(val)) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                outputSize_s_ref(outputSize_s);
            unsigned int slot = outputSize_s_ref.fetch_add(1,
                cuda::memory_order_relaxed);
            output_s[slot] = val;
        }
    }
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            outputSize_ref(*outputSize);
        blockStart = outputSize_ref.fetch_add(outputSize_s,
            cuda::memory_order_relaxed);
    }
    __syncthreads();  // barrier 3
    if (threadIdx.x < outputSize_s)
        output[blockStart + threadIdx.x] = output_s[threadIdx.x];
}

__global__ void filterNoBarrier3(const unsigned int* input, unsigned int* output, unsigned int N, unsigned int* outputSize) {
    __shared__ unsigned int output_s[BLOCK_DIM];
    __shared__ unsigned int outputSize_s;
    __shared__ unsigned int blockStart;
    if (threadIdx.x == 0) outputSize_s = 0;
    __syncthreads();  // barrier 1
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int val = input[i];
        if (cond(val)) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                outputSize_s_ref(outputSize_s);
            unsigned int slot = outputSize_s_ref.fetch_add(1,
                cuda::memory_order_relaxed);
            output_s[slot] = val;
        }
    }
    __syncthreads();  // barrier 2
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            outputSize_ref(*outputSize);
        blockStart = outputSize_ref.fetch_add(outputSize_s,
            cuda::memory_order_relaxed);
    }
    if (threadIdx.x < outputSize_s)
        output[blockStart + threadIdx.x] = output_s[threadIdx.x];
}

__global__ void filterPlainCounter(const unsigned int* input, unsigned int* output, unsigned int N, unsigned int* outputSize) {
    __shared__ unsigned int output_s[BLOCK_DIM];
    __shared__ unsigned int outputSize_s;
    __shared__ unsigned int blockStart;
    if (threadIdx.x == 0) outputSize_s = 0;
    __syncthreads();  // barrier 1
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int val = input[i];
        if (cond(val)) {
            unsigned int slot = outputSize_s++;
            output_s[slot] = val;
        }
    }
    __syncthreads();  // barrier 2
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            outputSize_ref(*outputSize);
        blockStart = outputSize_ref.fetch_add(outputSize_s,
            cuda::memory_order_relaxed);
    }
    __syncthreads();  // barrier 3
    if (threadIdx.x < outputSize_s)
        output[blockStart + threadIdx.x] = output_s[threadIdx.x];
}
