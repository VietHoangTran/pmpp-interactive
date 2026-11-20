// Section 9.4's bug hunt, "Seven broken privatised kernels". One privatised histogram
// kernel in the shape of Fig. 9.10 (p. 214), shrunk for the hunt: NUM_BINS bins, one
// pixel per thread, pixels already reduced to bin numbers. The hunt launches two
// blocks of four threads on eight pixels.
//
// histPrivatised is correct. Each of the seven kernels after it changes or removes one
// thing; the lesson works out which, by diffing each one against histPrivatised. The
// private count is read once into `count` (the book reads bins_s[b] in the test and
// again in the add). app/src/sim/histpriv.ts models every kernel, and
// app/src/sim/histpriv.test.ts runs this file on the course's emulator in its three
// thread orders and on the race-detection flavour, and compares both with the model.
#include <cuda/atomic>

#define NUM_BINS 8

__global__ void histPrivatised(const unsigned char* image, unsigned int* bins, unsigned int n) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        bins_s[b] = 0;
    __syncthreads();
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bin_ref(bins_s[image[i]]);
        bin_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        unsigned int count = bins_s[b];
        if (count > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bin_ref(bins[b]);
            bin_ref.fetch_add(count, cuda::memory_order_relaxed);
        }
    }
}

__global__ void histNoZeroing(const unsigned char* image, unsigned int* bins, unsigned int n) {
    __shared__ unsigned int bins_s[NUM_BINS];
    __syncthreads();
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bin_ref(bins_s[image[i]]);
        bin_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        unsigned int count = bins_s[b];
        if (count > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bin_ref(bins[b]);
            bin_ref.fetch_add(count, cuda::memory_order_relaxed);
        }
    }
}

__global__ void histNoBarrierAfterZeroing(const unsigned char* image, unsigned int* bins, unsigned int n) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        bins_s[b] = 0;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bin_ref(bins_s[image[i]]);
        bin_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        unsigned int count = bins_s[b];
        if (count > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bin_ref(bins[b]);
            bin_ref.fetch_add(count, cuda::memory_order_relaxed);
        }
    }
}

__global__ void histNoBarrierBeforeCommit(const unsigned char* image, unsigned int* bins, unsigned int n) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        bins_s[b] = 0;
    __syncthreads();
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bin_ref(bins_s[image[i]]);
        bin_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        unsigned int count = bins_s[b];
        if (count > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bin_ref(bins[b]);
            bin_ref.fetch_add(count, cuda::memory_order_relaxed);
        }
    }
}

__global__ void histPlainSharedUpdate(const unsigned char* image, unsigned int* bins, unsigned int n) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        bins_s[b] = 0;
    __syncthreads();
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ++bins_s[image[i]];
    }
    __syncthreads();
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        unsigned int count = bins_s[b];
        if (count > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bin_ref(bins[b]);
            bin_ref.fetch_add(count, cuda::memory_order_relaxed);
        }
    }
}

__global__ void histPlainCommit(const unsigned char* image, unsigned int* bins, unsigned int n) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        bins_s[b] = 0;
    __syncthreads();
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bin_ref(bins_s[image[i]]);
        bin_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        unsigned int count = bins_s[b];
        if (count > 0) {
            bins[b] += count;
        }
    }
}

__global__ void histBlockScopeCommit(const unsigned char* image, unsigned int* bins, unsigned int n) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        bins_s[b] = 0;
    __syncthreads();
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bin_ref(bins_s[image[i]]);
        bin_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        unsigned int count = bins_s[b];
        if (count > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bin_ref(bins[b]);
            bin_ref.fetch_add(count, cuda::memory_order_relaxed);
        }
    }
}

__global__ void histCommitBoundBlockDim(const unsigned char* image, unsigned int* bins, unsigned int n) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        bins_s[b] = 0;
    __syncthreads();
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bin_ref(bins_s[image[i]]);
        bin_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    for (unsigned int b = threadIdx.x; b < blockDim.x; b += blockDim.x) {
        unsigned int count = bins_s[b];
        if (count > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bin_ref(bins[b]);
            bin_ref.fetch_add(count, cuda::memory_order_relaxed);
        }
    }
}
