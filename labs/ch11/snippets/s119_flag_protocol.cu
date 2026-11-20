// Section 11.9, "Make the flag safe": the flag protocol of Fig. 11.17 (pp. 278-279) in the
// spelling of NVIDIA's message-passing example (notes/today/ch11-facts.md A5). The writer
// stores the sum with a plain store, then sets the flag with a device-scope release store;
// the reader spins on a device-scope acquire load of the flag, then loads the sum with a
// plain load. The flag is read with load(), NVIDIA's spelling, where the book writes
// fetch_add(0, ...).
//
// The page prints the bodies of publish() and waitFor() line for line, with the orders and
// the scope the reader picks. This file holds the one combination that is guaranteed;
// app/src/chapters/ch11/S119.snippets.test.ts ties all eight combinations the page offers
// to these lines and runs the CUDA front-end check on each. Nothing here is run: lab 11.5
// runs the whole single-pass kernel on the emulators.
#include <cuda/atomic>

// The leader of block b - 1: publish the sum of every segment up to and including its own.
__device__ void publish(float* partialSums, unsigned int* flags, unsigned int bid, float prev, float blockSum) {
    partialSums[bid] = prev + blockSum;          // plain store
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> f(flags[bid]);
    f.store(1, cuda::memory_order_release);
}

// The leader of block b: wait until block b - 1 has published, then read what it published.
__device__ void waitFor(const float* partialSums, unsigned int* flags, unsigned int bid, float& prev) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> f(flags[bid - 1]);
    while (f.load(cuda::memory_order_acquire) == 0) {}
    prev = partialSums[bid - 1];                 // plain load
}
