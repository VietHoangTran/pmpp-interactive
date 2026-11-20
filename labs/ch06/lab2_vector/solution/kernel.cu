// Lab 6.2 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Book: section 6.3 (pp. 138-139), Fig. 6.12, Exercise 4 (p. 155); section 6.5 (pp. 141-143).
//
// z = x + y, each thread owning the four consecutive elements 4i .. 4i+3, two ways:
//   vecAdd4Kernel       four elements per INSTRUCTION: one 16-byte load per input and one
//                       16-byte store for a full chunk (Fig. 6.12).
//   vecAddCoarseKernel  four elements per THREAD, one 4-byte access at a time: the
//                       coarsening loop of section 6.5 with a factor of 4.
// The provided vecAddKernel (provided/vecadd.cu) is the one-element-per-thread baseline.
//
// Where this file differs from the page:
// - Fig. 6.12 has no bounds check and never uses its parameter N; the book leaves the
//   boundaries to Exercise 4. Here a full chunk takes the figure's path, a partial last
//   chunk falls back to scalar accesses, and a thread past the data does nothing.
// - The inputs are const, so the casts are (const float4*).
// - Section 6.5's loop calls a placeholder foo(i) with no bounds check and unsigned
//   indices; here the body is the addition, each element is checked against n, and the
//   indices are int, as in Fig. 6.12.
//
// The cast is the book's technique. NVIDIA's reference guides do not describe it; its
// developer blog does, and requires the cast pointer to be aligned to the vector's size,
// 16 bytes (notes/today/ch06-facts.md, A11). cudaMalloc returns memory aligned to at
// least 256 bytes (A3), so x, y and z as allocated are fine; x + 1 would not be.

#define COARSENING_FACTOR 4

__global__ void vecAdd4Kernel(const float* x, const float* y, float* z, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;   // names the chunk 4i .. 4i+3
    int first = 4 * i;
    if (first + 3 < n) {
        // The whole chunk is inside the arrays: the body of Fig. 6.12.
        float4 x4 = ((const float4*)x)[i];
        float4 y4 = ((const float4*)y)[i];
        float4 z4;
        z4.x = x4.x + y4.x;
        z4.y = x4.y + y4.y;
        z4.z = x4.z + y4.z;
        z4.w = x4.w + y4.w;
        ((float4*)z)[i] = z4;
    } else if (first < n) {
        // The last chunk holds 1 to 3 elements. A 16-byte access would run past the end
        // of the arrays, so these go one float at a time.
        for (int j = first; j < n; ++j) z[j] = x[j] + y[j];
    }
    // first >= n: this thread's chunk starts past the data. It touches nothing.
}

void launchVecAdd4(const float* x, const float* y, float* z, int n) {
    int chunks = (n + 3) / 4;   // one thread per chunk; the last chunk may be partial
    vecAdd4Kernel<<<(chunks + 255) / 256, 256>>>(x, y, z, n);
}

__global__ void vecAddCoarseKernel(const float* x, const float* y, float* z, int n) {
    int iStart = COARSENING_FACTOR * (blockIdx.x * blockDim.x + threadIdx.x);
    #pragma unroll
    for (int c = 0; c < COARSENING_FACTOR; ++c) {
        int i = iStart + c;
        if (i < n) z[i] = x[i] + y[i];   // ordinary 4-byte accesses
    }
}

void launchVecAddCoarse(const float* x, const float* y, float* z, int n) {
    int threads = (n + COARSENING_FACTOR - 1) / COARSENING_FACTOR;
    vecAddCoarseKernel<<<(threads + 255) / 256, 256>>>(x, y, z, n);
}
