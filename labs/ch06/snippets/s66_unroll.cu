// Section 6.6, pp. 144-145: the book's 16-iteration loop, made compilable. A and B
// stand for the book's two statements: B(i) uses what A(i) produced, and A(i + 1)
// does not depend on A(i) or B(i). Each thread works on its own 16 elements.
__device__ float A(const float* in, unsigned int i) { return in[i]*in[i]; }
__device__ void B(float* out, unsigned int i, float a) { out[i] = a + 1.0f; }

__global__ void original(const float* in, float* out) {
    unsigned int base = 16*(blockIdx.x*blockDim.x + threadIdx.x);
    for (unsigned int i = 0; i < 16; ++i) {
        float a = A(in, base + i);
        B(out, base + i, a);
    }
}

// Unrolled by 4 and reordered by hand: four A's, then four B's, four times.
__global__ void unrolledReordered(const float* in, float* out) {
    unsigned int base = 16*(blockIdx.x*blockDim.x + threadIdx.x);
    for (unsigned int i = 0; i < 16; i += 4) {
        float a0 = A(in, base + i);
        float a1 = A(in, base + i + 1);
        float a2 = A(in, base + i + 2);
        float a3 = A(in, base + i + 3);
        B(out, base + i, a0);
        B(out, base + i + 1, a1);
        B(out, base + i + 2, a2);
        B(out, base + i + 3, a3);
    }
}

// The same request made to the compiler. #pragma unroll 1 would forbid unrolling.
__global__ void askedToUnroll(const float* in, float* out) {
    unsigned int base = 16*(blockIdx.x*blockDim.x + threadIdx.x);
#pragma unroll 4
    for (unsigned int i = 0; i < 16; ++i) {
        float a = A(in, base + i);
        B(out, base + i, a);
    }
}
