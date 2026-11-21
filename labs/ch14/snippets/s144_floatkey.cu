// Section 14.4: turn float logits into unsigned keys that a radix sort orders correctly.
// Ours, not the book's: the book only warns that non-comparison sorts "may not generalize
// to arbitrary types of keys" (p. 330). The transform is the one CUB documents for float
// keys (notes/today/ch14-facts.md A3) and rocPRIM uses (B2); E1 checks its bit patterns.
//
// Every function takes the float's 32 bits as an unsigned int and never holds a float:
//   - the unsigned view needs no bit-cast intrinsic on any backend, so the labs read a
//     float buffer through it (on CUDA and HIP, __float_as_uint(x) gives the same bits
//     from a float in a register: A8, B6);
//   - on Metal, fast and relaxed math both allow "no signed zeros", so a float test such
//     as x == 0.0f could be rewritten; done on the bits, nothing is left to rewrite (C3).
// The same source compiles for HIP through pmpp/vendors/hip.h. Shown in section 14.4; run
// on the CPU emulator against app/src/sim/floatkey.ts by
// app/src/chapters/ch14/S144.snippets.test.ts (driver: check/s144_check.cu).

// region: floatkey
__host__ __device__ inline unsigned int float_to_key(unsigned int u) {
    if(u == 0x80000000u) u = 0u;                    // -0.0 sorts as +0.0
    unsigned int mask = (u & 0x80000000u) ? 0xFFFFFFFFu  // negative: invert every bit
                                          : 0x80000000u; // otherwise: set the sign bit
    return u ^ mask;
}

__host__ __device__ inline unsigned int float_to_key_desc(unsigned int u) {
    return ~float_to_key(u);                        // descending: invert, sort ascending
}

__global__ void logits_to_keys(const unsigned int* logitBits, unsigned int* keys,
                               unsigned int n, bool descending) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) {
        unsigned int u = logitBits[i];              // the float's bits, never a float
        keys[i] = descending ? float_to_key_desc(u) : float_to_key(u);
    }
}
// endregion
