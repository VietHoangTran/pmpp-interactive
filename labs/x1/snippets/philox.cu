// Philox4x32-10 and M5's uniform, one per row and step (section X1.7). Ours: random numbers
// are not in the book. The round, its constants and the key schedule follow Random123's
// include/Random123/philox.h (notes/today/x1-facts.md A4), whose notice is reproduced below.
// The code is written here, not copied; cuRAND's headers are under NVIDIA's licence and are
// not used.
//
// The 32 x 32 -> 64-bit product is formed in unsigned long long, which compiles the same on
// CUDA, HIP and the course's emulators with no intrinsic (__umulhi would do on CUDA and HIP,
// x1-facts A5, B3). Checked against Random123's known-answer vectors by
// labs/x1/snippets/check/philox_kat.cu.
/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

struct PhiloxWords {
    unsigned int x, y, z, w;
};

// The high and low halves of the exact 64-bit product.
__host__ __device__ inline void mulhilo32(unsigned int a, unsigned int b, unsigned int* hi, unsigned int* lo) {
    const unsigned long long p = (unsigned long long)a * b;
    *hi = (unsigned int)(p >> 32);
    *lo = (unsigned int)p;
}

// One round: two products, the high halves XORed with the other words and the key, a swap.
__host__ __device__ inline PhiloxWords philoxRound(PhiloxWords c, unsigned int k0, unsigned int k1) {
    unsigned int hi0, lo0, hi1, lo1;
    mulhilo32(0xD2511F53u, c.x, &hi0, &lo0);
    mulhilo32(0xCD9E8D57u, c.z, &hi1, &lo1);
    PhiloxWords out = {hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0};
    return out;
}

// Ten rounds; the key grows by two fixed constants between them, not before the first.
__host__ __device__ inline PhiloxWords philox4x32_10(PhiloxWords c, unsigned int k0, unsigned int k1) {
    for (int r = 0; r < 10; ++r) {
        if (r > 0) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        c = philoxRound(c, k0, k1);
    }
    return c;
}

// M5's mapping (ours): counter = (step low, step high, stream low, stream high),
// key = (seed low, seed high); word 0 becomes the row's u.
__host__ __device__ inline unsigned int m5Bits(unsigned long long seed, unsigned long long stream,
                                               unsigned long long step) {
    PhiloxWords c = {(unsigned int)step, (unsigned int)(step >> 32), (unsigned int)stream,
                     (unsigned int)(stream >> 32)};
    return philox4x32_10(c, (unsigned int)seed, (unsigned int)(seed >> 32)).x;
}

// The top 24 bits times 2^-24: exact, so the same float on every backend; in [0, 1 - 2^-24].
__host__ __device__ inline float m5Uniform(unsigned int x) {
    return (float)(x >> 8) * (1.0f / 16777216.0f);
}

// One thread per row: each row brings its own seed, stream and step.
__global__ void uniformKernel(const unsigned long long* seed, const unsigned long long* stream,
                              const unsigned long long* step, float* u, int rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) u[r] = m5Uniform(m5Bits(seed[r], stream[r], step[r]));
}

void launchUniform(const unsigned long long* seed, const unsigned long long* stream, const unsigned long long* step,
                   float* u, int rows) {
    const int block = 128;
    uniformKernel<<<(rows + block - 1) / block, block>>>(seed, stream, step, u, rows);
}
