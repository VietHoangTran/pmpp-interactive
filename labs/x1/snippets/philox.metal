// Philox4x32-10 and M5's uniform in the Metal Shading Language (section X1.7). Ours: MSL has
// no random number function (notes/today/x1-facts.md C1), so the kernel computes Philox
// itself. The round, its constants and the key schedule follow Random123's
// include/Random123/philox.h (x1-facts A4), whose notice is reproduced below. Here the high
// half comes from MSL's mulhi and the low half from the wrapping 32-bit product (C4).
// Checked against Random123's known-answer vectors by labs/x1/snippets/check/philox_metal_run.mm.
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
#include <metal_stdlib>
using namespace metal;

// One round: two products, the high halves XORed with the other words and the key, a swap.
inline uint4 philoxRound(uint4 c, uint2 k) {
    const uint hi0 = mulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
    const uint hi1 = mulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
    return uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
}

// Ten rounds; the key grows by two fixed constants between them, not before the first.
inline uint4 philox4x32_10(uint4 c, uint2 k) {
    for (int r = 0; r < 10; ++r) {
        if (r > 0) k += uint2(0x9E3779B9u, 0xBB67AE85u);
        c = philoxRound(c, k);
    }
    return c;
}

// M5's mapping (ours): counter = (step low, step high, stream low, stream high),
// key = (seed low, seed high); word 0 becomes the row's u.
inline uint m5Bits(ulong seed, ulong stream, ulong step) {
    const uint4 c = uint4(uint(step), uint(step >> 32), uint(stream), uint(stream >> 32));
    return philox4x32_10(c, uint2(uint(seed), uint(seed >> 32))).x;
}

// The top 24 bits times 2^-24: exact, so the same float on every backend; in [0, 1 - 2^-24].
inline float m5Uniform(uint x) {
    return float(x >> 8) * (1.0f / 16777216.0f);
}

// One thread per row: each row brings its own seed, stream and step.
kernel void uniformKernel(device const ulong* seed [[buffer(0)]], device const ulong* stream [[buffer(1)]],
                          device const ulong* step [[buffer(2)]], device float* u [[buffer(3)]],
                          constant uint& rows [[buffer(4)]], uint r [[thread_position_in_grid]]) {
    if (r < rows) u[r] = m5Uniform(m5Bits(seed[r], stream[r], step[r]));
}
