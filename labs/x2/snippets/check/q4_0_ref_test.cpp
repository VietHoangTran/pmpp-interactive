// Driver for ../q4_0_ref.h. The unit test app/src/sim/quant.test.ts compiles this file
// (-std=c++17 -Wall -Wextra -O2 -ffp-contract=off), runs it on the blocks sections X2.2 and
// X2.3 use and compares every byte and every float, bit for bit, with app/src/sim/quant.ts.
//
//   q4_0_ref_test w0 .. w31 y0 .. y31      (each a float32 as 8 hex digits)
//
// prints the FP16 bits of d, the 16 bytes of qs, the 32 dequantised values, the four
// 32-bit words q4_0_word returns, the folded dot product, and sizeof(BlockQ4_0), one per line.
#include "../q4_0_ref.h"

#include <cstdio>
#include <cstdlib>

static float from_hex(const char* s) {
    const uint32_t u = static_cast<uint32_t>(std::strtoul(s, nullptr, 16));
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static unsigned bits_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<unsigned>(u);
}

int main(int argc, char** argv) {
    if (argc != 1 + 2 * QK4_0) return 2;
    float w[QK4_0], y[QK4_0], out[QK4_0];
    for (int i = 0; i < QK4_0; ++i) {
        w[i] = from_hex(argv[1 + i]);
        y[i] = from_hex(argv[1 + QK4_0 + i]);
    }
    BlockQ4_0 b;
    quantize_q4_0_block(w, &b);
    std::printf("%04x\n", static_cast<unsigned>(b.d));
    for (int j = 0; j < QK4_0 / 2; ++j) std::printf("%u\n", static_cast<unsigned>(b.qs[j]));
    dequantize_q4_0_block(&b, out);
    for (int j = 0; j < QK4_0; ++j) std::printf("%08x\n", bits_of(out[j]));
    for (int i = 0; i < 4; ++i) std::printf("%08x\n", static_cast<unsigned>(q4_0_word(&b, i)));
    std::printf("%08x\n", bits_of(dot_q4_0_block(&b, y)));
    std::printf("%zu\n", sizeof(BlockQ4_0));
    return 0;
}
