// Driver for ../mx_quantize.h. The unit test app/src/sim/mxblock.test.ts compiles this
// file, runs it on the blocks section A.5 uses and compares every byte and every decoded
// float with the model behind the MX block explorer and its scene (app/src/sim/mxblock.ts).
//
//   mx_quantize_test x0 x1 ... x31
//
// prints the scale byte, then the 32 four-bit codes, then the 32 decoded values as the
// hex bits of a float, each on its own line, then the block's size in bytes.
#include "../mx_quantize.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    if (argc != 1 + kBlock) return 2;
    float v[kBlock];
    for (int i = 0; i < kBlock; ++i) v[i] = std::strtof(argv[1 + i], nullptr);
    MxBlock b = mx_quantize(v);
    std::printf("%u\n", static_cast<unsigned>(b.scale));
    for (int i = 0; i < kBlock; ++i) {
        int c = (i % 2 == 0) ? (b.codes[i / 2] & 0xF) : (b.codes[i / 2] >> 4);
        std::printf("%d\n", c);
    }
    for (int i = 0; i < kBlock; ++i) {
        float d = mx_decode(b, i);
        std::uint32_t u;
        std::memcpy(&u, &d, sizeof u);
        std::printf("%08x\n", static_cast<unsigned>(u));
    }
    std::printf("%zu\n", sizeof(MxBlock));
    return 0;
}
