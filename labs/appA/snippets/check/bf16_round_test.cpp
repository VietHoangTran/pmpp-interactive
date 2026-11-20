// Driver for ../bf16_round.h. The unit test app/src/sim/fpadd.test.ts compiles this file,
// runs it on a list of FP32 bit patterns and compares every result, bit for bit, with the
// model behind section A.4's explorer (bf16FromBits in app/src/sim/fpadd.ts).
//
//   bf16_round_test 3f808000 7f7fffff ...   one line per input: "<rounded> <truncated>" in hex
#include "../bf16_round.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    for (int k = 1; k < argc; ++k) {
        std::uint32_t u = (std::uint32_t)std::strtoul(argv[k], nullptr, 16);
        float f;
        std::memcpy(&f, &u, sizeof f);
        std::uint16_t r = bf16_round(f);
        std::uint16_t t = bf16_truncate(f);
        // Converting back must give the same bits: bf16_to_float is the inverse on BF16 values.
        float back = bf16_to_float(r);
        std::uint32_t ub;
        std::memcpy(&ub, &back, sizeof ub);
        if ((ub >> 16) != r || (ub & 0xffffu) != 0) return 3;
        std::printf("%04x %04x\n", r, t);
    }
    return 0;
}
