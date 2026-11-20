// Driver for ../kahan.h. The unit test app/src/sim/fpsum.test.ts compiles this file with
// the host C++ compiler (ordinary optimisation, no fast math), runs it, and compares every
// float, bit for bit, with the model behind section A.6's bug hunt (kahanVariant and
// sumSequential in app/src/sim/fpsum.ts).
//
//   kahan_check bughunt          1.0f then 65,535 values of 2^-25
//   kahan_check list x0 x1 ...   the given values (decimal or hex floats)
// Output: five lines, "<name> <bits of the float result in hex>", for plain_sum and
// version1 to version4.
#include "../kahan.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static unsigned bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    std::vector<float> x;
    if (!std::strcmp(argv[1], "bughunt")) {
        x.push_back(1.0f);
        for (int i = 1; i < 65536; ++i) x.push_back(0x1p-25f);
    } else if (!std::strcmp(argv[1], "list")) {
        for (int k = 2; k < argc; ++k) x.push_back(std::strtof(argv[k], nullptr));
    } else {
        return 2;
    }
    const float* p = x.data();
    const std::size_t n = x.size();
    std::printf("plain %08x\n", bits(plain_sum(p, n)));
    std::printf("version1 %08x\n", bits(version1(p, n)));
    std::printf("version2 %08x\n", bits(version2(p, n)));
    std::printf("version3 %08x\n", bits(version3(p, n)));
    std::printf("version4 %08x\n", bits(version4(p, n)));
    return 0;
}
