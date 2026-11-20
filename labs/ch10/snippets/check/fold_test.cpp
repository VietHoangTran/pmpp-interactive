// Driver for ../fold.h. The unit test app/src/sim/reducetree.test.ts compiles this
// file, runs it with the lesson's inputs and compares every float bit for bit with the
// model behind section 10.1's explorer (app/src/sim/reducetree.ts).
//
//   fold_test trace  x0 x1 ...   float sum from 0: the accumulator's bits after each iteration
//   fold_test max    x0 x1 ...   max_logit: bits of the result
//   fold_test max0   x0 x1 ...   the same loop started at 0.0f instead of -FLT_MAX: bits
//   fold_test argmax x0 x1 ...   argmax_row: the index
#include "../fold.h"

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
    if (argc < 3) return 2;
    std::vector<float> x;
    for (int k = 2; k < argc; ++k) x.push_back(std::strtof(argv[k], nullptr));
    const char* cmd = argv[1];
    if (!std::strcmp(cmd, "trace")) {
        float acc = 0.0f;
        for (float v : x) {
            acc = reduce_seq(&v, 1, acc, [](float a, float b) { return a + b; });
            std::printf("%08x\n", bits(acc));
        }
        // The whole loop at once must end on the same bits.
        float all = reduce_seq(x.data(), x.size(), 0.0f, [](float a, float b) { return a + b; });
        std::printf("%08x\n", bits(all));
    } else if (!std::strcmp(cmd, "max")) {
        std::printf("%08x\n", bits(max_logit(x.data(), x.size())));
    } else if (!std::strcmp(cmd, "max0")) {
        float r = reduce_seq(x.data(), x.size(), 0.0f, [](float acc, float v) { return v > acc ? v : acc; });
        std::printf("%08x\n", bits(r));
    } else if (!std::strcmp(cmd, "argmax")) {
        std::printf("%zu\n", argmax_row(x.data(), x.size()).index);
    } else {
        return 2;
    }
    return 0;
}
