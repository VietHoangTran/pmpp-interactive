// Driver for ../s111_sequential.h. app/src/sim/scan.test.ts compiles this file with the
// host C++ compiler, runs it on the lesson's rows and compares every float, bit for bit,
// with the model behind section 11.1 (app/src/sim/scan.ts, float32 in loop order).
//
//   s111_sequential_test x0 x1 ...
//   -> four lines of hexadecimal float bits, one value per input position:
//      inclusive    sequential_scan
//      exclusive    sequential_exclusive_scan
//      to-excl      inclusive_to_exclusive applied to the inclusive line
//      to-incl      exclusive_to_inclusive applied to the exclusive line
#include "../s111_sequential.h"

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

static void print(const char* name, const std::vector<float>& v) {
    std::printf("%s", name);
    for (float f : v) std::printf(" %08x", bits(f));
    std::printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const unsigned n = unsigned(argc - 1);
    std::vector<float> x(n), inc(n), exc(n), toExc(n), toInc(n);
    for (unsigned k = 0; k < n; ++k) x[k] = std::strtof(argv[k + 1], nullptr);
    sequential_scan(x.data(), inc.data(), n);
    sequential_exclusive_scan(x.data(), exc.data(), n);
    inclusive_to_exclusive(inc.data(), toExc.data(), n);
    exclusive_to_inclusive(exc.data(), x.data(), toInc.data(), n);
    print("inclusive", inc);
    print("exclusive", exc);
    print("to-excl", toExc);
    print("to-incl", toInc);
    return 0;
}
