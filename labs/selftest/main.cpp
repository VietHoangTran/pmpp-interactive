// Driver shared by every emulator self-test: runs the cases a .cu file lists.
#include "pmpp/harness.hpp"
#include "selftest.h"

int main(int argc, char** argv) {
  return pmpp::lab_main(argc, argv,
      [](const pmpp::Args& a) {
        pmpp::Tests t(a);
        for (int i = 0; i < pmpp_selftest_count; ++i)
          t.check(pmpp_selftest_cases[i].name, [i] { return pmpp_selftest_cases[i].run(); });
        return t.finish();
      },
      [](const pmpp::Args&) { return 0; });
}
