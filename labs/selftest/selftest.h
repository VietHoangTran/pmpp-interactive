// Interface between a self-test .cu file and the shared driver (main.cpp).
// A .cu file defines its kernels, the host code that launches them and checks
// the results, and this table. A case returns "" on success, else what went wrong.
#pragma once
#include <string>

struct pmpp_selftest_case {
  const char* name;
  std::string (*run)();
};
extern const pmpp_selftest_case pmpp_selftest_cases[];
extern const int pmpp_selftest_count;
