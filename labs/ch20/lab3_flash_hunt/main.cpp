// Lab 20.3 driver: lab 20.2's tests (../shared-g11/flash_test.hpp: the same shapes, data,
// CPU reference and tolerances, with the emulators' sizes kept small) run on a flash
// attention kernel that ships with three defects. You do not need to edit this file; your
// work goes in work/.
#include "../shared-g11/flash_test.hpp"

static int tests(const pmpp::Args& a) { return flash::runTests(a); }

static int bench(const pmpp::Args&) {
  std::printf("this lab has no benchmark: lab 20.2 times the same kernel.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
