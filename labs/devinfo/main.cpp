// Prints what the device reports about itself. The course reads these values
// instead of assuming any particular GPU.
#include "pmpp/harness.hpp"
int main() {
  pmpp::emit_device();
  std::string why;
  return pmpp::device_available(&why) ? 0 : 3;
}
