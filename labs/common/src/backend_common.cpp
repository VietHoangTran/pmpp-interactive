#include <cstdlib>
#include "pmpp/backend.hpp"
namespace pmpp {
double device_prop(const std::string& key, double fallback) {
  for (auto& kv : device_props()) if (kv.first == key) return std::atof(kv.second.c_str());
  return fallback;
}
}  // namespace pmpp
