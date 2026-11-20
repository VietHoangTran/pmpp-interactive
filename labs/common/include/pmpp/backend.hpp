// Backend-neutral runtime used by every lab's main.cpp. One implementation per
// backend lives in labs/common/src. Student code never needs this header: it is
// written against the real API of its backend (CUDA, or Metal).
#pragma once
#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace pmpp {

struct Buf;  // a buffer in device memory

// "cuda", "hip", "emu" (CPU emulator of the CUDA subset) or "metal".
const char* backend();

// False when the backend was built but no usable device is present.
bool device_available(std::string* why);

// Properties queried from the device at run time, as (key, value) strings.
// Nothing in the course hard-codes these.
std::vector<std::pair<std::string, std::string>> device_props();
// Convenience: numeric property, or `fallback` if absent.
double device_prop(const std::string& key, double fallback);

Buf*  alloc(size_t bytes);
void  upload(Buf*, const void* src, size_t bytes);
void  download(Buf*, void* dst, size_t bytes);
void  release(Buf*);
// cuda/hip/emu: the device pointer. metal: the id<MTLBuffer>, bridged.
void* raw(Buf*);

// The emulator runs each test under several deterministic thread schedules;
// real devices report 1.
int  schedules();
void use_schedule(int i);
const char* schedule_name(int i);

struct Timing {
  double wall_s;  // host wall-clock around submit + wait
  double gpu_s;   // device-reported execution time, or < 0 if the backend has none
  double best() const { return gpu_s >= 0 ? gpu_s : wall_s; }
};

// Calls `enqueue` (which must launch the kernel), waits for it to finish, and
// returns how long it took. `pass` is nullptr except on Metal, where it points
// to a pmpp::metal::Pass (see pmpp/metal.h).
Timing run(const std::function<void(void* pass)>& enqueue);

// Last launch error reported by the backend, or empty.
std::string take_error();

}  // namespace pmpp
