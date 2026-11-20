// Shared by the drivers in this directory (see ../check.py). Plain C++: the drivers are
// .cu files that also pass the CUDA front-end check, and on the course's emulator
// (PMPP_EMU, defined by pmpp/emu/cuda_emu.h) they run every case under the three thread
// orders the labs use. The same helper as labs/ch13/snippets/check/check.h.
#pragma once
#include <cstdio>

static int g_failures = 0;

static void expect(bool ok, const char* what) {
    std::printf("  %s: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_failures;
}

static int finish() {
    std::printf("%s\n", g_failures ? "FAILED" : "PASSED");
    return g_failures ? 1 : 0;
}

#if defined(PMPP_EMU)
static const int kOrders = 3;
// The same three orders, and the same shuffle seed, as labs/common/src/backend_cudalike.cpp.
static const char* useOrder(int i) {
    using pmpp_emu::Schedule;
    pmpp_emu::set_schedule(i == 0 ? Schedule::Forward : i == 1 ? Schedule::Reverse : Schedule::Shuffle, 7);
    return i == 0 ? "threads in order" : i == 1 ? "threads reversed" : "threads shuffled";
}
#else
static const int kOrders = 1;
static const char* useOrder(int) { return "device"; }
#endif
