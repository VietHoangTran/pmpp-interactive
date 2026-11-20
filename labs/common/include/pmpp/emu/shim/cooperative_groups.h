// Stands in for <cooperative_groups.h> in emulator builds and in the CUDA front-end check.
#pragma once
#if defined(PMPP_CUDA_FRONTEND_CHECK)
#define PMPP_STUB_WANT_COOPERATIVE_GROUPS
#include "pmpp/check/cuda_stub.h"
#else
#include "pmpp/emu/cooperative_groups_emu.h"
#endif
