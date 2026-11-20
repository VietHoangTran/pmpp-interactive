// <cuda_runtime.h>: nvcc includes it implicitly. In emulator builds pmpp/emu/cuda_emu.h, and in the
// front-end check pmpp/check/cuda_stub.h, is force-included and already declares the runtime API.
#pragma once
