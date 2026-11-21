// pmpp CPU emulator, part 4 of 4: the host-side runtime API.
// Included by pmpp/emu/cuda_emu.h; do not include directly.
//
// Prototypes follow the CUDA Runtime API reference 13.5 (checked 2026-11-20).
// Everything is synchronous: a call has finished when it returns.
#pragma once
#include <chrono>

struct pmpp_emu_stream { int magic = 0x57AE; };
struct pmpp_emu_event {
  std::chrono::steady_clock::time_point at;
  bool recorded = false;
  unsigned flags = 0;
};

// ---- memory ---------------------------------------------------------------------
// cudaMalloc, cudaMallocHost and cudaMallocManaged come in the C form taking void**
// and the toolkit's templated T** form, so both `cudaMalloc((void**)&p, n)` and
// `cudaMalloc(&p, n)` compile, as they do with nvcc (ch02-facts A2).
inline cudaError_t cudaMalloc(void** devPtr, size_t size) {
  *devPtr = std::malloc(size ? size : 1);
  if (*devPtr) std::memset(*devPtr, 0xA5, size);            // device memory is not zeroed
  return *devPtr ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFree(void* devPtr) { std::free(devPtr); return cudaSuccess; }
inline cudaError_t cudaMallocHost(void** ptr, size_t size) { return cudaMalloc(ptr, size); }
inline cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int /*flags*/) { return cudaMalloc(pHost, size); }
inline cudaError_t cudaFreeHost(void* ptr) { std::free(ptr); return cudaSuccess; }
inline cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int /*flags*/ = cudaMemAttachGlobal) {
  return cudaMalloc(devPtr, size);
}
template <class T> inline cudaError_t cudaMalloc(T** devPtr, size_t size) {
  return cudaMalloc(reinterpret_cast<void**>(devPtr), size);
}
template <class T> inline cudaError_t cudaMallocHost(T** ptr, size_t size, unsigned int /*flags*/ = 0) {
  return cudaMalloc(reinterpret_cast<void**>(ptr), size);
}
template <class T> inline cudaError_t cudaMallocManaged(T** devPtr, size_t size, unsigned int /*flags*/ = cudaMemAttachGlobal) {
  return cudaMalloc(reinterpret_cast<void**>(devPtr), size);
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind) {
  std::memcpy(dst, src, count);
  return cudaSuccess;
}
inline cudaError_t cudaMemset(void* devPtr, int value, size_t count) { std::memset(devPtr, value, count); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t = nullptr) {
  return cudaMemcpy(dst, src, count, kind);
}
inline cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t = nullptr) {
  return cudaMemset(devPtr, value, count);
}

// ---- constant memory --------------------------------------------------------------
// A __constant__ variable is an ordinary global in the emulator, so copying to
// the symbol is a memcpy into it.
template <class T>
inline cudaError_t cudaMemcpyToSymbol(const T& symbol, const void* src, size_t count, size_t offset = 0,
                                      cudaMemcpyKind = cudaMemcpyHostToDevice) {
  if (offset + count > sizeof(T)) return ::pmpp_emu::launch_error(cudaErrorInvalidValue,
      ::pmpp_emu::fmt("cudaMemcpyToSymbol: offset %zu + count %zu exceeds the symbol's %zu bytes", offset, count, sizeof(T)));
  std::memcpy(reinterpret_cast<char*>(const_cast<std::remove_const_t<T>*>(&symbol)) + offset, src, count);
  return cudaSuccess;
}
template <class T>
inline cudaError_t cudaMemcpyFromSymbol(void* dst, const T& symbol, size_t count, size_t offset = 0,
                                        cudaMemcpyKind = cudaMemcpyDeviceToHost) {
  if (offset + count > sizeof(T)) return ::pmpp_emu::launch_error(cudaErrorInvalidValue,
      ::pmpp_emu::fmt("cudaMemcpyFromSymbol: offset %zu + count %zu exceeds the symbol's %zu bytes", offset, count, sizeof(T)));
  std::memcpy(dst, reinterpret_cast<const char*>(&symbol) + offset, count);
  return cudaSuccess;
}
template <class T>
inline cudaError_t cudaGetSymbolAddress(void** devPtr, const T& symbol) {
  *devPtr = const_cast<void*>(static_cast<const void*>(&symbol));
  return cudaSuccess;
}

// ---- errors -----------------------------------------------------------------------
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { auto e = ::pmpp_emu::S().lastError; ::pmpp_emu::S().lastError = cudaSuccess; return e; }
inline cudaError_t cudaPeekAtLastError() { return ::pmpp_emu::S().lastError; }
// For a failure the emulator itself detected, the string also carries the
// emulator's explanation (real CUDA returns only the fixed text).
inline const char* cudaGetErrorString(cudaError_t e) {
  const char* base = "error (pmpp emulator)";
  switch (e) {
    case cudaSuccess: return "no error";
    case cudaErrorInvalidValue: base = "invalid argument"; break;
    case cudaErrorMemoryAllocation: base = "out of memory"; break;
    case cudaErrorInvalidConfiguration: base = "invalid configuration argument"; break;
    case cudaErrorInvalidDeviceFunction: base = "invalid device function"; break;
    case cudaErrorInvalidResourceHandle: base = "invalid resource handle"; break;
    case cudaErrorLaunchFailure: base = "unspecified launch failure"; break;
    case cudaErrorCooperativeLaunchTooLarge: base = "too many blocks in cooperative launch"; break;
    case cudaErrorNotSupported: base = "operation not supported"; break;
    default: break;
  }
  ::pmpp_emu::Global& g = ::pmpp_emu::S();
  if (g.messageFor != e || g.message.empty()) return base;
  static std::string text;
  text = std::string(base) + " (pmpp emulator: " + g.message + ")";
  return text.c_str();
}

// ---- device -----------------------------------------------------------------------
inline cudaError_t cudaGetDeviceCount(int* count) { *count = 1; return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int /*device*/) {
  std::memset(prop, 0, sizeof(*prop));
#if defined(PMPP_EMU_RACE)
  std::snprintf(prop->name, sizeof(prop->name), "pmpp CPU emulator, race-detection flavour (not a GPU)");
#else
  std::snprintf(prop->name, sizeof(prop->name), "pmpp CPU emulator (not a GPU)");
#endif
  prop->sharedMemPerBlock = ::pmpp_emu::Limits::sharedMemPerBlock;
  prop->sharedMemPerBlockOptin = ::pmpp_emu::Limits::sharedMemPerBlockOptin;
  prop->sharedMemPerMultiprocessor = ::pmpp_emu::Limits::sharedMemPerBlockOptin;
  prop->maxThreadsPerBlock = ::pmpp_emu::Limits::maxThreadsPerBlock;
  prop->maxThreadsPerMultiProcessor = ::pmpp_emu::Limits::maxThreadsPerBlock;
  prop->maxBlocksPerMultiProcessor = 1;
  prop->multiProcessorCount = 1;
  prop->warpSize = warpSize;
  // The per-dimension limits are CUDA's (Programming Guide 13.5, Table 33), and the emulator
  // enforces them at launch (emu_runtime.h, Limits).
  for (int a = 0; a < 3; ++a) {
    prop->maxThreadsDim[a] = int(::pmpp_emu::Limits::maxBlockDim[a]);
    prop->maxGridSize[a] = int(::pmpp_emu::Limits::maxGridDim[a]);
  }
  prop->cooperativeLaunch = 1;
  prop->unifiedAddressing = 1;
  prop->managedMemory = 1;
  return cudaSuccess;
}
// There is one device, number 0 (Runtime API reference 13.5, group__CUDART__DEVICE).
inline cudaError_t cudaGetDevice(int* device) { *device = 0; return cudaSuccess; }
inline cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int /*device*/) {
  if (attr != cudaDevAttrCooperativeLaunch) return cudaErrorInvalidValue;
  *value = 1;
  return cudaSuccess;
}

// ---- streams and events (nothing is asynchronous here) ------------------------------
inline cudaError_t cudaStreamCreate(cudaStream_t* pStream) { *pStream = new pmpp_emu_stream; return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int /*flags*/) { return cudaStreamCreate(pStream); }
// Stream priorities (Runtime API reference 13.5; infra-facts E7). The emulator has none, so it
// answers as the reference says a device without priorities does: both ends of the range are
// 0, and a requested priority is clamped into that range, so every stream has priority 0.
inline cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  if (leastPriority) *leastPriority = 0;          // NULL is allowed for either pointer
  if (greatestPriority) *greatestPriority = 0;
  return cudaSuccess;
}
inline cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int /*flags*/, int /*priority*/) {
  return cudaStreamCreate(pStream);
}
inline cudaError_t cudaStreamGetPriority(cudaStream_t /*hStream*/, int* priority) { *priority = 0; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  if (!stream) return cudaErrorInvalidResourceHandle;
  delete stream;
  return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamQuery(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
  *event = new pmpp_emu_event;
  (*event)->flags = flags;
  return cudaSuccess;
}
inline cudaError_t cudaEventCreate(cudaEvent_t* event) { return cudaEventCreateWithFlags(event, cudaEventDefault); }
inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t = nullptr) {
  if (!event) return cudaErrorInvalidResourceHandle;
  event->at = std::chrono::steady_clock::now();
  event->recorded = true;
  return cudaSuccess;
}
inline cudaError_t cudaEventQuery(cudaEvent_t event) { return event ? cudaSuccess : cudaErrorInvalidResourceHandle; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t event) { return event ? cudaSuccess : cudaErrorInvalidResourceHandle; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t event, unsigned int /*flags*/ = 0) {
  return event ? cudaSuccess : cudaErrorInvalidResourceHandle;
}
// Host wall-clock time between the two cudaEventRecord calls, in milliseconds.
// It measures the emulator and says nothing about a GPU.
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
  if (!start || !end || !start->recorded || !end->recorded ||
      ((start->flags | end->flags) & cudaEventDisableTiming)) return cudaErrorInvalidResourceHandle;
  *ms = std::chrono::duration<float, std::milli>(end->at - start->at).count();
  return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
  if (!event) return cudaErrorInvalidResourceHandle;
  delete event;
  return cudaSuccess;
}

// ---- kernel attributes, occupancy, cooperative launch -------------------------------
// `func` is the kernel, usually written `(void*)kernel` or just `kernel`.
template <class T>
inline cudaError_t cudaFuncSetAttribute(T* func, cudaFuncAttribute attr, int value) {
  if (attr == cudaFuncAttributeMaxDynamicSharedMemorySize) {
    if (value < 0 || size_t(value) > ::pmpp_emu::Limits::sharedMemPerBlockOptin) return ::pmpp_emu::launch_error(
        cudaErrorInvalidValue, ::pmpp_emu::fmt("cudaFuncSetAttribute: %d bytes of dynamic shared memory exceeds the "
                                               "emulator's opt-in limit of %zu", value, ::pmpp_emu::Limits::sharedMemPerBlockOptin));
    ::pmpp_emu::S().dynSharedOptin[reinterpret_cast<const void*>(func)] = size_t(value);
  }
  return cudaSuccess;   // other attributes are accepted and ignored
}
// cudaFuncGetAttributes, the C form (const void* func) and the C++ template (T* entry), as the
// Runtime API reference 13.5 lists them. SYNTHETIC: the emulator compiles no device code, so
// numRegs, localSizeBytes, sharedSizeBytes, constSizeBytes, binaryVersion, ptxVersion and
// preferredShmemCarveout are 0;
// maxThreadsPerBlock is the emulator's 1024 and maxDynamicSharedSizeBytes what
// cudaFuncSetAttribute set for this kernel, else the 48 KiB default. None of it describes a GPU.
inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) {
  if (!attr || !func) return cudaErrorInvalidValue;
  std::memset(attr, 0, sizeof(*attr));
  attr->maxThreadsPerBlock = int(::pmpp_emu::Limits::maxThreadsPerBlock);
  auto& optin = ::pmpp_emu::S().dynSharedOptin;
  auto it = optin.find(func);
  attr->maxDynamicSharedSizeBytes = int(it != optin.end() ? it->second : ::pmpp_emu::Limits::sharedMemPerBlock);
  return cudaSuccess;
}
template <class T>
inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, T* entry) {
  return cudaFuncGetAttributes(attr, (const void*)entry);
}
// SYNTHETIC: the largest number of blocks of this size that one emulated
// "multiprocessor" (there is exactly one) accepts in a cooperative launch. It
// exists so that the usual grid-sizing code runs; it says nothing about occupancy.
template <class T>
inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, T /*func*/, int blockSize,
                                                                 size_t dynamicSMemSize) {
  if (blockSize < 1 || unsigned(blockSize) > ::pmpp_emu::Limits::maxThreadsPerBlock ||
      dynamicSMemSize > ::pmpp_emu::Limits::sharedMemPerBlockOptin) { *numBlocks = 0; return cudaSuccess; }
  *numBlocks = int(::pmpp_emu::Limits::maxCooperativeThreads / unsigned(blockSize));
  return cudaSuccess;
}
// Every block of the grid is alive at once, so that grid_group::sync() can work.
template <class T>
inline cudaError_t cudaLaunchCooperativeKernel(T* func, dim3 gridSize, dim3 blockSize, void** args,
                                               size_t sharedMem = 0, cudaStream_t = nullptr) {
  auto& table = ::pmpp_emu::S().kernels;
  auto it = table.find(reinterpret_cast<const void*>(func));
  if (it == table.end()) return ::pmpp_emu::launch_error(cudaErrorInvalidDeviceFunction,
      "cudaLaunchCooperativeKernel: the emulator does not know this function. It can launch non-template "
      "__global__ functions defined in a .cu file (labs/tools/cu2emu.py registers them).");
  return it->second(gridSize, blockSize, args, sharedMem);
}
