# Detect which backends can be built here. Each one that can becomes a static
# library pmpp_rt_<backend> and an entry in PMPP_BACKENDS. Each one that cannot
# is recorded with the reason in <build>/pmpp_backends.txt.
include(CheckLanguage)

set(PMPP_BACKENDS "")
set(PMPP_BACKEND_LOG ${CMAKE_BINARY_DIR}/pmpp_backends.txt)
file(WRITE ${PMPP_BACKEND_LOG} "")
file(WRITE ${CMAKE_BINARY_DIR}/pmpp_targets.txt "")

function(pmpp_backend_status name ok why)
  if(ok)
    message(STATUS "pmpp backend ${name}: enabled (${why})")
    file(APPEND ${PMPP_BACKEND_LOG} "${name}|enabled|${why}\n")
  else()
    message(STATUS "pmpp backend ${name}: skipped (${why})")
    file(APPEND ${PMPP_BACKEND_LOG} "${name}|skipped|${why}\n")
  endif()
endfunction()

set(PMPP_RT_COMMON ${PMPP_ROOT}/common/src/backend_common.cpp)
set(PMPP_INC ${PMPP_ROOT}/common/include)

# ---------------- emu ----------------
find_package(Python3 COMPONENTS Interpreter)
find_package(Threads)
if(Python3_FOUND AND Threads_FOUND)
  add_library(pmpp_rt_emu STATIC ${PMPP_ROOT}/common/src/backend_cudalike.cpp ${PMPP_RT_COMMON})
  # The second directory holds stand-ins for <cuda/atomic>, <cooperative_groups.h>
  # and <cuda_runtime.h>, so that the same #include lines work as under nvcc.
  target_include_directories(pmpp_rt_emu PUBLIC ${PMPP_INC} ${PMPP_INC}/pmpp/emu/shim)
  target_compile_definitions(pmpp_rt_emu PUBLIC PMPP_BACKEND_EMU)
  target_link_libraries(pmpp_rt_emu PUBLIC Threads::Threads)
  # AddressSanitizer turns an out-of-bounds read or write in a kernel (device
  # buffers are plain heap blocks in the emulator, __shared__ arrays are statics)
  # into an immediate report with the source line, instead of a silent wrong value.
  option(PMPP_EMU_SANITIZE "Build emulator targets with AddressSanitizer" ON)
  set(pmpp_emu_note "CPU emulator; python ${Python3_VERSION}")
  if(PMPP_EMU_SANITIZE)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")
    set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=address")
    check_cxx_source_compiles("int main() { return 0; }" PMPP_HAVE_ASAN)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_LINK_OPTIONS)
    if(PMPP_HAVE_ASAN)
      target_compile_options(pmpp_rt_emu PUBLIC -fsanitize=address -fno-omit-frame-pointer -g)
      target_link_options(pmpp_rt_emu PUBLIC -fsanitize=address)
      set(pmpp_emu_note "${pmpp_emu_note}; AddressSanitizer on")
    else()
      set(pmpp_emu_note "${pmpp_emu_note}; AddressSanitizer unavailable, out-of-bounds accesses are not detected")
    endif()
  endif()
  list(APPEND PMPP_BACKENDS emu)
  pmpp_backend_status(emu TRUE "${pmpp_emu_note}")
else()
  pmpp_backend_status(emu FALSE "needs python3 and a threads library")
endif()

# ---------------- race ----------------
# The emulator in its race-detection flavour: emulated threads run concurrently
# under ThreadSanitizer. Needs a compiler whose TSan runtime works on this
# machine and exports the dynamic annotations the emulator uses to keep its own
# bookkeeping out of the analysis. Labs opt in with `pmpp_add_lab(... RACE)`.
if(PMPP_DISABLE_RACE)
  pmpp_backend_status(race FALSE "disabled with PMPP_DISABLE_RACE")
elseif(NOT TARGET pmpp_rt_emu)
  pmpp_backend_status(race FALSE "needs the emu backend")
else()
  include(CheckCXXSourceCompiles)
  include(CheckCXXSourceRuns)
  set(pmpp_tsan_probe "
    #include <thread>
    extern \"C\" {
    void AnnotateIgnoreSyncBegin(const char*, int); void AnnotateIgnoreSyncEnd(const char*, int);
    void AnnotateIgnoreReadsBegin(const char*, int); void AnnotateIgnoreReadsEnd(const char*, int);
    void AnnotateIgnoreWritesBegin(const char*, int); void AnnotateIgnoreWritesEnd(const char*, int);
    void AnnotateHappensBefore(const char*, int, const volatile void*);
    void AnnotateHappensAfter(const char*, int, const volatile void*);
    }
    int main() {
      int token = 0, x = 0;
      std::thread t([&] {
        AnnotateIgnoreSyncBegin(\"\", 0); AnnotateIgnoreReadsBegin(\"\", 0); AnnotateIgnoreWritesBegin(\"\", 0);
        x = 1;
        AnnotateIgnoreWritesEnd(\"\", 0); AnnotateIgnoreReadsEnd(\"\", 0); AnnotateIgnoreSyncEnd(\"\", 0);
        AnnotateHappensBefore(\"\", 0, &token);
      });
      t.join();
      AnnotateHappensAfter(\"\", 0, &token);
      return x == 1 ? 0 : 1;
    }")
  set(CMAKE_REQUIRED_FLAGS "-fsanitize=thread")
  set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=thread")
  set(CMAKE_REQUIRED_LIBRARIES Threads::Threads)
  if(CMAKE_CROSSCOMPILING)
    check_cxx_source_compiles("${pmpp_tsan_probe}" PMPP_HAVE_TSAN)
  else()
    check_cxx_source_runs("${pmpp_tsan_probe}" PMPP_HAVE_TSAN)
  endif()
  unset(CMAKE_REQUIRED_FLAGS)
  unset(CMAKE_REQUIRED_LINK_OPTIONS)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(PMPP_HAVE_TSAN)
    add_library(pmpp_rt_race STATIC ${PMPP_INC}/pmpp/emu/rt/backend_race.cpp ${PMPP_RT_COMMON})
    target_include_directories(pmpp_rt_race PUBLIC ${PMPP_INC} ${PMPP_INC}/pmpp/emu/shim)
    target_compile_definitions(pmpp_rt_race PUBLIC PMPP_BACKEND_RACE)
    target_compile_options(pmpp_rt_race PUBLIC -fsanitize=thread -fno-omit-frame-pointer -g)
    target_link_options(pmpp_rt_race PUBLIC -fsanitize=thread)
    target_link_libraries(pmpp_rt_race PUBLIC Threads::Threads)
    list(APPEND PMPP_BACKENDS race)
    pmpp_backend_status(race TRUE "CPU emulator with concurrent threads; ThreadSanitizer on (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
  else()
    pmpp_backend_status(race FALSE "this compiler's ThreadSanitizer (-fsanitize=thread) does not build and run a test program here")
  endif()
endif()

# ---------------- metal ----------------
if(APPLE AND NOT PMPP_DISABLE_METAL)
  check_language(OBJCXX)
  if(CMAKE_OBJCXX_COMPILER)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    add_library(pmpp_rt_metal STATIC ${PMPP_ROOT}/common/src/backend_metal.mm ${PMPP_RT_COMMON})
    target_include_directories(pmpp_rt_metal PUBLIC ${PMPP_INC})
    target_compile_definitions(pmpp_rt_metal PUBLIC PMPP_BACKEND_METAL)
    target_compile_options(pmpp_rt_metal PUBLIC -fobjc-arc)
    target_link_libraries(pmpp_rt_metal PUBLIC "-framework Metal" "-framework Foundation")
    list(APPEND PMPP_BACKENDS metal)
    pmpp_backend_status(metal TRUE "Objective-C++ compiler ${CMAKE_OBJCXX_COMPILER_ID} ${CMAKE_OBJCXX_COMPILER_VERSION}")
  else()
    pmpp_backend_status(metal FALSE "no Objective-C++ compiler")
  endif()
else()
  pmpp_backend_status(metal FALSE "not macOS")
endif()

# ---------------- cuda ----------------
# UNVERIFIED: written on a machine without the CUDA toolkit.
if(NOT PMPP_DISABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
      add_library(pmpp_rt_cuda STATIC ${PMPP_ROOT}/common/src/backend_cudalike.cpp ${PMPP_RT_COMMON})
      target_include_directories(pmpp_rt_cuda PUBLIC ${PMPP_INC})
      target_compile_definitions(pmpp_rt_cuda PUBLIC PMPP_BACKEND_CUDA)
      target_link_libraries(pmpp_rt_cuda PUBLIC CUDA::cudart)
      list(APPEND PMPP_BACKENDS cuda)
      pmpp_backend_status(cuda TRUE "CUDA toolkit ${CUDAToolkit_VERSION}")
    else()
      pmpp_backend_status(cuda FALSE "CUDA compiler found but find_package(CUDAToolkit) failed")
    endif()
  else()
    pmpp_backend_status(cuda FALSE "no CUDA compiler (nvcc) found")
  endif()
else()
  pmpp_backend_status(cuda FALSE "disabled with PMPP_DISABLE_CUDA")
endif()

# ---------------- hip ----------------
# UNVERIFIED: written on a machine without ROCm, from AMD's and CMake's docs
# (see notes/today/ch05-facts.md, items B6 and B7):
#   - CMake's HIP language needs clang (amdclang++), not the legacy hipcc wrapper.
#   - Targets mixing C++ and HIP sources link hip::host only.
#   - ROCm is found through ROCM_PATH, or /opt/rocm where the packages provide it.
if(NOT PMPP_DISABLE_HIP)
  set(pmpp_rocm_hints "")
  if(DEFINED ENV{ROCM_PATH})
    list(APPEND pmpp_rocm_hints $ENV{ROCM_PATH})
  endif()
  if(EXISTS /opt/rocm)
    list(APPEND pmpp_rocm_hints /opt/rocm)
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${pmpp_rocm_hints})
  if(NOT CMAKE_HIP_COMPILER AND NOT DEFINED ENV{HIPCXX})
    set(pmpp_hip_bin_hints "")
    foreach(h ${pmpp_rocm_hints})
      list(APPEND pmpp_hip_bin_hints ${h}/bin ${h}/lib/llvm/bin ${h}/llvm/bin)
    endforeach()
    find_program(PMPP_AMDCLANG NAMES amdclang++ HINTS ${pmpp_hip_bin_hints})
    if(PMPP_AMDCLANG)
      set(CMAKE_HIP_COMPILER ${PMPP_AMDCLANG})
    endif()
  endif()
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 17)
    find_package(hip CONFIG)
    if(hip_FOUND)
      add_library(pmpp_rt_hip STATIC ${PMPP_ROOT}/common/src/backend_cudalike.cpp ${PMPP_RT_COMMON})
      target_include_directories(pmpp_rt_hip PUBLIC ${PMPP_INC})
      target_compile_definitions(pmpp_rt_hip PUBLIC PMPP_BACKEND_HIP)
      target_link_libraries(pmpp_rt_hip PUBLIC hip::host)
      # <cuda/atomic> (cuda::atomic_ref, used from chapter 9 on) comes from libhipcxx,
      # part of the ROCm Core SDK from 10.1; AMD's docs say to link this target
      # (infra-facts B2). Unverified here: no ROCm machine has built it yet.
      find_package(libhipcxx CONFIG QUIET)
      if(TARGET libhipcxx::libhipcxx)
        target_link_libraries(pmpp_rt_hip PUBLIC libhipcxx::libhipcxx)
        set(_pmpp_hipcxx "libhipcxx found")
      else()
        set(_pmpp_hipcxx "libhipcxx NOT found: labs that include <cuda/atomic> will not build")
      endif()
      list(APPEND PMPP_BACKENDS hip)
      pmpp_backend_status(hip TRUE "HIP ${hip_VERSION}; ${_pmpp_hipcxx}")
    else()
      pmpp_backend_status(hip FALSE "HIP compiler found but find_package(hip) failed")
    endif()
  else()
    pmpp_backend_status(hip FALSE "no HIP compiler found (is ROCm installed? set ROCM_PATH)")
  endif()
else()
  pmpp_backend_status(hip FALSE "disabled with PMPP_DISABLE_HIP")
endif()

message(STATUS "pmpp backends to build: ${PMPP_BACKENDS}")
