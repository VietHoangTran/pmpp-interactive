# Labs for the PMPP interactive textbook.
# Configures every backend this machine can build and skips the rest:
#   emu    CPU emulator of the CUDA subset (always, needs Python 3)
#   race   the same emulator with truly concurrent threads under ThreadSanitizer
#          (when the compiler's TSan works here; labs opt in, see cmake/Lab.cmake)
#   metal  Apple GPUs (macOS)
#   cuda   NVIDIA GPUs (needs the CUDA toolkit)
#   hip    AMD GPUs (needs ROCm)
# You normally drive this through ../verify.sh or `python3 labs/pmpp.py`.
cmake_minimum_required(VERSION 3.21)
project(pmpp_labs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(PMPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
include(cmake/Backends.cmake)
include(cmake/Lab.cmake)

add_subdirectory(devinfo)
# Self-tests of the emulator itself (see "The emulator" in README.md).
file(WRITE ${CMAKE_BINARY_DIR}/pmpp_selftests.txt "")
add_subdirectory(selftest)
# Every chNN/labM_name directory with a CMakeLists.txt is a lab.
# (appendix and extra-module labs live under appA, appC, x1, x2, x3.)
# Set PMPP_ONLY=ch06 in the environment to configure one chapter's labs only.
file(WRITE ${CMAKE_BINARY_DIR}/pmpp_incomplete.txt "")
if(DEFINED ENV{PMPP_ONLY})
  file(GLOB pmpp_lab_lists RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS $ENV{PMPP_ONLY}/lab*/CMakeLists.txt)
else()
  file(GLOB pmpp_lab_lists RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS
       ch*/lab*/CMakeLists.txt app*/lab*/CMakeLists.txt x[0-9]*/lab*/CMakeLists.txt)
endif()
list(SORT pmpp_lab_lists)
foreach(lst ${pmpp_lab_lists})
  get_filename_component(labdir ${lst} DIRECTORY)
  add_subdirectory(${labdir})
endforeach()
