# Labs for the PMPP interactive textbook.
# Configures every backend this machine can build and skips the rest:
#   emu    CPU emulator of the CUDA subset (always, needs Python 3)
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
# Every chNN/labM_name directory with a CMakeLists.txt is a lab.
file(GLOB pmpp_lab_lists RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS ch*/lab*/CMakeLists.txt)
list(SORT pmpp_lab_lists)
foreach(lst ${pmpp_lab_lists})
  get_filename_component(labdir ${lst} DIRECTORY)
  add_subdirectory(${labdir})
endforeach()
