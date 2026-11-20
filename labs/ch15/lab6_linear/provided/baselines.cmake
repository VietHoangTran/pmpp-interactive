# Lab 15.6, provided: the two baselines the benchmark times your M3 kernel against, and
# the vendor libraries they need. Included by ../CMakeLists.txt before pmpp_add_lab; call
# pmpp_ch15_m3_link() after it. Sets:
#   pmpp_ch15_m3_cu   provided CUDA sources (SHARED_CU)
#   pmpp_ch15_m3_mm   provided Metal host sources (SHARED_MM)
#
# 1. M2, lab 6.4's kernel, from that lab's reference solution. Copied into the build
#    directory at configure time with its symbols renamed, so that it can be linked beside
#    your M3 (launchLinear -> launchLinearM2, linearKernel -> linearM2Kernel,
#    encodeLinear -> encodeLinearM2, and its LinearDims struct -> LinearM2Dims). Its Metal
#    kernel is compiled at run time from lab 6.4's own kernel.metal. If lab 6.4's solution
#    is missing, the benchmark has no m2 rows.
# 2. The vendor library: cuBLAS (CUDA::cublas), hipBLAS (roc::hipblas) or Metal
#    Performance Shaders (ch15-facts A10, B6, C6). Where none is found, no vendor rows.
#    -DPMPP_CH15_VENDOR_BLAS=OFF leaves it out on CUDA and HIP (unverified there).
# 3. The occupancy row on CUDA and HIP (provided/occupancy.cu, unverified there);
#    -DPMPP_CH15_OCCUPANCY_API=OFF leaves it out.

option(PMPP_CH15_VENDOR_BLAS "Lab 15.6: time cuBLAS / hipBLAS beside your kernel" ON)
option(PMPP_CH15_OCCUPANCY_API "Lab 15.6: ask the CUDA / HIP runtime for registers and blocks per SM" ON)

set(_pmpp_m3_provided ${CMAKE_CURRENT_LIST_DIR})
get_filename_component(_pmpp_m2_dir ${CMAKE_CURRENT_LIST_DIR}/../../../ch06/lab4_linear/solution ABSOLUTE)

set(pmpp_ch15_m3_cu ${_pmpp_m3_provided}/vendor_linear.cu ${_pmpp_m3_provided}/occupancy.cu)
set(pmpp_ch15_m3_mm ${_pmpp_m3_provided}/vendor_linear_mps.mm)
set(_pmpp_m3_defs "")
set(_pmpp_m3_metal_defs "")
if(PMPP_CH15_OCCUPANCY_API)
  list(APPEND _pmpp_m3_defs PMPP_CH15_OCCUPANCY_API=1)
endif()

# Writes `content` to `path` only if it differs, so a reconfigure does not force a rebuild.
function(_pmpp_m3_write path content)
  if(EXISTS ${path})
    file(READ ${path} old)
    if(old STREQUAL content)
      return()
    endif()
  endif()
  file(WRITE ${path} "${content}")
endfunction()

if(EXISTS ${_pmpp_m2_dir}/kernel.cu AND EXISTS ${_pmpp_m2_dir}/launch.mm AND EXISTS ${_pmpp_m2_dir}/kernel.metal)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               ${_pmpp_m2_dir}/kernel.cu ${_pmpp_m2_dir}/launch.mm)
  set(_pmpp_m2_note "// Generated at configure time by labs/ch15/lab6_linear/provided/baselines.cmake\n// from labs/ch06/lab4_linear/solution, with its symbols renamed. Do not edit.\n")
  file(READ ${_pmpp_m2_dir}/kernel.cu _src)
  string(REPLACE "launchLinear" "launchLinearM2" _src "${_src}")
  string(REPLACE "linearKernel" "linearM2Kernel" _src "${_src}")
  _pmpp_m3_write(${CMAKE_CURRENT_BINARY_DIR}/m2_linear.cu "${_pmpp_m2_note}${_src}")
  file(READ ${_pmpp_m2_dir}/launch.mm _src)
  string(REPLACE "encodeLinear" "encodeLinearM2" _src "${_src}")
  string(REPLACE "LinearDims" "LinearM2Dims" _src "${_src}")
  _pmpp_m3_write(${CMAKE_CURRENT_BINARY_DIR}/m2_linear_launch.mm "${_pmpp_m2_note}${_src}")
  list(APPEND pmpp_ch15_m3_cu ${CMAKE_CURRENT_BINARY_DIR}/m2_linear.cu)
  list(APPEND pmpp_ch15_m3_mm ${CMAKE_CURRENT_BINARY_DIR}/m2_linear_launch.mm)
  list(APPEND _pmpp_m3_defs PMPP_HAVE_M2=1)
  list(APPEND _pmpp_m3_metal_defs PMPP_M2_METAL="${_pmpp_m2_dir}/kernel.metal")
else()
  message(STATUS "pmpp lab ch15/lab6_linear: lab 6.4's solution not found in ${_pmpp_m2_dir}; "
                 "the benchmark will have no m2 rows")
endif()

if(PMPP_CH15_VENDOR_BLAS AND "hip" IN_LIST PMPP_BACKENDS)
  find_package(hipblas CONFIG QUIET)
endif()

# Adds the vendor libraries and the definitions above to every target of lab `base`
# (e.g. ch15_lab6_linear) that pmpp_add_lab created.
function(pmpp_ch15_m3_link base)
  foreach(v work solution)
    foreach(b ${PMPP_BACKENDS})
      set(tgt ${base}.${v}.${b})
      if(NOT TARGET ${tgt})
        continue()
      endif()
      if(_pmpp_m3_defs)
        target_compile_definitions(${tgt} PRIVATE ${_pmpp_m3_defs})
      endif()
      if(b STREQUAL "metal")
        target_link_libraries(${tgt} PRIVATE "-framework MetalPerformanceShaders")
        if(_pmpp_m3_metal_defs)
          target_compile_definitions(${tgt} PRIVATE ${_pmpp_m3_metal_defs})
        endif()
      elseif(b STREQUAL "cuda" AND PMPP_CH15_VENDOR_BLAS AND TARGET CUDA::cublas)
        target_link_libraries(${tgt} PRIVATE CUDA::cublas)
        target_compile_definitions(${tgt} PRIVATE PMPP_HAVE_VENDOR_BLAS=1)
      elseif(b STREQUAL "hip" AND PMPP_CH15_VENDOR_BLAS AND TARGET roc::hipblas)
        target_link_libraries(${tgt} PRIVATE roc::hipblas)
        target_compile_definitions(${tgt} PRIVATE PMPP_HAVE_VENDOR_BLAS=1)
      endif()
    endforeach()
  endforeach()
endfunction()
