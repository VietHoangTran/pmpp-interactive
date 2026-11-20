# Lab 20.5, provided: lab 20.4's reference solution, which this lab needs twice:
#   - its per-head decode kernel, the baseline the benchmark times your grouped kernel against;
#   - its KV append, which the 16-step loop calls before your kernel.
# Included by ../CMakeLists.txt before pmpp_add_lab; call pmpp_ch20_perhead_link() after it.
# Modelled on labs/ch15/lab6_linear/provided/baselines.cmake.
#
# The CUDA source and the Metal host source are copied into the build directory at
# configure time with their names changed, so that they link beside your kernel:
#   launchDecodeAttention  -> launchDecodeAttentionPerHead
#   decodeAttentionKernel  -> decodeAttentionPerHeadKernel
#   encodeDecodeAttention  -> encodeDecodeAttentionPerHead
# launchKvAppend, kvAppendKernel and encodeKvAppend keep their names (your files define
# no append). Every one-line __device__ helper definition is made static, so that your
# kernel.cu may have its own laneIdx or warpSumAll. The Metal kernels are compiled at run
# time from lab 20.4's own kernel.metal (`decode_attention`, `kv_append`).
# Sets pmpp_ch20_perhead_cu (SHARED_CU) and pmpp_ch20_perhead_mm (SHARED_MM). If lab 20.4's
# solution is missing, both name files that do not exist and pmpp_add_lab skips the lab
# with a note instead of breaking every other lab.

get_filename_component(_ch20_l4 ${CMAKE_CURRENT_LIST_DIR}/../../lab4_decode/solution ABSOLUTE)

# Writes `content` to `path` only if it differs, so a reconfigure does not force a rebuild.
function(_pmpp_ch20_write path content)
  if(EXISTS ${path})
    file(READ ${path} old)
    if(old STREQUAL content)
      return()
    endif()
  endif()
  file(WRITE ${path} "${content}")
endfunction()

if(EXISTS ${_ch20_l4}/kernel.cu AND EXISTS ${_ch20_l4}/launch.mm AND EXISTS ${_ch20_l4}/kernel.metal)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_ch20_l4}/kernel.cu ${_ch20_l4}/launch.mm)
  set(_note "// Generated at configure time by labs/ch20/lab5_gqa_decode/provided/perhead.cmake\n// from labs/ch20/lab4_decode/solution, with its names changed. Do not edit.\n")
  file(READ ${_ch20_l4}/kernel.cu _src)
  string(REPLACE "launchDecodeAttention" "launchDecodeAttentionPerHead" _src "${_src}")
  string(REPLACE "decodeAttentionKernel" "decodeAttentionPerHeadKernel" _src "${_src}")
  string(REGEX REPLACE "\n__device__ ([^\n;]*\\)[ \t]*\\{)" "\nstatic __device__ \\1" _src "${_src}")
  _pmpp_ch20_write(${CMAKE_CURRENT_BINARY_DIR}/perhead_decode.cu "${_note}${_src}")
  file(READ ${_ch20_l4}/launch.mm _src)
  string(REPLACE "encodeDecodeAttention" "encodeDecodeAttentionPerHead" _src "${_src}")
  _pmpp_ch20_write(${CMAKE_CURRENT_BINARY_DIR}/perhead_decode_launch.mm "${_note}${_src}")
  set(pmpp_ch20_perhead_cu ${CMAKE_CURRENT_BINARY_DIR}/perhead_decode.cu)
  set(pmpp_ch20_perhead_mm ${CMAKE_CURRENT_BINARY_DIR}/perhead_decode_launch.mm)
else()
  message(STATUS "pmpp lab ch20/lab5_gqa_decode: lab 20.4's solution not found in ${_ch20_l4}")
  set(pmpp_ch20_perhead_cu ${_ch20_l4}/kernel.cu)
  set(pmpp_ch20_perhead_mm ${_ch20_l4}/launch.mm)
endif()

# Tells every Metal target of lab `base` (ch20_lab5_gqa_decode) where lab 20.4's kernels are.
function(pmpp_ch20_perhead_link base)
  foreach(v work solution)
    set(tgt ${base}.${v}.metal)
    if(TARGET ${tgt})
      target_compile_definitions(${tgt} PRIVATE PMPP_CH20_PERHEAD_METAL="${_ch20_l4}/kernel.metal")
    endif()
  endforeach()
endfunction()
