# Labs 20.1 and 20.2, provided: the kernels of the unfused attention pipeline that the two
# benchmarks time. Included by a lab's CMakeLists.txt before pmpp_add_lab; call
# pmpp_ch20_unfused_link(<base>) after it. Modelled on
# labs/ch15/lab6_linear/provided/baselines.cmake.
#
#   S = Q K^T   milestone M3: lab 15.6's reference launchLinear (Y = X W^T; X = Q, W = K)
#   P           your softmax (lab 20.1), or lab 20.1's reference (lab 20.2's benchmark)
#   O = P V     milestone M2: lab 5.3's reference launchGemm (C = A B), tile 16
#
# Each is copied from that lab's solution/ into the build directory at configure time with
# its symbols renamed, so that it links beside the lab's own kernel:
#   M3          launchLinear -> launchLinearM3, fastLinearKernel -> fastLinearM3Kernel,
#               encodeLinear -> encodeLinearM3, LinearDims -> LinearM3Dims
#   M2          launchGemm -> launchGemmM2, gemmKernel -> gemmM2Kernel,
#               encodeGemm -> encodeGemmM2, GemmDims -> GemmM2Dims, and TILE_WIDTH ->
#               PMPP_M2_TILE (left at the file's default, 16, whatever TILE_WIDTH the lab is
#               built with)
#   softmax ref launchCausalSoftmax -> launchCausalSoftmaxRef, causalSoftmaxKernel ->
#               causalSoftmaxRefKernel, encodeCausalSoftmax -> encodeCausalSoftmaxRef,
#               CausalDims -> CausalRefDims; its M4 helpers get a ch20ref_ prefix through
#               #define lines, so that they cannot clash with names in your kernel.cu.
#               Only when PMPP_CH20_SOFTMAX_REF is ON before the include (lab 20.2).
# The Metal kernels are compiled at run time from those labs' own kernel.metal files.
# A copy whose source lab is missing is left out, and the benchmark has no rows for it.
#
# Sets:
#   pmpp_ch20_unfused_cu   provided CUDA sources (SHARED_CU)
#   pmpp_ch20_unfused_mm   provided Metal host sources (SHARED_MM)
# pmpp_ch20_unfused_link(<base>) adds to every target of the lab:
#   PMPP_CH20_HAVE_M3, PMPP_CH20_HAVE_M2, PMPP_CH20_HAVE_SOFTMAX_REF (for the copies made)
#   PMPP_CH20_M3_METAL, PMPP_CH20_M2_METAL, PMPP_CH20_SOFTMAX_REF_METAL (Metal only: the
#   kernel.metal paths)

set(pmpp_ch20_unfused_cu "")
set(pmpp_ch20_unfused_mm "")
set(_pmpp_ch20_defs "")
set(_pmpp_ch20_metal_defs "")

get_filename_component(_pmpp_ch20_m3_dir ${CMAKE_CURRENT_LIST_DIR}/../../ch15/lab6_linear/solution ABSOLUTE)
get_filename_component(_pmpp_ch20_m2_dir ${CMAKE_CURRENT_LIST_DIR}/../../ch05/lab3_gemm/solution ABSOLUTE)
get_filename_component(_pmpp_ch20_sm_dir ${CMAKE_CURRENT_LIST_DIR}/../lab1_causal_softmax/solution ABSOLUTE)

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

# Copies <dir>/kernel.cu and <dir>/launch.mm to <stem>.cu and <stem>_launch.mm in the
# build directory, applying the (from, to) pairs in `pairs` to both and prepending
# `prefix` to the .cu file. Sets <ok_var> to TRUE when all three source files exist.
function(_pmpp_ch20_copy dir stem what prefix ok_var)
  set(pairs ${ARGN})
  if(NOT (EXISTS ${dir}/kernel.cu AND EXISTS ${dir}/launch.mm AND EXISTS ${dir}/kernel.metal))
    message(STATUS "pmpp ch20 unfused pipeline: ${what} not found in ${dir}; the benchmark has no rows for it")
    set(${ok_var} FALSE PARENT_SCOPE)
    return()
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${dir}/kernel.cu ${dir}/launch.mm)
  set(note "// Generated at configure time by labs/ch20/shared-g11/unfused.cmake from\n// ${what}, with its symbols renamed. Do not edit.\n")
  foreach(f kernel.cu launch.mm)
    file(READ ${dir}/${f} src)
    set(p ${pairs})
    while(p)
      list(POP_FRONT p from to)
      string(REPLACE "${from}" "${to}" src "${src}")
    endwhile()
    if(f STREQUAL "kernel.cu")
      _pmpp_ch20_write(${CMAKE_CURRENT_BINARY_DIR}/${stem}.cu "${note}${prefix}${src}")
    else()
      _pmpp_ch20_write(${CMAKE_CURRENT_BINARY_DIR}/${stem}_launch.mm "${note}${src}")
    endif()
  endforeach()
  set(${ok_var} TRUE PARENT_SCOPE)
endfunction()

# M3, lab 15.6. ("launchLinear" also renames launchLinearWidth, to launchLinearM3Width.)
_pmpp_ch20_copy(${_pmpp_ch20_m3_dir} m3_linear "lab 15.6's solution (M3)" "" _ok
  launchLinear launchLinearM3  fastLinearKernel fastLinearM3Kernel
  encodeLinear encodeLinearM3  LinearDims LinearM3Dims)
if(_ok)
  list(APPEND pmpp_ch20_unfused_cu ${CMAKE_CURRENT_BINARY_DIR}/m3_linear.cu)
  list(APPEND pmpp_ch20_unfused_mm ${CMAKE_CURRENT_BINARY_DIR}/m3_linear_launch.mm)
  list(APPEND _pmpp_ch20_defs PMPP_CH20_HAVE_M3=1)
  list(APPEND _pmpp_ch20_metal_defs PMPP_CH20_M3_METAL="${_pmpp_ch20_m3_dir}/kernel.metal")
endif()

# M2, lab 5.3.
_pmpp_ch20_copy(${_pmpp_ch20_m2_dir} m2_gemm "lab 5.3's solution (M2)" "" _ok
  launchGemm launchGemmM2  gemmKernel gemmM2Kernel
  encodeGemm encodeGemmM2  GemmDims GemmM2Dims  TILE_WIDTH PMPP_M2_TILE)
if(_ok)
  list(APPEND pmpp_ch20_unfused_cu ${CMAKE_CURRENT_BINARY_DIR}/m2_gemm.cu)
  list(APPEND pmpp_ch20_unfused_mm ${CMAKE_CURRENT_BINARY_DIR}/m2_gemm_launch.mm)
  list(APPEND _pmpp_ch20_defs PMPP_CH20_HAVE_M2=1)
  list(APPEND _pmpp_ch20_metal_defs PMPP_CH20_M2_METAL="${_pmpp_ch20_m2_dir}/kernel.metal")
endif()

# Lab 20.1's reference softmax, for lab 20.2's benchmark.
if(PMPP_CH20_SOFTMAX_REF)
  set(_prefix "")
  foreach(n warpIdx laneIdx warpsInBlock maxOp warpReduceMax warpReduceSum blockReduceMax blockReduceSum)
    string(APPEND _prefix "#define ${n} ch20ref_${n}\n")
  endforeach()
  _pmpp_ch20_copy(${_pmpp_ch20_sm_dir} softmax_ref "lab 20.1's solution (M6 part one)" "${_prefix}" _ok
    launchCausalSoftmax launchCausalSoftmaxRef  causalSoftmaxKernel causalSoftmaxRefKernel
    encodeCausalSoftmax encodeCausalSoftmaxRef  CausalDims CausalRefDims)
  if(_ok)
    list(APPEND pmpp_ch20_unfused_cu ${CMAKE_CURRENT_BINARY_DIR}/softmax_ref.cu)
    list(APPEND pmpp_ch20_unfused_mm ${CMAKE_CURRENT_BINARY_DIR}/softmax_ref_launch.mm)
    list(APPEND _pmpp_ch20_defs PMPP_CH20_HAVE_SOFTMAX_REF=1)
    list(APPEND _pmpp_ch20_metal_defs PMPP_CH20_SOFTMAX_REF_METAL="${_pmpp_ch20_sm_dir}/kernel.metal")
  endif()
endif()

# Adds the definitions above to every target of lab `base` (e.g. ch20_lab1_causal_softmax)
# that pmpp_add_lab created, extra tile widths included.
function(pmpp_ch20_unfused_link base)
  foreach(v work solution)
    foreach(b ${PMPP_BACKENDS})
      foreach(suffix "" ${ARGN})
        set(tgt ${base}.${v}.${b}${suffix})
        if(NOT TARGET ${tgt})
          continue()
        endif()
        if(_pmpp_ch20_defs)
          target_compile_definitions(${tgt} PRIVATE ${_pmpp_ch20_defs})
        endif()
        if(b STREQUAL "metal" AND _pmpp_ch20_metal_defs)
          target_compile_definitions(${tgt} PRIVATE ${_pmpp_ch20_metal_defs})
        endif()
      endforeach()
    endforeach()
  endforeach()
endfunction()
