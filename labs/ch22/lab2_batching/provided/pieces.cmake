# Lab 22.2, provided: the project's own pieces, which the decode step is built from.
# Included by ../CMakeLists.txt before pmpp_add_lab; call pmpp_ch22_pieces_link() after it.
# Sets:
#   pmpp_ch22_pieces_cu   provided CUDA sources (SHARED_CU)
#   pmpp_ch22_pieces_mm   provided Metal host sources (SHARED_MM)
#
# Each piece is an earlier lab's reference solution, copied into the build directory at
# configure time (as labs/ch15/lab6_linear/provided/baselines.cmake does), with its names
# changed where two pieces would clash, and with every one-line __device__ helper
# definition made static, so that two pieces may each have, say, a warpReduceSum. Its Metal
# kernels are compiled at run time from that lab's own kernel.metal; the driver passes the
# constants that lab's driver uses.
#
#   lab 2.3  (M0)  launchAddBias, launchAddResidual; encodeAddBias, encodeAddResidual   required
#   lab 6.4  (M2)  launchLinear -> launchLinearM2, encodeLinear -> encodeLinearM2         required
#   lab 10.3 (M4)  launchSoftmaxRows, launchArgmaxRows and their encoders                required
#   lab 15.6 (M3)  launchLinear -> launchLinearM3, encodeLinear -> encodeLinearM3         if present
#   lab 22.1       launchGemvGather, encodeGemvGather                                    if present
#   step_timer.cu  one pair of events around a step (the CUDA and HIP timing)            always
# A required piece that is missing leaves the lab skipped, with a note that names it.
#
# Lab 22.1 is used only if its solution has exactly the shape this lab calls:
#   void launchGemvGather(const float* X, const float* W, float* Y, int batch, int in, int out)
#   void encodeGemvGather(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
#                         id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
#                         uint32_t batch, uint32_t in, uint32_t out)
# and a Metal kernel named gemv_gather (or a kernel.metal with a single kernel). Its
# directory is the first ../../lab1_*/solution, or -DPMPP_CH22_GEMV_DIR=<dir>.

get_filename_component(_ch22_labs ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)
set(_ch22_m0 ${_ch22_labs}/ch02/lab3_elementwise/solution)
set(_ch22_m2 ${_ch22_labs}/ch06/lab4_linear/solution)
set(_ch22_m3 ${_ch22_labs}/ch15/lab6_linear/solution)
set(_ch22_m4 ${_ch22_labs}/ch10/lab3_softmax_argmax/solution)
if(PMPP_CH22_GEMV_DIR)
  set(_ch22_gemv ${PMPP_CH22_GEMV_DIR})
else()
  file(GLOB _ch22_gemv_dirs LIST_DIRECTORIES true ${_ch22_labs}/ch22/lab1_*/solution)
  set(_ch22_gemv "")
  if(_ch22_gemv_dirs)
    list(SORT _ch22_gemv_dirs)
    list(GET _ch22_gemv_dirs 0 _ch22_gemv)
  endif()
endif()

set(pmpp_ch22_pieces_cu ${CMAKE_CURRENT_LIST_DIR}/step_timer.cu)
set(pmpp_ch22_pieces_mm "")
set(_ch22_defs "")          # every backend
set(_ch22_metal_defs "")    # Metal only
set(_ch22_includes "")
set(_ch22_note "// Generated at configure time by labs/ch22/lab2_batching/provided/pieces.cmake")

# Writes `content` to `path` only if it differs, so a reconfigure does not force a rebuild.
function(_pmpp_ch22_write path content)
  if(EXISTS ${path})
    file(READ ${path} old)
    if(old STREQUAL content)
      return()
    endif()
  endif()
  file(WRITE ${path} "${content}")
endfunction()

# Copies `src` to `dst` in the build directory, replacing each OLD NEW pair in ARGN, and
# makes one-line __device__ helper definitions static (a declaration ending in `;` keeps
# its linkage). Appends `dst` to the variable named `list`.
function(_pmpp_ch22_copy list src dst)
  file(READ ${src} text)
  set(pairs ${ARGN})
  list(LENGTH pairs n)
  set(i 0)
  while(i LESS n)
    math(EXPR j "${i} + 1")
    list(GET pairs ${i} old)
    list(GET pairs ${j} new)
    string(REPLACE "${old}" "${new}" text "${text}")
    math(EXPR i "${i} + 2")
  endwhile()
  string(REGEX REPLACE "\n__device__ ([^\n;]*\\)[ \t]*\\{)" "\nstatic __device__ \\1" text "${text}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${src})
  _pmpp_ch22_write(${CMAKE_CURRENT_BINARY_DIR}/${dst} "${_ch22_note}\n// from ${src}. Do not edit.\n${text}")
  set(${list} ${${list}} ${CMAKE_CURRENT_BINARY_DIR}/${dst} PARENT_SCOPE)
endfunction()

# A required piece: copy it, or name what is missing so that pmpp_add_lab skips the lab.
macro(_pmpp_ch22_required dir name)
  foreach(_f kernel.cu kernel.metal launch.mm)
    if(NOT EXISTS ${dir}/${_f})
      message(STATUS "pmpp lab ch22/lab2_batching: ${name}'s solution has no ${_f} (${dir})")
      list(APPEND pmpp_ch22_pieces_cu ${dir}/${_f})
    endif()
  endforeach()
endmacro()

# ---- lab 2.3, M0
_pmpp_ch22_required(${_ch22_m0} "lab 2.3")
if(EXISTS ${_ch22_m0}/kernel.cu AND EXISTS ${_ch22_m0}/launch.mm)
  _pmpp_ch22_copy(pmpp_ch22_pieces_cu ${_ch22_m0}/kernel.cu m0_elementwise.cu)
  _pmpp_ch22_copy(pmpp_ch22_pieces_mm ${_ch22_m0}/launch.mm m0_elementwise_launch.mm)
  list(APPEND _ch22_metal_defs PMPP_M0_METAL="${_ch22_m0}/kernel.metal")
endif()

# ---- lab 6.4, M2
_pmpp_ch22_required(${_ch22_m2} "lab 6.4")
if(EXISTS ${_ch22_m2}/kernel.cu AND EXISTS ${_ch22_m2}/launch.mm)
  _pmpp_ch22_copy(pmpp_ch22_pieces_cu ${_ch22_m2}/kernel.cu m2_linear.cu
                  launchLinear launchLinearM2 linearKernel linearM2Kernel)
  _pmpp_ch22_copy(pmpp_ch22_pieces_mm ${_ch22_m2}/launch.mm m2_linear_launch.mm
                  encodeLinear encodeLinearM2 LinearDims LinearM2Dims)
  list(APPEND _ch22_metal_defs PMPP_M2_METAL="${_ch22_m2}/kernel.metal")
endif()

# ---- lab 10.3, M4
_pmpp_ch22_required(${_ch22_m4} "lab 10.3")
if(EXISTS ${_ch22_m4}/kernel.cu AND EXISTS ${_ch22_m4}/launch.mm)
  _pmpp_ch22_copy(pmpp_ch22_pieces_cu ${_ch22_m4}/kernel.cu m4_rowwise.cu)
  _pmpp_ch22_copy(pmpp_ch22_pieces_mm ${_ch22_m4}/launch.mm m4_rowwise_launch.mm)
  list(APPEND _ch22_metal_defs PMPP_M4_METAL="${_ch22_m4}/kernel.metal")
endif()

# ---- lab 15.6, M3 (optional)
if(EXISTS ${_ch22_m3}/kernel.cu AND EXISTS ${_ch22_m3}/launch.mm AND EXISTS ${_ch22_m3}/kernel.metal)
  _pmpp_ch22_copy(pmpp_ch22_pieces_cu ${_ch22_m3}/kernel.cu m3_linear.cu launchLinear launchLinearM3)
  _pmpp_ch22_copy(pmpp_ch22_pieces_mm ${_ch22_m3}/launch.mm m3_linear_launch.mm
                  encodeLinear encodeLinearM3 LinearDims LinearM3Dims)
  list(APPEND _ch22_defs PMPP_HAVE_M3=1)
  list(APPEND _ch22_metal_defs PMPP_M3_METAL="${_ch22_m3}/kernel.metal")
else()
  message(STATUS "pmpp lab ch22/lab2_batching: lab 15.6's solution not found in ${_ch22_m3}; no m3 cases or rows")
endif()

# ---- lab 22.1, the gather (optional; only with the expected signatures)
set(_ch22_ws "[ \t\r\n]")
set(_ch22_cuda_sig "void${_ch22_ws}+launchGemvGather${_ch22_ws}*\\(${_ch22_ws}*const${_ch22_ws}+float${_ch22_ws}*\\*[^,]*,${_ch22_ws}*const${_ch22_ws}+float${_ch22_ws}*\\*[^,]*,${_ch22_ws}*float${_ch22_ws}*\\*[^,]*,${_ch22_ws}*int[^,]*,${_ch22_ws}*int[^,]*,${_ch22_ws}*int[^,)]*\\)")
set(_ch22_metal_sig "encodeGemvGather${_ch22_ws}*\\(${_ch22_ws}*id<MTLComputeCommandEncoder>[^,]*,${_ch22_ws}*id<MTLComputePipelineState>[^,]*,${_ch22_ws}*id<MTLBuffer>[^,]*,${_ch22_ws}*id<MTLBuffer>[^,]*,${_ch22_ws}*id<MTLBuffer>[^,]*,${_ch22_ws}*uint32_t[^,]*,${_ch22_ws}*uint32_t[^,]*,${_ch22_ws}*uint32_t[^,)]*\\)")
if(_ch22_gemv AND EXISTS ${_ch22_gemv}/kernel.cu)
  file(READ ${_ch22_gemv}/kernel.cu _src)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_ch22_gemv}/kernel.cu)
  if(_src MATCHES "${_ch22_cuda_sig}")
    _pmpp_ch22_copy(pmpp_ch22_pieces_cu ${_ch22_gemv}/kernel.cu gemv_gather.cu)
    list(APPEND _ch22_defs PMPP_HAVE_GEMV=1)
    list(APPEND _ch22_includes ${_ch22_gemv})
  else()
    message(STATUS "pmpp lab ch22/lab2_batching: ${_ch22_gemv}/kernel.cu has no launchGemvGather with "
                   "the linear layer's signature; no gemv cases or rows on CUDA, HIP and the emulators")
  endif()
endif()
if(_ch22_gemv AND EXISTS ${_ch22_gemv}/launch.mm AND EXISTS ${_ch22_gemv}/kernel.metal)
  file(READ ${_ch22_gemv}/launch.mm _src)
  file(READ ${_ch22_gemv}/kernel.metal _msl)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_ch22_gemv}/launch.mm ${_ch22_gemv}/kernel.metal)
  string(REGEX MATCHALL "kernel${_ch22_ws}+void${_ch22_ws}+[A-Za-z_0-9]+" _kernels "${_msl}")
  set(_ch22_gemv_kernel "")
  list(LENGTH _kernels _nk)
  foreach(_k ${_kernels})
    string(REGEX REPLACE "^kernel${_ch22_ws}+void${_ch22_ws}+" "" _k "${_k}")
    if(_k STREQUAL "gemv_gather" OR _nk EQUAL 1)
      set(_ch22_gemv_kernel ${_k})
    endif()
  endforeach()
  if(_src MATCHES "${_ch22_metal_sig}" AND _ch22_gemv_kernel)
    _pmpp_ch22_copy(pmpp_ch22_pieces_mm ${_ch22_gemv}/launch.mm gemv_gather_launch.mm)
    list(APPEND _ch22_metal_defs PMPP_GEMV_METAL="${_ch22_gemv}/kernel.metal"
                                 PMPP_GEMV_KERNEL="${_ch22_gemv_kernel}")
    list(APPEND _ch22_includes ${_ch22_gemv})
  else()
    message(STATUS "pmpp lab ch22/lab2_batching: ${_ch22_gemv} has no encodeGemvGather with the expected "
                   "signature, or no kernel gemv_gather; no gemv cases or rows on Metal")
  endif()
endif()

# Adds the definitions above to every target of lab `base` (e.g. ch22_lab2_batching).
function(pmpp_ch22_pieces_link base)
  foreach(v work solution)
    foreach(b ${PMPP_BACKENDS})
      set(tgt ${base}.${v}.${b})
      if(NOT TARGET ${tgt})
        continue()
      endif()
      if(_ch22_defs)
        target_compile_definitions(${tgt} PRIVATE ${_ch22_defs})
      endif()
      if(_ch22_includes)
        target_include_directories(${tgt} PRIVATE ${_ch22_includes})
      endif()
      if(b STREQUAL "metal" AND _ch22_metal_defs)
        target_compile_definitions(${tgt} PRIVATE ${_ch22_metal_defs})
      endif()
    endforeach()
  endforeach()
endfunction()
