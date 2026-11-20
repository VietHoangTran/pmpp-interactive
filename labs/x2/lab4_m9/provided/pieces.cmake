# Lab X2.4, provided: the project's own pieces, which the quantised decode step is built from.
# Modelled on lab 22.2's provided/pieces.cmake. Included by ../CMakeLists.txt before
# pmpp_add_lab; call pmpp_x2m9_pieces_link() after it. Sets:
#   pmpp_x2m9_cu   provided CUDA sources (SHARED_CU)
#   pmpp_x2m9_mm   provided Metal host sources (SHARED_MM)
#
# Each piece is an earlier lab's reference solution, copied into the build directory at
# configure time with its one-line __device__ helper definitions made static, so that two
# pieces may each have, say, a warpSum. Its Metal kernels are compiled at run time from that
# lab's own kernel.metal; the driver passes the constants that lab's driver uses.
#
#   lab 2.3  (M0)  launchAddBias, launchAddResidual; encodeAddBias, encodeAddResidual   required
#   lab 10.3 (M4)  launchSoftmaxRows, launchArgmaxRows and their encoders                required
#   lab 22.1       launchGemvGather, encodeGemvGather (the F32 GEMV)                     required
#   lab X2.1       launchGemvQ8_0, encodeGemvQ8_0      if present with this shape, else the
#   lab X2.2       launchGemvQ4_0, encodeGemvQ4_0      stand-ins in ../shared-g12 (qpieces.cmake)
#   step_timer.cu  one pair of events around a step (the CUDA and HIP timing)            always
# A required piece that is missing leaves the lab skipped, with a note that names it.

include(${CMAKE_CURRENT_LIST_DIR}/../../shared-g12/qpieces.cmake)
get_filename_component(_x2m9_labs ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)
set(_x2m9_m0 ${_x2m9_labs}/ch02/lab3_elementwise/solution)
set(_x2m9_m4 ${_x2m9_labs}/ch10/lab3_softmax_argmax/solution)
set(_x2m9_gemv ${_x2m9_labs}/ch22/lab1_gemv/solution)

set(pmpp_x2m9_cu ${CMAKE_CURRENT_LIST_DIR}/step_timer.cu)
set(pmpp_x2m9_mm "")
set(_x2m9_metal_defs "")
set(_x2m9_includes "")

# A required piece: copy it (kernel.cu and launch.mm), or name what is missing so that
# pmpp_add_lab skips the lab.
macro(_x2m9_required dir name stem)
  set(_ok TRUE)
  foreach(_f kernel.cu kernel.metal launch.mm)
    if(NOT EXISTS ${dir}/${_f})
      message(STATUS "pmpp lab x2/lab4_m9: ${name}'s solution has no ${_f} (${dir})")
      list(APPEND pmpp_x2m9_cu ${dir}/${_f})
      set(_ok FALSE)
    endif()
  endforeach()
  if(_ok)
    _x2g12_copy(_c ${dir}/kernel.cu ${stem}.cu)
    list(APPEND pmpp_x2m9_cu ${_c})
    _x2g12_copy(_c ${dir}/launch.mm ${stem}_launch.mm)
    list(APPEND pmpp_x2m9_mm ${_c})
    list(APPEND _x2m9_includes ${dir})
  endif()
endmacro()

_x2m9_required(${_x2m9_m0} "lab 2.3" m0_elementwise)
list(APPEND _x2m9_metal_defs PMPP_M0_METAL="${_x2m9_m0}/kernel.metal")
_x2m9_required(${_x2m9_m4} "lab 10.3" m4_rowwise)
list(APPEND _x2m9_metal_defs PMPP_M4_METAL="${_x2m9_m4}/kernel.metal")
_x2m9_required(${_x2m9_gemv} "lab 22.1" gemv_gather)
list(APPEND _x2m9_metal_defs PMPP_GEMV_METAL="${_x2m9_gemv}/kernel.metal")

# Lab X2.1's and X2.2's GEMVs, or the stand-ins.
pmpp_x2g12_qgemv(_x2m9_q x2/lab4_m9 Q8_0 Q4_0)
list(APPEND pmpp_x2m9_cu ${_x2m9_q_cu})
list(APPEND pmpp_x2m9_mm ${_x2m9_q_mm})

function(pmpp_x2m9_pieces_link base)
  pmpp_x2g12_link(${base} DEFS ${_x2m9_q_defs} METAL_DEFS ${_x2m9_metal_defs} ${_x2m9_q_metal_defs}
                  INCLUDES ${_x2m9_includes} ${_x2m9_q_includes})
endfunction()
