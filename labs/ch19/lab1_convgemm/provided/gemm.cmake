# Lab 19.1, provided: which GEMM the unroll + GEMM path calls. Included by
# ../CMakeLists.txt before pmpp_add_lab; call pmpp_ch19_gemm_link() after it. Modelled on
# labs/ch15/lab6_linear/provided/baselines.cmake.
#
# The path calls the project's M2 kernel, lab 6.4's launchLinear(X, W, Y, batch, in, out),
# unchanged. Which copy of lab 6.4 it uses is read from the environment at configure time
# (labs/pmpp.py configures before every run, so a change takes effect at the next run):
#
#   PMPP_CH19_GEMM unset or "m2"   labs/ch06/lab4_linear/solution  (the reference M2)
#   PMPP_CH19_GEMM=work            labs/ch06/lab4_linear/work      (your own M2)
#
# Sets pmpp_ch19_gemm_cu, the CUDA source to pass in SHARED_CU: the chosen kernel.cu,
# copied into the build directory as m2_linear.cu. The copy keeps every symbol as it is
# (launchLinear, linearKernel); it only needs a name of its own, because the build turns
# each .cu file into <target>.<file name>.cpp for the emulators and .hip for HIP, and the
# lab's own kernel.cu already takes the name "kernel". configure_file re-runs the
# configure step whenever the chosen file changes, so edits to it are picked up.
# If the chosen kernel.cu does not exist, the lab is skipped with a note (pmpp_add_lab
# reports a missing provided source) instead of breaking every other lab.
#
# pmpp_ch19_gemm_link(<base>) adds to every target of the lab:
#   PMPP_CH19_GEMM_SOURCE="m2 reference" or "m2 yours"   (printed, and in the bench rows)
#   PMPP_CH19_M2_METAL="<dir>/kernel.metal"               (Metal only: lab 6.4's `linear`
#                                                         kernel, compiled at run time)

set(_pmpp_ch19_choice "$ENV{PMPP_CH19_GEMM}")
if(_pmpp_ch19_choice STREQUAL "" OR _pmpp_ch19_choice STREQUAL "m2")
  set(_pmpp_ch19_variant solution)
  set(_pmpp_ch19_source "m2 reference")
elseif(_pmpp_ch19_choice STREQUAL "work")
  set(_pmpp_ch19_variant work)
  set(_pmpp_ch19_source "m2 yours")
else()
  message(WARNING "pmpp lab ch19/lab1_convgemm: PMPP_CH19_GEMM='${_pmpp_ch19_choice}' is not "
                  "'m2' or 'work'; using lab 6.4's reference solution")
  set(_pmpp_ch19_variant solution)
  set(_pmpp_ch19_source "m2 reference")
endif()

get_filename_component(_pmpp_ch19_m2_dir
  ${CMAKE_CURRENT_LIST_DIR}/../../../ch06/lab4_linear/${_pmpp_ch19_variant} ABSOLUTE)

if(EXISTS ${_pmpp_ch19_m2_dir}/kernel.cu)
  configure_file(${_pmpp_ch19_m2_dir}/kernel.cu ${CMAKE_CURRENT_BINARY_DIR}/m2_linear.cu COPYONLY)
  set(pmpp_ch19_gemm_cu ${CMAKE_CURRENT_BINARY_DIR}/m2_linear.cu)
else()
  # A path that does not exist: pmpp_add_lab then skips the lab and says what is missing.
  set(pmpp_ch19_gemm_cu ${_pmpp_ch19_m2_dir}/kernel.cu)
endif()

# Adds the definitions above to every target of lab `base` (ch19_lab1_convgemm).
function(pmpp_ch19_gemm_link base)
  foreach(v work solution)
    foreach(b ${PMPP_BACKENDS})
      set(tgt ${base}.${v}.${b})
      if(NOT TARGET ${tgt})
        continue()
      endif()
      target_compile_definitions(${tgt} PRIVATE PMPP_CH19_GEMM_SOURCE="${_pmpp_ch19_source}")
      if(b STREQUAL "metal")
        target_compile_definitions(${tgt} PRIVATE PMPP_CH19_M2_METAL="${_pmpp_ch19_m2_dir}/kernel.metal")
      endif()
    endforeach()
  endforeach()
endfunction()
