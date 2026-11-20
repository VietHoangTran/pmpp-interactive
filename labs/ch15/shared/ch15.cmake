# Labs 15.1 to 15.5 (milestone M3, the five book-layout stages): provided build logic.
# Each stage's CMakeLists.txt includes this file and calls ch15_add_stage once.
#
#   ch15_add_stage(<id> KERNEL <kernel name> [TEMPLATE] [PREV <lab dir> PREV_NAME <name>])
#
#   KERNEL     the stage's __global__ function, for the occupancy query (stage_occupancy.cu)
#   TEMPLATE   the kernel is a template over the warp width (labs 15.3 to 15.5)
#   PREV       the previous stage's directory under labs/ch15 (lab2_regtile, ...); its
#              solution is built into this stage's benchmark as the row "previous"
#   PREV_NAME  a short name for the copies of its files (coarse, regtile, ...)
#
# Every stage's benchmark also times lab 5.3's solution (milestone M2, T = 16) and the
# vendor GEMM. Lab 5.3's and the previous stage's kernel.cu and launch.mm are copied into
# the build directory under distinct names (lab53_gemm.cu, prev_<name>.cu, ...), because
# pmpp_cu_sources names its generated files by the source's stem, and two files called
# kernel.cu in one target would collide. The .metal files are compiled at run time from
# where they are; their paths reach main.cpp as PMPP_CH15_LAB53_METAL and
# PMPP_CH15_PREV_METAL.
include_guard(GLOBAL)

# Like lab 4.4's occupancy option: the vendor files have not been built on CUDA or HIP
# (see the header of vendor_blas.cu). If they do not build somewhere, turn this off and
# the benchmark runs without the vendor's column.
option(PMPP_CH15_VENDOR "Labs 15.1 to 15.5: time the vendor GEMM (cuBLAS, hipBLAS or MPS) in the benchmark" ON)
option(PMPP_CH15_OCCUPANCY_API "Labs 15.1 to 15.5: ask the CUDA or HIP runtime how many blocks fit per SM" ON)

function(ch15_add_stage id)
  cmake_parse_arguments(S "TEMPLATE" "KERNEL;PREV;PREV_NAME" "" ${ARGN})
  set(shared ${CMAKE_CURRENT_FUNCTION_LIST_DIR})
  set(gen ${CMAKE_CURRENT_BINARY_DIR}/provided)

  # Lab 5.3's solution: the M2 kernel every stage is compared with.
  set(lab53 ${PMPP_ROOT}/ch05/lab3_gemm/solution)
  configure_file(${lab53}/kernel.cu ${gen}/lab53_gemm.cu COPYONLY)
  configure_file(${lab53}/launch.mm ${gen}/lab53_launch.mm COPYONLY)
  set(cu ${gen}/lab53_gemm.cu ${shared}/vendor_blas.cu ${shared}/stage_occupancy.cu)
  set(mm ${gen}/lab53_launch.mm ${shared}/vendor_mps.mm)

  # The previous stage's solution, if there is one.
  set(prev_metal "")
  if(S_PREV)
    set(prev ${shared}/../${S_PREV}/solution)
    configure_file(${prev}/kernel.cu ${gen}/prev_${S_PREV_NAME}.cu COPYONLY)
    configure_file(${prev}/launch.mm ${gen}/prev_${S_PREV_NAME}_launch.mm COPYONLY)
    list(APPEND cu ${gen}/prev_${S_PREV_NAME}.cu)
    list(APPEND mm ${gen}/prev_${S_PREV_NAME}_launch.mm)
    get_filename_component(prev_metal ${prev}/kernel.metal ABSOLUTE)
  endif()

  pmpp_add_lab(${id} VARIANTS work solution SHARED_CU ${cu} SHARED_MM ${mm} RACE)

  if(S_TEMPLATE)
    set(is_template 1)
  else()
    set(is_template 0)
  endif()
  if(PMPP_CH15_VENDOR AND "hip" IN_LIST PMPP_BACKENDS)
    find_package(hipblas CONFIG QUIET)
  endif()

  # Definitions and libraries for the executables pmpp_add_lab made (it skips a lab whose
  # main.cpp does not exist yet, so a target may be missing).
  string(REPLACE "/" "_" base ${id})
  foreach(v work solution)
    foreach(b ${PMPP_BACKENDS})
      set(tgt ${base}.${v}.${b})
      if(NOT TARGET ${tgt})
        continue()
      endif()
      target_compile_definitions(${tgt} PRIVATE
        PMPP_CH15_KERNEL=${S_KERNEL} PMPP_CH15_KERNEL_TEMPLATE=${is_template}
        PMPP_CH15_LAB53_METAL="${lab53}/kernel.metal" PMPP_CH15_PREV_METAL="${prev_metal}")
      if(NOT PMPP_CH15_OCCUPANCY_API)
        target_compile_definitions(${tgt} PRIVATE PMPP_CH15_NO_OCCUPANCY)
      endif()
      if(NOT PMPP_CH15_VENDOR)
        target_compile_definitions(${tgt} PRIVATE PMPP_CH15_NO_VENDOR)
      elseif(b STREQUAL "cuda")
        if(TARGET CUDA::cublas)
          target_link_libraries(${tgt} PRIVATE CUDA::cublas)
        else()
          target_compile_definitions(${tgt} PRIVATE PMPP_CH15_VENDOR_MISSING)
        endif()
      elseif(b STREQUAL "hip")
        if(TARGET roc::hipblas)
          target_link_libraries(${tgt} PRIVATE roc::hipblas)
        else()
          target_compile_definitions(${tgt} PRIVATE PMPP_CH15_VENDOR_MISSING)
        endif()
      elseif(b STREQUAL "metal")
        target_link_libraries(${tgt} PRIVATE "-framework MetalPerformanceShaders")
      endif()
    endforeach()
  endforeach()
endfunction()
