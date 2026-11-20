# Lab 20.2, milestone M6 part two. TILE_WIDTH is the kernel's B_c: 32 as in the book, 16 also
# built on CUDA and HIP for the benchmark (Metal chooses B_c at run time, in launch.mm).
# ../shared-g11/unfused.cmake copies in M3, M2 and lab 20.1's reference softmax for the
# benchmark's unfused pipeline.
set(PMPP_CH20_SOFTMAX_REF ON)
include(${CMAKE_CURRENT_SOURCE_DIR}/../shared-g11/unfused.cmake)
pmpp_add_lab(ch20/lab2_flash VARIANTS work solution RACE TILE 32 EXTRA_TILES 16
  SHARED_CU ${pmpp_ch20_unfused_cu} SHARED_MM ${pmpp_ch20_unfused_mm})
pmpp_ch20_unfused_link(ch20_lab2_flash .t16)
