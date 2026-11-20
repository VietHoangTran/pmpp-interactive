# Lab 19.1: a convolutional layer two ways. provided/gemm.cmake picks the project's M2 GEMM
# (lab 6.4's launchLinear) that your unroll + GEMM path calls; the provided implicit GEMM
# (corrected Fig. 19.11) is timed beside your two kernels.
include(${CMAKE_CURRENT_SOURCE_DIR}/provided/gemm.cmake)
pmpp_add_lab(ch19/lab1_convgemm VARIANTS work solution RACE TILE 16
  SHARED_CU ${pmpp_ch19_gemm_cu} ${CMAKE_CURRENT_SOURCE_DIR}/provided/implicit_gemm.cu
  SHARED_MM ${CMAKE_CURRENT_SOURCE_DIR}/provided/linear_at.mm ${CMAKE_CURRENT_SOURCE_DIR}/provided/implicit_gemm_launch.mm)
pmpp_ch19_gemm_link(ch19_lab1_convgemm)
