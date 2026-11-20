# Lab 15.6, milestone M3. provided/baselines.cmake prepares lab 6.4's M2 kernel and the
# vendor library (cuBLAS, hipBLAS or MPS) that the benchmark times beside your kernel.
include(${CMAKE_CURRENT_SOURCE_DIR}/provided/baselines.cmake)
pmpp_add_lab(ch15/lab6_linear VARIANTS work solution RACE SHARED_CU ${pmpp_ch15_m3_cu} SHARED_MM ${pmpp_ch15_m3_mm})
pmpp_ch15_m3_link(ch15_lab6_linear)
