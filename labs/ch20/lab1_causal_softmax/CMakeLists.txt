# Lab 20.1, milestone M6 part one. ../shared-g11/unfused.cmake copies the project's M3 and M2
# kernels in for the benchmark's unfused attention pipeline.
include(${CMAKE_CURRENT_SOURCE_DIR}/../shared-g11/unfused.cmake)
pmpp_add_lab(ch20/lab1_causal_softmax VARIANTS work solution RACE SHARED_CU ${pmpp_ch20_unfused_cu} SHARED_MM ${pmpp_ch20_unfused_mm})
pmpp_ch20_unfused_link(ch20_lab1_causal_softmax)
