include(${CMAKE_CURRENT_SOURCE_DIR}/../shared/ch15.cmake)
ch15_add_stage(ch15/lab3_remap KERNEL gemmRemapKernel TEMPLATE PREV lab2_regtile PREV_NAME regtile)
