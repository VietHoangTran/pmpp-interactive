include(${CMAKE_CURRENT_SOURCE_DIR}/../shared/ch15.cmake)
ch15_add_stage(ch15/lab4_padded KERNEL gemmPaddedKernel TEMPLATE PREV lab3_remap PREV_NAME remap)
