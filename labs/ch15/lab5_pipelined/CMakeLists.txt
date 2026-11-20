include(${CMAKE_CURRENT_SOURCE_DIR}/../shared/ch15.cmake)
ch15_add_stage(ch15/lab5_pipelined KERNEL gemmPipelinedKernel TEMPLATE PREV lab4_padded PREV_NAME padded)
