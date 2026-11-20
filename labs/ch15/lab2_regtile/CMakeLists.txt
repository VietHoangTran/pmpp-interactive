include(${CMAKE_CURRENT_SOURCE_DIR}/../shared/ch15.cmake)
ch15_add_stage(ch15/lab2_regtile KERNEL gemmRegTileKernel PREV lab1_coarse PREV_NAME coarse)
