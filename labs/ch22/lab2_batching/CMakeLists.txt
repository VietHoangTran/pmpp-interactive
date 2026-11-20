# Lab 22.2, milestone M8. provided/pieces.cmake copies the earlier labs' reference solutions
# (M0, M2, M3, M4, lab 22.1's gather) into the build directory for the step to call.
include(${CMAKE_CURRENT_SOURCE_DIR}/provided/pieces.cmake)
pmpp_add_lab(ch22/lab2_batching VARIANTS work solution SHARED_CU ${pmpp_ch22_pieces_cu} SHARED_MM ${pmpp_ch22_pieces_mm})
pmpp_ch22_pieces_link(ch22_lab2_batching)
