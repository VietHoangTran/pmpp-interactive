# Lab X2.4, milestone M9. provided/pieces.cmake copies the earlier labs' reference solutions
# (M0, M4, lab 22.1's gather, lab X2.1's Q8_0 and lab X2.2's Q4_0 GEMVs) into the build
# directory for the step to call.
include(${CMAKE_CURRENT_SOURCE_DIR}/provided/pieces.cmake)
pmpp_add_lab(x2/lab4_m9 VARIANTS work solution SHARED_CU ${pmpp_x2m9_cu} SHARED_MM ${pmpp_x2m9_mm})
pmpp_x2m9_pieces_link(x2_lab4_m9)
