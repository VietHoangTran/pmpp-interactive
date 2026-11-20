# Lab X3.2, milestone M10. provided/pieces.cmake copies the earlier labs' reference solutions
# (M0, M2, M4, M7, M8, lab X3.1; M3, lab 20.5, lab 22.1 and M5 when present) into the build
# directory for the step to call, and writes the test models with provided/make_model.py.
include(${CMAKE_CURRENT_SOURCE_DIR}/provided/pieces.cmake)
pmpp_add_lab(x3/lab2_tinyllama VARIANTS work solution SHARED_CU ${pmpp_x3l2_cu} SHARED_MM ${pmpp_x3l2_mm})
pmpp_x3l2_link(x3_lab2_tinyllama)
