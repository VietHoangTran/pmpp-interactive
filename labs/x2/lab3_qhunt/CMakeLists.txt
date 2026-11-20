# Lab X2.3, the integer-path bug hunt. ../shared-g12/qpieces.cmake supplies the float Q4_0 GEMV
# the benchmark compares against (lab X2.2's solution, or a stand-in); lab 22.2's step_timer.cu
# times a pass with one pair of events on CUDA and HIP.
include(${CMAKE_CURRENT_SOURCE_DIR}/../shared-g12/qpieces.cmake)
pmpp_x2g12_qgemv(qhunt x2/lab3_qhunt Q4_0)
pmpp_add_lab(x2/lab3_qhunt VARIANTS work solution RACE
  SHARED_CU ${qhunt_cu} ${CMAKE_CURRENT_SOURCE_DIR}/../../ch22/lab2_batching/provided/step_timer.cu
  SHARED_MM ${qhunt_mm})
pmpp_x2g12_link(x2_lab3_qhunt DEFS ${qhunt_defs} METAL_DEFS ${qhunt_metal_defs} INCLUDES ${qhunt_includes})
