# Lab X2.1. Lab 22.1's reference gather is linked as the benchmark's FP32 baseline. Its
# kernel.cu is copied unchanged under its own name, because the emulator build names its
# generated source after the file's stem and two kernel.cu files would clash.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../ch22/lab1_gemv/solution/kernel.cu
               ${CMAKE_CURRENT_BINARY_DIR}/ch22_gemv_gather.cu COPYONLY)
pmpp_add_lab(x2/lab1_q8gemv VARIANTS work solution
  SHARED_CU ${CMAKE_CURRENT_BINARY_DIR}/ch22_gemv_gather.cu
  SHARED_MM ${CMAKE_CURRENT_SOURCE_DIR}/../../ch22/lab1_gemv/solution/launch.mm)
