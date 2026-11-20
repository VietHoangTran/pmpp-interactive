# Lab X2.2. The benchmark's baselines are linked beside your files:
#   lab 22.1's reference gather (FP32; also the second half of the provided unfused Q4_0 path),
#   lab X2.1's reference Q8_0 kernel, if that solution is present (PMPP_HAVE_Q8),
#   provided/unfused.cu and provided/unfused_launch.mm (dequantise, then gather).
# The two kernel.cu files are copied unchanged under their own names, because the emulator
# build names its generated source after the file's stem and kernel.cu files would clash.
set(_x2_ch22 ${CMAKE_CURRENT_SOURCE_DIR}/../../ch22/lab1_gemv/solution)
set(_x2_q8 ${CMAKE_CURRENT_SOURCE_DIR}/../lab1_q8gemv/solution)
configure_file(${_x2_ch22}/kernel.cu ${CMAKE_CURRENT_BINARY_DIR}/ch22_gemv_gather.cu COPYONLY)
set(_x2_cu ${CMAKE_CURRENT_BINARY_DIR}/ch22_gemv_gather.cu ${CMAKE_CURRENT_SOURCE_DIR}/provided/unfused.cu)
set(_x2_mm ${_x2_ch22}/launch.mm ${CMAKE_CURRENT_SOURCE_DIR}/provided/unfused_launch.mm)
set(_x2_have_q8 OFF)
if(EXISTS ${_x2_q8}/kernel.cu AND EXISTS ${_x2_q8}/kernel.metal AND EXISTS ${_x2_q8}/launch.mm)
  configure_file(${_x2_q8}/kernel.cu ${CMAKE_CURRENT_BINARY_DIR}/x2_gemv_q8_0.cu COPYONLY)
  list(APPEND _x2_cu ${CMAKE_CURRENT_BINARY_DIR}/x2_gemv_q8_0.cu)
  list(APPEND _x2_mm ${_x2_q8}/launch.mm)
  set(_x2_have_q8 ON)
else()
  message(STATUS "pmpp lab x2/lab2_q4gemv: lab X2.1's solution not found in ${_x2_q8}; no Q8_0 rows")
endif()

pmpp_add_lab(x2/lab2_q4gemv VARIANTS work solution SHARED_CU ${_x2_cu} SHARED_MM ${_x2_mm})

if(_x2_have_q8)
  foreach(v work solution)
    foreach(b ${PMPP_BACKENDS})
      if(TARGET x2_lab2_q4gemv.${v}.${b})
        target_compile_definitions(x2_lab2_q4gemv.${v}.${b} PRIVATE PMPP_HAVE_Q8=1)
      endif()
    endforeach()
  endforeach()
endif()
