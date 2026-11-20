# The benchmark sets the reader's occupancy model beside the vendor's own occupancy
# function, which provided/occupancy_api.cu calls. That file has not been compiled by
# nvcc or a HIP compiler (see its header comment). If it does not build somewhere, turn
# this option off and the lab works without that column.
option(PMPP_CH04_OCCUPANCY_API "Lab 4.4: call the vendor's occupancy function in the benchmark" ON)
if(PMPP_CH04_OCCUPANCY_API)
  set(pmpp_ch04_occupancy ${CMAKE_CURRENT_SOURCE_DIR}/provided/occupancy_api.cu)
else()
  set(pmpp_ch04_occupancy ${CMAKE_CURRENT_SOURCE_DIR}/provided/occupancy_none.cu)
endif()
pmpp_add_lab(ch04/lab4_blocksize VARIANTS work solution SHARED_CU ${pmpp_ch04_occupancy})
