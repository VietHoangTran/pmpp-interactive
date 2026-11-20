pmpp_add_lab(ch06/lab3_banks
  VARIANTS work solution
  TILE 16
  EXTRA_TILES 32
  SHARED_CU ${CMAKE_CURRENT_SOURCE_DIR}/provided/bank_probe.cu ${CMAKE_CURRENT_SOURCE_DIR}/provided/copy.cu
  SHARED_MM ${CMAKE_CURRENT_SOURCE_DIR}/provided/bank_probe_launch.mm
  RACE)
