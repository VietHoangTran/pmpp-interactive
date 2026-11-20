pmpp_add_lab(ch05/lab2_tiled
  VARIANTS work solution
  TILE 16
  EXTRA_TILES 8 32
  SHARED_CU ${CMAKE_CURRENT_SOURCE_DIR}/../shared/naive_matmul.cu
  SHARED_MM ${CMAKE_CURRENT_SOURCE_DIR}/../shared/naive_launch.mm)
