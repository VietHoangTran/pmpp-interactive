pmpp_add_lab(ch06/lab5_doublebuf
  VARIANTS work solution
  TILE 16
  EXTRA_TILES 8 32
  SHARED_CU ${CMAKE_CURRENT_SOURCE_DIR}/provided/tiled_two_barriers.cu
  SHARED_MM ${CMAKE_CURRENT_SOURCE_DIR}/provided/tiled_two_barriers_launch.mm
  RACE)
