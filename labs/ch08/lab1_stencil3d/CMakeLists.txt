pmpp_add_lab(ch08/lab1_stencil3d
  VARIANTS work solution
  TILE 32
  EXTRA_TILES 16 8
  SHARED_CU ${CMAKE_CURRENT_SOURCE_DIR}/../shared/tiled_cube.cu ${CMAKE_CURRENT_SOURCE_DIR}/../shared/planes.cu
  SHARED_MM ${CMAKE_CURRENT_SOURCE_DIR}/../shared/provided_launch.mm
  RACE)
