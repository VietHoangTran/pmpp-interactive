pmpp_add_lab(ch07/lab2_tiledconv
  VARIANTS work solution
  TILE 32
  EXTRA_TILES 8 16
  SHARED_CU ${CMAKE_CURRENT_SOURCE_DIR}/../shared/conv_const.cu
  SHARED_MM ${CMAKE_CURRENT_SOURCE_DIR}/../shared/conv_const_launch.mm)
