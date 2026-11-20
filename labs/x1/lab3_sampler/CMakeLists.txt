pmpp_add_lab(x1/lab3_sampler VARIANTS work solution RACE
  SHARED_CU ${CMAKE_CURRENT_SOURCE_DIR}/../shared-g11/penalties_g11.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/../shared-g11/softmax_g11.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/../shared-g11/topp_g11.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/../shared-g11/sortrows_g11.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/../shared-g11/topk_g11.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/../shared-g11/minp_g11.cu
  SHARED_MM ${CMAKE_CURRENT_SOURCE_DIR}/../../ch09/lab3_tokencounts/solution/launch.mm
            ${CMAKE_CURRENT_SOURCE_DIR}/../../ch10/lab3_softmax_argmax/solution/launch.mm
            ${CMAKE_CURRENT_SOURCE_DIR}/../../ch11/lab6_topp/solution/launch.mm
            ${CMAKE_CURRENT_SOURCE_DIR}/../../ch14/lab3_sortrows/solution/launch.mm
            ${CMAKE_CURRENT_SOURCE_DIR}/../shared-g11/topk_g11_launch.mm
            ${CMAKE_CURRENT_SOURCE_DIR}/../shared-g11/minp_g11_launch.mm)
