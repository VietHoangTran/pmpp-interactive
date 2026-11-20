# pmpp_add_lab(<id>
#   VARIANTS work solution         directories holding kernel.cu / kernel.metal / launch.mm
#   [TILE 16]                      TILE_WIDTH the CUDA-family kernels are compiled with
#   [EXTRA_TILES 8 32]             extra CUDA-family builds of each variant, for benchmarks
#   [SHARED_CU a.cu ...]           provided CUDA sources compiled into every variant
#   [SHARED_MM a.mm ...]           provided Metal host sources compiled into every variant
# )
# <id> is e.g. ch05/lab2_tiled. One executable per (variant, backend[, tile]):
#   ch05_lab2_tiled.solution.metal, ch05_lab2_tiled.work.emu, ...cuda.t32
# Every executable is listed in <build>/pmpp_targets.txt for labs/pmpp.py.
function(pmpp_add_lab id)
  cmake_parse_arguments(L "" "TILE" "VARIANTS;EXTRA_TILES;SHARED_CU;SHARED_MM" ${ARGN})
  string(REPLACE "/" "_" base ${id})
  set(dir ${CMAKE_CURRENT_SOURCE_DIR})
  if(NOT L_TILE)
    set(L_TILE 16)
  endif()

  foreach(v ${L_VARIANTS})
    foreach(b ${PMPP_BACKENDS})
      set(tiles ${L_TILE})
      if(NOT b STREQUAL "metal")       # Metal picks the tile width at run time
        list(APPEND tiles ${L_EXTRA_TILES})
      endif()
      foreach(t ${tiles})
        if(t EQUAL L_TILE)
          set(tgt ${base}.${v}.${b})
        else()
          set(tgt ${base}.${v}.${b}.t${t})
        endif()

        if(b STREQUAL "metal")
          if(NOT EXISTS ${dir}/${v}/kernel.metal)
            continue()
          endif()
          set(wrap ${CMAKE_CURRENT_BINARY_DIR}/${tgt}.main.mm)
          file(WRITE ${wrap} "#include \"${dir}/main.cpp\"\n")
          add_executable(${tgt} ${wrap} ${dir}/${v}/launch.mm ${L_SHARED_MM})
          target_compile_definitions(${tgt} PRIVATE PMPP_METAL_SOURCE="${dir}/${v}/kernel.metal"
                                                    PMPP_METAL_SHARED_DIR="${dir}/../shared")
        else()
          if(NOT EXISTS ${dir}/${v}/kernel.cu)
            continue()
          endif()
          set(kernel_srcs "")
          foreach(src ${dir}/${v}/kernel.cu ${L_SHARED_CU})
            get_filename_component(stem ${src} NAME_WE)
            if(b STREQUAL "emu")
              set(gen ${CMAKE_CURRENT_BINARY_DIR}/${tgt}.${stem}.cpp)
              add_custom_command(OUTPUT ${gen}
                COMMAND ${Python3_EXECUTABLE} ${PMPP_ROOT}/tools/cu2emu.py ${src} ${gen}
                DEPENDS ${src} ${PMPP_ROOT}/tools/cu2emu.py VERBATIM)
              set_source_files_properties(${gen} PROPERTIES
                COMPILE_OPTIONS "-include;${PMPP_INC}/pmpp/emu/cuda_emu.h")
              list(APPEND kernel_srcs ${gen})
            elseif(b STREQUAL "hip")
              set(gen ${CMAKE_CURRENT_BINARY_DIR}/${tgt}.${stem}.hip)
              file(WRITE ${gen} "#include \"pmpp/vendors/hip.h\"\n#include \"${src}\"\n")
              set_source_files_properties(${gen} PROPERTIES LANGUAGE HIP)
              list(APPEND kernel_srcs ${gen})
            else()  # cuda: .cu files are compiled by nvcc as they are
              list(APPEND kernel_srcs ${src})
            endif()
          endforeach()
          add_executable(${tgt} ${dir}/main.cpp ${kernel_srcs})
        endif()

        target_link_libraries(${tgt} PRIVATE pmpp_rt_${b})
        target_include_directories(${tgt} PRIVATE ${dir} ${dir}/${v})
        target_compile_definitions(${tgt} PRIVATE TILE_WIDTH=${t}
          PMPP_LAB="${id}" PMPP_VARIANT="${v}")
        set_target_properties(${tgt} PROPERTIES EXCLUDE_FROM_ALL TRUE)
        file(APPEND ${CMAKE_BINARY_DIR}/pmpp_targets.txt
          "${id}|${v}|${b}|${t}|${tgt}|${CMAKE_BINARY_DIR}/bin/${tgt}\n")
      endforeach()
    endforeach()
  endforeach()
endfunction()
