# pmpp_add_lab(<id>
#   VARIANTS work solution         directories holding kernel.cu / kernel.metal / launch.mm
#   [TILE 16]                      TILE_WIDTH the CUDA-family kernels are compiled with
#   [EXTRA_TILES 8 32]             extra CUDA-family builds of each variant, for benchmarks
#   [SHARED_CU a.cu ...]           provided CUDA sources compiled into every variant
#   [SHARED_MM a.mm ...]           provided Metal host sources compiled into every variant
#   [RACE]                         also build each variant for the race-detection emulator
#                                  (backend `race`; tests only, default tile only)
# )
# <id> is e.g. ch05/lab2_tiled. One executable per (variant, backend[, tile]):
#   ch05_lab2_tiled.solution.metal, ch05_lab2_tiled.work.emu, ...cuda.t32
# Every executable is listed in <build>/pmpp_targets.txt for labs/pmpp.py.
#
# A driver that wants smaller problems on both emulator flavours can test
# `#if defined(PMPP_BACKEND_EMU) || defined(PMPP_BACKEND_RACE)`.

# Sets <out> to the sources to compile for `backend` from the given .cu files.
#   emu, race: translated by tools/cu2emu.py, with pmpp/emu/cuda_emu.h force-included
#   hip:       wrapped so that pmpp/vendors/hip.h comes first
#   cuda:      compiled by nvcc as they are
function(pmpp_cu_sources out tgt backend)
  set(result "")
  foreach(src ${ARGN})
    get_filename_component(stem ${src} NAME_WE)
    if(backend STREQUAL "emu" OR backend STREQUAL "race")
      set(gen ${CMAKE_CURRENT_BINARY_DIR}/${tgt}.${stem}.cpp)
      add_custom_command(OUTPUT ${gen}
        COMMAND ${Python3_EXECUTABLE} ${PMPP_ROOT}/tools/cu2emu.py ${src} ${gen}
        DEPENDS ${src} ${PMPP_ROOT}/tools/cu2emu.py VERBATIM)
      set_source_files_properties(${gen} PROPERTIES
        COMPILE_OPTIONS "-include;${PMPP_INC}/pmpp/emu/cuda_emu.h")
      list(APPEND result ${gen})
    elseif(backend STREQUAL "hip")
      set(gen ${CMAKE_CURRENT_BINARY_DIR}/${tgt}.${stem}.hip)
      file(WRITE ${gen} "#include \"pmpp/vendors/hip.h\"\n#include \"${src}\"\n")
      set_source_files_properties(${gen} PROPERTIES LANGUAGE HIP)
      list(APPEND result ${gen})
    else()
      list(APPEND result ${src})
    endif()
  endforeach()
  set(${out} ${result} PARENT_SCOPE)
endfunction()

function(pmpp_add_lab id)
  cmake_parse_arguments(L "RACE" "TILE" "VARIANTS;EXTRA_TILES;SHARED_CU;SHARED_MM" ${ARGN})
  string(REPLACE "/" "_" base ${id})
  set(dir ${CMAKE_CURRENT_SOURCE_DIR})
  if(NOT L_TILE)
    set(L_TILE 16)
  endif()
  # A lab that is still being written (no driver yet, or a provided source that does
  # not exist yet) is skipped with a note, so that it cannot break every other lab.
  set(missing "")
  foreach(f ${dir}/main.cpp ${L_SHARED_CU} ${L_SHARED_MM})
    if(NOT EXISTS ${f})
      list(APPEND missing ${f})
    endif()
  endforeach()
  if(missing)
    message(STATUS "pmpp lab ${id}: skipped, incomplete (missing ${missing})")
    file(APPEND ${CMAKE_BINARY_DIR}/pmpp_incomplete.txt "${id}|${missing}\n")
    return()
  endif()

  foreach(v ${L_VARIANTS})
    foreach(b ${PMPP_BACKENDS})
      if(b STREQUAL "race" AND NOT L_RACE)
        continue()
      endif()
      set(tiles ${L_TILE})
      if(NOT b STREQUAL "metal" AND NOT b STREQUAL "race")   # Metal picks the tile width at run time
        list(APPEND tiles ${L_EXTRA_TILES})
      endif()
      foreach(t ${tiles})
        if(t EQUAL L_TILE)
          set(tgt ${base}.${v}.${b})
        else()
          set(tgt ${base}.${v}.${b}.t${t})
        endif()

        if(b STREQUAL "metal")
          if(NOT EXISTS ${dir}/${v}/kernel.metal OR NOT EXISTS ${dir}/${v}/launch.mm)
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
          pmpp_cu_sources(kernel_srcs ${tgt} ${b} ${dir}/${v}/kernel.cu ${L_SHARED_CU})
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

# pmpp_add_selftest(<name> [EXPECT_RACE] [BACKENDS emu race cuda])
# A self-test of the emulator: <name>.cu (kernels, launches and checks, written
# in plain CUDA) plus the shared driver main.cpp in the same directory. Built by
# default for emu, race and cuda (on real CUDA the same file checks the
# emulator's model against the real thing; there is nothing to test on Metal,
# and the HIP name mapping in pmpp/vendors/hip.h does not cover these APIs).
#   EXPECT_RACE   on backend `race` the test passes only if ThreadSanitizer
#                 reports a data race in every case (a negative control for the
#                 race detector). On other backends it is an ordinary test.
#   BACKENDS      restrict the backends it is built for.
# Listed in <build>/pmpp_selftests.txt as name|backend|expect|target|exe.
function(pmpp_add_selftest name)
  cmake_parse_arguments(T "EXPECT_RACE" "" "BACKENDS" ${ARGN})
  if(NOT T_BACKENDS)
    set(T_BACKENDS emu race cuda)
  endif()
  set(dir ${CMAKE_CURRENT_SOURCE_DIR})
  foreach(b ${PMPP_BACKENDS})
    if(NOT b IN_LIST T_BACKENDS)
      continue()
    endif()
    set(tgt selftest_${name}.${b})
    pmpp_cu_sources(srcs ${tgt} ${b} ${dir}/${name}.cu)
    add_executable(${tgt} ${dir}/main.cpp ${srcs})
    target_link_libraries(${tgt} PRIVATE pmpp_rt_${b})
    target_include_directories(${tgt} PRIVATE ${dir})
    target_compile_definitions(${tgt} PRIVATE PMPP_LAB="selftest/${name}" PMPP_VARIANT="selftest")
    set_target_properties(${tgt} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    set(expect "clean")
    if(T_EXPECT_RACE AND b STREQUAL "race")
      set(expect "race")
    endif()
    file(APPEND ${CMAKE_BINARY_DIR}/pmpp_selftests.txt
      "${name}|${b}|${expect}|${tgt}|${CMAKE_BINARY_DIR}/bin/${tgt}\n")
  endforeach()
endfunction()
