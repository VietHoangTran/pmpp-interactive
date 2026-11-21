# pmpp_add_lab(<id>
#   VARIANTS work solution         directories holding kernel.cu / kernel.metal / launch.mm
#   [TILE 16]                      TILE_WIDTH the CUDA-family kernels are compiled with
#   [EXTRA_TILES 8 32]             extra CUDA-family builds of each variant, for benchmarks
#   [SHARED_DIR ../shared-g11]     the directory of the lab's provided shared sources, relative
#                                  to the lab (default ../shared). Defines PMPP_SHARED_DIR on
#                                  every backend and PMPP_METAL_SHARED_DIR on Metal (where the
#                                  driver finds provided .metal files at run time), adds it
#                                  to the include path, and lets SHARED_CU / SHARED_MM name
#                                  files in it without a path. Without SHARED_DIR only
#                                  PMPP_METAL_SHARED_DIR=<lab>/../shared is defined, as before.
#   [SHARED_CU a.cu ...]           provided CUDA sources compiled into every variant
#   [SHARED_MM a.mm ...]           provided Metal host sources compiled into every variant
#                                  (a relative name is looked up in SHARED_DIR, then the lab)
#   [LIBS lib ...]                 libraries every executable of the lab links
#   [LIBS_EMU ...] [LIBS_RACE ...] [LIBS_METAL ...] [LIBS_CUDA ...] [LIBS_HIP ...]
#                                  libraries for one backend only (CUDA::cublas, roc::hipblas,
#                                  "-framework MetalPerformanceShaders"). A CMake target name
#                                  (one with ::) that does not exist here skips that backend's
#                                  builds of the lab, with the reason recorded for pmpp.py.
#   [RACE]                         also build each variant for the race-detection emulator
#                                  (backend `race`; tests only, default tile only)
# )
# <id> is e.g. ch05/lab2_tiled. One executable per (variant, backend[, tile]):
#   ch05_lab2_tiled.solution.metal, ch05_lab2_tiled.work.emu, ...cuda.t32
# Every executable is listed in <build>/pmpp_targets.txt for labs/pmpp.py. A build that is
# not made (no kernel.metal for Metal, no RACE, a missing library) is listed with the reason
# in <build>/pmpp_skipped.txt as id|variant|backend|reason, so pmpp.py can say why.
# For each executable, <build>/pmpp_inputs/<target>.txt records its sources, compile
# definitions, include directories and binary directory (written at generate time, so it
# includes what a lab's own CMake adds after pmpp_add_lab). pmpp.py hashes those inputs,
# plus the headers in the compiler's dependency files, to tell whether an earlier test
# result still describes what was just built.
#
# A driver that wants smaller problems on both emulator flavours can test
# `#if defined(PMPP_BACKEND_EMU) || defined(PMPP_BACKEND_RACE)`.

file(WRITE ${CMAKE_BINARY_DIR}/pmpp_skipped.txt "")
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/pmpp_inputs)

# Records what target `tgt` is built from in <build>/pmpp_inputs/<tgt>.txt (see above).
function(pmpp_record_inputs tgt)
  file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/pmpp_inputs/${tgt}.txt CONTENT
"binary_dir=$<TARGET_PROPERTY:${tgt},BINARY_DIR>
sources=$<JOIN:$<TARGET_PROPERTY:${tgt},SOURCES>,|>
definitions=$<JOIN:$<TARGET_PROPERTY:${tgt},COMPILE_DEFINITIONS>,|>
include_dirs=$<JOIN:$<TARGET_PROPERTY:${tgt},INCLUDE_DIRECTORIES>,|>
")
endfunction()

# Records a build of lab `id` that is not made, and why (see above).
function(pmpp_record_skip id v b why)
  file(APPEND ${CMAKE_BINARY_DIR}/pmpp_skipped.txt "${id}|${v}|${b}|${why}\n")
endfunction()

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
  cmake_parse_arguments(L "RACE" "TILE;SHARED_DIR"
    "VARIANTS;EXTRA_TILES;SHARED_CU;SHARED_MM;LIBS;LIBS_EMU;LIBS_RACE;LIBS_METAL;LIBS_CUDA;LIBS_HIP" ${ARGN})
  string(REPLACE "/" "_" base ${id})
  set(dir ${CMAKE_CURRENT_SOURCE_DIR})
  if(NOT L_TILE)
    set(L_TILE 16)
  endif()
  set(shared ${dir}/../shared)
  if(L_SHARED_DIR)
    get_filename_component(shared ${L_SHARED_DIR} ABSOLUTE BASE_DIR ${dir})
  endif()
  # A relative provided source is looked up in SHARED_DIR first, then in the lab.
  foreach(kind CU MM)
    set(resolved "")
    foreach(f ${L_SHARED_${kind}})
      if(NOT IS_ABSOLUTE ${f})
        if(L_SHARED_DIR AND EXISTS ${shared}/${f})
          set(f ${shared}/${f})
        else()
          set(f ${dir}/${f})
        endif()
      endif()
      list(APPEND resolved ${f})
    endforeach()
    set(L_SHARED_${kind} ${resolved})
  endforeach()
  # A lab that is still being written (no driver yet, or a provided source that does
  # not exist yet) is skipped with a note, so that it cannot break every other lab.
  set(missing "")
  set(need ${dir}/main.cpp ${L_SHARED_CU} ${L_SHARED_MM})
  if(L_SHARED_DIR)
    list(APPEND need ${shared})
  endif()
  foreach(f ${need})
    if(NOT EXISTS ${f})
      list(APPEND missing ${f})
    endif()
  endforeach()
  if(missing)
    message(STATUS "pmpp lab ${id}: skipped, incomplete (missing ${missing})")
    file(APPEND ${CMAKE_BINARY_DIR}/pmpp_incomplete.txt "${id}|${missing}\n")
    return()
  endif()

  # Per-backend libraries. A target name that does not exist here leaves that backend out.
  foreach(b ${PMPP_BACKENDS})
    string(TOUPPER ${b} B)
    set(libs_${b} ${L_LIBS} ${L_LIBS_${B}})
    set(nolib_${b} "")
    foreach(lib ${libs_${b}})
      if(lib MATCHES "::" AND NOT TARGET ${lib})
        set(nolib_${b} "the library ${lib} it links on ${b} was not found here")
        break()
      endif()
    endforeach()
  endforeach()

  foreach(v ${L_VARIANTS})
    foreach(b ${PMPP_BACKENDS})
      if(b STREQUAL "race" AND NOT L_RACE)
        pmpp_record_skip(${id} ${v} ${b} "this lab is not built for the race-detection emulator (its CMakeLists.txt does not pass RACE to pmpp_add_lab)")
        continue()
      endif()
      if(nolib_${b})
        message(STATUS "pmpp lab ${id}: no ${b} build of ${v} (${nolib_${b}})")
        pmpp_record_skip(${id} ${v} ${b} "${nolib_${b}}")
        continue()
      endif()
      if(b STREQUAL "metal")
        set(have "")
        foreach(f kernel.metal launch.mm)
          if(NOT EXISTS ${dir}/${v}/${f})
            list(APPEND have ${f})
          endif()
        endforeach()
        if(have)
          string(REPLACE ";" " and " have "${have}")
          pmpp_record_skip(${id} ${v} ${b} "no Metal build for this lab: ${v}/ has no ${have} (the lab is CUDA-family only; its README says why)")
          continue()
        endif()
      elseif(NOT EXISTS ${dir}/${v}/kernel.cu)
        pmpp_record_skip(${id} ${v} ${b} "no ${b} build for this lab: ${v}/ has no kernel.cu (the lab is Metal only)")
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
          set(wrap ${CMAKE_CURRENT_BINARY_DIR}/${tgt}.main.mm)
          file(WRITE ${wrap} "#include \"${dir}/main.cpp\"\n")
          add_executable(${tgt} ${wrap} ${dir}/${v}/launch.mm ${L_SHARED_MM})
          target_compile_definitions(${tgt} PRIVATE PMPP_METAL_SOURCE="${dir}/${v}/kernel.metal"
                                                    PMPP_METAL_SHARED_DIR="${shared}")
        else()
          pmpp_cu_sources(kernel_srcs ${tgt} ${b} ${dir}/${v}/kernel.cu ${L_SHARED_CU})
          add_executable(${tgt} ${dir}/main.cpp ${kernel_srcs})
        endif()

        target_link_libraries(${tgt} PRIVATE pmpp_rt_${b} ${libs_${b}})
        target_include_directories(${tgt} PRIVATE ${dir} ${dir}/${v})
        if(L_SHARED_DIR)
          target_include_directories(${tgt} PRIVATE ${shared})
          target_compile_definitions(${tgt} PRIVATE PMPP_SHARED_DIR="${shared}")
        endif()
        target_compile_definitions(${tgt} PRIVATE TILE_WIDTH=${t}
          PMPP_LAB="${id}" PMPP_VARIANT="${v}")
        set_target_properties(${tgt} PROPERTIES EXCLUDE_FROM_ALL TRUE)
        pmpp_record_inputs(${tgt})
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
    pmpp_record_inputs(${tgt})
    set(expect "clean")
    if(T_EXPECT_RACE AND b STREQUAL "race")
      set(expect "race")
    endif()
    file(APPEND ${CMAKE_BINARY_DIR}/pmpp_selftests.txt
      "${name}|${b}|${expect}|${tgt}|${CMAKE_BINARY_DIR}/bin/${tgt}\n")
  endforeach()
endfunction()
