# Labs X2.3 and X2.4, provided: where the float quantised GEMVs come from.
#
#   pmpp_x2g12_qgemv(<prefix> <lab name> [Q8_0] [Q4_0])
#
# For each format asked for, uses the reference solution of the lab that builds it, if that
# solution exists with exactly the shape these labs call, and otherwise a stand-in from this
# directory (gemv_q8_0_g12.*, gemv_q4_0_g12.*, gemv_q_g12_launch.mm):
#
#   Q8_0, lab X2.1 (first ../lab1_*/solution, or -DPMPP_X2_Q8_0_DIR=<dir>)
#     void launchGemvQ8_0(const float* X, const int8_t* Wq, const uint16_t* Wd, float* Y,
#                         int batch, int in, int out)
#     void encodeGemvQ8_0(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
#                         id<MTLBuffer> X, id<MTLBuffer> Wq, id<MTLBuffer> Wd, id<MTLBuffer> Y,
#                         uint32_t batch, uint32_t in, uint32_t out)
#     and a Metal kernel named gemv_q8_0 (or a kernel.metal with a single kernel)
#   Q4_0, lab X2.2 (first ../lab2_*/solution, or -DPMPP_X2_Q4_0_DIR=<dir>)
#     void launchGemvQ4_0(const float* X, const BlockQ4_0* W, float* Y, int batch, int in, int out)
#     void encodeGemvQ4_0(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
#                         id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
#                         uint32_t batch, uint32_t in, uint32_t out)
#     and a Metal kernel named gemv_q4_0 (or a kernel.metal with a single kernel)
#
# A copied solution goes into the build directory with its one-line __device__ helper
# definitions made static (as lab 22.2's pieces.cmake does), and its own directory on the
# include path. Sets, in the caller's scope:
#   <prefix>_cu         CUDA sources (SHARED_CU)
#   <prefix>_mm         Metal host sources (SHARED_MM)
#   <prefix>_defs       definitions for every backend: PMPP_Q8_0_FROM, PMPP_Q4_0_FROM
#                       ("the reference solution of lab X2.1" or "the stand-in ...")
#   <prefix>_metal_defs Metal only: PMPP_Q8_0_METAL, PMPP_Q8_0_KERNEL, PMPP_Q4_0_METAL,
#                       PMPP_Q4_0_KERNEL, G12_WANT_Q8_0, G12_WANT_Q4_0
#   <prefix>_includes   include directories

set(_x2g12_dir ${CMAKE_CURRENT_LIST_DIR})
get_filename_component(_x2g12_x2 ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)

function(_x2g12_write path content)
  if(EXISTS ${path})
    file(READ ${path} old)
    if(old STREQUAL content)
      return()
    endif()
  endif()
  file(WRITE ${path} "${content}")
endfunction()

# Copies src to the binary dir as dst with one-line __device__ definitions made static.
function(_x2g12_copy outvar src dst)
  file(READ ${src} text)
  string(REGEX REPLACE "\n__device__ ([^\n;]*\\)[ \t]*\\{)" "\nstatic __device__ \\1" text "${text}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${src})
  _x2g12_write(${CMAKE_CURRENT_BINARY_DIR}/${dst}
               "// Generated at configure time by labs/x2/shared-g12/qpieces.cmake\n// from ${src}. Do not edit.\n${text}")
  set(${outvar} ${CMAKE_CURRENT_BINARY_DIR}/${dst} PARENT_SCOPE)
endfunction()

# The first directory matching pattern, or "".
function(_x2g12_first outvar pattern)
  file(GLOB dirs LIST_DIRECTORIES true ${pattern})
  set(r "")
  if(dirs)
    list(SORT dirs)
    list(GET dirs 0 r)
  endif()
  set(${outvar} ${r} PARENT_SCOPE)
endfunction()

# The kernel to compile from a kernel.metal: `want` if present, or its only kernel.
function(_x2g12_kernel outvar msl want)
  set(ws "[ \t\r\n]")
  string(REGEX MATCHALL "kernel${ws}+void${ws}+[A-Za-z_0-9]+" ks "${msl}")
  list(LENGTH ks nk)
  set(r "")
  foreach(k ${ks})
    string(REGEX REPLACE "^kernel${ws}+void${ws}+" "" k "${k}")
    if(k STREQUAL want OR nk EQUAL 1)
      set(r ${k})
    endif()
  endforeach()
  set(${outvar} "${r}" PARENT_SCOPE)
endfunction()

function(pmpp_x2g12_qgemv prefix labname)
  cmake_parse_arguments(Q "Q8_0;Q4_0" "" "" ${ARGN})
  set(ws "[ \t\r\n]")
  set(cu "")
  set(mm "")
  set(defs "")
  set(mdefs "")
  set(incs "")
  set(stand_in_mm FALSE)

  foreach(fmt Q8_0 Q4_0)
    if(NOT Q_${fmt})
      continue()
    endif()
    if(fmt STREQUAL "Q8_0")
      set(lab "lab X2.1")
      set(fn launchGemvQ8_0)
      set(enc encodeGemvQ8_0)
      set(kern gemv_q8_0)
      set(stem gemv_q8_0_g12)
      set(cu_sig "void${ws}+launchGemvQ8_0${ws}*\\(${ws}*const${ws}+float${ws}*\\*[^,]*,${ws}*const${ws}+int8_t${ws}*\\*[^,]*,${ws}*const${ws}+uint16_t${ws}*\\*[^,]*,${ws}*float${ws}*\\*[^,]*,${ws}*int[^,]*,${ws}*int[^,]*,${ws}*int[^,)]*\\)")
      set(mm_sig "encodeGemvQ8_0${ws}*\\(${ws}*id<MTLComputeCommandEncoder>[^,]*,${ws}*id<MTLComputePipelineState>[^,]*,${ws}*id<MTLBuffer>[^,]*,${ws}*id<MTLBuffer>[^,]*,${ws}*id<MTLBuffer>[^,]*,${ws}*id<MTLBuffer>[^,]*,${ws}*uint32_t[^,]*,${ws}*uint32_t[^,]*,${ws}*uint32_t[^,)]*\\)")
      if(PMPP_X2_Q8_0_DIR)
        set(sol ${PMPP_X2_Q8_0_DIR})
      else()
        _x2g12_first(sol ${_x2g12_x2}/lab1_*/solution)
      endif()
    else()
      set(lab "lab X2.2")
      set(fn launchGemvQ4_0)
      set(enc encodeGemvQ4_0)
      set(kern gemv_q4_0)
      set(stem gemv_q4_0_g12)
      set(cu_sig "void${ws}+launchGemvQ4_0${ws}*\\(${ws}*const${ws}+float${ws}*\\*[^,]*,${ws}*const${ws}+BlockQ4_0${ws}*\\*[^,]*,${ws}*float${ws}*\\*[^,]*,${ws}*int[^,]*,${ws}*int[^,]*,${ws}*int[^,)]*\\)")
      set(mm_sig "encodeGemvQ4_0${ws}*\\(${ws}*id<MTLComputeCommandEncoder>[^,]*,${ws}*id<MTLComputePipelineState>[^,]*,${ws}*id<MTLBuffer>[^,]*,${ws}*id<MTLBuffer>[^,]*,${ws}*id<MTLBuffer>[^,]*,${ws}*uint32_t[^,]*,${ws}*uint32_t[^,]*,${ws}*uint32_t[^,)]*\\)")
      if(PMPP_X2_Q4_0_DIR)
        set(sol ${PMPP_X2_Q4_0_DIR})
      else()
        _x2g12_first(sol ${_x2g12_x2}/lab2_*/solution)
      endif()
    endif()
    string(TOLOWER ${fmt} lfmt)

    # ---- CUDA, HIP, emulators
    set(used_cu FALSE)
    if(sol AND EXISTS ${sol}/kernel.cu)
      file(READ ${sol}/kernel.cu src)
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${sol}/kernel.cu)
      if(src MATCHES "${cu_sig}")
        _x2g12_copy(copied ${sol}/kernel.cu x2_${lfmt}_gemv.cu)
        list(APPEND cu ${copied})
        list(APPEND incs ${sol})
        list(APPEND defs "PMPP_${fmt}_FROM=\"the reference solution of ${lab}\"")
        set(used_cu TRUE)
      else()
        message(STATUS "pmpp ${labname}: ${sol}/kernel.cu has no ${fn} with the shape this lab calls; "
                       "using the stand-in in labs/x2/shared-g12")
      endif()
    endif()
    if(NOT used_cu)
      list(APPEND cu ${_x2g12_dir}/${stem}.cu)
      list(APPEND defs "PMPP_${fmt}_FROM=\"the stand-in in labs/x2/shared-g12 (no solution of ${lab} with this shape)\"")
    endif()

    # ---- Metal
    set(used_mm FALSE)
    if(sol AND EXISTS ${sol}/launch.mm AND EXISTS ${sol}/kernel.metal)
      file(READ ${sol}/launch.mm src)
      file(READ ${sol}/kernel.metal msl)
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${sol}/launch.mm ${sol}/kernel.metal)
      _x2g12_kernel(k "${msl}" ${kern})
      if(src MATCHES "${mm_sig}" AND k)
        _x2g12_copy(copied ${sol}/launch.mm x2_${lfmt}_gemv_launch.mm)
        list(APPEND mm ${copied})
        list(APPEND incs ${sol})
        list(APPEND mdefs "PMPP_${fmt}_METAL=\"${sol}/kernel.metal\"" "PMPP_${fmt}_KERNEL=\"${k}\"")
        set(used_mm TRUE)
      else()
        message(STATUS "pmpp ${labname}: ${sol} has no ${enc} with the shape this lab calls, or no kernel "
                       "${kern}; using the stand-in in labs/x2/shared-g12 on Metal")
      endif()
    endif()
    if(NOT used_mm)
      set(stand_in_mm TRUE)
      list(APPEND mdefs "PMPP_${fmt}_METAL=\"${_x2g12_dir}/${stem}.metal\"" "PMPP_${fmt}_KERNEL=\"${kern}\""
                        G12_WANT_${fmt})
    endif()
  endforeach()
  if(stand_in_mm)
    list(APPEND mm ${_x2g12_dir}/gemv_q_g12_launch.mm)
  endif()

  set(${prefix}_cu ${cu} PARENT_SCOPE)
  set(${prefix}_mm ${mm} PARENT_SCOPE)
  set(${prefix}_defs ${defs} PARENT_SCOPE)
  set(${prefix}_metal_defs ${mdefs} PARENT_SCOPE)
  set(${prefix}_includes ${incs} PARENT_SCOPE)
endfunction()

# Adds what pmpp_x2g12_qgemv found (and any extra lists) to every target of lab `base`.
#   pmpp_x2g12_link(<base> DEFS ... METAL_DEFS ... INCLUDES ...)
function(pmpp_x2g12_link base)
  cmake_parse_arguments(L "" "" "DEFS;METAL_DEFS;INCLUDES" ${ARGN})
  foreach(v work solution)
    foreach(b ${PMPP_BACKENDS})
      set(tgt ${base}.${v}.${b})
      if(NOT TARGET ${tgt})
        continue()
      endif()
      if(L_DEFS)
        target_compile_definitions(${tgt} PRIVATE ${L_DEFS})
      endif()
      if(L_INCLUDES)
        target_include_directories(${tgt} PRIVATE ${L_INCLUDES})
      endif()
      if(b STREQUAL "metal" AND L_METAL_DEFS)
        target_compile_definitions(${tgt} PRIVATE ${L_METAL_DEFS})
      endif()
    endforeach()
  endforeach()
endfunction()
