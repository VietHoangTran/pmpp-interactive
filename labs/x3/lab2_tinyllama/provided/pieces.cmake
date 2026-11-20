# Lab X3.2, provided: the project's own pieces, which the Llama step is built from, and the
# test models. Included by ../CMakeLists.txt before pmpp_add_lab; call pmpp_x3l2_link() after.
# Modelled on labs/ch22/lab2_batching/provided/pieces.cmake. Sets:
#   pmpp_x3l2_cu   provided CUDA sources (SHARED_CU)
#   pmpp_x3l2_mm   provided Metal host sources (SHARED_MM)
#
# Each piece is an earlier lab's reference solution, copied into the build directory at
# configure time with its names changed where two pieces would clash, its `#include "../x"`
# lines pointed back at its own lab, and every one-line __device__ helper definition made
# static, so that two pieces may each have, say, a laneIdx. Its Metal kernels are compiled
# at run time from that lab's own kernel.metal; the driver passes the constants that lab's
# driver uses.
#
#   lab 2.3  (M0)  launchAddResidual; encodeAddResidual                               required
#   lab 6.4  (M2)  launchLinear -> launchLinearM2, encodeLinear -> encodeLinearM2     required
#   lab 10.3 (M4)  launchArgmaxRows; encodeArgmaxRows                                 required
#   lab 20.4 (M7)  launchKvAppend, launchDecodeAttention and their encoders           required
#   lab 22.2 (M8)  launchEmbed; encodeEmbed (launchDecodeStep -> launchDecodeStepM8)  required
#   lab X3.1       launchRmsNorm, launchRope, launchSwiGLU and their encoders         required
#   lab 15.6 (M3)  launchLinear -> launchLinearM3, encodeLinear -> encodeLinearM3     if present
#   lab 20.5 (M7)  launchDecodeAttentionGrouped, encodeDecodeAttentionGrouped         if present
#   lab 22.1       launchGemvGather, encodeGemvGather                                 if present
#   lab X1.3 (M5)  launchSampleRows, encodeSampleRows, with its own pieces            if present
#   step_timer.cu  one pair of events around a step (the CUDA and HIP timing)         always
# A required piece that is missing, or whose signature differs from the one llama.h
# declares, leaves the lab skipped, with a note that names it.
#
# Lab X3.1's solution is the first ../../lab1_*/solution, or the directory in the CMake
# variable or environment variable PMPP_X3_PIECES_DIR. M9 (lab X2's quantised linear layer)
# would plug in as one more LinearFn; labs/x2 has no LinearFn-compatible solution yet, so
# M10 runs float32 without M9 (README.md).

get_filename_component(_x3_labs ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)
set(_x3_m0 ${_x3_labs}/ch02/lab3_elementwise/solution)
set(_x3_m2 ${_x3_labs}/ch06/lab4_linear/solution)
set(_x3_m3 ${_x3_labs}/ch15/lab6_linear/solution)
set(_x3_m4 ${_x3_labs}/ch10/lab3_softmax_argmax/solution)
set(_x3_m7 ${_x3_labs}/ch20/lab4_decode/solution)
set(_x3_m7g ${_x3_labs}/ch20/lab5_gqa_decode/solution)
set(_x3_m8 ${_x3_labs}/ch22/lab2_batching/solution)
set(_x3_m5lab ${_x3_labs}/x1/lab3_sampler)
set(_x3_m5 ${_x3_m5lab}/solution)

function(_x3_first_dir out pattern)
  file(GLOB _dirs LIST_DIRECTORIES true ${pattern})
  set(_d "")
  if(_dirs)
    list(SORT _dirs)
    list(GET _dirs 0 _d)
  endif()
  set(${out} ${_d} PARENT_SCOPE)
endfunction()
_x3_first_dir(_x3_gemv "${_x3_labs}/ch22/lab1_*/solution")
if(PMPP_X3_PIECES_DIR)
  set(_x3_x31 ${PMPP_X3_PIECES_DIR})
elseif(DEFINED ENV{PMPP_X3_PIECES_DIR})
  set(_x3_x31 $ENV{PMPP_X3_PIECES_DIR})
else()
  _x3_first_dir(_x3_x31 "${CMAKE_CURRENT_LIST_DIR}/../../lab1_*/solution")
endif()

set(pmpp_x3l2_cu ${CMAKE_CURRENT_LIST_DIR}/step_timer.cu)
set(pmpp_x3l2_mm "")
set(_x3_defs "")          # every backend
set(_x3_metal_defs "")    # Metal only
set(_x3_includes "")
set(_x3_note "// Generated at configure time by labs/x3/lab2_tinyllama/provided/pieces.cmake")

# Writes `content` to `path` only if it differs, so a reconfigure does not force a rebuild.
function(_x3_write path content)
  if(EXISTS ${path})
    file(READ ${path} old)
    if(old STREQUAL content)
      return()
    endif()
  endif()
  file(WRITE ${path} "${content}")
endfunction()

# Copies `src` to `dst` in the build directory, replacing each OLD NEW pair in ARGN, pointing
# `#include "../` at the source's own directory, and making one-line __device__ helper
# definitions static (a declaration ending in `;` keeps its linkage). Appends `dst` to `list`.
function(_x3_copy list src dst)
  file(READ ${src} text)
  set(pairs ${ARGN})
  list(LENGTH pairs n)
  set(i 0)
  while(i LESS n)
    math(EXPR j "${i} + 1")
    list(GET pairs ${i} old)
    list(GET pairs ${j} new)
    string(REPLACE "${old}" "${new}" text "${text}")
    math(EXPR i "${i} + 2")
  endwhile()
  get_filename_component(srcdir ${src} DIRECTORY)
  string(REPLACE "#include \"../" "#include \"${srcdir}/../" text "${text}")
  string(REGEX REPLACE "\n__device__ ([^\n;]*\\)[ \t]*\\{)" "\nstatic __device__ \\1" text "${text}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${src})
  _x3_write(${CMAKE_CURRENT_BINARY_DIR}/${dst} "${_x3_note}\n// from ${src}. Do not edit.\n${text}")
  set(${list} ${${list}} ${CMAKE_CURRENT_BINARY_DIR}/${dst} PARENT_SCOPE)
endfunction()

# A required piece that is not usable: name a file that does not exist, so pmpp_add_lab skips
# the lab and lists it in pmpp_incomplete.txt, and say why.
macro(_x3_missing why)
  message(STATUS "pmpp lab x3/lab2_tinyllama: ${why}")
  string(MAKE_C_IDENTIFIER "${why}" _x3_id)
  string(SUBSTRING "${_x3_id}" 0 120 _x3_id)
  list(APPEND pmpp_x3l2_cu ${CMAKE_CURRENT_BINARY_DIR}/missing-piece/${_x3_id})
endmacro()
macro(_x3_required dir name)
  foreach(_f kernel.cu kernel.metal launch.mm)
    if(NOT EXISTS ${dir}/${_f})
      _x3_missing("${name}'s solution has no ${_f} (${dir})")
    endif()
  endforeach()
endmacro()

# A signature as a regular expression: return type, name, then one entry per parameter type
# ("const float*", "int", "id<MTLBuffer>", ...), each followed by any parameter name.
set(_ws "[ \t\r\n]")
function(_x3_sig out ret name)
  set(re "${ret}${_ws}+${name}${_ws}*\\(${_ws}*")
  set(first TRUE)
  foreach(t ${ARGN})
    string(REPLACE " " "${_ws}+" p "${t}")
    string(REPLACE "*" "${_ws}*\\*" p "${p}")
    if(NOT first)
      set(re "${re},${_ws}*")
    endif()
    set(re "${re}${p}[^,()]*")
    set(first FALSE)
  endforeach()
  set(${out} "${re}\\)" PARENT_SCOPE)
endfunction()
set(_enc "id<MTLComputeCommandEncoder>" "id<MTLComputePipelineState>")
set(_B "id<MTLBuffer>")

# ---- lab 2.3, M0
_x3_required(${_x3_m0} "lab 2.3")
if(EXISTS ${_x3_m0}/kernel.cu AND EXISTS ${_x3_m0}/launch.mm)
  _x3_copy(pmpp_x3l2_cu ${_x3_m0}/kernel.cu m0_elementwise.cu)
  _x3_copy(pmpp_x3l2_mm ${_x3_m0}/launch.mm m0_elementwise_launch.mm)
  list(APPEND _x3_metal_defs PMPP_M0_METAL="${_x3_m0}/kernel.metal")
endif()

# ---- lab 6.4, M2
_x3_required(${_x3_m2} "lab 6.4")
if(EXISTS ${_x3_m2}/kernel.cu AND EXISTS ${_x3_m2}/launch.mm)
  _x3_copy(pmpp_x3l2_cu ${_x3_m2}/kernel.cu m2_linear.cu launchLinear launchLinearM2 linearKernel linearM2Kernel)
  _x3_copy(pmpp_x3l2_mm ${_x3_m2}/launch.mm m2_linear_launch.mm encodeLinear encodeLinearM2 LinearDims LinearM2Dims)
  list(APPEND _x3_metal_defs PMPP_M2_METAL="${_x3_m2}/kernel.metal")
endif()

# ---- lab 10.3, M4
_x3_required(${_x3_m4} "lab 10.3")
if(EXISTS ${_x3_m4}/kernel.cu AND EXISTS ${_x3_m4}/launch.mm)
  _x3_copy(pmpp_x3l2_cu ${_x3_m4}/kernel.cu m4_rowwise.cu)
  _x3_copy(pmpp_x3l2_mm ${_x3_m4}/launch.mm m4_rowwise_launch.mm)
  list(APPEND _x3_metal_defs PMPP_M4_METAL="${_x3_m4}/kernel.metal")
endif()

# ---- lab 20.4, M7
_x3_required(${_x3_m7} "lab 20.4")
if(EXISTS ${_x3_m7}/kernel.cu AND EXISTS ${_x3_m7}/launch.mm)
  _x3_copy(pmpp_x3l2_cu ${_x3_m7}/kernel.cu m7_decode.cu)
  _x3_copy(pmpp_x3l2_mm ${_x3_m7}/launch.mm m7_decode_launch.mm)
  list(APPEND _x3_metal_defs PMPP_M7_METAL="${_x3_m7}/kernel.metal")
endif()

# ---- lab 22.2, M8 (its launchDecodeStep is renamed so that nothing mistakes it for yours)
_x3_required(${_x3_m8} "lab 22.2")
if(EXISTS ${_x3_m8}/kernel.cu AND EXISTS ${_x3_m8}/launch.mm)
  _x3_copy(pmpp_x3l2_cu ${_x3_m8}/kernel.cu m8_embed.cu launchDecodeStep launchDecodeStepM8)
  _x3_copy(pmpp_x3l2_mm ${_x3_m8}/launch.mm m8_embed_launch.mm encodeDecodeStep encodeDecodeStepM8)
  list(APPEND _x3_metal_defs PMPP_M8_METAL="${_x3_m8}/kernel.metal")
endif()

# ---- lab X3.1: RMSNorm, RoPE, SwiGLU, with exactly the signatures llama.h declares
_x3_sig(_s1 void launchRmsNorm "const float*" "const float*" "float*" int int float)
_x3_sig(_s2 void launchRope "float*" "const int*" "const float*" "const float*" int int int int)
_x3_sig(_s3 void launchSwiGLU "const float*" "const float*" "float*" int)
_x3_sig(_m1 bool encodeRmsNorm ${_enc} ${_B} ${_B} ${_B} uint32_t uint32_t float)
_x3_sig(_m2 void encodeRope ${_enc} ${_B} ${_B} ${_B} ${_B} uint32_t uint32_t uint32_t uint32_t)
_x3_sig(_m3 void encodeSwiGLU ${_enc} ${_B} ${_B} ${_B} uint32_t)
if(NOT _x3_x31)
  _x3_missing("lab X3.1's solution (labs/x3/lab1_*/solution) was not found; set PMPP_X3_PIECES_DIR to use another")
else()
  _x3_required(${_x3_x31} "lab X3.1")
  if(EXISTS ${_x3_x31}/kernel.cu AND EXISTS ${_x3_x31}/launch.mm AND EXISTS ${_x3_x31}/kernel.metal)
    file(READ ${_x3_x31}/kernel.cu _cu)
    file(READ ${_x3_x31}/launch.mm _mm)
    file(READ ${_x3_x31}/kernel.metal _msl)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                 ${_x3_x31}/kernel.cu ${_x3_x31}/launch.mm ${_x3_x31}/kernel.metal)
    set(_x3_ok TRUE)
    foreach(_p "_s1;launchRmsNorm" "_s2;launchRope" "_s3;launchSwiGLU")
      list(GET _p 0 _v)
      list(GET _p 1 _n)
      if(NOT _cu MATCHES "${${_v}}")
        _x3_missing("lab X3.1's kernel.cu has no ${_n} with the signature llama.h declares")
        set(_x3_ok FALSE)
      endif()
    endforeach()
    foreach(_p "_m1;encodeRmsNorm" "_m2;encodeRope" "_m3;encodeSwiGLU")
      list(GET _p 0 _v)
      list(GET _p 1 _n)
      if(NOT _mm MATCHES "${${_v}}")
        _x3_missing("lab X3.1's launch.mm has no ${_n} with the signature llama.h declares")
        set(_x3_ok FALSE)
      endif()
    endforeach()
    foreach(_k rms_norm_rows rope_rows swiglu)
      if(NOT _msl MATCHES "kernel${_ws}+void${_ws}+${_k}${_ws}*\\(")
        _x3_missing("lab X3.1's kernel.metal has no kernel ${_k}")
        set(_x3_ok FALSE)
      endif()
    endforeach()
    if(_x3_ok)
      _x3_copy(pmpp_x3l2_cu ${_x3_x31}/kernel.cu x31_pieces.cu)
      _x3_copy(pmpp_x3l2_mm ${_x3_x31}/launch.mm x31_pieces_launch.mm)
      list(APPEND _x3_metal_defs PMPP_X31_METAL="${_x3_x31}/kernel.metal")
      list(APPEND _x3_defs PMPP_X31_DIR="${_x3_x31}")
    endif()
  endif()
endif()

# ---- lab 15.6, M3 (optional)
if(EXISTS ${_x3_m3}/kernel.cu AND EXISTS ${_x3_m3}/launch.mm AND EXISTS ${_x3_m3}/kernel.metal)
  _x3_copy(pmpp_x3l2_cu ${_x3_m3}/kernel.cu m3_linear.cu launchLinear launchLinearM3)
  _x3_copy(pmpp_x3l2_mm ${_x3_m3}/launch.mm m3_linear_launch.mm encodeLinear encodeLinearM3 LinearDims LinearM3Dims)
  list(APPEND _x3_defs PMPP_HAVE_M3=1)
  list(APPEND _x3_metal_defs PMPP_M3_METAL="${_x3_m3}/kernel.metal")
else()
  message(STATUS "pmpp lab x3/lab2_tinyllama: lab 15.6's solution not found in ${_x3_m3}; no m3 rows")
endif()

# ---- lab 20.5, the grouped decode kernel (optional)
_x3_sig(_g1 void launchDecodeAttentionGrouped "const float*" "const float*" "const float*" "const int*" "float*"
        int int int int float)
if(EXISTS ${_x3_m7g}/kernel.cu AND EXISTS ${_x3_m7g}/launch.mm AND EXISTS ${_x3_m7g}/kernel.metal)
  file(READ ${_x3_m7g}/kernel.cu _cu)
  if(_cu MATCHES "${_g1}")
    _x3_copy(pmpp_x3l2_cu ${_x3_m7g}/kernel.cu m7_grouped.cu)
    _x3_copy(pmpp_x3l2_mm ${_x3_m7g}/launch.mm m7_grouped_launch.mm)
    list(APPEND _x3_defs PMPP_HAVE_GQA=1)
    list(APPEND _x3_metal_defs PMPP_M7G_METAL="${_x3_m7g}/kernel.metal")
  endif()
endif()

# ---- lab 22.1, the gather (optional; only with the linear layer's signature)
_x3_sig(_v1 void launchGemvGather "const float*" "const float*" "float*" int int int)
_x3_sig(_v2 void encodeGemvGather ${_enc} ${_B} ${_B} ${_B} uint32_t uint32_t uint32_t)
if(_x3_gemv AND EXISTS ${_x3_gemv}/kernel.cu AND EXISTS ${_x3_gemv}/launch.mm AND EXISTS ${_x3_gemv}/kernel.metal)
  file(READ ${_x3_gemv}/kernel.cu _cu)
  file(READ ${_x3_gemv}/launch.mm _mm)
  file(READ ${_x3_gemv}/kernel.metal _msl)
  if(_cu MATCHES "${_v1}" AND _mm MATCHES "${_v2}" AND _msl MATCHES "kernel${_ws}+void${_ws}+gemv_gather${_ws}*\\(")
    _x3_copy(pmpp_x3l2_cu ${_x3_gemv}/kernel.cu gemv_gather.cu)
    _x3_copy(pmpp_x3l2_mm ${_x3_gemv}/launch.mm gemv_gather_launch.mm)
    list(APPEND _x3_defs PMPP_HAVE_GEMV=1)
    list(APPEND _x3_metal_defs PMPP_GEMV_METAL="${_x3_gemv}/kernel.metal")
  else()
    message(STATUS "pmpp lab x3/lab2_tinyllama: ${_x3_gemv} has no launchGemvGather/encodeGemvGather/gemv_gather "
                   "with the linear layer's signature; no gemv rows")
  endif()
endif()

# ---- lab X1.3, the sampler (optional). It is built from its own copies of earlier labs
# (labs/x1/shared-g11) and, on Metal, from those labs' solutions, as its CMakeLists.txt does;
# its copy of lab 10.3 is left out, since lab 10.3 itself is linked above.
_x3_sig(_x1 void launchSampleRows "const float*" "const unsigned int*" "const unsigned long long*"
        "const unsigned long long*" "const unsigned long long*" "int*" "float*" "void*" int int int float float
        float float float float)
set(_x3_g11 ${_x3_labs}/x1/shared-g11)
set(_x3_m5_cu penalties_g11.cu topp_g11.cu sortrows_g11.cu topk_g11.cu minp_g11.cu)
set(_x3_m5_mm ${_x3_labs}/ch09/lab3_tokencounts/solution/launch.mm ${_x3_labs}/ch11/lab6_topp/solution/launch.mm
              ${_x3_labs}/ch14/lab3_sortrows/solution/launch.mm ${_x3_g11}/topk_g11_launch.mm
              ${_x3_g11}/minp_g11_launch.mm ${_x3_m5}/launch.mm)
set(_x3_m5_ok FALSE)
if(EXISTS ${_x3_m5}/kernel.cu AND EXISTS ${_x3_m5}/launch.mm AND EXISTS ${_x3_m5lab}/sampler_metal.h)
  file(READ ${_x3_m5}/kernel.cu _cu)
  set(_x3_m5_ok TRUE)
  if(NOT _cu MATCHES "${_x1}")
    set(_x3_m5_ok FALSE)
  endif()
  foreach(_f ${_x3_m5_cu})
    if(NOT EXISTS ${_x3_g11}/${_f})
      set(_x3_m5_ok FALSE)
    endif()
  endforeach()
  foreach(_f ${_x3_m5_mm})
    if(NOT EXISTS ${_f})
      set(_x3_m5_ok FALSE)
    endif()
  endforeach()
endif()
if(_x3_m5_ok)
  _x3_copy(pmpp_x3l2_cu ${_x3_m5}/kernel.cu m5_sampler.cu)
  foreach(_f ${_x3_m5_cu})
    _x3_copy(pmpp_x3l2_cu ${_x3_g11}/${_f} m5_${_f})
  endforeach()
  set(_k 0)
  foreach(_f ${_x3_m5_mm})
    math(EXPR _k "${_k} + 1")
    _x3_copy(pmpp_x3l2_mm ${_f} m5_launch_${_k}.mm)
  endforeach()
  list(APPEND _x3_defs PMPP_HAVE_M5=1)
  list(APPEND _x3_metal_defs PMPP_M5_METAL="${_x3_m5}/kernel.metal" PMPP_LABS_DIR="${_x3_labs}")
  list(APPEND _x3_includes ${_x3_m5lab})
else()
  message(STATUS "pmpp lab x3/lab2_tinyllama: lab X1.3's solution and pieces not all found; no sampler test")
endif()

# ---- the test models: make_model.py at configure time, into the build directory
# Seeds and the checksums make_model.py printed for them when this lab was written (they
# check that the generator writes the same bytes on every machine). PMPP_X3_SEED in the
# environment replaces both seeds; the checksum check then says it was not run.
set(_x3_seed_emu 11)
set(_x3_seed_gpu 1)
set(_x3_expect_emu "472374e93a74f773")
set(_x3_expect_gpu "15505ca2bc0f1b11")
if(DEFINED ENV{PMPP_X3_SEED})
  set(_x3_seed_emu $ENV{PMPP_X3_SEED})
  set(_x3_seed_gpu $ENV{PMPP_X3_SEED})
endif()
set(_x3_script ${CMAKE_CURRENT_LIST_DIR}/make_model.py)
file(SHA256 ${_x3_script} _x3_script_hash)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_x3_script})
set(_x3_model_dir ${CMAKE_CURRENT_BINARY_DIR}/models)
file(MAKE_DIRECTORY ${_x3_model_dir})

# Runs make_model.py with ARGN into models/<name>.bin unless the same script and arguments
# made it already; sets <name>_path and <name>_checksum (the value the script printed).
function(_x3_model name)
  set(bin ${_x3_model_dir}/${name}.bin)
  set(log ${_x3_model_dir}/${name}.txt)
  set(stamp "${_x3_script_hash} ${ARGN}")
  set(old "")
  if(EXISTS ${bin}.stamp)
    file(READ ${bin}.stamp old)
  endif()
  if(NOT EXISTS ${bin} OR NOT EXISTS ${log} OR NOT old STREQUAL stamp)
    string(REPLACE ";" " " _args "${ARGN}")
    message(STATUS "pmpp lab x3/lab2_tinyllama: writing models/${name}.bin (make_model.py ${_args})")
    execute_process(COMMAND ${Python3_EXECUTABLE} ${_x3_script} ${ARGN} --out ${bin}
                    OUTPUT_VARIABLE outp ERROR_VARIABLE errp RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
      message(STATUS "pmpp lab x3/lab2_tinyllama: make_model.py failed: ${errp}")
      file(REMOVE ${bin} ${log} ${bin}.stamp)
    else()
      file(WRITE ${log} "${outp}")
      file(WRITE ${bin}.stamp "${stamp}")
    endif()
  endif()
  set(sum "")
  if(EXISTS ${log})
    file(READ ${log} outp)
    string(REGEX MATCH "checksum: ([0-9a-f]+)" _m "${outp}")
    set(sum ${CMAKE_MATCH_1})
  endif()
  set(${name}_path ${bin} PARENT_SCOPE)
  set(${name}_checksum "${sum}" PARENT_SCOPE)
endfunction()

if(Python3_EXECUTABLE)
  set(_x3_emu_shape --dim 256 --hidden 128 --layers 2 --heads 2 --kv-heads 1 --vocab 256 --seq-len 16)
  set(_x3_gpu_shape --dim 512 --hidden 1376 --layers 4 --heads 4 --kv-heads 2 --vocab 4099 --seq-len 128)
  _x3_model(emu ${_x3_emu_shape} --seed ${_x3_seed_emu})
  _x3_model(emu_perm ${_x3_emu_shape} --seed ${_x3_seed_emu} --permute-qk)
  _x3_model(gpu ${_x3_gpu_shape} --seed ${_x3_seed_gpu})
  _x3_model(gpu_perm ${_x3_gpu_shape} --seed ${_x3_seed_gpu} --permute-qk)
  foreach(_n emu emu_perm gpu gpu_perm)
    string(TOUPPER ${_n} _u)
    list(APPEND _x3_defs PMPP_X3_MODEL_${_u}="${${_n}_path}" PMPP_X3_SUM_${_u}="${${_n}_checksum}")
  endforeach()
  list(APPEND _x3_defs PMPP_X3_SEED_EMU=${_x3_seed_emu} PMPP_X3_SEED_GPU=${_x3_seed_gpu}
       PMPP_X3_EXPECT_EMU="${_x3_expect_emu}" PMPP_X3_EXPECT_GPU="${_x3_expect_gpu}"
       PMPP_X3_PYTHON="${Python3_EXECUTABLE}" PMPP_X3_MAKE_MODEL="${_x3_script}"
       PMPP_X3_MODEL_DIR="${_x3_model_dir}")
else()
  _x3_missing("no Python 3 to run provided/make_model.py")
endif()
get_filename_component(_x3_repo ${_x3_labs}/.. ABSOLUTE)
list(APPEND _x3_defs PMPP_X3_REPORTS="${_x3_repo}/reports/runs")

# Adds the definitions above to every target of lab `base` (x3_lab2_tinyllama).
function(pmpp_x3l2_link base)
  foreach(v work solution)
    foreach(b ${PMPP_BACKENDS})
      set(tgt ${base}.${v}.${b})
      if(NOT TARGET ${tgt})
        continue()
      endif()
      target_compile_definitions(${tgt} PRIVATE ${_x3_defs})
      if(_x3_includes)
        target_include_directories(${tgt} PRIVATE ${_x3_includes})
      endif()
      if(b STREQUAL "metal" AND _x3_metal_defs)
        target_compile_definitions(${tgt} PRIVATE ${_x3_metal_defs})
      endif()
    endforeach()
  endforeach()
endfunction()
