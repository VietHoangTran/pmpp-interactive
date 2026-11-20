# Lab 20.5 (milestone M7, grouped-query decode). provided/perhead.cmake copies lab 20.4's
# reference solution into the build directory, renamed: the per-head kernel the benchmark
# times beside yours, and the KV append the 16-step loop calls. RACE: the warps merge
# through shared memory, one query head after another.
include(${CMAKE_CURRENT_SOURCE_DIR}/provided/perhead.cmake)
pmpp_add_lab(ch20/lab5_gqa_decode VARIANTS work solution RACE
  SHARED_CU ${pmpp_ch20_perhead_cu}
  SHARED_MM ${pmpp_ch20_perhead_mm})
pmpp_ch20_perhead_link(ch20_lab5_gqa_decode)
