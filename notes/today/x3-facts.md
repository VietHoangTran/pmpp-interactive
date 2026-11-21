# Module X3 fact check (reading real backends): llama.cpp, MLX, PyTorch, and the M10 checkpoint formats

Checked on 2026-11-20 against the live project sources and vendor documents listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md` and `x1-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY, and judges the claim in the item title (the assumption a course author or the reader would most likely bring). Where an item has several sub-claims, the text says which one failed.
- **Quote** is copied from the source (raw files at the pinned commit below, HF Hub API JSON, the MSL PDF flattened to text, PyTorch's HTML docs flattened to text). Line breaks inside quotes are removed.
- "ch10 D1", "infra D3" and so on point at items in `notes/today/ch10-facts.md`, `infra-facts.md`, etc. Anything settled there is cited, not re-checked.
- **There is no `notes/x3.md`.** Module X3 is ours: PLAN.md lists it as "Reading real backends (not in the book)", goal 3, milestone M10 "run a small real checkpoint and compare tokens with a reference implementation". So part (a) of the brief (do the book's dated statements still hold) has nothing to check, and there is no "Unclear in the PDF" list. Every lesson claim built on this file is labelled ours, and each claim about a codebase as "read in the facts pass on 2026-11-20 at commit ...".
- **Snapshot.** Every project file below was fetched from `raw.githubusercontent.com` at the commit shown here, and each file's "last commit" date is the GitHub API's last commit touching that path at or before that commit. Lessons should link to these commits (permalinks), not to `master`/`main`, so excerpts and line numbers stay valid.

| Project | Branch | Commit read (committed) | Licence (GitHub API / LICENSE file) |
|---|---|---|---|
| llama.cpp `ggml-org/llama.cpp` | `master` | `919794500475862f6768c353255469dc661cb734` (2026-11-20) | MIT, "Copyright (c) 2023-2026 The ggml authors" |
| MLX `ml-explore/mlx` | `main` | `bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a` (2026-11-20) | MIT, "Copyright © 2023 Apple Inc." |
| PyTorch `pytorch/pytorch` | `main` | `a6ec1fa444f247690030fe73106156341c95fcc3` (2026-11-20) | BSD-style three clause (API reports `NOASSERTION`; see A4) |
| mlx-lm `ml-explore/mlx-lm` | `main` | `2d4df8d94a80c0f9f8794e54fafee08728b04fc6` (2026-11-19) | MIT |
| transformers `huggingface/transformers` | `main` | `4d9ef623e3b0a1fb6c12891a907e0e50d41f0af8` (2026-11-20) | Apache-2.0 |
| safetensors `huggingface/safetensors` | `main` | `11319d360db5170565db914601fbf375762bf8eb` (2026-11-20) | Apache-2.0 |
| ggml `ggml-org/ggml` (GGUF spec) | `master` | `d9fe7bff3d2bf4d077e25c654486d02c321734aa` | MIT |
| llama2.c `karpathy/llama2.c` | `master` | `350e04fe35433e6d2941dce5a1f53308f87058eb` (2024-05-29; last push 2024-08-06) | MIT |

- Also used: Metal Shading Language Specification "Version 4.1" (PDF dated 2026-06-04, downloaded again today); PyTorch docs 2.15; Hugging Face Hub API for `karpathy/tinyllamas` (revision `0bd21da7...`, 2023-08-15) and `HuggingFaceTB/SmolLM2-135M` (revision `93efa2f0...`, 2025-02-06). Two checkpoint headers were read by HTTP range request (28 bytes and 30 KB) into the session scratchpad, outside the repository.
- Not checked: anything by running code on a GPU, or building any of the projects. What a kernel "does" below is our reading of the source, labelled as such.

Totals: 17 CONFIRMED, 5 WRONG, 6 PARTLY, 0 COULD NOT VERIFY (28 items: 4 reading method, 9 tour, 2 Metal mechanisms, 9 M10 formats and references, 2 arithmetic, 2 contributing). Sub-claims that could not be verified are called out inside A4, B5 and D4 and collected under "Summary of corrections"; D6 is a reading of source, not a run.

---

## Already settled elsewhere (cite, do not re-check)

| X3 needs | Settled in |
|---|---|
| llama.cpp directory layout: one directory per backend; `ggml-hip` is build files only | ch01 D1, ch24 D1, infra D2 |
| llama.cpp HIP: `vendors/hip.h` renames, `common.cuh` vendor selection, `ggml-hip/CMakeLists.txt` compiles `../ggml-cuda/*.cu` as HIP, ROCm 6.3 minimum | ch05 B9 (re-read today: unchanged, `hip.h` last commit 2026-11-16, CMake line 54 to 55 "At least ROCM/HIP V6.3 is required") |
| llama.cpp logical warp 32 (`#define WARP_SIZE 32`) vs `ggml_cuda_get_physical_warp_size()` (64 on GFX8/9) | ch04 D1, ch15 D1 |
| llama.cpp typed lane mask `ggml_cuda_thread_mask_t` (32-bit CUDA, 64-bit HIP) | ch05 B9, ch10 D1 |
| llama.cpp programmatic dependent launch (`ggml_cuda_pdl_sync`, `ggml_cuda_pdl_lc`), CUDA only | ch18 D3, ch18 B5 |
| llama.cpp CUDA graphs and HIP graphs | ch22 D7, ch18 D3 |
| llama.cpp CUDA kernel file map (`mmq`, `mmf`, `mmvq`, `mmvf`, `mma.cuh`, `fattn-*`, `softmax.cu`) | infra D1 |
| llama.cpp Metal kernel file map, embedded-source build, runtime compile | infra D2, ch05 C10 |
| llama.cpp elementwise: `scale.cu` (the M0 kernel), `ggml_cuda_kernel_launch`, Metal `dispatchThreadgroups` | ch02 D1, D2 |
| llama.cpp `k_bin_bcast`: a 3D grid over a 4D tensor, `fastdiv`/`fastmodulo` | ch03 D1 |
| llama.cpp warp and block reduce, `soft_max_f32`, the barrier between two reductions sharing a buffer (#26385) | ch10 D1 |
| llama.cpp argmax on CUDA and Metal, host greedy, three tie rules | ch10 D2 |
| llama.cpp GEMV/GEMM switch by batch (`MMVF_MAX_BATCH_SIZE 8`, `MMVQ_MAX_BATCH_SIZE 8`), cuBLAS above it; Metal `mul_mv_ext` for 2 to 8 | ch22 D6, ch15 D1 |
| llama.cpp Metal `kernel_mul_mm` tiles and SIMD-group matrices | ch15 D2 |
| llama.cpp flash attention on CUDA/HIP (online softmax in `fattn-vec.cuh`, split-KV combine, GQA) and Metal (`fa_common.metal`, `fa_aux.metal`) | ch20 D1, D2 |
| KV cache storage in Meta's reference code, llama.cpp, mlx-lm | ch20 D3 |
| Samplers (llama.cpp host and backend chains, transformers, `torch.multinomial`, MLX `categorical`, mlx-lm) | x1 D1 to D8; ch11 D1 to D4; ch12 D1 to D6; ch14 D1 to D7 |
| MLX layout: `mlx/backend/{metal,cuda,cpu,common}`, no HIP; Metal file map (steel GEMM, softmax, quantized, SDPA) | ch01 D3, infra D3 |
| MLX Metal elementwise `binary.h`, `dispatch_threads`, allocator; MLX CUDA allocator uses managed memory | ch02 D3 |
| MLX Metal argmax pair operator and `softmax_single_row` (two-stage, `fast::exp`, `SIMD_SIZE = 32`) | ch10 D3 |
| MLX steel GEMM tile shapes per device | ch15 D3, ch06 D3 |
| MLX Metal SDPA: vector kernels for 8 or fewer queries, 2-pass for long caches, steel attention otherwise | ch20 D4 |
| MLX `mx.compile` fusion; small work is "overhead bound" | ch18 D6 |
| PyTorch: CUDA sources hipified at build time (`tools/amd_build/build_amd.py`, `torch/utils/hipify`) | ch01 D4, ch24 D3 |
| PyTorch MPS layout (`aten/src/ATen/mps`, `native/mps/operations/*.mm`, `native/mps/kernels/*.metal`, `MetalShaderLibrary`) | infra D4, ch24 D2 |
| PyTorch CUDA elementwise launch (`elementwise_kernel`, `num_threads()` 128 on CUDA vs 256 on ROCm), vectorised loads chosen from pointer alignment | ch03 D6, ch06 D4 |
| PyTorch argmax comparator, `BlockReduceSum`, warp-per-row persistent softmax ("Forward supports values up to 2048") | ch10 D4 |
| PyTorch float matmul route: cuBLAS / cuBLASLt / CK on ROCm; gemv disables TF32 | ch15 D4 |
| PyTorch `scaled_dot_product_attention`: semantics, upper-left `is_causal`, MPS kernels | ch20 D5 |
| Attention on AMD: FlashAttention CK and Triton backends; PyTorch ROCm uses AOTriton and CK | ch20 B5 |
| `float*` to `float4*` cast is documented by no vendor; Metal's alternative | infra A6, B6, C7 |
| Metal default math mode, `fast::` vs `precise::`; MSL may round toward zero | ch05 C7, ch10 C4, x1 C5 |
| Summation order changes float results; `expf` accuracy per backend | ch10 A9, A11, B4, C4 |
| bfloat16 layout (1-8-7, same exponent range as float) | appA A2; llama.cpp's BF16 conversion appA D3 |
| `nn.Linear` weight is (out_features, in_features); `y = x A^T + b` | ch03 D6 |

---

## Group A: How to find an op in each codebase (the reading method)

The transferable skill for the lessons: each project has one place where a graph or framework op is turned into a backend call. Start there, follow one name, and you land in the kernel. All three items below were followed by hand today.

### A1. llama.cpp: an op is a `GGML_OP_*` enum value, and each backend has one switch that maps it to a host function

**Verdict: CONFIRMED.**

- CUDA (and HIP, same file): `static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) { switch (dst->op) {` (`ggml/src/ggml-cuda/ggml-cuda.cu` line 2812, last commit 2026-11-20). Cases read today: `case GGML_OP_GET_ROWS: ggml_cuda_op_get_rows(ctx, dst);` (2826), `case GGML_OP_ADD: ggml_cuda_op_add(ctx, dst);` (2847), `case GGML_UNARY_OP_SILU: ggml_cuda_op_silu(ctx, dst);` (2882), `case GGML_GLU_OP_SWIGLU: ggml_cuda_op_swiglu(ctx, dst);` (2945), `case GGML_OP_RMS_NORM: ggml_cuda_op_rms_norm(ctx, dst);` (2997), `case GGML_OP_MUL_MAT: ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);` (3003), `case GGML_OP_SOFT_MAX: ggml_cuda_op_soft_max(ctx, dst);` (3042), `case GGML_OP_ROPE: ggml_cuda_op_rope(ctx, dst);` (3048), `case GGML_OP_FLASH_ATTN_EXT: ggml_cuda_flash_attn_ext(ctx, dst);` (3111). Each host function lives in the `.cu` file of the same name (`binbcast.cu`, `unary.cu`, `norm.cu`, `softmax.cu`, `rope.cu`, `getrows.cu`, `fattn.cu`). Whether a backend accepts an op at all is `ggml_backend_cuda_device_supports_op` (line 6336).
- Before that switch, a fusion pass looks at runs of graph nodes: `if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) { ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]); return 1; }` (lines 5268 to 5271), with `{ GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }` fused one step earlier: `if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) { ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);` (lines 5263 to 5264, re-read 2026-11-21 at the same commit). It can be switched off with the environment variable read at line 181: `ggml_cuda_env_flag("GGML_CUDA_DISABLE_FUSION")`. For a reader this matters: the kernel that runs for `RMS_NORM` is usually the fused one, not `ggml_cuda_op_rms_norm`.
- Metal: `static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx)` (`ggml/src/ggml-metal/ggml-metal-ops.cpp` line 187, last commit 2026-11-20): `case GGML_OP_ADD: case GGML_OP_SUB: case GGML_OP_MUL: case GGML_OP_DIV: { n_fuse = ggml_metal_op_bin(ctx, idx);` (291), `GGML_OP_UNARY` → `ggml_metal_op_unary` (319), `GGML_OP_GLU` → `ggml_metal_op_glu` (327), `GGML_OP_SOFT_MAX` → `ggml_metal_op_soft_max`, `GGML_OP_MUL_MAT` → `ggml_metal_op_mul_mat`, `GGML_OP_GET_ROWS` → `ggml_metal_op_get_rows`, `GGML_OP_RMS_NORM` → `ggml_metal_op_norm` (414), `GGML_OP_ROPE` → `ggml_metal_op_rope`, `GGML_OP_FLASH_ATTN_EXT` → `ggml_metal_op_flash_attn_ext` (487). Each returns `n_fuse`, the number of graph nodes it consumed. The host function asks for a pipeline by a kernel name string (for example `snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));` in `ggml-metal-device.cpp` line 154), which is the `[[host_name(...)]]` of a kernel in `kernels/*.metal` (C2).
- The op enum is the shared vocabulary: the same `GGML_OP_RMS_NORM` runs on the CPU backend as the reference in `test-backend-ops` (D7, F1).

- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/ggml-cuda.cu (lines 176 to 183, 2812 to 3115, 5255 to 5272, 6336)
- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-metal/ggml-metal-ops.cpp (lines 187 to 490)
- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-metal/ggml-metal-device.cpp (line 154)

### A2. MLX: an op is a `Primitive` subclass, and each backend implements `eval_gpu` for it

**Verdict: CONFIRMED.**

- Python layers call core ops or `mx.fast.*`: `nn.RMSNorm.__call__` is `return mx.fast.rms_norm(x, self["weight"], self.eps)` (`python/mlx/nn/layers/normalization.py` line 166); `nn.Embedding.__call__` is `return self.weight[x]` (`embedding.py` line 32, a gather); `nn.silu` is `@partial(mx.compile, shapeless=True) def silu(x): ... return x * mx.sigmoid(x)` (`activations.py` lines 136 to 145).
- The backend side is a method per primitive, one per directory: `void RMSNorm::eval_gpu(` in `mlx/backend/metal/normalization.cpp` (line 17) and in `mlx/backend/cuda/rms_norm.cu` (line 539); `void Softmax::eval_gpu(` in `mlx/backend/cuda/softmax.cu` (line 108) and `metal/softmax.cpp` (infra D3); `void RoPE::eval_gpu(` in `cuda/rope.cu` (249) and `metal/rope.cpp`; `void Matmul::eval_gpu(` in `cuda/matmul.cpp` (143) and `metal/matmul.cpp`; `void ScaledDotProductAttention::eval_gpu(` in `cuda/scaled_dot_product_attention.cpp` (623) and the Metal file of the same name (ch20 D4). Elementwise ops are stamped out by a macro: `#define BINARY_GPU(func) void func::eval_gpu(...)` in `cuda/binary/binary.cuh` (line 399) with `BINARY_GPU(Add)` in `cuda/binary/add.cu`, and the same macro name in `metal/binary.cpp` (`BINARY_GPU(Add)` line 217).
- Fused elementwise code (anything under `mx.compile`, such as `nn.silu` and mlx-lm's `swiglu`) is generated as source at run time: `Compiled::eval_gpu` in `metal/compiled.cpp` (line 257) writes `[[host_name(\"{0}\")]]\n[[kernel]] void {0}(` and compiles it through `d.get_library(...)`; `cuda/compiled.cpp` (line 225) writes `__global__ void {}(` and builds it through `cu::get_jit_module`.
- mlx-lm's Llama model (`mlx_lm/models/llama/layers.py`, 2026-11-19, now a package) uses `nn.RMSNorm`, `swiglu` from `mlx_lm.layers.common`, `scaled_dot_product_attention` from `mlx_lm.layers.attention`, and RoPE with `args.rope_traditional` (B6).

- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/python/mlx/nn/layers/normalization.py
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/cuda/binary/binary.cuh (lines 93 to 118, 399 to 406)
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/metal/compiled.cpp (lines 16 to 40, 257 to 290)
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/cuda/compiled.cpp (lines 63, 225 to 245)
- https://github.com/ml-explore/mlx-lm/blob/2d4df8d94a80c0f9f8794e54fafee08728b04fc6/mlx_lm/models/llama/layers.py

### A3. PyTorch: every CUDA op has a `.cu` file you can open

**Verdict: WRONG** for the most common op. The map from op to kernel is `aten/src/ATen/native/native_functions.yaml`, but some CUDA kernels are generated at build time and exist in no source file.

- Dispatch keys, `native_functions.yaml` (last commit 2026-11-20): `_softmax.out` → `CUDA: softmax_cuda_out`, `MPS: softmax_mps_out` (line 5607 to 5609); `mm.out` → `CUDA: mm_out_cuda`, `MPS: mm_out_mps` (4151, 4153); `silu.out` → `CPU, CUDA, MPS, MTIA, XPU: silu_out` (5300); `_fused_rms_norm` → `CUDA: _fused_rms_norm_cuda`, `MPS: _fused_rms_norm_mps`, `CompositeImplicitAutograd: rms_norm_composite` (3304 to 3309); `_scaled_dot_product_flash_attention` → `CUDA: _scaled_dot_product_flash_attention_cuda` (15280 to 15282); `embedding` → `CompositeExplicitAutograd: embedding_symint` (2261 to 2263), which is `weight.index_select(0, indices)` (`Embedding.cpp`).
- `add.out` has no CUDA entry: it is `structured: True` with `ufunc_inner_loop: Generic: add (AllAndComplex, BFloat16, Half, ComplexHalf)` (lines 565 to 571). The scalar body is `return self + alpha * other;` in `aten/src/ATen/native/ufunc/add.h`; torchgen writes `UfuncCUDA_{name}.cu` at build time (`torchgen/gen.py` line 2536) containing `gpu_kernel(iter, ufunctor);` and `REGISTER_DISPATCH(...)` (`torchgen/dest/ufunc.py` lines 279, 288, 328). The MPS side says so in a comment: "`add.out` is a ufunc op, and ufunc codegen only emits CPU/CUDA impls, so MPS needs this entry" (`mps/operations/BinaryOps.mm` lines 16 to 19), and calls `add_stub`, which reaches `lib.exec_binary_kernel(iter, "add")` or `"add_alpha"` (`BinaryKernel.mm` lines 182, 187) and `struct add_alpha_functor { ... return static_cast<T>(a + c10::metal::mul(alpha, b));` (`kernels/BinaryKernel.metal` lines 22 to 25).
- Many other CUDA ops are a lambda passed to the same `gpu_kernel` loop (ch03 D6): SiLU is `gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) -> scalar_t { using opmath_t = at::opmath_type<scalar_t>; const opmath_t x_acc = static_cast<opmath_t>(x); return x_acc / (opmath_t(1) + ::exp(-x_acc)); });` registered with `REGISTER_DISPATCH(silu_stub, &silu_kernel)` (`native/cuda/ActivationSiluKernel.cu` lines 30 to 33, 57).
- So the reading rule for PyTorch: find the op's `- func:` line, read its `dispatch:` keys; if there are none, look for `structured_delegate`, `ufunc_inner_loop` or a `_stub` and follow `REGISTER_DISPATCH`.

- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/native_functions.yaml
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/ufunc/add.h
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/torchgen/dest/ufunc.py
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/mps/operations/BinaryOps.mm
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/mps/kernels/BinaryKernel.metal
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/cuda/ActivationSiluKernel.cu

### A4. Short excerpts from the three projects may be shown in lessons with attribution

**Verdict: PARTLY.** The licences are permissive and confirmed; whether a short excerpt needs the full notice is a legal question this pass cannot settle (COULD NOT VERIFY).

- llama.cpp and MLX are MIT. Both `LICENSE` files contain (line 12): "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software." Notices: "Copyright (c) 2023-2026 The ggml authors" (llama.cpp, last commit to the file 2026-02-02), "Copyright © 2023 Apple Inc." (MLX). mlx-lm and llama2.c are MIT too (GitHub API).
- PyTorch's `LICENSE` is the three-clause BSD text with extra copyright lines: "1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer." and "3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America and IDIAP Research Institute nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission." GitHub's API reports `NOASSERTION` because of the extra text.
- transformers and safetensors are Apache-2.0 (GitHub API).
- Not verified: whether a 5 to 15 line excerpt is a "substantial portion". Safe practice for the lessons (ours): quote short, name the file, link the permalink at the pinned commit, and show the project's licence name and copyright line next to each excerpt.

- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/LICENSE
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/LICENSE
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/LICENSE

---

## Group B: The guided tour, op by op

One item per op. Each gives the file and function in each project, what the reader will recognise from the course, and what is different. "Course" means our labs and chapters (for the lesson to map), not a vendor fact.

### B1. Elementwise (M0, chapter 2; coarsening and vector loads, chapter 6): the three projects write the chapter 2 kernel

**Verdict: CONFIRMED.**

- llama.cpp CUDA, `unary.cu` (last commit 2026-10-25), the whole kernel behind SiLU, GELU and the other unary ops:

```cpp
template <float (*op)(float), typename T>
static __global__ void unary_op_kernel(const T * x, T * dst, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(x[i])));
}
```

  (lines 116 to 127). The op is a template argument: `ggml_cuda_op_silu` is `ggml_cuda_op_unary<op_silu>(ctx, dst)` (line 229 to 230), with `ggml_cuda_op_silu_single(float x) { return x / (1.0f + expf(-x)); }` (`unary.cuh` line 98 to 100). SwiGLU is `unary_gated_op_kernel`, one thread per output, `dst[i] = ...op(x[j0]) * g[j1]` (lines 307 to 321). The two `pdl` calls are no-ops unless programmatic dependent launch is on (ch18 D3). Launch: `(k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE` blocks (line 131). Binary ops (`k_bin_bcast`, `binbcast.cu`) are ch03 D1; `ggml_cuda_op_fused_add` fuses 2 to 8 consecutive adds (`GGML_ASSERT(2 <= n_fuse && n_fuse <= 8)`, line 499).
- llama.cpp Metal, `kernels/unary.metal` (2026-10-29): one template `kernel_unary_impl` serves every unary op; the op is a function constant, `constant short FC_unary_op [[function_constant(FC_UNARY + 0)]];`, tested inside the body: `if (FC_OP == OP_UNARY_NUM_SILU) { dst_ptr[i0] = ggml_metal_unary_cast<T>(x / (1 + exp(-x))); }` (lines 3, 132 to 134). A `float4` instantiation exists (`kernel_unary_f32_f32_4`, line 210). SwiGLU: `kernel_swiglu`, a grid-stride loop `for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x)` (lines 290 to 309). Binary ops: `kernel_bin_fuse_impl` in `binbcast.metal`, which also fuses several ops.
- MLX CUDA, `backend/cuda/binary/binary.cuh` (2026-10-09), the coarsened form (chapter 6):

```cpp
template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ __launch_bounds__(BINARY_MAX_BLOCK_DIM) void binary_vv(
    const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();
  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[i], b[i]);
    }
  } else {
    auto a_vec = load_vector<N_READS>(a, index);
    auto b_vec = load_vector<N_READS>(b, index);
    ...
    store_vector<N_READS>(out, index, out_vec);
  }
}
```

  (lines 93 to 118). `load_vector` is the book's pointer cast, guarded at run time: `if (is_aligned<N>(ptr)) { auto* from = reinterpret_cast<const AlignedVector<T, N>*>(ptr); return from[offset]; } else { ...scalar loop... }` with `struct alignas(sizeof(T) * N) AlignedVector` (`backend/cuda/device/utils.cuh` lines 33, 58 to 70). Same design as PyTorch's (ch06 D4); neither is a documented vendor guarantee (infra A6).
- MLX Metal: `binary.h` `binary_ss`/`binary_vv` with `WorkPerThread` (ch02 D3).
- PyTorch CUDA: `gpu_kernel` lambdas over `elementwise_kernel` (ch03 D6, A3 above). PyTorch MPS: `exec_binary_kernel(iter, "add")` and `add_functor` / `add_alpha_functor` in `kernels/BinaryKernel.metal` (A3).
- SiLU formula: `x / (1 + exp(-x))` in llama.cpp CUDA and Metal, PyTorch CUDA, llama2.c (D1); `x * sigmoid(x)` in MLX. Mathematically equal; in float they need not round alike (ours, not measured).

Course mapping (ours): `unary_op_kernel` is lab 2.3's kernel (`launchScaleShift`, `launchAddResidual`) with the op as a template parameter; `binary_vv` is chapter 6's coarsening plus the vector load, with the tail handled by a scalar loop instead of a second launch.

- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/unary.cu
- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/unary.cuh
- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/binbcast.cu
- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-metal/kernels/unary.metal
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/cuda/binary/binary.cuh
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/cuda/device/utils.cuh
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/python/mlx/nn/layers/activations.py

### B2. RMSNorm (a row reduction, chapter 10's technique; not a course lab yet): one block or threadgroup per row, a two-stage sum, then a scale

**Verdict: CONFIRMED.** Every project computes `x * rsqrt(mean(x^2) + eps) * w` per row, with one block (CUDA) or threadgroup (Metal) per row.

- llama.cpp CUDA, `norm.cu` (2026-11-18), `rms_norm_f32` (lines 127 to 272), scalar path:

```cpp
float tmp = 0.0f; // partial sum for thread in warp
for (int col = tid; col < ncols; col += block_size) {
    const float xi = ggml_cuda_cast<float>(xc[col]);
    tmp += xi * xi;
}
// sum up partial sums
tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
const float mean = tmp / ncols;
const float scale = rsqrtf(mean + eps);
```

  then `dstc[col] = ggml_cuda_cast<T>(scale * xi * mulc[mul_col]);` when the weight multiply is fused (`do_multiply`). `const int row = blockIdx.x;` and `extern __shared__ float s_sum[];`. A vector path loads `float4` through `rms_norm_load4` and casts the weight pointer, `const float4 * mul4 = (const float4 *) mulc;`, chosen only when `ncols % 4 == 0` and the destination is 16-byte aligned (`((uintptr_t) dst % (4*sizeof(T))) == 0`, line 473). Block size: on the vector path 32, 128, 256, 512 or 1024 threads by row length, with 4 `float4` per thread above 1024 `float4`; on the scalar path 256 threads for rows under 1024 elements, else 1024 (lines 486 to 505). `block_reduce` is ch10 D1.
- llama.cpp Metal, `kernels/norm.metal` (2026-11-07), `kernel_rms_norm_fuse_impl` (lines 111 to 176): SIMD-group 0 zeroes `shmem_f32[tiisg]`, a strided loop does `sumf += dot(x[i00], x[i00]);`, then `sumf = simd_sum(sumf);`, barrier, lane 0 of each SIMD-group writes `shmem_f32[sgitg]`, barrier, every lane reads `shmem_f32[tiisg]` and `simd_sum`s again, `const float scale = 1.0f/sqrt(mean + args.eps);`. Fusion is a template argument `F` (1 plain, 2 times weight, 3 times weight plus add), instantiated as `kernel_rms_norm_f32`, `kernel_rms_norm_mul_f32`, `kernel_rms_norm_mul_add_f32` and `_4` forms (lines 178 to 185).
- MLX Metal, `kernels/rms_norm.metal` (2026-08-19): `rms_single_row` and `rms_looped` (lines 13, 84), `simd_sum`, threadgroup `local_sums`, `local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);` (line 61); `RMSNorm::eval_gpu` switches to the looped kernel above `RMS_LOOPED_LIMIT` (`normalization.cpp` lines 57 to 60).
- MLX CUDA, `backend/cuda/rms_norm.cu` (2026-08-13): `rms_norm_small` and `rms_norm` kernels with a CUB-like `BlockBroadcastReduce` ("Similar to cub::BlockReduce, but result is broadcasted to every thread.", line 25), `normalizer = rsqrt(normalizer / axis_size + eps);` (lines 107, 162).
- PyTorch CUDA: `torch.nn.functional.rms_norm` → `rms_norm_symint` → `at::_fused_rms_norm(...)` (`native/layer_norm.cpp` line 372; it falls back to `rms_norm_composite` for complex input, and warns "Cannot dispatch to fused implementation." when weight and input dtypes differ, lines 345 to 352) → `_fused_rms_norm_cuda` (`native/cuda/layer_norm_kernel.cu` line 2009), which shares LayerNorm's kernels through a template flag: `template <typename T, typename T_ACC, bool rms_norm> __global__ void RowwiseMomentsCUDAKernel(` and `vectorized_layer_norm_kernel_impl` (lines 57, 270).
- PyTorch MPS: `_fused_rms_norm_mps` (`operations/RMSNorm.mm` line 22) picks a pipeline by name from `kernels/RMSNorm.metal`, whose reduction is MLX's structure (`simd_sum`, `local_sums`, `metal::precise::rsqrt(acc / axis_size + eps)`, lines 54 to 74).
- References: Hugging Face `LlamaRMSNorm.forward`: `hidden_states = hidden_states.to(torch.float32)`, `variance = hidden_states.pow(2).mean(-1, keepdim=True)`, `hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)`, `return self.weight * hidden_states.to(input_dtype)` (`modeling_llama.py` lines 62 to 67): the weight multiply happens after casting back to the input type. llama2.c `rmsnorm`: `ss /= size; ss += 1e-5f; ss = 1.0f / sqrtf(ss);` then `o[j] = weight[j] * (ss * x[j]);` (`run.c` lines 182 to 195).

Course mapping (ours): chapter 10's two-stage block reduction (M4's `launchRowMax` structure) with sum of squares instead of max; the identity-padded second stage is visible in both llama.cpp files. M10 must build it (project notes).

- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/norm.cu (lines 127 to 272, 420 to 520)
- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-metal/kernels/norm.metal (lines 111 to 185)
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/metal/kernels/rms_norm.metal
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/cuda/rms_norm.cu
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/layer_norm.cpp (lines 265, 329 to 372)
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/cuda/layer_norm_kernel.cu
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/mps/kernels/RMSNorm.metal
- https://github.com/huggingface/transformers/blob/4d9ef623e3b0a1fb6c12891a907e0e50d41f0af8/src/transformers/models/llama/modeling_llama.py (lines 53 to 70)

### B3. Softmax (M4, chapter 10): one project computes it the same way on both of its GPU backends

**Verdict: WRONG** for MLX. Its Metal kernel takes the max, then the sum; its CUDA kernel is the one-pass online normaliser (chapter 20's recurrence applied to a row).

- MLX Metal `softmax_single_row`: max reduction, broadcast, then sum reduction, `fast::exp` (ch10 D3).
- MLX CUDA, `backend/cuda/softmax.cu` (2026-02-26), one block per row (`in += grid.block_rank() * axis_size;`):

```cpp
prevmax = maxval;
for (int i = 0; i < N_READS; ++i) {
  maxval = max_op(maxval, static_cast<AccT>(vals[i]));
}
// Online normalizer calculation for softmax:
// https://github.com/NVIDIA/online-softmax
normalizer = normalizer * softmax_exp(prevmax - maxval);
for (int i = 0; i < N_READS; i++) {
  normalizer = normalizer + softmax_exp(static_cast<AccT>(vals[i]) - maxval);
}
```

  (lines 49 to 62), then a warp `cg::reduce` of max and a rescale of the normaliser, a second stage through `__shared__ AccT local_max[WARP_SIZE]; __shared__ AccT local_normalizer[WARP_SIZE];` with identity padding (`warp.thread_rank() < warp.meta_group_size() ? local_max[warp.thread_rank()] : Limits<AccT>::min()`), and a final pass that writes `softmax_exp(x - maxval) * normalizer`. The exponential is `__expf`, with the comment "Softmax doesn't need high precision exponential cause x is gonna be in (-oo, 0] anyway and subsequently it will be divided by sum(exp(x_i))." (lines 22 to 28). It uses cooperative groups (`cg::this_grid()`, `cg::tiled_partition<WARP_SIZE>(block)`, `cg::reduce`), which is CUDA-only in this course's portability rules (CLAUDE.md, infra B4).
- llama.cpp CUDA `soft_max_f32`: max, `expf`, barrier, sum (ch10 D1). llama.cpp Metal `kernel_soft_max` (`kernels/softmax.metal`, 2026-08-24): `simd_max`, `buf[tiisg] = -INFINITY` padding, two barriers, `simd_max` again; `exp(...)`, `simd_sum`, `buf[tiisg] = 0.0f`, two barriers, `simd_sum` again (lines 42 to 99). Same two-pass structure on both llama.cpp backends.
- PyTorch CUDA `host_softmax` (`native/cuda/SoftMax.cu`, 2026-10-27): rows of at most 2048 elements and 8192 bytes go to the warp-per-row persistent kernel, `if (dim_size <= 2048 && dim_size*sizeof(scalar_t) <= 8192) { ... dispatch_softmax_forward<...>` (line 1125 to 1129, consistent with ch10 D4); longer rows to `cunn_SoftMaxForwardSmem` or `cunn_SoftMaxForward` (lines 762, 916, 1203 to 1206). PyTorch MPS `kernels/SoftMax.metal` (2026-10-30) uses `precise::exp` throughout (lines 30, 39) and `c10::metal::simd_max` / `simd_sum`; MLX's Metal softmax uses `fast::exp` (ch10 D3). The two Metal frameworks made opposite accuracy choices for the same op.

Course mapping (ours): MLX CUDA's kernel is M4's softmax with chapter 20's (m, D) update moved into the row reduction, so it reads the row twice instead of three times; a good "what does it do differently" exercise.

- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/cuda/softmax.cu (lines 20 to 106)
- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-metal/kernels/softmax.metal
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/cuda/SoftMax.cu
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/mps/kernels/SoftMax.metal

### B4. GEMV and GEMM (M1 to M3, M8; chapters 3, 5, 15, 22): a matrix product goes to one kernel

**Verdict: WRONG.** In all three projects the shape picks the kernel: a vector kernel for one (or a few) tokens, a tiled GEMM or the vendor library otherwise.

- llama.cpp CUDA and HIP, `ggml_cuda_mul_mat` (`ggml-cuda.cu` lines 2535 to 2596), the decision chain in order: `ggml_cuda_mul_mat_cublas` if the activations or the output are not `F32`, or the weights sit in a padded view of a compute buffer: `if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) { ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);` (2547 to 2548; re-read 2026-11-21 at the same commit: the type test reads `src1`, the activations, and `dst`, never `src0`, the weights; `src0`'s type is what the later `should_use_*` checks read); `ggml_cuda_should_use_mmvf` → `ggml_cuda_mul_mat_vec_f` (float GEMV, 2555 to 2558); `ggml_cuda_should_use_mmf` → `ggml_cuda_mul_mat_f` (2575 to 2576); `ggml_cuda_should_use_mmvq` → `ggml_cuda_mul_mat_vec_q` (quantised GEMV, 2581 to 2582); `ggml_cuda_should_use_mmq` → `ggml_cuda_mul_mat_q` (2589 to 2590); an FP8 path; else `ggml_cuda_mul_mat_cublas` (2596). Thresholds: ch22 D6.
- llama.cpp Metal: `mul_mv_ext` for 2 to 8 tokens, `mul_mm` (SIMD-group matrices) above (ch22 D6, ch15 D2), `mul_mv.metal` for one.
- MLX CUDA, `backend/cuda/matmul.cpp` (2026-11-02): `if (!bias && cu::can_use_gemv(M, N, K, a_transposed, b_transposed)) { cu::gemv(` (line 99), else "Invoke cublasLt" through `CublasGemm` (line 115; `gemms/cublas_gemm.cpp` calls `cublasLtMatmulDescSetAttribute`). `can_use_gemv` is `return (M == 1 && b_transposed) || (N == 1 && !a_transposed);` (`gemms/gemv.cu` line 188). The GEMV is a warp per row:

```cpp
int row = g_idx.x * rows_per_block + t_idx.y;
if (row < rows) {
  using Acc = typename GemvAccType<T>::type;
  Acc sum = Acc(0);
  for (int col = n_per_thread * warp.thread_rank(); col < cols;
       col += (WARP_SIZE * n_per_thread)) {
    auto local_mat = unsafe_load_vector<n_per_thread>(mat + row * cols + col, 0);
    auto local_vec = unsafe_load_vector<n_per_thread>(vec + col, 0);
    for (int j = 0; j < n_per_thread; ++j) {
      sum += static_cast<Acc>(local_mat[j]) * static_cast<Acc>(local_vec[j]);
    }
  }
  sum = cg::reduce(warp, sum, cg::plus<Acc>{});
  if (warp.thread_rank() == 0) { out[row] = static_cast<T>(sum); }
}
```

  (`gemv_impl`, lines 56 to 83; half and bfloat16 accumulate in `float` via `GemvAccType`).
- MLX Metal, `metal/matmul.cpp` (2026-10-09): a "wide" GEMV first ("The wide gemv route streams the weight matrix once per <= 5 input vectors instead of running a row-padded GEMM tile.", line 1620), then `if (std::min(M, N) == 1) { return gemv(` (line 1647), else steel GEMM (ch15 D3). `kernels/gemv.h` `GEMVKernel` asserts `static_assert(SM * SN == 32, "simdgroup can only have 32 threads");` (line 50), a hard-coded SIMD width (ch04 D3).
- PyTorch CUDA: cuBLAS/cuBLASLt `gemm`, `gemv` without TF32 (ch15 D4). PyTorch MPS: `Linear.mm` uses `MPSNDArrayMatrixMultiplication` (lines 68 to 82) and notes "addmm fuses the bias and routes rank-1 shapes to the GEMV kernels" (line 161); `Matmul.mm` builds names `gemv_t_...`, `gemv_t2d_...`, `gemv_nt_...` (lines 116 to 141) for kernels in `kernels/Gemv.h` (`kernel void gemv_t(`, line 71, a `simd_sum` of per-SIMD-group partials).

Course mapping (ours): MLX CUDA's `gemv_impl` is lab 22.1's GEMV (`launchGemvGather`) with vector loads; llama.cpp's chain is the M8 lesson ("8 is a code constant, not a device figure", ch22 D6).

- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/ggml-cuda.cu (lines 2535 to 2596)
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/cuda/matmul.cpp (lines 95 to 135)
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/cuda/gemms/gemv.cu (lines 24 to 90, 187 to 189)
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/metal/matmul.cpp (lines 1584 to 1660)
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/metal/kernels/gemv.h (lines 30 to 60)
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/mps/operations/Matmul.mm (lines 94 to 160)
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/mps/kernels/Gemv.h

### B5. Attention (M6, M7; chapter 20): each framework runs its own attention kernels on every GPU backend

**Verdict: PARTLY.** llama.cpp does (ch20 D1, D2); MLX on Metal does (ch20 D4); MLX on CUDA uses cuDNN for prefill and for long decodes and its own kernels only for short decodes; PyTorch on CUDA ships FlashAttention-2 from a submodule and a selection order.

- MLX CUDA, `backend/cuda/scaled_dot_product_attention.cpp` (2026-11-02): `ScaledDotProductAttention::eval_gpu` (line 623) calls `sdpa_cudnn(` when `supports_sdpa_cudnn(q, k, v, has_arr_mask, do_causal_, s)` (lines 650 to 654), built with the cuDNN frontend graph API (`fe::DataType_t`, `.set_name("sdpa_cudnn")`, line 228). For one query (`q.shape(2) == 1`), cuDNN is used only without an array mask and from 256 keys: "The cuDNN SDPA is faster than vector kernel but for small sequence the overhead would kill the advantage." `constexpr int kv_cache_step = 256; // number is from mlx-lm` (`use_cudnn_for_decoding`, lines 72 to 88). Otherwise `sdpa_vector` (line 544).
- MLX CUDA decode kernels, `scaled_dot_product_attention.cu` (2026-10-16): `kernel_sdpav_1pass` and `kernel_sdpav_2pass_1` (lines 42, 195), the online softmax in base 2: `const U scale_log2 = params.scale * M_LOG2E;` (line 67), `U factor = exp2f(max_score - new_max); U exp_score = exp2f(score - new_max);` (lines 143 to 144), scores summed with `cg::reduce(warp, score, cg::plus<U>())`. Same recurrence as MLX Metal's `sdpa_vector.h` (`fast::exp`, ch20 D4) with the scale folded into `log2 e`.
- PyTorch CUDA: `select_sdp_backend` (`native/transformers/cuda/sdp_utils.cpp` line 1337) walks `priority_order`; the default is `sdp_priority_order = { at::SDPBackend::flash_attention, at::SDPBackend::efficient_attention, at::SDPBackend::math, at::SDPBackend::cudnn_attention, at::SDPBackend::overrideable};` (`aten/src/ATen/Context.h` lines 509 to 514), replaced by cuDNN first when `check_prefer_cudnn_attention()` (sdp_utils lines 121 to 131). "Note [SDPA Runtime Dispatch] ... Each backend defines a use_<backend> function that returns true if the backend can be run with the given SDP parameters." (lines 48 to 58). The flash kernels are not in PyTorch's tree: `native/transformers/cuda/flash_attn/` holds `flash_api.cpp`, `flash_api.h`, `philox.cuh`, `static_switch.h`; `flash_api.cpp` includes `<flash.h>` and calls `run_mha_fwd_<elem_type, kHeadDim, Is_causal>` or `run_mha_fwd_splitkv_dispatch` (lines 294 to 301), from the submodule `third_party/flash-attention` (`url = https://github.com/Dao-AILab/flash-attention.git`, `.gitmodules` lines 115 to 117; pinned at `bb25ba3ce4b1c8df5d4ef6a94807f0d3910599ca` today). Memory-efficient attention is in `mem_eff_attention/`. On ROCm: AOTriton and CK (ch20 B5).
- PyTorch MPS: `sdpa_vector_2pass_mps`, `sdpa_vector_fast_mps`, `sdpa_prefill_mps` (ch20 D5).
- Not verified: which cuDNN version MLX requires, and what `supports_sdpa_cudnn` checks beyond the lines read.

Course mapping (ours): MLX CUDA's `kernel_sdpav_2pass_1` is lab 20.4/20.5's decode kernel with split-KV (ch20 D1's combine); the `exp2f` with `scale * M_LOG2E` is a rewrite the course does not do.

- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/cuda/scaled_dot_product_attention.cpp (lines 72 to 90, 317 to 380, 538 to 560, 623 to 660)
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/cuda/scaled_dot_product_attention.cu (lines 42 to 190)
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp (lines 48 to 131, 1337 to 1350)
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/Context.h (lines 509 to 515)
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp (lines 33 to 50, 294 to 301)
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/.gitmodules (lines 115 to 117)

### B6. RoPE (not taught in the course; an elementwise kernel over pairs): every implementation rotates the same pairs

**Verdict: WRONG.** There are two pairings. "Normal" (also "traditional", "interleaved") rotates consecutive elements (0, 1), (2, 3), ...; "NeoX" (Hugging Face's `rotate_half`) rotates element i with element i + d/2. Weights trained with one must be permuted to run with the other. This is the main trap for M10's "real checkpoint" path.

- llama.cpp CUDA, `rope.cu` (2026-11-19): `rope_norm` (line 44) reads `x[ix + 0]` and `x[ix + 1]` and stores `x0 * cos_theta - x1 * sin_theta, x0 * sin_theta + x1 * cos_theta` (lines 119 to 122); `rope_neox` (line 126) reads `x[ix + n_offs/2 + 0]` and `x[ix + n_offs/2 + n_dims/2]`. The angle: `const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);` (line 110), then `cos_theta = cosf(theta) * mscale; sin_theta = sinf(theta) * mscale;` inside `rope_yarn`. There is also a fused `rms_norm_mul_rope_f32` (line 766). Metal: `kernel_rope_norm`, `kernel_rope_neox` in `kernels/rope.metal` (lines 48, 106).
- Which pairing a model uses, `llama_model_rope_type` (`src/llama-model.cpp`, 2026-11-20): "// use what we call a normal RoPE, operating on pairs of consecutive head values" then `case LLM_ARCH_LLAMA:` ... `return LLAMA_ROPE_TYPE_NORM;` (lines 3741 to 3791); "// the pairs of head values are offset by n_rot/2" → `LLAMA_ROPE_TYPE_NEOX` (lines 3799 to 3885). `include/llama.h`: `LLAMA_ROPE_TYPE_NORM = 0, LLAMA_ROPE_TYPE_NEOX = GGML_ROPE_TYPE_NEOX,` (lines 86 to 87).
- Hugging Face Llama (`modeling_llama.py`): `def rotate_half(x): ... x1 = x[..., : x.shape[-1] // 2]; x2 = x[..., x.shape[-1] // 2 :]; return torch.cat((-x2, x1), dim=-1)` and `emb = torch.cat((freqs, freqs), dim=-1)`, `inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))` (lines 73 to 123). That is the NeoX pairing.
- So llama.cpp converts HF Llama weights by permuting the rows of Q and K: `def permute(weights, n_head, n_head_kv): if n_head_kv is not None and n_head != n_head_kv: n_head = n_head_kv; return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:]).swapaxes(1, 2).reshape(weights.shape))`, applied to `q_proj.weight` with `(n_head, n_head)` and `k_proj.weight` with `(n_head, n_kv_head)` (`conversion/llama.py` lines 177 to 182, 262 to 265; the converter moved from one file into a `conversion/` package).
- MLX: `mx.fast.rope(a, dims, *, traditional, base, scale, offset, freqs=None)`, "traditional (bool): If set to ``True`` choose the traditional implementation which rotates consecutive dimensions." (`python/src/fast.cpp` lines 211 to 245); on Metal `traditional` is a function constant, `constant bool traditional [[function_constant(2)]];` (`kernels/rope.metal` line 8); on CUDA a template parameter (`template <typename T, bool traditional, bool forward>`, `cuda/rope.cu` line 15). mlx-lm's Llama config has `rope_traditional: bool = False` (`mlx_lm/models/llama/config.py` line 28), the HF pairing.
- llama2.c rotates consecutive pairs: `float v0 = vec[i]; float v1 = vec[i+1]; vec[i] = v0 * fcr - v1 * fci; vec[i+1] = v0 * fci + v1 * fcr;` with `float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);` (`run.c` lines 265 to 278).
- PyTorch has no RoPE operator: "rotary" does not occur in `native_functions.yaml` at the pinned commit (searched); Hugging Face runs `rotate_half` as ordinary tensor ops.
- SmolLM2-135M's `config.json` carries `"rope_interleaved": false` (D4).

- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/rope.cu (lines 15 to 200)
- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/src/llama-model.cpp (lines 3708 to 3915)
- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/conversion/llama.py (lines 177 to 195, 255 to 270)
- https://github.com/huggingface/transformers/blob/4d9ef623e3b0a1fb6c12891a907e0e50d41f0af8/src/transformers/models/llama/modeling_llama.py (lines 73 to 150)
- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/python/src/fast.cpp (lines 205 to 260)
- https://github.com/ml-explore/mlx-lm/blob/2d4df8d94a80c0f9f8794e54fafee08728b04fc6/mlx_lm/models/llama/config.py
- https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c (lines 264 to 279)

### B7. Embedding lookup (lab 22.2's `launchEmbed`; chapter 22's output-centric gather): a row gather in every project

**Verdict: CONFIRMED.**

- llama.cpp: `GGML_OP_GET_ROWS` → `ggml_cuda_op_get_rows` with kernels `k_get_rows`, `k_get_rows_float`, `k_get_rows_float_vec` (`getrows.cu` lines 6, 73, 105, 2026-11-11), quantised rows dequantised on the fly; Metal `kernel_get_rows_f` and `kernel_get_rows_q` (`kernels/quantize.metal` lines 266, 297), named `kernel_get_rows_<type>` by the host (A1).
- PyTorch: `embedding_symint` → `return weight.index_select(0, indices.reshape(-1)).view_symint(size);` (`native/Embedding.cpp`, 2026-08-27). Backward is ch21 D1, D2.
- MLX: `nn.Embedding.__call__` is `self.weight[x]` (A2), a gather primitive.

- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/getrows.cu
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/aten/src/ATen/native/Embedding.cpp (lines 35 to 55)

### B8. Sampling (M5, module X1): where each project's sampler runs

**Verdict: CONFIRMED** (all settled in x1 D1 to D8; one addition).

- MLX's CUDA backend uses the same counter-based generator as Metal: `__device__ rbits threefry2x32_hash(uint2 key, uint2 count)` with kernels `rbitsc` and `rbits` (`backend/cuda/random.cu` lines 27, 47, 87, 2026-04-21), the CUDA twin of `kernels/random.metal` (x1 D7).
- Tour pointers for the lesson: llama.cpp host chain `src/llama-sampler.cpp` and the experimental backend graph (`ggml_soft_max` → `ggml_cumsum` → `ggml_step` → `ggml_sum`) (x1 D3); its GPU top-k is `top-k.cu` and `ggml-metal` argsort/top-k (ch14 D2); transformers draws with `torch.multinomial(probs, 1)` = argmax(p / q) (x1 D4, D5); PyTorch's CUDA inverse-CDF path is `native/cuda/MultinomialKernel.cu` (x1 D5); mlx-lm `sample_utils.py` (x1 D8); FlashInfer `SamplingFromProbKernel` (x1 A6).

- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend/cuda/random.cu

### B9. HIP: the tour's CUDA files are also the AMD files

**Verdict: PARTLY.** True for llama.cpp and PyTorch (both compile their CUDA sources for AMD: ch05 B9, ch24 D3), not for MLX, which has no AMD backend.

- llama.cpp: `ggml-hip/CMakeLists.txt` (2026-11-15) still globs `../ggml-cuda/*.cu` and the `template-instances/fattn-tile*`, `fattn-mma*`, `mmq*`, `mmf*` files and sets `LANGUAGE HIP` (lines 60 to 70, 133). Every file in B1 to B7 above is therefore the HIP file. Visible vendor branches in the files read: `softmax.cu` includes `<hip/hip_cooperative_groups.h>` under `#ifdef GGML_USE_HIP` (lines 5 to 10, ch10 D1); PDL is CUDA-only (ch18 D3); the warp width is a logical 32 (ch04 D1).
- PyTorch: the same `native/cuda/*` and `native/transformers/cuda/attention.cu` are hipified at build time (ch24 D3); attention on ROCm goes to AOTriton or CK (ch20 B5); `sdp_utils.cpp` carries `#ifndef AOTRITON_VERSION_MINOR` (line 43).
- MLX: `mlx/backend/` lists `common cpu cuda gpu metal no_cpu no_gpu` at the pinned commit (no HIP or ROCm). An open pull request "[Experiment] ROCm backend" (`ml-explore/mlx#2300`, open, last updated 2026-08-14) and an open issue "Add ROCm Support for AMD GPUs" (#2556) exist; neither is merged. MLX's CUDA backend also depends on cuBLASLt, cuDNN's frontend and cooperative groups (B4, B5, B3), none of which is a HIP-compatible spelling (infra B4, B12).

- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-hip/CMakeLists.txt
- https://github.com/ml-explore/mlx/tree/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend
- https://github.com/ml-explore/mlx/pull/2300
- https://github.com/ml-explore/mlx/issues/2556

---

## Group C: Metal mechanisms the reader meets when reading these kernels

### C1. Metal kernels in llama.cpp and MLX pick variants with function constants; CUDA code uses templates

**Verdict: CONFIRMED.**

- MSL spec 4.1, section 5.8 "Program Scope Function Constants" (page 190): "All OS: Metal 1.2 and later support function constants." "Function constants enable the generation of multiple variants of a function. Without using function constants, you can compile one function many times with different preprocessor macro defines to enable different features (an ubershader). ... Function constants provide the same ease of use as preprocessor macros but moves the generation of the specific variants to the creation of the pipeline state, so you don't have to compile the variants offline." Syntax `[[function_constant(index)]]`, "The value index needs to be between 0 and 65535." "You don't initialize function constants in the Metal function source. Instead, you specify their values when creating a specialized function (MTLFunction) using an MTLFunctionDescriptor in the Metal API."
- In the tour: llama.cpp's unary op selector `FC_unary_op` (B1); MLX's RoPE `traditional` (B6). The CUDA files do the same job with template parameters (`template <float (*op)(float), typename T>` in llama.cpp, `template <typename T, bool traditional, bool forward>` in MLX CUDA), and HIP compiles those templates unchanged.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 5.8, pages 190 to 191)

### C2. A Metal kernel is found by the string in `[[host_name(...)]]`, not by its C++ name

**Verdict: CONFIRMED.**

- MSL spec 4.1: "In Metal 2.2 and later, you can override the default name that the Metal Framework API uses to refer to a qualified function. Add the [[host_name(name)]] attribute to the function declaration, where name is the string literal that the Metal Framework API will use to reference the function name." Templates need it: "[[host_name(name)]] attribute to provide a different name for each specialization."
- llama.cpp: `template [[host_name("kernel_rms_norm_mul_f32")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 2>;` (`norm.metal` line 179); the host builds the same string (A1). MLX's JIT writes `[[host_name("...")]]` into generated source (A2). PyTorch MPS builds names with `fmt::format("{}_{}", name, scalarToMetalTypeString(output))` and calls `lib.getPipelineStateForFunc(kernel)` (`RMSNorm.mm` lines 48 to 49).
- Reading rule (ours): to find the Metal kernel behind a host call, search the `.metal` files for the string the host formats, not for a function name.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (the `host_name` paragraphs, page 132)

---

## Group D: M10, the formats and references for "a small real checkpoint"

M10's lab is built on a synthetic model and downloads nothing. These items are for the README's "point the same pipeline at a real checkpoint you download yourself" section and for the reference implementation's conventions.

### D1. llama2.c's `.bin` format is the simplest real format: a 28-byte header and raw float32 weights in a fixed order

**Verdict: CONFIRMED.** `run.c` (last commit 2023-11-28; repository last pushed 2024-08-06, so it is a stable reference, not a moving one).

- Header: `typedef struct { int dim; int hidden_dim; int n_layers; int n_heads; int n_kv_heads; int vocab_size; int seq_len; } Config;` read with `fread(config, sizeof(Config), 1, file)`. "negative vocab size is hacky way of signaling unshared weights. bit yikes." `int shared_weights = config->vocab_size > 0 ? 1 : 0;` (lines 18 to 26, 142 to 150).
- Order after the header (`memory_map_weights`, lines 111 to 140): `token_embedding_table` (vocab × dim), `rms_att_weight` (layers × dim), `wq`, `wk`, `wv`, `wo`, `rms_ffn_weight`, `w1`, `w2`, `w3`, `rms_final_weight` (dim), then two skipped blocks "what used to be freq_cis_real (for RoPE)" and "freq_cis_imag", each `seq_len * head_size / 2`, then `w->wcls = shared_weights ? w->token_embedding_table : ptr;`.
- Matmul convention: `void matmul(float* xout, float* x, float* w, int n, int d) { // W (d,n) @ x (n,) -> xout (d,)` with `val += w[i * n + j] * x[j];` (lines 217 to 228): row-major (out, in), the `nn.Linear` convention (ch03 D6).
- Forward pass (`forward`, lines 231 to 361), the whole decoder step M10 composes: copy the embedding row; per layer `rmsnorm`, `q,k,v` matmuls into the KV cache at `pos`, RoPE on consecutive pairs (B6), per-head scores over `t <= pos` divided by `sqrtf(head_size)`, `softmax(att, pos + 1)`, weighted sum of V, `wo`, residual add, `rmsnorm`, `w1` and `w3`, SwiGLU (`val *= (1.0f / (1.0f + expf(-val))); val *= s->hb2[i];`), `w2`, residual add; final `rmsnorm`, classifier matmul. GQA by `(h / kv_mul) * head_size` with `kv_mul = p->n_heads / p->n_kv_heads`. The prompt is processed one token at a time through the same function.
- Constants not in the header: RMSNorm epsilon `1e-5f` and RoPE base `10000.0f` are hard-coded (lines 189, 267).
- Greedy: `if (sampler->temperature == 0.0f) { // greedy argmax sampling ... next = sample_argmax(logits, sampler->vocab_size);` (line 694); `sample_argmax` keeps the first maximum (`if (probabilities[i] > max_p)`, lines 590 to 601). Generation starts from BOS = 1 ("add optional BOS (=1) token") and stops when `next == 1` (lines 476, 762 to 763). Usage: `-t <float> temperature in [0,inf], default 1.0`, `-n <int> number of steps to run for, default 256`.
- `run.c` reads only the legacy ("version 0") layout; the README's to-do list says "add support in run.c of reading version 1+ files from export, later deprecate "version 0"" (README line 406).

- https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c
- https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/README.md

### D2. Small real checkpoints in that format exist, down to about 1 MB, under the MIT licence

**Verdict: CONFIRMED.**

- Hugging Face Hub API for `karpathy/tinyllamas` (revision `0bd21da7698eaf29a0d7de3992de8a46ef624add`, last modified 2023-08-15, tag `license:mit`): files and sizes `stories260K/stories260K.bin` 1,056,540 bytes, `stories260K/tok512.bin` 6,227, `stories15M.bin` 60,816,028, `stories42M.bin` 167,020,572, `stories110M.bin` 438,381,596.
- Headers read today by a 28-byte range request: `stories15M.bin` → `dim 288, hidden_dim 768, n_layers 6, n_heads 6, n_kv_heads 6, vocab_size 32000, seq_len 256`; `stories260K.bin` → `dim 64, hidden_dim 172, n_layers 5, n_heads 8, n_kv_heads 4, vocab_size 512, seq_len 512`. Both have positive `vocab_size`, so the classifier is the embedding. Our arithmetic (E1) reproduces both file sizes exactly from these headers.
- `stories260K/readme.md`: "You'll notice that `n_kv_heads` is 4 while `n_heads` is 8, so two heads at a time share their key,value projections, i.e. this model is 2X multiquery." It documents a greedy run: `./run stories260K/stories260K.bin -z stories260K/tok512.bin -t 0.0` beginning "Once upon a time, there was a little girl named Lily." (That text is the model's output on the author's machine; not reproduced here.)
- Head sizes: 288 / 6 = 48 (15M), 64 / 8 = 8 (260K). Neither is M7's fixed `HEAD_DIM 128` (project notes).

- https://huggingface.co/karpathy/tinyllamas (API: https://huggingface.co/api/models/karpathy/tinyllamas?blobs=true)
- https://huggingface.co/karpathy/tinyllamas/blob/main/stories260K/readme.md

### D3. safetensors: an 8-byte length, a JSON header, a row-major little-endian byte buffer

**Verdict: CONFIRMED.** `README.md` section "Format" (last commit 2026-11-20):

- "8 bytes: `N`, an unsigned little-endian 64-bit integer, containing the size of the header"; "N bytes: a JSON UTF-8 string representing the header."; "The header data MUST begin with a `{` character (0x7B)."; "The header is a dict like `{"TENSOR_NAME": {"dtype": "F16", "shape": [1, 16, 256], "data_offsets": [BEGIN, END]}, ...}`"; "`data_offsets` point to the tensor data relative to the beginning of the byte buffer (i.e. not an absolute position in the file), with `BEGIN` as the starting offset and `END` as the one-past offset"; "A special key `__metadata__` is allowed to contain free form string-to-string map."; "Endianness: Little-endian."; "Order: 'C' or row-major."; "Tensor values are not checked against, in particular NaN and +/-Inf could be in the file".
- So a reader can parse it with the standard library only (a length, `json`, slices), consistent with PLAN.md's "Python 3 (standard library only)" rule.

- https://github.com/huggingface/safetensors/blob/11319d360db5170565db914601fbf375762bf8eb/README.md (section "Format")

### D4. A Hugging Face Llama-architecture checkpoint (SmolLM2-135M) maps onto the same pipeline with four conversions

**Verdict: PARTLY.** The architecture matches llama2.c's step (RMSNorm, GQA attention with RoPE, SwiGLU, tied head), but the weights are bfloat16, the RoPE pairing is the other one, and the shapes differ from M7's.

- `HuggingFaceTB/SmolLM2-135M` (Hub API revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, 2025-02-06, `license:apache-2.0`): files include `config.json`, `model.safetensors` (269,060,552 bytes), `tokenizer.json`. `config.json` as served: `"architectures": ["LlamaForCausalLM"]`, `"hidden_size": 576`, `"intermediate_size": 1536`, `"num_attention_heads": 9`, `"num_hidden_layers": 30`, `"num_key_value_heads": 3`, `"rms_norm_eps": 1e-05`, `"rope_interleaved": false`, `"rope_theta": 100000`, `"tie_word_embeddings": true`, `"torch_dtype": "bfloat16"`, `"vocab_size": 49152`, `"bos_token_id": 0`, `"eos_token_id": 0`.
- Header of `model.safetensors`, read by range request (N = 30,528 bytes): `__metadata__ = {'format': 'pt'}`, 272 tensors, every dtype `BF16`; names and shapes `model.embed_tokens.weight [49152, 576]`, `model.layers.0.input_layernorm.weight [576]`, `model.layers.0.self_attn.q_proj.weight [576, 576]`, `k_proj.weight [192, 576]`, `v_proj.weight [192, 576]`, `o_proj.weight [576, 576]`, `model.layers.0.post_attention_layernorm.weight [576]`, `mlp.gate_proj.weight [1536, 576]`, `mlp.up_proj.weight [1536, 576]`, `mlp.down_proj.weight [576, 1536]`, `model.norm.weight [576]`; no `lm_head.weight` (tied).
- Hugging Face's Llama code: `LlamaConfig` fields `vocab_size`, `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `hidden_act = "silu"`, `rms_norm_eps = 1e-6` default, `tie_word_embeddings = False` default, `rope_parameters`, `head_dim` defaulting to `hidden_size // num_attention_heads` (`configuration_llama.py` lines 45 to 70); `LlamaMLP.forward`: `self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))` (`modeling_llama.py` lines 164 to 166). A top-level `rope_theta` in an older `config.json` is moved into `rope_parameters` on load: `rope_theta = kwargs.pop("rope_theta", getattr(self, "rope_theta", self.default_theta))` then `rope_parameters.setdefault("rope_theta", rope_theta)`, with `default_theta = 10_000.0` (`modeling_rope_utils.py` lines 760, 785 to 794).
- The four conversions for our pipeline (ours, from the facts above): (1) bfloat16 to float32 by placing the 16 stored bits in the top half of a 32-bit word (exact; layout appA A2); (2) the NeoX pairing, or permute Q and K rows as llama.cpp's converter does (B6); (3) GQA with 9 query heads over 3 KV heads, head size 576 / 9 = 64; (4) RoPE base 100000 and epsilon 1e-5 from the config, not llama2.c's constants. A tokenizer is not needed if the README compares token ids.
- Not verified: that a Hugging Face greedy `generate` on this checkpoint matches any course output; nothing was run.

- https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json
- https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M?blobs=true
- https://github.com/huggingface/transformers/blob/4d9ef623e3b0a1fb6c12891a907e0e50d41f0af8/src/transformers/models/llama/configuration_llama.py
- https://github.com/huggingface/transformers/blob/4d9ef623e3b0a1fb6c12891a907e0e50d41f0af8/src/transformers/modeling_rope_utils.py (lines 755 to 800)

### D5. GGUF is the format to use for M10's real-checkpoint path

**Verdict: PARTLY.** GGUF is documented and is what llama.cpp loads, but for this lab it adds work the other two formats do not: typed metadata, alignment padding, and usually quantised tensor types (that is M9's subject, and M10 runs FP32).

- `ggml/docs/gguf.md` (last commit 2026-11-08): header `uint32_t magic;` "Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`."; `uint32_t version;` "Must be `3` for version described in this spec"; `uint64_t tensor_count;`, `uint64_t metadata_kv_count;`. Tensor info: name ("at most 64 bytes long"), `uint32_t n_dimensions;` ("Currently at most 4"), `uint64_t dimensions[n_dimensions];`, `ggml_type type;`, `uint64_t offset;` "relative to `tensor_data`" and "a multiple of `ALIGNMENT`". "`general.alignment: uint32` ... If the alignment is **not** specified, assume it is `32`." "Models are little-endian by default."
- Our recommendation for the README: llama2.c `.bin` (D1, D2) as the first real checkpoint (float32, no parsing beyond 7 integers, a C reference that runs on any CPU), safetensors plus `config.json` (D3, D4) as the second.

- https://github.com/ggml-org/ggml/blob/d9fe7bff3d2bf4d077e25c654486d02c321734aa/docs/gguf.md (lines 147 to 360, 390)

### D6. llama2.c's Hugging Face import permutes Q and K correctly for grouped-query models

**Verdict: WRONG** by our reading of the source (not run).

- `export.py` (last commit 2023-10-09), `load_hf_model`: `def permute_reverse(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim): return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)`, applied as `permute_reverse(hf_dict[f'model.layers.{i}.self_attn.k_proj.weight'])` with the defaults (lines 468 to 475). For a GQA model the K projection has `n_kv_heads * head_size` rows, not `dim`, so `view(n_heads, 2, dim // n_heads // 2, dim)` does not match its shape. llama.cpp's `permute` switches to `n_head_kv` for K (B6).
- Consequence (ours): converting SmolLM2-135M (3 KV heads) with llama2.c's script would fail or mis-permute; the README should not send the reader there. Not verified by running it.

- https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/export.py (lines 437 to 480)

### D7. How real projects compare a GPU result with a reference: a normalised error, not elementwise equality

**Verdict: CONFIRMED.**

- llama.cpp `tests/test-backend-ops.cpp` (last commit 2026-11-20), header comment: "For the forward pass it asserts that the results of multiple backends computing the same GGML ops are consistent." Usage: "test (default) compare with CPU backend for correctness", options `-o <op|regex,..>`, `-b <backend> run tests on the given backend (e.g. CPU, MTL0, CUDA0)`, example `-o ADD,MUL_MAT` (lines 1 to 10, 14882 to 14902).
- Metric: `static double nmse(const float * a, const float * b, size_t n) { ... mse_a_b += (a_i - b_i) * (a_i - b_i); mse_a_0 += a_i * a_i; } return mse_a_b / mse_a_0; }` (lines 326 to 339), with a default threshold `virtual double max_nmse_err() { return 1e-7; }` (line 1266 to 1267), overridden per op.
- For M10 (ours): "exact greedy agreement" and "a stated tolerance on logits" are two different tests. Logits can differ by rounding across backends (x1 C5, ch10 A9, A11), and a greedy step whose top two logits are closer than that difference can flip the token, after which every later token may differ. So the lab should report the logit error (an NMSE or max absolute error per step), and check greedy agreement on steps whose top-two margin exceeds the tolerance, flagging the others. The 1e-7 figure is llama.cpp's per-op default, not a value for a whole model.

- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/tests/test-backend-ops.cpp

### D8. Each framework lets you run your own kernel inside it, on CUDA and on Metal

**Verdict: CONFIRMED.** These are the bridges for "your course kernel, called from the framework you use".

- MLX: `mx.fast.metal_kernel(name, input_names, output_names, source, header="", ensure_row_contiguous=True, atomic_outputs=False, ...)`: "A jit-compiled custom Metal kernel defined from a source string." "source (str): Source code. This is the body of a function in Metal, the function signature will be automatically generated." Compile option `"math_mode"`: "``"safe"`` preserves IEEE behavior for special values such as ``exp(-inf) == 0``. Default: ``"safe"``." `mx.fast.cuda_kernel(...)`: "A jit-compiled custom CUDA kernel defined from a source string. This is the CUDA equivalent of :ref:`custom_metal_kernels`."; "grid (tuple[int, int, int]): ... For compatibility with :func:`metal_kernel` the grid is in threads and not in threadgroups." (`python/src/fast.cpp` lines 441 to 640, last commit 2026-11-20).
- PyTorch MPS, docs 2.15, `torch.mps.compile_shader(source)`: "Compiles compute shader from source and allows one to invoke kernels defined there from the comfort of Python runtime", example `lib = torch.mps.compile_shader("kernel void full(device float* out, constant float& val, uint idx [[thread_position_in_grid]]) { out[idx] = val; }")` then `lib.full(x, 3.14)`.
- PyTorch CUDA (and ROCm), docs 2.15, `torch.utils.cpp_extension.load_inline(name, cpp_sources, cuda_sources=None, ...)`: "Load a PyTorch C++ extension just-in-time (JIT) from string sources."
- llama.cpp: no plug-in kernel API; a new op is a `GGML_OP_*` plus a CPU implementation first ("When adding support for a new model or feature, focus on **CPU support only** in the initial PR"), then backends, each with a `test-backend-ops` case (F1).

- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/python/src/fast.cpp
- https://docs.pytorch.org/docs/2.15/generated/torch.mps.compile_shader.html
- https://docs.pytorch.org/docs/2.15/cpp_extension.html

### D9. The reference implementation's greedy rule is "first maximum"

**Verdict: PARTLY.** True for the two references M10's README would name (llama2.c's `sample_argmax`, D1; `torch.argmax`, documented "the indices of the first maximal value", ch10 D4, used by transformers' greedy path, x1 D4). Not true for every GPU argmax: llama.cpp's CUDA and Metal kernels break ties differently (ch10 D2). The lab's own GPU argmax (M4) must state its rule and the CPU reference must use the same one.

---

## Group E: Our arithmetic

### E1. The tinyllamas file sizes follow from their headers and llama2.c's layout

**Verdict: CONFIRMED** (computed in Python on the host).

Floats per file = vocab·dim + L·dim (attention norm) + L·dim·dim (wq) + 2·L·dim·kv_dim (wk, wv) + L·dim·dim (wo) + L·dim (ffn norm) + 3·L·dim·hidden + dim (final norm) + seq_len·head_size (the two skipped RoPE tables), with kv_dim = dim·n_kv_heads / n_heads; bytes = 4·floats + 28.

| File | Header | Floats | Bytes computed | Bytes on the Hub |
|---|---|---|---|---|
| `stories15M.bin` | 288, 768, 6, 6, 6, 32000, 256 | 15,204,000 | 60,816,028 | 60,816,028 |
| `stories260K.bin` | 64, 172, 5, 8, 4, 512, 512 | 264,128 | 1,056,540 | 1,056,540 |

So both are legacy-format files with a shared classifier, and a reader's loader can check a download by this formula before running it.

### E2. Head sizes of the candidate checkpoints against M7's kernel

**Verdict: CONFIRMED.**

| Model | dim / heads | Q heads / KV heads | M7 (`HEAD_DIM 128`) fits? |
|---|---|---|---|
| stories260K | 64 / 8 = 8 | 8 / 4 | No |
| stories15M | 288 / 6 = 48 | 6 / 6 | No |
| SmolLM2-135M | 576 / 9 = 64 | 9 / 3 | No |

None of the small real checkpoints has head size 128. M10's synthetic model can use 128; the README's real-checkpoint path needs a decode kernel with the head size as a parameter (or a compile-time constant per model), which is a change to M7 (project notes).

---

## Group F: Contributing (goal 3)

### F1. llama.cpp's contribution rules for a kernel change

**Verdict: CONFIRMED.** `CONTRIBUTING.md` (last commit 2026-11-16):

- "If you modified the `ggml` source, run the `test-backend-ops` tool to check whether different backend implementations of the `ggml` operators produce consistent results (this requires access to at least two different `ggml` backends)"; "If you modified a `ggml` operator or added a new one, add the corresponding test cases to `test-backend-ops`".
- "Bug-fix PRs must include a reproducible issue and a regression test that fails before your change and passes after."
- AI policy: "AI-generated code is allowed. You are 100% responsible for every line, however it was produced." "Undisclosed AI usage may result in your account being permanently banned from contributing to the project." Contributors must "Explicitly disclose the manner in which AI was employed" and "It is strictly prohibited to use AI to write your posts for you (bug reports, feature requests, pull request descriptions, Github discussions, responding to humans, ...)."
- Style: "use 4 spaces for indentation, brackets on the same line, `void * ptr`, `int & a`".

- https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/CONTRIBUTING.md

### F2. MLX and PyTorch require tests with every change and allow AI assistance under the contributor's responsibility

**Verdict: CONFIRMED.**

- MLX `CONTRIBUTING.md` (2026-09-10): "All contributions require thorough testing." "Ensure all existing tests pass before submitting: `python3 python/tests/run.py`"; "Format the code with `pre-commit run --all` before submitting a pull request."; "You may use productivity tools, including AI-assisted coding, to help you work more efficiently. AI-generated content must meet the same standards as human-written contributions."
- PyTorch `CONTRIBUTING.md` (2026-11-11): tests run with `python test/run_test.py` or, for one file, for example `pytest test/test_nn.py -k Loss -v` (lines 540, 582); a section "AI-Assisted Development" exists (table of contents line 22; not read in detail).

- https://github.com/ml-explore/mlx/blob/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/CONTRIBUTING.md
- https://github.com/pytorch/pytorch/blob/a6ec1fa444f247690030fe73106156341c95fcc3/CONTRIBUTING.md

---

## Project notes (not vendor facts)

Recorded so the lab author does not have to rediscover them; these describe this repository on the check date.

- **No `labs/x2` today** (directory absent). M9's quantised linear layer does not exist yet, so M10 runs FP32 weights without an M9 switch unless X2 lands one before integration.
- **What M0 to M8 provide** (from the solution `kernel.cu` files): M0 `launchAddBias`, `launchScaleShift`, `launchAddResidual` (lab 2.3); M2 `launchGemm` (lab 5.3); M3 `launchLinear`, `launchLinearWidth` (lab 15.6); M4 `launchRowMax`, `launchSoftmaxRows`, `launchArgmaxRows` (lab 10.3); M5 sampler pieces including `launchPhiloxKat` (lab X1.3); M6/M7 flash attention and decode with KV append (labs 20.2, 20.4, 20.5; `#define HEAD_DIM 128` in `lab4_decode` and `lab5_gqa_decode`, Metal too); M8 `launchGemvGather`, `launchGemvSplitK` (lab 22.1), `launchEmbed` and `launchDecodeStep` over a `DecodeModel` (lab 22.2: embedding, per-layer linear plus bias plus residual, LM head with the embedding as weight, softmax, argmax; no attention, no normalisation).
- **Missing pieces for a Llama-style step** (searched the solutions for RMSNorm, RoPE, rotary, SiLU, SwiGLU: none): RMSNorm (B2), RoPE (B6), the SiLU-gated feed-forward (B1), and a head size other than 128 (E2). Smallest versions: RMSNorm as M4's row reduction with a sum of squares; RoPE as an elementwise kernel over pairs with one pairing stated; SwiGLU as M0's elementwise kernel over two inputs.
- **Reference conventions to copy** (ours, from D1 and D4): llama2.c's step is a complete CPU reference in about 130 lines; its weight order and `W (d,n) @ x` convention match the project's row-major `float`, batch-first rule.

## Answers to the brief's questions

| Question | Answer | Where |
|---|---|---|
| Where an op lives (method) | llama.cpp: `GGML_OP_*` → `ggml_cuda_compute_forward` / `ggml_metal_op_encode_impl`, after a fusion pass; MLX: `Primitive::eval_gpu` in `backend/metal` and `backend/cuda`, plus JIT for `mx.compile`; PyTorch: `native_functions.yaml` dispatch keys, ufunc codegen for `add` | A1, A2, A3 |
| Elementwise | llama.cpp `unary_op_kernel`, `k_bin_bcast`, Metal `kernel_unary_impl`; MLX `binary_vv` (CUDA), `binary.h` (Metal); PyTorch `gpu_kernel` lambdas, MPS `BinaryKernel.metal` | B1 |
| Row reduction | RMSNorm: llama.cpp `rms_norm_f32`, `kernel_rms_norm_fuse_impl`; MLX `rms_single_row`, CUDA `rms_norm`; PyTorch `_fused_rms_norm_cuda`, MPS `RMSNorm.metal`. Softmax: ch10 D1 to D4 plus MLX CUDA's online softmax | B2, B3 |
| GEMM / GEMV | Shape-dependent everywhere; llama.cpp `ggml_cuda_mul_mat` chain; MLX `gemv_impl` or cuBLASLt; PyTorch MPS `gemv_*` or `MPSNDArrayMatrixMultiplication` | B4 |
| Attention | llama.cpp own kernels on every backend; MLX own on Metal, cuDNN plus own decode on CUDA; PyTorch FlashAttention-2 submodule, priority order, MPS own kernels | B5 |
| Sampling | x1 D1 to D8; MLX CUDA uses the same Threefry | B8 |
| HIP path | llama.cpp and PyTorch: the CUDA files are the HIP files; MLX: none | B9 |
| Real checkpoint for M10's README | llama2.c `.bin` (stories260K, 1 MB, MIT) first; SmolLM2-135M safetensors (Apache-2.0) second, with bf16, RoPE pairing, GQA 9/3, head 64; not GGUF | D1 to D6, E1, E2 |
| How to compare with a reference | NMSE as llama.cpp does; greedy agreement checked with a margin | D7, D9 |

---

## Summary of corrections

Every WRONG and PARTLY item, and every failed or unfound sub-claim:

- A3 (WRONG): PyTorch's CUDA `add` has no source file; `add.out` is a ufunc op whose CUDA kernel `UfuncCUDA_add.cu` is generated by torchgen at build time; MPS needs a hand-written `add_out_mps` for that reason.
- A4: the three licences are permissive (llama.cpp and MLX MIT, PyTorch three-clause BSD-style, which GitHub reports as `NOASSERTION`); whether a short excerpt is a "substantial portion" needing the full notice could not be verified; attribute every excerpt.
- B3 (WRONG): MLX computes softmax differently on its two GPU backends (Metal: max then sum with `fast::exp`; CUDA: one-pass online normaliser with `__expf` and cooperative groups); PyTorch MPS uses `precise::exp` where MLX Metal uses `fast::exp`.
- B4 (WRONG): no project sends every matrix product to one kernel; MLX CUDA uses a warp-per-row GEMV when M or N is 1 and cuBLASLt otherwise; MLX Metal adds a "wide" GEMV for up to 5 vectors.
- B5: MLX on CUDA uses cuDNN's SDPA for prefill and for unmasked decodes over at least 256 keys, its own `exp2f` vector kernels otherwise; PyTorch's CUDA flash kernels live in the `third_party/flash-attention` submodule; default order flash, efficient, math, cuDNN. Which cuDNN version MLX needs could not be verified.
- B6 (WRONG): RoPE has two pairings; llama.cpp runs Llama with consecutive pairs and permutes HF's Q/K weights to fit; HF, mlx-lm (default `rope_traditional=False`) and SmolLM2 (`"rope_interleaved": false`) use the half-split pairing; llama2.c uses consecutive pairs; PyTorch has no RoPE op.
- B9: MLX has no AMD backend (an experimental ROCm pull request, #2300, is open and unmerged); llama.cpp's and PyTorch's CUDA files are their HIP files.
- D4: SmolLM2-135M fits the Llama step but is bfloat16, half-split RoPE, 9 query and 3 KV heads with head size 64, RoPE base 100000; old `config.json` files' top-level `rope_theta` is moved into `rope_parameters` by transformers.
- D5: GGUF is not the right first format for M10 (typed metadata, 32-byte default alignment, usually quantised tensors); llama2.c `.bin` and safetensors are simpler.
- D6 (WRONG by reading): llama2.c's `export.py` permutes K with `n_heads` and `dim`, which does not fit a grouped-query K projection; do not use it for SmolLM2.
- D9: "first maximum" holds for llama2.c and `torch.argmax`, not for llama.cpp's GPU argmax kernels.
- E2: none of the small real checkpoints has head size 128, the constant in M7's decode kernels.

Path and status facts worth knowing even where the verdict is CONFIRMED:

- A1: llama.cpp's CUDA backend fuses `RMS_NORM` + `MUL` (+ `ADD`) before dispatch; `GGML_CUDA_DISABLE_FUSION` turns it off. The converter moved from `convert_hf_to_gguf.py` into a `conversion/` package (`conversion/llama.py`).
- A2: mlx-lm's models are now packages (`mlx_lm/models/llama/{config,layers,model}.py`), with shared layers in `mlx_lm/layers/`.
- B1, B2: MLX CUDA and llama.cpp both use the `float*` to vector-pointer cast behind a run-time alignment test (infra A6).
- C1, C2: Metal variants are function constants (Metal 1.2 and later), Metal kernels are found by `[[host_name]]` strings (Metal 2.2 and later).
- D1, D2: `stories260K.bin` (1,056,540 bytes, MIT) is a complete GQA Llama-style model with a 512-token vocabulary, and `run.c` is its CPU reference with greedy `-t 0.0`.
- D7: llama.cpp's `test-backend-ops` compares each backend with the CPU by NMSE, default threshold 1e-7 per op.
- D8: `mx.fast.metal_kernel` (default math mode "safe"), `mx.fast.cuda_kernel` (grid in threads), `torch.mps.compile_shader`, `torch.utils.cpp_extension.load_inline` run a hand-written kernel inside the framework.
- F1: llama.cpp requires disclosure of AI use and forbids AI-written PR descriptions and posts.

Not checked: any build or run of the three projects; MLX's `looped_softmax` structure; PyTorch's `cunn_SoftMaxForward` internals; the "AI-Assisted Development" section of PyTorch's guide; the TinyStories training data licence; any tokenizer.

## Where the backends really differ

One line per cell, for what X3's tour and M10 use. "Same" means the same source file serves both. Nothing here was run on CUDA or AMD hardware; the Metal cells are from source and the MSL spec.

| Feature | CUDA | HIP (ROCm) | Metal |
|---|---|---|---|
| Where llama.cpp's kernels are | `ggml/src/ggml-cuda/*.cu` | Same files, compiled as HIP (`ggml-hip/CMakeLists.txt`) | `ggml/src/ggml-metal/kernels/*.metal`, separate code |
| Where MLX's kernels are | `mlx/backend/cuda/` (cuBLASLt, cuDNN frontend, cooperative groups) | None (experimental PR unmerged) | `mlx/backend/metal/kernels/` |
| Where PyTorch's kernels are | `aten/src/ATen/native/cuda/`, generated `UfuncCUDA_*.cu`, FlashAttention submodule | Same sources hipified at build; AOTriton / CK for attention | `aten/src/ATen/native/mps/kernels/*.metal` + `operations/*.mm`, some MPSGraph |
| Op to kernel map | llama.cpp `ggml_cuda_compute_forward`; MLX `eval_gpu` in `backend/cuda`; PyTorch `CUDA:` dispatch key | Same as CUDA | llama.cpp `ggml_metal_op_encode_impl`; MLX `eval_gpu` in `backend/metal`; PyTorch `MPS:` key |
| Kernel variants | Template parameters | Same templates | Function constants (`[[function_constant]]`, Metal 1.2+) and `[[host_name]]` instantiations |
| Finding a kernel from the host | Function symbol | Same | The name string passed to the pipeline lookup |
| Elementwise | One thread per element (llama.cpp); coarsened with aligned vector loads (MLX, PyTorch) | Same source | One thread per element or grid-stride loop; `float4` instantiations |
| Row reduction (RMSNorm, softmax) | Block per row, warp shuffle + shared memory; MLX: `cg::reduce`, online softmax | Same source (llama.cpp, PyTorch); logical warp 32 in llama.cpp | Threadgroup per row, `simd_sum`/`simd_max` + threadgroup memory; `fast::exp` (MLX) vs `precise::exp` (PyTorch MPS) |
| GEMV vs GEMM | llama.cpp `mmvf`/`mmvq` up to 8 tokens; MLX warp-per-row GEMV or cuBLASLt; PyTorch cuBLAS | llama.cpp same chain with AMD tuning; PyTorch hipBLAS / CK | llama.cpp `mul_mv`, `mul_mv_ext`, `mul_mm`; MLX `gemv`, `gemv_wide`, steel; PyTorch MPS `gemv_*` or `MPSNDArrayMatrixMultiplication` |
| Attention | llama.cpp own kernels; MLX cuDNN + own decode; PyTorch FA2 / mem-efficient / math / cuDNN | llama.cpp same; PyTorch AOTriton, CK | llama.cpp own; MLX own vector and steel kernels; PyTorch MPS own kernels |
| RoPE pairing | Both exist (`rope_norm`, `rope_neox`; MLX `traditional`) | Same | Both exist (`kernel_rope_norm`, `kernel_rope_neox`; MLX function constant) |
| Running your own kernel in the framework | `mx.fast.cuda_kernel`; `torch.utils.cpp_extension.load_inline` | `load_inline` on ROCm builds (not checked separately); no MLX | `mx.fast.metal_kernel` (math mode "safe" by default); `torch.mps.compile_shader` |
| Backend correctness test | llama.cpp `test-backend-ops -b CUDA0` against CPU by NMSE | Same tool on the HIP backend | `test-backend-ops -b MTL0` |
