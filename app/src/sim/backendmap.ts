// Where an op becomes a kernel in llama.cpp, MLX and PyTorch, on CUDA, HIP and Metal, as read
// in the X3 facts pass (notes/today/x3-facts.md, 2026-11-20, at the commits in x3sources.ts).
//
// Every cell cites the facts items it rests on. Line numbers appear only where x3-facts gives
// them, at the pinned commit. Cells that rest on another chapter's facts item (read at master
// the same day) say so with readAt: 'master' and carry that item's master URL, never pinned
// line numbers. A cell whose file was not read says so with the tag 'not read in the facts
// pass' instead of guessing.
//
// Also here, because the lessons draw them from the same data: the side-by-side entry points
// (ENTRY_POINTS), the routes the "Follow the call" stepper walks (ROUTES), the variant
// mechanisms X3.2 sorts (VARIANT_MECHANISMS), and the graph and names the "One RMSNorm call"
// scene animates (ONE_CALL, fuseGraph).

import { permalink, type Lines, type ProjectId } from './x3sources';

export type OpId = 'embedding' | 'add' | 'silu' | 'rmsnorm' | 'softmax' | 'matmul' | 'rope' | 'attention' | 'sampling';
export type Project = 'llama.cpp' | 'mlx' | 'pytorch';
export type Backend = 'cuda' | 'hip' | 'metal';
export const PROJECTS: Project[] = ['llama.cpp', 'mlx', 'pytorch'];
export const BACKENDS: Backend[] = ['cuda', 'hip', 'metal'];
export const PROJECT_NAME: Record<Project, string> = { 'llama.cpp': 'llama.cpp', mlx: 'MLX', pytorch: 'PyTorch' };
export const BACKEND_NAME: Record<Backend, string> = { cuda: 'CUDA', hip: 'HIP', metal: 'Metal' };

export const TAGS = [
  'own kernel',
  'same source as CUDA',
  'hipified at build time',
  'generated at build time',
  'generated at run time (mx.compile)',
  'vendor library',
  'third-party submodule',
  'composite of other ops',
  'fused before dispatch',
  'found by name string',
  'not read in the facts pass',
  'no backend',
] as const;
export type Tag = (typeof TAGS)[number];

/** A further permalink at the pinned commit: an entry point, a host function, a build file. */
export interface Where {
  label: string;
  path: string;
  lines?: Lines;
  /** Defaults to the cell's project. */
  project?: ProjectId;
  tree?: boolean;
}

export interface Cell {
  /** Where the op enters this backend. */
  entry: string;
  /** The host function that launches or encodes the work. */
  host: string | null;
  /** The kernel's file, relative to the repository root; null when there is no file to open. */
  file: string | null;
  /** The kernel function(s), or what stands in for them. */
  kernel: string | null;
  /** Lines of `file`, only where x3-facts.md gives them. */
  lines?: Lines;
  /** 'pinned': read at the commit in x3sources. 'master': from another chapter's facts item. */
  readAt: 'pinned' | 'master';
  /** For readAt 'master': the URL that facts item gives. */
  url?: string;
  also?: Where[];
  /** Facts items: "A1" for x3-facts.md, "ch10 D2" for notes/today/ch10-facts.md. */
  facts: string[];
  tags: Tag[];
  note?: string;
}

export interface CourseLab {
  num: string;
  /** Chapter id for the link to its labs page. */
  chapter: string;
  /** Directory under labs/, or null for a lab still planned. */
  dir: string | null;
  what: string;
  planned?: boolean;
}

export interface OpRow {
  id: OpId;
  label: string;
  /** The tour's eight ops group add and SiLU as "elementwise". */
  tour: string;
  /** The section of module X3 that reads it. */
  section: string;
  course: CourseLab[];
  cells: Record<Project, Record<Backend, Cell>>;
}

// ---------------------------------------------------------------- helpers for the HIP columns
const LLAMA_HIP_NOTE = 'ggml-hip/CMakeLists.txt compiles ../ggml-cuda/*.cu with LANGUAGE HIP (lines 60 to 70).';
const LLAMA_HIP_CMAKE: Where = { label: 'ggml-hip/CMakeLists.txt', path: 'ggml/src/ggml-hip/CMakeLists.txt', lines: [60, 70] };

/** llama.cpp's HIP cell: the CUDA file itself. */
function llamaHip(c: Cell): Cell {
  return { ...c, also: [...(c.also ?? []), LLAMA_HIP_CMAKE], facts: [...c.facts, 'B9'], tags: ['same source as CUDA'], note: LLAMA_HIP_NOTE };
}
/** PyTorch's HIP cell: the CUDA source after the build-time hipify step. */
function torchHip(c: Cell): Cell {
  return {
    ...c,
    facts: [...c.facts, 'B9', 'ch24 D3'],
    tags: [...c.tags.filter((t) => t === 'composite of other ops'), 'hipified at build time'],
    note: 'tools/amd_build/build_amd.py translates aten/src/ATen/native/cuda/* at build time (ch24 D3).',
  };
}
const MLX_NO_HIP: Cell = {
  entry: 'none', host: null, file: null, kernel: null, readAt: 'pinned',
  also: [{ label: 'mlx/backend at the pinned commit', path: 'mlx/backend', tree: true }],
  facts: ['B9'], tags: ['no backend'],
  note: 'mlx/backend holds common, cpu, cuda, gpu, metal, no_cpu and no_gpu. An experimental ROCm pull request (#2300) is open and unmerged.',
};

// ---------------------------------------------------------------- llama.cpp entry points
const CU = 'ggml/src/ggml-cuda/';
const MT = 'ggml/src/ggml-metal/';
const CUDA_SWITCH = (line: number): Where => ({ label: `ggml_cuda_compute_forward, line ${line}`, path: CU + 'ggml-cuda.cu', lines: line });
const METAL_SWITCH = (line?: number): Where => ({ label: `ggml_metal_op_encode_impl${line ? ', line ' + line : ' (line 187)'}`, path: MT + 'ggml-metal-ops.cpp', lines: line ?? 187 });
const MLX_CU = 'mlx/backend/cuda/';
const MLX_MT = 'mlx/backend/metal/';
const YAML = 'aten/src/ATen/native/native_functions.yaml';
const TN = 'aten/src/ATen/native/';

const lab = (num: string, chapter: string, dir: string | null, what: string, planned = false): CourseLab => ({ num, chapter, dir, what, planned });
const LAB_X31 = lab('X3.1', 'x3', null, 'the Llama pieces the course has not built', true);

export const OPS: OpRow[] = [
  {
    id: 'embedding', label: 'embedding', tour: 'embedding', section: 'X3.2',
    course: [lab('22.2', 'ch22', 'ch22/lab2_batching', 'launchEmbed, one output row per token')],
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          entry: 'case GGML_OP_GET_ROWS in ggml_cuda_compute_forward', host: 'ggml_cuda_op_get_rows',
          file: CU + 'getrows.cu', kernel: 'k_get_rows (line 6), k_get_rows_float (73), k_get_rows_float_vec (105)', lines: [6, 105],
          readAt: 'pinned', also: [CUDA_SWITCH(2826)], facts: ['A1', 'B7'], tags: ['own kernel'],
          note: 'Quantised rows are dequantised as they are gathered.',
        };
        return {
          cuda,
          hip: llamaHip(cuda),
          metal: {
            entry: 'case GGML_OP_GET_ROWS in ggml_metal_op_encode_impl',
            host: 'ggml_metal_op_get_rows, which asks for "kernel_get_rows_%s" with the type name',
            file: MT + 'kernels/quantize.metal', kernel: 'kernel_get_rows_f (line 266), kernel_get_rows_q (297)', lines: [266, 297],
            readAt: 'pinned', also: [METAL_SWITCH(), { label: 'the snprintf, ggml-metal-device.cpp line 154', path: MT + 'ggml-metal-device.cpp', lines: 154 }],
            facts: ['A1', 'B7', 'C2'], tags: ['own kernel', 'found by name string'],
          },
        };
      })(),
      mlx: {
        cuda: {
          entry: 'nn.Embedding: return self.weight[x], a gather', host: null, file: null,
          kernel: 'the gather primitive’s CUDA kernel', readAt: 'pinned',
          also: [{ label: 'embedding.py line 32', path: 'python/mlx/nn/layers/embedding.py', lines: 32 }],
          facts: ['A2', 'B7'], tags: ['composite of other ops', 'not read in the facts pass'],
        },
        hip: MLX_NO_HIP,
        metal: {
          entry: 'nn.Embedding: return self.weight[x], a gather', host: null, file: null,
          kernel: 'the gather primitive’s Metal kernel', readAt: 'pinned',
          also: [{ label: 'embedding.py line 32', path: 'python/mlx/nn/layers/embedding.py', lines: 32 }],
          facts: ['A2', 'B7'], tags: ['composite of other ops', 'not read in the facts pass'],
        },
      },
      pytorch: (() => {
        const cuda: Cell = {
          entry: '- func: embedding, dispatch CompositeExplicitAutograd: embedding_symint',
          host: 'embedding_symint: weight.index_select(0, indices.reshape(-1))', file: null,
          kernel: 'index_select’s CUDA kernel', readAt: 'pinned',
          also: [{ label: 'native_functions.yaml lines 2261 to 2263', path: YAML, lines: [2261, 2263] }, { label: 'Embedding.cpp lines 35 to 55', path: TN + 'Embedding.cpp', lines: [35, 55] }],
          facts: ['A3', 'B7'], tags: ['composite of other ops', 'not read in the facts pass'],
        };
        return {
          cuda,
          hip: torchHip(cuda),
          metal: { ...cuda, kernel: 'index_select’s MPS kernel', note: 'The same composite: there is no embedding entry per backend.' },
        };
      })(),
    },
  },
  {
    id: 'add', label: 'add', tour: 'elementwise', section: 'X3.2',
    course: [lab('2.3', 'ch02', 'ch02/lab3_elementwise', 'launchAddResidual, one thread per element'), lab('6.2', 'ch06', 'ch06/lab2_vector', 'four elements per thread, float4 loads')],
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          entry: 'case GGML_OP_ADD in ggml_cuda_compute_forward', host: 'ggml_cuda_op_add', file: CU + 'binbcast.cu', kernel: 'k_bin_bcast',
          readAt: 'pinned', also: [CUDA_SWITCH(2847)], facts: ['A1', 'B1', 'ch03 D1'], tags: ['own kernel'],
          note: 'ggml_cuda_op_fused_add fuses 2 to 8 consecutive adds.',
        };
        return {
          cuda,
          hip: llamaHip(cuda),
          metal: {
            entry: 'case GGML_OP_ADD (with SUB, MUL, DIV) in ggml_metal_op_encode_impl', host: 'ggml_metal_op_bin',
            file: MT + 'kernels/binbcast.metal', kernel: 'kernel_bin_fuse_impl, which also fuses several ops', readAt: 'pinned',
            also: [METAL_SWITCH(291)], facts: ['A1', 'B1'], tags: ['own kernel', 'found by name string'],
          },
        };
      })(),
      mlx: {
        cuda: {
          entry: 'BINARY_GPU(Add) in cuda/binary/add.cu', host: 'Add::eval_gpu, stamped out by the BINARY_GPU macro',
          file: MLX_CU + 'binary/binary.cuh', kernel: 'binary_vv<Op, In, Out, IdxT, N_READS>', lines: [93, 118], readAt: 'pinned',
          also: [{ label: 'the macro, binary.cuh lines 399 to 406', path: MLX_CU + 'binary/binary.cuh', lines: [399, 406] }],
          facts: ['A2', 'B1'], tags: ['own kernel'],
        },
        hip: MLX_NO_HIP,
        metal: {
          entry: 'BINARY_GPU(Add) in metal/binary.cpp', host: 'Add::eval_gpu, stamped out by the same macro name',
          file: MLX_MT + 'kernels/binary.h', kernel: 'binary_ss, binary_vv with WorkPerThread elements per thread', readAt: 'master',
          url: 'https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/binary.h',
          also: [{ label: 'binary.cpp line 217', path: MLX_MT + 'binary.cpp', lines: 217 }],
          facts: ['A2', 'B1', 'ch02 D3'], tags: ['own kernel'],
        },
      },
      pytorch: {
        cuda: {
          entry: '- func: add.out: structured, ufunc_inner_loop, no CUDA: key', host: 'the scalar body in ufunc/add.h: return self + alpha * other;',
          file: null, kernel: 'UfuncCUDA_add.cu, written by torchgen at build time, calling gpu_kernel(iter, ufunctor)', readAt: 'pinned',
          also: [
            { label: 'native_functions.yaml lines 565 to 571', path: YAML, lines: [565, 571] },
            { label: 'ufunc/add.h', path: TN + 'ufunc/add.h' },
            { label: 'torchgen/gen.py line 2536', path: 'torchgen/gen.py', lines: 2536 },
            { label: 'torchgen/dest/ufunc.py lines 279 to 328', path: 'torchgen/dest/ufunc.py', lines: [279, 328] },
          ],
          facts: ['A3'], tags: ['generated at build time'],
          note: 'grep native/cuda for an add kernel and you find nothing: the file exists only in a build tree.',
        },
        hip: {
          entry: '- func: add.out: structured, ufunc_inner_loop, no CUDA: key', host: null, file: null,
          kernel: 'how the ROCm build gets its add kernel', readAt: 'pinned',
          facts: ['A3', 'B9', 'ch24 D3'], tags: ['generated at build time', 'not read in the facts pass'],
          note: 'The CUDA kernel is generated by torchgen; whether hipify sees the generated file or torchgen writes a HIP one was not read.',
        },
        metal: {
          entry: 'add_out_mps, hand-written: “ufunc codegen only emits CPU/CUDA impls, so MPS needs this entry”',
          host: 'add_stub → lib.exec_binary_kernel(iter, "add") or "add_alpha"', file: TN + 'mps/kernels/BinaryKernel.metal',
          kernel: 'add_functor, add_alpha_functor', lines: [22, 25], readAt: 'pinned',
          also: [{ label: 'BinaryOps.mm lines 16 to 19', path: TN + 'mps/operations/BinaryOps.mm', lines: [16, 19] }, { label: 'BinaryKernel.mm lines 182 to 187', path: TN + 'mps/operations/BinaryKernel.mm', lines: [182, 187] }],
          facts: ['A3'], tags: ['own kernel', 'found by name string'],
        },
      },
    },
  },
  {
    id: 'silu', label: 'SiLU', tour: 'elementwise', section: 'X3.2',
    course: [lab('2.3', 'ch02', 'ch02/lab3_elementwise', 'the elementwise kernel SiLU is'), LAB_X31],
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          entry: 'case GGML_UNARY_OP_SILU in ggml_cuda_compute_forward', host: 'ggml_cuda_op_silu = ggml_cuda_op_unary<op_silu>',
          file: CU + 'unary.cu', kernel: 'unary_op_kernel<op, T>, the op a template parameter', lines: [116, 127], readAt: 'pinned',
          also: [CUDA_SWITCH(2882), { label: 'ggml_cuda_op_silu_single, unary.cuh lines 98 to 100', path: CU + 'unary.cuh', lines: [98, 100] }],
          facts: ['A1', 'B1'], tags: ['own kernel'],
          note: 'SwiGLU is case GGML_GLU_OP_SWIGLU (line 2945): unary_gated_op_kernel, one thread per output (lines 307 to 321).',
        };
        return {
          cuda,
          hip: llamaHip(cuda),
          metal: {
            entry: 'case GGML_OP_UNARY in ggml_metal_op_encode_impl', host: 'ggml_metal_op_unary',
            file: MT + 'kernels/unary.metal', kernel: 'kernel_unary_impl, the op a function constant FC_unary_op', lines: [132, 134], readAt: 'pinned',
            also: [METAL_SWITCH(319)], facts: ['A1', 'B1', 'C1'], tags: ['own kernel', 'found by name string'],
            note: 'SwiGLU is case GGML_OP_GLU (line 327): kernel_swiglu, a grid-stride loop (lines 290 to 309).',
          },
        };
      })(),
      mlx: {
        cuda: {
          entry: 'nn.silu is decorated with mx.compile', host: 'Compiled::eval_gpu, cuda/compiled.cpp line 225', file: null,
          kernel: 'source written at run time (__global__ void {}( ...), built through cu::get_jit_module', readAt: 'pinned',
          also: [{ label: 'activations.py lines 136 to 145', path: 'python/mlx/nn/layers/activations.py', lines: [136, 145] }, { label: 'cuda/compiled.cpp lines 225 to 245', path: MLX_CU + 'compiled.cpp', lines: [225, 245] }],
          facts: ['A2'], tags: ['generated at run time (mx.compile)'],
          note: 'There is no file to open: the kernel exists after the first call. MLX computes x * sigmoid(x).',
        },
        hip: MLX_NO_HIP,
        metal: {
          entry: 'nn.silu is decorated with mx.compile', host: 'Compiled::eval_gpu, metal/compiled.cpp line 257', file: null,
          kernel: 'source written at run time with [[host_name("{0}")]], compiled through d.get_library(...)', readAt: 'pinned',
          also: [{ label: 'activations.py lines 136 to 145', path: 'python/mlx/nn/layers/activations.py', lines: [136, 145] }, { label: 'metal/compiled.cpp lines 257 to 290', path: MLX_MT + 'compiled.cpp', lines: [257, 290] }],
          facts: ['A2', 'C2'], tags: ['generated at run time (mx.compile)', 'found by name string'],
          note: 'There is no file to open: the kernel exists after the first call.',
        },
      },
      pytorch: (() => {
        const cuda: Cell = {
          entry: '- func: silu.out, dispatch CPU, CUDA, MPS, MTIA, XPU: silu_out', host: 'silu_stub, filled by REGISTER_DISPATCH(silu_stub, &silu_kernel)',
          file: TN + 'cuda/ActivationSiluKernel.cu', kernel: 'a lambda passed to gpu_kernel, computed in at::opmath_type', lines: [30, 57], readAt: 'pinned',
          also: [{ label: 'native_functions.yaml line 5300', path: YAML, lines: 5300 }], facts: ['A3', 'B1', 'ch03 D6'], tags: ['own kernel'],
        };
        return {
          cuda,
          hip: torchHip(cuda),
          metal: {
            entry: '- func: silu.out, dispatch CPU, CUDA, MPS, MTIA, XPU: silu_out', host: null, file: null, kernel: 'the MPS implementation behind silu_stub',
            readAt: 'pinned', also: [{ label: 'native_functions.yaml line 5300', path: YAML, lines: 5300 }], facts: ['A3'], tags: ['not read in the facts pass'],
          },
        };
      })(),
    },
  },
  {
    id: 'rmsnorm', label: 'RMSNorm', tour: 'RMSNorm', section: 'X3.3',
    course: [lab('10.3', 'ch10', 'ch10/lab3_softmax_argmax', 'the two-stage row reduction (launchRowMax)'), LAB_X31],
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          entry: 'the fusion pass first (lines 5268 to 5271); else case GGML_OP_RMS_NORM',
          host: 'ggml_cuda_op_rms_norm_fused when RMS_NORM is followed by MUL; ggml_cuda_op_rms_norm otherwise',
          file: CU + 'norm.cu', kernel: 'rms_norm_f32, one block per row', lines: [127, 272], readAt: 'pinned',
          also: [{ label: 'the fusion pass, lines 5255 to 5272', path: CU + 'ggml-cuda.cu', lines: [5255, 5272] }, CUDA_SWITCH(2997), { label: 'GGML_CUDA_DISABLE_FUSION, line 181', path: CU + 'ggml-cuda.cu', lines: 181 }],
          facts: ['A1', 'B2'], tags: ['own kernel', 'fused before dispatch'],
          note: 'The host function you would expect is often not the one that runs: set GGML_CUDA_DISABLE_FUSION to check.',
        };
        return {
          cuda,
          hip: { ...llamaHip(cuda), tags: ['same source as CUDA', 'fused before dispatch'] },
          metal: {
            entry: 'case GGML_OP_RMS_NORM in ggml_metal_op_encode_impl', host: 'ggml_metal_op_norm, which returns how many nodes it consumed',
            file: MT + 'kernels/norm.metal', kernel: 'kernel_rms_norm_fuse_impl<float, F>; F = 2 is "kernel_rms_norm_mul_f32" (line 179)', lines: [111, 185],
            readAt: 'pinned', also: [METAL_SWITCH(414)], facts: ['A1', 'B2', 'C2'], tags: ['own kernel', 'fused before dispatch', 'found by name string'],
          },
        };
      })(),
      mlx: {
        cuda: {
          entry: 'nn.RMSNorm calls mx.fast.rms_norm', host: 'RMSNorm::eval_gpu, rms_norm.cu line 539', file: MLX_CU + 'rms_norm.cu',
          kernel: 'rms_norm_small, rms_norm, with a broadcasting block reduce', lines: 539, readAt: 'pinned',
          also: [{ label: 'normalization.py line 166', path: 'python/mlx/nn/layers/normalization.py', lines: 166 }], facts: ['A2', 'B2'], tags: ['own kernel'],
        },
        hip: MLX_NO_HIP,
        metal: {
          entry: 'nn.RMSNorm calls mx.fast.rms_norm', host: 'RMSNorm::eval_gpu, metal/normalization.cpp line 17', file: MLX_MT + 'kernels/rms_norm.metal',
          kernel: 'rms_single_row (line 13), rms_looped (84) above RMS_LOOPED_LIMIT', lines: [13, 84], readAt: 'pinned',
          also: [{ label: 'normalization.cpp lines 17 to 60', path: MLX_MT + 'normalization.cpp', lines: [17, 60] }], facts: ['A2', 'B2'], tags: ['own kernel'],
        },
      },
      pytorch: (() => {
        const cuda: Cell = {
          entry: '- func: _fused_rms_norm, dispatch CUDA: _fused_rms_norm_cuda', host: '_fused_rms_norm_cuda, layer_norm_kernel.cu line 2009',
          file: TN + 'cuda/layer_norm_kernel.cu', kernel: 'LayerNorm’s kernels with a flag: RowwiseMomentsCUDAKernel<T, T_ACC, rms_norm> (line 57)', lines: [57, 270],
          readAt: 'pinned',
          also: [{ label: 'native_functions.yaml lines 3304 to 3309', path: YAML, lines: [3304, 3309] }, { label: 'rms_norm_symint, layer_norm.cpp lines 329 to 372', path: TN + 'layer_norm.cpp', lines: [329, 372] }],
          facts: ['A3', 'B2'], tags: ['own kernel'],
        };
        return {
          cuda,
          hip: torchHip(cuda),
          metal: {
            entry: '- func: _fused_rms_norm, dispatch MPS: _fused_rms_norm_mps', host: '_fused_rms_norm_mps (RMSNorm.mm line 22), names the kernel with fmt::format',
            file: TN + 'mps/kernels/RMSNorm.metal', kernel: 'simd_sum, local_sums, metal::precise::rsqrt: MLX’s structure', lines: [54, 74], readAt: 'pinned',
            also: [{ label: 'RMSNorm.mm lines 22 to 49', path: TN + 'mps/operations/RMSNorm.mm', lines: [22, 49] }],
            facts: ['A3', 'B2', 'C2'], tags: ['own kernel', 'found by name string'],
          },
        };
      })(),
    },
  },
  {
    id: 'softmax', label: 'softmax', tour: 'softmax', section: 'X3.3',
    course: [lab('10.3', 'ch10', 'ch10/lab3_softmax_argmax', 'launchSoftmaxRows, max then sum')],
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          entry: 'case GGML_OP_SOFT_MAX in ggml_cuda_compute_forward', host: 'ggml_cuda_op_soft_max', file: CU + 'softmax.cu', kernel: 'soft_max_f32: max, expf, barrier, sum',
          readAt: 'pinned', also: [CUDA_SWITCH(3042)], facts: ['A1', 'B3', 'ch10 D1'], tags: ['own kernel'],
        };
        return {
          cuda,
          hip: { ...llamaHip(cuda), note: LLAMA_HIP_NOTE + ' softmax.cu includes <hip/hip_cooperative_groups.h> under #ifdef GGML_USE_HIP (lines 5 to 10).' },
          metal: {
            entry: 'case GGML_OP_SOFT_MAX in ggml_metal_op_encode_impl', host: 'ggml_metal_op_soft_max', file: MT + 'kernels/softmax.metal',
            kernel: 'kernel_soft_max: simd_max, two barriers, simd_sum', lines: [42, 99], readAt: 'pinned', also: [METAL_SWITCH()],
            facts: ['A1', 'B3'], tags: ['own kernel', 'found by name string'],
          },
        };
      })(),
      mlx: {
        cuda: {
          entry: 'the Softmax primitive', host: 'Softmax::eval_gpu, cuda/softmax.cu line 108', file: MLX_CU + 'softmax.cu',
          kernel: 'one block per row, an online normaliser in one pass, __expf', lines: [49, 62], readAt: 'pinned', facts: ['A2', 'B3'], tags: ['own kernel'],
          note: 'Not the algorithm of MLX’s own Metal softmax.',
        },
        hip: MLX_NO_HIP,
        metal: {
          entry: 'the Softmax primitive', host: 'Softmax::eval_gpu, metal/softmax.cpp', file: MLX_MT + 'kernels/softmax.h',
          kernel: 'softmax_single_row: max, then sum, fast::exp', readAt: 'master', url: 'https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/softmax.h',
          facts: ['A2', 'B3', 'ch10 D3'], tags: ['own kernel'],
        },
      },
      pytorch: (() => {
        const cuda: Cell = {
          entry: '- func: _softmax.out, dispatch CUDA: softmax_cuda_out', host: 'host_softmax', file: TN + 'cuda/SoftMax.cu',
          kernel: 'a warp per row up to 2048 elements; cunn_SoftMaxForwardSmem or cunn_SoftMaxForward above', lines: [1125, 1129], readAt: 'pinned',
          also: [{ label: 'native_functions.yaml lines 5607 to 5609', path: YAML, lines: [5607, 5609] }], facts: ['A3', 'B3', 'ch10 D4'], tags: ['own kernel'],
        };
        return {
          cuda,
          hip: torchHip(cuda),
          metal: {
            entry: '- func: _softmax.out, dispatch MPS: softmax_mps_out', host: 'softmax_mps_out', file: TN + 'mps/kernels/SoftMax.metal',
            kernel: 'precise::exp throughout, simd_max and simd_sum', lines: [30, 39], readAt: 'pinned',
            also: [{ label: 'native_functions.yaml lines 5607 to 5609', path: YAML, lines: [5607, 5609] }], facts: ['A3', 'B3'], tags: ['own kernel'],
          },
        };
      })(),
    },
  },
  {
    id: 'matmul', label: 'matmul', tour: 'GEMV and GEMM', section: 'X3.4',
    course: [lab('5.3', 'ch05', 'ch05/lab3_gemm', 'the tiled GEMM'), lab('15.6', 'ch15', 'ch15/lab6_linear', 'M3, the linear layer'), lab('22.1', 'ch22', 'ch22/lab1_gemv', 'GEMV for one token')],
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          entry: 'case GGML_OP_MUL_MAT in ggml_cuda_compute_forward', host: 'ggml_cuda_mul_mat: a chain of “should use” tests',
          file: CU + 'ggml-cuda.cu', kernel: 'mmvf, mmf, mmvq, mmq or cuBLAS, picked by shape and weight type', lines: [2535, 2596], readAt: 'pinned',
          also: [CUDA_SWITCH(3003)], facts: ['A1', 'B4', 'ch22 D6'], tags: ['own kernel', 'vendor library'],
          note: 'Up to 8 tokens go to the vector kernels: MMVF_MAX_BATCH_SIZE and MMVQ_MAX_BATCH_SIZE are 8 (ch22 D6).',
        };
        return {
          cuda,
          hip: llamaHip(cuda),
          metal: {
            entry: 'case GGML_OP_MUL_MAT in ggml_metal_op_encode_impl', host: 'ggml_metal_op_mul_mat', file: MT + 'kernels/mul_mv.metal',
            kernel: 'mul_mv for one token, mul_mv_ext for 2 to 8, kernel_mul_mm (mul_mm.metal) above', readAt: 'pinned', also: [METAL_SWITCH()],
            facts: ['A1', 'B4', 'ch22 D6'], tags: ['own kernel', 'found by name string'],
          },
        };
      })(),
      mlx: {
        cuda: {
          entry: 'the Matmul primitive', host: 'Matmul::eval_gpu, cuda/matmul.cpp line 143', file: MLX_CU + 'gemms/gemv.cu',
          kernel: 'gemv_impl, a warp per row, when M or N is 1; cuBLASLt otherwise', lines: [56, 83], readAt: 'pinned',
          also: [{ label: 'matmul.cpp lines 95 to 135', path: MLX_CU + 'matmul.cpp', lines: [95, 135] }, { label: 'can_use_gemv, gemv.cu line 188', path: MLX_CU + 'gemms/gemv.cu', lines: 188 }],
          facts: ['A2', 'B4'], tags: ['own kernel', 'vendor library'],
        },
        hip: MLX_NO_HIP,
        metal: {
          entry: 'the Matmul primitive', host: 'Matmul::eval_gpu, metal/matmul.cpp', file: MLX_MT + 'kernels/gemv.h',
          kernel: 'a wide GEMV for up to 5 vectors, GEMVKernel when min(M, N) is 1, steel GEMM otherwise', lines: [30, 60], readAt: 'pinned',
          also: [{ label: 'matmul.cpp lines 1584 to 1660', path: MLX_MT + 'matmul.cpp', lines: [1584, 1660] }], facts: ['A2', 'B4', 'ch15 D3'], tags: ['own kernel'],
        },
      },
      pytorch: {
        cuda: {
          entry: '- func: mm.out, dispatch CUDA: mm_out_cuda', host: 'mm_out_cuda, then gemm_internal', file: 'aten/src/ATen/cuda/CUDABlas.cpp',
          kernel: 'cuBLAS or cuBLASLt; gemv with TF32 disabled', readAt: 'master', url: 'https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp',
          also: [{ label: 'native_functions.yaml line 4151', path: YAML, lines: 4151 }], facts: ['A3', 'B4', 'ch15 D4'], tags: ['vendor library'],
        },
        hip: {
          entry: '- func: mm.out, dispatch CUDA: mm_out_cuda (the same key on ROCm)', host: 'gemm_internal', file: 'aten/src/ATen/cuda/CUDABlas.cpp',
          kernel: 'hipBLAS or Composable Kernel', readAt: 'master', url: 'https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp',
          facts: ['ch15 D4', 'B9'], tags: ['vendor library', 'hipified at build time'],
        },
        metal: {
          entry: '- func: mm.out, dispatch MPS: mm_out_mps', host: 'Matmul.mm builds names gemv_t_..., gemv_nt_... (lines 116 to 141)',
          file: TN + 'mps/kernels/Gemv.h', kernel: 'gemv_t (line 71) for rank-1 shapes; MPSNDArrayMatrixMultiplication otherwise', lines: 71, readAt: 'pinned',
          also: [{ label: 'native_functions.yaml line 4153', path: YAML, lines: 4153 }, { label: 'Matmul.mm lines 94 to 160', path: TN + 'mps/operations/Matmul.mm', lines: [94, 160] }],
          facts: ['B4'], tags: ['own kernel', 'vendor library', 'found by name string'],
        },
      },
    },
  },
  {
    id: 'rope', label: 'RoPE', tour: 'RoPE and SwiGLU', section: 'X3.7',
    course: [LAB_X31],
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          entry: 'case GGML_OP_ROPE in ggml_cuda_compute_forward', host: 'ggml_cuda_op_rope', file: CU + 'rope.cu',
          kernel: 'rope_norm (line 44), consecutive pairs; rope_neox (126), pairs half a head apart', lines: [44, 126], readAt: 'pinned',
          also: [CUDA_SWITCH(3048)], facts: ['A1', 'B6'], tags: ['own kernel'],
        };
        return {
          cuda,
          hip: llamaHip(cuda),
          metal: {
            entry: 'case GGML_OP_ROPE in ggml_metal_op_encode_impl', host: 'ggml_metal_op_rope', file: MT + 'kernels/rope.metal',
            kernel: 'kernel_rope_norm (line 48), kernel_rope_neox (106)', lines: [48, 106], readAt: 'pinned', also: [METAL_SWITCH()],
            facts: ['A1', 'B6'], tags: ['own kernel', 'found by name string'],
          },
        };
      })(),
      mlx: {
        cuda: {
          entry: 'mx.fast.rope(a, dims, *, traditional, base, scale, offset)', host: 'RoPE::eval_gpu, cuda/rope.cu line 249', file: MLX_CU + 'rope.cu',
          kernel: 'the pairing a template parameter: template <typename T, bool traditional, bool forward>', lines: 15, readAt: 'pinned',
          also: [{ label: 'fast.cpp lines 211 to 245', path: 'python/src/fast.cpp', lines: [211, 245] }], facts: ['A2', 'B6', 'C1'], tags: ['own kernel'],
        },
        hip: MLX_NO_HIP,
        metal: {
          entry: 'mx.fast.rope(a, dims, *, traditional, base, scale, offset)', host: 'RoPE::eval_gpu, metal/rope.cpp', file: MLX_MT + 'kernels/rope.metal',
          kernel: 'the pairing a function constant: constant bool traditional [[function_constant(2)]]', lines: 8, readAt: 'pinned',
          also: [{ label: 'fast.cpp lines 211 to 245', path: 'python/src/fast.cpp', lines: [211, 245] }], facts: ['A2', 'B6', 'C1'], tags: ['own kernel'],
        },
      },
      pytorch: (() => {
        const none: Cell = {
          entry: 'none: “rotary” does not occur in native_functions.yaml', host: 'Hugging Face’s rotate_half, run as ordinary tensor ops', file: null, kernel: null,
          readAt: 'pinned', also: [{ label: 'transformers modeling_llama.py lines 73 to 123', path: 'src/transformers/models/llama/modeling_llama.py', lines: [73, 123], project: 'transformers' }],
          facts: ['B6'], tags: ['composite of other ops'],
        };
        return { cuda: none, hip: { ...none, facts: ['B6', 'B9'] }, metal: none };
      })(),
    },
  },
  {
    id: 'attention', label: 'attention', tour: 'attention', section: 'X3.5',
    course: [lab('20.2', 'ch20', 'ch20/lab2_flash', 'flash attention'), lab('20.4', 'ch20', 'ch20/lab4_decode', 'decode with a KV cache'), lab('20.5', 'ch20', 'ch20/lab5_gqa_decode', 'GQA decode')],
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          entry: 'case GGML_OP_FLASH_ATTN_EXT in ggml_cuda_compute_forward', host: 'ggml_cuda_flash_attn_ext', file: CU + 'fattn.cu',
          kernel: 'its own: the vec, tile and mma kernels (fattn-vec.cuh, fattn-tile, fattn-mma-f16.cuh)', readAt: 'pinned', also: [CUDA_SWITCH(3111)],
          facts: ['A1', 'ch20 D1', 'infra D1'], tags: ['own kernel'],
        };
        return {
          cuda,
          hip: llamaHip(cuda),
          metal: {
            entry: 'case GGML_OP_FLASH_ATTN_EXT in ggml_metal_op_encode_impl', host: 'ggml_metal_op_flash_attn_ext', file: MT + 'kernels/fa_common.metal',
            kernel: 'its own: one file per K/V type (fa_f32.metal, fa_f16.metal, ...), vec variants, fa_aux.metal', readAt: 'pinned', also: [METAL_SWITCH(487)],
            facts: ['A1', 'ch20 D2', 'infra D2'], tags: ['own kernel', 'found by name string'],
          },
        };
      })(),
      mlx: {
        cuda: {
          entry: 'mx.fast.scaled_dot_product_attention', host: 'ScaledDotProductAttention::eval_gpu, line 623: sdpa_cudnn if supported, else sdpa_vector',
          file: MLX_CU + 'scaled_dot_product_attention.cu', kernel: 'its own for short decodes: kernel_sdpav_1pass (line 42), kernel_sdpav_2pass_1 (195); cuDNN SDPA for prefill and long decodes',
          lines: [42, 195], readAt: 'pinned', also: [{ label: 'scaled_dot_product_attention.cpp lines 623 to 660', path: MLX_CU + 'scaled_dot_product_attention.cpp', lines: [623, 660] }],
          facts: ['B5'], tags: ['own kernel', 'vendor library'],
        },
        hip: MLX_NO_HIP,
        metal: {
          entry: 'mx.fast.scaled_dot_product_attention', host: 'ScaledDotProductAttention::eval_gpu, metal/scaled_dot_product_attention.cpp', file: MLX_MT + 'kernels/sdpa_vector.h',
          kernel: 'its own: vector kernels for 8 or fewer queries, steel attention otherwise', readAt: 'master',
          url: 'https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h', facts: ['ch20 D4'], tags: ['own kernel'],
        },
      },
      pytorch: {
        cuda: {
          entry: '- func: _scaled_dot_product_flash_attention, dispatch CUDA: _scaled_dot_product_flash_attention_cuda', host: 'select_sdp_backend (sdp_utils.cpp line 1337): flash, efficient, math, cuDNN',
          file: TN + 'transformers/cuda/flash_attn/flash_api.cpp', kernel: 'run_mha_fwd_ and run_mha_fwd_splitkv_dispatch, from the third_party/flash-attention submodule', lines: [294, 301],
          readAt: 'pinned',
          also: [{ label: 'native_functions.yaml lines 15280 to 15282', path: YAML, lines: [15280, 15282] }, { label: 'sdp_utils.cpp lines 1337 to 1350', path: TN + 'transformers/cuda/sdp_utils.cpp', lines: [1337, 1350] }, { label: '.gitmodules lines 115 to 117', path: '.gitmodules', lines: [115, 117] }],
          facts: ['B5'], tags: ['third-party submodule', 'vendor library'],
        },
        hip: {
          entry: 'the same dispatch, hipified', host: 'select_sdp_backend', file: TN + 'transformers/hip',
          kernel: 'AOTriton or Composable Kernel (transformers/hip: aotriton_adapter.h, flash_attn/{aot,ck})', readAt: 'master',
          url: 'https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/transformers/hip', facts: ['B9', 'ch20 B5'], tags: ['vendor library', 'hipified at build time'],
        },
        metal: {
          entry: 'scaled_dot_product_attention on MPS: Attention.mm', host: 'sdpa_vector_2pass_mps, sdpa_vector_fast_mps or sdpa_prefill_mps', file: TN + 'mps/kernels/Attention.metal',
          kernel: 'its own Metal kernels, “largely influenced” by MLX’s', readAt: 'master', url: 'https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/Attention.metal',
          facts: ['B5', 'ch20 D5'], tags: ['own kernel'],
        },
      },
    },
  },
  {
    id: 'sampling', label: 'argmax and sampling', tour: 'sampling', section: 'X3.6',
    course: [lab('10.3', 'ch10', 'ch10/lab3_softmax_argmax', 'launchArgmaxRows'), lab('X1.3', 'x1', 'x1/lab3_sampler', 'M5, the GPU sampler')],
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          entry: 'the host chain by default (src/llama-sampler.cpp); the experimental backend greedy sampler builds ggml_argmax', host: 'llama_sampler_dist_apply on the host',
          file: CU + 'argmax.cu', kernel: 'argmax_f32: ties go to whichever lane held the index', readAt: 'master', url: 'https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/argmax.cu',
          facts: ['B8', 'ch10 D2', 'x1 D3'], tags: ['own kernel'],
        };
        return {
          cuda,
          hip: { ...cuda, facts: [...cuda.facts, 'B9'], tags: ['same source as CUDA'], note: LLAMA_HIP_NOTE },
          metal: {
            entry: 'the experimental backend greedy sampler builds ggml_argmax', host: null, file: MT + 'kernels/misc.metal',
            kernel: 'kernel_argmax_f32: the largest index among lanes holding the maximum', readAt: 'master', url: 'https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/misc.metal',
            facts: ['ch10 D2', 'x1 D3'], tags: ['own kernel'],
          },
        };
      })(),
      mlx: {
        cuda: {
          entry: 'mlx-lm sample_utils.py: mx.random.categorical or mx.argmax', host: null, file: MLX_CU + 'random.cu',
          kernel: 'threefry2x32_hash (line 27), rbitsc (47), rbits (87); the rest is ordinary MLX ops', lines: [27, 87], readAt: 'pinned',
          facts: ['B8', 'x1 D7', 'x1 D8'], tags: ['own kernel', 'composite of other ops'],
        },
        hip: MLX_NO_HIP,
        metal: {
          entry: 'mlx-lm sample_utils.py: mx.random.categorical or mx.argmax', host: null, file: MLX_MT + 'kernels/random.metal',
          kernel: 'threefry2x32_hash, 20 rounds; the rest is ordinary MLX ops', readAt: 'master', url: 'https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/random.metal',
          facts: ['B8', 'x1 D7', 'x1 D8'], tags: ['own kernel', 'composite of other ops'],
        },
      },
      pytorch: (() => {
        const cuda: Cell = {
          entry: '- func: multinomial, dispatch CPU, CUDA, MPS, XPU: multinomial', host: 'multinomial_out: one sample is argmax(p / q) with q exponential',
          file: TN + 'cuda/MultinomialKernel.cu', kernel: 'the inverse-CDF search, used only with replacement and several samples', readAt: 'master',
          url: 'https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MultinomialKernel.cu', facts: ['x1 D5', 'B8'], tags: ['composite of other ops', 'own kernel'],
        };
        return {
          cuda,
          hip: torchHip(cuda),
          metal: {
            ...cuda, file: TN + 'mps/operations/Distributions.mm', kernel: 'with replacement: cumsum, then searchsorted', url: 'https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Distributions.mm',
            tags: ['composite of other ops'],
          },
        };
      })(),
    },
  },
];

export const OP_IDS = OPS.map((o) => o.id);
export function op(id: OpId): OpRow {
  const o = OPS.find((x) => x.id === id);
  if (!o) throw new Error('no op ' + id);
  return o;
}
export const cell = (o: OpId, p: Project, b: Backend) => op(o).cells[p][b];

/** The facts file an item lives in: "B1" → x3-facts.md, "ch10 D2" → ch10-facts.md. */
export function factsFile(ref: string): { file: string; item: string } {
  const m = /^(\w+) ([A-Z]\d+)$/.exec(ref);
  return m ? { file: `notes/today/${m[1]}-facts.md`, item: m[2] } : { file: 'notes/today/x3-facts.md', item: ref };
}

/** The link for a cell's kernel file: a pinned permalink, the master URL of its facts item, or none. */
export function cellLink(project: Project, c: Cell): string | null {
  if (c.readAt === 'master') return c.url ?? null;
  if (!c.file) return null;
  return permalink(project, c.file, c.lines ?? null, !/\.\w+$/.test(c.file));
}
export const whereLink = (project: Project, w: Where) => permalink(w.project ?? project, w.path, w.lines ?? null, !!w.tree);

// ---------------------------------------------------------------- the three entry points
export interface EntryPoint {
  project: Project;
  what: string;
  cuda: string;
  hip: string;
  metal: string;
  rule: string;
  facts: string[];
  where: Where[];
}
export const ENTRY_POINTS: EntryPoint[] = [
  {
    project: 'llama.cpp', what: 'an op is a GGML_OP_* value in a graph',
    cuda: 'ggml_cuda_compute_forward: one switch on dst->op (after the fusion pass)', hip: 'the same switch: ggml-cuda/*.cu compiled as HIP',
    metal: 'ggml_metal_op_encode_impl: its own switch; each case returns n_fuse and asks for a pipeline by name',
    rule: 'Find the case for the op, open the .cu named after it. Check the fusion pass first. On Metal, follow the formatted name string.',
    facts: ['A1', 'B9', 'C2'],
    where: [{ label: 'ggml-cuda.cu line 2812', path: CU + 'ggml-cuda.cu', lines: 2812 }, { label: 'ggml-metal-ops.cpp line 187', path: MT + 'ggml-metal-ops.cpp', lines: 187 }],
  },
  {
    project: 'mlx', what: 'an op is a Primitive subclass',
    cuda: 'Primitive::eval_gpu in mlx/backend/cuda', hip: 'none: MLX has no AMD backend',
    metal: 'Primitive::eval_gpu in mlx/backend/metal; anything under mx.compile is source written at run time',
    rule: 'Search for Name::eval_gpu in the backend’s directory; elementwise ops come from a macro; compiled functions have no file.',
    facts: ['A2', 'B9'],
    where: [{ label: 'mlx/backend', path: 'mlx/backend', tree: true }],
  },
  {
    project: 'pytorch', what: 'an op is a - func: entry in native_functions.yaml',
    cuda: 'the CUDA: dispatch key; or structured_delegate, ufunc_inner_loop, or a _stub filled by REGISTER_DISPATCH', hip: 'the same CUDA sources, hipified at build time',
    metal: 'the MPS: dispatch key, an operations/*.mm file that names a kernel in kernels/*.metal by a string',
    rule: 'Find the - func: line, read dispatch:; with no backend key, follow structured_delegate, ufunc_inner_loop or the stub to REGISTER_DISPATCH.',
    facts: ['A3', 'B9', 'C2', 'ch24 D3'],
    where: [{ label: 'native_functions.yaml', path: YAML }],
  },
];

/** Three rows of x3-facts "Where the backends really differ", verbatim. */
export const DIFFER_ROWS: { feature: string; cuda: string; hip: string; metal: string }[] = [
  {
    feature: 'Op to kernel map',
    cuda: 'llama.cpp `ggml_cuda_compute_forward`; MLX `eval_gpu` in `backend/cuda`; PyTorch `CUDA:` dispatch key',
    hip: 'Same as CUDA',
    metal: 'llama.cpp `ggml_metal_op_encode_impl`; MLX `eval_gpu` in `backend/metal`; PyTorch `MPS:` key',
  },
  {
    feature: 'Kernel variants', cuda: 'Template parameters', hip: 'Same templates',
    metal: 'Function constants (`[[function_constant]]`, Metal 1.2+) and `[[host_name]]` instantiations',
  },
  { feature: 'Finding a kernel from the host', cuda: 'Function symbol', hip: 'Same', metal: 'The name string passed to the pipeline lookup' },
];

// ---------------------------------------------------------------- "Follow the call" routes
export type SearchKind = 'function name' | 'yaml key' | 'switch case' | 'formatted string' | 'REGISTER_DISPATCH' | 'macro' | 'file name' | 'environment variable';
export interface HopOption {
  label: string;
  kind?: SearchKind;
  correct?: boolean;
  why?: string;
}
export interface Hop {
  prompt: string;
  options: HopOption[];
  /** What you find when you follow the right option. */
  found: string;
  project: ProjectId;
  path: string;
  lines?: Lines;
  facts: string[];
}
export interface Route {
  id: string;
  label: string;
  project: Project;
  backend: Backend;
  op: OpId;
  start: string;
  hops: Hop[];
  /** What this route teaches, shown at the end. */
  lesson: string;
}

export const ROUTES: Route[] = [
  {
    id: 'torch-rmsnorm-cuda', label: 'PyTorch rms_norm on CUDA', project: 'pytorch', backend: 'cuda', op: 'rmsnorm',
    start: 'torch.nn.functional.rms_norm(x, (d,), weight) on CUDA tensors of the same dtype.',
    hops: [
      {
        prompt: 'The Python function calls into C++. What do you look for in aten/src/ATen/native?',
        options: [
          { label: 'The C++ function it binds to, rms_norm_symint', kind: 'function name', correct: true },
          { label: 'A switch on an op enum', kind: 'switch case', why: 'That is llama.cpp’s shape. PyTorch has no op switch to search.' },
          { label: 'A kernel-name string', kind: 'formatted string', why: 'Name strings are how Metal pipelines are found, not how a Python call reaches C++.' },
        ],
        found: 'rms_norm_symint in layer_norm.cpp calls at::_fused_rms_norm(...) (line 372). It falls back to rms_norm_composite for complex input, and warns “Cannot dispatch to fused implementation.” when weight and input dtypes differ.',
        project: 'pytorch', path: TN + 'layer_norm.cpp', lines: [329, 372], facts: ['B2'],
      },
      {
        prompt: 'at::_fused_rms_norm is an operator. Where is the map from it to a CUDA function?',
        options: [
          { label: 'Its - func: _fused_rms_norm entry in native_functions.yaml', kind: 'yaml key', correct: true },
          { label: 'A REGISTER_DISPATCH for a _fused_rms_norm_stub', kind: 'REGISTER_DISPATCH', why: 'Some ops do go through a stub, but this one names its backends directly in the yaml.' },
          { label: 'A file called _fused_rms_norm.cu', kind: 'file name', why: 'PyTorch does not name files after operators.' },
        ],
        found: 'dispatch: CUDA: _fused_rms_norm_cuda, MPS: _fused_rms_norm_mps, CompositeImplicitAutograd: rms_norm_composite (lines 3304 to 3309).',
        project: 'pytorch', path: YAML, lines: [3304, 3309], facts: ['A3'],
      },
      {
        prompt: 'The CUDA key names _fused_rms_norm_cuda. What do you search for now?',
        options: [
          { label: 'That symbol, _fused_rms_norm_cuda, in native/cuda', kind: 'function name', correct: true },
          { label: 'A generated UfuncCUDA_ file in the build tree', kind: 'file name', why: 'Only ufunc ops (add and friends) are generated. This one has a CUDA key and a hand-written function.' },
          { label: 'The MPS operations file', kind: 'file name', why: 'That is the MPS: key, another backend.' },
        ],
        found: '_fused_rms_norm_cuda is in native/cuda/layer_norm_kernel.cu (line 2009).',
        project: 'pytorch', path: TN + 'cuda/layer_norm_kernel.cu', lines: 2009, facts: ['B2'],
      },
      {
        prompt: 'Predict: what does the kernel it launches look like?',
        options: [
          { label: 'LayerNorm’s kernels, with RMSNorm selected by a template flag', correct: true },
          { label: 'A separate RMSNorm kernel file', why: 'There is none: the two norms share kernels.' },
          { label: 'A cuDNN normalisation call', why: 'No library is involved here; these are PyTorch’s own kernels.' },
        ],
        found: 'template <typename T, typename T_ACC, bool rms_norm> __global__ void RowwiseMomentsCUDAKernel( (line 57) and vectorized_layer_norm_kernel_impl (line 270): one source, two norms.',
        project: 'pytorch', path: TN + 'cuda/layer_norm_kernel.cu', lines: [57, 270], facts: ['B2'],
      },
    ],
    lesson: 'One name at a time: Python function, C++ function, yaml key, dispatch target, kernel. The yaml is the map.',
  },
  {
    id: 'llama-rmsnorm-fused', label: 'llama.cpp RMS_NORM on CUDA, fusion on', project: 'llama.cpp', backend: 'cuda', op: 'rmsnorm',
    start: 'A Llama layer’s graph: an RMS_NORM node, then a MUL by the norm weight, on the CUDA backend with default settings.',
    hops: [
      {
        prompt: 'Before ggml_cuda_compute_forward’s switch sees the RMS_NORM node, what looks at it?',
        options: [
          { label: 'A fusion pass over runs of nodes', kind: 'function name', correct: true },
          { label: 'Nothing: the switch is first', kind: 'switch case', why: 'That is true only with fusion disabled.' },
          { label: 'ggml_backend_cuda_device_supports_op, which runs it', kind: 'function name', why: 'supports_op (line 6336) only says whether the backend accepts the op. It runs nothing.' },
        ],
        found: 'if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) { ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]); return 1; } (lines 5268 to 5271). RMS_NORM, MUL, ADD is tried one step earlier.',
        project: 'llama.cpp', path: CU + 'ggml-cuda.cu', lines: [5255, 5272], facts: ['A1'],
      },
      {
        prompt: 'Where is ggml_cuda_op_rms_norm_fused?',
        options: [
          { label: 'In norm.cu, the .cu file named after the op family', kind: 'file name', correct: true },
          { label: 'In a fusion.cu with every fused kernel', kind: 'file name', why: 'llama.cpp keeps a fused variant beside the plain one.' },
          { label: 'Nowhere: it is generated at build time', why: 'llama.cpp generates template instances, not this function.' },
        ],
        found: 'norm.cu, rms_norm_f32 (lines 127 to 272): one block per row, a sum of squares, block_reduce, rsqrtf(mean + eps), and the weight multiply when it is fused.',
        project: 'llama.cpp', path: CU + 'norm.cu', lines: [127, 272], facts: ['B2'],
      },
      {
        prompt: 'A breakpoint in plain ggml_cuda_op_rms_norm never hits. How do you confirm why?',
        options: [
          { label: 'Rerun with GGML_CUDA_DISABLE_FUSION set', kind: 'environment variable', correct: true },
          { label: 'Rebuild for HIP', why: 'The HIP build compiles the same files and runs the same fusion pass.' },
          { label: 'Remove the MUL from the model file', why: 'That changes the model, not the dispatch.' },
        ],
        found: 'ggml_cuda_env_flag("GGML_CUDA_DISABLE_FUSION") is read at line 181. With it set, the switch’s case GGML_OP_RMS_NORM (line 2997) runs ggml_cuda_op_rms_norm.',
        project: 'llama.cpp', path: CU + 'ggml-cuda.cu', lines: [176, 183], facts: ['A1'],
      },
      {
        prompt: 'And on an AMD GPU?',
        options: [
          { label: 'The same norm.cu, compiled as HIP', correct: true },
          { label: 'ggml-hip/norm.hip, a separate port', kind: 'file name', why: 'ggml-hip holds build files only.' },
          { label: 'A copy hipified at build time', why: 'That is PyTorch’s approach. llama.cpp renames with #defines in vendors/hip.h.' },
        ],
        found: 'ggml-hip/CMakeLists.txt globs ../ggml-cuda/*.cu and sets LANGUAGE HIP (lines 60 to 70).',
        project: 'llama.cpp', path: 'ggml/src/ggml-hip/CMakeLists.txt', lines: [60, 70], facts: ['B9'],
      },
    ],
    lesson: 'The fusion pass runs before the switch, so the host function you expect may never run.',
  },
  {
    id: 'llama-rmsnorm-plain', label: 'llama.cpp RMS_NORM on CUDA, fusion off', project: 'llama.cpp', backend: 'cuda', op: 'rmsnorm',
    start: 'The same graph, run with GGML_CUDA_DISABLE_FUSION set.',
    hops: [
      {
        prompt: 'With fusion off, what do you search ggml-cuda.cu for?',
        options: [
          { label: 'case GGML_OP_RMS_NORM in ggml_cuda_compute_forward', kind: 'switch case', correct: true },
          { label: 'ggml_cuda_op_rms_norm_fused', kind: 'function name', why: 'That is the fused path, which is now off.' },
          { label: 'A kernel-name string', kind: 'formatted string', why: 'CUDA host code calls the kernel by its symbol.' },
        ],
        found: 'case GGML_OP_RMS_NORM: ggml_cuda_op_rms_norm(ctx, dst); (line 2997), inside switch (dst->op) of ggml_cuda_compute_forward (line 2812).',
        project: 'llama.cpp', path: CU + 'ggml-cuda.cu', lines: 2997, facts: ['A1'],
      },
      {
        prompt: 'Where is ggml_cuda_op_rms_norm?',
        options: [
          { label: 'norm.cu', kind: 'file name', correct: true },
          { label: 'rmsnorm.cu', kind: 'file name', why: 'The file is named for the op family: norm.cu holds every norm.' },
          { label: 'ggml-cuda.cu itself', why: 'ggml-cuda.cu holds the switch; each host function lives in its op’s file.' },
        ],
        found: 'norm.cu, the same rms_norm_f32 kernel (lines 127 to 272), now without the fused multiply.',
        project: 'llama.cpp', path: CU + 'norm.cu', lines: [127, 272], facts: ['A1', 'B2'],
      },
      {
        prompt: 'What runs the MUL by the weight now?',
        options: [
          { label: 'A second launch, from the switch’s own case for MUL: a binary op', kind: 'switch case', correct: true },
          { label: 'Nothing: RMS_NORM always multiplies', why: 'Only the fused call multiplies; the plain one does not.' },
          { label: 'The CPU backend', why: 'The CUDA backend accepts MUL; nothing falls back here.' },
        ],
        found: 'Binary ops go to k_bin_bcast in binbcast.cu (ch03 D1). Two launches and a round trip through memory where the fused path had one.',
        project: 'llama.cpp', path: CU + 'binbcast.cu', facts: ['A1', 'B1', 'ch03 D1'],
      },
    ],
    lesson: 'With fusion off, the one-name rule works as is: switch case, host function, the .cu file of the same name.',
  },
  {
    id: 'llama-getrows-metal', label: 'llama.cpp GET_ROWS on Metal', project: 'llama.cpp', backend: 'metal', op: 'embedding',
    start: 'The embedding lookup of a llama.cpp graph, a GGML_OP_GET_ROWS node with F16 rows, on the Metal backend.',
    hops: [
      {
        prompt: 'Where does the Metal backend turn the node into work?',
        options: [
          { label: 'A case in ggml_metal_op_encode_impl’s switch', kind: 'switch case', correct: true },
          { label: 'The CUDA switch, compiled for Metal', why: 'Metal has its own host code and its own kernels.' },
          { label: 'A Metal function named ggml_get_rows', kind: 'function name', why: 'Metal kernels are found by a name string the host builds, and the kernels are called kernel_*.' },
        ],
        found: 'static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) (line 187): GGML_OP_GET_ROWS goes to ggml_metal_op_get_rows. Each case returns n_fuse, the number of nodes it consumed.',
        project: 'llama.cpp', path: MT + 'ggml-metal-ops.cpp', lines: [187, 490], facts: ['A1'],
      },
      {
        prompt: 'ggml_metal_op_get_rows needs a pipeline. How does it ask for one?',
        options: [
          { label: 'By a kernel name it formats as a string', kind: 'formatted string', correct: true },
          { label: 'By a C++ function pointer to the kernel', kind: 'function name', why: 'The host cannot call an MSL function by symbol; it asks the library for a function by name.' },
          { label: 'By including the .metal file', kind: 'file name', why: 'The kernels are compiled into a library; the host looks functions up in it by name.' },
        ],
        found: 'snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc)); (ggml-metal-device.cpp line 154).',
        project: 'llama.cpp', path: MT + 'ggml-metal-device.cpp', lines: 154, facts: ['A1'],
      },
      {
        prompt: 'What do you search the .metal files for?',
        options: [
          { label: 'The string the host builds: kernel_get_rows_ followed by the type’s name', kind: 'formatted string', correct: true },
          { label: 'ggml_metal_op_get_rows', kind: 'function name', why: 'That is the host function you came from.' },
          { label: 'The template’s C++ name only', kind: 'function name', why: 'One template serves every type; the string tells you which instantiation runs.' },
        ],
        found: 'kernels/quantize.metal holds kernel_get_rows_f (line 266) for float rows and kernel_get_rows_q (line 297) for quantised rows; the string matches the [[host_name(...)]] of an instantiation. The exact instantiation string for F16 was not read in the facts pass.',
        project: 'llama.cpp', path: MT + 'kernels/quantize.metal', lines: [266, 297], facts: ['A1', 'B7', 'C2'],
      },
    ],
    lesson: 'On Metal, the name is a string: search the .metal files for what the host formats.',
  },
  {
    id: 'mlx-softmax-cuda', label: 'MLX softmax on CUDA', project: 'mlx', backend: 'cuda', op: 'softmax',
    start: 'MLX’s Softmax primitive on an NVIDIA GPU. (How the Python softmax builds the primitive was not read in the facts pass.)',
    hops: [
      {
        prompt: 'MLX ops are Primitive subclasses. What do you search mlx/backend/cuda for?',
        options: [
          { label: 'Softmax::eval_gpu', kind: 'function name', correct: true },
          { label: 'A dispatch key in a yaml file', kind: 'yaml key', why: 'That is PyTorch’s map.' },
          { label: 'case SOFTMAX in a switch', kind: 'switch case', why: 'That is llama.cpp’s shape.' },
        ],
        found: 'void Softmax::eval_gpu( in mlx/backend/cuda/softmax.cu (line 108).',
        project: 'mlx', path: MLX_CU + 'softmax.cu', lines: 108, facts: ['A2'],
      },
      {
        prompt: 'Predict: how does that kernel reduce a row?',
        options: [
          { label: 'In one pass, rescaling a running sum whenever the max grows', correct: true },
          { label: 'Max, then sum, as lab 10.3 and MLX’s Metal kernel do', why: 'That is the Metal side. The two backends of one project differ here.' },
          { label: 'Through cuDNN’s softmax', why: 'MLX uses cuDNN for attention, not for this op.' },
        ],
        found: 'normalizer = normalizer * softmax_exp(prevmax - maxval); then add softmax_exp(vals[i] - maxval) (lines 49 to 62): the online normaliser of chapter 20, with __expf and cooperative groups.',
        project: 'mlx', path: MLX_CU + 'softmax.cu', lines: [20, 106], facts: ['B3'],
      },
      {
        prompt: 'Is there an AMD version of this file?',
        options: [
          { label: 'No: MLX has no AMD backend', correct: true },
          { label: 'Yes, the same file hipified at build time', why: 'That is PyTorch. MLX’s backend directory has no HIP or ROCm entry.' },
          { label: 'Yes, under mlx/backend/rocm', kind: 'file name', why: 'There is no such directory at the pinned commit.' },
        ],
        found: 'mlx/backend lists common, cpu, cuda, gpu, metal, no_cpu, no_gpu. An experimental ROCm pull request (#2300) is open and unmerged; the CUDA backend also uses cooperative groups, cuBLASLt and cuDNN’s frontend.',
        project: 'mlx', path: 'mlx/backend', facts: ['B9'],
      },
    ],
    lesson: 'In MLX, search for Name::eval_gpu in the backend directory. Do not assume its two backends compute alike.',
  },
  {
    id: 'torch-add-cuda', label: 'PyTorch add on CUDA', project: 'pytorch', backend: 'cuda', op: 'add',
    start: 'add.out, the out= form of torch.add, on CUDA tensors.',
    hops: [
      {
        prompt: 'Where do you start?',
        options: [
          { label: 'Its - func: add.out entry in native_functions.yaml', kind: 'yaml key', correct: true },
          { label: 'A file Add.cu in native/cuda', kind: 'file name', why: 'Try it: there is none.' },
          { label: 'REGISTER_DISPATCH(add_stub, ...) in native/cuda', kind: 'REGISTER_DISPATCH', why: 'Close, but the registration is in a file that does not exist in the source tree.' },
        ],
        found: 'structured: True, ufunc_inner_loop: Generic: add (AllAndComplex, BFloat16, Half, ComplexHalf), and no CUDA: key (lines 565 to 571).',
        project: 'pytorch', path: YAML, lines: [565, 571], facts: ['A3'],
      },
      {
        prompt: 'ufunc_inner_loop names add. Where is the scalar body?',
        options: [
          { label: 'native/ufunc/add.h', kind: 'file name', correct: true },
          { label: 'native/cuda/BinaryAddKernel.cu', kind: 'file name', why: 'A guess at a name; the ufunc body is device-neutral and lives in native/ufunc.' },
          { label: 'cuBLAS’s axpy', why: 'No library: the loop is PyTorch’s own.' },
        ],
        found: 'return self + alpha * other; in aten/src/ATen/native/ufunc/add.h.',
        project: 'pytorch', path: TN + 'ufunc/add.h', facts: ['A3'],
      },
      {
        prompt: 'So where is the CUDA kernel?',
        options: [
          { label: 'In UfuncCUDA_add.cu, which torchgen writes at build time', correct: true },
          { label: 'In native/cuda under another name', why: 'Search all you like: it is not in the source tree.' },
          { label: 'Nowhere: add runs on the CPU', why: 'It runs on the GPU, from a file that exists only after a build.' },
        ],
        found: 'torchgen/gen.py writes UfuncCUDA_{name}.cu (line 2536); torchgen/dest/ufunc.py emits gpu_kernel(iter, ufunctor); and REGISTER_DISPATCH(...) (lines 279, 288, 328).',
        project: 'pytorch', path: 'torchgen/dest/ufunc.py', lines: [279, 328], facts: ['A3'],
      },
      {
        prompt: 'And on MPS?',
        options: [
          { label: 'A hand-written add_out_mps, because ufunc codegen emits only CPU and CUDA', correct: true },
          { label: 'The same generated file, compiled for Metal', why: 'Codegen does not emit an MPS version, and the MPS source says so.' },
          { label: 'MPSGraph only', why: 'PyTorch MPS has its own Metal kernel for add.' },
        ],
        found: '“add.out is a ufunc op, and ufunc codegen only emits CPU/CUDA impls, so MPS needs this entry” (BinaryOps.mm lines 16 to 19), then lib.exec_binary_kernel(iter, "add") and add_alpha_functor in kernels/BinaryKernel.metal (lines 22 to 25).',
        project: 'pytorch', path: TN + 'mps/operations/BinaryOps.mm', lines: [16, 19], facts: ['A3'],
      },
    ],
    lesson: 'Some PyTorch CUDA kernels exist in no source file. When the yaml has no CUDA key, follow ufunc_inner_loop or the stub.',
  },
];
export function route(id: string): Route {
  const r = ROUTES.find((x) => x.id === id);
  if (!r) throw new Error('no route ' + id);
  return r;
}
/** The five routes the stepper offers; RMS_NORM's two are one route with a fusion switch. */
export const ROUTE_CHOICES: { id: string; label: string; variants?: { on: string; off: string } }[] = [
  { id: 'torch-rmsnorm-cuda', label: 'PyTorch rms_norm, CUDA' },
  { id: 'llama-rmsnorm', label: 'llama.cpp RMS_NORM, CUDA', variants: { on: 'llama-rmsnorm-fused', off: 'llama-rmsnorm-plain' } },
  { id: 'llama-getrows-metal', label: 'llama.cpp GET_ROWS, Metal' },
  { id: 'mlx-softmax-cuda', label: 'MLX softmax, CUDA' },
  { id: 'torch-add-cuda', label: 'PyTorch add, CUDA' },
];

/** Score a walk: for each hop answered, whether the first pick was right. */
export function scoreWalk(r: Route, firstPicks: (number | null)[]): { answered: number; right: number } {
  let answered = 0;
  let right = 0;
  r.hops.forEach((h, i) => {
    const p = firstPicks[i];
    if (p == null) return;
    answered++;
    if (h.options[p]?.correct) right++;
  });
  return { answered, right };
}

// ---------------------------------------------------------------- variant mechanisms (X3.2)
export type Mechanism = 'compile-time' | 'pipeline-constant' | 'name-string' | 'registration' | 'build-time-codegen' | 'run-time-codegen';
export const MECHANISMS: { id: Mechanism; label: string }[] = [
  { id: 'compile-time', label: 'compile time: a template or a macro' },
  { id: 'pipeline-constant', label: 'fixed when the pipeline is created' },
  { id: 'name-string', label: 'looked up by a name string' },
  { id: 'registration', label: 'a registration table' },
  { id: 'build-time-codegen', label: 'generated at build time' },
  { id: 'run-time-codegen', label: 'generated at run time' },
];
export interface VariantItem {
  id: string;
  code: string;
  project: Project;
  backend: Backend;
  mechanism: Mechanism;
  why: string;
  facts: string;
}
export const VARIANT_MECHANISMS: VariantItem[] = [
  {
    id: 'llama-template', code: 'template <float (*op)(float), typename T>', project: 'llama.cpp', backend: 'cuda', mechanism: 'compile-time',
    why: 'Each op is a separate instantiation of unary_op_kernel, compiled into the binary. HIP compiles the same template.', facts: 'B1',
  },
  {
    id: 'llama-fc', code: 'constant short FC_unary_op [[function_constant(FC_UNARY + 0)]];', project: 'llama.cpp', backend: 'metal', mechanism: 'pipeline-constant',
    why: 'Its value is supplied when the specialised function and pipeline are created; the source is compiled once.', facts: 'B1, C1',
  },
  {
    id: 'llama-hostname', code: 'template [[host_name("kernel_rms_norm_mul_f32")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 2>;',
    project: 'llama.cpp', backend: 'metal', mechanism: 'name-string',
    why: 'The instantiation is compiled ahead, and the host finds it by the string in [[host_name]].', facts: 'C2',
  },
  {
    id: 'llama-snprintf', code: 'snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));', project: 'llama.cpp', backend: 'metal', mechanism: 'name-string',
    why: 'The host side of the same mechanism: it builds the string it will look up.', facts: 'A1',
  },
  {
    id: 'mlx-binary-gpu', code: 'BINARY_GPU(Add)', project: 'mlx', backend: 'cuda', mechanism: 'compile-time',
    why: 'A macro that stamps out Add::eval_gpu when MLX is compiled.', facts: 'A2',
  },
  {
    id: 'mlx-rope-template', code: 'template <typename T, bool traditional, bool forward>', project: 'mlx', backend: 'cuda', mechanism: 'compile-time',
    why: 'The RoPE pairing as a template parameter, on CUDA.', facts: 'B6, C1',
  },
  {
    id: 'mlx-rope-fc', code: 'constant bool traditional [[function_constant(2)]];', project: 'mlx', backend: 'metal', mechanism: 'pipeline-constant',
    why: 'The same choice on Metal, as a function constant.', facts: 'B6, C1',
  },
  {
    id: 'torch-register', code: 'REGISTER_DISPATCH(silu_stub, &silu_kernel)', project: 'pytorch', backend: 'cuda', mechanism: 'registration',
    why: 'The stub is a table entry per device; the CUDA file fills it when the library loads.', facts: 'A3',
  },
  {
    id: 'torch-ufunc', code: 'gpu_kernel(iter, ufunctor);', project: 'pytorch', backend: 'cuda', mechanism: 'build-time-codegen',
    why: 'torchgen writes this line into UfuncCUDA_add.cu while PyTorch builds.', facts: 'A3',
  },
  {
    id: 'torch-fmt', code: 'fmt::format("{}_{}", name, scalarToMetalTypeString(output))', project: 'pytorch', backend: 'metal', mechanism: 'name-string',
    why: 'PyTorch MPS builds a kernel name and asks for its pipeline with getPipelineStateForFunc.', facts: 'C2',
  },
  {
    id: 'mlx-compiled', code: '[[host_name(\\"{0}\\")]]\\n[[kernel]] void {0}(', project: 'mlx', backend: 'metal', mechanism: 'run-time-codegen',
    why: 'Compiled::eval_gpu writes this into kernel source the first time an mx.compile function runs, and compiles it.', facts: 'A2',
  },
];
export function gradeSort(picks: Record<string, Mechanism | undefined>): { right: number; total: number; wrong: string[] } {
  const wrong = VARIANT_MECHANISMS.filter((v) => picks[v.id] !== v.mechanism).map((v) => v.id);
  return { right: VARIANT_MECHANISMS.length - wrong.length, total: VARIANT_MECHANISMS.length, wrong };
}

// ---------------------------------------------------------------- the "One RMSNorm call" scene
export type GraphOp = 'GET_ROWS' | 'RMS_NORM' | 'MUL' | 'ADD' | 'MUL_MAT';
/** The CUDA fusion rules read in A1 that start with RMS_NORM, longest first, as the pass tries them. */
export const CUDA_FUSIONS: { ops: GraphOp[]; call: string }[] = [
  // A1 quotes this rule but not the name of the function it calls.
  { ops: ['RMS_NORM', 'MUL', 'ADD'], call: 'the fused RMS_NORM, MUL, ADD call' },
  { ops: ['RMS_NORM', 'MUL'], call: 'ggml_cuda_op_rms_norm_fused' },
];
export const PLAIN_CALL: Record<GraphOp, string> = {
  GET_ROWS: 'ggml_cuda_op_get_rows',
  RMS_NORM: 'ggml_cuda_op_rms_norm',
  MUL: 'binary op (binbcast.cu)',
  ADD: 'ggml_cuda_op_add',
  MUL_MAT: 'ggml_cuda_mul_mat',
};
export interface Dispatch {
  nodes: number[];
  call: string;
  fused: boolean;
}
/** What the CUDA backend calls for each run of nodes, with or without the fusion pass. */
export function fuseGraph(graph: GraphOp[], fusion: boolean): Dispatch[] {
  const out: Dispatch[] = [];
  for (let i = 0; i < graph.length; ) {
    const rule = fusion ? CUDA_FUSIONS.find((f) => f.ops.every((o, k) => graph[i + k] === o)) : undefined;
    if (rule) {
      out.push({ nodes: rule.ops.map((_, k) => i + k), call: rule.call, fused: true });
      i += rule.ops.length;
    } else {
      out.push({ nodes: [i], call: PLAIN_CALL[graph[i]], fused: false });
      i++;
    }
  }
  return out;
}
/** llama.cpp Metal's RMSNorm instantiations (norm.metal lines 178 to 185, float forms), by fusion level F. */
export const METAL_NORM_NAMES: { F: number; ops: GraphOp[]; name: string }[] = [
  { F: 1, ops: ['RMS_NORM'], name: 'kernel_rms_norm_f32' },
  { F: 2, ops: ['RMS_NORM', 'MUL'], name: 'kernel_rms_norm_mul_f32' },
  { F: 3, ops: ['RMS_NORM', 'MUL', 'ADD'], name: 'kernel_rms_norm_mul_add_f32' },
];
/** The Metal name for a run of nodes starting at an RMS_NORM: the longest matching fusion level. */
export function metalNormName(graph: GraphOp[], i: number): { F: number; consumed: number; name: string } {
  const m = [...METAL_NORM_NAMES].reverse().find((x) => x.ops.every((o, k) => graph[i + k] === o))!;
  return { F: m.F, consumed: m.ops.length, name: m.name };
}
export const ONE_CALL = {
  /** The part of a Llama layer the scene shows (ours: the order of a pre-norm attention block's start). */
  graph: ['GET_ROWS', 'RMS_NORM', 'MUL', 'MUL_MAT'] as GraphOp[],
  cudaKernel: { file: 'norm.cu', kernel: 'rms_norm_f32', lines: [127, 272] as [number, number] },
  metalSwitch: { fn: 'ggml_metal_op_encode_impl', line: 187, host: 'ggml_metal_op_norm', caseLine: 414 },
  metalTemplate: 'kernel_rms_norm_fuse_impl',
};

// ---------------------------------------------------------------- the start page's prediction
/** Where a project's CUDA attention kernels come from, read off the attention cells' tags. */
export type AttnHome = 'own' | 'mixed' | 'submodule';
export const ATTN_HOME_OPTIONS: { id: AttnHome; label: string }[] = [
  { id: 'own', label: 'its own source tree, for every shape' },
  { id: 'mixed', label: 'its own kernels for some shapes, a vendor library for others' },
  { id: 'submodule', label: 'first choice is another project’s kernels, pulled in as a git submodule' },
];
export function attentionHome(p: Project): AttnHome {
  const t = cell('attention', p, 'cuda').tags;
  if (t.includes('third-party submodule')) return 'submodule';
  if (t.includes('vendor library')) return 'mixed';
  return 'own';
}

/**
 * The order a hop's options are shown in: a permutation seeded by the route and hop, the same
 * on every visit, so the right option is not always where the data lists it (first).
 */
export function hopOrder(routeId: string, hop: number, n: number): number[] {
  let h = 2166136261;
  const key = `${routeId}#${hop}`;
  for (let i = 0; i < key.length; i++) h = Math.imul(h ^ key.charCodeAt(i), 16777619) >>> 0;
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
    const j = h % (i + 1);
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx;
}
