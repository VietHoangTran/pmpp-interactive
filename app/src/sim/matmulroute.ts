// Which kernel a matrix product runs, by project and shape (module X3, section X3.4). Not in
// the book. Every check below is one read in notes/today/x3-facts.md (B4), ch22-facts.md (D6)
// or ch15-facts.md (D1 to D4), at the commits listed there; the line numbers are the facts'.
//
// What it models: the decision chains in their source order, for a product
//   Y[M x N] = X[M x K] * W^T   (M tokens, W stored (N, K) as nn.Linear stores it).
// Each check is evaluated as true or false only when the condition the facts quote decides it;
// everything decided inside a function the facts did not read (ggml_cuda_should_use_mmf,
// ggml_cuda_should_use_mmq, the rest of should_use_mmvf and should_use_mmvq, the FP8 path,
// ggml_metal_op_mul_mat_use_mm, MLX's wide-GEMV conditions other than its limit of 5, cuBLAS's
// own algorithm choice) is returned as 'not-modelled'. The route is then the first check not
// ruled out, and it is certain only when that check is true. One reading is ours, not the
// facts': mmvf and mmf (the "_f" functions) serve float weights and mmvq and mmq (the "_q"
// ones) quantised weights, so a weight of the other class rules a candidate out.
//
// What it leaves out: steel and cuBLAS tile selection, data types other than the weight's
// class, strides and batching of 3-D operands, and every per-GPU exception (it names them).

import { gpuBackends, stepSeries, type ReportsLike } from './batching';

export type Project = 'llama-cuda' | 'llama-metal' | 'mlx-cuda' | 'mlx-metal' | 'torch-cuda' | 'torch-mps';
export const PROJECTS: Project[] = ['llama-cuda', 'llama-metal', 'mlx-cuda', 'mlx-metal', 'torch-cuda', 'torch-mps'];
export const PROJECT_LABEL: Record<Project, string> = {
  'llama-cuda': 'llama.cpp, CUDA and HIP',
  'llama-metal': 'llama.cpp, Metal',
  'mlx-cuda': 'MLX, CUDA',
  'mlx-metal': 'MLX, Metal',
  'torch-cuda': 'PyTorch, CUDA (and ROCm)',
  'torch-mps': 'PyTorch, MPS (addmm and mm)',
};

export type WeightType = 'f32' | 'f16' | 'quant' | 'kquant';
export const WEIGHT_LABEL: Record<WeightType, string> = {
  f32: 'F32',
  f16: 'F16',
  quant: 'quantised (not a k-quant)',
  kquant: 'quantised (k-quant)',
};
const isQuant = (w: WeightType) => w === 'quant' || w === 'kquant';

export interface Shape {
  /** Tokens: rows of X (llama.cpp's ne11). */
  M: number;
  /** Output features: rows of W. */
  N: number;
  /** Input features. */
  K: number;
  weight: WeightType;
  /** llama.cpp CUDA: is "the input" F32? (the facts' wording at ggml-cuda.cu 2547; we read it as the activations). */
  inputF32: boolean;
  /** MLX: b is a transposed view (x @ W.T with W stored (out, in)). */
  bTransposed: boolean;
  /** MLX: a is a transposed view. */
  aTransposed: boolean;
  /** MLX CUDA: a bias is fused (addmm); the GEMV branch requires none. */
  bias: boolean;
}

export const DEFAULT_SHAPE: Shape = { M: 1, N: 4096, K: 4096, weight: 'f16', inputF32: true, bTransposed: true, aTransposed: false, bias: false };

export type RouteId =
  | 'cublas' | 'mmvf' | 'mmf' | 'mmvq' | 'mmq' | 'fp8'
  | 'mul_mv' | 'mul_mv_ext' | 'mul_mm' | 'metal-unread'
  | 'mlx-gemv' | 'cublaslt'
  | 'wide' | 'gemv' | 'steel'
  | 'torch-blas'
  | 'mps-gemv' | 'mps-matmul';
export type Kind = 'gemv' | 'gemm' | 'library' | 'unknown';

export const ROUTES: Record<RouteId, { name: string; kind: Kind; where: string }> = {
  cublas: { name: 'cuBLAS (ggml_cuda_mul_mat_cublas)', kind: 'library', where: 'ggml/src/ggml-cuda/ggml-cuda.cu (the HIP build goes through the renames in vendors/hip.h)' },
  mmvf: { name: 'mmvf: float GEMV (ggml_cuda_mul_mat_vec_f)', kind: 'gemv', where: 'ggml/src/ggml-cuda/mmvf.cu' },
  mmf: { name: 'mmf (ggml_cuda_mul_mat_f)', kind: 'gemm', where: 'ggml/src/ggml-cuda/mmf.cu' },
  mmvq: { name: 'mmvq: quantised GEMV (ggml_cuda_mul_mat_vec_q)', kind: 'gemv', where: 'ggml/src/ggml-cuda/mmvq.cu' },
  mmq: { name: 'mmq: quantised GEMM (ggml_cuda_mul_mat_q)', kind: 'gemm', where: 'ggml/src/ggml-cuda/mmq.cu' },
  fp8: { name: 'the FP8 path', kind: 'unknown', where: 'ggml/src/ggml-cuda/ggml-cuda.cu' },
  mul_mv: { name: 'mul_mv (matrix-vector)', kind: 'gemv', where: 'ggml/src/ggml-metal/kernels/mul_mv.metal' },
  mul_mv_ext: { name: 'mul_mv_ext (small-batch matrix-vector)', kind: 'gemv', where: 'ggml/src/ggml-metal/ggml-metal-ops.cpp' },
  mul_mm: { name: 'mul_mm (SIMD-group matrices)', kind: 'gemm', where: 'ggml/src/ggml-metal/kernels/mul_mm.metal' },
  'metal-unread': { name: 'a path the facts did not read', kind: 'unknown', where: 'ggml/src/ggml-metal/ggml-metal-ops.cpp' },
  'mlx-gemv': { name: 'gemv: a warp per row (gemv_impl)', kind: 'gemv', where: 'mlx/backend/cuda/gemms/gemv.cu' },
  cublaslt: { name: 'cuBLASLt (CublasGemm)', kind: 'library', where: 'mlx/backend/cuda/gemms/cublas_gemm.cpp' },
  wide: { name: 'the wide GEMV', kind: 'gemv', where: 'mlx/backend/metal/matmul.cpp' },
  gemv: { name: 'gemv (GEMVKernel)', kind: 'gemv', where: 'mlx/backend/metal/kernels/gemv.h' },
  steel: { name: 'steel GEMM, tiles chosen per device', kind: 'gemm', where: 'mlx/backend/metal/kernels/steel/gemm/' },
  'torch-blas': { name: 'cuBLAS or cuBLASLt (CK on ROCm), by the preferred BLAS backend', kind: 'library', where: 'aten/src/ATen/cuda/CUDABlas.cpp' },
  'mps-gemv': { name: 'gemv_t / gemv_nt kernels', kind: 'gemv', where: 'aten/src/ATen/native/mps/kernels/Gemv.h' },
  'mps-matmul': { name: "PyTorch's Metal matmul kernels or MPSGraph (which one was not read)", kind: 'gemm', where: 'aten/src/ATen/native/mps/operations/Matmul.mm' },
};

export type Verdict = 'true' | 'false' | 'not-modelled';
export interface Check {
  label: string;
  verdict: Verdict;
  /** What runs when the check is true. */
  route: RouteId;
  detail: string;
  source: string;
}
export interface RouteResult {
  project: Project;
  checks: Check[];
  /** Index of the first check not ruled out. */
  at: number;
  /** The first route not ruled out. */
  candidate: RouteId;
  kind: Kind;
  /** True when that check is true, not merely undecided. */
  certain: boolean;
  /** Routes still possible after the candidate (undecided checks and the fallback). */
  alternatives: RouteId[];
  notes: string[];
}

// Constants read in the facts.
/** mmvf.cuh: "Max. batch size for which to use MMVF kernels." (ch22 D6) */
export const MMVF_MAX_BATCH_SIZE = 8;
/** mmvq.cuh: "Max. batch size for which to use MMVQ kernels." (ch22 D6) */
export const MMVQ_MAX_BATCH_SIZE = 8;
/** llama.cpp Metal mul_mv_ext: (ne11 >= 2 && ne11 <= 8), or (ne11 >= 4 && ne11 <= 8) for k-quants (ch22 D6). */
export const METAL_EXT = { lo: 2, loKQuant: 4, hi: 8 };
/** MLX Metal: "streams the weight matrix once per <= 5 input vectors" (x3 B4, matmul.cpp line 1620). */
export const MLX_WIDE_MAX = 5;

const T = (b: boolean): Verdict => (b ? 'true' : 'false');

/** MLX CUDA, gemv.cu line 188, as quoted: (M == 1 && b_transposed) || (N == 1 && !a_transposed). */
export const canUseGemv = (M: number, N: number, aT: boolean, bT: boolean) => (M === 1 && bT) || (N === 1 && !aT);

function checksFor(p: Project, s: Shape): { checks: Check[]; notes: string[] } {
  switch (p) {
    case 'llama-cuda': {
      const overVec = s.M > MMVF_MAX_BATCH_SIZE;
      return {
        checks: [
          { label: 'the input is not F32', verdict: T(!s.inputF32), route: 'cublas', detail: s.inputF32 ? 'F32: go on' : 'not F32: straight to cuBLAS', source: 'x3 B4, ggml-cuda.cu 2547 to 2548' },
          {
            label: 'ggml_cuda_should_use_mmvf',
            verdict: overVec ? 'false' : isQuant(s.weight) ? 'false' : 'not-modelled',
            route: 'mmvf',
            detail: overVec
              ? `${s.M} tokens > MMVF_MAX_BATCH_SIZE ${MMVF_MAX_BATCH_SIZE}: ruled out`
              : isQuant(s.weight)
                ? 'a quantised weight: mmvf is the float GEMV (our reading of the names in the facts)'
                : `${s.M} ≤ ${MMVF_MAX_BATCH_SIZE}; the rest, including AMD branches, is decided inside should_use_mmvf (not modelled)`,
            source: 'x3 B4, ch22 D6',
          },
          {
            label: 'ggml_cuda_should_use_mmf',
            verdict: isQuant(s.weight) ? 'false' : 'not-modelled',
            route: 'mmf',
            detail: isQuant(s.weight) ? 'a quantised weight: mmf is the float matmul, beside mmq (our reading of the names)' : 'decided inside should_use_mmf (not modelled)',
            source: 'x3 B4, ggml-cuda.cu 2575 to 2576',
          },
          {
            label: 'ggml_cuda_should_use_mmvq',
            verdict: s.M > MMVQ_MAX_BATCH_SIZE ? 'false' : !isQuant(s.weight) ? 'false' : 'not-modelled',
            route: 'mmvq',
            detail: s.M > MMVQ_MAX_BATCH_SIZE
              ? `${s.M} tokens > MMVQ_MAX_BATCH_SIZE ${MMVQ_MAX_BATCH_SIZE}: ruled out`
              : !isQuant(s.weight)
                ? 'a float weight: mmvq is the quantised GEMV (our reading of the names in the facts)'
                : `${s.M} ≤ ${MMVQ_MAX_BATCH_SIZE}, but the limit is lower for some types on some GPUs ("tuned on RTX 4090"): not modelled`,
            source: 'x3 B4, ch22 D6',
          },
          {
            label: 'ggml_cuda_should_use_mmq',
            verdict: isQuant(s.weight) ? 'not-modelled' : 'false',
            route: 'mmq',
            detail: isQuant(s.weight) ? 'decided inside should_use_mmq (not modelled)' : 'a float weight: mmq is the quantised matmul (our reading of the names)',
            source: 'x3 B4, ggml-cuda.cu 2589 to 2590',
          },
          { label: 'an FP8 path', verdict: 'not-modelled', route: 'fp8', detail: 'its condition was not read', source: 'x3 B4' },
          { label: 'otherwise', verdict: 'true', route: 'cublas', detail: 'ggml_cuda_mul_mat_cublas', source: 'x3 B4, ggml-cuda.cu 2596' },
        ],
        notes: ['HIP compiles the same file: the chain is the same, with AMD branches inside should_use_mmvf (ch22 D6, x3 B9).'],
      };
    }
    case 'llama-metal': {
      const lo = s.weight === 'kquant' ? METAL_EXT.loKQuant : METAL_EXT.lo;
      const inExt = s.M >= lo && s.M <= METAL_EXT.hi;
      return {
        checks: [
          { label: 'one token', verdict: T(s.M === 1), route: 'mul_mv', detail: s.M === 1 ? 'mul_mv.metal' : `${s.M} tokens`, source: 'x3 B4 ("mul_mv.metal for one"; where this test sits in the code was not read)' },
          {
            label: `${lo} ≤ tokens ≤ ${METAL_EXT.hi}${s.weight === 'kquant' ? ' (k-quants)' : ''}`,
            verdict: inExt ? 'not-modelled' : 'false',
            route: 'mul_mv_ext',
            detail: inExt
              ? 'in the range; the condition is "for most types" and the full type list was not read'
              : `${s.M} tokens: outside ${lo} to ${METAL_EXT.hi}`,
            source: 'ch22 D6, ggml-metal-ops.cpp',
          },
          { label: 'ggml_metal_op_mul_mat_use_mm', verdict: 'not-modelled', route: 'mul_mm', detail: 'decided inside use_mm (not modelled)', source: 'ch22 D6, ch15 D2' },
          { label: 'otherwise', verdict: 'true', route: 'metal-unread', detail: 'what runs when use_mm declines was not read', source: 'ch22 D6' },
        ],
        notes: [],
      };
    }
    case 'mlx-cuda': {
      const g = !s.bias && canUseGemv(s.M, s.N, s.aTransposed, s.bTransposed);
      return {
        checks: [
          {
            label: '!bias && can_use_gemv: (M == 1 && b transposed) || (N == 1 && !a transposed)',
            verdict: T(g),
            route: 'mlx-gemv',
            detail: s.bias
              ? 'a bias: the GEMV branch needs none'
              : `M == 1: ${s.M === 1}, b transposed: ${s.bTransposed}, N == 1: ${s.N === 1}, a transposed: ${s.aTransposed}`,
            source: 'x3 B4, matmul.cpp line 99, gemv.cu line 188',
          },
          { label: 'otherwise', verdict: 'true', route: 'cublaslt', detail: '"Invoke cublasLt"', source: 'x3 B4, matmul.cpp line 115' },
        ],
        notes: ['No AMD backend: MLX has no HIP build (x3 B9).'],
      };
    }
    case 'mlx-metal': {
      return {
        checks: [
          {
            label: `the wide GEMV: at most ${MLX_WIDE_MAX} input vectors`,
            verdict: s.M <= MLX_WIDE_MAX ? 'not-modelled' : 'false',
            route: 'wide',
            detail: s.M <= MLX_WIDE_MAX ? `${s.M} ≤ ${MLX_WIDE_MAX}; the route's other conditions were not read` : `${s.M} vectors > ${MLX_WIDE_MAX}`,
            source: 'x3 B4, matmul.cpp line 1620',
          },
          { label: 'std::min(M, N) == 1', verdict: T(Math.min(s.M, s.N) === 1), route: 'gemv', detail: `min(${s.M}, ${s.N}) = ${Math.min(s.M, s.N)}`, source: 'x3 B4, matmul.cpp line 1647' },
          { label: 'otherwise', verdict: 'true', route: 'steel', detail: 'tile shape picked per device class (not evaluated)', source: 'ch15 D3' },
        ],
        notes: ['GEMVKernel asserts SM * SN == 32: a SIMD width of 32 written into the code (x3 B4, ch04 D3).'],
      };
    }
    case 'torch-cuda':
      return {
        checks: [
          {
            label: 'gemm_internal: the preferred BLAS backend',
            verdict: 'true',
            route: 'torch-blas',
            detail: "the library picks its own kernel; a gemv entry point exists, with TF32 turned off, but which shapes reach it was not read",
            source: 'ch15 D4, CUDABlas.cpp',
          },
        ],
        notes: ['On ROCm the same sources are hipified; gemm_internal may prefer Composable Kernel (ch15 D4, x3 B9).'],
      };
    case 'torch-mps':
      return {
        checks: [
          { label: 'a rank-1 shape: min(M, N) == 1', verdict: T(Math.min(s.M, s.N) === 1), route: 'mps-gemv', detail: `min(${s.M}, ${s.N}) = ${Math.min(s.M, s.N)}`, source: 'x3 B4, Matmul.mm lines 116 to 141, Linear.mm line 161' },
          { label: 'otherwise', verdict: 'true', route: 'mps-matmul', detail: 'not read in detail (infra D4)', source: 'x3 B4' },
        ],
        notes: [
          "nn.Linear's own path (Linear.mm) uses MPSNDArrayMatrixMultiplication (lines 68 to 82); its comment says addmm routes rank-1 shapes to the GEMV kernels. Which shapes Linear.mm hands to addmm was not read.",
        ],
      };
  }
}

export function route(p: Project, s: Shape): RouteResult {
  const { checks, notes } = checksFor(p, s);
  const at = checks.findIndex((c) => c.verdict !== 'false');
  const c = checks[at];
  const alternatives = checks.slice(at + 1).filter((x) => x.verdict !== 'false').map((x) => x.route);
  return {
    project: p,
    checks,
    at,
    candidate: c.route,
    kind: ROUTES[c.route].kind,
    certain: c.verdict === 'true',
    alternatives: c.verdict === 'true' ? [] : [...new Set(alternatives)],
    notes,
  };
}

/** Projects whose shape rule the facts read (PyTorch CUDA hands every shape to a library). */
export const SHAPE_RULE_READ: Project[] = ['llama-cuda', 'llama-metal', 'mlx-cuda', 'mlx-metal', 'torch-mps'];

/** Switch points written in code, for the strip beside the reader's knee. */
export const CODE_SWITCHES = [
  { label: "llama.cpp's MMVF_MAX_BATCH_SIZE", tokens: MMVF_MAX_BATCH_SIZE, source: 'ch22 D6' },
  { label: "MLX Metal's wide GEMV limit", tokens: MLX_WIDE_MAX, source: 'x3 B4' },
];

// ---------------------------------------------------------------- the reader's knee (ch22's definition)
export interface ReaderKnee {
  backend: string;
  device: string;
  /** Series label, e.g. "lab 22.2, M3 (lab 15.6's register-tiled kernel)". */
  label: string;
  /** The kernel's short name, e.g. "M3". */
  short: string;
  /** First measured batch at 1.5 times the smallest batch's step time; null if none in range. */
  knee: number | null;
  /** Largest batch measured. */
  upTo: number;
  /** The measured (batch, seconds per step) points, by batch. */
  points: { batch: number; seconds: number }[];
}

/** Lab 22.2's 'step' rows on every GPU backend, one entry per GEMM kernel, with ch22's knee (sim/batching.ts kneeOf). */
export function readerKnees(reports: ReportsLike): ReaderKnee[] {
  return gpuBackends(reports).flatMap((b) =>
    stepSeries(reports, b)
      .filter((s) => s.points.length >= 2)
      .map((s) => ({
        backend: b,
        device: String(reports[b].device?.name ?? b),
        label: s.label,
        short: s.label.replace(/^lab 22\.2, /, '').replace(/ \(.*$/, ''),
        knee: s.knee,
        upTo: Math.max(...s.points.map((p) => p.batch)),
        points: s.points,
      })),
  );
}
