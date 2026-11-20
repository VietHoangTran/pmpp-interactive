// Which attention kernel each project runs (module X3, section X3.5). Not in the book. Every
// check is one read in notes/today/x3-facts.md (B5, B9) or ch20-facts.md (B5, D1, D2, D4, D5),
// at the commits listed there.
//
// What it models:
//   - MLX CUDA: cuDNN's SDPA when supports_sdpa_cudnn accepts the call; for one query that
//     needs no array mask and at least 256 keys (use_cudnn_for_decoding, kv_cache_step = 256,
//     "number is from mlx-lm"); otherwise its own sdpa_vector kernels for one query.
//   - MLX Metal: vector kernels for 8 or fewer queries; among them the 2-pass kernel when the
//     device class is 'd' or 's' with at least 1024 keys, or with GQA from 4096 keys; tiled
//     steel attention for more queries.
//   - PyTorch CUDA: select_sdp_backend walks a priority order (default flash, efficient, math,
//     cuDNN, overrideable; cuDNN first when it is preferred) and takes the first backend whose
//     use_<backend> check passes.
//   - llama.cpp: its own kernels on every backend.
//
// What it leaves out: the real conditions of supports_sdpa_cudnn and of PyTorch's use_<backend>
// checks (not read: the reader answers them), how MLX CUDA picks its 1-pass or 2-pass vector
// kernel, what MLX CUDA does for several queries when cuDNN declines, which llama.cpp variant
// (vector, tile, MMA) runs for a shape, the MPS kernels' thresholds, head sizes and dtypes.

export type AttnProject = 'llama' | 'mlx-cuda' | 'mlx-metal' | 'torch-cuda' | 'torch-rocm' | 'torch-mps';
export const ATTN_PROJECTS: AttnProject[] = ['llama', 'mlx-cuda', 'mlx-metal', 'torch-cuda', 'torch-rocm', 'torch-mps'];
export const ATTN_LABEL: Record<AttnProject, string> = {
  llama: 'llama.cpp (CUDA, HIP, Metal)',
  'mlx-cuda': 'MLX, CUDA',
  'mlx-metal': 'MLX, Metal',
  'torch-cuda': 'PyTorch, CUDA',
  'torch-rocm': 'PyTorch, ROCm',
  'torch-mps': 'PyTorch, MPS',
};

export type Mask = 'none' | 'causal' | 'array';
export type TorchBackend = 'flash' | 'efficient' | 'math' | 'cudnn' | 'overrideable';
export const TORCH_BACKEND_LABEL: Record<TorchBackend, string> = {
  flash: 'flash_attention',
  efficient: 'efficient_attention',
  math: 'math',
  cudnn: 'cudnn_attention',
  overrideable: 'overrideable',
};

export interface AttnInput {
  /** Queries per sequence (q.shape(2) in MLX). */
  queries: number;
  keys: number;
  mask: Mask;
  /** Fewer K/V heads than query heads. */
  gqa: boolean;
  /** MLX Metal: the device class letter is 'd' or 's' (devc == 'd' || devc == 's'). */
  deviceDS: boolean;
  /** MLX CUDA: the reader's answer to supports_sdpa_cudnn for the other conditions (not read). */
  cudnnSupports: boolean;
  /** PyTorch CUDA: check_prefer_cudnn_attention(). */
  preferCudnn: boolean;
  /** PyTorch CUDA: the reader's answers to each use_<backend> check (not read). */
  use: Record<TorchBackend, boolean>;
}

export const DEFAULT_INPUT: AttnInput = {
  queries: 1,
  keys: 1000,
  mask: 'none',
  gqa: false,
  deviceDS: false,
  cudnnSupports: true,
  preferCudnn: false,
  use: { flash: true, efficient: true, math: true, cudnn: true, overrideable: false },
};

/** Context.h lines 509 to 514. */
export const SDP_PRIORITY_ORDER: TorchBackend[] = ['flash', 'efficient', 'math', 'cudnn', 'overrideable'];
/** use_cudnn_for_decoding: "constexpr int kv_cache_step = 256; // number is from mlx-lm" */
export const KV_CACHE_STEP = 256;
/** MLX Metal: q_pre.shape(2) <= 8 selects the vector kernels. */
export const MLX_VECTOR_MAX_QUERIES = 8;
/** MLX Metal 2-pass: n_keys >= 1024 on a 'd' or 's' device, or GQA with n_keys >= 4096. */
export const MLX_2PASS_KEYS = 1024;
export const MLX_2PASS_GQA_KEYS = 4096;

export type Verdict = 'true' | 'false' | 'not-modelled' | 'reader';
export interface AttnCheck {
  label: string;
  verdict: Verdict;
  detail: string;
  source: string;
}
export type Home = 'own' | 'library' | 'submodule' | 'unknown';
export interface AttnKernel {
  name: string;
  where: string;
  home: Home;
}
export interface AttnResult {
  project: AttnProject;
  checks: AttnCheck[];
  kernel: AttnKernel;
  /** PyTorch CUDA: the order walked. */
  order?: TorchBackend[];
  notes: string[];
}

const T = (b: boolean): Verdict => (b ? 'true' : 'false');

export const TORCH_KERNEL: Record<TorchBackend, AttnKernel> = {
  flash: {
    name: 'FlashAttention-2 (run_mha_fwd_ or run_mha_fwd_splitkv_dispatch)',
    where: 'third_party/flash-attention, a submodule pinned at bb25ba3; PyTorch holds the shim flash_attn/flash_api.cpp',
    home: 'submodule',
  },
  efficient: { name: 'memory-efficient attention', where: 'aten/src/ATen/native/transformers/cuda/mem_eff_attention/', home: 'own' },
  math: { name: 'the math path: "a PyTorch implementation defined in C++" (the docs)', where: 'in PyTorch itself; its file was not read', home: 'own' },
  cudnn: { name: "cuDNN's attention", where: 'the cuDNN library (no source)', home: 'library' },
  overrideable: { name: 'an overrideable backend', where: 'registered from outside (not read)', home: 'unknown' },
};

export function attnRoute(p: AttnProject, x: AttnInput): AttnResult {
  switch (p) {
    case 'llama':
      return {
        project: p,
        checks: [{ label: 'which variant for this shape', verdict: 'not-modelled', detail: 'vector, tile or MMA kernels: the selection was not read', source: 'ch20 D1, D2' }],
        kernel: {
          name: 'its own flash-attention kernels: fattn-vec (online softmax, split-KV combine, GQA ratio) and the other fattn-* variants; Metal fa_common and fa_aux',
          where: 'ggml/src/ggml-cuda/fattn-*.cu (also the HIP build); ggml/src/ggml-metal/kernels/fa_*.metal',
          home: 'own',
        },
        notes: ['The HIP backend compiles the same fattn files (x3 B9).'],
      };
    case 'mlx-cuda': {
      const one = x.queries === 1;
      const decodeOk = !one || (x.mask !== 'array' && x.keys >= KV_CACHE_STEP);
      const cudnn = x.cudnnSupports && decodeOk;
      const checks: AttnCheck[] = [
        {
          label: 'supports_sdpa_cudnn (its other conditions)',
          verdict: 'reader',
          detail: x.cudnnSupports ? 'you said yes; the real checks were not read' : 'you said no',
          source: 'x3 B5, scaled_dot_product_attention.cpp 650 to 654',
        },
      ];
      if (one)
        checks.push({
          label: `one query: no array mask and at least ${KV_CACHE_STEP} keys (use_cudnn_for_decoding)`,
          verdict: T(decodeOk),
          detail: `mask: ${x.mask}, keys: ${x.keys}. "for small sequence the overhead would kill the advantage"; 256 is mlx-lm's cache step`,
          source: 'x3 B5, lines 72 to 88',
        });
      if (cudnn)
        return {
          project: p,
          checks,
          kernel: { name: "cuDNN's SDPA, built with the cuDNN frontend graph API (sdpa_cudnn)", where: 'the cuDNN library (no source); MLX holds the graph in scaled_dot_product_attention.cpp', home: 'library' },
          notes: ['Which cuDNN version MLX needs was not verified (x3 B5).'],
        };
      if (one)
        return {
          project: p,
          checks: [...checks, { label: '1-pass or 2-pass', verdict: 'not-modelled', detail: 'kernel_sdpav_1pass or kernel_sdpav_2pass_1: the choice was not read', source: 'x3 B5' }],
          kernel: { name: 'its own sdpa_vector kernels (kernel_sdpav_1pass, kernel_sdpav_2pass_1), online softmax in base 2', where: 'mlx/backend/cuda/scaled_dot_product_attention.cu', home: 'own' },
          notes: [],
        };
      return {
        project: p,
        checks,
        kernel: { name: 'not read: the facts give no MLX CUDA kernel for several queries when cuDNN declines', where: 'mlx/backend/cuda/scaled_dot_product_attention.cpp', home: 'unknown' },
        notes: [],
      };
    }
    case 'mlx-metal': {
      const vector = x.queries <= MLX_VECTOR_MAX_QUERIES;
      const checks: AttnCheck[] = [
        { label: `q_pre.shape(2) <= ${MLX_VECTOR_MAX_QUERIES} (queries)`, verdict: T(vector), detail: `${x.queries} queries`, source: 'ch20 D4, line 1820' },
      ];
      if (!vector)
        return {
          project: p,
          checks,
          kernel: { name: 'tiled steel attention (sdpa_full_self_attention_metal)', where: 'called from mlx/backend/metal/scaled_dot_product_attention.cpp, line 1929', home: 'own' },
          notes: [],
        };
      const big = x.deviceDS && x.keys >= MLX_2PASS_KEYS;
      const gqaLong = x.gqa && x.keys >= MLX_2PASS_GQA_KEYS;
      checks.push(
        { label: `device class 'd' or 's' and n_keys >= ${MLX_2PASS_KEYS}`, verdict: T(big), detail: `devc d or s: ${x.deviceDS}, keys: ${x.keys}`, source: 'ch20 D4, lines 1885 to 1896' },
        { label: `k.shape(1) < q.shape(1) (GQA) and n_keys >= ${MLX_2PASS_GQA_KEYS}`, verdict: T(gqaLong), detail: `GQA: ${x.gqa}, keys: ${x.keys}`, source: 'ch20 D4, lines 1885 to 1896' },
      );
      return {
        project: p,
        checks,
        kernel: big || gqaLong
          ? { name: 'sdpa_vector_2pass (split over keys, then merged)', where: 'called from mlx/backend/metal/scaled_dot_product_attention.cpp, lines 1885 to 1896', home: 'own' }
          : { name: 'sdpa_vector (one pass, fast::exp)', where: 'mlx/backend/metal/kernels/sdpa_vector.h', home: 'own' },
        notes: ['Which Apple chips report which device class letter is not documented in the file (ch15 D3).'],
      };
    }
    case 'torch-cuda': {
      const order: TorchBackend[] = x.preferCudnn ? ['cudnn', ...SDP_PRIORITY_ORDER.filter((b) => b !== 'cudnn')] : SDP_PRIORITY_ORDER.slice();
      const checks: AttnCheck[] = [];
      let chosen: TorchBackend | null = null;
      for (const b of order) {
        if (chosen) {
          checks.push({ label: `use_${TORCH_BACKEND_LABEL[b]}`, verdict: 'not-modelled', detail: 'not reached', source: 'x3 B5' });
          continue;
        }
        checks.push({ label: `use_${TORCH_BACKEND_LABEL[b]}`, verdict: 'reader', detail: x.use[b] ? 'you said it passes' : 'you said it fails', source: 'x3 B5, sdp_utils.cpp 48 to 58 (Note [SDPA Runtime Dispatch])' });
        if (x.use[b]) chosen = b;
      }
      return {
        project: p,
        checks,
        order,
        kernel: chosen ? TORCH_KERNEL[chosen] : { name: 'no backend accepts the call', where: '', home: 'unknown' },
        notes: x.preferCudnn
          ? ['When cuDNN is preferred, the order of the others was not read; they are shown in the default order.']
          : [],
      };
    }
    case 'torch-rocm':
      return {
        project: p,
        checks: [{ label: 'AOTriton or Composable Kernel', verdict: 'not-modelled', detail: 'the selection on ROCm was not read', source: 'ch20 B5, x3 B9' }],
        kernel: { name: 'AOTriton or CK flash attention', where: 'aten/src/ATen/native/transformers/hip/ (aotriton_adapter.h, flash_attn/{aot,ck}); AOTriton pinned in cmake/External/aotriton.cmake', home: 'library' },
        notes: ['The CUDA sources are hipified at build time (x3 B9).'],
      };
    case 'torch-mps':
      return {
        project: p,
        checks: [{ label: 'long cache, short query or prefill', verdict: 'not-modelled', detail: 'Attention.mm routes to three kernels; the thresholds were not read', source: 'ch20 D5, Attention.mm 403 to 410' }],
        kernel: {
          name: x.queries === 1
            ? 'its own decode kernels: sdpa_vector_2pass_mps (long caches) or sdpa_vector_fast_mps'
            : 'its own kernels: sdpa_vector_fast_mps (short queries) or sdpa_prefill_mps',
          where: 'aten/src/ATen/native/mps/kernels/Attention.metal ("Largely influeneced by" MLX)',
          home: 'own',
        },
        notes: [],
      };
  }
}

// ---------------------------------------------------------------- the start page's prediction
export interface HomeRow {
  project: 'llama.cpp' | 'MLX' | 'PyTorch';
  backend: string;
  /** Where the attention kernels live, by the facts. */
  home: string;
  /** Every kernel for every shape is source in the project's own tree. */
  allOwn: boolean;
}
/** Where each project's attention kernels live, backend by backend (x3 B5, ch20 B5, D4, D5). */
export const ATTENTION_HOME: HomeRow[] = [
  { project: 'llama.cpp', backend: 'CUDA and HIP', home: 'its own fattn-* kernels', allOwn: true },
  { project: 'llama.cpp', backend: 'Metal', home: 'its own fa_common / fa_aux kernels', allOwn: true },
  { project: 'MLX', backend: 'Metal', home: 'its own vector and steel kernels', allOwn: true },
  { project: 'MLX', backend: 'CUDA', home: "cuDNN for prefill and long decodes, its own vector kernels for short decodes", allOwn: false },
  { project: 'PyTorch', backend: 'CUDA', home: 'FlashAttention-2 from a submodule, its own memory-efficient and math paths, cuDNN', allOwn: false },
  { project: 'PyTorch', backend: 'ROCm', home: 'AOTriton or Composable Kernel', allOwn: false },
  { project: 'PyTorch', backend: 'MPS', home: 'its own Metal kernels', allOwn: true },
];
/** Projects whose own tree holds the attention kernel on every GPU backend they support. */
export const OWN_EVERYWHERE = [...new Set(ATTENTION_HOME.map((r) => r.project))].filter((p) =>
  ATTENTION_HOME.filter((r) => r.project === p).every((r) => r.allOwn),
);

/** Where a project's CUDA attention kernels come from, as the start page's prediction asks it. */
export type CudaHome = 'own' | 'mixed' | 'submodule';
/** Every shape the explorer offers, with the library checks answered yes (the defaults). */
const SHAPES: AttnInput[] = [1, 8, 128].flatMap((queries) =>
  [1, 100, 256, 1000, 5000].flatMap((keys) =>
    (['none', 'causal', 'array'] as Mask[]).flatMap((mask) => [false, true].map((gqa) => ({ ...DEFAULT_INPUT, queries, keys, mask, gqa }))),
  ),
);
/**
 * Derived from attnRoute over every shape: 'submodule' when the default first choice is another
 * project's kernels, 'own' when every shape runs the project's own source, else 'mixed'.
 */
export function cudaHome(p: 'llama' | 'mlx-cuda' | 'torch-cuda'): CudaHome {
  if (attnRoute(p, DEFAULT_INPUT).kernel.home === 'submodule') return 'submodule';
  return SHAPES.every((x) => attnRoute(p, x).kernel.home === 'own') ? 'own' : 'mixed';
}
/** The start page stores its pick by these project ids (sim/backendmap.ts). */
export const CUDA_HOME_OF = { 'llama.cpp': 'llama', mlx: 'mlx-cuda', pytorch: 'torch-cuda' } as const;
