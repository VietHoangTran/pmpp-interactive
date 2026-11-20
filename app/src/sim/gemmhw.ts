// What a GEMM kernel can reach on each target: matrix instructions, asynchronous copies,
// template libraries and the vendor baseline (book §15.9, pp. 368–369, checked against
// notes/today/ch15-facts.md and the files it refers to). Every cell names the facts item
// it rests on; "not documented" means the facts files found nothing to point to, which is
// a statement about the documentation, not a claim that the hardware lacks it.
//
// Fact references: a bare id ("A6") is ch15-facts.md; others name their file
// ("infra C10" = infra-facts.md, "ch06 C7" = ch06-facts.md, "ch03 A7" = ch03-facts.md).

export type Backend = 'CUDA' | 'HIP' | 'Metal';
export type FeatureId = 'warpMma' | 'bigMma' | 'asyncCopy' | 'bulkCopy' | 'templateLib' | 'vendorGemm';
/** yes: documented and available; no: documented as unavailable here; undocumented: nothing usable is documented. */
export type Status = 'yes' | 'no' | 'undocumented';

export const FEATURES: readonly { id: FeatureId; label: string; book: string }[] = [
  { id: 'warpMma', label: 'Warp-level matrix multiply', book: 'WMMA (p. 369)' },
  { id: 'bigMma', label: 'Larger matrix multiply (warp group, tensor memory, or a library operation in the kernel)', book: 'WGMMA, TMEM (p. 369)' },
  { id: 'asyncCopy', label: 'Asynchronous global-to-shared copy', book: 'LDGSTS (p. 369)' },
  { id: 'bulkCopy', label: 'Bulk multi-dimensional tensor copy', book: 'TMA (p. 369)' },
  { id: 'templateLib', label: 'Template GEMM library', book: 'CUTLASS, cuTile (p. 368)' },
  { id: 'vendorGemm', label: 'Vendor GEMM to measure against (FP32)', book: 'cuBLAS (p. 368)' },
];

export interface Cell {
  status: Status;
  /** What is there, or why nothing is. */
  what: string;
  /** Facts items behind the cell. */
  items: string[];
}

export interface Target {
  id: string;
  backend: Backend;
  label: string;
  /** CUDA: compute capability; the generation tests below compare it numerically. */
  cc?: number;
  /** CUDA: built for the architecture-specific "a" target (sm_90a). */
  archSpecific?: boolean;
  /** HIP: gfx generation (9 = CDNA, 11 = RDNA3, 12 = RDNA4: ch15-facts B1, B4) and, on CDNA, its version. */
  gfx?: number;
  cdna?: number;
  /** Metal: compiled for Metal 4. */
  metal4?: boolean;
  /** A word about the target itself. */
  note?: string;
}

export const TARGETS: readonly Target[] = [
  { id: 'cuda-7.0', backend: 'CUDA', label: 'CUDA, compute capability 7.0 (Volta)', cc: 7.0 },
  { id: 'cuda-7.5', backend: 'CUDA', label: 'CUDA, compute capability 7.5 (Turing)', cc: 7.5 },
  { id: 'cuda-8.0', backend: 'CUDA', label: 'CUDA, compute capability 8.0', cc: 8.0 },
  { id: 'cuda-8.6', backend: 'CUDA', label: 'CUDA, compute capability 8.6 or 8.9 (RTX 30, RTX 40)', cc: 8.6 },
  { id: 'cuda-9.0', backend: 'CUDA', label: 'CUDA, compute capability 9.0, built as sm_90', cc: 9.0 },
  { id: 'cuda-9.0a', backend: 'CUDA', label: 'CUDA, compute capability 9.0, built as sm_90a', cc: 9.0, archSpecific: true },
  { id: 'cuda-10.0', backend: 'CUDA', label: 'CUDA, compute capability 10.0 (data-centre Blackwell)', cc: 10.0 },
  { id: 'cuda-12', backend: 'CUDA', label: 'CUDA, compute capability 12.x (RTX 50)', cc: 12.0 },
  { id: 'hip-cdna2', backend: 'HIP', label: 'HIP, CDNA 2', gfx: 9, cdna: 2 },
  { id: 'hip-cdna3', backend: 'HIP', label: 'HIP, CDNA 3', gfx: 9, cdna: 3 },
  { id: 'hip-cdna4', backend: 'HIP', label: 'HIP, CDNA 4 (gfx950)', gfx: 9, cdna: 4 },
  { id: 'hip-rdna3', backend: 'HIP', label: 'HIP, RDNA 3 (gfx11)', gfx: 11 },
  { id: 'hip-rdna4', backend: 'HIP', label: 'HIP, RDNA 4 (gfx12)', gfx: 12 },
  { id: 'metal-apple7', backend: 'Metal', label: 'Metal, Apple7 or later, without Metal 4', metal4: false },
  { id: 'metal-apple7-m4', backend: 'Metal', label: 'Metal, Apple7 or later, with Metal 4', metal4: true },
];

export const target = (id: string): Target => {
  const t = TARGETS.find((x) => x.id === id);
  if (!t) throw new Error('unknown target ' + id);
  return t;
};

// ---------------------------------------------------------------- the rules, one per fact
/** nvcc 13.5 lists real architectures from sm_75 up; sm_70 is absent (A5). */
export const cuda13Targets = (cc: number) => cc >= 7.5;
/** nvcuda::wmma: compute capability 7.0 or higher (infra A12). PTX mma: "Requires sm_70 or higher" (A6). */
export const hasWmma = (cc: number) => cc >= 7.0;
/** PTX mma .f16 m16n8k16 and the .tf32 shapes require sm_80 (A6). */
export const hasMmaTf32 = (cc: number) => cc >= 8.0;
/** wgmma "Requires sm_90a"; "a" targets cannot run on later generations (A6). */
export const hasWgmma = (cc: number, archSpecific: boolean) => cc === 9.0 && archSpecific;
/** tcgen05 (with TMEM): sm_100 and sm_110 families only; sm_120 is not in the list (A6). */
export const hasTcgen05 = (cc: number) => (cc >= 10.0 && cc < 11.0) || (cc >= 11.0 && cc < 12.0);
/** LDGSTS / cp.async: CC 8.0+ (A7). */
export const hasLdgsts = (cc: number) => cc >= 8.0;
/** TMA: CC 9.0+ (A8). */
export const hasTma = (cc: number) => cc >= 9.0;
/** BF16x9 emulation in cuBLAS: compute capability 10.0 and 10.3 only (A10). */
export const hasBf16x9 = (cc: number) => cc === 10.0 || cc === 10.3;

/** MFMA builtins: the "mai-insts" feature of the CDNA (gfx9 MI) sets (B4). */
export const hasMfma = (t: Target) => t.backend === 'HIP' && t.gfx === 9;
/** The xf32 MFMA form: llama.cpp uses it on CDNA3; "CDNA4 (gfx950) does not support xf32 MFMA" (B4, D1). */
export const hasXf32Mfma = (t: Target) => t.backend === 'HIP' && t.cdna === 3;
/** WMMA builtins: wmma-256b-insts in the gfx11 sets, wmma-128b-insts in gfx12 (B4). */
export const hasRdnaWmma = (t: Target) => t.backend === 'HIP' && (t.gfx === 11 || t.gfx === 12);
/** global_load_lds: "vmem-to-lds-load-insts", in the GFX9 and GFX10 generation sets, not GFX11/12 (B5). */
export const hasGlobalLoadLds = (t: Target) => t.backend === 'HIP' && (t.gfx === 9 || t.gfx === 10);
/** The data movement engine is described for CDNA 3 and 4 (B5). */
export const hasDme = (t: Target) => t.backend === 'HIP' && (t.cdna === 3 || t.cdna === 4);

// ---------------------------------------------------------------- the table
function cuda(t: Target, f: FeatureId): Cell {
  const cc = t.cc!;
  switch (f) {
    case 'warpMma':
      return {
        status: 'yes',
        what: hasMmaTf32(cc)
          ? 'nvcuda::wmma in <mma.h>, still documented (not "pre-Hopper only"); PTX mma.sync, including the f16 m16n8k16 and tf32 shapes.'
          : 'nvcuda::wmma in <mma.h>; PTX mma.sync. The f16 m16n8k16 and tf32 shapes need sm_80.',
        items: ['A6', 'infra A12'],
      };
    case 'bigMma':
      if (hasWgmma(cc, !!t.archSpecific))
        return {
          status: 'yes',
          what: 'WGMMA (wgmma.mma_async): four warps; A from registers or shared memory, B from shared memory; issue, commit and wait are separate. Code built for sm_90a cannot run on later generations.',
          items: ['A6'],
        };
      if (hasTcgen05(cc))
        return {
          status: 'yes',
          what: 'tcgen05.mma with tensor memory (TMEM): the accumulator is written to TMEM, out of the register file. WGMMA is not available: it is sm_90a only, and "a" code does not run here.',
          items: ['A6'],
        };
      if (cc === 9.0)
        return { status: 'no', what: 'WGMMA "Requires sm_90a": built as plain sm_90 the kernel cannot use it.', items: ['A6'] };
      if (cc >= 12.0)
        return {
          status: 'no',
          what: 'No TMEM: tcgen05 targets are the sm_100 and sm_110 families, and sm_120 is not in the list. No WGMMA either (sm_90a only).',
          items: ['A6'],
        };
      return { status: 'no', what: 'WGMMA needs sm_90a and tcgen05 the sm_100 or sm_110 families.', items: ['A6'] };
    case 'asyncCopy':
      return hasLdgsts(cc)
        ? {
            status: 'yes',
            what: 'LDGSTS / cp.async (CC 8.0+). cuda::memcpy_async does not guarantee it; the __pipeline_memcpy_async primitives do. Check the SASS or Nsight Compute\'s memory chart.',
            items: ['A7'],
          }
        : { status: 'no', what: 'LDGSTS needs compute capability 8.0 (cp.async "Requires sm_80 or higher").', items: ['A7'] };
    case 'bulkCopy':
      return hasTma(cc)
        ? {
            status: 'yes',
            what: 'TMA (CC 9.0+). A 2D tile needs a tensor map built on the host (cuTensorMapEncodeTiled) and cuda::ptx::cp_async_bulk_tensor; cuda::memcpy_async reaches TMA only for 1D bulk copies.',
            items: ['A8'],
          }
        : { status: 'no', what: 'TMA needs compute capability 9.0.', items: ['A8'] };
    case 'templateLib':
      return {
        status: 'yes',
        what: 'CUTLASS; CUDA Tile (cuTile Python, and CUDA Tile C++ from toolkit 13.3). Which generations each supports is not in our facts files.',
        items: ['A9', 'infra A12'],
      };
    case 'vendorGemm':
      return {
        status: 'yes',
        what:
          'cublasSgemm (cublas_v2.h, CUDA::cublas), column-major. The default math mode keeps FP32; TF32 is opt-in' +
          (hasBf16x9(cc) ? ', and so is BF16x9 emulation, which exists on this compute capability.' : '.'),
        items: ['A10', 'ch03 A7'],
      };
  }
}

function hip(t: Target, f: FeatureId): Cell {
  switch (f) {
    case 'warpMma':
      if (hasMfma(t))
        return {
          status: 'yes',
          what:
            'MFMA builtins, including FP32 in and out (__builtin_amdgcn_mfma_f32_16x16x4f32); rocWMMA as the C++ route. ' +
            (hasXf32Mfma(t)
              ? 'The reduced-precision xf32 form exists here (llama.cpp uses it on CDNA3).'
              : t.cdna === 4
                ? 'No xf32 form: "CDNA4 (gfx950) does not support xf32 MFMA".'
                : 'llama.cpp takes the plain f32 path here, not xf32.'),
          items: ['B4', 'D1', 'infra B12'],
        };
      return {
        status: 'yes',
        what:
          (t.gfx === 12 ? '__builtin_amdgcn_wmma_*_w32_gfx12 builtins (wmma-128b-insts).' : '__builtin_amdgcn_wmma_*_w32 builtins (wmma-256b-insts).') +
          ' The HIP features table shows no Matrix Cores on RDNA, but the ROCm 10.2 compiler has these. No MFMA.',
        items: ['B4'],
      };
    case 'bigMma':
      return { status: 'undocumented', what: 'Nothing beyond MFMA or WMMA is documented.', items: ['ch15 table'] };
    case 'asyncCopy':
      return hasGlobalLoadLds(t)
        ? {
            status: 'undocumented',
            what: '__builtin_amdgcn_global_load_lds exists in the compiler (GFX9 generation) but not in the HIP guide, so it is not taught. The documented name, __builtin_amdgcn_async_work_group_copy, does not exist in ROCm 10.2.',
            items: ['B5'],
          }
        : {
            status: 'undocumented',
            what: 'None documented. The compiler\'s global-to-LDS load feature is in the GFX9 and GFX10 sets, not GFX11 or GFX12.',
            items: ['B5'],
          };
    case 'bulkCopy':
      return hasDme(t)
        ? {
            status: 'undocumented',
            what: 'AMD describes a data movement engine on CDNA 3 and 4, but the intrinsic it names does not exist in ROCm 10.2: no usable documented API.',
            items: ['B5'],
          }
        : { status: 'undocumented', what: 'None documented.', items: ['B5'] };
    case 'templateLib':
      return {
        status: 'yes',
        what:
          'Composable Kernel (with CK Tile); rocWMMA.' +
          (t.gfx === 11 || t.gfx === 12 ? ' PyTorch\'s code notes "gfx11 and gfx12 have no CK fp32 GEMM".' : ''),
        items: ['B7', 'infra B12', 'D4'],
      };
    case 'vendorGemm':
      return {
        status: 'yes',
        what: 'hipblasSgemm (roc::hipblas): the same call shape as cublasSgemm, column-major, FP32 by default; XF32 is opt-in.',
        items: ['B6', 'ch03 B6'],
      };
  }
}

function metal(t: Target, f: FeatureId): Cell {
  switch (f) {
    case 'warpMma':
      return {
        status: 'yes',
        what: 'simdgroup_float8x8 with simdgroup_multiply_accumulate (8 x 8 only, Apple7 and later). Apple documents no matrix unit behind it: not "tensor cores".',
        items: ['C5', 'infra C10'],
      };
    case 'bigMma':
      return t.metal4
        ? {
            status: 'yes',
            what: 'Metal Performance Primitives matmul2d over one or more SIMD-groups, float x float -> float from Metal 4; half x int4 formats from Metal 4 with OS 26.4. Apple recommends it over simdgroup_matrix.',
            items: ['C5', 'infra C10'],
          }
        : { status: 'no', what: 'MPP matmul2d needs Metal 4.', items: ['infra C10'] };
    case 'asyncCopy':
      return { status: 'undocumented', what: 'None documented: the MSL 4.1 specification has no "async".', items: ['ch06 C7'] };
    case 'bulkCopy':
      return { status: 'undocumented', what: 'None documented.', items: ['ch15 table'] };
    case 'templateLib':
      return { status: 'undocumented', what: 'No template GEMM library; MPP is the nearest thing.', items: ['ch15 table'] };
    case 'vendorGemm':
      return {
        status: 'yes',
        what: 'MPSMatrixMultiplication (-framework MetalPerformanceShaders), row-major, with transpose flags. Uses higher internal precision by default; reduced precision is an option you set.',
        items: ['C6', 'ch03 C7'],
      };
  }
}

export function cell(t: Target, f: FeatureId): Cell {
  return t.backend === 'CUDA' ? cuda(t, f) : t.backend === 'HIP' ? hip(t, f) : metal(t, f);
}

export interface TargetSummary {
  target: Target;
  /** CUDA only: whether today's toolkit (CUDA 13) compiles for it at all. */
  toolkit?: Cell;
  cells: Record<FeatureId, Cell>;
}

export function summarize(id: string): TargetSummary {
  const t = target(id);
  const cells = Object.fromEntries(FEATURES.map((f) => [f.id, cell(t, f.id)])) as Record<FeatureId, Cell>;
  const toolkit =
    t.backend === 'CUDA'
      ? cuda13Targets(t.cc!)
        ? { status: 'yes' as const, what: 'A CUDA 13 target (nvcc 13.5 lists sm_75 and later).', items: ['A5'] }
        : {
            status: 'no' as const,
            what: 'Tensor cores date from Volta, but nvcc 13.5 does not list sm_70 and NVIDIA\'s libraries dropped pre-Turing GPUs. (We found no single toolkit-wide sentence saying so.)',
            items: ['A5'],
          }
      : undefined;
  return { target: t, toolkit, cells };
}
