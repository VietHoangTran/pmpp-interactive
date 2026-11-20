// The library view of a convolutional layer: chapter 19, section 19.4 (book pp. 472–474).
//
// What the book gives (notes/ch19.md, §19.4): cuDNN's parameter names (Table 19.1,
// p. 473), the output size only as P = f(H; R; u; pad_h) with no formula, and the
// algorithm families "GEMM, Winograd, FFT, etc." (p. 473). Everything else is from
// notes/today/ch19-facts.md, read on 2026-11-20, and names its item:
// - A2: cuDNN's output-size formula with padding, stride and dilation, and PyTorch's and
//   llama.cpp's spellings of the same formula.
// - A3, B3, B4: the forward algorithms of cuDNN (8) and MIOpen (5); HIPIFY's mapping.
// - A7: CUTLASS's GEMM orientation, GEMM_M = N·P·Q, GEMM_N = K, GEMM_K = C·R·S.
// - C2: Metal's documented floor for one buffer, 256 MB.
// - D1: llama.cpp's CUDA dispatch (pointwise, small_conv, implicit GEMM).
// - D3: MLX's Winograd preconditions, as far as the facts give them.
// - D4: the requires_columns test of PyTorch's fallback, Slow2d.
// - E1, E2: duplication of the unfolded input (Eq. 19.4 with stride, padding, dilation).
//
// Also here, for the chapter's start page: the row-major offset of a 4D tensor in NCHW
// and NHWC (ch. 3 §3.2, p. 55; cuDNN's formats, A5) and ceiling division (phases of a
// tiled GEMM, ch. 5 §5.5; Fig. 19.11 as printed truncates instead, notes p. 471).
//
// Simplifications: 2D, groups 1, a square filter, and the same padding, stride and
// dilation on both axes (the libraries let each differ). The dispatch rules model shape
// conditions only, as read on one day. Data types, hardware checks and every condition
// the facts file does not record are left out, and the results say "not modelled".

// ---------------------------------------------------------------- the layer

/** A layer in the chapter's names, plus the library-only parameters. */
export interface Layer {
  /** Samples in the batch. */
  N: number;
  /** Input feature maps. */
  C: number;
  /** Input height and width. */
  H: number;
  W: number;
  /** Output feature maps (cuDNN's K). */
  M: number;
  /** Filter side (cuDNN's R and S). */
  K: number;
  /** Zero rows and columns added on every side (cuDNN's pad_h, pad_w). Not in the book's kernels. */
  pad: number;
  /** Step between windows (cuDNN's u, v). Not in the book's kernels. */
  stride: number;
  /** Spacing between filter taps (cuDNN's dilation). In neither the book's kernels nor Table 19.1. */
  dilation: number;
}

/** The book's kernels: no padding, stride 1, dilation 1. */
export const BOOK_DEFAULTS = { pad: 0, stride: 1, dilation: 1 } as const;

/** Bounds the explorer offers; values outside are clamped. */
export const LIMITS: Record<keyof Layer, [number, number]> = {
  N: [1, 256],
  C: [1, 4096],
  H: [1, 4096],
  W: [1, 4096],
  M: [1, 4096],
  K: [1, 31],
  pad: [0, 15],
  stride: [1, 32],
  dilation: [1, 8],
};

/** Clamp one field to an integer inside LIMITS. */
export function clampField(key: keyof Layer, v: number): number {
  const [lo, hi] = LIMITS[key];
  if (!Number.isFinite(v)) return lo;
  return Math.min(hi, Math.max(lo, Math.round(v)));
}

export const ceilDiv = (a: number, b: number) => Math.ceil(a / b);

// ---------------------------------------------------------------- output size (A2)

/** Height or width the dilated filter covers: (K − 1)·dilation + 1. */
export const span = (filter: number, dilation = 1) => (filter - 1) * dilation + 1;

/**
 * cuDNN, cudnnGetConvolution2dForwardOutputDim (A2):
 * outputDim = 1 + (inputDim + 2·pad − ((filterDim − 1)·dilation + 1)) / convolutionStride,
 * integer division. Returns 0 when the dilated filter does not fit in the padded input.
 */
export function outputDim(input: number, filter: number, pad = 0, stride = 1, dilation = 1): number {
  const room = input + 2 * pad - span(filter, dilation);
  if (room < 0) return 0;
  return 1 + Math.trunc(room / stride);
}

/** PyTorch's spelling (torch.nn.Conv2d, A2): floor((H_in + 2·padding − dilation·(kernel_size − 1) − 1)/stride + 1). */
export const outputDimTorch = (input: number, filter: number, pad = 0, stride = 1, dilation = 1) =>
  Math.floor((input + 2 * pad - dilation * (filter - 1) - 1) / stride + 1);

/** llama.cpp's spelling, ggml_calc_conv_output_size (A2): (ins + 2·p − d·(ks − 1) − 1)/s + 1 in C integer arithmetic. */
export const outputDimGgml = (input: number, filter: number, pad = 0, stride = 1, dilation = 1) =>
  Math.trunc((input + 2 * pad - dilation * (filter - 1) - 1) / stride) + 1;

/** The chapter's formula, H − K + 1 (p. 457): correct only with no padding, stride 1, dilation 1. */
export const chapterOutputDim = (input: number, filter: number) => input - filter + 1;

export interface Shape {
  P: number;
  Q: number;
  /** False when the filter does not fit: no output at all. */
  valid: boolean;
  /** The filter's footprint on the input, (K − 1)·dilation + 1. */
  span: number;
  /** True when the chapter's H − K + 1 applies (pad 0, stride 1, dilation 1). */
  chapterApplies: boolean;
}

export function shape(l: Layer): Shape {
  const P = outputDim(l.H, l.K, l.pad, l.stride, l.dilation);
  const Q = outputDim(l.W, l.K, l.pad, l.stride, l.dilation);
  return {
    P,
    Q,
    valid: P > 0 && Q > 0,
    span: span(l.K, l.dilation),
    chapterApplies: l.pad === 0 && l.stride === 1 && l.dilation === 1,
  };
}

/** The arithmetic of cuDNN's formula for one axis, written out for display. */
export function outputDimSteps(input: number, filter: number, pad: number, stride: number, dilation: number) {
  const s = span(filter, dilation);
  const room = input + 2 * pad - s;
  return { span: s, room, quotient: room < 0 ? null : Math.trunc(room / stride), result: outputDim(input, filter, pad, stride, dilation) };
}

// ---------------------------------------------------------------- GEMM views

export interface Gemm {
  /** Rows of the left operand and of the result. */
  rows: number;
  /** The reduction length: columns of the left operand, rows of the right. */
  inner: number;
  /** Columns of the right operand and of the result. */
  cols: number;
  /** How many such products the layer needs. */
  count: number;
}

/**
 * The book's orientation, one product per sample (pp. 464–466): F (M × C·K·K) times
 * B (C·K·K × P·Q) gives that sample's Y (M × P·Q), already in NCHW order.
 */
export function bookGemm(l: Layer): Gemm {
  const { P, Q } = shape(l);
  return { rows: l.M, inner: l.C * l.K * l.K, cols: P * Q, count: l.N };
}

/**
 * The orientation CUTLASS documents (A7), one product for the batch: activations
 * (N·P·Q × C·R·S) times filters (C·R·S × K) gives N·P·Q × K. GEMM_M = N·P·Q, GEMM_N = K
 * (the chapter's M), GEMM_K = C·R·S. Rows are output pixels: the transpose of the book's Y.
 */
export function libraryGemm(l: Layer): Gemm {
  const { P, Q } = shape(l);
  return { rows: l.N * P * Q, inner: l.C * l.K * l.K, cols: l.M, count: 1 };
}

/** Multiply-adds count two operations each: 2·N·M·P·Q·C·K·K. */
export function flops(l: Layer): number {
  const { P, Q } = shape(l);
  return 2 * l.N * l.M * P * Q * l.C * l.K * l.K;
}

// ---------------------------------------------------------------- the unfolded input (Eq. 19.4, E1, E2)

/** Elements of the unfolded input B for one sample: C·K·K rows by P·Q columns (p. 467). */
export function unfoldedPerSample(l: Layer): number {
  const { P, Q } = shape(l);
  return l.C * l.K * l.K * P * Q;
}

/** Elements of the stored input X for one sample, without padding. */
export const inputPerSample = (l: Layer) => l.C * l.H * l.W;

/**
 * Eq. 19.4 with stride, padding and dilation: B's elements over X's. Overlapping windows
 * make it larger than 1 (it approaches K² at stride 1); stride equal to K with no padding
 * makes it exactly 1 when K divides the image (E1), and below 1 when pixels are skipped.
 */
export const duplication = (l: Layer) => unfoldedPerSample(l) / inputPerSample(l);

/** The largest element count a signed 32-bit index reaches (E2). */
export const INT32_MAX = 2 ** 31 - 1;
/** Apple: maxBufferLength "is at least 256 MB (268,435,456 bytes)" (C2). A floor, not any device's value. */
export const METAL_BUFFER_FLOOR = 268_435_456;

export interface UnfoldSize {
  perSample: number;
  batch: number;
  /** FP32 bytes of B for the whole batch, unfolded at once. */
  batchBytes: number;
  perSampleBytes: number;
  overInt32: boolean;
  overMetalFloor: boolean;
}

export function unfoldSize(l: Layer, bytesPerElement = 4): UnfoldSize {
  const perSample = unfoldedPerSample(l);
  const batch = perSample * l.N;
  return {
    perSample,
    batch,
    perSampleBytes: perSample * bytesPerElement,
    batchBytes: batch * bytesPerElement,
    overInt32: batch > INT32_MAX,
    overMetalFloor: batch * bytesPerElement > METAL_BUFFER_FLOOR,
  };
}

// ---------------------------------------------------------------- layouts and indexing (start page)

export type Layout = 'NCHW' | 'NHWC';

/** Row-major offset of element (n, c, h, w) of an N × C × H × W tensor stored in the given order. */
export function offset(layout: Layout, dims: { C: number; H: number; W: number }, n: number, c: number, h: number, w: number): number {
  const { C, H, W } = dims;
  return layout === 'NCHW' ? ((n * C + c) * H + h) * W + w : ((n * H + h) * W + w) * C + c;
}

/** Phases of a tiled GEMM over a reduction of length `inner`, with bounds checks (ch. 5 §5.5). */
export const phases = (inner: number, tile: number) => ceilDiv(inner, tile);
/** Phases Fig. 19.11 runs as printed: (C·K·K)/TILE_WIDTH, truncated (notes, p. 471). */
export const printedPhases = (inner: number, tile: number) => Math.floor(inner / tile);

// ---------------------------------------------------------------- names: Table 19.1 both ways

export type Convention = 'chapter' | 'cudnn';

export interface NameRow {
  /** cuDNN's symbol. */
  cudnn: string;
  /** Our wording of what it means. */
  meaning: string;
  /** The chapter's symbol, or null when the chapter's kernels have no such parameter. */
  chapter: string | null;
  /** Why there is no counterpart, when there is none. */
  none?: string;
  /** Where the name comes from: Table 19.1, the book's text on p. 472–473, or cuDNN's docs (A2). */
  from: 'Table 19.1' | 'p. 472–473' | 'cuDNN docs (A2)';
}

export const NAMES: NameRow[] = [
  { cudnn: 'N', meaning: 'images in the batch', chapter: 'N', from: 'Table 19.1' },
  { cudnn: 'C', meaning: 'input feature maps', chapter: 'C', from: 'Table 19.1' },
  { cudnn: 'H', meaning: 'input height', chapter: 'H', from: 'Table 19.1' },
  { cudnn: 'W', meaning: 'input width', chapter: 'W', from: 'Table 19.1' },
  { cudnn: 'K', meaning: 'output feature maps', chapter: 'M', from: 'Table 19.1' },
  { cudnn: 'R', meaning: 'filter height', chapter: 'K', from: 'Table 19.1' },
  { cudnn: 'S', meaning: 'filter width', chapter: 'K', from: 'Table 19.1' },
  { cudnn: 'u', meaning: 'vertical stride', chapter: null, none: 'the chapter always steps by 1', from: 'Table 19.1' },
  { cudnn: 'v', meaning: 'horizontal stride', chapter: null, none: 'the chapter always steps by 1', from: 'Table 19.1' },
  { cudnn: 'pad_h', meaning: 'zero rows added at the top and at the bottom', chapter: null, none: 'the chapter adds no zeros', from: 'Table 19.1' },
  { cudnn: 'pad_w', meaning: 'zero columns added at the left and at the right', chapter: null, none: 'the chapter adds no zeros', from: 'Table 19.1' },
  { cudnn: 'dilation', meaning: 'spacing between filter taps', chapter: null, none: 'the chapter\'s taps are adjacent', from: 'cuDNN docs (A2)' },
  { cudnn: 'P', meaning: 'output height', chapter: 'H_out', from: 'p. 472–473' },
  { cudnn: 'Q', meaning: 'output width', chapter: 'W_out', from: 'p. 472–473' },
  { cudnn: 'D', meaning: 'the input tensor, N × C × H × W', chapter: 'X', from: 'p. 472–473' },
  { cudnn: 'F', meaning: 'the filter tensor, K × C × R × S', chapter: 'F', from: 'p. 472–473' },
  { cudnn: 'O', meaning: 'the output tensor, N × K × P × Q', chapter: 'Y', from: 'p. 472–473' },
];

/** The chapter's name for a cuDNN symbol, or null when the chapter has none. */
export function toChapter(cudnn: string): string | null {
  const row = NAMES.find((r) => r.cudnn === cudnn);
  if (!row) throw new Error(`not a cuDNN name: ${cudnn}`);
  return row.chapter;
}

/** cuDNN's names for a chapter symbol (the chapter's K is two of them). */
export function toCudnn(chapter: string): string[] {
  const rows = NAMES.filter((r) => r.chapter === chapter).map((r) => r.cudnn);
  if (!rows.length) throw new Error(`not a chapter name: ${chapter}`);
  return rows;
}

/** The value each cuDNN symbol takes for a layer. */
export function cudnnValues(l: Layer): Record<string, number> {
  const { P, Q } = shape(l);
  return {
    N: l.N, C: l.C, H: l.H, W: l.W, K: l.M, R: l.K, S: l.K,
    u: l.stride, v: l.stride, pad_h: l.pad, pad_w: l.pad, dilation: l.dilation, P, Q,
  };
}

// ---------------------------------------------------------------- flashcards

export interface Card {
  id: string;
  /** The convention the symbol is shown in; the answer is in the other one. */
  from: Convention;
  symbol: string;
  options: string[];
  answer: string;
  why: string;
}

export const NO_COUNTERPART = 'no counterpart';

export const CARDS: Card[] = [
  { id: 'M', from: 'chapter', symbol: 'M', options: ['K', 'M', 'R', 'C'], answer: 'K', why: 'cuDNN calls the number of output feature maps K.' },
  { id: 'K', from: 'chapter', symbol: 'K', options: ['K', 'R and S', 'P and Q', 'u and v'], answer: 'R and S', why: 'The chapter\'s square K × K filter is R rows by S columns in cuDNN.' },
  { id: 'cK', from: 'cudnn', symbol: 'K', options: ['K', 'M', 'C', 'H_out'], answer: 'M', why: 'cuDNN\'s K counts output feature maps: the chapter\'s M, not its filter side.' },
  { id: 'R', from: 'cudnn', symbol: 'R', options: ['K', 'M', 'H', 'H_out'], answer: 'K', why: 'R is the filter height, which the chapter calls K.' },
  { id: 'P', from: 'cudnn', symbol: 'P', options: ['H_out', 'H', 'K', 'M'], answer: 'H_out', why: 'P is the output height, the chapter\'s H_out = H − K + 1 when nothing is padded or strided.' },
  { id: 'Y', from: 'chapter', symbol: 'Y', options: ['O', 'D', 'F', 'P'], answer: 'O', why: 'The book calls cuDNN\'s output tensor O and its input D.' },
  { id: 'u', from: 'cudnn', symbol: 'u', options: [NO_COUNTERPART, 'K', 'W', 'M'], answer: NO_COUNTERPART, why: 'u is the vertical stride. The chapter\'s kernels always step by 1, so they have no such parameter.' },
  { id: 'pad_h', from: 'cudnn', symbol: 'pad_h', options: [NO_COUNTERPART, 'K − 1', 'H − H_out', 'C'], answer: NO_COUNTERPART, why: 'pad_h adds zero rows at the top and the bottom. The chapter adds none: its output shrinks instead.' },
];

/** The answer a card should have, from the translation table. */
export function expectedAnswer(card: Card): string {
  if (card.from === 'chapter') return toCudnn(card.symbol).join(' and ');
  return toChapter(card.symbol) ?? NO_COUNTERPART;
}

export const grade = (card: Card, pick: string) => pick === card.answer;

// ---------------------------------------------------------------- algorithms (A3, A4, B3, B4)

export interface Algo {
  /** The enum value's name. */
  name: string;
  /** Our short label. */
  short: string;
  /** Where it sits in this chapter, or that it does not. */
  chapter: string;
  /** What the documentation says about its workspace. */
  workspace: string;
  /** HIPIFY's MIOpen counterpart (B4), or null when none is listed. Only for cuDNN. */
  hipify?: string | null;
  note?: string;
}

export const CUDNN_FWD_ALGOS: Algo[] = [
  {
    name: 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM', short: 'implicit GEMM',
    chapter: '§19.3, Fig. 19.11: B formed tile by tile, never stored', workspace: 'none mentioned',
    hipify: 'miopenConvolutionFwdAlgoImplicitGEMM',
  },
  {
    name: 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM', short: 'implicit GEMM, precomputed indices',
    chapter: 'Fig. 19.11 with the index map precomputed', workspace: 'some, for precomputed indices',
    hipify: null,
  },
  {
    name: 'CUDNN_CONVOLUTION_FWD_ALGO_GEMM', short: 'explicit GEMM',
    chapter: '§19.3, explicit unfolding (the cost on p. 467)', workspace: 'significant: holds the unfolded input',
    hipify: 'miopenConvolutionFwdAlgoGEMM',
  },
  {
    name: 'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT', short: 'direct',
    chapter: '§19.2, no matrix multiplication', workspace: 'none mentioned',
    hipify: 'miopenConvolutionFwdAlgoDirect',
  },
  {
    name: 'CUDNN_CONVOLUTION_FWD_ALGO_FFT', short: 'FFT',
    chapter: 'not in this chapter', workspace: 'significant: intermediate results',
    hipify: 'miopenConvolutionFwdAlgoFFT',
  },
  {
    name: 'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING', short: 'FFT on tiles',
    chapter: 'not in this chapter', workspace: 'less than FFT for large images',
    hipify: null,
  },
  {
    name: 'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD', short: 'Winograd',
    chapter: 'not in this chapter', workspace: 'reasonably sized',
    hipify: 'miopenConvolutionFwdAlgoWinograd',
    note: 'not supported on Hopper and later GPUs',
  },
  {
    name: 'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED', short: 'Winograd, not fused',
    chapter: 'not in this chapter', workspace: 'may be significant: intermediate results',
    hipify: null,
  },
];

/** MIOpen 3.6.2's forward algorithms with their enum values (B3). */
export const MIOPEN_FWD_ALGOS: (Algo & { value: number })[] = [
  { name: 'miopenConvolutionFwdAlgoGEMM', value: 0, short: 'GEMM', chapter: '§19.3', workspace: 'not checked here' },
  { name: 'miopenConvolutionFwdAlgoDirect', value: 1, short: 'direct', chapter: '§19.2', workspace: 'not checked here' },
  { name: 'miopenConvolutionFwdAlgoFFT', value: 2, short: 'FFT', chapter: 'not in this chapter', workspace: 'not checked here' },
  { name: 'miopenConvolutionFwdAlgoWinograd', value: 3, short: 'Winograd', chapter: 'not in this chapter', workspace: 'not checked here' },
  { name: 'miopenConvolutionFwdAlgoImplicitGEMM', value: 5, short: 'implicit GEMM', chapter: '§19.3, Fig. 19.11', workspace: 'not checked here' },
];

/** Apple exposes no algorithm choice in MPSGraph, MPSCNNConvolution or MPP (C3 to C5). */
export const APPLE_FWD_ALGOS: Algo[] = [];

// ---------------------------------------------------------------- dispatch, as read on 2026-11-20

export interface Check {
  /** The condition as the source writes it, in the chapter's names. */
  label: string;
  /** The values substituted. */
  shown: string;
  holds: boolean;
}

export type LlamaPath = 'cublas' | 'direct' | 'implicit' | 'cublas-or-direct';

export interface LlamaChoice {
  path: LlamaPath;
  pointwise: boolean;
  small: boolean;
  checks: Check[];
}

/**
 * llama.cpp, ggml-cuda/conv2d.cu (D1), shape conditions only:
 *   pointwise  = KW == 1 && KH == 1 && stride 1 && no padding   → cublasGemmEx on the input
 *   small_conv = IC·KW·KH < 64 && OW·OH < 512                   → the direct kernel
 *   otherwise                                                    → conv2d_implicit_gemm_f16
 * When both hold, the facts file does not record which the code tests first.
 */
export function llamaCuda(l: Layer): LlamaChoice {
  const { P, Q } = shape(l);
  const reduction = l.C * l.K * l.K;
  const pixels = P * Q;
  const k1 = l.K === 1;
  const s1 = l.stride === 1;
  const p0 = l.pad === 0;
  const pointwise = k1 && s1 && p0;
  const shortReduction = reduction < 64;
  const smallMap = pixels < 512;
  const small = shortReduction && smallMap;
  const path: LlamaPath = pointwise && small ? 'cublas-or-direct' : pointwise ? 'cublas' : small ? 'direct' : 'implicit';
  return {
    path,
    pointwise,
    small,
    checks: [
      { label: 'pointwise: K = 1', shown: `K = ${l.K}`, holds: k1 },
      { label: 'pointwise: stride 1', shown: `stride ${l.stride}`, holds: s1 },
      { label: 'pointwise: no padding', shown: `pad ${l.pad}`, holds: p0 },
      { label: 'small_conv: C·K·K < 64', shown: `${l.C}·${l.K}·${l.K} = ${reduction}`, holds: shortReduction },
      { label: 'small_conv: P·Q < 512', shown: `${P}·${Q} = ${pixels}`, holds: smallMap },
    ],
  };
}

export const LLAMA_PATH_TEXT: Record<LlamaPath, string> = {
  cublas: 'a cuBLAS GEMM on the input as it is (cublasGemmEx)',
  direct: 'the direct kernel, one thread per output (conv2d_kernel)',
  implicit: 'implicit GEMM (conv2d_implicit_gemm_f16), Fig. 19.11 in production',
  'cublas-or-direct': 'either the cuBLAS GEMM or the direct kernel: both conditions hold',
};

export interface Slow2dChoice {
  /** False when the explorer cannot say (a dilated layer). */
  modelled: boolean;
  requiresColumns: boolean;
  checks: Check[];
}

/**
 * PyTorch's CUDA fallback, Slow2d (ConvolutionMM2d.cu, D4):
 *   requires_columns = (kW != 1 || kH != 1 || dW != 1 || dH != 1 || padH != 0 || padW != 0)
 * When it is false the GEMM reads the input as it is. The test has no dilation; which
 * ConvBackend member serves a dilated layer is not traced in the facts file.
 */
export function slow2d(l: Layer): Slow2dChoice {
  const k = l.K !== 1;
  const s = l.stride !== 1;
  const p = l.pad !== 0;
  return {
    modelled: l.dilation === 1,
    requiresColumns: k || s || p,
    checks: [
      { label: 'K ≠ 1', shown: `K = ${l.K}`, holds: k },
      { label: 'stride ≠ 1', shown: `stride ${l.stride}`, holds: s },
      { label: 'pad ≠ 0', shown: `pad ${l.pad}`, holds: p },
    ],
  };
}

export type MlxPath = 'winograd-candidate' | 'gemm';

export interface MlxChoice {
  path: MlxPath;
  checks: Check[];
  /** The facts file does not say whether MLX's Winograd test looks at dilation. */
  dilationUnknown: boolean;
}

/**
 * MLX, backend/metal/conv.cpp, dispatch_conv_2D_gpu (D3): Winograd for 3 × 3, stride 1,
 * C % 32 == 0 and O % 32 == 0, "and large enough inputs" (not modelled). Otherwise an
 * implicit GEMM (specialised or general; the conditions are not in the facts file), with
 * an explicit unfold, tiled to fit maxBufferLength, as the last resort.
 */
export function mlx(l: Layer): MlxChoice {
  const checks: Check[] = [
    { label: '3 × 3 filter', shown: `${l.K} × ${l.K}`, holds: l.K === 3 },
    { label: 'stride 1', shown: `stride ${l.stride}`, holds: l.stride === 1 },
    { label: 'C a multiple of 32', shown: `C = ${l.C}`, holds: l.C % 32 === 0 },
    { label: 'output maps a multiple of 32', shown: `M = ${l.M}`, holds: l.M % 32 === 0 },
  ];
  const all = checks.every((c) => c.holds);
  return { path: all ? 'winograd-candidate' : 'gemm', checks, dilationUnknown: all && l.dilation !== 1 };
}

export interface Preset {
  id: string;
  label: string;
  layer: Layer;
}

const L = (N: number, C: number, H: number, W: number, M: number, K: number, pad = 0, stride = 1, dilation = 1): Layer => ({
  N, C, H, W, M, K, pad, stride, dilation,
});

/** Layers for the "describe a layer" explorer. */
export const PRESETS: Preset[] = [
  { id: 'fig19-2', label: 'The book’s example, Fig. 19.2', layer: L(1, 3, 3, 3, 2, 2) },
  { id: 'lenet-c1', label: 'LeNet C1', layer: L(1, 1, 32, 32, 6, 5) },
  { id: 'lenet-c3', label: 'LeNet C3', layer: L(1, 6, 14, 14, 16, 5) },
  { id: 'stem', label: '7 × 7, stride 2, padded, on 224 × 224', layer: L(1, 3, 224, 224, 64, 7, 3, 2) },
  { id: 'same3', label: '3 × 3, 64 channels, padded to keep the size', layer: L(8, 64, 56, 56, 64, 3, 1) },
  { id: 'dilated', label: 'Dilated 3 × 3 on 224 × 224', layer: L(1, 64, 224, 224, 64, 3, 1, 1, 2) },
  { id: 'patch', label: 'Vision patch embedding, 16 × 16, stride 16', layer: L(1, 3, 224, 224, 768, 16, 0, 16) },
  { id: 'pointwise', label: '1 × 1, 256 to 512 channels', layer: L(1, 256, 14, 14, 512, 1) },
];

/** The three layers the dispatch explorer asks you to predict, with llama.cpp's answer. */
export const DISPATCH_PRESETS: (Preset & { expect: LlamaPath })[] = [
  { id: 'lenet-c1', label: 'LeNet C1', layer: L(1, 1, 32, 32, 6, 5), expect: 'implicit' },
  { id: 'tiny', label: 'A tiny layer: C 2, 3 × 3, 16 × 16 output', layer: L(1, 2, 18, 18, 8, 3), expect: 'direct' },
  { id: 'pointwise', label: '1 × 1, 256 to 512 channels', layer: L(1, 256, 14, 14, 512, 1), expect: 'cublas' },
];

/** The three choices the dispatch prediction offers. */
export const LLAMA_GUESSES: LlamaPath[] = ['cublas', 'direct', 'implicit'];
