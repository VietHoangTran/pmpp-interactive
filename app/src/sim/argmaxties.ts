// Argmax tie rules in the projects module X3 reads (section X3.6), and where each sampling
// stage runs. Everything here is our transcription of source read in the facts passes:
//
//   llama.cpp CUDA   ggml-cuda/argmax.cu: a pair is replaced only if (val > maxval), in the
//                    strided loop and in the xor butterfly (notes/today/ch10-facts.md D2)
//   llama.cpp Metal  kernels/misc.metal: simd_max of the values, then simd_max of the indices
//                    of the lanes holding that maximum (ch10 D2)
//   llama.cpp host   src/llama-sampler.cpp greedy: a strict > scan from index 0 (ch10 D2)
//   torch.argmax     documented "the indices of the first maximal value"; comparator: NaN
//                    wins, lower index on equal values (ch10 D4)
//   course M4        lab 10.3 argmaxCombine: otherV > v || (otherV == v && otherI < i), with
//                    the identity pair (-FLT_MAX, INT_MAX)
//
// The model is one warp and one column per lane, so a row has exactly WARP columns. The
// strided loop over longer rows and the second stage across warps are left out.

export const WARP = 32;
/** The butterfly's offsets in the order the CUDA kernel visits them. */
export const OFFSETS = [16, 8, 4, 2, 1] as const;
export const FLT_MAX = 3.4028234663852886e38;
export const INT_MAX = 2147483647;

export interface Pair {
  v: number;
  i: number;
}

export interface ButterflyStage {
  /** null for the state after the per-lane load, else the xor offset just applied. */
  offset: number | null;
  lanes: Pair[];
  /** Lanes whose pair changed at this stage. */
  changed: number[];
}

/** Each lane's pair after its share of the row: one column per lane, kept only if val > -FLT_MAX. */
function cudaLoad(row: number[]): Pair[] {
  return Array.from({ length: WARP }, (_, lane) => {
    const p: Pair = { v: -FLT_MAX, i: -1 };
    for (let c = lane; c < row.length; c += WARP) if (row[c] > p.v) { p.v = row[c]; p.i = c; }
    return p;
  });
}

/** llama.cpp CUDA argmax_f32 for one warp: every lane's (value, index) after the load and after each offset. */
export function cudaButterfly(row: number[]): ButterflyStage[] {
  if (row.length !== WARP) throw new Error(`the model takes a row of ${WARP} columns`);
  let lanes = cudaLoad(row);
  const stages: ButterflyStage[] = [{ offset: null, lanes, changed: [] }];
  for (const off of OFFSETS) {
    const prev = lanes;
    lanes = prev.map((own, lane) => {
      const other = prev[lane ^ off];
      return other.v > own.v ? { ...other } : { ...own };
    });
    const changed = lanes.flatMap((p, k) => (p.i !== prev[k].i || p.v !== prev[k].v ? [k] : []));
    stages.push({ offset: off, lanes, changed });
  }
  return stages;
}

/** The index llama.cpp's CUDA kernel writes: lane 0's after the butterfly (ch10 D2). -1 if nothing beat -FLT_MAX. */
export function cudaArgmax(row: number[]): number {
  const st = cudaButterfly(row);
  return st[st.length - 1].lanes[0].i;
}

/**
 * llama.cpp Metal kernel_argmax_f32 for one SIMD-group of 32, one column per lane:
 * max_val = simd_max(lmax); arg = simd_max(select(-1, larg, lmax == max_val)).
 * The per-lane start (-INFINITY, -1) with a strict > is our assumption; the facts pass quotes only
 * the two simd_max lines. NaN handling in simd_max is undocumented (ch10 C1), so NaN is skipped by
 * the per-lane test before it reaches simd_max.
 */
export function metalLanes(row: number[]): Pair[] {
  if (row.length !== WARP) throw new Error(`the model takes a row of ${WARP} columns`);
  return row.map((x, lane) => (x > -Infinity ? { v: x, i: lane } : { v: -Infinity, i: -1 }));
}
export function metalArgmax(row: number[]): number {
  const lanes = metalLanes(row);
  const max = Math.max(...lanes.map((p) => p.v));
  return Math.max(...lanes.map((p) => (p.v === max ? p.i : -1)));
}
/** The lanes whose value equals the SIMD-group maximum: the ones simd_max over indices looks at. */
export function metalMaxLanes(row: number[]): number[] {
  const lanes = metalLanes(row);
  const max = Math.max(...lanes.map((p) => p.v));
  return lanes.flatMap((p, k) => (p.v === max && p.i >= 0 ? [k] : []));
}

/** llama.cpp host greedy: selected = 0, then i replaces it only if logit[i] > logit[selected]. */
export function hostGreedy(row: number[]): number {
  let s = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[s]) s = i;
  return s;
}

/** torch.argmax by its comparator: NaN wins (the lowest NaN index), else the first maximum. */
export function torchArgmax(row: number[]): number {
  let bi = 0;
  for (let i = 1; i < row.length; i++) {
    const a = row[i];
    const b = row[bi];
    const better = Number.isNaN(a) ? !Number.isNaN(b) : Number.isNaN(b) ? false : a > b;
    if (better) bi = i;
  }
  return bi;
}

/** Course M4 (lab 10.3): combine with the identity (-FLT_MAX, INT_MAX); the result does not depend on order. */
export function m4Argmax(row: number[]): number {
  let v = -FLT_MAX;
  let i = INT_MAX;
  row.forEach((x, c) => {
    if (x > v || (x === v && c < i)) { v = x; i = c; }
  });
  return i;
}

export type Impl = 'cuda' | 'metal' | 'host' | 'torch' | 'm4';
export const IMPLS: { id: Impl; label: string; rule: string; cite: string }[] = [
  { id: 'cuda', label: 'llama.cpp CUDA (argmax.cu, also the HIP file)', rule: "lane 0's pair after the xor butterfly; a pair is replaced only by a strictly larger value", cite: 'ch10 D2' },
  { id: 'metal', label: 'llama.cpp Metal (kernel_argmax_f32)', rule: 'the largest index among the lanes that hold the maximum', cite: 'ch10 D2' },
  { id: 'host', label: 'llama.cpp host greedy sampler', rule: 'the lowest index: a strict > scan from index 0', cite: 'ch10 D2' },
  { id: 'torch', label: 'torch.argmax (transformers greedy)', rule: 'documented: the first maximal value', cite: 'ch10 D4, x1 D4' },
  { id: 'm4', label: "the course's M4 (lab 10.3)", rule: 'the lowest index among the maxima, by its combine rule', cite: 'lab 10.3 argmaxCombine' },
];
export function argmaxBy(impl: Impl, row: number[]): number {
  switch (impl) {
    case 'cuda': return cudaArgmax(row);
    case 'metal': return metalArgmax(row);
    case 'host': return hostGreedy(row);
    case 'torch': return torchArgmax(row);
    case 'm4': return m4Argmax(row);
  }
}

/**
 * The row the tie stepper shows: 32 distinct values below TIE_VALUE, and TIE_VALUE at every planted
 * column. Values are a fixed permutation so that no two unplanted columns tie.
 */
export const TIE_VALUE = 4;
export function tieRow(planted: number[]): number[] {
  const row = Array.from({ length: WARP }, (_, c) => ((c * 13) % WARP) / 10);
  for (const c of planted) row[c] = TIE_VALUE;
  return row;
}

// ---------------------------------------------------------------- where each stage runs

export type Stage = 'penalties' | 'top-k' | 'top-p' | 'temperature' | 'draw';
export const STAGE_LIST: Stage[] = ['penalties', 'top-k', 'top-p', 'temperature', 'draw'];
export type System = 'llama-default' | 'llama-bs' | 'transformers' | 'mlx-lm';
export const SYSTEMS: { id: System; label: string }[] = [
  { id: 'llama-default', label: 'llama.cpp, defaults' },
  { id: 'llama-bs', label: 'llama.cpp with -bs' },
  { id: 'transformers', label: 'transformers generate' },
  { id: 'mlx-lm', label: 'mlx-lm' },
];
export type Bin = 'host' | 'kernels' | 'framework';
export const BINS: { id: Bin; label: string }[] = [
  { id: 'host', label: 'host CPU code' },
  { id: 'kernels', label: "the project's GPU graph (one backend kernel per op)" },
  { id: 'framework', label: "framework tensor ops on the logits' device" },
];

export interface Cell {
  system: System;
  stage: Stage;
  bin: Bin;
  /** Facts item that settles it. */
  cite: string;
  /** What the source shows, in one line. */
  evidence: string;
}

/**
 * Only cells a facts item settles; every one of the 4 × 5 here is settled. A stage or system added
 * later goes in only with the item that settles it, never by analogy.
 */
export const STAGES: Cell[] = [
  { system: 'llama-default', stage: 'penalties', bin: 'host', cite: 'x1 D1, D2; ch09 D4', evidence: 'backend_sampling = false; llama_sampler_penalties_apply walks the candidate array on the host' },
  { system: 'llama-default', stage: 'top-k', bin: 'host', cite: 'x1 D1; ch14 D3', evidence: 'host top-k is a partial sort of the candidate array' },
  { system: 'llama-default', stage: 'top-p', bin: 'host', cite: 'x1 D1; ch14 D3', evidence: 'host top-p: softmax, adaptive partial sort, running sum' },
  { system: 'llama-default', stage: 'temperature', bin: 'host', cite: 'x1 D1, D2', evidence: 'llama_sampler_temp_impl divides each logit by temp' },
  { system: 'llama-default', stage: 'draw', bin: 'host', cite: 'x1 D3', evidence: 'dist: std::mt19937 and a running sum compared with u × total' },
  { system: 'llama-bs', stage: 'penalties', bin: 'kernels', cite: 'ch09 D4', evidence: 'host counts uploaded, scattered with ggml_set_rows and gathered with ggml_get_rows' },
  { system: 'llama-bs', stage: 'top-k', bin: 'kernels', cite: 'ch14 D3, D2', evidence: 'backend top-k calls ggml_top_k (top-k.cu, Metal argsort.metal)' },
  { system: 'llama-bs', stage: 'top-p', bin: 'kernels', cite: 'ch14 D3; ch11 D2', evidence: 'ggml_argsort, ggml_soft_max, ggml_cumsum, a ggml_step mask' },
  { system: 'llama-bs', stage: 'temperature', bin: 'kernels', cite: 'x1 D2', evidence: 'ggml_scale(ctx, logits, 1.0f / temp)' },
  { system: 'llama-bs', stage: 'draw', bin: 'kernels', cite: 'x1 D3', evidence: 'ggml_soft_max, ggml_cumsum, ggml_step, ggml_sum; u still drawn on the host' },
  { system: 'transformers', stage: 'penalties', bin: 'framework', cite: 'x1 D4', evidence: 'RepetitionPenaltyLogitsProcessor: torch.where on the scores' },
  { system: 'transformers', stage: 'top-k', bin: 'framework', cite: 'x1 D4', evidence: 'TopKLogitsWarper: torch.topk and a mask' },
  { system: 'transformers', stage: 'top-p', bin: 'framework', cite: 'x1 D4; ch11 D4', evidence: 'TopPLogitsWarper, tensor ops' },
  { system: 'transformers', stage: 'temperature', bin: 'framework', cite: 'x1 D4', evidence: 'TemperatureLogitsWarper: scores / temperature' },
  { system: 'transformers', stage: 'draw', bin: 'framework', cite: 'x1 D4, D5', evidence: 'torch.multinomial(probs, 1): argmax(p / q) on the tensor’s device' },
  { system: 'mlx-lm', stage: 'penalties', bin: 'framework', cite: 'x1 D8', evidence: 'mx.where(selected < 0, selected * penalty, selected / penalty)' },
  { system: 'mlx-lm', stage: 'top-k', bin: 'framework', cite: 'x1 D8; ch12 D6', evidence: 'mx.argpartition and mx.put_along_axis' },
  { system: 'mlx-lm', stage: 'top-p', bin: 'framework', cite: 'x1 D8; ch11 D3', evidence: 'mx.where(logprobs < threshold, -inf, logprobs)' },
  { system: 'mlx-lm', stage: 'temperature', bin: 'framework', cite: 'x1 D8', evidence: 'logits * (1.0 / temp) inside categorical_sampling' },
  { system: 'mlx-lm', stage: 'draw', bin: 'framework', cite: 'x1 D7, D8', evidence: 'mx.random.categorical with a Threefry key' },
];
export const cellOf = (system: System, stage: Stage) => STAGES.find((c) => c.system === system && c.stage === stage);

/** The order each system applies its stages in (x1 D1, D4, D8): the reason the table's columns are not one pipeline. */
export const ORDER: Partial<Record<System, string[]>> = {
  'llama-default': ['penalties', 'DRY', 'top-n-sigma', 'top-k', 'typical', 'top-p', 'min-p', 'XTC', 'temperature', 'dist or greedy'],
  transformers: ['repetition penalty and other processors', 'temperature', 'top-k', 'top-p', 'min-p', 'softmax', 'torch.multinomial'],
  'mlx-lm': ['logits processors (penalties)', 'top-p', 'min-p', 'XTC', 'top-k', 'temperature and categorical'],
};
