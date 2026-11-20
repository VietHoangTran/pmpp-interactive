// Comparing a GPU run with a reference, for section X3.9 and the scene "A test that fails
// late". Ours: the toy model, the perturbation and the two-test design are not from any
// project. What comes from the facts (notes/today/x3-facts.md D7): llama.cpp's
// test-backend-ops compares backends by NMSE = sum (a - b)^2 / sum a^2, with a default
// threshold of 1e-7 per op; greedy decoding takes the first maximum (D1, D9).
//
// The toy model: 16 tokens, 24 steps, logits a fixed function of (previous token, step).
// The top logit is token (5·prev + 3·t + 1) mod 16 and the runner-up 7 places after it, at
// a margin that depends on the step; three steps are built with small margins. Because
// 5 is invertible mod 16, two different previous tokens never lead to the same top token,
// so once the two runs disagree they keep disagreeing. The "backend" adds s · u to every
// logit, u a fixed pseudo-random number in [-1, 1] per (step, token): independent per
// logit, and the same pattern scaled by s.

export const V = 16;
export const STEPS = 24;
export const BOS = 1;
/** llama.cpp's default per-op NMSE threshold (D7). */
export const NMSE_LLAMACPP = 1e-7;

/** The three steps whose top two logits are close, and their margins. */
export const CLOSE_STEPS: Record<number, number> = { 6: 8e-3, 13: 8e-4, 19: 6e-6 };

/** splitmix32-style hash of three integers to [0, 1). */
function hash01(a: number, b: number, c: number): number {
  let h = (Math.imul(a + 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + 0x7f4a7c15, 0xc2b2ae35) ^ Math.imul(c + 0x165667b1, 0x27d4eb2f)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h / 4294967296;
}

const MODEL_SEED = 11;
export const NOISE_SEED = 54;

export const topOf = (prev: number, t: number) => (5 * prev + 3 * t + 1) % V;
export const runnerOf = (prev: number, t: number) => (topOf(prev, t) + 7) % V;
/** The margin between the top two logits at step t (the same whatever the previous token). */
export const marginAt = (t: number) => CLOSE_STEPS[t] ?? 0.4 + 2 * hash01(MODEL_SEED, t, 777);

/** The reference logits for one step. */
export function modelLogits(prev: number, t: number): number[] {
  const top = topOf(prev, t);
  const runner = runnerOf(prev, t);
  const high = 6 + hash01(MODEL_SEED, prev * 31 + t, 99);
  return Array.from({ length: V }, (_, i) =>
    i === top ? high : i === runner ? high - marginAt(t) : -2 + 5 * hash01(MODEL_SEED, prev * 31 + t, i),
  );
}

/** The backend's fixed noise pattern, in [-1, 1], for one step. */
export const noiseAt = (t: number, seed = NOISE_SEED) => Array.from({ length: V }, (_, i) => 2 * hash01(seed, t, i + 1000) - 1);

/** The backend's logits: the reference plus s times the step's noise pattern. */
export const backendLogits = (prev: number, t: number, s: number, seed = NOISE_SEED) => {
  const u = noiseAt(t, seed);
  return modelLogits(prev, t).map((x, i) => x + s * u[i]);
};

/** Greedy: the first maximum (llama2.c's sample_argmax, torch.argmax). */
export function argmax(x: number[]): number {
  let k = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[k]) k = i;
  return k;
}

/** NMSE as test-backend-ops computes it: sum (a - b)^2 / sum a^2, a the reference. */
export function nmse(a: number[], b: number[]): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    num += (a[i] - b[i]) * (a[i] - b[i]);
    den += a[i] * a[i];
  }
  return num / den;
}

export const maxAbsErr = (a: number[], b: number[]) => a.reduce((m, x, i) => Math.max(m, Math.abs(x - b[i])), 0);

/** Top logit minus the runner-up. */
export function topTwoMargin(x: number[]): number {
  const s = [...x].sort((p, q) => q - p);
  return s[0] - s[1];
}

export interface StepRow {
  t: number;
  /** Reference run: input token, logits, chosen token, top-two margin. */
  refIn: number;
  ref: number[];
  refTok: number;
  margin: number;
  /** Teacher forced: the backend is fed the reference's input token. */
  tf: number[];
  tfTok: number;
  tfNmse: number;
  tfMaxErr: number;
  /** Free running: the backend is fed its own previous token. */
  frIn: number;
  fr: number[];
  frTok: number;
  /** NMSE of the free-running logits against the reference's (inputs differ after a divergence). */
  frNmse: number;
  /** Margin at most twice the run's largest teacher-forced error: too close to judge. */
  flagged: boolean;
}

export interface Comparison {
  s: number;
  steps: StepRow[];
  /** The tolerance the flagging uses: the largest logit error in the teacher-forced run. */
  tol: number;
  /** First step whose free-running token differs, or null. */
  firstDivergence: number | null;
  /** Steps whose teacher-forced token differs. */
  tfDiffers: number[];
  /** Steps whose free-running token differs. */
  frDiffers: number[];
  flagged: number[];
  /** Largest per-step teacher-forced NMSE. */
  maxTfNmse: number;
}

/** Run both tests for perturbation size s. */
export function compare(s: number, seed = NOISE_SEED): Comparison {
  const rows: Omit<StepRow, 'flagged'>[] = [];
  let refPrev = BOS;
  let frPrev = BOS;
  for (let t = 0; t < STEPS; t++) {
    const ref = modelLogits(refPrev, t);
    const tf = backendLogits(refPrev, t, s, seed);
    const fr = backendLogits(frPrev, t, s, seed);
    const refTok = argmax(ref);
    const row = {
      t,
      refIn: refPrev,
      ref,
      refTok,
      margin: topTwoMargin(ref),
      tf,
      tfTok: argmax(tf),
      tfNmse: nmse(ref, tf),
      tfMaxErr: maxAbsErr(ref, tf),
      frIn: frPrev,
      fr,
      frTok: argmax(fr),
      frNmse: nmse(ref, fr),
    };
    rows.push(row);
    refPrev = refTok;
    frPrev = row.frTok;
  }
  const tol = Math.max(...rows.map((r) => r.tfMaxErr));
  const steps = rows.map((r) => ({ ...r, flagged: r.margin <= 2 * tol }));
  const frDiffers = steps.filter((r) => r.frTok !== r.refTok).map((r) => r.t);
  return {
    s,
    steps,
    tol,
    firstDivergence: frDiffers.length ? frDiffers[0] : null,
    tfDiffers: steps.filter((r) => r.tfTok !== r.refTok).map((r) => r.t),
    frDiffers,
    flagged: steps.filter((r) => r.flagged).map((r) => r.t),
    maxTfNmse: Math.max(...steps.map((r) => r.tfNmse)),
  };
}

/**
 * The perturbation size above which step t's teacher-forced token flips, or null if it never
 * does. Every logit moves linearly in s, so token j overtakes the top one at
 * s = (l_top - l_j) / (u_j - u_top) when u_j > u_top; the smallest such s wins.
 */
export function flipThreshold(t: number, seed = NOISE_SEED): number | null {
  const ref = compare(0, seed).steps[t];
  const u = noiseAt(t, seed);
  const k = ref.refTok;
  let best: number | null = null;
  for (let j = 0; j < V; j++) {
    if (j === k || u[j] <= u[k]) continue;
    const s = (ref.ref[k] - ref.ref[j]) / (u[j] - u[k]);
    if (best == null || s < best) best = s;
  }
  return best;
}

/** The slider runs from 1e-7 to 1e-2 in steps of a tenth of a decade. */
export const SLIDER = { min: -7, max: -2, step: 0.1 };
/** The scene's perturbation size: one late flip. */
export const SCENE_S = 1e-4;
/** The explorer's starting perturbation size. */
export const START_S = 1e-6;
