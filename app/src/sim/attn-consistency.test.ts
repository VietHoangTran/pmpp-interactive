import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { attention as refAttention, visibleCount, type AttnProblem } from './attnref';
import { TOY, makeProblem as makeKProblem, runKernel, type KProblem } from './flashkernel';
import { BOOK, plan } from './flashplan';
import { attention as cacheAttention, makeProblem as makeCacheProblem } from './kvcache';
import { mergeStates, partialState, runOnline, SENTINEL } from './onlinesoftmax';

// Chapter 20's attention lives in four models built by different hands: attnref.ts (sections
// 20.2 and 20.3), onlinesoftmax.ts and flashkernel.ts (section 20.5) and kvcache.ts (section
// 20.4). Each is tested on its own. This file checks that they compute the same attention
// for the same inputs, so that no page teaches one answer while another page shows a
// different one, and ties the flash plan to the constants lab 20.2 compiles with.

const repo = path.resolve(__dirname, '../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');

/** A flashkernel problem (flat row-major arrays) as attnref's 2D problem, square and causal. */
function asAttn(p: KProblem): AttnProblem {
  const rows = (a: number[]) => Array.from({ length: p.n }, (_, r) => a.slice(r * p.d, (r + 1) * p.d));
  return { nq: p.n, nk: p.n, d: p.d, Q: rows(p.Q), K: rows(p.K), V: rows(p.V) };
}

describe('chapter 20: one attention, four models', () => {
  const prob = makeKProblem(TOY.n, TOY.d);
  const ref = refAttention(asAttn(prob));

  it('attnref and flashkernel use the same scale, 1/sqrt(d)', () => {
    expect(ref.scale).toBeCloseTo(prob.scaling, 15);
  });

  it("flashkernel's exact reference equals attnref's O and logsumexp", () => {
    for (let r = 0; r < prob.n; r++) {
      expect(ref.L[r]).toBeCloseTo(prob.L[r], 12);
      for (let c = 0; c < prob.d; c++) expect(ref.O[r][c]).toBeCloseTo(prob.O[r * prob.d + c], 12);
    }
  });

  it("the corrected kernel model (section 20.5's stepper) reproduces attnref's O and L", () => {
    const run = runKernel(TOY, prob, 'apart', 'ahead');
    expect(run.correct).toBe(true);
    for (let r = 0; r < prob.n; r++) {
      expect(run.L[r]).toBeCloseTo(ref.L[r], 9);
      for (let c = 0; c < prob.d; c++) expect(run.O[r * prob.d + c]).toBeCloseTo(ref.O[r][c], 9);
    }
  });

  it("the online softmax (section 20.5's first stepper) reproduces attnref's row, in every order and form", () => {
    const row = prob.n - 1;
    const scores = ref.logits[row];
    for (const order of ['ltr', 'rtl', 'shuffled'] as const)
      for (const form of ['deferred', 'normalised'] as const)
        for (const bc of [1, 3, 4, 12]) {
          const t = runOnline({ scores, V: asAttn(prob).V, bc, row, order, init: 'sentinel', form });
          // The stepper's exact answer is attnref's; its running values are rounded to float32.
          t.exact.forEach((x, c) => expect(x).toBeCloseTo(ref.O[row][c], 12));
          t.final.forEach((x, c) => expect(x).toBeCloseTo(ref.O[row][c], 6));
        }
  });

  it("two partial states merged by Eqs. 20.4 and 20.6 give attnref's row (the split-KV merge)", () => {
    const row = prob.n - 1;
    const scores = ref.logits[row];
    const V = asAttn(prob).V;
    const half = Math.floor(prob.n / 2);
    const a = partialState(scores, V, Array.from({ length: half }, (_, j) => j));
    const b = partialState(scores, V, Array.from({ length: prob.n - half }, (_, j) => half + j));
    const m = mergeStates(a, b);
    m.O.forEach((x, c) => expect(x / m.D).toBeCloseTo(ref.O[row][c], 12));
    expect(m.m + Math.log(m.D)).toBeCloseTo(ref.L[row], 12);
  });

  it("kvcache's attention (section 20.4) equals attnref's on the same Q, K and V", () => {
    const cp = makeCacheProblem(7, 4);
    const mats = cacheAttention(cp, 7);
    const rows = (m: { rows: number; cols: number; v: number[] }) =>
      Array.from({ length: m.rows }, (_, r) => m.v.slice(r * m.cols, (r + 1) * m.cols));
    const r2 = refAttention({ nq: 7, nk: 7, d: 4, Q: rows(mats.Q), K: rows(mats.K), V: rows(mats.V) });
    const P = rows(mats.P);
    const O = rows(mats.O);
    for (let r = 0; r < 7; r++)
      for (let c = 0; c < 7; c++) {
        expect(P[r][c]).toBeCloseTo(r2.P[r][c], 12);
        if (c < 4) expect(O[r][c]).toBeCloseTo(r2.O[r][c], 10);
      }
  });

  it("the decode step's single query sees every key under lab 20.1's lower-right mask", () => {
    for (const nk of [1, 2, 500, 4097]) expect(visibleCount(0, 1, nk)).toBe(nk);
  });

  it('the start value of the running max is the labs\' -FLT_MAX/2', () => {
    expect(SENTINEL).toBe(Math.fround(-3.4028234663852886e38 / 2));
    for (const lab of ['lab2_flash', 'lab4_decode', 'lab5_gqa_decode'])
      expect(read(`labs/ch20/${lab}/solution/kernel.cu`)).toMatch(/-FLT_MAX\s*\/\s*2/);
  });
});

describe("chapter 20: the flash plan against lab 20.2's constants", () => {
  const cu = read('labs/ch20/lab2_flash/solution/kernel.cu');
  const define = (name: string) => {
    const m = new RegExp(`#define ${name} (\\d+)`).exec(cu);
    if (!m) throw new Error(`no #define ${name} in lab 20.2`);
    return Number(m[1]);
  };

  it('B_r, the block size, the warp width and d are the book\'s', () => {
    expect(define('B_r')).toBe(BOOK.br);
    expect(define('BLOCK_SIZE')).toBe(BOOK.threads);
    expect(define('WARP_SIZE')).toBe(BOOK.warp);
    expect(define('HEAD_DIM')).toBe(BOOK.d);
    expect(define('LOG_NUM_BANKS')).toBe(Math.log2(BOOK.banks));
  });

  it('the tiles take 37,376 bytes at B_c = 32 and 18,688 at B_c = 16, as sections 20.5 and the lab README say', () => {
    expect(plan({ ...BOOK, bc: 32 }).bytes.total).toBe(37376);
    expect(plan({ ...BOOK, bc: 16 }).bytes.total).toBe(18688);
    expect(read('labs/ch20/lab2_flash/README.md')).toContain('37,376');
    expect(read('labs/ch20/lab2_flash/README.md')).toContain('18,688');
  });
});
