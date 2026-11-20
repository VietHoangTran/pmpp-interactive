import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  ONE_PLANE,
  SHARE_2D_PCT,
  SHARE_CUBE_PCT,
  SHARE_SLAB_PCT,
  THREE_PLANES,
  TWO_PLANES_BYTES,
  questionsWrap,
  share2D,
} from './Wrap';
import { stencilIdealAI } from '../../sim/stencil3d';

// The summary page shows one kernel of its own, for a question. Its checked copy is
// labs/ch08/snippets/regtile_two_planes.cu, which the CUDA front-end check parses. This
// test fails if the page's copy and that file drift apart, checks that the kernel is the
// two-buffer variant the question describes, and pins the numbers the questions use.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'Wrap.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in Wrap.tsx`);
  return m[1].trim();
};
const file = fs.readFileSync(path.join(repo, 'labs/ch08/snippets/regtile_two_planes.cu'), 'utf8').trim();

/**
 * One 2D block of the two-plane kernel, run one thread at a time between barriers in a
 * given order (as the emulator does), on a small grid. Returns the block's outputs, or
 * the first difference from a sequential reference. Weights are all different, so a
 * stale or swapped neighbour shows. The model executes the kernel's statements: load the
 * next plane, barrier, compute from inCurr_s[b], rotate, write the other buffer.
 */
function runTwoPlanes(order: (n: number) => number[], buffers: 1 | 2): string {
  const T = 6;
  const OUT = T - 2;
  const N = 10;
  const c = [0.4, 0.11, 0.13, 0.07, 0.17, 0.05, 0.19];
  const at = (i: number, j: number, k: number) => i * N * N + j * N + k;
  const input = Array.from({ length: N * N * N }, (_, q) => Math.sin(q * 1.7) + (q % 5));
  const want = (i: number, j: number, k: number) =>
    c[0] * input[at(i, j, k)] +
    c[1] * input[at(i, j, k - 1)] + c[2] * input[at(i, j, k + 1)] +
    c[3] * input[at(i, j - 1, k)] + c[4] * input[at(i, j + 1, k)] +
    c[5] * input[at(i - 1, j, k)] + c[6] * input[at(i + 1, j, k)];
  // Block (1, 1, 0): outputs j, k in 4..7 (interior), planes 0..3.
  const bx = 1, by = 1, bz = 0;
  const iStart = bz * OUT;
  const threads = T * T;
  const shared = [new Array(T * T).fill(NaN), new Array(T * T).fill(NaN)];
  const s = (b: number) => shared[buffers === 2 ? b : 0];
  type R = { prev: number; curr: number; next: number };
  const regs: R[] = [];
  const jk = (t: number) => [by * OUT + Math.floor(t / T) - 1, bx * OUT + (t % T) - 1];
  const inside = (v: number) => v >= 0 && v < N;
  for (let t = 0; t < threads; t++) {
    const [j, k] = jk(t);
    const r: R = { prev: 0, curr: 0, next: 0 };
    if (inside(j) && inside(k) && iStart >= 1) r.prev = input[at(iStart - 1, j, k)];
    if (inside(j) && inside(k)) r.curr = input[at(iStart, j, k)];
    s(0)[t] = r.curr;
    regs.push(r);
  }
  // Phase p of thread t: from barrier p to barrier p + 1, i.e. compute plane i = iStart + p - 1,
  // rotate, write the other buffer, then load plane i + 2 for the next iteration.
  for (let t = 0; t < threads; t++) {
    const [j, k] = jk(t);
    if (inside(j) && inside(k)) regs[t].next = input[at(iStart + 1, j, k)];
  }
  for (let p = 0; p < OUT; p++) {
    const i = iStart + p;
    const b = p % 2;
    for (const t of order(threads)) {
      const [j, k] = jk(t);
      const ty = Math.floor(t / T), tx = t % T;
      const r = regs[t];
      if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1 && ty >= 1 && ty < T - 1 && tx >= 1 && tx < T - 1) {
        const cur = s(b);
        const got =
          c[0] * r.curr + c[1] * cur[ty * T + tx - 1] + c[2] * cur[ty * T + tx + 1] +
          c[3] * cur[(ty - 1) * T + tx] + c[4] * cur[(ty + 1) * T + tx] + c[5] * r.prev + c[6] * r.next;
        if (!(Math.abs(got - want(i, j, k)) < 1e-9)) return `wrong at (${i}, ${j}, ${k})`;
      }
      r.prev = r.curr;
      r.curr = r.next;
      s(1 - b)[t] = r.next;
      // The next iteration's load, before its barrier.
      if (inside(j) && inside(k) && i + 2 < N) r.next = input[at(i + 2, j, k)];
    }
  }
  return 'ok';
}
const inOrder = (n: number) => Array.from({ length: n }, (_, t) => t);
const reversed = (n: number) => inOrder(n).reverse();
const shuffled = (n: number) => inOrder(n).sort((a, b) => ((a * 7919) % 13) - ((b * 7919) % 13) || a - b);

describe('summary page', () => {
  it('the two-plane kernel is the checked snippet', () => {
    expect(lesson('TWO_PLANES')).toBe(file);
  });

  it('the snippet has one barrier, reads buffer b and writes buffer 1 - b', () => {
    const loop = file.slice(file.indexOf('for (int i'));
    expect(loop.match(/__syncthreads\(\)/g)?.length).toBe(1);
    expect(file.match(/__syncthreads\(\)/g)?.length).toBe(1);
    expect(loop).toMatch(/inCurr_s\[b\]\[ty\]\[tx - 1\]/);
    expect(loop).toMatch(/inCurr_s\[1 - b\]\[ty\]\[tx\] = inNext;/);
    expect(file).toMatch(/__shared__ float inCurr_s\[2\]\[IN_TILE_DIM\]\[IN_TILE_DIM\]/);
  });

  it('with two buffers and one barrier the block is right under every order; with one buffer it is not', () => {
    for (const o of [inOrder, reversed, shuffled]) expect(runTwoPlanes(o, 2)).toBe('ok');
    // The same loop on a single buffer is Fig. 8.12 without its second barrier.
    expect([inOrder, reversed].map((o) => runTwoPlanes(o, 1)).some((r) => r !== 'ok')).toBe(true);
  });

  it('a 32-wide 2D tile reaches 93.6% of its ideal; the 3D cube 59%, the slab as coded 90%', () => {
    expect(SHARE_2D_PCT).toBe(93.6);
    expect(share2D(32)).toBeCloseTo(1800 / 1924, 12);
    expect(SHARE_CUBE_PCT).toBe(59);
    expect(SHARE_SLAB_PCT).toBe(90);
    expect(stencilIdealAI(7)).toBe(1.625);
    const qq = questionsWrap['ch08/wrap/q-2d-tile'];
    expect(qq.kind === 'numeric' && qq.answer).toBe(93.6);
  });

  it('shared memory per 32 x 32 block: 4,096, 8,192 and 12,288 bytes for one, two and three planes', () => {
    expect([ONE_PLANE, TWO_PLANES_BYTES, THREE_PLANES]).toEqual([4096, 8192, 12288]);
  });

  it('every question has exactly one correct option or a numeric answer, and its own id', () => {
    for (const [key, q] of Object.entries(questionsWrap)) {
      expect(q.id).toBe(key);
      expect(q.id.startsWith('ch08/wrap/')).toBe(true);
      expect(q.source).toBeTruthy();
      if (q.kind === 'choice') expect(q.options.filter((o) => o.correct).length).toBe(1);
      else expect(Number.isFinite(q.answer)).toBe(true);
    }
  });
});
