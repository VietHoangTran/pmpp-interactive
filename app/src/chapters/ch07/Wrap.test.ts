import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  CAUSAL,
  EX7,
  EX7_BLOCKS,
  EX7_EXTRA,
  EX7_THREADS_712,
  EX7_THREADS_715,
  EX10,
  EX10_AI,
  EX10_IDEAL,
  EX10_OUT,
  ai3d,
  questionsWrap,
} from './Wrap';
import { cxx, python, runOnEmulator } from './snippetEmu';

// The summary page shows one kernel of its own, for a question. Its checked copy is
// labs/ch07/snippets/wrap_causal_conv1d.cu, which the CUDA front-end check parses. This test
// fails if the page's copy and that file drift apart, runs a model of the kernel to pin the
// answer, runs the file itself on the course emulator through
// labs/ch07/snippets/check/wrap_causal_conv1d_test.cu (skipped when no C++ compiler or
// python3 is found), and pins the numbers the other questions state.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'Wrap.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in Wrap.tsx`);
  return m[1].trim();
};
const file = fs.readFileSync(path.join(repo, 'labs/ch07/snippets/wrap_causal_conv1d.cu'), 'utf8').trim();

/**
 * The causal kernel, block by block, as the source says: threads with i >= n return before
 * loading; the others load their own input and, for t < K - 1, one halo element; then the
 * barrier (which waits only for threads that have not exited); then the sum. Shared memory
 * nobody wrote reads as NaN. Returns y, or the reference when `fixed` (no early return).
 */
function causal(x: number[], w: number[], block: number, fixed = false): number[] {
  const K = w.length;
  const n = x.length;
  const y = new Array<number>(n).fill(NaN);
  for (let b = 0; b * block < n; b++) {
    const tile = new Array<number>(block + K - 1).fill(NaN);
    const alive = (t: number) => fixed || b * block + t < n;
    for (let t = 0; t < block; t++) {
      if (!alive(t)) continue;
      const i = b * block + t;
      tile[t + K - 1] = i < n ? x[i] : 0;
      if (t < K - 1) {
        const j = i - (K - 1);
        tile[t] = j >= 0 ? x[j] : 0;
      }
    }
    for (let t = 0; t < block; t++) {
      const i = b * block + t;
      if (!alive(t) || i >= n) continue;
      let acc = 0;
      for (let k = 0; k < K; k++) acc += w[k] * tile[t + k];
      y[i] = acc;
    }
  }
  return y;
}
const reference = (x: number[], w: number[]) =>
  x.map((_, i) => w.reduce((a, wk, k) => a + wk * (i - (w.length - 1) + k >= 0 ? x[i - (w.length - 1) + k] : 0), 0));

describe('summary page', () => {
  it('the causal kernel is the checked snippet, with the sizes the question uses', () => {
    expect(lesson('CAUSAL_CONV1D')).toBe(file);
    expect(file).toContain(`#define BLOCK ${CAUSAL.block}`);
    expect(file).toContain(`#define K ${CAUSAL.taps}`);
    expect(file).toContain('if (i >= n) return;');
  });

  it('the causal kernel is wrong exactly when the last block has 1 or 2 outputs', () => {
    const w = [0.5, -1, 2, 0.25];
    const wrong: number[] = [];
    for (let n = 1; n <= 400; n++) {
      const x = Array.from({ length: n }, (_, i) => ((i * 37) % 11) - 5);
      const want = reference(x, w);
      expect(causal(x, w, CAUSAL.block, true)).toEqual(want);
      const got = causal(x, w, CAUSAL.block);
      if (got.some((v, i) => !(Math.abs(v - want[i]) < 1e-9))) wrong.push(n);
    }
    // The emulator run below must give the same list.
    expect(wrong).toEqual([1, 2, 129, 130, 257, 258, 385, 386]);
    expect(wrong.every((n) => (CAUSAL.wrongRemainders as readonly number[]).includes(n % CAUSAL.block))).toBe(true);
    const q = questionsWrap['ch07/wrap/q-early-return'];
    expect(q.kind === 'choice' && q.options.findIndex((o) => o.correct)).toBe(2);
  });

  it('exercise 7: blocks, threads and the difference', () => {
    expect(EX7_BLOCKS).toBe(36 * 36);
    expect(EX7_THREADS_712).toBe(1024);
    expect(EX7_THREADS_715).toBe(784);
    expect(EX7_EXTRA).toBe(311040);
    expect(EX7).toEqual({ n: 1000, m: 5, t: 28 });
    const q = questionsWrap['ch07/wrap/q-ex7'];
    expect(q.kind === 'numeric' && q.answer).toBe(311040);
  });

  it('exercise 10: a 10-wide cube, a 6-wide output tile, 11.1 FLOP/B against 31.25', () => {
    expect(EX10.inTile).toBe(10);
    expect(EX10.inTile ** 3).toBeLessThanOrEqual(1024);
    expect((EX10.inTile + 1) ** 3).toBeGreaterThan(1024);
    expect(EX10_OUT).toBe(6);
    expect(EX10_AI).toBeCloseTo(54000 / 4864, 10);
    expect(EX10_AI.toFixed(1)).toBe('11.1');
    expect(EX10_IDEAL).toBe(31.25);
    // The 3D count reduces to the 2D one's shape: the ideal times an attenuation factor below 1.
    expect(ai3d(1e6, 5)).toBeCloseTo(EX10_IDEAL, 3);
    const q = questionsWrap['ch07/wrap/q-3d'];
    expect(q.kind === 'numeric' && q.answer).toBe(11.1);
  });

  it('every question names its source', () => {
    for (const q of Object.values(questionsWrap)) expect(q.source && q.source.length > 10).toBe(true);
    expect(Object.keys(questionsWrap).every((k) => k.startsWith('ch07/wrap/q-'))).toBe(true);
  });
});

describe.skipIf(!cxx || !python)('the causal kernel on the course emulator', () => {
  it('is wrong for exactly n = 1, 2, 129, 130, 257, 258, 385 and 386 in every thread order; the fix is right for every n', () => {
    const r = runOnEmulator('wrap_causal_conv1d_test.cu');
    expect(r.status, r.stderr).toBe(0);
    const lines = r.stdout.split('\n').filter((l) => /^(shown|fixed) /.test(l));
    expect(lines.length).toBe(6);
    for (const l of lines) {
      const m = /^(shown|fixed) (in order|reversed|shuffled): wrong for n =([ \d]*)$/.exec(l);
      expect(m, l).not.toBeNull();
      const wrong = m![3].trim() ? m![3].trim().split(' ').map(Number) : [];
      expect(wrong, l).toEqual(m![1] === 'shown' ? [1, 2, 129, 130, 257, 258, 385, 386] : []);
    }
    expect(r.stderr).not.toMatch(/AddressSanitizer/);
  }, 300000);
});
