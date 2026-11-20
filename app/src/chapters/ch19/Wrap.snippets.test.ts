import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { C1_PRINTED, C5_DIRECT, DEPTHWISE, SCATTER_WRITERS, questionsWrap } from './Wrap';

// The summary page shows one kernel of its own, for one question. Its checked copy is
// labs/ch19/snippets/wrap_backward_scatter.cu, which the CUDA front-end check parses. This
// test fails if the page's copy and that file drift apart, checks with an independent
// count that the scatter really collides, and pins the numbers the questions take from
// the models.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'Wrap.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in Wrap.tsx`);
  return m[1].trim();
};
const file = fs.readFileSync(path.join(repo, 'labs/ch19/snippets/wrap_backward_scatter.cu'), 'utf8').trim();

/** Run the scatter kernel's loop nest for every thread and count the writers of each dX element. */
function scatterWriters(C: number, H: number, W: number, M: number, K: number): number[] {
  const Hout = H - K + 1;
  const Wout = W - K + 1;
  const writers = new Array(C * H * W).fill(0);
  for (let m = 0; m < M; m++)
    for (let pixel = 0; pixel < Hout * Wout; pixel++) {
      const touched = new Set<number>();
      const h = Math.floor(pixel / Wout);
      const w = pixel % Wout;
      for (let c = 0; c < C; c++) for (let p = 0; p < K; p++) for (let q = 0; q < K; q++) touched.add((c * H + h + p) * W + w + q);
      for (const i of touched) writers[i]++;
    }
  return writers;
}

describe('summary page, chapter 19', () => {
  it('the scatter kernel is the checked snippet', () => {
    expect(lesson('SCATTER')).toBe(file);
  });

  it('the scatter collides, as the question says, and only K = 1 with M = 1 is safe', () => {
    const w = scatterWriters(3, 3, 3, 2, 2);
    expect(Math.max(...w)).toBe(SCATTER_WRITERS.centre);
    expect(w[0]).toBe(SCATTER_WRITERS.corner);
    expect(SCATTER_WRITERS).toEqual({ centre: 8, corner: 2 });
    expect(Math.max(...scatterWriters(4, 6, 6, 1, 1))).toBe(1);
    expect(Math.max(...scatterWriters(4, 6, 6, 2, 1))).toBe(2);
  });

  it('pins the numbers the questions use', () => {
    expect(C1_PRINTED).toEqual({ pixels: 784, missed: 528 });
    expect(C5_DIRECT.outputs).toBe(120);
    expect(C5_DIRECT.threads).toBe(120 * 256);
    expect(C5_DIRECT.printedBlocks).toBe(0);
    expect(C5_DIRECT.inner).toBe(400);
    const q = questionsWrap['ch19/wrap/q-c5-direct'];
    expect(q.kind === 'numeric' && q.answer).toBe(0.39);
    expect(C5_DIRECT.pct).toBeCloseTo(100 / 256, 10);
    expect([DEPTHWISE.per.rows, DEPTHWISE.per.inner, DEPTHWISE.per.cols]).toEqual([1, 9, 144]);
    expect([DEPTHWISE.dense.rows, DEPTHWISE.dense.inner, DEPTHWISE.dense.cols]).toEqual([256, 2304, 144]);
    expect(DEPTHWISE.dense.phases).toBe(144);
  });

  it('each choice question has exactly one correct option, and every wrong one says why', () => {
    for (const q of Object.values(questionsWrap)) {
      if (q.kind !== 'choice') continue;
      expect(q.options.filter((o) => o.correct).length).toBe(1);
      for (const o of q.options) if (!o.correct) expect(o.why).toBeTruthy();
    }
  });

  it('question ids are the chapter wrap namespace and match their keys', () => {
    for (const [k, q] of Object.entries(questionsWrap)) {
      expect(q.id).toBe(k);
      expect(k.startsWith('ch19/wrap/q-')).toBe(true);
    }
  });
});
