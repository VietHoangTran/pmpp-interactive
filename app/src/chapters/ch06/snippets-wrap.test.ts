import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  PERMUTE_READ,
  PERMUTE_WRITE,
  SCORES_K,
  SCORES_K4,
  SCORES_Q,
  SCORES_REMAPPED,
  SHAPE,
  alignedRows,
  questionsWrap,
  rowsAligned,
} from './Wrap';

// The summary page shows one kernel of its own, for two questions. Its checked copy is
// labs/ch06/snippets/wrap_scores.cu, which the CUDA front-end check parses. This test
// fails if the page's copy and that file drift apart, and pins the numbers the
// questions take from the coalescing model (src/sim/coalesce.ts).
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'Wrap.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in Wrap.tsx`);
  return m[1].trim();
};
const file = fs.readFileSync(path.join(repo, 'labs/ch06/snippets/wrap_scores.cu'), 'utf8').trim();

describe('summary page, chapter 6', () => {
  it('the scores kernel is the checked snippet', () => {
    expect(lesson('SCORES')).toBe(file);
  });

  it('the model reads the same K and q expressions as the kernel', () => {
    expect(file).toContain('K[(t * H + h) * D + d]');
    expect(file).toContain('q[h * D + d]');
    expect(file).toContain('int t = blockIdx.x * blockDim.x + threadIdx.x;');
  });

  it('as written, one warp touches 32 sectors of K at 12.5% and one sector of q', () => {
    expect(SHAPE).toMatchObject({ H: 32, D: 128 });
    expect(SCORES_K.transactions).toBe(32);
    expect(SCORES_K.efficiency).toBe(0.125);
    expect(SCORES_Q.transactions).toBe(1);
    const q = questionsWrap['ch06/wrap/q-scores-sectors'];
    expect(q.kind === 'numeric' && q.answer).toBe(32);
  });

  it('float4 loads still need a sector per lane; lanes along d need four', () => {
    expect(SCORES_K4.transactions).toBe(32);
    expect(SCORES_K4.efficiency).toBe(0.5);
    expect(SCORES_REMAPPED.transactions).toBe(4);
    expect(SCORES_REMAPPED.efficiency).toBe(1);
  });

  it('a permutation of the outer two dimensions reads and writes four sectors per warp', () => {
    expect(PERMUTE_WRITE.transactions).toBe(4);
    expect(PERMUTE_READ.transactions).toBe(4);
    expect(PERMUTE_READ.efficiency).toBe(1);
  });

  it('every row starts on 16 bytes exactly when K is a multiple of 4', () => {
    for (let K = 1; K <= 64; K++) expect(rowsAligned(K)).toBe(K % 4 === 0);
    expect(alignedRows(4097, 8)).toBe(2);
    expect(alignedRows(4100, 8)).toBe(8);
    expect(rowsAligned(12)).toBe(true);
  });

  it('every question has exactly one correct option or a numeric answer, under ch06/wrap/', () => {
    expect(Object.keys(questionsWrap)).toHaveLength(6);
    for (const [id, q] of Object.entries(questionsWrap)) {
      expect(q.id).toBe(id);
      expect(id.startsWith('ch06/wrap/')).toBe(true);
      if (q.kind === 'choice') expect(q.options.filter((o) => o.correct).length).toBe(1);
      else expect(Number.isFinite(q.answer)).toBe(true);
    }
  });
});
