import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  BINS_2,
  BINS_4,
  INTENSITY,
  INTENSITY_1024,
  OPS_PER_ATOM,
  PAIRS,
  SCANNED_2,
  SCANNED_4,
  SECTORS_ADJ_ALL,
  SECTORS_ADJ_F4,
  SECTORS_ADJ_STMT,
  SECTORS_STR_ALL,
  SECTORS_STR_STMT,
  SHARE_2,
  SHARE_4,
  VOCAB,
  questionsWrap,
} from './Wrap';

// The summary page shows one kernel of its own, for a question. Its checked copy is
// labs/ch21/snippets/bias_gather.cu, which the CUDA front-end check parses. This test fails
// if the page's copy and that file drift apart, and pins the numbers the other questions
// take from the models (dcs.ts, coalesce.ts, cutoff.ts), so a model change that moves them
// is seen here first.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'Wrap.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in Wrap.tsx`);
  return m[1].trim();
};
const shown = (name: string, marker = '// ---- shown in the lesson ----\n') => {
  const parts = fs.readFileSync(path.join(repo, 'labs/ch21/snippets', name), 'utf8').split(marker);
  if (parts.length !== 2) throw new Error(`${name}: expected one marker line`);
  return parts[1].trim();
};

describe('the bias kernel on the summary page', () => {
  it('is the part of labs/ch21/snippets/bias_gather.cu below its marker', () =>
    expect(lesson('BIAS')).toBe(shown('bias_gather.cu')));
  it('is the gather form the question describes: one writer per logit, every pair scanned, no atomic', () => {
    const k = lesson('BIAS');
    expect(k).toContain('if (v >= vocab) return;');
    expect(k).toContain('for (int e = 0; e < n; ++e)');
    expect(k).toContain('if (ids[e] == v) add += bias[e];');
    expect(k).toContain('logits[v] += add;');
    expect(k).not.toMatch(/atomic(Add|_ref)\s*\(/);
  });
  it('the question counts every pair', () => {
    expect(VOCAB * PAIRS).toBe(6_400_000);
  });
});

describe('the numbers the mixed check takes from the models', () => {
  it('gather: 10 arithmetic operations per atom per point, 80 per byte at chunk 64, 1,280 at 1,024', () => {
    expect(OPS_PER_ATOM).toBe(10);
    expect(INTENSITY).toBe(80);
    expect(INTENSITY_1024).toBe(1280);
    expect(questionsWrap['ch21/wrap/q-intensity']).toMatchObject({ kind: 'numeric', answer: 80 });
  });
  it('coarsening by 8, NVIDIA sectors: 32 per adjacent statement, 256 for eight; strided 4 and 32', () => {
    expect(SECTORS_ADJ_STMT).toBe(32);
    expect(SECTORS_ADJ_ALL).toBe(256);
    expect(SECTORS_STR_STMT).toBe(4);
    expect(SECTORS_STR_ALL).toBe(32);
    expect(SECTORS_ADJ_F4).toBe(16);
    expect(questionsWrap['ch21/wrap/q-factor8']).toMatchObject({ kind: 'numeric', answer: 256 });
  });
  it('the book\'s 3D case: 461 bins and about 25 % accepted at 4 Å; 1,863 and about 49 % at 2 Å', () => {
    expect(BINS_4).toBe(461);
    expect(SHARE_4).toBe(25);
    expect(BINS_2).toBe(1863);
    expect(SHARE_2).toBe(49);
    expect(SCANNED_4).toBe(29_504);
    expect(SCANNED_2).toBe(14_904);
  });
  it('every choice question has exactly one correct option and a reason for every other one', () => {
    for (const q of Object.values(questionsWrap)) {
      if (q.kind !== 'choice') continue;
      expect(q.options.filter((o) => o.correct).length).toBe(1);
      expect(q.options.filter((o) => !o.correct).every((o) => o.why)).toBe(true);
      expect(q.source).toBeTruthy();
    }
  });
});
