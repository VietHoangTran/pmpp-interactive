import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { CORANK_LINES, coRankTrace } from '../../sim/corank';

// Section 13.3 shows co_rank, merge_sequential and merge_basic_kernel. The checked file is
// labs/ch13/snippets/merge_basic.cu, which ./verify.sh front-end-checks and
// labs/ch13/snippets/check.py runs on the emulator. This test fails if a lesson copy
// drifts from it, and pins every line number the lesson and the explorer rely on.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'S133.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in S133.tsx`);
  return m[1].trim();
};
const file = fs.readFileSync(path.join(repo, 'labs/ch13/snippets/merge_basic.cu'), 'utf8');
const CO_RANK = lesson('CO_RANK');
const MERGE_SEQ = lesson('MERGE_SEQ');
const MERGE_BASIC = lesson('MERGE_BASIC');
const line = (code: string, n: number) => code.split('\n')[n - 1];

describe('section 13.3 shows the checked file', () => {
  it('each of the three functions appears verbatim in the file', () => {
    for (const c of [MERGE_SEQ, CO_RANK, MERGE_BASIC]) expect(file).toContain(c);
  });
  it('the file is those three functions and comments, nothing else', () => {
    let rest = file;
    for (const c of [MERGE_SEQ, CO_RANK, MERGE_BASIC]) rest = rest.replace(c, '');
    for (const l of rest.split('\n')) expect(l.trim() === '' || l.trim().startsWith('//')).toBe(true);
  });
});

describe('the book\'s line numbering', () => {
  it('co_rank has Fig. 13.5\'s 24 lines, with the lines the explorer highlights where they should be', () => {
    expect(CO_RANK.split('\n')).toHaveLength(24);
    expect(line(CO_RANK, 2)).toContain('int i = k < m ? k : m;');
    expect(line(CO_RANK, 5)).toContain('j_low = max(0, k - m)');
    expect(line(CO_RANK, 8)).toContain('while (active)');
    expect(line(CO_RANK, 9)).toContain('if (i > 0 && j < n && A[i - 1] > B[j])');
    expect(line(CO_RANK, 10)).toContain('delta = ((i - i_low + 1) >> 1);');
    expect(line(CO_RANK, 14)).toContain('else if (j > 0 && i < m && B[j - 1] >= A[i])');
    expect(line(CO_RANK, 20)).toContain('active = false;');
    expect(line(CO_RANK, 23)).toContain('return i;');
    expect(CORANK_LINES.init).toEqual([2, 3, 4, 5]);
    expect(CORANK_LINES['i-too-high'][0]).toBe(9);
    expect(CORANK_LINES['j-too-high'][0]).toBe(14);
    expect(CORANK_LINES.done).toContain(20);
    // The model reports the same lines for the book's trace.
    expect(coRankTrace(3, [1, 7, 8, 9, 10], [7, 10, 10, 12]).steps.map((s) => s.line)).toEqual([9, 14, 20]);
  });
  it('merge_basic_kernel has Fig. 13.9\'s 11 lines; line 03 is the corrected division', () => {
    expect(MERGE_BASIC.split('\n')).toHaveLength(11);
    expect(line(MERGE_BASIC, 3)).toContain('int elementsPerThread = (m + n + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);');
    expect(line(MERGE_BASIC, 3)).not.toContain('ceil');
    expect(line(MERGE_BASIC, 4)).toContain('int k_curr =');
    expect(line(MERGE_BASIC, 5)).toContain('int k_next =');
    expect(line(MERGE_BASIC, 10)).toContain('merge_sequential(&A[i_curr]');
  });
  it('merge_sequential has Fig. 13.2\'s 21 lines and its <= tie rule on line 6', () => {
    expect(MERGE_SEQ.split('\n')).toHaveLength(21);
    expect(line(MERGE_SEQ, 6)).toContain('if (A[i] <= B[j])');
    expect(line(MERGE_SEQ, 12)).toContain('if (i == m)');
  });
  it('the printed line 03 the page quotes is the book\'s, ceil of an integer quotient', () => {
    expect(src).toContain("const LINE03_PRINTED = 'int elementsPerThread = ceil((m+n)/(blockDim.x*gridDim.x));';");
  });
});

describe('the "swap the tie tests" listing', () => {
  it('is derived from co_rank by exchanging the two comparisons, on lines 9 and 14 only', () => {
    expect(src).toContain(".replace('A[i - 1] > B[j]', 'A[i - 1] >= B[j]').replace('B[j - 1] >= A[i]', 'B[j - 1] > A[i]')");
    const swapped = CO_RANK.replace('A[i - 1] > B[j]', 'A[i - 1] >= B[j]').replace('B[j - 1] >= A[i]', 'B[j - 1] > A[i]');
    const a = CO_RANK.split('\n');
    const b = swapped.split('\n');
    const changed = a.flatMap((l, q) => (l !== b[q] ? [q + 1] : []));
    expect(changed).toEqual([9, 14]);
    expect(b[8]).toContain('A[i - 1] >= B[j]');
    expect(b[13]).toContain('B[j - 1] > A[i]');
  });
});
