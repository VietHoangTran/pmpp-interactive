import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { BOOK_SHAPE, fmas, sharedReadsV1, sharedReadsV2 } from '../../sim/gemmreads';
import { SCENE } from './OuterProductScene';
import { functionOf as functionOf153 } from './S153';
import { MM_FILES, functionOf } from './S154';

// Sections 15.3 and 15.4 show code by slicing functions out of the labs' reference
// solutions at run time (no copies to drift). This test checks that the slices exist and
// that the lab files really have the loop orders the lessons describe, and that the
// narration of "Filling a register tile with outer products" says the model's numbers.
const repo = path.resolve(__dirname, '../../../..');
const lab = (rel: string) => fs.readFileSync(path.join(repo, 'labs', rel), 'utf8');
const loopVars = (code: string) => [...code.matchAll(/for \(int (\w+) = 0;/g)].map((m) => m[1]);

describe('the functions sections 15.3 and 15.4 slice out of the labs', () => {
  const v1 = lab(MM_FILES.v1);
  const v2 = lab(MM_FILES.v2);
  it('lab 15.1 has loadTile, mm and writeTile as top-level functions', () => {
    for (const fn of ['loadTile', 'mm', 'writeTile']) {
      const s = functionOf(v1, fn);
      expect(s, fn).not.toBeNull();
      expect(s!.code.split('\n')[0]).toMatch(new RegExp(`void ${fn}\\(`));
      expect(s!.code.trimEnd().endsWith('}')).toBe(true);
      expect(v1.split('\n')[s!.firstLine - 1]).toBe(s!.code.split('\n')[0]);
    }
  });
  it('lab 15.1\'s mm is Fig. 15.6: row, col, then i innermost; both operands read from a and b', () => {
    const m = functionOf(v1, 'mm')!.code;
    expect(loopVars(m)).toEqual(['row', 'col', 'i']);
    expect(m).toContain('c[row][col] += a[row * lda + i] * b[i * ldb + col];');
    expect(m).not.toMatch(/a_r|b_r/);
  });
  it('lab 15.2\'s mm is Fig. 15.9: i outermost, a strip, a b strip, then the rank-1 update', () => {
    const m = functionOf(v2, 'mm')!.code;
    expect(loopVars(m)).toEqual(['i', 'row', 'col', 'row', 'col']);
    expect(m).toContain('a_r[row] = a[row * lda + i];');
    expect(m).toContain('b_r[col] = b[i * ldb + col];');
    expect(m).toContain('c[row][col] += a_r[row] * b_r[col];');
    expect((m.match(/#pragma unroll/g) ?? []).length).toBe(5);
  });
  it('only mm differs between the two solutions\' helpers', () => {
    for (const fn of ['clear', 'loadTile', 'writeTile']) expect(functionOf(v2, fn)?.code, fn).toBe(functionOf(v1, fn)?.code);
  });
  it('a call or a comment is not mistaken for the definition (both sections\' copies of the helper)', () => {
    for (const f of [functionOf, functionOf153]) {
      const s = f('// mm(a, b)\n    mm(x);\nstatic void mm(int a) {\n  return;\n}\n', 'mm');
      expect(s).toEqual({ code: 'static void mm(int a) {\n  return;\n}', firstLine: 3 });
      expect(f(v1, 'writeTile')).toEqual(functionOf(v1, 'writeTile'));
    }
  });
});

describe('the outer-product narration is written for the ear and says the model\'s numbers', () => {
  const script = (n: number) => fs.readFileSync(path.join(repo, `content/ch15/narration/outer-product-${n}.txt`), 'utf8').trim();
  const words = (s: string) => s.replace(/\[[^\]]*\]/g, '').split(/\s+/).filter(Boolean).length;
  it('six scripts of 35 to 60 words, with no symbols read aloud', () => {
    for (let n = 1; n <= 6; n++) {
      expect(words(script(n)), `part ${n}`).toBeGreaterThanOrEqual(35);
      expect(words(script(n)), `part ${n}`).toBeLessThanOrEqual(60);
      expect(script(n), `part ${n}`).not.toMatch(/[=×*+/<>%_]/);
    }
  });
  it('the scene runs the book\'s 8 x 8 x 8 thread', () => {
    expect(SCENE.shape).toEqual(BOOK_SHAPE);
    expect(SCENE.v2.C).toEqual(SCENE.v1.C);
  });
  it('part 1: the first output reads 8 values of A and 8 of B, 16 reads', () => {
    expect(SCENE.v1.ops.slice(0, 8).every((o) => o.kind === 'fma' && o.row === 0 && o.col === 0)).toBe(true);
    expect(script(1)).toContain('reads 8 values of A and 8 of B: 16 reads.');
  });
  it('part 2: 1024 reads for 512 multiply-adds', () => {
    expect([SCENE.readsV1, fmas(BOOK_SHAPE)]).toEqual([1024, 512]);
    expect(script(2)).toContain(`the count stands at ${SCENE.readsV1} reads for ${fmas(BOOK_SHAPE)} multiply adds.`);
  });
  it('parts 3 and 4: one step reads 16 values and feeds 64 multiply-adds', () => {
    expect(BOOK_SHAPE.tM + BOOK_SHAPE.tN).toBe(16);
    expect(script(3)).toContain('That is 16 reads from shared memory');
    expect(script(4)).toContain('All 64 cells move together.');
    expect(script(4)).toContain('16 reads from shared memory have fed 64 multiply adds.');
  });
  it('part 5: 128 against 1024, 8 times fewer', () => {
    expect([sharedReadsV2(BOOK_SHAPE), sharedReadsV1(BOOK_SHAPE) / sharedReadsV2(BOOK_SHAPE)]).toEqual([128, 8]);
    expect(script(5)).toContain(`the count is ${SCENE.readsV2} reads, against ${SCENE.readsV1} in the first order: 8 times fewer.`);
  });
});
