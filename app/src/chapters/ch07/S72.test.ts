import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { basicLineNumbers, basicTaps, firstKernel, mapThread, partOf, warpDivergence } from '../../sim/convolve';

// Section 7.2 shows lab 7.1's reference solution (the part marked "@part basic") and its
// stepper highlights lines of it found by their text. This test fails if the lab file
// loses that part or a statement the stepper looks for. It also ties the numbers the
// narrated animation speaks to the model it draws from.
const repo = path.resolve(__dirname, '../../../..');
const lab = (f: string) => fs.readFileSync(path.join(repo, 'labs/ch07/lab1_constconv/solution', f), 'utf8');
const script = (n: number) => fs.readFileSync(path.join(repo, 'content/ch07/narration', `one-thread-per-output-${n}.txt`), 'utf8').trim();
const words = (s: string) => s.split(/\s+/).length;

describe('the kernel section 7.2 shows', () => {
  it('each solution file has a basic part', () => {
    for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm']) {
      const src = lab(f);
      expect(src).toContain('// @part basic');
      expect(partOf(src, 'basic').length).toBeLessThan(src.length);
    }
  });
  it('the stepper finds every statement of the CUDA kernel exactly once', () => {
    const code = firstKernel(partOf(lab('kernel.cu'), 'basic'));
    expect(code.startsWith('__global__')).toBe(true);
    const lines = basicLineNumbers(code);
    expect(Object.keys(lines).sort()).toEqual(['accumulate', 'guard', 'inCol', 'inRow', 'init', 'outCol', 'outRow', 'store', 'test']);
    expect(lines.outCol!).toBeLessThan(lines.inRow!);
    expect(lines.inRow!).toBeLessThan(lines.test!);
    expect(lines.test!).toBeLessThan(lines.accumulate!);
    expect(lines.accumulate!).toBeLessThan(lines.store!);
  });
  it('the corrected kernel has the output guard and the linearised F and P the lesson names', () => {
    const code = partOf(lab('kernel.cu'), 'basic');
    expect(code).toMatch(/if\s*\(\s*outRow\s*>=\s*height\s*\|\|\s*outCol\s*>=\s*width\s*\)\s*return/);
    expect(code).toMatch(/F\[fRow \* \(2 \* r \+ 1\) \+ fCol\]/);
    expect(code).toMatch(/P\[outRow \* width \+ outCol\] = Pvalue/);
    expect(code).toMatch(/int inRow = outRow - r \+ fRow/);
  });
  it('the Metal kernel converts its unsigned position to int before subtracting r', () => {
    const code = partOf(lab('kernel.metal'), 'basic');
    expect(code).toMatch(/uint2 gid \[\[thread_position_in_grid\]\]/);
    expect(code).toMatch(/int outCol = int\(gid\.x\)/);
  });
});

describe('the narrated animation says what the model draws', () => {
  const s = [1, 2, 3, 4, 5, 6].map(script);
  it('six scripts of 35 to 60 words, nothing read aloud as a symbol', () => {
    for (const t of s) {
      expect(words(t)).toBeGreaterThanOrEqual(35);
      expect(words(t)).toBeLessThanOrEqual(60);
      expect(t).not.toMatch(/[<>=×%/_(){}[\]]/);
    }
  });
  it('part 2: thread 1, 1 of block 1, 1 computes P 5, 5 from a patch at row 3, column 3', () => {
    const m = mapThread({ bx: 1, by: 1, tx: 1, ty: 1, bdx: 4, bdy: 4 }, 2);
    expect(s[1]).toContain(`row ${m.outRow} and column ${m.outCol}`);
    expect(s[1]).toContain(`row ${m.cornerRow}, column ${m.cornerCol}`);
  });
  it('part 4: 16 of the corner thread\'s 25 taps are ghost cells', () => {
    expect(s[3]).toContain(`${basicTaps(16, 16, 2, 0, 0).skipped} of its 25 taps`);
  });
  it('part 6: about 6 percent of the warps of a 1024 x 1024 image with 32 x 32 blocks diverge', () => {
    const d = warpDivergence(1024, 1024, 32, 32, 2, 32);
    expect(s[5]).toContain(`about ${Math.round((d.ghost / d.warps) * 100)} percent`);
  });
});
