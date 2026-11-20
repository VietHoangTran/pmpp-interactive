import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { BOOK_F, BOOK_LAYER, BOOK_X, directTrace, readsOfThread } from '../../sim/cnnlayer';

// Section 19.2 shows the "direct" region of lab 19.1's reference solution in three tabs
// and lists how it differs from Figs. 19.5 and 19.7. This test fails if a solution file
// loses its region or the corrections the page names. It also ties the numbers the
// narrated animation "What the direct kernel reads" speaks to the model it draws from.
const repo = path.resolve(__dirname, '../../../..');
const SOL = path.join(repo, 'labs/ch19/lab1_convgemm/solution');
const region = (src: string) => {
  const lines = src.split('\n');
  const a = lines.findIndex((l) => l.includes('[lesson:direct]'));
  const b = lines.findIndex((l, k) => k > a && l.includes('[/lesson:direct]'));
  return a >= 0 && b > a ? lines.slice(a + 1, b).join('\n') : null;
};
const script = (n: number) => fs.readFileSync(path.join(repo, 'content/ch19/narration', `direct-reads-${n}.txt`), 'utf8').trim();
const words = (s: string) => s.split(/\s+/).length;

describe('the direct kernel section 19.2 shows', () => {
  for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm'])
    it(`solution/${f} has a "direct" region`, () => {
      expect(fs.existsSync(path.join(SOL, f))).toBe(true);
      const src = fs.readFileSync(path.join(SOL, f), 'utf8');
      expect(src).toContain('[lesson:direct]');
      expect(src).toContain('[/lesson:direct]');
      expect(region(src)).not.toBeNull();
    });
  it('the CUDA region has a bounds check on h and w and keeps the sum in acc', () => {
    const code = region(fs.readFileSync(path.join(SOL, 'kernel.cu'), 'utf8'))!;
    expect(code).toMatch(/__global__/);
    expect(code).toMatch(/\bacc\b/);
    expect(code).toMatch(/\bh\s*<\s*\w+/);
    expect(code).toMatch(/\bw\s*<\s*\w+/);
    expect(code).toMatch(/blockIdx\.y\s*\/\s*\w+/);
    expect(code).toMatch(/blockIdx\.y\s*%\s*\w+/);
  });
  it('the Metal region takes its positions as uint3', () => {
    const code = region(fs.readFileSync(path.join(SOL, 'kernel.metal'), 'utf8'))!;
    expect(code).toMatch(/uint3\s+\w+\s*\[\[threadgroup_position_in_grid\]\]/);
    expect(code).toMatch(/\[\[thread_position_in_threadgroup\]\]/);
  });
});

describe('the narrated animation says what the model draws', () => {
  const s = [1, 2, 3, 4, 5, 6].map(script);
  const tr = directTrace(BOOK_LAYER, BOOK_X, BOOK_F, 2);
  it('six scripts of 35 to 60 words, nothing read aloud as a symbol', () => {
    for (const t of s) {
      expect(words(t)).toBeGreaterThanOrEqual(35);
      expect(words(t)).toBeLessThanOrEqual(60);
      expect(t).not.toMatch(/[<>=×%/_(){}[\]*+]/);
    }
  });
  it('part 1: 2 blocks of 4 threads', () => {
    expect(tr.blocks.length).toBe(2);
    expect(s[0]).toContain(`make ${tr.blocks.length} blocks of ${tr.tile * tr.tile} threads`);
  });
  it('part 2: 12 steps and the register climbs to 14', () => {
    const t0 = readsOfThread(tr.blocks[0], 0);
    expect(s[1]).toContain(`${t0.length} steps`);
    expect(s[1]).toContain(`climbs to ${t0[t0.length - 1].sum}`);
  });
  it('part 3: centre 4 times, edges twice, corners once; 48 reads for 27 values', () => {
    const b0 = tr.blocks[0];
    expect(b0.xHeat.slice(0, 9)).toEqual([1, 2, 1, 2, 4, 2, 1, 2, 1]);
    expect(s[2]).toContain('centre value is fetched 4 times');
    expect(s[2]).toContain(`${b0.xReads} input reads to cover only ${b0.distinctX} different values`);
  });
  it('part 4: all 4 threads read one filter address per step', () => {
    expect(tr.blocks[0].fAddressesPerStep.every((n) => n === 1)).toBe(true);
    expect(s[3]).toContain('all 4 threads of the block read the same filter weight');
  });
  it('part 5: block 1 fetches all 48 again', () => {
    expect(s[4]).toContain(`same ${tr.distinctX} input values`);
    expect(s[4]).toContain(`all ${tr.blocks[1].xReads} of them again`);
  });
  it('part 6: 96 reads for 27 values, a quarter of an operation per byte', () => {
    expect(tr.ratio).toBe(0.25);
    expect(s[5]).toContain(`${tr.xReads} input reads for ${tr.distinctX} different values`);
    expect(s[5]).toContain('a quarter of an operation per byte');
  });
});
