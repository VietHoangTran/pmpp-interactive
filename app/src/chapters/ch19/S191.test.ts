import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { lessonRegion } from '../../sim/cnnlayer';

// Section 19.1 shows the CPU reference of lab 19.1's driver, cut out of main.cpp between
// "[lesson:cpu-reference]" and "[/lesson:cpu-reference]". The page lists how it differs
// from Figs. 19.3 and 19.4; this test fails if the file loses the region or stops being
// the seven-loop nest the page describes.
const repo = path.resolve(__dirname, '../../../..');
const MAIN = path.join(repo, 'labs/ch19/lab1_convgemm/main.cpp');

describe('the CPU reference section 19.1 shows', () => {
  it('lab 19.1 has a main.cpp with a complete cpu-reference region', () => {
    expect(fs.existsSync(MAIN)).toBe(true);
    const src = fs.readFileSync(MAIN, 'utf8');
    expect(src).toContain('[lesson:cpu-reference]');
    expect(src).toContain('[/lesson:cpu-reference]');
    const r = lessonRegion(src, 'cpu-reference');
    expect(r.found).toBe(true);
    expect(r.code.length).toBeLessThan(src.length);
  });
  it('the region is the batched loop nest: loops over n, m, h, w, c, p and q, in the book\'s order', () => {
    const { code } = lessonRegion(fs.readFileSync(MAIN, 'utf8'), 'cpu-reference');
    const at = (v: string) => {
      const m = new RegExp(`for\\s*\\(\\s*(?:int|long|size_t|int64_t|std::size_t)?\\s*${v}\\s*=`).exec(code);
      expect(m, `a for loop over ${v}`).not.toBeNull();
      return m!.index;
    };
    const order = ['n', 'm', 'h', 'w', 'c', 'p', 'q'].map(at);
    expect([...order].sort((a, b) => a - b)).toEqual(order);
  });
  it('it computes the output size and accumulates in double, as the page says', () => {
    const { code } = lessonRegion(fs.readFileSync(MAIN, 'utf8'), 'cpu-reference');
    expect(code).toMatch(/H\s*-\s*K\s*\+\s*1/);
    expect(code).toMatch(/W\s*-\s*K\s*\+\s*1/);
    expect(code).toMatch(/\bdouble\b/);
  });
});
