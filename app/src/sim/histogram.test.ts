import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  PIXELS,
  PRESETS,
  SIDE,
  WIDTHS,
  binCount,
  binOf,
  binRange,
  flatIndex,
  generateImage,
  histogram,
  hottest,
  preset,
} from './histogram';

describe('binning (p. 203: divide by the interval width)', () => {
  it('width 64 gives the book\'s four intervals: 63 -> 0, 64 -> 1, 191 -> 2, 192 -> 3, 255 -> 3', () => {
    expect([0, 63, 64, 127, 128, 191, 192, 255].map((v) => binOf(v, 64))).toEqual([0, 0, 1, 1, 2, 2, 3, 3]);
    expect(binCount(64)).toBe(4);
  });
  it('the explorer\'s widths give 4, 8, 16 and 256 bins', () => {
    expect(WIDTHS.map(binCount)).toEqual([4, 8, 16, 256]);
  });
  it('width 1 is one bin per value, as Fig. 9.2 assumes', () => {
    expect(binOf(200, 1)).toBe(200);
    expect(binRange(200, 1)).toEqual([200, 200]);
  });
  it('bin ranges cover 0..255 without gaps', () => {
    for (const w of WIDTHS) {
      const r = Array.from({ length: binCount(w) }, (_, b) => binRange(b, w));
      expect(r[0][0]).toBe(0);
      expect(r[r.length - 1][1]).toBe(255);
      for (let b = 1; b < r.length; b++) expect(r[b][0]).toBe(r[b - 1][1] + 1);
    }
  });
  it('rejects values that are not bytes', () => {
    expect(() => binOf(256, 1)).toThrow();
    expect(() => binOf(-1, 1)).toThrow();
  });
});

describe('the presets (ours, never the book\'s tree)', () => {
  it('every preset is 8 x 8 bytes', () => {
    for (const p of PRESETS) {
      expect(p.values).toHaveLength(PIXELS);
      expect(p.values.every((v) => Number.isInteger(v) && v >= 0 && v <= 255)).toBe(true);
    }
  });
  it('the bars sum to the pixel count for every preset and every width', () => {
    for (const p of PRESETS) for (const w of WIDTHS) expect(histogram(p.values, w).reduce((s, c) => s + c, 0)).toBe(64);
  });
  it('the sky preset is skewed: 31 of 64 pixels in bin 255, 35 in the top interval of 64', () => {
    const sky = preset('sky').values;
    expect(histogram(sky, 1)[255]).toBe(31);
    expect(histogram(sky, 64)).toEqual([15, 9, 5, 35]);
    expect(hottest(histogram(sky, 1))).toEqual({ bin: 255, count: 31, share: 31 / 64 });
  });
  it('the flat preset puts every pixel in one bin', () => {
    expect(hottest(histogram(preset('flat').values, 1))).toEqual({ bin: 128, count: 64, share: 1 });
  });
  it('the night preset is dark: bin 0 of width 64 holds 55 pixels, the moon and stars 7', () => {
    expect(histogram(preset('night').values, 64)).toEqual([55, 2, 0, 7]);
  });
  it('the noise preset is the same on every run', () => {
    expect(preset('noise').values.slice(0, 4)).toEqual(PRESETS.find((p) => p.id === 'noise')!.values.slice(0, 4));
    expect(new Set(preset('noise').values).size).toBeGreaterThan(50);
  });
});

describe('flat indices are row-major (chapter 3)', () => {
  it('(0,0) is 0, (0,7) is 7, (1,0) is 8, (7,7) is 63', () => {
    expect([flatIndex(0, 0, SIDE), flatIndex(0, 7, SIDE), flatIndex(1, 0, SIDE), flatIndex(7, 7, SIDE)]).toEqual([0, 7, 8, 63]);
  });
  it('is row * width + col for other widths', () => {
    expect(flatIndex(3, 5, 1920)).toBe(3 * 1920 + 5);
  });
  it('refuses a column outside the row', () => {
    expect(() => flatIndex(0, 8, 8)).toThrow();
  });
});

describe('generated images', () => {
  it('are seeded and repeatable', () => {
    expect(generateImage(1000, 'uniform', 3)).toEqual(generateImage(1000, 'uniform', 3));
  });
  it('constant: one bin holds everything', () => {
    expect(hottest(histogram(generateImage(500, 'constant'), 1)).share).toBe(1);
  });
  it('bright: about half of the pixels are 255, uniform: none near that', () => {
    expect(hottest(histogram(generateImage(20000, 'bright', 5), 1)).share).toBeGreaterThan(0.43);
    expect(hottest(histogram(generateImage(20000, 'uniform', 5), 1)).share).toBeLessThan(0.01);
  });
});

// ------------------------------------------------------------------ the checked file
// labs/ch09/snippets/histogram_sequential.cpp is what section 9.1 shows. Compile it with
// a small driver, run it, and require the same 256 counts as histogram(values, 1).
const repo = path.resolve(__dirname, '../../..');
const snippetDir = path.join(repo, 'labs/ch09/snippets');

describe('labs/ch09/snippets/histogram_sequential.cpp still contains the loop the model mirrors', () => {
  const src = fs.readFileSync(path.join(snippetDir, 'histogram_sequential.cpp'), 'utf8');
  it('is Fig. 9.2 with const on the image', () => {
    expect(src).toContain('void histogram_sequential(const unsigned char* image, unsigned int* bins,');
    expect(src).toContain('for(unsigned int i = 0; i < width*height; ++i) {');
    expect(src).toContain('unsigned char b = image[i];');
    expect(src).toContain('++bins[b];');
  });
});

const compiler = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
describe.skipIf(!compiler)('histogram_sequential.cpp, compiled and run', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pmpp-ch09-'));
  const exe = path.join(dir, 'histogram_sequential_test');
  const run = (w: number, h: number, values: number[]) => {
    const r = spawnSync(exe, [w, h, ...values].map(String), { encoding: 'utf8' });
    expect(r.status).toBe(0);
    return r.stdout.trim().split('\n').map(Number);
  };
  it('compiles without warnings', () => {
    const r = spawnSync(compiler!, ['-std=c++17', '-Wall', '-Wextra', '-Werror', '-O2', path.join(snippetDir, 'check/histogram_sequential_test.cpp'), '-o', exe], { encoding: 'utf8' });
    expect(r.stderr).toBe('');
    expect(r.status).toBe(0);
  });
  it('counts every preset exactly as histogram(values, 1) does', () => {
    for (const p of PRESETS) expect(run(SIDE, SIDE, p.values)).toEqual(histogram(p.values, 1));
  });
  it('and a seeded 40 x 25 image of each kind', () => {
    for (const kind of ['uniform', 'bright', 'constant'] as const) {
      const img = generateImage(1000, kind, 11);
      expect(run(40, 25, img)).toEqual(histogram(img, 1));
    }
  });
});
