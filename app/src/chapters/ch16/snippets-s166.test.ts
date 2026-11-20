import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadsOf, cellAt, type Scheme } from '../../sim/wavetile';

// Section 16.6's stepper shows the load helpers of lab 16.1's reference solution as
// constants (it highlights lines in them). This test fails if those copies drift from the
// lab file, if a highlighted line stops being the tile branch of its load, or if the model
// the stepper runs on reads a different slot from the one the file's helper reads.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'S166.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in S166.tsx`);
  return m[1];
};
const file = fs.readFileSync(path.join(repo, 'labs/ch16/lab1_wavefront/solution/kernel.cu'), 'utf8');
const SQUARE = lesson('SQUARE_LOADS');
const SHEARED = lesson('SHEARED_LOADS');
const line = (code: string, n: number) => code.split('\n')[n - 1];

describe('section 16.6 shows the lab\'s load helpers verbatim', () => {
  it('square: sqLoadN, sqLoadW, sqLoadNW as in the solution file', () => {
    expect(file).toContain(SQUARE);
    expect(SQUARE).toMatch(/^__device__ inline int sqLoadN\(/);
    expect(SQUARE).toContain('sqLoadW(');
    expect(SQUARE).toContain('sqLoadNW(');
  });
  it('sheared: hyLoadN, hyLoadW, hyLoadNW as in the solution file', () => {
    expect(file).toContain(SHEARED);
    expect(SHEARED).toMatch(/^__device__ inline int hyLoadN\(/);
  });
});

describe('the lines the stepper highlights', () => {
  it('are the tile branch of each load, as LOAD_LINE says', () => {
    expect(src).toContain("square: { n: 3, w: 7, nw: 11 }");
    expect(src).toContain("sheared: { n: 3, w: 7, nw: 12 }");
    expect(line(SQUARE, 3)).toContain(': swTile[(r_tile - 1) * stride + q_tile]');
    expect(line(SQUARE, 7)).toContain(': swTile[r_tile * stride + (q_tile - 1)]');
    expect(line(SQUARE, 11)).toContain(': swTile[(r_tile - 1) * stride + (q_tile - 1)]');
    expect(line(SHEARED, 3)).toContain(': swTile[(r_tile - 1) * stride + (q_tile - 1)]');
    expect(line(SHEARED, 7)).toContain(': swTile[r_tile * stride + (q_tile - 1)]');
    expect(line(SHEARED, 12)).toContain(': swTile[(r_tile - 1) * stride + (q_tile - 2)]');
  });
});

describe('the model reads what the file reads', () => {
  // Each helper is `return (edge test) ? H[...] : swTile[(row) * stride + (col)];`. Evaluate
  // the edge test and the slot for every thread and step of a tile and compare with loadsOf.
  const helpers = (code: string) =>
    [...code.matchAll(/return \(([^)]*)\) \? H\[[^\]]*\]\s*:\s*swTile\[([^\]]*)\]/g)].map((m) => ({ test: m[1], slot: m[2] }));
  const evalIn = (expr: string, env: Record<string, number>) =>
    // eslint-disable-next-line @typescript-eslint/no-implied-eval
    Function(...Object.keys(env), `return (${expr});`)(...Object.values(env)) as number | boolean;
  for (const [scheme, code] of [['square', SQUARE], ['sheared', SHEARED]] as [Scheme, string][])
    it(`${scheme}: same edge tests and same slots, W = 6, every thread and step`, () => {
      const hs = helpers(code);
      expect(hs).toHaveLength(3);
      const W = 6;
      const stride = W; // unpadded
      const g = { Lseq: 3 * W, W, scheme };
      const steps = scheme === 'square' ? 2 * W - 1 : W;
      for (let k = 0; k < steps; k++)
        for (let t = 0; t < W; t++) {
          const c = cellAt(g, { row: 1, col: 1 }, k, t);
          if (!c.active) continue;
          const env = { r_tile: c.rTile, q_tile: c.qTile, stride };
          loadsOf(g, c).forEach((ld, i) => {
            const edge = !!evalIn(hs[i].test, env);
            expect(edge).toBe(ld.from === 'global');
            if (!edge) expect(evalIn(hs[i].slot, env)).toBe(ld.slot![0] * stride + ld.slot![1]);
          });
        }
    });
});
