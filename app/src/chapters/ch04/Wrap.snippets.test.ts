import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  AFTER_PACK,
  BEFORE_PACK,
  CAP_AFTER,
  CAP_BEFORE,
  NEEDED_AFTER,
  NEEDED_BEFORE,
  ON_BOOK,
  ON_SMALLER,
  ROW_LIMITS,
  SMALL_GRID,
  WARPS_IN_GRID,
  divergentWarps,
  questionsWrap,
} from './Wrap';
import { waves } from '../../sim/blocksched';
import { pct } from '../../sim/occupancy';

// The summary page shows one kernel of its own, for a question. Its checked copy is
// labs/ch04/snippets/warp_tail_sum.cu, which the CUDA front-end check parses. This test
// fails if the page's copy and that file drift apart, and pins the numbers the other
// questions take from the models.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'Wrap.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in Wrap.tsx`);
  return m[1].trim();
};
const file = fs.readFileSync(path.join(repo, 'labs/ch04/snippets/warp_tail_sum.cu'), 'utf8').trim();

/**
 * The six warp lines of blockSum on 256 values, run two ways: in lockstep (every
 * thread of the group reads, then every thread writes) and one thread at a time in a
 * given order with no synchronisation. Returns s[0].
 */
function tailSum(values: number[], order: 'lockstep' | 'descending'): number {
  const s = [...values];
  for (let stride = 128; stride > 32; stride /= 2) for (let t = 0; t < stride; t++) s[t] += s[t + stride];
  for (const d of [32, 16, 8, 4, 2, 1]) {
    if (order === 'lockstep') {
      const read = Array.from({ length: 32 }, (_, t) => s[t + d]);
      for (let t = 0; t < 32; t++) s[t] += read[t];
    } else {
      for (let t = 31; t >= 0; t--) s[t] += s[t + d];
    }
  }
  return s[0];
}

describe('summary page', () => {
  it('the warp-tail kernel is the checked snippet', () => {
    expect(lesson('WARP_TAIL_SUM')).toBe(file);
  });

  it('the six warp lines are right in lockstep and wrong when one thread runs ahead of another', () => {
    const values = Array.from({ length: 256 }, (_, i) => (i % 7) + 1);
    const total = values.reduce((a, b) => a + b, 0);
    expect(tailSum(values, 'lockstep')).toBe(total);
    expect(tailSum(values, 'descending')).not.toBe(total);
    // The kernel's six offsets are the ones simulated, and nothing synchronises them.
    const body = file.slice(file.indexOf('if (t < 32)'), file.indexOf('if (t == 0)'));
    expect([...body.matchAll(/v\[t \+ (\d+)\]/g)].map((m) => Number(m[1]))).toEqual([32, 16, 8, 4, 2, 1]);
    expect(body).not.toMatch(/sync/);
  });

  it('a row test on a 16 x 8 tile: only K = 4 splits no warp at widths 32 and 64', () => {
    const at = (k: number) => [divergentWarps(k, 32), divergentWarps(k, 64)];
    expect(ROW_LIMITS.map(at)).toEqual([
      [0, 1],
      [0, 0],
      [0, 1],
    ]);
    // "Any K" is wrong: an odd K splits a warp even at 32.
    expect(divergentWarps(3, 32)).toBe(1);
    const q = questionsWrap['ch04/wrap/q-rowmask'];
    expect(q.kind === 'choice' && q.options.findIndex((o) => o.correct)).toBe(2);
  });

  it('four times the work per load cuts the warps needed from 16 to 5', () => {
    expect(BEFORE_PACK).toEqual({ work: 2, latency: 30 });
    expect(AFTER_PACK).toEqual({ work: 8, latency: 30 });
    expect(NEEDED_BEFORE).toBe(16);
    expect(NEEDED_AFTER).toBe(5);
    const q = questionsWrap['ch04/wrap/q-pack'];
    expect(q.kind === 'numeric' && q.answer).toBe(5);
  });

  it('64 blocks on room for 80, then 160: under one wave either way, 512 warps in all', () => {
    expect(SMALL_GRID.blocks).toBe(64);
    expect([CAP_BEFORE, CAP_AFTER]).toEqual([80, 160]);
    expect([waves(64, CAP_BEFORE), waves(64, CAP_AFTER)]).toEqual([0.8, 0.4]);
    expect(WARPS_IN_GRID).toBe(512);
  });

  it('40 registers at 256 threads: 6 blocks and 48 warps on both SMs, 75% and 100%', () => {
    expect([ON_BOOK.blocks, ON_BOOK.residentWarps, ON_BOOK.warpSlots, pct(ON_BOOK.byWarps)]).toEqual([6, 48, 64, 75]);
    expect(ON_BOOK.binding).toEqual(['regs']);
    expect([ON_SMALLER.blocks, ON_SMALLER.residentWarps, ON_SMALLER.warpSlots, pct(ON_SMALLER.byWarps)]).toEqual([6, 48, 48, 100]);
    expect(ON_SMALLER.binding).toContain('regs');
  });

  it('every question has exactly one correct option or a numeric answer', () => {
    for (const q of Object.values(questionsWrap)) {
      if (q.kind === 'choice') expect(q.options.filter((o) => o.correct).length).toBe(1);
      else expect(Number.isFinite(q.answer)).toBe(true);
      expect(q.id.startsWith('ch04/wrap/')).toBe(true);
    }
  });
});
