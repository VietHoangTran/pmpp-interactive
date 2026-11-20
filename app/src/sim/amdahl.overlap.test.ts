// Chapter 22's additions to amdahl.ts: overlap and modules (§22.3, pp. 537–538).
// Chapter 1's tests stay in amdahl.test.ts, untouched.
import { describe, expect, it } from 'vitest';
import {
  DECODE_STEP,
  MD_STEP,
  ceiling,
  moduleSpeedup,
  moduleTime,
  overlapped,
  rebalance,
  speedup,
  type AppModule,
} from './amdahl';

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const place = (ms: AppModule[], i: number, where: 'host' | 'device') => ms.map((m, j) => (j === i ? { ...m, place: where } : m));
const withS = (ms: AppModule[], s: Record<number, number>) => ms.map((m, j) => (j in s ? { ...m, s: s[j] } : m));

describe("the book's worked example (pp. 537–538; ch22-facts A9)", () => {
  it('95% made 100 times faster: 1 / 5.95% = 16.807, printed as 17x', () => {
    expect(speedup(0.95, 100)).toBeCloseTo(16.807, 3);
    expect(Math.round(speedup(0.95, 100))).toBe(17);
  });
  it('with host and device overlapped the device hides inside the host: 20x, which is the ceiling', () => {
    expect(overlapped(0.95, 100)).toBeCloseTo(20, 12);
    expect(overlapped(0.95, 100)).toBeCloseTo(ceiling(0.95), 12);
  });
  it('the preset modules reproduce both numbers', () => {
    expect(sum(MD_STEP.map((m) => m.share))).toBeCloseTo(1, 12);
    const plain = moduleSpeedup(MD_STEP, false);
    expect(plain.speedup).toBeCloseTo(16.807, 3);
    expect(plain.total).toBeCloseTo(0.0595, 12);
    expect(plain.host).toBeCloseTo(0.05, 12);
    expect(plain.device).toBeCloseTo(0.0095, 12);
    expect(plain.ceiling).toBeCloseTo(20, 9);
    const both = moduleSpeedup(MD_STEP, true);
    expect(both.speedup).toBeCloseTo(20, 9);
    expect(both.hidden).toBeCloseTo(0.0095, 12); // the whole kernel is hidden
    expect(both.chained).toBeCloseTo(0.03, 12); // neighbour list and update
  });
  it('the 20x does not depend on how the 5% is split, while the kernel fits inside the unchained host work', () => {
    for (const [a, b, c] of [
      [0.01, 0.03, 0.01],
      [0.03, 0.01, 0.01],
      [0.0, 0.05, 0.0],
    ]) {
      const ms = MD_STEP.map((m, j) => ({ ...m, share: j === 1 ? 0.95 : [a, 0, b, c][j] }));
      expect(moduleSpeedup(ms, false).speedup).toBeCloseTo(16.807, 3);
      expect(moduleSpeedup(ms, true).speedup).toBeCloseTo(20, 9);
    }
  });
});

describe('numbers the lesson states', () => {
  it('p-overlap: 90% at 50x gives 8.47x without overlap, 10x with it, and 10x is the ceiling', () => {
    expect(speedup(0.9, 50)).toBeCloseTo(8.4746, 4);
    expect(1 / (0.1 + 0.018)).toBeCloseTo(speedup(0.9, 50), 12);
    expect(overlapped(0.9, 50)).toBeCloseTo(10, 12);
    expect(ceiling(0.9)).toBeCloseTo(10, 12);
  });
  it('q-host-share: with the device hidden, 20x needs the host at 5% or less', () => {
    expect(overlapped(0.95, 1000)).toBeCloseTo(20, 9);
    expect(overlapped(0.94, 1000)).toBeLessThan(20);
  });
  it('q-decode-kernels: kernels 9 parts, host 1, kernels 3x faster, no overlap: 2.5x, host now a quarter', () => {
    expect(speedup(0.9, 3)).toBeCloseTo(2.5, 12);
    const r = moduleSpeedup(
      [
        { name: 'kernels', share: 0.9, place: 'device', s: 3 },
        { name: 'host', share: 0.1, place: 'host', s: 1, chained: true },
      ],
      false,
    );
    expect(r.speedup).toBeCloseTo(2.5, 12);
    expect(r.host / r.total).toBeCloseTo(0.25, 12);
    // a second 3x on the kernels buys only 2x more; 10x is the ceiling
    expect(speedup(0.9, 9) / speedup(0.9, 3)).toBeCloseTo(2, 12);
    expect(ceiling(0.9)).toBeCloseTo(10, 12);
  });
  it('the explorer task, book preset: pushing the kernel from 100x to 1,000x moves 16.8x only to 19.6x', () => {
    expect(moduleSpeedup(withS(MD_STEP, { 1: 1000 }), false).speedup).toBeCloseTo(19.627, 3);
  });
  it('moving vibrational and rotational forces to the GPU (4x): 22.5x, and overlap no longer adds anything', () => {
    const ms = place(MD_STEP, 2, 'device');
    const a = moduleSpeedup(ms, false);
    const b = moduleSpeedup(ms, true);
    expect(a.total).toBeCloseTo(0.0095 + 0.005 + 0.03, 12);
    expect(a.speedup).toBeCloseTo(22.472, 3);
    expect(b.speedup).toBeCloseTo(a.speedup, 9); // no unchained host work left to hide behind
    expect(b.hidden).toBe(0);
    expect(a.ceiling).toBeCloseTo(1 / 0.03, 9);
  });
  it('everything on the GPU: no ceiling', () => {
    const all = MD_STEP.map((m) => ({ ...m, place: 'device' as const }));
    expect(moduleSpeedup(all, true).ceiling).toBe(Infinity);
    expect(moduleSpeedup(all, false).speedup).toBeCloseTo(1 / (0.0095 + 0.05 / 4), 9);
  });
});

describe('the decode-step preset (ours, illustrative)', () => {
  it('shares add up to 1; kernels 86%, host 14%; today it is 1x', () => {
    expect(sum(DECODE_STEP.map((m) => m.share))).toBeCloseTo(1, 12);
    const r = moduleSpeedup(DECODE_STEP, false);
    expect(r.host).toBeCloseTo(0.14, 12);
    expect(r.speedup).toBeCloseTo(1, 12);
    expect(r.ceiling).toBeCloseTo(1 / 0.14, 9);
  });
  it('kernels 10x faster: 4.42x; with overlap 6.85x; the ceiling is 7.14x either way', () => {
    const fast = withS(DECODE_STEP, { 0: 10, 1: 10 });
    expect(moduleSpeedup(fast, false).speedup).toBeCloseTo(1 / 0.226, 9);
    expect(moduleSpeedup(fast, true).speedup).toBeCloseTo(1 / (0.06 + 0.086), 9);
    expect(moduleSpeedup(fast, true).ceiling).toBeCloseTo(7.1429, 4);
  });
  it('overlap cannot hide the host sampler: the next step waits for it', () => {
    const fast = withS(DECODE_STEP, { 0: 1000, 1: 1000 });
    const r = moduleSpeedup(fast, true);
    expect(r.total).toBeGreaterThanOrEqual(0.06 + 0.08 - 1e-12);
  });
  it('moving the sampler to the GPU lifts the ceiling to 12.5x, and overlap then reaches 10.9x', () => {
    const moved = place(withS(DECODE_STEP, { 0: 10, 1: 10 }), 2, 'device');
    const r = moduleSpeedup(moved, true);
    expect(r.ceiling).toBeCloseTo(12.5, 9);
    expect(r.total).toBeCloseTo(0.086 + 0.006, 12);
    expect(r.speedup).toBeCloseTo(10.87, 2);
  });
});

describe('properties', () => {
  const ps = [0, 0.1, 0.5, 0.9, 0.95, 0.99];
  const ss = [1, 2, 10, 100, 1000];
  it('overlap is never slower than no overlap, never above the ceiling, and equals it once p/s <= 1 - p', () => {
    for (const p of ps)
      for (const s of ss) {
        expect(overlapped(p, s)).toBeGreaterThanOrEqual(speedup(p, s) - 1e-12);
        if (p < 1) expect(overlapped(p, s)).toBeLessThanOrEqual(ceiling(p) + 1e-9);
        if (p / s <= 1 - p) expect(overlapped(p, s)).toBeCloseTo(ceiling(p), 9);
      }
  });
  it('two modules, nothing chained: moduleSpeedup is speedup() and overlapped()', () => {
    for (const p of ps)
      for (const s of ss) {
        const ms: AppModule[] = [
          { name: 'a', share: 1 - p, place: 'host', s: 1 },
          { name: 'b', share: p, place: 'device', s },
        ];
        expect(moduleSpeedup(ms, false).speedup).toBeCloseTo(speedup(p, s), 9);
        expect(moduleSpeedup(ms, true).speedup).toBeCloseTo(overlapped(p, s), 9);
      }
  });
  it('a chained host module adds its whole time even with overlap', () => {
    const ms: AppModule[] = [
      { name: 'feed', share: 0.1, place: 'host', s: 1, chained: true },
      { name: 'k', share: 0.9, place: 'device', s: 9 },
    ];
    expect(moduleSpeedup(ms, true).total).toBeCloseTo(0.2, 12);
    expect(moduleSpeedup(ms, false).total).toBeCloseTo(0.2, 12);
  });
  it('the host is not sped up: moduleTime ignores s on the host', () => {
    expect(moduleTime({ name: 'x', share: 0.3, place: 'host', s: 50 })).toBe(0.3);
    expect(moduleTime({ name: 'x', share: 0.3, place: 'device', s: 3 })).toBeCloseTo(0.1, 12);
  });
  it('spans tile the timeline: lanes never overlap themselves and the last end is the total', () => {
    for (const preset of [MD_STEP, DECODE_STEP])
      for (const overlap of [false, true]) {
        const r = moduleSpeedup(withS(preset, { 0: 7, 1: 13 }), overlap);
        expect(r.spans.map((s) => s.module)).toEqual(preset.map((_, i) => i));
        for (const lane of ['host', 'device'] as const) {
          const ss = r.spans.filter((s) => s.lane === lane).sort((a, b) => a.start - b.start);
          for (let i = 1; i < ss.length; i++) expect(ss[i].start).toBeGreaterThanOrEqual(ss[i - 1].end - 1e-12);
        }
        expect(Math.max(...r.spans.map((s) => s.end))).toBeCloseTo(r.total, 12);
        if (!overlap) expect(r.total).toBeCloseTo(r.host + r.device, 12);
        else expect(r.total).toBeCloseTo(r.chained + Math.max(r.host - r.chained, r.device), 12);
      }
  });
  it('with overlap, the MD step runs neighbour list, then forces side by side, then the update', () => {
    const r = moduleSpeedup(MD_STEP, true);
    const at = (i: number) => r.spans[i];
    expect(at(0)).toMatchObject({ lane: 'host', start: 0 });
    expect(at(1).start).toBeCloseTo(0.02, 12);
    expect(at(2).start).toBeCloseTo(0.02, 12);
    expect(at(3).start).toBeCloseTo(0.04, 12);
    expect(at(3).end).toBeCloseTo(0.05, 12);
  });
});

describe('rebalance', () => {
  it('sets one share and scales the others to keep the total at 1', () => {
    const r = rebalance(MD_STEP, 1, 0.99);
    expect(r[1].share).toBe(0.99);
    expect(sum(r.map((m) => m.share))).toBeCloseTo(1, 12);
    expect(r[0].share).toBeCloseTo(0.004, 12);
    expect(r[3].share).toBeCloseTo(0.002, 12);
  });
  it('clamps to 0..1, and shares out equally when the others were all 0', () => {
    const r = rebalance(MD_STEP, 1, 1.5);
    expect(r[1].share).toBe(1);
    expect(r[0].share).toBe(0);
    const back = rebalance(r, 1, 0.7);
    expect(back[0].share).toBeCloseTo(0.1, 12);
    expect(sum(back.map((m) => m.share))).toBeCloseTo(1, 12);
  });
  it('leaves the other fields alone', () => {
    const r = rebalance(DECODE_STEP, 0, 0.5);
    expect(r.map((m) => [m.name, m.place, m.s, m.fixed, m.chained])).toEqual(
      DECODE_STEP.map((m) => [m.name, m.place, m.s, m.fixed, m.chained]),
    );
  });
});
