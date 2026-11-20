import { describe, expect, it } from 'vitest';
import {
  CH6_DEVICE_R,
  FIG1021,
  RUN_SEED,
  SCENE,
  SOFTMAX_BATCHES,
  SOFTMAX_ROW,
  VOCABS,
  blockSum,
  blockSums,
  blockView,
  coarsenSteps,
  commit,
  f32,
  f32bits,
  f32hex,
  fewestStepsC,
  firstDifferentRun,
  halvingLevels,
  halvingTree,
  history,
  justEnoughC,
  logits,
  maxUsefulC,
  reference,
  runOf,
  sceneData,
  seededOrder,
  segments,
  softmaxPreset,
  sum64,
  sweepC,
  threadLoads,
  threadSum,
  ulpDistance,
} from './multiblock';

describe('segments (Figs. 10.17, 10.18 and 10.20; exercise 5)', () => {
  it('1,000,000 elements, 1024-thread blocks: 489 blocks, 576 real elements in the last (q-lastblock)', () => {
    const s = segments(1_000_000, 1024);
    expect(s.perBlock).toBe(2048);
    expect(s.blocks).toBe(489);
    expect(s.lastStart).toBe(488 * 2048);
    expect(s.lastStart).toBe(999_424);
    expect(s.lastReal).toBe(576);
    expect(s.pastEnd).toBe(2048 - 576);
  });
  it('block b starts at 2 x blockDim x b: 0, 2048, 4096 for 1024 threads (p. 244)', () => {
    expect([0, 1, 2].map((b) => threadLoads(1 << 20, 1024, 1, b, 0)[0].index)).toEqual([0, 2048, 4096]);
  });
  it('each thread loads i and i + blockDim (Fig. 10.18 line 05)', () => {
    expect(threadLoads(1 << 20, 1024, 1, 1, 5).map((l) => l.index)).toEqual([2048 + 5, 2048 + 5 + 1024]);
  });
  it('Fig. 10.19: 8 threads coarsened by two cover 32 elements, thread t loads t, t + 8, t + 16, t + 24', () => {
    expect(segments(32, 8, 2)).toMatchObject({ perBlock: 32, blocks: 1, lastReal: 32, pastEnd: 0 });
    expect(threadLoads(32, 8, 2, 0, 3).map((l) => l.index)).toEqual([3, 11, 19, 27]);
  });
  it('in the last block of the 1,000,000 case only the first load of threads 0 to 575 is real', () => {
    expect(threadLoads(1_000_000, 1024, 1, 488, 575)).toEqual([
      { k: 0, index: 999_999, inRange: true },
      { k: 1, index: 1_001_023, inRange: false },
    ]);
    expect(threadLoads(1_000_000, 1024, 1, 488, 576).every((l) => !l.inRange)).toBe(true);
    expect(blockView(1_000_000, 1024, 1, 488)).toMatchObject({ real: 576, busyThreads: 576, idleThreads: 448, pastEnd: 1472 });
    expect(blockView(1_000_000, 1024, 1, 0)).toMatchObject({ real: 2048, busyThreads: 1024, idleThreads: 0, pastEnd: 0 });
  });
  it('an exact multiple leaves nothing past the end; an empty input launches nothing', () => {
    expect(segments(4096, 1024).pastEnd).toBe(0);
    expect(segments(0, 256).blocks).toBe(0);
  });
});

describe('block sums in float32', () => {
  it('guarded loads contribute 0: ten ones, 4-thread blocks give 8 and 2', () => {
    const ones = new Float32Array(10).fill(1);
    expect(blockSums(ones, 4)).toEqual([8, 2]);
    expect(threadSum(ones, 4, 1, 1, 1)).toBe(1); // index 9
    expect(threadSum(ones, 4, 1, 1, 2)).toBe(0); // indices 10 and 14: past the end
  });
  it('coarsening changes how a block adds, not what it adds: small integers sum exactly either way', () => {
    const v = Float32Array.from({ length: 1000 }, (_, i) => i % 7);
    let loop = 0;
    for (const x of v) loop += x;
    expect(loop).toBe(2997);
    for (const c of [1, 2, 4]) expect(blockSums(v, 8, c).reduce((a, b) => a + b, 0)).toBe(loop);
  });
  it('the halving tree pads with zeros and rounds every addition', () => {
    expect(halvingTree([1, 2, 3])).toBe(6);
    expect(halvingTree([])).toBe(0);
    expect(halvingTree([16777216, 1, 1, 0])).toBe(f32(f32(16777216 + 1) + f32(1 + 0)));
  });
  it('a block sum is the same every time it is computed: the order inside a block is fixed', () => {
    const v = logits(4096, 3);
    expect(blockSum(v, 256, 1, 3)).toBe(blockSum(v, 256, 1, 3));
  });
});

describe('commits: atomics give mutual exclusion, not order (p. 245)', () => {
  const v = logits(65536, 2026);
  const sums = blockSums(v, 256);
  it('128 blocks, 20 seeded orders: the atomic totals take more than one float32 value', () => {
    expect(sums).toHaveLength(128);
    const h = history(sums, 'atomic', 20);
    expect(h.distinct.length).toBeGreaterThan(1);
    expect(h.totals).toHaveLength(20);
  });
  it('the two-pass total is the same on every run', () => {
    const h = history(sums, 'two-pass', 20);
    expect(h.distinct).toHaveLength(1);
    expect(h.distinct[0]).toBe(f32hex(commit(sums, 'fixed').total));
  });
  it('every total sits within a few float32 steps of the float64 reference; none is it exactly', () => {
    const ref = reference(v);
    for (const t of history(sums, 'atomic', 20).totals) {
      expect(ulpDistance(t, f32(ref))).toBeLessThan(32);
      expect(t).not.toBe(ref);
    }
  });
  it('two blocks cannot disagree: float addition is commutative, only not associative', () => {
    const two = blockSums(logits(4096, 2026), 1024);
    expect(two).toHaveLength(2);
    expect(history(two, 'atomic', 20).distinct).toHaveLength(1);
  });
  it('the running total ends on the total, and run k uses order seed RUN_SEED + k', () => {
    const r = runOf(sums, 'atomic', 3);
    expect(r.order).toEqual(seededOrder(128, RUN_SEED + 3));
    expect(r.running[r.running.length - 1]).toBe(r.total);
    expect([...r.order].sort((a, b) => a - b)).toEqual(sums.map((_, i) => i));
  });
  it('the float64 reference is a compensated sum', () => {
    expect(sum64([1e16, 1, -1e16])).toBe(1);
  });
});

describe('the commit-order scene (60 logits, 4-thread blocks)', () => {
  const s = sceneData();
  it('8 segments of 8; the last holds 4 real elements', () => {
    expect(s.seg).toMatchObject({ blocks: 8, perBlock: 8, lastStart: 56, lastReal: 4, pastEnd: 4 });
    expect(SCENE.N).toBe(60);
  });
  it('runs 0 and 1 commit in different orders and end on different bits', () => {
    expect(s.runA.order).not.toEqual(s.runB.order);
    expect(f32bits(s.runA.total)).not.toBe(f32bits(s.runB.total));
    expect(firstDifferentRun(s.sums)).toBe(1);
    // "differ in their last bits": a couple of float32 steps, nothing more.
    expect(ulpDistance(s.runA.total, s.runB.total)).toBeGreaterThan(0);
    expect(ulpDistance(s.runA.total, s.runB.total)).toBeLessThan(8);
  });
  it('the two-pass total is one fixed value, the last level of its tree', () => {
    expect(history(s.sums, 'two-pass', 10).distinct).toEqual([f32hex(s.twoPass.total)]);
    const levels = halvingLevels(s.sums);
    expect(levels.map((l) => l.length)).toEqual([8, 4, 2, 1]);
    expect(levels[1][0]).toBe(f32(s.sums[0] + s.sums[4]));
    expect(levels[3][0]).toBe(s.twoPass.total);
  });
  it('the three totals are distinct bit patterns, with the two-pass one between the atomic runs', () => {
    const hexes = [s.runA.total, s.runB.total, s.twoPass.total].map(f32hex);
    expect(new Set(hexes).size).toBe(3);
    expect(hexes).toEqual(['0xc2532b58', '0xc2532b56', '0xc2532b57']);
    expect(f32hex(f32(s.ref))).toBe('0xc2532b57');
  });
  it('the cost line: one float per block, 489 for a million logits at 1024 threads', () => {
    expect(segments(1_000_000, 1024).blocks * 4).toBe(1956);
  });
});

describe('Fig. 10.21 step accounting (pp. 247-248)', () => {
  it('two original blocks one after the other: 8 steps, 2 fully used, 6 underused', () => {
    const r = coarsenSteps({ ...FIG1021, C: 1 });
    expect(r).toMatchObject({ blocks: 2, waves: 2, fullSteps: 2, underSteps: 6, totalSteps: 8, commits: 2 });
  });
  it('one block coarsened by two: 6 steps, 3 and 3', () => {
    const r = coarsenSteps({ ...FIG1021, C: 2 });
    expect(r).toMatchObject({ blocks: 1, waves: 1, fullSteps: 3, underSteps: 3, totalSteps: 6, commits: 1 });
  });
  it('q-fig1021: N 64, 8 threads, R 1: C = 4 takes 10 steps (7 and 3), C = 1 takes 16 (4 blocks x 4)', () => {
    expect(coarsenSteps({ N: 64, block: 8, C: 4, R: 1 })).toMatchObject({ blocks: 1, fullSteps: 7, underSteps: 3, totalSteps: 10 });
    expect(coarsenSteps({ N: 64, block: 8, C: 1, R: 1 })).toMatchObject({ blocks: 4, waves: 4, totalSteps: 16 });
  });
});

describe('just enough blocks (p. 248)', () => {
  const base = { N: 1 << 20, block: 256, R: 256 };
  it('q-just-enough: C = 8 launches exactly 256 blocks, one wave', () => {
    expect(coarsenSteps({ ...base, C: 8 })).toMatchObject({ blocks: 256, waves: 1, tooFew: false, busyShare: 1 });
    expect(justEnoughC(base, 64)).toBe(8);
    expect(fewestStepsC(base, 64)).toBe(8);
  });
  it('C = 4: 512 blocks, two waves, twice the tree phases and commits; C = 16: 128 blocks, half the device idle', () => {
    const c4 = coarsenSteps({ ...base, C: 4 });
    const c8 = coarsenSteps({ ...base, C: 8 });
    const c16 = coarsenSteps({ ...base, C: 16 });
    expect(c4).toMatchObject({ blocks: 512, waves: 2, commits: 512 });
    expect(c4.underSteps).toBe(2 * c8.underSteps);
    expect(c16).toMatchObject({ blocks: 128, tooFew: true, busyShare: 0.5 });
  });
  it('wave quantisation: on 264 slots C = 7 launches 293 blocks, two waves, and is worse than both neighbours', () => {
    const sw = sweepC({ N: 1 << 20, block: 256, R: CH6_DEVICE_R }, 64);
    const at = (c: number) => sw[c - 1];
    expect(at(7)).toMatchObject({ blocks: 293, waves: 2, lastWave: 29, totalSteps: 42 });
    expect(at(6).totalSteps).toBe(38);
    expect(at(8)).toMatchObject({ blocks: 256, waves: 1, totalSteps: 23, tooFew: true });
    expect(at(7).totalSteps).toBeGreaterThan(at(6).totalSteps);
    expect(at(7).totalSteps).toBeGreaterThan(at(8).totalSteps);
    // The book's rule (largest C still filling the device) picks 7; the fewest steps sit at 8.
    expect(justEnoughC({ N: 1 << 20, block: 256, R: CH6_DEVICE_R }, 64)).toBe(7);
    expect(fewestStepsC({ N: 1 << 20, block: 256, R: CH6_DEVICE_R }, 64)).toBe(8);
  });
  it('past one wave, steps grow by 2 per unit of C', () => {
    const sw = sweepC({ N: 1 << 20, block: 256, R: CH6_DEVICE_R }, 64);
    expect(sw[20].totalSteps - sw[19].totalSteps).toBe(2);
  });
});

describe('softmax presets: one 256-thread block per row', () => {
  it('the row length is a vocabulary size from the facts file', () => {
    expect(VOCABS.map((v) => v.size)).toContain(SOFTMAX_ROW);
    expect(SOFTMAX_BATCHES).toEqual([1, 8, 64]);
  });
  it('one block covers the row: C = 297, 594 loads per thread, 593 additions then 8 tree steps', () => {
    const r = coarsenSteps(softmaxPreset(1));
    expect(maxUsefulC(SOFTMAX_ROW, 256)).toBe(297);
    expect(SOFTMAX_ROW / 256).toBe(593.5);
    expect(r).toMatchObject({ blocksPerRow: 1, blocks: 1, waves: 1, fullPerBlock: 593, underPerBlock: 8, totalSteps: 601, tooFew: true });
  });
  it('blocks = batch: 1, 8 and 64 of 264 slots', () => {
    expect(SOFTMAX_BATCHES.map((b) => coarsenSteps(softmaxPreset(b)).blocks)).toEqual([1, 8, 64]);
    expect(coarsenSteps(softmaxPreset(64)).busyShare).toBeCloseTo(64 / 264, 12);
  });
  it('q-batch1: one block on a device that runs 100 at once keeps 1% of the slots busy', () => {
    expect(coarsenSteps(softmaxPreset(1, 100)).busyShare).toBe(0.01);
  });
});
