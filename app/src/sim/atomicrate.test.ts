import { describe, expect, it } from 'vitest';
import {
  BOOK_MEMORY,
  SCENE,
  SKEWED4,
  UNIFORM4,
  atomicCycles,
  elementRate,
  ex1,
  ex2,
  ex3,
  ex4,
  histogramBound,
  longestChain,
  meanTime,
  peakBandwidth,
  perLocationRate,
  sceneBins,
  sceneLoads,
  sceneLockstep,
  sceneOneBin,
  scenePrivatised,
  skewSlowdown,
  timeline,
  uniformShare,
} from './atomicrate';

const close = (a: number, b: number, rel = 1e-9) => expect(Math.abs(a - b)).toBeLessThanOrEqual(Math.abs(b) * rel);

describe('the book\'s hypothetical memory system (pp. 209-210)', () => {
  it('8 bytes x 2 transfers x 1 GHz x 8 channels = 128 GB/s', () => {
    expect(peakBandwidth(BOOK_MEMORY)).toBe(128e9);
  });
  it('which is 32 G four-byte elements per second', () => {
    expect(elementRate(BOOK_MEMORY)).toBe(32e9);
  });
  it('one location: 200 + 200 cycles per atomic, 2.5 M atomics per second', () => {
    expect(atomicCycles(200)).toBe(400);
    expect(perLocationRate(1e9, 200)).toBe(2.5e6);
  });
  it('256 uniformly hit bins: 640 M atomics per second', () => {
    expect(histogramBound(2.5e6, uniformShare(256)).rate).toBe(640e6);
  });
  it('640 M/s is still 50 times below the 32 G elements/s the memory system delivers', () => {
    close(elementRate(BOOK_MEMORY) / histogramBound(2.5e6, 1 / 256).rate, 50);
  });
});

describe('the busiest bin sets the pace', () => {
  it('half the pixels in one bin: 5 M/s, whatever the other 255 bins do', () => {
    expect(histogramBound(2.5e6, 0.5).rate).toBe(5e6);
  });
  it('an all-one-value image runs at the one-location rate', () => {
    expect(histogramBound(2.5e6, 1).rate).toBe(2.5e6);
  });
  it('the memory system caps the bound', () => {
    expect(histogramBound(1e9 / 4, 1 / 256, 32e9)).toEqual({ rate: 32e9, capped: true });
    expect(histogramBound(2.5e6, 1 / 256, 32e9).capped).toBe(false);
  });
  it('in the book\'s model a constant image is 256 times slower than uniform bytes', () => {
    expect(skewSlowdown(2.5e6, 256, 32e9)).toBe(256);
  });
  it('shorter latency shortens every link: 20-cycle accesses give 25 M/s on one location', () => {
    expect(perLocationRate(1e9, 20)).toBe(25e6);
  });
  it('rejects a share outside (0, 1]', () => {
    expect(() => histogramBound(1, 0)).toThrow();
    expect(() => histogramBound(1, 1.5)).toThrow();
  });
});

describe('exercises 1 to 4 (p. 220)', () => {
  it('1: 100 ns per atomic gives 10 M atomics/s', () => {
    close(ex1(), 10e6);
  });
  it('2: 0.9 x 4 + 0.1 x 100 = 13.6 ns, about 73.5 M atomics/s', () => {
    close(meanTime(0.9, 4, 100), 13.6, 1e-12);
    close(ex2().rate, 73.53e6, 1e-3);
  });
  it('3: 5 FLOP per atomic at 10 M/s is 50 M FLOP/s', () => {
    close(ex3(), 50e6);
  });
  it('4, 1 ns per shared-memory atomic: 1.1 ns, about 909 M/s and 4.55 GFLOP/s, 90 times exercise 3', () => {
    const r = ex4(1);
    close(r.nsPerAtomic, 1.1, 1e-12);
    close(r.rate, 909.09e6, 1e-4);
    close(r.flops, 4.545e9, 1e-3);
    close(r.flops / ex3(), 90.9, 1e-3);
  });
  it('4, read plus write at 1 ns each: 2.2 ns, about 2.27 GFLOP/s', () => {
    close(ex4(2).flops, 2.273e9, 1e-3);
  });
});

describe('timelines for the animation', () => {
  it('eight independent loads end at one latency plus seven issue slots', () => {
    const t = sceneLoads();
    expect(t.end).toBe(SCENE.dram + 7 * SCENE.issue);
    expect(t.bars.every((b) => b.end - b.start === SCENE.dram)).toBe(true);
  });
  it('eight same-address atomics end at 8 x 400 cycles, each waiting for the one before', () => {
    const t = sceneOneBin();
    expect(t.end).toBe(8 * 400);
    expect(t.bars.map((b) => b.start)).toEqual([0, 400, 800, 1200, 1600, 2000, 2400, 2800]);
    expect(t.bars.map((b) => b.read)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(t.memory.bin).toBe(8);
    expect(t.lost).toBe(0);
    expect(t.bars[7].waited).toBe(2800 - 70);
  });
  it('eight lockstep plain increments leave 1: seven updates lost, and it is fast', () => {
    const t = sceneLockstep();
    expect(t.memory.bin).toBe(1);
    expect(t.lost).toBe(7);
    expect(t.bars.every((b) => b.read === 0 && b.wrote === 1)).toBe(true);
    expect(t.end).toBe(SCENE.dram + 7 * SCENE.issue + 7 * SCENE.issue + SCENE.dram);
    expect(t.end).toBeLessThan(sceneOneBin().end / 5);
  });
  it('uniform data over four bins: four chains of two, done after about two atomics', () => {
    const t = sceneBins(UNIFORM4);
    expect(longestChain(UNIFORM4)).toBe(2);
    expect(t.end).toBe(800 + 3 * SCENE.issue);
    expect(Object.values(t.memory)).toEqual([2, 2, 2, 2]);
  });
  it('skewed data: the hottest bin\'s chain of five sets the finish at 2000 cycles', () => {
    const t = sceneBins(SKEWED4);
    expect(longestChain(SKEWED4)).toBe(5);
    expect(t.end).toBe(5 * 400);
    const others = t.bars.filter((b) => b.address !== 'bin 0');
    expect(Math.max(...others.map((b) => b.end))).toBeLessThan(500);
  });
  it('the same chains with 20-cycle cache atomics: every link ten times shorter', () => {
    expect(sceneBins(SKEWED4, SCENE.cache).end).toBe(5 * 40);
    expect(sceneOneBin(8, SCENE.cache).end).toBe(8 * 40);
  });
  it('privatised: two short private chains, then one commit per block that queue', () => {
    const t = scenePrivatised();
    const priv0 = t.bars.filter((b) => b.address === 'private 0');
    expect(priv0).toHaveLength(4);
    // A shared-memory atomic (8 cycles) is shorter than the issue slot (10), so nobody waits.
    expect(priv0.every((b) => b.waited === 0)).toBe(true);
    const done = 3 * SCENE.issue + 2 * SCENE.shared;
    expect(Math.max(...priv0.map((b) => b.end))).toBe(done);
    const commits = t.bars.filter((b) => b.address === 'bin');
    expect(commits).toHaveLength(2);
    expect(commits.map((b) => [b.start, b.read, b.wrote])).toEqual([[done, 0, 4], [done + 400, 4, 8]]);
    expect(t.end).toBe(done + 800);
    expect(t.memory.bin).toBe(8);
    expect(t.lost).toBe(0);
    expect(t.end).toBeLessThan(sceneOneBin().end / 3);
  });
  it('a mixed list: loads do not wait on atomics, atomics on different addresses do not wait on each other', () => {
    const t = timeline(
      [
        { row: 'a', address: 'x', kind: 'atomic' },
        { row: 'b', address: 'y', kind: 'atomic' },
        { row: 'c', address: 'x', kind: 'load' },
      ],
      { load: 10, store: 10, issue: 1 },
    );
    expect(t.bars.map((b) => [b.start, b.end])).toEqual([[0, 20], [1, 21], [2, 12]]);
  });
});
