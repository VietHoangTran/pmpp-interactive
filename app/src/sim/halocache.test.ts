import { describe, expect, it } from 'vitest';
import {
  SCENE_A,
  SCENE_B,
  SCENE_C,
  SCENE_ONE_L2,
  SCENE_SPLIT_L2,
  blockLines,
  bytesPerOutput,
  effectiveIntensity,
  idealLines,
  noReuseLines,
  placeBlocks,
  sceneFacts,
  simulate,
  summarise,
  type CacheConfig,
} from './halocache';

const BASE: CacheConfig = {
  tilesX: 8, tilesY: 8, tile: 32, r: 2, lineFloats: 32, capacity: 1024, resident: 8, slices: 1, placement: 'in-order', seed: 1,
};

describe('the lines a block touches', () => {
  it('an interior 32 x 32 block, radius 2, 128-byte lines: 32 lines of its own, 76 of halo', () => {
    // 2 rows above and 2 below span 3 lines each (12); 32 rows of left and right columns, one line each side (64).
    const l = blockLines(BASE, 3, 3);
    expect(l.load).toHaveLength(32);
    expect(l.halo).toHaveLength(76);
  });
  it('Fig. 7.12 (input tile 32, output tile 28): an interior block touches 2 lines per row, 64', () => {
    expect(blockLines({ ...BASE, kernel: 'halo' }, 3, 3).load).toHaveLength(64);
  });
});

describe('the two ends of the scale', () => {
  const configs: CacheConfig[] = [
    BASE,
    { ...BASE, tile: 16, r: 1, lineFloats: 16, resident: 4 },
    { ...BASE, tilesX: 5, tilesY: 3, tile: 8, r: 3, lineFloats: 32 },
    { ...BASE, tilesX: 6, tilesY: 6, tile: 16, r: 2, lineFloats: 8, resident: 1 },
  ];
  for (const cfg of configs)
    for (const kernel of ['cached', 'halo'] as const) {
      const c = { ...cfg, kernel };
      it(`capacity 0 is the all-miss count (${kernel}, T = ${cfg.tile}, r = ${cfg.r})`, () => {
        expect(simulate({ ...c, capacity: 0 }).misses).toBe(noReuseLines(c));
      });
      it(`an unbounded single cache fetches each line once: the ideal (${kernel}, T = ${cfg.tile}, r = ${cfg.r})`, () => {
        expect(simulate({ ...c, capacity: Infinity }).misses).toBe(idealLines(c));
      });
    }
  it('the ideal is 8 bytes per output (one load, one store) when rows are whole lines', () => {
    expect(bytesPerOutput(BASE, idealLines(BASE))).toBe(8);
    expect(effectiveIntensity(BASE, 8)).toBe(6.25);
  });
  it('8 x 8 tiles of 32, radius 2, 128-byte lines: no reuse costs 16.2 bytes per output for Fig. 7.15 and 14.3 for Fig. 7.12', () => {
    const s = summarise(BASE);
    expect(s.bytes.allMiss).toBe(16.203125);
    expect(s.bytes.fig712).toBe(14.265625);
    expect(s.bytes.allMiss).toBeGreaterThan(s.bytes.fig712);
  });
});

describe('the explorer\'s default and what changing it shows', () => {
  it('one shared L2 of 1024 lines (four rows of tiles): the run reaches the ideal and 97% of halo lines hit', () => {
    const s = summarise(BASE);
    expect(s.lines.run).toBe(s.lines.ideal);
    expect(s.haloHitFraction!).toBeGreaterThan(0.97);
  });
  it('shrinking the cache moves traffic toward the all-miss count', () => {
    const at = (capacity: number) => summarise({ ...BASE, capacity }).bytes.run;
    expect(at(256)).toBeGreaterThan(at(1024));
    expect(at(64)).toBeGreaterThan(at(256));
    expect(at(0)).toBe(summarise(BASE).bytes.allMiss);
  });
  it('more slices with random placement give more misses (seeds 1 to 5, finite and unbounded capacity)', () => {
    for (const capacity of [1024, 4096, Infinity])
      for (let seed = 1; seed <= 5; seed++) {
        let prev = -1;
        for (const slices of [1, 2, 4, 8]) {
          const m = simulate({ ...BASE, capacity, slices, placement: 'random', seed }).misses;
          expect(m).toBeGreaterThanOrEqual(prev);
          prev = m;
        }
      }
  });
  it('with eight random slices and the default capacity, traffic is worse than Fig. 7.12 with no reuse', () => {
    const s = summarise({ ...BASE, slices: 8, placement: 'random', capacity: 4096 });
    expect(s.bytes.run).toBeGreaterThan(s.bytes.fig712);
  });
  it('the same seed gives the same run; random placement depends on the seed', () => {
    const c = { ...BASE, slices: 4, placement: 'random' as const, seed: 3 };
    expect(simulate(c, { trace: true }).events).toEqual(simulate(c, { trace: true }).events);
    expect(placeBlocks(64, c)).not.toEqual(placeBlocks(64, { ...c, seed: 4 }));
  });
  it('in-order placement gives each wave to the slices in contiguous runs; round-robin deals them in turn', () => {
    expect(placeBlocks(8, { slices: 4, placement: 'in-order', seed: 1, resident: 8 })).toEqual([0, 0, 1, 1, 2, 2, 3, 3]);
    expect(placeBlocks(6, { slices: 4, placement: 'round-robin', seed: 1, resident: 8 })).toEqual([0, 1, 2, 3, 0, 1]);
  });
});

describe('the scene "Where the halo comes from"', () => {
  const f = sceneFacts();
  it('A is the corner block, B its right neighbour, C the first block of the next row', () => {
    expect([f.one.blocks[SCENE_A].bx, f.one.blocks[SCENE_A].by]).toEqual([0, 0]);
    expect([f.one.blocks[SCENE_B].bx, f.one.blocks[SCENE_B].by]).toEqual([1, 0]);
    expect([f.one.blocks[SCENE_C].bx, f.one.blocks[SCENE_C].by]).toEqual([0, 1]);
  });
  it('part 1: every line A loads comes from DRAM', () => {
    expect(f.one.blocks[SCENE_A].loadHits).toBe(0);
    expect(f.one.blocks[SCENE_A].loadLines).toBe(8);
  });
  it('part 2: all 8 of B\'s reads of A\'s lines hit, and so do B\'s own 8 loads (A\'s halo read pulled them in)', () => {
    expect([f.bLeftHits, f.bLeft]).toEqual([8, 8]);
    expect([f.bLoadHits, f.bLoads]).toEqual([8, 8]);
  });
  it('part 3: C reads A\'s bottom 2 rows as halo and both reads miss; the 48-line cache has turned over since B', () => {
    expect([f.cOnABottomHits, f.cOnABottom]).toEqual([0, 2]);
    expect(SCENE_ONE_L2.capacity).toBe(48);
    expect(f.evictedBeforeC).toBeGreaterThan(0);
  });
  it('part 4: on two slices, A and B land on different ones and all 8 of B\'s reads of A\'s lines miss', () => {
    expect(f.sliceA).not.toBe(f.sliceB);
    expect([f.splitBLeftHits, f.splitBLeft]).toEqual([0, 8]);
    expect(SCENE_SPLIT_L2.slices).toBe(2);
  });
  it('part 5: ideal 8, one L2 9.5, split 16.3, no reuse 19.1 bytes per output', () => {
    expect(f.bytes.ideal).toBe(8);
    expect(f.bytes.one).toBe(9.5);
    expect(f.bytes.split).toBeCloseTo(16.34, 2);
    expect(f.bytes.noReuse).toBe(19.125);
  });
});
