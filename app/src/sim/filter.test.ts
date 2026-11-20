import { describe, expect, it } from 'vitest';
import {
  CURVE_FRACTIONS,
  FIG12_1,
  FRACTIONS,
  RUN_LENGTH,
  SCENE,
  arrivalOrder,
  atomicCounts,
  crossings,
  curve,
  expectedAtomics,
  fromFlags,
  inPlaceOverwrites,
  keepFrom,
  patternSource,
  randomFlags,
  sameKeys,
  sceneKeep,
  serveOneByOne,
  stableFilter,
  unstableFilter,
} from './filter';

const fig = () => keepFrom(FIG12_1.n, FIG12_1.kept);

describe('Fig. 12.1: stable and unstable', () => {
  it('the stable filter keeps the book\'s eight keys in input order, and its arrows never cross', () => {
    const out = stableFilter(fig());
    expect(out).toEqual([1, 3, 4, 6, 8, 10, 11, 14]);
    expect(crossings(out)).toBe(0);
  });
  it('the book\'s unstable panel holds the same keys in another order, with crossing arrows', () => {
    const out = unstableFilter(fig(), FIG12_1.bookUnstable);
    expect(sameKeys(out, stableFilter(fig()))).toBe(true);
    expect(crossings(out)).toBeGreaterThan(0);
  });
  it('every unstable run is a permutation of the stable output: same keys, same count', () => {
    const stable = stableFilter(fig());
    const orders = new Set<string>();
    for (let seed = 1; seed <= 300; seed++) {
      const out = unstableFilter(fig(), arrivalOrder(fig(), seed));
      expect(out.length).toBe(stable.length);
      expect(sameKeys(out, stable)).toBe(true);
      expect(crossings(out) === 0).toBe(out.every((k, i) => k === stable[i]));
      orders.add(out.join(','));
    }
    expect(orders.size).toBeGreaterThan(250); // the order does change between runs
  });
  it('the same seed gives the same order (a run can be replayed), another seed another order', () => {
    expect(arrivalOrder(fig(), 7)).toEqual(arrivalOrder(fig(), 7));
    expect(arrivalOrder(fig(), 7)).not.toEqual(arrivalOrder(fig(), 8));
  });
  it('an arrival order that is not the kept keys is refused', () => {
    expect(() => unstableFilter(fig(), [1, 3, 4])).toThrow();
    expect(() => unstableFilter(fig(), [1, 3, 4, 6, 8, 10, 11, 13])).toThrow();
  });
  it('works for any keep set, including none and all', () => {
    for (const kept of [[], Array.from({ length: 16 }, (_, i) => i), [0], [15], [2, 9]]) {
      const keep = keepFrom(16, kept);
      expect(sameKeys(unstableFilter(keep, arrivalOrder(keep, 3)), stableFilter(keep))).toBe(true);
    }
  });
});

describe('in place (p. 290, section 12.7)', () => {
  it('with the Fig. 12.1 keys, k3 lands where k1 lived, and k1 moves to slot 0', () => {
    const w = inPlaceOverwrites(fig());
    expect(w[1]).toEqual({ slot: 1, writer: 3, held: 1, heldMovesTo: 0, kind: 'moved' });
    expect(w[0]).toEqual({ slot: 0, writer: 1, held: 0, heldMovesTo: null, kind: 'tested' }); // k0 is dropped, but it must still be read first
  });
  it('question q-inplace-slot: keys k0..k7 keeping k1, k3, k4, k6: k6 goes to slot 3, which held k3 (bound for slot 1)', () => {
    const w = inPlaceOverwrites(keepFrom(8, [1, 3, 4, 6]));
    expect(w.map((x) => x.writer)).toEqual([1, 3, 4, 6]);
    expect(w[3]).toEqual({ slot: 3, writer: 6, held: 3, heldMovesTo: 1, kind: 'moved' });
  });
  it('every write onto another key needs a read first: kept keys to move them, dropped keys to test them', () => {
    expect(inPlaceOverwrites(fig()).map((o) => o.kind)).toEqual(['tested', 'moved', 'tested', 'moved', 'moved', 'tested', 'moved', 'tested']);
    // A kept prefix stays where it is: every key is written over itself.
    expect(inPlaceOverwrites(keepFrom(8, [0, 1, 2])).map((o) => o.kind)).toEqual(['own', 'own', 'own']);
    expect(inPlaceOverwrites(keepFrom(8, [0, 2])).map((o) => o.kind)).toEqual(['own', 'tested']);
  });
});

describe('atomics on the counter (sections 12.2 to 12.4)', () => {
  const ratio = (f: number, W = 32) => {
    const e = expectedAtomics(1 << 20, f, W, 256);
    return e.perKey / e.perWarp;
  };
  it('independent keys: one atomic per warp beats one per kept key 16 times at 50 percent, 32 at 100, about 1.16 at 1', () => {
    expect(ratio(0.5)).toBeCloseTo(16, 6);
    expect(ratio(1)).toBe(32);
    expect(ratio(0.01)).toBeCloseTo(1.1637, 3);
    expect(Math.abs(ratio(0.01) - 1.16) / 1.16).toBeLessThan(0.03); // question q-lowfraction's tolerance
    expect(ratio(0.5, 64)).toBeCloseTo(32, 6);
  });
  it('the q-lowfraction arithmetic: 0.32 kept per warp, 27.5 percent of warps busy', () => {
    expect(0.01 * 32).toBeCloseTo(0.32, 12);
    expect(1 - Math.pow(0.99, 32)).toBeCloseTo(0.275, 3);
  });
  it('the 12.2 prediction: 2^20 keys, half kept: 524,288 atomics per key against 32,768 per warp', () => {
    const e = expectedAtomics(1 << 20, 0.5, 32, 256);
    expect(e.perKey).toBe(524288);
    expect(e.perWarp).toBeCloseTo(32768, 3);
    expect(Math.round(e.perKey / e.perWarp)).toBe(16);
  });
  it('one per block is N / block size whatever the fraction, even when nothing is kept', () => {
    for (const f of FRACTIONS) expect(expectedAtomics(1 << 20, f, 32, 256).perBlock).toBe(4096);
    for (const p of ['runs', 'spread', 'sorted'] as const) expect(atomicCounts(patternSource(p, 1000, 0, 1), 32, 128).perBlock).toBe(8);
  });
  it('a partial last warp counts with its own lanes', () => {
    const e = expectedAtomics(40, 1, 32, 64);
    expect(e.perWarp).toBe(2);
    expect(expectedAtomics(40, 0.5, 32, 64).perWarp).toBeCloseTo(1 - 2 ** -32 + 1 - 2 ** -8, 12);
  });
  it('a concrete random input agrees with the expectation', () => {
    for (const f of [0.01, 0.1, 0.5]) {
      const c = atomicCounts(fromFlags(randomFlags(1 << 18, f, 5)), 32, 256);
      const e = expectedAtomics(1 << 18, f, 32, 256);
      expect(Math.abs(c.perWarp - e.perWarp) / e.perWarp).toBeLessThan(0.05);
      expect(Math.abs(c.perKey - e.perKey) / e.perKey).toBeLessThan(0.05);
    }
  });
  it('patterns: every other key at 50 percent keeps 16 per warp; sorted fills whole warps; runs keep most warps busy', () => {
    const n = 1 << 16;
    const spread = atomicCounts(patternSource('spread', n, 0.5), 32, 256);
    expect(spread.perKey).toBe(n / 2);
    expect(spread.perWarp).toBe(n / 32);
    const sorted = atomicCounts(patternSource('sorted', n, 0.3), 32, 256);
    expect(sorted.perKey).toBe(Math.round(0.3 * n));
    expect(sorted.perWarp).toBe(Math.ceil(Math.round(0.3 * n) / 32));
    // At 1 percent, independent keys give almost one atomic per kept key; runs of 50 do not.
    const runs = atomicCounts(patternSource('runs', n, 0.01, 3), 32, 256);
    expect(runs.perKey).toBeGreaterThan(0);
    expect(runs.perKey / runs.perWarp).toBeGreaterThan(10);
    const random = atomicCounts(patternSource('random', n, 0.01, 3), 32, 256);
    expect(random.perKey / random.perWarp).toBeLessThan(1.3);
  });
  it('the spread pattern keeps exactly the odd keys at one half', () => {
    const s = patternSource('spread', 10, 0.5);
    expect(Array.from({ length: 10 }, (_, i) => s.keptIn(i, i + 1))).toEqual([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
  });
  it('the runs pattern keeps or drops whole runs', () => {
    const s = patternSource('runs', 1000, 0.4, 9);
    for (let r = 0; r < 1000 / RUN_LENGTH; r++) expect([0, RUN_LENGTH]).toContain(s.keptIn(r * RUN_LENGTH, (r + 1) * RUN_LENGTH));
  });
  it('Fig. 12.3 seeing two snapshots per warp: two atomics where a warp keeps two or more keys', () => {
    const keep = keepFrom(64, [1, 2, 4, 5, 40]);
    const c = atomicCounts(fromFlags(keep), 32, 64, 2);
    expect(c.perWarp).toBe(2);
    expect(c.perSnapshot).toBe(3);
    expect(atomicCounts(fromFlags(keep), 32, 64, 1).perSnapshot).toBe(2);
  });
  it('Fig. 12.6: one private atomic per kept key, or per busy warp with a vote', () => {
    const c = atomicCounts(fromFlags(keepFrom(64, [1, 2, 4, 5, 40])), 32, 64);
    expect(c.privatePerKey).toBe(5);
    expect(c.privatePerWarp).toBe(2);
    expect(c.perBlock).toBe(1);
  });
  it('a block size that is not a multiple of the warp is refused', () => {
    expect(() => atomicCounts(patternSource('sorted', 100, 0.5), 64, 96)).toThrow();
  });
  it('curves: per-warp never exceeds per-key, and the gap closes at low fractions for random keys', () => {
    const pts = curve('random', 1 << 20, 32, 256);
    expect(pts.length).toBe(CURVE_FRACTIONS.length);
    for (const p of pts) expect(p.perWarp).toBeLessThanOrEqual(p.perKey + 1e-9);
    expect(pts[0].perKey / pts[0].perWarp).toBeLessThan(1.02);
    expect(pts[pts.length - 1].perKey / pts[pts.length - 1].perWarp).toBe(32);
    for (const p of ['runs', 'spread', 'sorted'] as const)
      for (const q of curve(p, 1 << 16, 32, 256)) expect(q.perWarp).toBeLessThanOrEqual(q.perKey);
  });
});

describe('the scene\'s toy warp (beats 1 to 3)', () => {
  it('keys below 5 pass: lanes 1, 2, 4 and 5', () => {
    expect(sceneKeep().flatMap((k, l) => (k ? [l] : []))).toEqual([1, 2, 4, 5]);
  });
  it('two runs serve the four lanes in two different orders, neither the lane order, handing out 40 to 43', () => {
    const [a, b] = SCENE.seeds.map((s) => serveOneByOne(sceneKeep(), SCENE.counter, s));
    for (const run of [a, b]) {
      expect(run.map((x) => x.slot)).toEqual([40, 41, 42, 43]);
      expect([...run.map((x) => x.lane)].sort()).toEqual([1, 2, 4, 5]);
      expect(run.map((x) => x.lane)).not.toEqual([1, 2, 4, 5]);
    }
    expect(a.map((x) => x.lane)).not.toEqual(b.map((x) => x.lane));
  });
});
