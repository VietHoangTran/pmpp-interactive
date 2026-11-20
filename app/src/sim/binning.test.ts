import { describe, expect, it } from 'vitest';
import {
  BINS,
  BOOK_SHARE,
  DISTRIBUTIONS,
  GRID,
  POINTS_PER_BIN,
  bestCapacity,
  binCounts,
  binWith,
  capacityForShare,
  countsFor,
  histogram,
  makeAtoms,
  neighbourCounts,
  neighbourPairs,
  work,
} from './binning';

const sum = (a: readonly number[]) => a.reduce((s, x) => s + x, 0);
const SIZES = [1024, 4096, 16384];

describe('the generator', () => {
  it('puts every atom inside the domain', () => {
    for (const d of DISTRIBUTIONS)
      for (const a of makeAtoms(d, 4096)) {
        expect(a.x).toBeGreaterThanOrEqual(0);
        expect(a.x).toBeLessThan(GRID);
        expect(a.y).toBeGreaterThanOrEqual(0);
        expect(a.y).toBeLessThan(GRID);
      }
  });
  it('is deterministic for a seed and changes with it', () => {
    expect(countsFor('clustered', 4096)).toEqual(countsFor('clustered', 4096));
    expect(countsFor('clustered', 4096, GRID, 1)).not.toEqual(countsFor('clustered', 4096, GRID, 2));
  });
  it('conserves atoms: the bins hold all of them', () => {
    for (const d of DISTRIBUTIONS) for (const n of SIZES) expect(sum(countsFor(d, n))).toBe(n);
  });
  it('a dense pocket makes the fullest bin far fuller than a uniform spread does', () => {
    const u = Math.max(...countsFor('uniform', 4096));
    const p = Math.max(...countsFor('pocket', 4096));
    expect(p).toBeGreaterThan(2 * u);
  });
  it('bins by the integer part of each coordinate', () => {
    const c = binCounts([{ x: 0.5, y: 0.5 }, { x: 15.9, y: 0.1 }, { x: 1.2, y: 2.7 }]);
    expect(c[0]).toBe(1);
    expect(c[15]).toBe(1);
    expect(c[2 * GRID + 1]).toBe(1);
  });
  it('the histogram counts every bin once', () => {
    const c = countsFor('pocket', 4096);
    const h = histogram(c);
    expect(sum(h)).toBe(BINS);
    expect(h.length).toBe(Math.max(...c) + 1);
  });
});

describe('fixed capacity', () => {
  it('real atoms plus overflow atoms are all the atoms, for every capacity', () => {
    for (const d of DISTRIBUTIONS) {
      const c = countsFor(d, 4096);
      for (let k = 0; k <= Math.max(...c); k += 3) {
        const b = binWith(c, k);
        expect(b.real + b.overflow).toBe(4096);
      }
    }
  });
  it('padding plus real atoms fills bins x capacity', () => {
    const c = countsFor('clustered', 4096);
    for (const k of [0, 1, 7, 16, 40, 200]) {
      const b = binWith(c, k);
      expect(b.dummies + b.real).toBe(BINS * k);
      expect(b.slots).toBe(BINS * k);
      expect(b.dummies).toBeGreaterThanOrEqual(0);
    }
  });
  it('the overflow share never grows with the capacity', () => {
    for (const d of DISTRIBUTIONS)
      for (const n of SIZES) {
        const c = countsFor(d, n);
        let prev = Infinity;
        for (let k = 0; k <= Math.max(...c) + 2; k++) {
          const s = binWith(c, k).overflowShare;
          expect(s).toBeLessThanOrEqual(prev);
          prev = s;
        }
      }
  });
  it('a capacity equal to the fullest bin leaves nothing to overflow, and one less does', () => {
    for (const d of DISTRIBUTIONS) {
      const c = countsFor(d, 4096);
      const full = Math.max(...c);
      expect(binWith(c, full).overflow).toBe(0);
      expect(binWith(c, full - 1).overflow).toBeGreaterThan(0);
    }
  });
  it("the smallest capacity under the book's 3% is that, and one less is not", () => {
    for (const d of DISTRIBUTIONS) {
      const c = countsFor(d, 4096);
      const k = capacityForShare(c, BOOK_SHARE);
      expect(binWith(c, k).overflowShare).toBeLessThanOrEqual(BOOK_SHARE);
      expect(binWith(c, k - 1).overflowShare).toBeGreaterThan(BOOK_SHARE);
    }
  });
});

describe('work', () => {
  it('the clipped 3 x 3 neighbourhoods add up to (3G - 2)^2 pairs', () => {
    expect(sum(neighbourCounts())).toBe(neighbourPairs());
    expect(neighbourPairs()).toBe(46 ** 2);
    expect(neighbourCounts()[0]).toBe(4);
    expect(neighbourCounts()[GRID + 1]).toBe(9);
  });
  it('the device scans every slot of every neighbour bin for every grid point', () => {
    const c = countsFor('uniform', 4096);
    expect(work(c, 20, 10).device).toBe(POINTS_PER_BIN * 20 * neighbourPairs());
  });
  it('with no overflow the host has nothing to do; the wasted share is the dummies', () => {
    const c = countsFor('pocket', 4096);
    const full = Math.max(...c);
    const w = work(c, full, 10);
    expect(w.host).toBe(0);
    expect(w.serial).toBe(w.deviceTime);
    expect(w.wasted).toBeGreaterThan(0.5 * w.device);
  });
  it('each overflow atom costs the host what one stored atom costs the device', () => {
    // One overflow atom alone in a corner bin, capacity 0: the host tests it against the
    // 4 corner-neighbourhood bins' grid points.
    const c = new Array(BINS).fill(0);
    c[0] = 1;
    expect(work(c, 0, 10).host).toBe(POINTS_PER_BIN * 4);
  });
  it('serial = device / S + host; overlapped = the larger of the two', () => {
    const c = countsFor('clustered', 4096);
    const w = work(c, 30, 25);
    expect(w.deviceTime).toBeCloseTo(w.device / 25, 9);
    expect(w.serial).toBeCloseTo(w.deviceTime + w.host, 6);
    expect(w.overlapped).toBe(Math.max(w.deviceTime, w.host));
  });
  it('the best capacity is a local minimum, and beats sizing for the fullest bin on a dense pocket', () => {
    for (const s of [5, 20, 100]) {
      const c = countsFor('pocket', 4096);
      const k = bestCapacity(c, s);
      const t = (x: number) => work(c, x, s).serial;
      expect(t(k)).toBeLessThanOrEqual(t(k + 1));
      if (k > 0) expect(t(k)).toBeLessThan(t(k - 1));
      expect(t(k)).toBeLessThan(t(Math.max(...c)));
    }
  });
  it('a very fast device makes overflow expensive: the best capacity rises with S', () => {
    const c = countsFor('clustered', 4096);
    expect(bestCapacity(c, 100)).toBeGreaterThan(bestCapacity(c, 5));
  });
});
