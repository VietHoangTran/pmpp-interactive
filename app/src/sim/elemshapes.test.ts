import { describe, expect, it } from 'vitest';
import {
  BLOCK_CHOICES,
  N_READS_CHOICES,
  WARP,
  divergentWarps,
  footprint,
  isAligned,
  launch,
  tailSummary,
  threadsFor,
  type LaunchCfg,
  type Style,
} from './elemshapes';

const cfg = (p: Partial<LaunchCfg>): LaunchCfg => ({ style: 'binary_vv', n: 100, block: 32, nReads: 4, offset: 0, ...p });
const STYLES: Style[] = ['per-element', 'binary_vv', 'grid-stride'];

describe('every element is written exactly once', () => {
  it('for all three styles, n 1 to 300, every block size, N_READS and offset', () => {
    for (const style of STYLES)
      for (const block of BLOCK_CHOICES)
        for (const nReads of style === 'binary_vv' ? N_READS_CHOICES : [1])
          for (const offset of style === 'binary_vv' ? [0, 1, 2, 3] : [0])
            for (let n = 1; n <= 300; n++) {
              const l = launch({ style, n, block, nReads, offset });
              expect(l.writes.every((w) => w === 1), `${style} n=${n} B=${block} N=${nReads} off=${offset}`).toBe(true);
            }
  });
  it('a thread never writes an element past n', () => {
    const l = launch(cfg({ n: 7, nReads: 8 }));
    expect(l.plans.flatMap((p) => p.elements).every((e) => e < 7)).toBe(true);
  });
});

describe('binary_vv: the tail', () => {
  it('the question: N_READS 4 on 1002 elements, 251 threads, tail is 2 elements in thread 250', () => {
    // The question states ceil(1002 / 4) = 251 threads; the model rounds up to whole blocks,
    // and the extra threads take the tail branch with nothing to do.
    expect(Math.ceil(1002 / 4)).toBe(251);
    const l = launch(cfg({ n: 1002, nReads: 4, block: 32 }));
    const tail = l.plans.filter((p) => p.path === 'tail');
    expect(tail.map((p) => p.t)).toEqual([250]);
    expect(tail[0].elements).toEqual([1000, 1001]);
    expect(tailSummary(l)).toEqual({ elements: 2, threads: 1 });
    expect(l.plans.slice(0, 250).every((p) => p.path === 'vector')).toBe(true);
  });
  it('touches n mod N_READS elements in at most one thread, for every n and N_READS', () => {
    for (const nReads of N_READS_CHOICES)
      for (let n = 1; n <= 300; n++) {
        const s = tailSummary(launch(cfg({ n, nReads })));
        expect(s.elements).toBe(n % nReads);
        expect(s.threads).toBe(n % nReads ? 1 : 0);
      }
  });
  it('splits the branch in at most one warp', () => {
    for (const nReads of N_READS_CHOICES)
      for (const block of BLOCK_CHOICES)
        for (let n = 1; n <= 300; n++) expect(divergentWarps(launch(cfg({ n, nReads, block }))).length).toBeLessThanOrEqual(1);
  });
});

describe('binary_vv: the alignment test', () => {
  it('is_aligned is a multiple of the vector width in bytes', () => {
    expect(isAligned(0, 4)).toBe(true);
    expect(isAligned(1, 4)).toBe(false);
    expect(isAligned(2, 2)).toBe(true); // 8 bytes is enough for a 2-wide float vector
    expect(isAligned(2, 4)).toBe(false);
    expect(isAligned(3, 1)).toBe(true); // N_READS 1 is always "aligned"
  });
  it('an offset that is not a multiple of N_READS sends every main-branch thread to the scalar path, same result', () => {
    for (const nReads of [2, 4, 8])
      for (const offset of [1, 2, 3]) {
        if (offset % nReads === 0) continue;
        const l = launch(cfg({ n: 257, nReads, offset }));
        expect(l.plans.some((p) => p.path === 'vector')).toBe(false);
        expect(l.plans.filter((p) => p.branch === 'main').every((p) => p.path === 'misaligned')).toBe(true);
        expect(l.writes.every((w) => w === 1)).toBe(true);
        const same = launch(cfg({ n: 257, nReads, offset: 0 }));
        expect(l.writer).toEqual(same.writer); // each thread still owns the same elements
      }
  });
  it('for N_READS 4 and 8, every nonzero offset of 1 to 3 floats is misaligned', () => {
    for (const nReads of [4, 8]) for (const offset of [1, 2, 3]) expect(isAligned(offset, nReads)).toBe(false);
  });
  it('a warp of vector loads reads one unbroken run; the misaligned loop reads with a stride of N_READS', () => {
    const v = launch(cfg({ n: 256, nReads: 4 }));
    const fv = footprint(v.warps[0].loads[0], 0);
    expect(v.warps[0].loads.length).toBe(1);
    expect(fv).toEqual({ lo: 0, hi: 127, read: 128, contiguous: true });
    const m = launch(cfg({ n: 256, nReads: 4, offset: 1 }));
    expect(m.warps[0].loads.length).toBe(4);
    const f0 = footprint(m.warps[0].loads[0], 1);
    expect(f0).toEqual({ lo: 1, hi: 125, read: 32, contiguous: false });
    const lanes = m.warps[0].loads[0].lanes.map((x) => x.element);
    expect(lanes.slice(0, 3)).toEqual([0, 4, 8]);
  });
});

describe('one thread per element and the grid-stride loop', () => {
  it('the guard idles the threads past n, and splits at most one warp', () => {
    for (const block of BLOCK_CHOICES)
      for (let n = 1; n <= 300; n++) {
        const l = launch(cfg({ style: 'per-element', n, block }));
        expect(l.threads).toBe(Math.ceil(n / block) * block);
        expect(l.plans.filter((p) => p.path === 'idle').length).toBe(l.threads - n);
        expect(divergentWarps(l).length).toBe(n % WARP ? 1 : 0);
      }
  });
  it('a grid-stride loop covers any n with a fixed grid', () => {
    for (const block of BLOCK_CHOICES) {
      const sizes = new Set<number>();
      for (let n = 1; n <= 300; n++) {
        const l = launch(cfg({ style: 'grid-stride', n, block }));
        sizes.add(l.threads);
        expect(l.writes.every((w) => w === 1)).toBe(true);
      }
      expect([...sizes]).toEqual([threadsFor(cfg({ style: 'grid-stride', block }))]);
    }
  });
  it('thread t of a 64-thread grid writes t, t + 64, t + 128 ...', () => {
    const l = launch(cfg({ style: 'grid-stride', n: 150, block: 32 }));
    expect(l.threads).toBe(64);
    expect(l.plans[5].elements).toEqual([5, 69, 133]);
    expect(l.plans[40].elements).toEqual([40, 104]);
    // 150 = 2 * 64 + 22: threads 0..21 run three trips, the rest two, so warp 0 splits.
    expect(divergentWarps(l)).toEqual([0]);
  });
  it('the trip count splits at most one warp', () => {
    for (const block of BLOCK_CHOICES)
      for (let n = 1; n <= 300; n++) expect(divergentWarps(launch(cfg({ style: 'grid-stride', n, block }))).length).toBeLessThanOrEqual(1);
  });
});
