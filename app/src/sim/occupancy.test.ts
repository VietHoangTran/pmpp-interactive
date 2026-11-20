import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { EX7_EXPECTED, EX8_EXPECTED, TABLE_1536, questionsS47, sizeTimes } from '../chapters/ch04/S47';
import type { BenchRow } from '../lib/reports';
import {
  BOOK, BOOK_BLOCK_SIZES, EX6, EX6_OPTIONS, EX7, EX7_ASSIGNMENTS, EX8, EX8_KERNELS, MAX_REGS_PER_THREAD, assignment,
  isSynthetic, limitsFromReport, notReported, occupancy, pct, regValues, regsAtFullOccupancy, scan, smallestFillingBlock,
  steps, threadValues, type Kernel, type Limits,
} from './occupancy';

const k = (threads: number, regs = 0, shared = 0): Kernel => ({ threads, regs, shared });
/** The book's example without registers or shared memory: the worked cases on pp. 85 to 86 use slots only. */
const SLOTS: Limits = { ...BOOK, regs: null, shared: null };
/** One row of NVIDIA's current table (facts A12): 16 blocks, 1536 threads, 64 K registers per SM. */
const T1536: Limits = { threadSlots: 1536, blockSlots: 16, blockSlotsDerived: false, regs: 65536, shared: null, width: 32, maxThreadsPerBlock: 1024 };

describe("the book's seven worked block sizes (pp. 85 to 86)", () => {
  const want: Record<number, { blocks: number; threads: number; occ: number; binding: string }> = {
    1024: { blocks: 2, threads: 2048, occ: 100, binding: 'threads' },
    512: { blocks: 4, threads: 2048, occ: 100, binding: 'threads' },
    256: { blocks: 8, threads: 2048, occ: 100, binding: 'threads' },
    128: { blocks: 16, threads: 2048, occ: 100, binding: 'threads' },
    64: { blocks: 32, threads: 2048, occ: 100, binding: 'threads+blocks' },
    32: { blocks: 32, threads: 1024, occ: 50, binding: 'blocks' },
    768: { blocks: 2, threads: 1536, occ: 75, binding: 'threads' },
  };
  for (const size of BOOK_BLOCK_SIZES)
    it(`${size} threads per block`, () => {
      const o = occupancy(SLOTS, k(size));
      expect(o.blocks).toBe(want[size].blocks);
      expect(o.residentThreads).toBe(want[size].threads);
      expect(pct(o.byWarps)).toBe(want[size].occ);
      expect(pct(o.byThreads)).toBe(want[size].occ);
      expect(o.binding.join('+')).toBe(want[size].binding);
    });
  it('768 leaves 512 thread slots idle with neither limit reached exactly', () => {
    const o = occupancy(SLOTS, k(768));
    expect(o.idleSlots).toBe(512);
    expect(o.rows.find((r) => r.key === 'blocks')?.fit).toBe(32);
  });
  it('64 is the smallest block that can fill the thread slots', () => {
    expect(smallestFillingBlock(BOOK)).toBe(64);
    expect(pct(occupancy(SLOTS, k(64)).byWarps)).toBe(100);
    expect(pct(occupancy(SLOTS, k(32)).byWarps)).toBe(50);
  });
});

describe('registers (p. 86)', () => {
  it('65,536 registers over 2048 threads is 32 per thread at full occupancy', () => {
    expect(regsAtFullOccupancy(BOOK)).toBe(32);
    expect(pct(occupancy(BOOK, k(512, 32)).byWarps)).toBe(100);
  });
  it('the cliff: 31 registers at 512 threads fits 4 blocks, 33 fits 3', () => {
    const a = occupancy(BOOK, k(512, 31));
    expect(a.blocks).toBe(4);
    expect(a.rows.find((r) => r.key === 'regs')?.need).toBe(15872);
    expect(a.blocks * 15872).toBe(63488);
    expect(pct(a.byWarps)).toBe(100);
    const b = occupancy(BOOK, k(512, 33));
    expect(b.blocks).toBe(3);
    expect(b.residentThreads).toBe(1536);
    expect(b.blocks * b.rows.find((r) => r.key === 'regs')!.need).toBe(50688);
    expect(2048 * 33).toBe(67584);
    expect(pct(b.byWarps)).toBe(75);
    expect(b.binding).toEqual(['regs']);
  });
  it('64 registers per thread caps occupancy at 50% whatever the block size', () => {
    const best = Math.max(...threadValues(BOOK).map((t) => occupancy(BOOK, k(t, 64)).byWarps));
    expect(pct(best)).toBe(50);
    for (const t of [64, 128, 256, 512, 1024]) expect(pct(occupancy(BOOK, k(t, 64)).byWarps)).toBe(50);
    expect(pct(occupancy(BOOK, k(32, 64)).byWarps)).toBe(50); // 32 blocks of one warp: block slots and registers together
  });
  it('a kernel too large for one SM gets no blocks', () => {
    const o = occupancy(BOOK, k(1024, 100));
    expect(o.blocks).toBe(0);
    expect(o.byWarps).toBe(0);
    expect(o.binding).toEqual(['regs']);
  });
});

describe('exercises 6, 7 and 8 (pp. 91 to 92)', () => {
  it('exercise 6: 512 threads per block puts the most threads on the SM', () => {
    const resident = EX6_OPTIONS.map((t) => occupancy(EX6, k(t)).residentThreads);
    expect(resident).toEqual([512, 1024, 1536, 1024]);
    expect(EX6_OPTIONS[resident.indexOf(Math.max(...resident))]).toBe(512);
    expect(occupancy(EX6, k(512)).blocks).toBe(3);
  });
  it('exercise 7: all five assignments are possible, at 50, 50, 50, 100 and 100%', () => {
    const r = EX7_ASSIGNMENTS.map((a) => assignment(EX7, a.blocks, a.threads));
    expect(r.map((x) => x.possible)).toEqual([true, true, true, true, true]);
    expect(r.map((x) => pct(x.occupancy))).toEqual([50, 50, 50, 100, 100]);
  });
  it('exercise 7: an assignment that needs more slots than there are is refused', () => {
    expect(assignment(EX7, 65, 32).possible).toBe(false);
    expect(assignment(EX7, 3, 1024).possible).toBe(false);
  });
  it('exercise 8: full; 50% by block slots; 87.5% by registers', () => {
    const [a, b, c] = EX8_KERNELS.map((kern) => occupancy(EX8, kern));
    expect([a.blocks, pct(a.byWarps)]).toEqual([16, 100]);
    expect(a.blocks * a.rows.find((r) => r.key === 'regs')!.need).toBe(61440);
    expect([b.blocks, pct(b.byWarps), b.binding.join('+')]).toEqual([32, 50, 'blocks']);
    expect([c.blocks, c.residentThreads, pct(c.byWarps), c.binding.join('+')]).toEqual([7, 1792, 87.5, 'regs']);
    expect(c.rows.find((r) => r.key === 'regs')?.need).toBe(8704);
  });
});

describe('a device with 1536 thread slots and 16 block slots (facts A12)', () => {
  it('768-thread blocks give 100%, not 75%', () => {
    const o = occupancy(T1536, k(768));
    expect([o.blocks, pct(o.byWarps)]).toEqual([2, 100]);
  });
  it('1024-thread blocks give 66.7%', () => {
    const o = occupancy(T1536, k(1024));
    expect([o.blocks, pct(o.byWarps)]).toEqual([1, 66.7]);
  });
  it('full occupancy allows 42 registers per thread, and 96 is the smallest block that fills it', () => {
    expect(regsAtFullOccupancy(T1536)).toBe(42);
    expect(smallestFillingBlock(T1536)).toBe(96);
    expect(pct(occupancy(T1536, k(96, 42)).byWarps)).toBe(100);
    expect(pct(occupancy(T1536, k(64, 42)).byWarps)).toBe(66.7); // 16 block slots of 2 warps
  });
  it('the summary page question: 96 threads at 48 registers gives 87.5%', () => {
    const o = occupancy(T1536, k(96, 48));
    expect([o.blocks, o.residentWarps, pct(o.byWarps), o.binding.join('+')]).toEqual([14, 42, 87.5, 'regs']);
  });
});

describe('two definitions of occupancy (pp. 85 and 90)', () => {
  it('48-thread blocks: 100% by warps, 75% by threads', () => {
    const o = occupancy(SLOTS, k(48));
    expect(o.warpsPerBlock).toBe(2);
    expect(o.blocks).toBe(32);
    expect(o.residentWarps).toBe(64);
    expect(o.residentThreads).toBe(1536);
    expect(pct(o.byWarps)).toBe(100);
    expect(pct(o.byThreads)).toBe(75);
    expect(o.idleSlots).toBe(0);
    expect(o.paddedLanes).toBe(512);
  });
  it('the two agree whenever the block is whole warps', () => {
    for (const t of threadValues(BOOK)) {
      const o = occupancy(BOOK, k(t, 20, 4096));
      expect(o.byWarps).toBeCloseTo(o.byThreads, 12);
      expect(o.paddedLanes).toBe(0);
    }
  });
});

describe('shared memory joins as a fourth budget (chapter 5, p. 116)', () => {
  it("the book's 38 KB blocks of 256 threads: 6 blocks, 75%", () => {
    const o = occupancy(BOOK, k(256, 0, 38 * 1024));
    expect([o.blocks, pct(o.byWarps), o.binding.join('+')]).toEqual([6, 75, 'shared']);
  });
  it('a budget the kernel does not use is not counted', () => {
    const o = occupancy(BOOK, k(256));
    expect(o.rows.find((r) => r.key === 'regs')?.why).toBe('unused');
    expect(o.rows.find((r) => r.key === 'shared')?.fit).toBeNull();
  });
});

describe('the staircase', () => {
  it('occupancy against registers at 512 threads per block falls 100, 75, 50, 25, 0', () => {
    const pts = scan(BOOK, k(512), 'regs', regValues());
    expect(pts).toHaveLength(MAX_REGS_PER_THREAD);
    expect(steps(pts).map((s) => [s.at, pct(s.from), pct(s.to)])).toEqual([
      [33, 100, 75],
      [43, 75, 50],
      [65, 50, 25],
      [129, 25, 0],
    ]);
  });
  it('occupancy against block size at 0 registers dips wherever the size does not divide the slots', () => {
    const pts = scan(SLOTS, k(0), 'threads', threadValues(SLOTS));
    expect(pts[0]).toMatchObject({ x: 32, blocks: 32 });
    expect(pct(pts.find((p) => p.x === 768)!.occupancy)).toBe(75);
    expect(pct(pts.find((p) => p.x === 1024)!.occupancy)).toBe(100);
    expect(pct(pts.find((p) => p.x === 96)!.occupancy)).toBe(98.4); // 21 blocks of 3 warps
    expect(Math.min(...pts.map((p) => p.occupancy))).toBe(0.5);
  });
  it('a step can cost half of the SM', () => {
    const pts = scan(BOOK, k(1024), 'regs', regValues());
    expect(steps(pts).map((s) => [s.at, pct(s.from), pct(s.to)])).toEqual([
      [33, 100, 50],
      [65, 50, 0],
    ]);
  });
  it('threadValues lists whole warps up to the largest block', () => {
    expect(threadValues(BOOK)).toHaveLength(32);
    expect(threadValues({ ...BOOK, width: 64 })).toHaveLength(16);
  });
});

describe("section 4.7's answers come from this model", () => {
  const num = (id: string) => {
    const q = questionsS47[id];
    if (q.kind !== 'numeric') throw new Error(id + ' is not numeric');
    return q.answer;
  };
  it('the exercise checker expects 50, 50, 50, 100, 100 for exercise 7 and full, block slots, registers for 8', () => {
    expect(EX7_EXPECTED).toEqual(['50', '50', '50', '100', '100']);
    expect(EX8_EXPECTED).toEqual(['full', 'blocks', 'regs']);
  });
  it('the numeric questions: 87.5 for exercise 8(c), 100 for 768 threads on 1536 / 16, 50 at 64 registers', () => {
    expect(num('ch04/4.7/q-ex8c')).toBe(87.5);
    expect(num('ch04/4.7/q-768')).toBe(100);
    expect(num('ch04/4.7/q-regs64')).toBe(50);
    expect(TABLE_1536).toEqual({ threadSlots: 1536, blockSlots: 16, blockSlotsDerived: false, regs: 65536, shared: null, width: 32, maxThreadsPerBlock: 1024 });
  });
  it("the measured reveal reads lab 4.4's sweep rows, one per block size, and nothing else", () => {
    const main = fs.readFileSync(path.join(path.resolve(__dirname, '../../..'), 'labs/ch04/lab4_blocksize/main.cpp'), 'utf8');
    expect(main).toContain('bench_row(a, "sweep").n("threads_per_block"');
    const rows: BenchRow[] = [
      { name: 'problem', outputs: 1000, width: 32 },
      { name: 'sweep', threads_per_block: 64, seconds: 0.002 },
      { name: 'sweep', threads_per_block: 32, seconds: 0.004 },
      { name: 'sweep', threads_per_block: 32, seconds: 0.009 }, // a repeat of a size keeps the first
      { name: 'summary', fastest_threads_per_block: 64, fastest_seconds: 0.002 },
    ];
    expect(sizeTimes(rows)).toEqual([{ size: 32, seconds: 0.004 }, { size: 64, seconds: 0.002 }]);
  });
});

describe('limits from a device report', () => {
  const cuda = {
    name: 'some CUDA device', warpSize: '32', maxThreadsPerBlock: '1024', maxThreadsPerMultiProcessor: '1536',
    maxBlocksPerMultiProcessor: '16', regsPerMultiprocessor: '65536', sharedMemPerMultiprocessor: '102400',
  };
  it('reads the six fields of a CUDA report', () => {
    expect(limitsFromReport(cuda, 'cuda')).toEqual({
      threadSlots: 1536, blockSlots: 16, blockSlotsDerived: false, regs: 65536, shared: 102400, width: 32, maxThreadsPerBlock: 1024,
    });
  });
  it('marks HIP block slots as derived and does not count them', () => {
    const hip = { ...cuda, warpSize: '64', maxThreadsPerMultiProcessor: '2048', maxBlocksPerMultiProcessor: '32' };
    const l = limitsFromReport(hip, 'hip')!;
    expect(l.blockSlotsDerived).toBe(true);
    const o = occupancy({ ...l, regs: null, shared: null }, k(64));
    expect(o.rows.find((r) => r.key === 'blocks')).toMatchObject({ fit: null, why: 'derived' });
    // One-wave blocks fill every wave slot: the book's 50% case has no counterpart here.
    expect([o.blocks, pct(o.byWarps), o.binding.join('+')]).toEqual([32, 100, 'threads']);
    expect(smallestFillingBlock(l)).toBeNull();
  });
  it('rounds a block up to whole waves with the width the device reports', () => {
    const l: Limits = { threadSlots: 2048, blockSlots: 32, blockSlotsDerived: true, regs: null, shared: null, width: 64, maxThreadsPerBlock: 1024 };
    const o = occupancy(l, k(65));
    expect([o.warpsPerBlock, o.slotsPerBlock, o.blocks]).toEqual([2, 128, 16]);
  });
  it('treats a field of 0 as not reported, and says which', () => {
    const hip = { ...cuda, maxThreadsPerMultiProcessor: '0', regsPerMultiprocessor: 0 };
    const l = limitsFromReport(hip, 'hip')!;
    expect(l.threadSlots).toBeNull();
    expect(l.regs).toBeNull();
    expect(notReported(hip)).toEqual(['maxThreadsPerMultiProcessor', 'regsPerMultiprocessor']);
    const o = occupancy(l, k(256, 30));
    expect(o.usable).toBe(false);
    expect(o.byWarps).toBe(0);
    expect(o.rows.find((r) => r.key === 'threads')?.why).toBe('not-reported');
  });
  it("refuses the emulators' synthetic values", () => {
    const emu = { ...cuda, name: 'pmpp CPU emulator (not a GPU)', note: 'synthetic values describing the CPU emulator, not a GPU' };
    expect(limitsFromReport(emu, 'emu')).toBeNull();
    expect(limitsFromReport(emu, 'race')).toBeNull();
    expect(limitsFromReport(emu, 'cuda')).toBeNull(); // the note alone is enough
    expect(isSynthetic('cuda', cuda)).toBe(false);
  });
  it('returns nothing for a Metal report, which has none of the per-SM fields', () => {
    expect(limitsFromReport({ name: 'a Metal device', maxThreadgroupMemoryLength: '32768', 'maxThreadsPerThreadgroup.width': '1024' }, 'metal')).toBeNull();
  });
  it('without a reported width a block is counted thread by thread, and says so', () => {
    const o = occupancy({ ...SLOTS, width: null }, k(48));
    expect(o.widthAssumed).toBe(true);
    expect(o.blocks).toBe(32);
    expect(pct(o.byThreads)).toBe(75);
  });
  it('flags a block larger than the device accepts', () => {
    expect(occupancy(BOOK, k(2048)).tooLarge).toBe(true);
    expect(occupancy(BOOK, k(1024)).tooLarge).toBe(false);
  });
});
