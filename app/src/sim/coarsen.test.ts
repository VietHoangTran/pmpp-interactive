import { describe, expect, it } from 'vitest';
import { warpAccess } from './coalesce';
import {
  BOOK_CAPACITY,
  PITFALLS,
  SECTOR_BYTES,
  calculate,
  deviceSlots,
  elementsOf,
  grid,
  largestWholeWaveFactor,
  residentUpperBound,
  threadsByRegisters,
  warpInstruction,
  warpVectorInstruction,
  waveFill,
  waveTime,
  waves,
  wholeWaveFactors,
} from './coarsen';

const TPB = 256;

describe("the book's wave arithmetic (p. 143)", () => {
  it('1056 blocks on 264 at once are 4 full waves', () => {
    expect(waves(1056, BOOK_CAPACITY)).toBe(4);
    expect(waveFill(1056, BOOK_CAPACITY)).toMatchObject({ waveCount: 4, whole: true, idle: 0, tail: 0 });
  });
  it('coarsening by 8 leaves 132 blocks: half a wave, half the device idle', () => {
    const r = calculate(1056 * TPB, TPB, 8, BOOK_CAPACITY);
    expect(r.blocks).toBe(132);
    expect(r.waves).toBe(0.5);
    expect(r.fill.perWave).toEqual([132]);
    expect(r.fill.idle).toBe(0.5);
  });
  it('1584 blocks are 6 waves; coarsening by 4 leaves 396 = 1.5 waves', () => {
    expect(waves(1584, BOOK_CAPACITY)).toBe(6);
    const r = calculate(1584 * TPB, TPB, 4, BOOK_CAPACITY);
    expect(r.blocks).toBe(396);
    expect(r.waves).toBe(1.5);
    expect(r.fill.perWave).toEqual([264, 132]);
    expect(r.fill.idle).toBe(0.25);
    expect(r.fill.tail).toBe(132);
  });
  it('whole-wave factors for the two cases; 6 is the largest for 1584 blocks', () => {
    expect(wholeWaveFactors(1056 * TPB, TPB, BOOK_CAPACITY)).toEqual([1, 2, 4]);
    expect(largestWholeWaveFactor(1056 * TPB, TPB, BOOK_CAPACITY)).toBe(4);
    expect(wholeWaveFactors(1584 * TPB, TPB, BOOK_CAPACITY)).toEqual([1, 2, 3, 6]);
    expect(largestWholeWaveFactor(1584 * TPB, TPB, BOOK_CAPACITY)).toBe(6);
  });
  it('wave time, with no overhead saved: a partial wave costs, whole waves do not', () => {
    // Uncoarsened: 4 waves of one unit each.
    expect(waveTime(1056, BOOK_CAPACITY, 1)).toBe(4);
    expect(waveTime(528, BOOK_CAPACITY, 2)).toBe(4);
    expect(waveTime(264, BOOK_CAPACITY, 4)).toBe(4);
    // Half a wave of blocks eight times as long.
    expect(waveTime(132, BOOK_CAPACITY, 8)).toBe(8);
    expect(waveTime(396, BOOK_CAPACITY, 4)).toBe(8);
    expect(waveTime(1584, BOOK_CAPACITY, 1)).toBe(6);
  });
});

describe('q-factor: 102,400 units, 256 threads per block, 100 blocks at once', () => {
  it('400 blocks at factor 1 (4 waves), 100 at factor 4 (one wave), 50 at factor 8', () => {
    expect(grid(102400, TPB, 1).blocks).toBe(400);
    expect(grid(102400, TPB, 4).blocks).toBe(100);
    expect(grid(102400, TPB, 8).blocks).toBe(50);
    expect(waveFill(50, 100).idle).toBe(0.5);
  });
  it('the largest factor that fills every slot of every wave is 4', () => {
    expect(largestWholeWaveFactor(102400, TPB, 100)).toBe(4);
    expect(wholeWaveFactors(102400, TPB, 100)).toEqual([1, 2, 4]);
  });
});

describe('the reader-device bound', () => {
  it('is SMs times the smaller of block slots and thread slots', () => {
    // The book's example SM (ch. 4, p. 85): 32 blocks, 2048 threads; 132 SMs.
    const book = { units: 132, blocksPerUnit: 32, threadsPerUnit: 2048, warpWidth: 32 };
    expect(residentUpperBound(book, 1024)).toBe(264);
    expect(residentUpperBound(book, 256)).toBe(1056);
    expect(residentUpperBound(book, 32)).toBe(4224);
  });
  it('rounds a block up to whole warps and ignores a thread limit reported as zero', () => {
    expect(residentUpperBound({ units: 10, blocksPerUnit: 40, threadsPerUnit: 2560, warpWidth: 64 }, 65)).toBe(200);
    expect(residentUpperBound({ units: 10, blocksPerUnit: 40, threadsPerUnit: 0, warpWidth: 64 }, 65)).toBe(400);
  });
  it('reads device reports, whose values are strings, and rejects a report without the fields', () => {
    expect(deviceSlots({ multiProcessorCount: '132', maxBlocksPerMultiProcessor: '32', maxThreadsPerMultiProcessor: '2048', warpSize: '32' }))
      .toEqual({ units: 132, blocksPerUnit: 32, threadsPerUnit: 2048, warpWidth: 32 });
    expect(deviceSlots({ name: 'Apple M3 Max', maxThreadgroupMemoryLength: '32768' })).toBeNull();
  });
});

describe('who handles what (our analysis with the 6.1 rule, NVIDIA 32-byte sectors)', () => {
  it("the book's mapping: thread 0 handles 0..3, thread 1 handles 4..7 (p. 142)", () => {
    expect(elementsOf(0, 4, 'contiguous', 64)).toEqual([0, 1, 2, 3]);
    expect(elementsOf(1, 4, 'contiguous', 64)).toEqual([4, 5, 6, 7]);
    expect(elementsOf(1, 4, 'strided', 64)).toEqual([1, 65, 129, 193]);
  });
  it('contiguous mapping at C = 4: lanes 16 bytes apart, 16 sectors per instruction, a quarter of each used', () => {
    for (let c = 0; c < 4; c++) {
      const a = warpAccess(warpInstruction(c, 4, 'contiguous', 32, 64), 4, SECTOR_BYTES);
      expect(a.transactions).toBe(16);
      expect(a.efficiency).toBe(0.25);
    }
  });
  it('strided mapping at C = 4: 4 sectors per instruction, fully used', () => {
    for (let c = 0; c < 4; c++) {
      const a = warpAccess(warpInstruction(c, 4, 'strided', 32, 64), 4, SECTOR_BYTES);
      expect(a.transactions).toBe(4);
      expect(a.efficiency).toBe(1);
    }
  });
  it('the contiguous mapping costs C times the sectors per instruction for C = 2 and 8 too', () => {
    expect(warpAccess(warpInstruction(0, 2, 'contiguous', 32, 64), 4, SECTOR_BYTES).transactions).toBe(8);
    expect(warpAccess(warpInstruction(0, 8, 'contiguous', 32, 64), 4, SECTOR_BYTES).transactions).toBe(32);
    expect(warpAccess(warpInstruction(0, 1, 'contiguous', 32, 64), 4, SECTOR_BYTES).transactions).toBe(4);
  });
  it('a 16-byte vector load is 16 sectors, fully used, in one instruction (ch06-facts A11)', () => {
    const a = warpAccess(warpVectorInstruction(32), 16, SECTOR_BYTES);
    expect(a.transactions).toBe(16);
    expect(a.efficiency).toBe(1);
  });
});

describe('pitfall cards and the occupancy question', () => {
  it('answers', () => {
    expect(Object.fromEntries(PITFALLS.map((p) => [p.id, p.answer]))).toEqual({
      relu: 'little or no change',
      vector: 'little or no change',
      'matmul-tiles': 'helps',
      'below-wave': 'can hurt',
      registers: 'can hurt',
    });
  });
  it("q-occupancy: 64 registers per thread on the book's SM leave 1024 of 2048 threads, 50%", () => {
    expect(threadsByRegisters(65536, 2048, 32)).toBe(2048);
    expect(threadsByRegisters(65536, 2048, 64)).toBe(1024);
    expect((100 * threadsByRegisters(65536, 2048, 64)) / 2048).toBe(50);
  });
});
