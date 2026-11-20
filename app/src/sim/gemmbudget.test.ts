import { describe, expect, it } from 'vitest';
import {
  BOOK_LIMITS,
  BOOK_SHAPE,
  budget,
  occupancyPct,
  regsForBlocks,
  sharedA,
  sharedB,
  sharedBytes,
  tileRegisters,
} from './gemmbudget';

// Every number section 15.7 states, and the ones its questions use.
const at = (extra: number, threadsPerSm = BOOK_LIMITS.threadsPerSm) =>
  budget({ ...BOOK_SHAPE, extra }, { ...BOOK_LIMITS, threadsPerSm });

describe('registers (book p. 363)', () => {
  it('an 8 x 8 tile with its two strips needs 80 registers', () => {
    expect(tileRegisters(8, 8)).toBe(80);
    expect(at(0).wanted).toBe(80);
  });
  it('at 255 registers, 65,536 / 255 = 257 threads: one block of 256, 12.5 % of 2048', () => {
    const b = at(175);
    expect(b.regs).toBe(255);
    expect(b.threadBound).toBe(257);
    expect(b.blocks).toBe(1);
    expect(b.binding).toEqual(['registers']);
    expect(b.resident).toBe(256);
    expect(b.occupancy).toBe(12.5);
  });
  it('the same block is 16.7 % of a 1536-thread SM and 25 % of a 1024-thread SM (ch15-facts A3)', () => {
    expect(at(175, 1536).blocks).toBe(1);
    expect(at(175, 1536).occupancy!.toFixed(1)).toBe('16.7');
    expect(at(175, 1024).occupancy).toBe(25);
  });
  it('at 128 registers two blocks fit: 512 threads, 25 %', () => {
    const b = at(48);
    expect(b.regs).toBe(128);
    expect(b.blocks).toBe(2);
    expect(b.resident).toBe(512);
    expect(b.occupancy).toBe(25);
    expect(regsForBlocks(65536, 256, 2)).toBe(128);
    expect(regsForBlocks(65536, 256, 8)).toBe(32);
  });
  it('at 80 registers registers still bind: 3 blocks, 37.5 %', () => {
    const b = at(0);
    expect(b.byLimit.registers).toBe(3);
    expect(b.byLimit.threads).toBe(8);
    expect(b.blocks).toBe(3);
    expect(b.occupancy).toBe(37.5);
  });
  it('asking for more than 255 is capped at 255 and flagged as spilling', () => {
    const b = at(300);
    expect(b.wanted).toBe(380);
    expect(b.regs).toBe(255);
    expect(b.spills).toBe(true);
    expect(at(175).spills).toBe(false);
  });
  it('256 threads of 8 x 8 cover a 128 x 128 tile exactly', () => {
    expect(at(0).covers).toBe(true);
    expect(budget({ ...BOOK_SHAPE, threads: 128 }, BOOK_LIMITS).covers).toBe(false);
  });
  it('occupancy helper', () => {
    expect(occupancyPct(1, 256, 2048)).toBe(12.5);
  });
});

describe('shared memory (p. 363, sections 15.6 and 15.8)', () => {
  it('a 128 x 8 tile and an 8 x 128 tile are 4 KB each', () => {
    expect(sharedA(BOOK_SHAPE)).toBe(4096);
    expect(sharedB(BOOK_SHAPE)).toBe(4096);
    expect(sharedBytes(BOOK_SHAPE)).toBe(8192);
  });
  it('one padding column makes A 4608 B and the pair 8704 B', () => {
    const padded = { ...BOOK_SHAPE, pad: 1 };
    expect(sharedA(padded)).toBe(4608);
    expect(sharedBytes(padded)).toBe(8704);
  });
  it('padded and double-buffered: 17,408 B', () => {
    expect(sharedBytes({ ...BOOK_SHAPE, pad: 1, buffers: 2 })).toBe(17408);
  });
  it('shared memory does not bind on the book example: 228 KB / 8 KB = 28 blocks', () => {
    expect(at(175).byLimit.shared).toBe(28);
  });
  it('per-block limit check', () => {
    const l = { ...BOOK_LIMITS, sharedPerBlock: 16384 };
    expect(budget({ ...BOOK_SHAPE, pad: 1, buffers: 2 }, l).fitsBlock).toBe(false);
    expect(budget({ ...BOOK_SHAPE, pad: 1 }, l).fitsBlock).toBe(true);
    expect(budget(BOOK_SHAPE, BOOK_LIMITS).fitsBlock).toBeNull();
  });
});

describe('unknown limits (Metal)', () => {
  it('without registers or thread slots no block count or occupancy is computed', () => {
    const b = budget(BOOK_SHAPE, { regsPerSm: 0, threadsPerSm: 0, maxRegsPerThread: 0, sharedPerBlock: 32768, sharedPerSm: 0 });
    expect(b.blocks).toBeNull();
    expect(b.occupancy).toBeNull();
    expect(b.threadBound).toBeNull();
    expect(b.spills).toBe(false);
    expect(b.fitsBlock).toBe(true);
  });
});
