import { describe, expect, it } from 'vitest';
import {
  LAYOUTS, blockStride, codeAddress, iterationPlan, iterationSpan, loadStep, loadsPerLane, problem, rowSteps, scaleAddress,
  scaleSpan, segmentBytesOf, type Mapping,
} from './qlayout';

const q4 = (o: Partial<Mapping> = {}): Mapping => ({ layout: 'q4_0-blocks', lanesPerBlock: 1, loadBytes: 2, warpWidth: 32, inW: 4096, ...o });

describe('layouts', () => {
  it("Q4_0 blocks are 18 bytes, codes after the 2-byte scale (llama.cpp's block_q4_0)", () => {
    expect(blockStride(LAYOUTS['q4_0-blocks'])).toBe(18);
    expect(codeAddress(q4(), 0, 0)).toBe(2);
    expect(codeAddress(q4(), 1, 0)).toBe(20);
    expect(scaleAddress(q4(), 3)).toBe(54);
  });
  it('split layouts keep codes contiguous and scales in their own array', () => {
    const m: Mapping = { layout: 'q8_0-split', lanesPerBlock: 8, loadBytes: 4, warpWidth: 32, inW: 4096 };
    expect(codeAddress(m, 2, 5)).toBe(69);
    expect(scaleAddress(m, 2)).toBe(4);
    expect(codeAddress({ ...m, row: 1 }, 0, 0)).toBe(4096);
  });
});

describe('coalescing of one load instruction (32-byte segments)', () => {
  it('one lane per Q4_0 block, 2-byte loads: 18 segments for 64 useful bytes, about 11 %', () => {
    const s = loadStep(q4(), 0, 0, 32);
    expect(s.code.transactions).toBe(18);
    expect(s.code.bytesNeeded).toBe(64);
    expect(s.code.efficiency).toBeCloseTo(64 / 576, 6);
    expect(Math.min(...s.codeLanes.map((x) => x.addr))).toBe(2);
    expect(Math.max(...s.codeLanes.map((x) => x.addr + x.bytes)) - 1).toBe(561);
  });
  it('four lanes per Q4_0 block, 2-byte loads: 8 blocks, 144 bytes, 5 segments for 64 useful bytes', () => {
    const m = q4({ lanesPerBlock: 4 });
    const s = loadStep(m, 0, 0, 32);
    expect(iterationSpan(m, 0)).toEqual([0, 144]);
    expect(s.code.transactions).toBe(5);
    expect(s.code.bytesNeeded).toBe(64);
    expect(loadsPerLane(m)).toBe(2);
  });
  it("lab X2.1's mapping: lane L reads code bytes 4L to 4L + 3, 4 segments, all useful; the scales are one segment", () => {
    const m: Mapping = { layout: 'q8_0-split', lanesPerBlock: 8, loadBytes: 4, warpWidth: 32, inW: 4096 };
    const s = loadStep(m, 0, 0, 32);
    expect(s.codeLanes.map((x) => x.addr)).toEqual(Array.from({ length: 32 }, (_, L) => 4 * L));
    expect(s.code.transactions).toBe(4);
    expect(s.code.efficiency).toBe(1);
    expect(s.scale.transactions).toBe(1);
    expect(s.scale.bytesNeeded).toBe(8);
    expect(scaleSpan(m, 1)).toEqual([8, 16]);
    expect(s.misaligned).toBe(0);
  });
});

describe("lab X2.2's mapping (piece order)", () => {
  it('lane s of 4 reads qs bytes 4s and 4s + 2 of its block in two 2-byte loads; still 5 segments', () => {
    const m = q4({ lanesPerBlock: 4, order: 'piece' });
    const a = loadStep(m, 0, 0, 32);
    const b = loadStep(m, 0, 1, 32);
    expect(a.codeLanes.slice(0, 5).map((x) => x.addr)).toEqual([2, 6, 10, 14, 20]);
    expect(b.codeLanes.slice(0, 4).map((x) => x.addr)).toEqual([4, 8, 12, 16]);
    expect(a.code.transactions).toBe(5);
    expect(a.misaligned + b.misaligned).toBe(0);
  });
  it('with one lane per block the two orders agree', () => {
    for (const k of [0, 3])
      expect(loadStep(q4({ order: 'piece' }), 0, k, 32).codeLanes.map((x) => x.addr)).toEqual(loadStep(q4(), 0, k, 32).codeLanes.map((x) => x.addr));
  });
});

describe('alignment', () => {
  it('4-byte loads of Q4_0 codes: even blocks start 2 mod 4 and are misaligned, odd blocks are aligned', () => {
    const s = loadStep(q4({ loadBytes: 4 }), 0, 0, 32);
    expect(s.codeLanes[0].misaligned).toBe(true);
    expect(s.codeLanes[1].misaligned).toBe(false);
    expect(s.misaligned).toBe(16);
  });
  it('2-byte and 1-byte loads of Q4_0 codes are always aligned; separate arrays allow 4-byte loads', () => {
    expect(loadStep(q4({ loadBytes: 2 }), 0, 0, 32).misaligned).toBe(0);
    expect(loadStep(q4({ loadBytes: 1 }), 0, 3, 32).misaligned).toBe(0);
    expect(loadStep({ layout: 'mlx4-split', lanesPerBlock: 4, loadBytes: 4, warpWidth: 32, inW: 4096 }, 0, 1, 32).misaligned).toBe(0);
  });
});

describe('the tail of a row', () => {
  it('in = 1056 (33 Q4_0 blocks), one lane per block: 31 idle lanes in the last iteration on 32 and on 64 lanes', () => {
    const p32 = iterationPlan(1056, 'q4_0-blocks', 1, 32);
    expect(p32.blocks).toBe(33);
    expect(p32.iterations).toBe(2);
    expect(p32.idleLast).toBe(31);
    const p64 = iterationPlan(1056, 'q4_0-blocks', 1, 64);
    expect(p64.iterations).toBe(1);
    expect(p64.idleLast).toBe(31);
    const s = loadStep(q4({ inW: 1056 }), 1, 0, 32);
    expect(s.idle).toBe(31);
    expect(s.code.bytesNeeded).toBe(2);
  });
  it('short rows leave much of a 64-wide wavefront idle', () => {
    expect(iterationPlan(1024, 'q4_0-blocks', 1, 64).idleFraction).toBe(0.5);
    expect(iterationPlan(1024, 'q4_0-blocks', 1, 32).idleFraction).toBe(0);
    expect(iterationPlan(1024, 'q4_0-blocks', 2, 64).idleFraction).toBe(0);
  });
});

describe('warp width is a parameter', () => {
  it('a 64-wide wavefront with four lanes per block covers 16 blocks per iteration', () => {
    const m = q4({ lanesPerBlock: 4, warpWidth: 64 });
    expect(iterationPlan(4096, 'q4_0-blocks', 4, 64).blocksPerIteration).toBe(16);
    const s = loadStep(m, 0, 0, 64);
    expect(s.codeLanes).toHaveLength(64);
    expect(new Set(s.codeLanes.map((x) => x.block)).size).toBe(16);
    expect(rowSteps(m)).toHaveLength(8 * 2);
  });
  it('the same mapping on 32 lanes covers 8 blocks', () => {
    expect(iterationPlan(4096, 'q4_0-blocks', 4, 32).blocksPerIteration).toBe(8);
  });
});

describe('mappings that cannot run', () => {
  it('explains itself', () => {
    expect(problem(q4({ lanesPerBlock: 8, loadBytes: 4 }))).toMatch(/16 code bytes/);
    expect(problem({ layout: 'mlx4-split', lanesPerBlock: 1, loadBytes: 4, warpWidth: 32, inW: 1056 })).toMatch(/64 weights/);
    expect(problem(q4())).toBeNull();
  });
  it("Apple's segment size is an assumption the reader makes", () => {
    expect(segmentBytesOf('nvidia')).toBe(32);
    expect(segmentBytesOf('amd128')).toBe(128);
    expect(segmentBytesOf('metal', 64)).toBe(64);
  });
});
