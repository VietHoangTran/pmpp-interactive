import { describe, expect, it } from 'vitest';
import {
  SECTORS_PER_REQUEST, aligned, bytesPerWarpLoad, instructions, misalignedChunks, plan, weightsPerLoad, widestVector,
} from './vectorize';

// Every number section 6.3 states is pinned here.
describe('the chunk plan (p. 138, Exercise 4)', () => {
  it('n = 1003, 256 threads per block, float4: 251 chunks in one block; thread 250 has a tail of 3; 5 threads idle', () => {
    const p = plan(1003, 256, 4);
    expect(p.chunks).toBe(251);
    expect(p.blocks).toBe(1);
    expect(p.threads).toBe(256);
    expect(p.tailThread).toBe(250);
    expect(p.tailCount).toBe(3);
    expect(p.idle).toBe(5);
    expect(p.roles[250]).toEqual({ thread: 250, block: 0, role: 'tail', first: 1000, count: 3 });
    expect(p.roles[249]).toEqual({ thread: 249, block: 0, role: 'vector', first: 996, count: 4 });
    expect(p.roles.slice(251).every((r) => r.role === 'idle')).toBe(true);
  });
  it('a quarter of the threads: 1024 floats in float4 chunks need 256 threads, with no tail and none idle', () => {
    const p = plan(1024, 256, 4);
    expect([p.chunks, p.threads, p.tailThread, p.idle]).toEqual([256, 256, null, 0]);
    expect(plan(1024, 256, 1).threads).toBe(1024);
  });
  it('one element past a whole number of blocks costs a whole extra block', () => {
    const p = plan(1025, 256, 4);
    expect([p.chunks, p.blocks, p.threads, p.tailThread, p.tailCount, p.idle]).toEqual([257, 2, 512, 256, 1, 255]);
  });
  it('scalar code has idle threads but never a tail', () => {
    const p = plan(1003, 256, 1);
    expect([p.threads, p.tailThread, p.idle]).toEqual([1024, null, 21]);
  });
  it('every element is owned by exactly one thread', () => {
    for (const [n, b, w] of [[1003, 256, 4], [77, 8, 2], [5, 32, 4], [1, 1, 4]] as const) {
      const p = plan(n, b, w);
      const owned = p.roles.flatMap((r) => Array.from({ length: r.count }, (_, j) => r.first + j));
      expect(owned).toEqual(Array.from({ length: n }, (_, i) => i));
    }
  });
});

describe('instruction counts (p. 139)', () => {
  it('two float4 loads replace eight scalar loads: 75 % fewer', () => {
    const one = instructions(4, 4);
    expect([one.loads, one.scalarLoads]).toEqual([2, 8]);
    expect(one.fewerLoadsPct).toBe(75);
    expect(instructions(4096, 4).fewerLoadsPct).toBe(75);
  });
  it('stores fall by the same factor', () => {
    const c = instructions(4096, 4);
    expect([c.stores, c.scalarStores]).toEqual([1024, 4096]);
  });
  it('a scalar tail eats into the saving: n = 1003 saves a little less than 75 %', () => {
    const c = instructions(1003, 4);
    expect([c.loads, c.stores, c.scalarLoads]).toEqual([506, 253, 2006]);
    expect(c.fewerLoadsPct).toBeCloseTo(74.78, 2);
  });
  it('float2 halves them', () => {
    expect(instructions(4096, 2).fewerLoadsPct).toBe(50);
  });
});

describe('alignment (facts A11, D4)', () => {
  it("PyTorch's rule: offset 0 allows 8 floats, 16 bytes allows 4, 8 bytes allows 2, 4 bytes only 1", () => {
    expect(widestVector(0, 4)).toBe(8);
    expect(widestVector(16, 4)).toBe(4);
    expect(widestVector(8, 4)).toBe(2);
    expect(widestVector(4, 4)).toBe(1);
    expect(widestVector(48, 4)).toBe(4);
    expect(widestVector(64, 4)).toBe(8);
  });
  it('q-align: x + 1 is 4 bytes past a 16-byte boundary, so no float4 access there is aligned', () => {
    expect(aligned(4, 4, 4)).toBe(false);
    expect(aligned(16, 4, 4)).toBe(true);
    const p = plan(64, 32, 4);
    expect(misalignedChunks(p, 4)).toHaveLength(16);
    expect(misalignedChunks(p, 0)).toHaveLength(0);
    expect(misalignedChunks(p, 32)).toHaveLength(0);
    expect(misalignedChunks(plan(64, 32, 2), 8)).toHaveLength(0);
    expect(misalignedChunks(plan(64, 32, 1), 4)).toHaveLength(0);
  });
});

describe('per warp (facts A11)', () => {
  it("a warp of 32 moving float4 moves 512 bytes, 16 sectors, in one instruction (Nsight's optimum)", () => {
    expect(bytesPerWarpLoad(32, 4)).toBe(512);
    expect(bytesPerWarpLoad(32, 1)).toBe(128);
    expect(SECTORS_PER_REQUEST[16]).toBe(16);
    expect(SECTORS_PER_REQUEST[4]).toBe(4);
    expect(bytesPerWarpLoad(32, 4) / 32).toBe(SECTORS_PER_REQUEST[16]);
  });
});

describe('the quantised-weights bridge (ours)', () => {
  it('one 16-byte load carries 32 four-bit weights, 16 int8 weights', () => {
    expect(weightsPerLoad(16, 4)).toBe(32);
    expect(weightsPerLoad(16, 8)).toBe(16);
  });
});
