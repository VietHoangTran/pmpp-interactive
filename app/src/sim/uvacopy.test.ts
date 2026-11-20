import { describe, expect, it } from 'vitest';
import { REGISTERED, hex4, inferKind, locate, ranges } from './uvacopy';

describe('one shared address space (CUDA 4 and later, p. 625)', () => {
  it('ranges are disjoint, so every value lies in at most one', () => {
    const rs = ranges('uva');
    for (let a = 0; a < 0x10000; a += 0x80) expect(rs.filter((r) => a >= r.lo && a < r.hi).length).toBeLessThanOrEqual(1);
  });
  it('q-direction: dst on GPU 1, src in pinned host memory: host to device, into GPU 1', () => {
    const r = inferKind(0xc400, 0x3000, 'uva');
    expect(r.kind).toBe('cudaMemcpyHostToDevice');
    expect(r.dst[0].space).toBe('gpu1');
    expect(r.detail).toBe('From host memory to GPU 1, read off the two values.');
  });
  it('all four directions are inferred', () => {
    expect(inferKind(0x8100, 0x1100, 'uva').kind).toBe('cudaMemcpyHostToDevice');
    expect(inferKind(0x1100, 0x8100, 'uva').kind).toBe('cudaMemcpyDeviceToHost');
    expect(inferKind(0xc100, 0x8100, 'uva').kind).toBe('cudaMemcpyDeviceToDevice');
    expect(inferKind(0x1100, 0x2100, 'uva').kind).toBe('cudaMemcpyHostToHost');
  });
  it('a value in a gap is no allocation', () => {
    expect(inferKind(0x7800, 0x1100, 'uva').kind).toBe('unmapped');
    expect(inferKind(0x7800, 0x1100, 'uva').detail).toMatch(/^The destination value/);
  });
});

describe('separate address spaces (before CUDA 4)', () => {
  it('the same value names a location in three spaces, so the direction cannot be inferred', () => {
    expect(locate(0x3000, 'separate')).toHaveLength(3);
    const r = inferKind(0x3000, 0x2000, 'separate');
    expect(r.kind).toBe('ambiguous');
    expect(r.detail).toContain('3 separate address spaces');
  });
  it('even values that look different are ambiguous: the value never says which space', () => {
    expect(inferKind(0x6f00, 0x1000, 'separate').kind).toBe('ambiguous');
  });
});

describe("a registered buffer: one buffer, two addresses (A3, A5)", () => {
  it('appears at a host address and at a distinct device address', () => {
    const rs = ranges('uva', true).filter((r) => r.registered);
    expect(rs.map((r) => r.registered)).toEqual(['host-side', 'device-side']);
    expect(rs[0].hi - rs[0].lo).toBe(rs[1].hi - rs[1].lo);
    expect(rs[1].lo).toBe(REGISTERED.device[0]);
  });
  it('the host-side address behaves as host memory; the device-side address is reported, not guessed', () => {
    expect(inferKind(0x8100, REGISTERED.host[0] + 0x10, 'uva', true).kind).toBe('cudaMemcpyHostToDevice');
    const r = inferKind(0x8100, REGISTERED.device[0] + 0x10, 'uva', true);
    expect(r.kind).toBe('alias');
    expect(r.detail).toContain('one buffer, two addresses');
  });
  it('without the option the device-side range is a gap', () => {
    expect(inferKind(0x8100, REGISTERED.device[0] + 0x10, 'uva', false).kind).toBe('unmapped');
  });
  it('hex formatting', () => {
    expect(hex4(0x2c00)).toBe('0x2c00');
    expect(hex4(0x10)).toBe('0x0010');
  });
});
