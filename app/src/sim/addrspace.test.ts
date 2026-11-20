import { describe, expect, it } from 'vitest';
import { CARDS, STRIP_TOP, VARIABLES, WINDOWS, classify, definitionsNeeded, judge } from './addrspace';

describe('windows in the generic address space (p. 624; A2, B1)', () => {
  it('an address outside every window is global', () => {
    expect(classify(0x0000).window).toBe('global');
    expect(classify(STRIP_TOP - 1).window).toBe('global');
    expect(classify(0x1200).load).toBe('ld.global');
  });
  it('inside a window, the matching state space and load', () => {
    const [local, shared] = WINDOWS;
    // Local memory takes the global colour, as in chapter 5: it lives in global memory.
    expect(classify(local.lo)).toEqual({ window: 'local', state: '.local', load: 'ld.local', tier: 'global' });
    expect(classify(shared.lo + 4)).toEqual({ window: 'shared', state: '.shared', load: 'ld.shared', tier: 'shared' });
    expect(classify(shared.hi).window).toBe('global');
  });
  it('each preset variable lands in its own window, so one helper serves all three', () => {
    for (const v of VARIABLES) expect(classify(v.addr).window).toBe(v.id);
  });
});

describe('the eight cards, under the MSL specification (C1)', () => {
  const bins = Object.fromEntries(CARDS.map((c) => [c.id, judge(c).bin]));
  it('CUDA and HIP: one generic helper, the argument decides the memory', () => {
    expect(bins['cuda-shared']).toBe('shared');
    expect(bins['cuda-global']).toBe('global');
    expect(bins['hip-shared']).toBe('shared');
    expect(judge(CARDS.find((c) => c.id === 'hip-shared')!).source).toContain('B1');
  });
  it('MSL: no address space is an error; a device-only helper does not take a threadgroup tile', () => {
    expect(judge(CARDS.find((c) => c.id === 'msl-none')!)).toMatchObject({ bin: 'error', error: 'missing-space' });
    expect(judge(CARDS.find((c) => c.id === 'msl-device-only')!)).toMatchObject({ bin: 'error', error: 'no-match' });
  });
  it('MSL: the overload pair sends each call to its own space; a threadgroup helper called from a kernel compiles', () => {
    expect(bins['msl-pair-tile']).toBe('shared');
    expect(bins['msl-pair-row']).toBe('global');
    expect(bins['msl-threadgroup']).toBe('shared');
  });
  it('every bin is used and every outcome names its source', () => {
    expect(new Set(Object.values(bins))).toEqual(new Set(['global', 'shared', 'error']));
    for (const c of CARDS) expect(judge(c).source.length).toBeGreaterThan(5);
  });
  it('q-overloads: two definitions under the specification, one in CUDA and HIP', () => {
    expect(definitionsNeeded('msl')).toBe(2);
    expect(definitionsNeeded('cuda')).toBe(1);
    expect(definitionsNeeded('hip')).toBe(1);
  });
});

describe('the compiler on the build machine at its default, MSL 4.1 (observed, not documented)', () => {
  it('accepts the unqualified pointer as generic; everything else is unchanged', () => {
    for (const c of CARDS) {
      const spec = judge(c, 'spec');
      const seen = judge(c, 'observed-4.1');
      if (c.id === 'msl-none') {
        expect(seen.bin).toBe('shared');
        expect(seen.source).toContain('observed');
      } else expect(seen).toEqual(spec);
    }
    expect(definitionsNeeded('msl', 'observed-4.1')).toBe(1);
  });
});
