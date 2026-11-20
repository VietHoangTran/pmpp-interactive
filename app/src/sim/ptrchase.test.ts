import { describe, expect, it } from 'vitest';
import { cell } from './derefmap';
import { ALLOCS, SETTINGS, addresses, chase, firstFailure, goodBeforeFailure, hex, type Alloc } from './ptrchase';

describe('the question appC/C.3/q-chase (book p. 625)', () => {
  const list: Alloc[] = ['cudaHostAlloc', 'cudaHostAlloc', 'malloc', 'cudaHostAlloc'];
  it('CUDA 4: two values read correctly, then the hop into malloc memory is invalid', () => {
    expect(goodBeforeFailure(list, 'uva')).toBe(2);
    const hops = chase(list, 'uva');
    expect(hops.map((h) => h.verdict)).toEqual(['remote', 'remote', 'invalid']);
    expect(hops[2].value).toBeNull();
    expect(hops[2].sum).toBe(3 + 1);
  });
  it('the same list works with HMM or ATS, where malloc memory faults in (A12)', () => {
    expect(firstFailure(list, 'hmm')).toBeNull();
    expect(chase(list, 'hmm').map((h) => h.verdict)).toEqual(['remote', 'remote', 'fault', 'remote']);
  });
  it('Pascal without HMM does not rescue the malloc node (A12)', () => {
    expect(goodBeforeFailure(list, 'pascal')).toBe(2);
  });
});

describe('the stepper\'s rules are the map\'s rules', () => {
  it('the head uses the "argument" cell, every later node the "read from memory" cell', () => {
    for (const s of SETTINGS)
      for (const a of ALLOCS) {
        const row = { malloc: 'malloc', cudaHostAlloc: 'pinned', cudaMalloc: 'device', cudaMallocManaged: 'managed' }[a] as
          | 'malloc' | 'pinned' | 'device' | 'managed';
        const hops = chase([a, a], s.id);
        expect(hops[0].rule).toEqual(cell('cuda', s.era, row, 'arg'));
        if (hops.length > 1) expect(hops[1].rule).toEqual(cell('cuda', s.era, row, 'chased'));
      }
  });
  it('managed nodes: invalid nowhere, read locally in the limited mode (moved at launch), faulting from Pascal on', () => {
    const m: Alloc[] = ['cudaMallocManaged', 'cudaMallocManaged', 'cudaMallocManaged'];
    expect(chase(m, 'um-limited').map((h) => h.verdict)).toEqual(['local', 'local', 'local']);
    expect(chase(m, 'pascal').map((h) => h.verdict)).toEqual(['fault', 'fault', 'fault']);
  });
  it('managed memory does not exist under CUDA 4, so a managed node is invalid there', () => {
    expect(chase(['cudaMallocManaged'], 'uva')[0].verdict).toBe('invalid');
  });
  it('cudaMalloc nodes, filled by the host with cudaMemcpy, are read locally in every setting', () => {
    for (const s of SETTINGS) expect(chase(['cudaMalloc', 'cudaMalloc'], s.id).every((h) => h.verdict === 'local')).toBe(true);
  });
});

describe('a list can be fine for two hops and fail on the third', () => {
  it('five nodes, the third in malloc memory: two good reads, sum 4, then the failure', () => {
    const list: Alloc[] = ['cudaMalloc', 'cudaHostAlloc', 'malloc', 'cudaMalloc', 'cudaMalloc'];
    const hops = chase(list, 'uva');
    expect(hops).toHaveLength(3);
    expect(hops.map((h) => h.sum)).toEqual([3, 4, 4]);
    expect(firstFailure(list, 'uva')).toBe(2);
  });
  it('a list of all-malloc nodes fails at the head everywhere but HMM or ATS', () => {
    const list: Alloc[] = ['malloc', 'malloc'];
    for (const s of ['uva', 'pascal', 'um-limited'] as const) expect(firstFailure(list, s)).toBe(0);
    expect(firstFailure(list, 'hmm')).toBeNull();
  });
});

describe('addresses', () => {
  it('one range per allocator, nodes 64 bytes apart inside it', () => {
    const a = addresses(['cudaHostAlloc', 'malloc', 'cudaHostAlloc']);
    expect(a[2] - a[0]).toBe(64);
    expect(hex(a[0])).toBe('0x7f30_0000_0000');
    expect(hex(a[2])).toBe('0x7f30_0000_0040');
  });
});
