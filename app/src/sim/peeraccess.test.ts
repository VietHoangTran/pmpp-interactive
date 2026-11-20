import { describe, expect, it } from 'vitest';
import {
  MAX_PEERS_WITHOUT_NVSWITCH,
  allToAll,
  allToAllPossible,
  canEnable,
  deref,
  emptyGrants,
  enable,
  enableAll,
  haloExchange,
  outgoing,
  revoke,
  type PeerSystem,
} from './peeraccess';

const sys = (n: number, nvswitch = false, iommu = false): PeerSystem => ({ n, nvswitch, iommu });

describe('enabling peer access (appC-facts A8)', () => {
  it('is one-directional: GPU 0 → 1 lets GPU 0 load from GPU 1, not the reverse', () => {
    const { grants, result } = enable(sys(2), emptyGrants(2), 0, 1);
    expect(result.ok).toBe(true);
    expect(deref(grants, 0, 1)).toMatchObject({ ok: true, how: 'peer' });
    expect(deref(grants, 1, 0)).toMatchObject({ ok: false, how: 'refused' });
    expect(deref(grants, 1, 0).text).toContain('cudaMemcpyPeer');
  });
  it('the call is made on the accessing device', () => {
    expect(canEnable(sys(2), emptyGrants(2), 0, 1).call).toBe('on GPU 0: cudaDeviceEnablePeerAccess(1, 0)');
  });
  it('own memory needs no grant and always loads', () => {
    expect(canEnable(sys(4), emptyGrants(4), 2, 2)).toMatchObject({ ok: false, reason: 'self' });
    expect(deref(emptyGrants(4), 2, 2)).toMatchObject({ ok: true, how: 'local' });
  });
  it('bare-metal Linux with the IOMMU on refuses every PCIe grant (A8)', () => {
    expect(canEnable(sys(2, false, true), emptyGrants(2), 0, 1)).toMatchObject({ ok: false, reason: 'iommu' });
    const r = canEnable(sys(2, false, true), emptyGrants(2), 0, 1);
    expect(r.ok ? '' : r.text).toContain('PCIe peer-to-peer');
    expect(allToAllPossible(sys(2, false, true))).toBe(false);
    expect(allToAll(sys(2, false, true))).toBe('impossible');
    expect(enableAll(sys(4, false, true))).toMatchObject({ granted: 0, refused: 12, unknown: 0 });
  });
  it('with NVSwitch and the IOMMU on, A8 says nothing: not documented, not refused, nothing granted', () => {
    const r = canEnable(sys(2, true, true), emptyGrants(2), 0, 1);
    expect(r).toMatchObject({ ok: false, reason: 'undocumented' });
    expect(r.ok ? '' : r.text).toContain('Not documented');
    expect(allToAll(sys(16, true, true))).toBe('unknown');
    expect(allToAllPossible(sys(16, true, true))).toBe(false);
    expect(enableAll(sys(4, true, true))).toMatchObject({ granted: 0, refused: 0, unknown: 12 });
  });
  it('without NVSwitch a GPU gets at most eight peers; the ninth is refused', () => {
    let g = emptyGrants(10);
    for (let j = 1; j <= 8; j++) g = enable(sys(10), g, 0, j).grants;
    expect(outgoing(g, 0)).toBe(MAX_PEERS_WITHOUT_NVSWITCH);
    expect(canEnable(sys(10), g, 0, 9)).toMatchObject({ ok: false, reason: 'limit' });
    // other GPUs are unaffected
    expect(canEnable(sys(10), g, 9, 0).ok).toBe(true);
    // re-enabling an existing grant is not a ninth peer
    expect(canEnable(sys(10), g, 0, 3).ok).toBe(true);
    // taking one back frees a slot
    expect(canEnable(sys(10), revoke(g, 0, 3), 0, 9).ok).toBe(true);
  });
  it('with NVSwitch the limit does not apply', () => {
    let g = emptyGrants(16);
    for (let j = 1; j < 16; j++) g = enable(sys(16, true), g, 0, j).grants;
    expect(outgoing(g, 0)).toBe(15);
  });
});

describe('all to all (the C.5 question on sixteen GPUs)', () => {
  it('nine GPUs without NVSwitch: every GPU needs eight peers, which just fits', () => {
    expect(allToAllPossible(sys(9))).toBe(true);
    const r = enableAll(sys(9));
    expect(r.granted).toBe(72);
    expect(r.refused).toBe(0);
  });
  it('sixteen GPUs without NVSwitch: each needs 15 peers, only 8 are allowed', () => {
    expect(allToAllPossible(sys(16))).toBe(false);
    const r = enableAll(sys(16));
    expect(r.granted).toBe(16 * 8);
    expect(r.refused).toBe(16 * 7);
    for (let i = 0; i < 16; i++) expect(outgoing(r.grants, i)).toBe(8);
  });
  it('sixteen GPUs with NVSwitch: all 240 grants succeed', () => {
    expect(allToAllPossible(sys(16, true))).toBe(true);
    expect(enableAll(sys(16, true)).granted).toBe(240);
  });
});

describe('one halo exchange between two GPUs (book p. 626)', () => {
  it('host-driven copies: two staging buffers, two host waits, two copies set up by the host', () => {
    const h = haloExchange('copy');
    expect([h.staging, h.hostWaits, h.hostCopies, h.ordering]).toEqual([2, 2, 2, 1]);
    expect(h.steps.filter((s) => s.who === 'host')).toHaveLength(4);
  });
  it('peer loads: no staging buffer and no host copy, but ordering between the GPUs remains', () => {
    const h = haloExchange('peer');
    expect([h.staging, h.hostWaits, h.hostCopies, h.ordering]).toEqual([0, 0, 0, 1]);
    expect(h.steps.some((s) => s.kind === 'order')).toBe(true);
    expect(h.steps.filter((s) => s.kind === 'load')).toHaveLength(2);
  });
  it('peer loads need a grant in each direction', () => {
    const one = enable(sys(2), emptyGrants(2), 0, 1).grants;
    expect(haloExchange('peer', one)).toMatchObject({ ready: false, missing: [[1, 0]] });
    const both = enable(sys(2), one, 1, 0).grants;
    expect(haloExchange('peer', both)).toMatchObject({ ready: true, missing: [] });
    expect(haloExchange('peer', emptyGrants(2)).missing).toEqual([
      [0, 1],
      [1, 0],
    ]);
  });
});
