import { describe, expect, it } from 'vitest';
import {
  LLAMA2_7B_BYTES_PER_TOKEN,
  MiB,
  apply,
  bookExample,
  bookThreeCalls,
  clean,
  grow,
  initial,
  kvBytesPerToken,
  run,
  slots,
  uncovered,
  type Op,
} from './vmmgrow';

const G = 2 * MiB;

describe('the stepper: reserve, create, map, grant (book p. 629; appC-facts A15)', () => {
  it("the book's three calls alone: the kernel's read crashes, because nothing granted access", () => {
    const { log } = run(G, bookThreeCalls(G));
    expect(log.map((o) => o.kind)).toEqual(['ok', 'ok', 'ok', 'crash']);
    expect(log[3].code).toBe('no-access');
    expect(log[3].text).toContain('cuMemSetAccess');
    expect(log.map((o) => o.call)).toEqual(['cuMemAddressReserve', 'cuMemCreate', 'cuMemMap', 'kernel']);
  });
  it('granting access for the reading device fixes it', () => {
    const ops: Op[] = [...bookThreeCalls(G).slice(0, 3), { op: 'setAccess', offset: 0, size: 4 * G, device: 0 }, { op: 'read', device: 0, offset: 0, size: 4 * G }];
    const { log } = run(G, ops);
    expect(log.map((o) => o.kind)).toEqual(['ok', 'ok', 'ok', 'ok', 'ok']);
    expect(log[4].pieces?.every((p) => p.status === 'local')).toBe(true);
  });
  it("the book's example: one pointer over memory on two GPUs, half local and half remote for GPU 0", () => {
    const { state, log } = run(G, bookExample(G));
    expect(log.every((o) => o.kind === 'ok')).toBe(true);
    const read = log[log.length - 1];
    expect(read.pieces?.map((p) => [p.offset / G, p.size / G, p.status])).toEqual([
      [0, 4, 'local'],
      [4, 4, 'remote'],
    ]);
    const s = slots(state);
    expect(s.map((x) => x.device)).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
    expect(s.every((x) => x.access[0] && !x.access[1])).toBe(true);
  });
  it('access is per device: GPU 1 reading the same range crashes until it gets its own grant', () => {
    let { state } = run(G, bookExample(G));
    let r = apply(state, { op: 'read', device: 1, offset: 0, size: 8 * G });
    expect(r.outcome).toMatchObject({ kind: 'crash', code: 'no-access' });
    state = apply(state, { op: 'setAccess', offset: 0, size: 8 * G, device: 1 }).state;
    r = apply(state, { op: 'read', device: 1, offset: 0, size: 8 * G });
    expect(r.outcome.kind).toBe('ok');
    expect(r.outcome.pieces?.map((p) => p.status)).toEqual(['remote', 'local']);
  });
  it('reading an address with nothing mapped crashes', () => {
    const { state } = run(G, bookExample(G).slice(0, 4)); // GPU 0's half only is mapped
    const r = apply(apply(state, { op: 'setAccess', offset: 0, size: 4 * G, device: 0 }).state, { op: 'read', device: 0, offset: 0, size: 8 * G });
    expect(r.outcome).toMatchObject({ kind: 'crash', code: 'unmapped' });
  });
  it('a chunk size that is not a whole number of granules is refused, citing HIP\'s documented error', () => {
    const r = apply(initial(G), { op: 'create', device: 0, size: 1.5 * G });
    expect(r.outcome).toMatchObject({ kind: 'error', code: 'misaligned' });
    expect(r.outcome.text).toContain('out-of-memory');
    expect(r.state.chunks).toHaveLength(0);
  });
  it('mapping past the end of the reservation or over a mapped chunk is refused', () => {
    let { state } = run(G, [
      { op: 'reserve', size: 4 * G },
      { op: 'create', device: 0, size: 2 * G },
      { op: 'create', device: 1, size: 2 * G },
      { op: 'map', chunk: 1, offset: 0 },
    ]);
    expect(apply(state, { op: 'map', chunk: 2, offset: 3 * G }).outcome.code).toBe('outside');
    expect(apply(state, { op: 'map', chunk: 2, offset: G }).outcome.code).toBe('overlap');
    expect(apply(state, { op: 'map', chunk: 1, offset: 2 * G }).outcome.code).toBe('mapped');
    state = apply(state, { op: 'map', chunk: 2, offset: 2 * G }).state;
    expect(state.maps).toHaveLength(2);
  });
  it('mapping before reserving, and granting over unmapped addresses, are refused', () => {
    const s = apply(initial(G), { op: 'create', device: 0, size: G }).state;
    expect(apply(s, { op: 'map', chunk: 1, offset: 0 }).outcome.code).toBe('no-range');
    const r = apply(s, { op: 'reserve', size: 2 * G }).state;
    expect(apply(r, { op: 'setAccess', offset: 0, size: G, device: 0 }).outcome.code).toBe('unmapped');
  });
  it('teardown in the documented order (unmap, release, free) leaves nothing behind', () => {
    const { state } = run(G, [
      ...bookExample(G),
      { op: 'unmap', chunk: 1 },
      { op: 'unmap', chunk: 2 },
      { op: 'release', chunk: 1 },
      { op: 'release', chunk: 2 },
      { op: 'free' },
    ]);
    expect(clean(state)).toBe(true);
  });
  it('release before unmap, and free with chunks still mapped, are flagged and not applied', () => {
    const { state } = run(G, bookExample(G));
    const r1 = apply(state, { op: 'release', chunk: 1 });
    expect(r1.outcome).toMatchObject({ kind: 'order', code: 'order' });
    expect(r1.state).toBe(state);
    const r2 = apply(state, { op: 'free' });
    expect(r2.outcome.kind).toBe('order');
    expect(r2.state.reserved).toBe(8 * G);
  });
  it('unmapping takes the grant over that range with it', () => {
    let { state } = run(G, bookExample(G));
    state = apply(state, { op: 'unmap', chunk: 2 }).state;
    expect(state.grants).toEqual([{ offset: 0, size: 4 * G, device: 0 }]);
  });
  it('uncovered() finds the gaps in a set of intervals', () => {
    expect(uncovered([{ offset: 2, size: 2 }], 0, 8)).toEqual([
      { offset: 0, size: 2 },
      { offset: 4, size: 4 },
    ]);
    expect(uncovered([{ offset: 0, size: 8 }], 2, 3)).toEqual([]);
  });
});

describe('growth: a buffer that grows one chunk at a time (the C.9 question)', () => {
  const opts = { tokens: 16, bytesPerToken: 1, chunk: 1 };
  it('doubling from 1 chunk to 16 copies 1 + 2 + 4 + 8 = 15 chunks and moves the pointer 4 times', () => {
    const r = grow('double', opts);
    expect(r.copiedTotal).toBe(15);
    expect(r.pointerChanges).toBe(4);
    expect(r.steps.filter((s) => s.moved).map((s) => s.copied)).toEqual([1, 2, 4, 8]);
    expect(r.finalAllocated).toBe(16);
    expect(r.unused).toBe(0);
  });
  it('doubling peaks at old plus new, three times the data being copied', () => {
    const r = grow('double', opts);
    expect(r.peak).toBe(8 + 16);
    expect(r.peakRatio).toBe(3);
  });
  it('reserve-and-map copies nothing, keeps the pointer, and maps exactly what is needed', () => {
    const r = grow('vmm', opts);
    expect(r.copiedTotal).toBe(0);
    expect(r.pointerChanges).toBe(0);
    expect(r.steps.every((s) => s.base === 0)).toBe(true);
    expect(r.peak).toBe(16);
    expect(r.unused).toBe(0);
  });
  it('exact fit copies 1 + 2 + ... + 15 = 120 chunks: quadratic', () => {
    const r = grow('exact', opts);
    expect(r.copiedTotal).toBe(120);
    expect(r.pointerChanges).toBe(15);
    expect(r.peak).toBe(15 + 16);
  });
});

describe("growth of the book's Llama 2 7B KV cache (ch. 20 §20.6, Eq. 20.7, pp. 505–506)", () => {
  it('512 KiB per token at 16 bits; 4096 tokens make 2 GiB, the book\'s figure', () => {
    expect(LLAMA2_7B_BYTES_PER_TOKEN).toBe(512 * 1024);
    expect(kvBytesPerToken(32, 4096, 2) * 4096).toBe(2 * 1024 * MiB);
  });
  const o = { tokens: 4096, bytesPerToken: LLAMA2_7B_BYTES_PER_TOKEN, chunk: 2 * MiB };
  it('doubling copies about as much as the final size, and peaks at 1.5 times it', () => {
    const r = grow('double', o);
    expect(r.copiedTotal).toBe(2048 * MiB - 2 * MiB);
    expect(r.copyRatio).toBeCloseTo(1, 2);
    expect(r.peak).toBe(3 * 1024 * MiB);
    expect(r.peakRatio).toBe(3);
  });
  it('exact fit copies about 4 TiB', () => {
    const r = grow('exact', o);
    expect(r.copiedTotal).toBe(LLAMA2_7B_BYTES_PER_TOKEN * ((4096 * 4095) / 2));
    expect(r.copiedTotal / 2 ** 40).toBeCloseTo(4, 2);
  });
  it('reserve-and-map wastes less than one chunk at any length', () => {
    for (const tokens of [1, 3, 5, 1000, 4095]) {
      const r = grow('vmm', { ...o, tokens });
      expect(r.unused).toBeLessThan(o.chunk);
      expect(r.unused).toBeGreaterThanOrEqual(0);
    }
  });
  it('a reservation smaller than the data overflows', () => {
    const r = grow('vmm', { ...o, tokens: 100, reserve: 8 * MiB });
    expect(r.overflow).toBe(true);
    expect(r.finalAllocated).toBe(8 * MiB);
  });
});

describe('the scene\'s sequence: 32 tokens, 4 per chunk', () => {
  it('doubling copies 4 + 8 + 16 = 28 token slots (7 chunks); mapping copies none', () => {
    const d = grow('double', { tokens: 32, bytesPerToken: 1, chunk: 4 });
    expect(d.copiedTotal).toBe(28);
    expect(d.steps.filter((s) => s.moved).map((s) => s.token)).toEqual([5, 9, 17]);
    const v = grow('vmm', { tokens: 32, bytesPerToken: 1, chunk: 4 });
    expect(v.copiedTotal).toBe(0);
    expect(v.steps.filter((s) => s.newChunks).map((s) => s.token)).toEqual([1, 5, 9, 13, 17, 21, 25, 29]);
  });
});
