import { describe, expect, it } from 'vitest';
import {
  AMD_CDNA_DOC,
  BOOK_DEVICE,
  BOOK_DEVICE_TODAY,
  FIG42,
  FIG45,
  TAIL_EXAMPLE,
  capacity,
  completionOrder,
  coresPerGroup,
  deviceFromReport,
  equalDurations,
  evenSplit,
  isGpuBackend,
  levels,
  noReport,
  profile,
  residentBlocks,
  residentThreads,
  schedule,
  shuffledOrder,
  stateAt,
  tail,
  threadsPerCore,
  totalCores,
  unevenDurations,
  usage,
  waitingAtLaunch,
  waveSizes,
  waves,
} from './blocksched';

describe("the book's example device (pp. 68, 76, 85)", () => {
  it('132 SMs of 128 streaming processors make 16,896', () => {
    expect(totalCores(BOOK_DEVICE)).toBe(16896);
  });
  it('an SM holds 2048 threads, 16 for each streaming processor', () => {
    expect(BOOK_DEVICE.threadsPerUnit).toBe(2048);
    expect(threadsPerCore(BOOK_DEVICE)).toBe(16);
    expect(residentThreads(BOOK_DEVICE)).toBe(270336);
  });
  it('four processing blocks of 32', () => {
    expect(coresPerGroup(BOOK_DEVICE)).toBe(32);
  });
  it('at most 32 blocks per SM, 4224 on the device', () => {
    expect(residentBlocks(BOOK_DEVICE)).toBe(4224);
  });
  it('132 SMs do not divide evenly into 8 GPCs (notes, "Unclear in the PDF" 7)', () => {
    expect(evenSplit(132, 8)).toBeNull();
    expect(evenSplit(BOOK_DEVICE.units!, BOOK_DEVICE.clusters!)).toBeNull();
  });
  it('today: 66 TPCs of 2 SMs make the 132, and 66 TPCs do not divide into 8 GPCs either (facts A2)', () => {
    expect(BOOK_DEVICE_TODAY.tpcs * BOOK_DEVICE_TODAY.unitsPerTpc).toBe(BOOK_DEVICE.units);
    expect(evenSplit(BOOK_DEVICE_TODAY.tpcs, BOOK_DEVICE_TODAY.clusters)).toBeNull();
    expect(BOOK_DEVICE_TODAY.unitsOtherVariant).toBe(114);
  });
});

describe('devices from reports', () => {
  it('a CUDA report gives the SM count and the per-SM limits, and no core count', () => {
    const d = deviceFromReport('cuda', {
      name: 'Some GPU', multiProcessorCount: '80', maxThreadsPerMultiProcessor: '1536', maxBlocksPerMultiProcessor: '16', warpSize: '32',
    });
    expect(d).toMatchObject({ name: 'Some GPU', vocabulary: 'cuda', units: 80, threadsPerUnit: 1536, blocksPerUnit: 16, groupWidth: 32 });
    expect(d.coresPerUnit).toBeNull();
    expect(d.clusters).toBeNull();
    expect(totalCores(d)).toBeNull();
  });
  it('a field of 0 or a missing field is "not reported" (facts B9)', () => {
    const d = deviceFromReport('hip', { name: 'AMD', multiProcessorCount: '120', maxThreadsPerMultiProcessor: '0', warpSize: 64 });
    expect(d.vocabulary).toBe('hip');
    expect(d.units).toBe(120);
    expect(d.threadsPerUnit).toBeNull();
    expect(d.blocksPerUnit).toBeNull();
    expect(d.groupWidth).toBe(64);
  });
  it('the Metal report of this repository has none of the fields (facts C1)', () => {
    const d = deviceFromReport('metal', {
      name: 'Apple GPU', hasUnifiedMemory: '1', 'maxThreadsPerThreadgroup.width': '1024', maxThreadgroupMemoryLength: '32768',
    });
    expect(d.vocabulary).toBe('metal');
    expect([d.units, d.clusters, d.coresPerUnit, d.threadsPerUnit, d.blocksPerUnit, d.groupWidth].every((v) => v === null)).toBe(true);
  });
});

describe('which reports are devices', () => {
  it('the CPU emulators are not', () => {
    expect(['emu', 'race', 'metal', 'cuda', 'hip'].filter(isGpuBackend)).toEqual(['metal', 'cuda', 'hip']);
  });
});

describe('levels in each vendor\'s words', () => {
  const row = (d: Parameters<typeof levels>[0], id: string) => levels(d).find((l) => l.id === id)!;
  it("the book's example: every count is the book's, and only the SM count can be asked for", () => {
    const ls = levels(BOOK_DEVICE);
    expect(ls.map((l) => l.id)).toEqual(['device', 'cluster', 'unit', 'group', 'core', 'onchip', 'global']);
    expect(row(BOOK_DEVICE, 'unit')).toMatchObject({ count: 132, provenance: 'book', ask: 'multiProcessorCount' });
    expect(row(BOOK_DEVICE, 'cluster')).toMatchObject({ count: 8, provenance: 'book', ask: null });
    expect(row(BOOK_DEVICE, 'group')).toMatchObject({ count: 4, ask: null });
    expect(row(BOOK_DEVICE, 'core')).toMatchObject({ count: 128, per: 'in each SM', ask: null });
    expect(row(BOOK_DEVICE, 'unit').holds.map((h) => h.where)).toEqual(['4.2', '4.6', '4.7', 'chapter 5']);
  });
  it('a CUDA report: SM count reported, the levels inside an SM not reported', () => {
    const d = deviceFromReport('cuda', { name: 'X', multiProcessorCount: 80 });
    expect(row(d, 'unit')).toMatchObject({ count: 80, provenance: 'reported' });
    expect(row(d, 'cluster')).toMatchObject({ count: null, provenance: 'none' });
    expect(row(d, 'group')).toMatchObject({ count: null, provenance: 'none' });
    expect(row(d, 'core')).toMatchObject({ count: null, provenance: 'none' });
  });
  it('HIP: AMD\'s names, and the inside of a CU from AMD\'s documentation, not from the device', () => {
    const d = noReport('hip');
    expect(row(d, 'unit')).toMatchObject({ name: 'compute unit (CU)', count: null, ask: 'multiProcessorCount' });
    expect(row(d, 'group')).toMatchObject({ name: 'SIMD', count: AMD_CDNA_DOC.simdsPerCu, provenance: 'vendor-doc' });
    expect(row(d, 'core')).toMatchObject({ count: 64, per: 'in each CU', provenance: 'vendor-doc' });
    expect(row(d, 'unit').holds[0].what).toContain('whole blocks');
  });
  it('Metal: no level under the device has a name, a count or a query', () => {
    const d = deviceFromReport('metal', { name: 'Apple GPU' });
    for (const id of ['cluster', 'unit', 'group', 'core']) {
      expect(row(d, id)).toMatchObject({ name: null, count: null, ask: null, provenance: 'none' });
      expect(row(d, id).holds).toEqual([]);
    }
    expect(row(d, 'device')).toMatchObject({ name: 'MTLDevice', count: 1, provenance: 'reported' });
    expect(row(d, 'onchip').name).toBe('threadgroup memory');
  });
  it('a row without a vendor name still says which of the book\'s levels it stands for', () => {
    const ls = levels(deviceFromReport('metal', { name: 'Apple GPU' }));
    expect(ls.filter((l) => l.name == null).map((l) => l.bookName)).toEqual(['GPC', 'SM', 'processing block', 'streaming processor']);
    expect(levels(BOOK_DEVICE).every((l) => l.bookName.length > 0)).toBe(true);
  });
});

describe('waves, in closed form (pp. 73-74)', () => {
  it('Fig. 4.5: 8 blocks give 4 waves at 2 at once and 2 waves at 4 at once', () => {
    expect(waves(FIG45.blocks, capacity(FIG45.small.units, FIG45.small.slotsPerUnit))).toBe(4);
    expect(waves(FIG45.blocks, capacity(FIG45.large.units, FIG45.large.slotsPerUnit))).toBe(2);
  });
  it('660 blocks at 264 at once: 2.5 waves, 132 in the last, half the capacity idle', () => {
    const cap = capacity(TAIL_EXAMPLE.units, TAIL_EXAMPLE.slotsPerUnit);
    expect(cap).toBe(264);
    expect(waves(660, cap)).toBe(2.5);
    expect(waveSizes(660, cap)).toEqual([264, 264, 132]);
    expect(tail(660, cap)).toEqual({ last: 132, idle: 132, idleShare: 0.5 });
  });
  it('1000 blocks at 264 at once: 208 in the last wave, 56 slots idle, 21%', () => {
    const t = tail(1000, 264);
    expect(t.last).toBe(208);
    expect(t.idle).toBe(56);
    expect(Math.round(t.idleShare * 100)).toBe(21);
  });
  it('a multiple of the capacity has no tail', () => {
    expect(tail(792, 264)).toEqual({ last: 264, idle: 0, idleShare: 0 });
    expect(waves(792, 264)).toBe(3);
  });
  it('blocks waiting just after launch', () => {
    expect(waitingAtLaunch(20, capacity(FIG42.units, FIG42.slotsPerUnit))).toBe(14);
    expect(waitingAtLaunch(1000, capacity(40, 6))).toBe(760);
    expect(waitingAtLaunch(4, 6)).toBe(0);
  });
  it('a grid smaller than the capacity is a fraction of a wave', () => {
    expect(waves(132, 264)).toBe(0.5);
    expect(waveSizes(132, 264)).toEqual([132]);
    expect(tail(132, 264).idle).toBe(132);
  });
});

describe('the scheduler', () => {
  it('20 blocks on 2 SMs with 3 slots: 6 start, 14 wait', () => {
    const s = schedule({ blocks: 20, ...FIG42 });
    const st = stateAt(s, 6);
    expect(st.resident).toBe(6);
    expect(st.pending.length).toBe(14);
    expect(st.pending[0]).toBe(6);
    expect(st.slots).toEqual([[0, 2, 4], [1, 3, 5]]);
    expect(s.events.slice(0, 6).every((e) => e.kind === 'assign' && e.time === 0)).toBe(true);
  });
  it('a finished block frees its slot and the next block on the list takes it', () => {
    const s = schedule({ blocks: 20, ...FIG42 });
    expect(s.events[6]).toMatchObject({ kind: 'complete', block: 0, unit: 0, slot: 0, time: 1 });
    expect(s.events[7]).toMatchObject({ kind: 'assign', block: 6, unit: 0, slot: 0, time: 1 });
    expect(s.events.length).toBe(40);
  });
  it('the resident count never exceeds SMs x slots, and every block runs once on one SM', () => {
    for (const spec of [
      { blocks: 20, units: 2, slotsPerUnit: 3 },
      { blocks: 37, units: 5, slotsPerUnit: 2, durations: unevenDurations(37, 9) },
      { blocks: 3, units: 4, slotsPerUnit: 4 },
    ]) {
      const s = schedule(spec);
      for (let n = 0; n <= s.events.length; n++) {
        const st = stateAt(s, n);
        expect(st.resident).toBeLessThanOrEqual(s.capacity);
        expect(st.slots.flat().filter((b) => b !== null).length).toBe(st.resident);
        expect(st.pending.length + st.resident + st.done.length).toBe(spec.blocks);
      }
      expect(s.placements.length).toBe(spec.blocks);
      expect(s.placements.every((p, b) => p.block === b && p.unit < spec.units && p.slot < spec.slotsPerUnit && p.end > p.start)).toBe(true);
      expect(s.events.filter((e) => e.kind === 'assign').length).toBe(spec.blocks);
      // No two blocks overlap in the same slot.
      const bySlot = new Map<string, { start: number; end: number }[]>();
      for (const p of s.placements) bySlot.set(`${p.unit}/${p.slot}`, [...(bySlot.get(`${p.unit}/${p.slot}`) ?? []), p]);
      for (const list of bySlot.values()) {
        list.sort((a, b) => a.start - b.start);
        for (let i = 1; i < list.length; i++) expect(list[i].start).toBeGreaterThanOrEqual(list[i - 1].end);
      }
    }
  });
  it('equal blocks finish in launch order; uneven blocks do not, but the same set finishes', () => {
    const even = schedule({ blocks: 12, ...FIG42 });
    expect(completionOrder(even)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    const uneven = schedule({ blocks: 12, ...FIG42, durations: unevenDurations(12) });
    expect(completionOrder(uneven)).not.toEqual(completionOrder(even));
    expect([...completionOrder(uneven)].sort((a, b) => a - b)).toEqual(completionOrder(even));
  });
  it('any hand-out order finishes the same set of blocks in the same time when blocks are equal', () => {
    const order = shuffledOrder(8, 3);
    expect(order).not.toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    const a = schedule({ blocks: 8, ...FIG45.small });
    const b = schedule({ blocks: 8, ...FIG45.small, order });
    expect(completionOrder(b)).toEqual(order);
    expect([...completionOrder(b)].sort((x, y) => x - y)).toEqual(completionOrder(a));
    expect(b.makespan).toBe(a.makespan);
  });
  it('an order that is not a permutation is rejected', () => {
    expect(() => schedule({ blocks: 3, units: 1, slotsPerUnit: 1, order: [0, 0, 1] })).toThrow();
    expect(() => schedule({ blocks: 3, units: 1, slotsPerUnit: 1, durations: [1] })).toThrow();
  });
});

describe('waves, as the scheduler produces them', () => {
  it('Fig. 4.5: the small device takes 4 time units, the large one 2', () => {
    const small = schedule({ blocks: 8, ...FIG45.small });
    const large = schedule({ blocks: 8, ...FIG45.large });
    expect(profile(small)).toEqual([0, 1, 2, 3].map((t) => ({ from: t, to: t + 1, resident: 2 })));
    expect(profile(large)).toEqual([0, 1].map((t) => ({ from: t, to: t + 1, resident: 4 })));
    expect(profile(small).length).toBe(waves(8, small.capacity));
    expect(small.makespan).toBe(4);
    expect(large.makespan).toBe(2);
    expect(small.placements.map((p) => p.start)).toEqual([0, 0, 1, 1, 2, 2, 3, 3]);
    expect(large.placements.map((p) => p.start)).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
  });
  it('660 on 264: two full waves, then 132 blocks', () => {
    const s = schedule(TAIL_EXAMPLE);
    expect(profile(s)).toEqual([
      { from: 0, to: 1, resident: 264 },
      { from: 1, to: 2, resident: 264 },
      { from: 2, to: 3, resident: 132 },
    ]);
    expect(profile(s).map((g) => g.resident)).toEqual(waveSizes(660, 264));
    expect(s.makespan).toBe(Math.ceil(waves(660, 264)));
    const u = usage(s);
    expect(u.work).toBe(660);
    expect(u.available).toBe(792);
    expect(u.utilisation).toBeCloseTo(660 / 792, 12);
    expect(u.peak).toBe(264);
  });
  it('a multiple of the capacity uses every slot the whole time', () => {
    expect(usage(schedule({ blocks: 792, units: 132, slotsPerUnit: 2 })).utilisation).toBe(1);
  });
  it('uneven blocks: one wave leaves slots idle, many waves even the load out', () => {
    const cap = { units: 4, slotsPerUnit: 2 };
    const one = usage(schedule({ blocks: 8, ...cap, durations: unevenDurations(8) }));
    const many = usage(schedule({ blocks: 80, ...cap, durations: unevenDurations(80) }));
    expect(one.utilisation).toBeLessThan(0.8);
    expect(many.utilisation).toBeGreaterThan(0.9);
    // In one wave each slot runs one block, so the device waits for the longest block.
    expect(schedule({ blocks: 8, ...cap, durations: unevenDurations(8) }).makespan).toBe(Math.max(...unevenDurations(8)));
  });
  it('durations are seeded and stay within 1 to 4', () => {
    expect(unevenDurations(50)).toEqual(unevenDurations(50));
    expect(unevenDurations(50).every((d) => Number.isInteger(d) && d >= 1 && d <= 4)).toBe(true);
    expect(new Set(unevenDurations(50)).size).toBe(4);
    expect(equalDurations(3)).toEqual([1, 1, 1]);
  });
  it('a grid that fits at once is a single wave, and a large grid is handled', () => {
    expect(profile(schedule({ blocks: 4, units: 4, slotsPerUnit: 2 }))).toEqual([{ from: 0, to: 1, resident: 4 }]);
    const big = schedule({ blocks: 50000, units: 132, slotsPerUnit: 32 });
    expect(big.makespan).toBe(Math.ceil(50000 / 4224));
  });
});
