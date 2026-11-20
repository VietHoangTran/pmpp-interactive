import { describe, expect, it } from 'vitest';
import {
  AMD_CDNA_L2, BUS_PRESETS, DEMAND, FIG_6_9, banksNeeded, busBandwidth, busBusy, busGaps, busesFor, bytesInFlight,
  channelTimeline, columnChannels, deal, elementsToFill, inFlight, interleave, phaseChannels, utilisationBound,
} from './dram';

// Every number section 6.2 states is pinned here, with the page it comes from.
describe('bus bandwidth (p. 133)', () => {
  it('64-bit DDR5-6400 at 3.2 GHz, double data rate: 51.2 GB/s', () => {
    const d = BUS_PRESETS.ddr5;
    expect(busBandwidth(d.widthBits, d.clockGHz, d.ddr, d.buses)).toBeCloseTo(51.2, 9);
    expect(busBandwidth(64, 3.2, false)).toBeCloseTo(25.6, 9);
  });
  it('four buses for a CPU at 200 GB/s, twenty for a GPU at 1000 GB/s', () => {
    const per = busBandwidth(64, 3.2, true);
    expect(DEMAND.cpu / per).toBeCloseTo(3.906, 3);
    expect(busesFor(DEMAND.cpu, per)).toBe(4);
    expect(DEMAND.gpu / per).toBeCloseTo(19.53, 2);
    expect(busesFor(DEMAND.gpu, per)).toBe(20);
  });
  it("the book's HBM3E inputs give 1254.4 GB/s, not the printed 1 TB/s (erratum)", () => {
    const h = BUS_PRESETS.hbm3e;
    expect(busBandwidth(h.widthBits, h.clockGHz, h.ddr, h.buses)).toBeCloseTo(1254.4, 6);
  });
  it('an exact multiple needs no extra bus', () => {
    expect(busesFor(102.4, 51.2)).toBe(2);
    expect(busesFor(102.5, 51.2)).toBe(3);
  });
});

describe('one channel and its banks (pp. 133 to 135, Fig. 6.8)', () => {
  it('R = 20, one bank: the bus is busy 1/21 of the time, 4.76 %', () => {
    const tl = channelTimeline({ R: 20, banks: 1 });
    expect(tl.utilisation).toBeCloseTo(1 / 21, 12);
    expect(Math.round(tl.utilisation * 1000) / 10).toBe(4.8);
  });
  it('so a 16 GB/s channel delivers 0.76 GB/s', () => {
    const tl = channelTimeline({ R: 20, banks: 1 });
    expect(tl.utilisation * 16).toBeCloseTo(0.762, 3);
  });
  it('a second bank doubles it, and each bank adds the same until R + 1', () => {
    for (const R of [1, 5, 20, 40])
      for (let banks = 1; banks <= 48; banks++)
        expect(channelTimeline({ R, banks }).utilisation).toBeCloseTo(utilisationBound(R, banks), 9);
    expect(channelTimeline({ R: 20, banks: 2 }).utilisation).toBeCloseTo(2 / 21, 12);
  });
  it('R + 1 banks fill the bus: 21 for R = 20 (p. 134), 31 for R = 30', () => {
    expect(banksNeeded(20)).toBe(21);
    expect(banksNeeded(30)).toBe(31);
    expect(channelTimeline({ R: 20, banks: 21 }).utilisation).toBe(1);
    expect(channelTimeline({ R: 20, banks: 20 }).utilisation).toBeLessThan(1);
    expect(channelTimeline({ R: 30, banks: 31 }).utilisation).toBe(1);
  });
  it('q-util: R = 15, two banks, a 32 GB/s bus delivers 4 GB/s (12.5 %)', () => {
    const tl = channelTimeline({ R: 15, banks: 2 });
    expect(tl.utilisation).toBeCloseTo(0.125, 12);
    expect(tl.utilisation * 32).toBeCloseTo(4, 9);
  });
  it('one bank serves one access at a time, and bursts never overlap on the bus', () => {
    const tl = channelTimeline({ R: 7, banks: 3, conflictShare: 0.4, seed: 5 });
    for (const segs of tl.banks)
      for (let i = 1; i < segs.length; i++) expect(segs[i].start).toBeGreaterThanOrEqual(segs[i - 1].end);
    for (let i = 1; i < tl.bus.length; i++) expect(tl.bus[i].start).toBeGreaterThanOrEqual(tl.bus[i - 1].end);
    for (const a of tl.accesses) expect(a.burst - a.start).toBe(7);
  });
  it('without conflicts the banks are used in turn, one start per burst time', () => {
    const tl = channelTimeline({ R: 20, banks: 21 });
    expect(tl.accesses.slice(0, 23).map((a) => a.bank)).toEqual([...Array.from({ length: 21 }, (_, i) => i), 0, 1]);
    expect(tl.accesses.slice(0, 5).map((a) => a.start)).toEqual([0, 1, 2, 3, 4]);
    expect(tl.accesses[0].burst).toBe(20);
  });
  it('a conflict makes the request wait out the previous access and opens a gap of R on the bus', () => {
    const tl = channelTimeline({ R: 20, banks: 21, forced: [30] });
    const a = tl.accesses[30];
    expect(a.conflict).toBe(true);
    expect(a.bank).toBe(tl.accesses[29].bank);
    expect(a.start).toBe(tl.accesses[29].burst + 1);
    const gaps = busGaps(tl, tl.accesses[21].burst, tl.accesses[40].burst);
    expect(gaps).toEqual([[tl.accesses[29].burst + 1, a.burst]]);
    expect(a.burst - (tl.accesses[29].burst + 1)).toBe(20);
  });
  it('conflicts lower the utilisation, more of them lower it more, and the draws repeat', () => {
    const u = (s: number) => channelTimeline({ R: 20, banks: 32, conflictShare: s, seed: 3 }).utilisation;
    expect(u(0)).toBe(1);
    expect(u(0.1)).toBeLessThan(1);
    expect(u(0.3)).toBeLessThan(u(0.1));
    expect(u(0.3)).toBe(u(0.3));
  });
  it('with one bank there is nothing to conflict with', () => {
    const tl = channelTimeline({ R: 20, banks: 1, conflictShare: 1 });
    expect(tl.accesses.some((a) => a.conflict)).toBe(false);
    expect(tl.utilisation).toBeCloseTo(1 / 21, 12);
  });
  it('a full bus needs R + 1 = 21 requests in flight at once', () => {
    const tl = channelTimeline({ R: 20, banks: 21 });
    expect(inFlight(tl, 100.5)).toBe(21);
    expect(inFlight(channelTimeline({ R: 20, banks: 48 }), 100.5)).toBe(21);
    expect(inFlight(channelTimeline({ R: 20, banks: 2 }), 100.5)).toBe(2);
  });
  it('the busy fraction of a window agrees with the steady-state figure over whole rounds', () => {
    const tl = channelTimeline({ R: 20, banks: 1 });
    expect(busBusy(tl, 20, 20 + 21 * 5)).toBeCloseTo(1 / 21, 12);
    expect(busBusy(channelTimeline({ R: 20, banks: 21 }), 20, 120)).toBe(1);
  });
});

describe('interleaving (p. 135, Fig. 6.9)', () => {
  const fig = (i: number) => interleave(i, 4, 8, 4, 2);
  it('reproduces the figure: channels first, then banks', () => {
    const table = [
      [0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0], [4, 2, 0], [5, 2, 0], [6, 3, 0], [7, 3, 0],
      [8, 0, 1], [9, 0, 1], [10, 1, 1], [11, 1, 1], [12, 2, 1], [13, 2, 1], [14, 3, 1], [15, 3, 1],
    ];
    for (const [i, ch, bank] of table) expect([i, fig(i).channel, fig(i).bank]).toEqual([i, ch, bank]);
  });
  it('wraps: M[16..17] back to channel 0, bank 0; M[18..19] to channel 1, bank 0', () => {
    expect(fig(16)).toEqual({ burst: 8, channel: 0, bank: 0 });
    expect(fig(17)).toEqual({ burst: 8, channel: 0, bank: 0 });
    expect(fig(18)).toEqual({ burst: 9, channel: 1, bank: 0 });
  });
  it('dealing 16 elements fills every channel and bank with the pairs of the figure', () => {
    expect(deal(16, FIG_6_9)).toEqual([
      [[0, 1], [8, 9]],
      [[2, 3], [10, 11]],
      [[4, 5], [12, 13]],
      [[6, 7], [14, 15]],
    ]);
  });
  it('16 elements give every channel and bank of the toy one burst each (p. 135)', () => {
    expect(elementsToFill(FIG_6_9)).toEqual({ channels: 8, banks: 16 });
    const cells = deal(16, FIG_6_9);
    expect(cells.flat().every((c) => c.length === 2)).toBe(true);
    // Fewer leave some bank empty.
    expect(deal(15, FIG_6_9).flat().filter((c) => c.length === 2)).toHaveLength(7);
  });
  it("AMD's interleave: 2048 floats (8192 bytes) give all 32 L2 channels 256 bytes each; 16 floats sit in channel 0", () => {
    expect(elementsToFill(AMD_CDNA_L2).channels).toBe(2048);
    expect(deal(16, AMD_CDNA_L2)[0][0]).toHaveLength(16);
    expect(deal(2048, AMD_CDNA_L2).every((ch) => ch[0].length === 64)).toBe(true);
  });
  it('q-onechannel: column 0 of a 2048-wide float matrix stays in one of AMD\'s 32 L2 channels', () => {
    expect(AMD_CDNA_L2).toEqual({ elemBytes: 4, burstBytes: 256, channels: 32, banks: 1 });
    expect(columnChannels(2048, 0, 256, AMD_CDNA_L2)).toEqual([0]);
    // A row of 2048 floats is 8192 bytes = 32 x 256: every row starts in the same channel.
    expect((2048 * 4) / 256).toBe(32);
    // One element wider, and the column walks across all 32.
    expect(columnChannels(2049, 0, 2048, AMD_CDNA_L2)).toHaveLength(32);
    // A whole row, read along, uses all 32.
    expect(new Set(Array.from({ length: 2048 }, (_, c) => interleave(c, 4, 256, 32, 1).channel)).size).toBe(32);
  });
});

describe('the toy matrix multiplication (pp. 136 to 137, Figs. 6.10 and 6.11)', () => {
  const four = phaseChannels(4, 2);
  it('phase 0: blocks 0,0 and 0,1 load M[0], M[1], M[4], M[5]; blocks 1,0 and 1,1 load M[8], M[9], M[12], M[13]', () => {
    const at = (row: number, col: number) => four[0].blocks.find((b) => b.row === row && b.col === col)!.elements;
    expect(at(0, 0)).toEqual([0, 1, 4, 5]);
    expect(at(0, 1)).toEqual([0, 1, 4, 5]);
    expect(at(1, 0)).toEqual([8, 9, 12, 13]);
    expect(at(1, 1)).toEqual([8, 9, 12, 13]);
  });
  it('phase 1: M[2], M[3], M[6], M[7] and M[10], M[11], M[14], M[15]', () => {
    expect(four[1].blocks[0].elements).toEqual([2, 3, 6, 7]);
    expect(four[1].blocks[3].elements).toEqual([10, 11, 14, 15]);
  });
  it('under Fig. 6.9: channels 0 and 2 in phase 0, then 1 and 3', () => {
    expect(four.map((p) => p.channels)).toEqual([[0, 2], [1, 3]]);
    expect(four[0].blocks[0].channels).toEqual([0, 2]);
  });
  it('8 x 8: every phase\'s M loads land in a single channel, channel ph (notes item 8, facts E1)', () => {
    const eight = phaseChannels(8, 2);
    expect(eight).toHaveLength(4);
    expect(eight.map((p) => p.channels)).toEqual([[0], [1], [2], [3]]);
    expect(eight[0].blocks).toHaveLength(16);
    // Element 8r + c is in burst 4r + floor(c/2).
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) expect(interleave(8 * r + c, 4, 8, 4, 2).burst).toBe(4 * r + Math.floor(c / 2));
  });
});

describe('bytes in flight (ours: Little\'s law)', () => {
  it('GB/s times ns is bytes', () => {
    const f = bytesInFlight(3350, 50, 4, 32);
    expect(f.bytes).toBeCloseTo(167500, 6);
    expect(f.loads).toBeCloseTo(41875, 6);
    expect(f.warpLoads).toBeCloseTo(1308.59, 2);
  });
  it('16-byte loads need a quarter of the loads for the same bytes; a 64-wide wavefront half the instructions', () => {
    const a = bytesInFlight(1000, 100, 4, 32);
    const b = bytesInFlight(1000, 100, 16, 32);
    const c = bytesInFlight(1000, 100, 4, 64);
    expect(b.loads).toBeCloseTo(a.loads / 4, 9);
    expect(c.warpLoads).toBeCloseTo(a.warpLoads / 2, 9);
  });
});
