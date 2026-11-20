import { describe, expect, it } from 'vitest';
import {
  BOOK_PROBLEM,
  C_CHOICES,
  PROBLEMS,
  R_CHOICES,
  VOCAB,
  blocks,
  directTraffic,
  fastest,
  keysPerBucket,
  localTraffic,
  measuredLocal,
  measuredOneBit,
  passes,
  sampled,
  sampledDirect,
  sharedBytes,
  sweepC,
  sweepR,
  tableEntries,
  tileBytes,
  type LabRow,
} from './radixcost';

const N22 = { rows: 1, rowLength: 2 ** 22 };

describe('passes and launches (p. 342)', () => {
  it('ceil(key bits / r)', () => {
    expect(passes(16, 6)).toBe(3); // q-passes: bf16 keys at r = 6
    expect(passes(16, 8)).toBe(2);
    expect(passes(32, 8)).toBe(4);
    expect(passes(32, 7)).toBe(5);
    expect(passes(32, 1)).toBe(32);
    expect(passes(4, 2)).toBe(2); // the book's 4-bit keys in two 2-bit passes (Fig. 14.10)
  });
  it('three launches per pass', () => {
    expect(localTraffic(N22, 32, 8, 256, 1).launches).toBe(12);
  });
});

describe('the table (pp. 343-345)', () => {
  it('q-table (14.7): 2^22 keys, 256 threads, c = 1, r = 8: as many entries as keys', () => {
    expect(tableEntries(N22, 8, 256, 1)).toBe(4194304);
    expect(blocks(N22, 256, 1)).toBe(16384);
  });
  it('q-table (14.8): c = 16 shrinks it sixteen times', () => {
    expect(tableEntries(N22, 8, 256, 16)).toBe(262144);
    expect(blocks(N22, 256, 16)).toBe(1024);
  });
  it("the book's 2-bit example: 16 entries over four blocks, 8 over two (Figs. 14.12, 14.13)", () => {
    expect(tableEntries(BOOK_PROBLEM, 2, 4, 1)).toBe(16);
    expect(tableEntries(BOOK_PROBLEM, 2, 4, 2)).toBe(8);
    expect(keysPerBucket(4, 1, 2)).toBe(1);
    expect(keysPerBucket(4, 2, 2)).toBe(2);
  });
  it('a batch of rows: no block spans two rows', () => {
    const batch = PROBLEMS.find((p) => p.id === 'batch')!;
    expect(blocks(batch, 256, 1)).toBe(64 * 594);
    expect(Math.ceil(VOCAB / 256)).toBe(594); // keys per thread when one block of 256 threads takes a whole row
  });
});

describe('shared memory (ours, from chapters 5 and 6)', () => {
  it('q-smem: c = 8, 256 threads, (uint32, uint32) pairs: 16,384 bytes of tile', () => {
    expect(tileBytes(256, 8)).toBe(16384);
    expect(sharedBytes(256, 8, 8)).toBe(16384 + 1024);
  });
});

describe('traffic under the model', () => {
  it('falls as 1/r until the table catches up with the data, then explodes', () => {
    const p = PROBLEMS[0];
    const t = R_CHOICES.map((r) => localTraffic(p, 32, r, 256, 1).perKey);
    expect(t[0]).toBe(32 * 20 + 32 * (2 * 4 * 4 * 4096) / 2 ** 20); // 32 passes of 4 + 2 x 8 bytes, plus the table
    for (let i = 1; i < 7; i++) expect(t[i]).toBeLessThan(t[i - 1]);
    expect(t[R_CHOICES.indexOf(16)]).toBeGreaterThan(10 * t[0]);
    expect(t[R_CHOICES.indexOf(16)]).toBeGreaterThan(50 * t[R_CHOICES.indexOf(8)]);
  });
  it('14.7 bridge: bf16 keys at r = 8 take 2 passes, and the table is a larger share of each pass than with 32-bit keys', () => {
    for (const p of PROBLEMS) {
      const k16 = localTraffic(p, 16, 8, 256, 1);
      const k32 = localTraffic(p, 32, 8, 256, 1);
      expect(k16.passes).toBe(2);
      expect(k32.passes).toBe(4);
      expect(k16.tablePerPass).toBe(k32.tablePerPass);
      const share = (t: typeof k16) => t.tablePerPass / (t.tablePerPass + t.keysPerPass);
      expect(share(k16)).toBeGreaterThan(share(k32));
    }
  });
  it('the one-bit sort of section 14.5 for reference: 32 passes of pairs and a bits array', () => {
    const d = directTraffic(PROBLEMS[0], 32);
    expect(d.passes).toBe(32);
    expect(d.perKey).toBe(32 * (16 + 16));
  });
});

describe('sampled runs and sectors', () => {
  it('is deterministic for a seed, and changes with it', () => {
    const s = { dist: 'uniform' as const, keyBits: 32 as const, rowLength: 2 ** 20, r: 4, threads: 256, c: 1 };
    expect(sampled(s)).toEqual(sampled(s));
    expect(sampled({ ...s, seed: 2 }).meanUnits).not.toBe(sampled(s).meanUnits);
  });
  it('wider digits lengthen nothing: runs per warp climb with r at one key per thread', () => {
    const rows = sweepR(PROBLEMS[0], 32, 256);
    for (let i = 1; i < 8; i++) expect(rows[i].local.meanRuns).toBeGreaterThan(rows[i - 1].local.meanRuns);
    expect(rows[0].local.meanMin).toBe(4);
    expect(rows[7].local.meanUnits).toBeGreaterThan(4 * rows[0].local.meanUnits);
  });
  it('coarsening lengthens runs and shrinks the table', () => {
    const rows = sweepC(PROBLEMS[0], 32, 8, 256);
    expect(rows.map((r) => r.c)).toEqual(C_CHOICES);
    for (let i = 1; i < rows.length; i++) {
      expect(rows[i].local.meanUnits).toBeLessThan(rows[i - 1].local.meanUnits);
      expect(rows[i].tableEntries).toBe(rows[i - 1].tableEntries / 2);
    }
  });
  it('the local design never costs more units than the direct scatter on the same sample', () => {
    for (const r of [1, 4, 8]) {
      const l = sampled({ dist: 'uniform', keyBits: 32, rowLength: 2 ** 20, r, threads: 256, c: 1 });
      expect(l.meanUnits).toBeLessThanOrEqual(sampledDirect(PROBLEMS[0], 32, r, 256).meanUnits);
    }
  });
  it('16-bit keys: two bytes each, so a coalesced warp store is two sectors', () => {
    expect(sampled({ dist: 'softmax', keyBits: 16, rowLength: VOCAB, r: 8, threads: 256, c: 1 }).meanMin).toBe(2);
  });
});

describe("reading lab 14.2's rows", () => {
  const rows: LabRow[] = [
    { name: 'onebit_global', seconds: 0.04 },
    { name: 'local', radix_bits: 1, keys_per_thread: 1, seconds: 0.02 },
    { name: 'local', radix_bits: 4, keys_per_thread: 1, seconds: 0.006 },
    { name: 'local', radix_bits: 8, keys_per_thread: 1, seconds: 0.008 },
    { name: 'local', radix_bits: 8, keys_per_thread: 4, seconds: 0.005 },
    { name: 'local', radix_bits: 8, keys_per_thread: 'x', seconds: 0.001 },
  ];
  it('picks rows by name, digit width and keys per thread', () => {
    expect(measuredOneBit(rows)).toBe(0.04);
    expect(measuredLocal(rows, 1, 1)).toBe(0.02);
    expect(measuredLocal(rows, 8, 4)).toBe(0.005);
    expect(measuredLocal(rows, 2, 1)).toBeUndefined();
    expect(measuredOneBit([])).toBeUndefined();
  });
  it('finds the fastest setting that ran', () => {
    expect(fastest([1, 2, 4, 8].map((r) => ({ at: r, seconds: measuredLocal(rows, r, 1) })))).toEqual({ at: 4, seconds: 0.006 });
    expect(fastest([{ at: 1, seconds: undefined }])).toBeUndefined();
  });
});
