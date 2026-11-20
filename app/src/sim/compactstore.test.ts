import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { DESIGNS, PRESET, SCENE, makeKeep, presetMap, sceneData, storeMap, type Config, type Design } from './compactstore';
import { BOOK_KEEP, exclusiveScan } from './stablescan';

const cfg = (design: Design, over: Partial<Config> = {}): Config => ({
  keep: makeKeep(4096, 0.5, 'random', 1), blockDim: 128, warp: 32, coarse: 8, segmentBytes: 32, design, ...over,
});

describe('every design writes the same output', () => {
  it('each kept key lands at its exclusive-scan offset, written exactly once', () => {
    for (const d of DESIGNS)
      for (const f of [0, 0.03, 0.5, 1])
        for (const pattern of ['random', 'runs', 'alternate'] as const) {
          const keep = makeKeep(1000, f, pattern, 7);
          const m = storeMap({ keep, blockDim: 64, warp: 32, coarse: 4, segmentBytes: 32, design: d });
          const slots = m.stores.flatMap((s) => s.slots).sort((a, b) => a - b);
          const kept = keep.reduce((a, b) => a + b, 0);
          expect(slots).toEqual(Array.from({ length: kept }, (_, i) => i));
          expect(m.writer.every((w) => w >= 0)).toBe(true);
          expect(m.offsets).toEqual(exclusiveScan(keep));
        }
  });
});

describe('what each figure does to the stores', () => {
  it('Fig. 12.8: within a warp the kept lanes write consecutive slots, in lane order (q-direct)', () => {
    const m = storeMap(cfg('direct'));
    for (const s of m.stores) {
      for (let k = 1; k < s.slots.length; k++) expect(s.slots[k]).toBe(s.slots[k - 1] + 1);
      for (let k = 1; k < s.lanes.length; k++) expect(s.lanes[k]).toBeGreaterThan(s.lanes[k - 1]);
    }
    // 12 of 32 lanes kept: 12 consecutive slots.
    const keep = new Array<number>(32).fill(0);
    for (const l of [0, 3, 4, 5, 9, 13, 14, 20, 21, 25, 30, 31]) keep[l] = 1;
    const one = storeMap({ keep, blockDim: 32, warp: 32, coarse: 1, segmentBytes: 32, design: 'direct' });
    expect(one.stores).toHaveLength(1);
    expect(one.stores[0].slots).toEqual(Array.from({ length: 12 }, (_, i) => i));
  });
  it('Figs. 12.9 and 12.10: a staged block writes exactly one contiguous run, with consecutive lanes', () => {
    for (const d of ['staged', 'staged-coarse'] as const)
      for (const c of [1, 2, 8]) {
        const m = storeMap(cfg(d, { coarse: c, keep: makeKeep(3000, 0.37, 'runs', 3) }));
        for (let b = 0; b < m.totals.blocks; b++) {
          const slots = m.stores.filter((s) => s.block === b).flatMap((s) => s.slots).sort((x, y) => x - y);
          const { start, length } = m.runs[b];
          expect(slots).toEqual(Array.from({ length }, (_, i) => start + i));
        }
        for (const s of m.stores) for (let k = 1; k < s.lanes.length; k++) expect(s.lanes[k]).toBe(s.lanes[k - 1] + 1);
      }
  });
  it('coarsening by c divides the chunk count by c (q-chunks: 4096 keys, 128 threads, c = 4: 8 chunks)', () => {
    const keep = makeKeep(4096, 0.5, 'random', 1);
    const chunks = (c: number) => storeMap({ keep, blockDim: 128, warp: 32, coarse: c, segmentBytes: 32, design: 'staged-coarse' }).totals.chunks;
    expect(chunks(1)).toBe(32);
    expect(chunks(2)).toBe(16);
    expect(chunks(4)).toBe(8);
    expect(chunks(8)).toBe(4);
    const m = storeMap({ keep, blockDim: 128, warp: 32, coarse: 4, segmentBytes: 32, design: 'staged-coarse' }).totals;
    expect(m.keysPerBlock).toBe(512);
    expect(m.scanInputsPerKey).toBe(0.25);
  });
  it('q-budget: staging 4-byte keys for 256 threads and c = 8 takes 8192 bytes', () => {
    expect(storeMap(cfg('staged-coarse', { blockDim: 256, coarse: 8 })).totals.stagingBytes).toBe(8192);
    expect(storeMap(cfg('direct-coarse', { blockDim: 256, coarse: 8 })).totals.stagingBytes).toBe(0);
  });
  it('designs without coarsening ignore the factor', () => {
    expect(storeMap(cfg('direct', { coarse: 8 })).totals).toEqual(storeMap(cfg('direct', { coarse: 1 })).totals);
    expect(storeMap(cfg('staged', { coarse: 8 })).totals.keysPerBlock).toBe(128);
  });
});

describe('the explorer\'s preset (Predict ch12/12.6/p-stores)', () => {
  it('1024 keys, half kept at random, blocks of 256, warps of 32, 32-byte segments', () => {
    expect(PRESET).toMatchObject({ n: 1024, fraction: 0.5, blockDim: 256, warp: 32, coarse: 1, segmentBytes: 32 });
    const d = presetMap('direct').totals;
    const s = presetMap('staged').totals;
    expect(d.kept).toBe(500);
    expect(d.stores).toBe(32); // every warp keeps something
    expect(s.stores).toBe(18);
    expect(d.segments).toBe(88);
    expect(s.segments).toBe(77);
    expect(s.chunks).toBe(4);
  });
});

// The sentences the lesson writes about the explorer, each pinned here.
describe('what the explorer shows (4096 keys, blocks of 128, warps of 32, 32-byte segments)', () => {
  const run = (d: Design, f: number, c = 1) =>
    storeMap({ keep: makeKeep(4096, f, 'random', 1), blockDim: 128, warp: 32, coarse: c, segmentBytes: 32, design: d }).totals;
  it('one key per thread: staging cuts store instructions by more than half at a quarter kept, and not at all with everything kept', () => {
    expect(run('staged', 0.25).stores).toBeLessThan(run('direct', 0.25).stores / 2);
    expect(run('staged', 1).stores).toBe(run('direct', 1).stores);
    expect(run('staged', 1).segments).toBe(run('direct', 1).segments);
    expect(run('direct', 1).efficiency).toBe(1);
  });
  it('one key per thread: the saving in segments shrinks as the fraction grows', () => {
    const saving = (f: number) => 1 - run('staged', f).segments / run('direct', f).segments;
    expect(saving(0.1)).toBeGreaterThan(0.4);
    expect(saving(0.5)).toBeLessThan(0.2);
    expect(saving(0.95)).toBeLessThan(0.01);
  });
  it('coarsened without staging, the lanes\' stores are strided: one segment per key with everything kept', () => {
    expect(run('direct-coarse', 1, 8).segments).toBe(4096);
    expect(run('staged-coarse', 1, 8).segments).toBe(512);
    expect(run('direct-coarse', 0.5, 8).segments).toBeGreaterThan(5 * run('staged-coarse', 0.5, 8).segments);
  });
  it('at low fractions the stores are a small part of the traffic whichever way they are written (the case where CUB skips staging)', () => {
    expect(run('direct-coarse', 0.02, 8).storeShare).toBeLessThan(0.15);
    expect(run('direct-coarse', 0.5, 8).storeShare).toBeGreaterThan(0.7);
    expect(run('staged-coarse', 0.5, 8).storeShare).toBeLessThan(0.4);
  });
});

describe('the narrated scene "Gather, then write together"', () => {
  const D = sceneData();
  it('the book\'s sixteen keys, two blocks of eight threads, warps of four, 16-byte segments', () => {
    expect(D.keep).toEqual([...BOOK_KEEP]);
    expect(D.offsets).toEqual([0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 7, 8]);
  });
  it('direct scatter: four stores of 2, 2, 3 and 1 lanes, four part-filled segments, 64 bytes fetched for 32 useful', () => {
    expect(D.direct.stores.map((s) => s.lanes.length)).toEqual([2, 2, 3, 1]);
    expect(D.direct.stores.map((s) => s.slots)).toEqual([[0, 1], [2, 3], [4, 5, 6], [7]]);
    expect(D.direct.totals).toMatchObject({ stores: 4, segments: 4, bytesFetched: 64, bytesUseful: 32, chunks: 2 });
  });
  it('staged: two stores of four lanes, two full segments', () => {
    expect(D.staged.stores.map((s) => [s.block, s.lanes, s.slots])).toEqual([
      [0, [0, 1, 2, 3], [0, 1, 2, 3]],
      [1, [0, 1, 2, 3], [4, 5, 6, 7]],
    ]);
    expect(D.staged.totals).toMatchObject({ stores: 2, segments: 2, bytesFetched: 32, efficiency: 1, chunks: 2, stagingBytes: 32 });
  });
  it('coarsened by two: one block takes all sixteen keys, one chunk of eight, still two stores, half a scan input per key', () => {
    expect(D.coarse.totals).toMatchObject({ blocks: 1, keysPerBlock: 16, stores: 2, segments: 2, chunks: 1, meanChunk: 8, scanInputsPerKey: 0.5 });
  });
  const repo = path.resolve(__dirname, '../../..');
  const script = (n: number) => fs.readFileSync(path.join(repo, `content/ch12/narration/stage-write-${n}.txt`), 'utf8');
  it('the scripts say the model\'s numbers', () => {
    expect(script(1)).toContain(`${SCENE.warp} lanes`);
    expect(script(4)).toContain(`${D.direct.totals.stores} store instructions, touching ${D.direct.totals.segments} segments, ${D.direct.totals.bytesFetched} bytes moved for ${D.direct.totals.bytesUseful} useful`);
    expect(script(4)).toContain(`${D.staged.totals.stores} store instructions and ${D.staged.totals.segments} full segments`);
    expect(script(5)).toContain(`Coarsening by ${SCENE.coarse}`);
    expect(script(5)).toContain(`Its run is ${D.coarse.totals.meanChunk} keys long: ${D.coarse.totals.chunks} chunk instead of ${D.staged.totals.chunks}`);
    expect(script(5)).toContain(`It still needs ${D.coarse.totals.stores} stores`);
    expect(script(6)).toContain(`each block keeps ${D.staged.runs[0].length} keys for ${SCENE.blockDim} threads`);
    expect(D.staged.runs.every((r) => r.length <= SCENE.blockDim)).toBe(true); // CUB's rule would not stage (ch12-facts A9)
  });
  it('each script is 35 to 60 words', () => {
    for (let n = 1; n <= 6; n++) {
      const words = script(n).replace(/\[[^\]]*\]/g, '').trim().split(/\s+/).length;
      expect(words, `stage-write-${n}`).toBeGreaterThanOrEqual(35);
      expect(words, `stage-write-${n}`).toBeLessThanOrEqual(60);
    }
  });
});
