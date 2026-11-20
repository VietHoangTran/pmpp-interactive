import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  BOOK_LAUNCH, CUDA_BLOCK_DIMS, CUDA_GRID_DIMS, LAB_PROBLEMS, RULES, appleRecipe, checkLaunch, launchLimitsFromReport,
  occupancyLimits, ruleName, suggest, type LaunchLimits, type RuleKey,
} from './launchconfig';
import { BOOK, occupancy, pct } from './occupancy';
import { AS_PRINTED, FIELDS, HUNT_ANSWER, metalQueryExcerpt } from '../chapters/ch04/S48';

const repo = path.resolve(__dirname, '../../..');
const BIG: [number, number, number] = [2147483647, 65535, 65535]; // large enough never to bind, as in the lab

/** One row of lab 4.1's table of synthetic devices (main.cpp, synthetic()). 0 means not reported. */
const synthetic = (
  warp: number, perBlock: number, dims: [number, number, number], threadSlots: number, blockSlots: number,
  grid: [number, number, number] = BIG,
): LaunchLimits => ({
  width: warp, maxThreadsPerBlock: perBlock, maxThreadsDim: dims, maxGridSize: grid,
  threadSlots: threadSlots || null, blockSlots: blockSlots || null, blockSlotsDerived: false,
});

const TABLE: { label: string; d: LaunchLimits; impossible?: boolean }[] = [
  { label: "width 32, 2048 / 32 slots (the book's example limits)", d: synthetic(32, 1024, [1024, 1024, 64], 2048, 32) },
  { label: 'width 32, 1536 / 16 slots', d: synthetic(32, 1024, [1024, 1024, 64], 1536, 16) },
  { label: 'width 64, 256 threads per block, 2048 / 32 slots', d: synthetic(64, 256, [256, 256, 256], 2048, 32) },
  { label: 'width 32, maxThreadsDim 128 x 128 x 64', d: synthetic(32, 1024, [128, 128, 64], 2048, 32) },
  { label: 'thread slots and block slots not reported', d: synthetic(32, 1024, [1024, 1024, 64], 0, 0) },
  { label: '1536 thread slots, block slots not reported', d: synthetic(32, 1024, [1024, 1024, 64], 1536, 0) },
  { label: 'grid limit 2 x 2 x 1', d: synthetic(32, 1024, [1024, 1024, 64], 2048, 32, [2, 2, 1]), impossible: true },
];

const status = (d: LaunchLimits, rows: number, cols: number, x: number, y: number) =>
  Object.fromEntries(checkLaunch(d, rows, cols, { x, y }).rules.map((r) => [r.key, r.status])) as Record<RuleKey, string>;

describe("the rules are lab 4.1's", () => {
  const main = fs.readFileSync(path.join(repo, 'labs/ch04/lab1_launchconfig/main.cpp'), 'utf8');
  it('every rule name is one the lab driver prints', () => {
    for (const r of RULES) expect(main, r.name).toContain(`rule("${r.name}"`);
  });
  it("the lab's first synthetic device is the book's example as this model has it", () => {
    expect(main).toContain('synthetic(32, 1024, 1024, 1024, 64, 2048, 32, kMaxX, kMaxYZ, kMaxYZ)');
    expect(BOOK_LAUNCH).toEqual(synthetic(32, 1024, CUDA_BLOCK_DIMS, 2048, 32, CUDA_GRID_DIMS));
    expect(main).toContain('{{1, 4096}, {3, 40}, {64, 4096}, {1000, 1000}}');
    expect(LAB_PROBLEMS.map((p) => [p.rows, p.cols])).toEqual([[1, 4096], [3, 40], [64, 4096], [1000, 1000]]);
  });
  it('the per-SM part agrees with the occupancy model of section 4.7', () => {
    expect(occupancyLimits(BOOK_LAUNCH)).toEqual({ ...BOOK, regs: null, shared: null });
  });
});

describe('suggest finds a configuration that passes every rule', () => {
  for (const dev of TABLE)
    it(dev.label, () => {
      for (const p of LAB_PROBLEMS) {
        const s = suggest(dev.d, p.rows, p.cols);
        if (dev.impossible && p.rows !== 3) {
          expect(s, `${p.rows} x ${p.cols}`).toBeNull(); // no block within the limits covers it with a 2 x 2 grid
          continue;
        }
        expect(s, `${p.rows} x ${p.cols}`).not.toBeNull();
        const c = checkLaunch(dev.d, p.rows, p.cols, s!);
        expect(c.failed, `${p.rows} x ${p.cols}: ${s!.x} x ${s!.y}`).toEqual([]);
        expect(c.ok).toBe(true);
        // Whole warps along x: every warp is a run of consecutive outputs of one row.
        expect(s!.x % (dev.d.width as number)).toBe(0);
      }
    });
  it("the configurations the lesson quotes for the book's example", () => {
    expect(suggest(BOOK_LAUNCH, 1, 4096)).toEqual({ x: 64, y: 1 });
    expect(suggest(BOOK_LAUNCH, 64, 4096)).toEqual({ x: 32, y: 32 });
    expect(suggest(BOOK_LAUNCH, 3, 40)).toEqual({ x: 32, y: 2 });
  });
  it('the 1536 / 16 device needs a wider block at one row: 96, not 64', () => {
    expect(suggest(TABLE[1].d, 1, 4096)).toEqual({ x: 96, y: 1 });
    expect(status(TABLE[1].d, 1, 4096, 64, 1)['block-slots']).toBe('fail'); // 16 x 64 = 1024 of 1536
    expect(status(TABLE[1].d, 1, 4096, 96, 1)['block-slots']).toBe('pass');
  });
  it('a 64-wide device gets rows of 64', () => {
    expect(suggest(TABLE[2].d, 1, 4096)).toEqual({ x: 64, y: 1 });
    expect(suggest(TABLE[2].d, 64, 4096)).toEqual({ x: 64, y: 4 });
  });
  it('with nothing reported about an SM, one warp per row is enough', () => {
    expect(suggest(TABLE[4].d, 1, 4096)).toEqual({ x: 32, y: 1 });
    const c = checkLaunch(TABLE[4].d, 1, 4096, { x: 32, y: 1 });
    expect(c.skipped).toEqual(['divides', 'block-slots']);
    expect(c.ok).toBe(true);
  });
  it('with thread slots and no block slots, only divisibility is tested', () => {
    const c = checkLaunch(TABLE[5].d, 1, 4096, { x: 32, y: 1 });
    expect(c.skipped).toEqual(['block-slots']);
    expect(c.ok).toBe(true); // 1536 / 32 = 48 exactly
    expect(status(TABLE[5].d, 1, 4096, 64, 1).divides).toBe('pass');
    expect(status(TABLE[5].d, 1, 4096, 1024, 1).divides).toBe('fail'); // 512 of 1536 left over
  });
  it('no width, no suggestion', () => {
    expect(suggest({ ...BOOK_LAUNCH, width: null }, 1, 4096)).toBeNull();
  });
});

describe('each rule fails on its own', () => {
  it('16 x 16 at one row fails only "no taller than the rows"', () => {
    const c = checkLaunch(BOOK_LAUNCH, 1, 4096, { x: 16, y: 16 });
    expect(c.failed).toEqual(['rows']);
    expect(c.grid).toEqual({ x: 256, y: 1 });
    expect([c.launched, c.working, c.surplus]).toEqual([65536, 4096, 61440]);
    expect(c.rules.find((r) => r.key === 'rows')!.detail).toContain('15 of every 16 rows of threads own nothing');
  });
  it('16 x 16 at 64 rows passes everything on the book example', () => {
    const c = checkLaunch(BOOK_LAUNCH, 64, 4096, { x: 16, y: 16 });
    expect(c.ok).toBe(true);
    expect([c.launched, c.working, c.surplus]).toEqual([262144, 262144, 0]);
  });
  it('within maxThreadsPerBlock', () => {
    expect(checkLaunch(BOOK_LAUNCH, 64, 4096, { x: 64, y: 32 }).failed).toEqual(['per-block']); // 2048 threads
  });
  it('within maxThreadsDim', () => {
    expect(checkLaunch(TABLE[3].d, 64, 4096, { x: 256, y: 4 }).failed).toEqual(['dims']);
    expect(checkLaunch(TABLE[3].d, 64, 4096, { x: 128, y: 8 }).ok).toBe(true);
  });
  it('whole number of warps', () => {
    const c = checkLaunch(TABLE[4].d, 1, 4096, { x: 48, y: 1 });
    expect(c.failed).toEqual(['warps']);
    expect(c.rules.find((r) => r.key === 'warps')!.detail).toContain('16 padded lanes');
    expect(checkLaunch(TABLE[2].d, 1, 4096, { x: 32, y: 1 }).failed).toContain('warps'); // half a 64-wide warp
  });
  it('divides the thread slots: 768 on the book example leaves 512 empty (p. 86)', () => {
    const c = checkLaunch(BOOK_LAUNCH, 64, 4096, { x: 32, y: 24 });
    expect(c.failed).toEqual(['divides']);
    expect(c.rules.find((r) => r.key === 'divides')!.detail).toBe('768 threads per block leave 512 of 2,048 thread slots empty');
    expect(checkLaunch(TABLE[1].d, 64, 4096, { x: 32, y: 24 }).ok).toBe(true); // 1536 = 2 x 768
  });
  it('block slots do not run out first: 32 on the book example fills half (pp. 85 to 86)', () => {
    const c = checkLaunch(BOOK_LAUNCH, 1, 4096, { x: 32, y: 1 });
    expect(c.failed).toEqual(['block-slots']);
    expect(c.rules.find((r) => r.key === 'block-slots')!.detail).toBe('32 block slots x 32 threads fill 1,024 of 2,048 thread slots');
    expect(pct(occupancy(occupancyLimits(BOOK_LAUNCH), { threads: 32, regs: 0, shared: 0 }).byWarps)).toBe(50);
  });
  it('within maxGridSize', () => {
    expect(checkLaunch(TABLE[6].d, 1, 4096, { x: 64, y: 1 }).failed).toEqual(['grid']); // 64 blocks along x, 2 allowed
    expect(checkLaunch({ ...BOOK_LAUNCH, maxGridSize: [null, null, null] }, 1, 4096, { x: 64, y: 1 }).skipped).toEqual(['grid']);
  });
  it('rule names resolve', () => {
    expect(ruleName('rows')).toBe('no taller than the rows');
    expect(RULES).toHaveLength(7);
  });
});

describe('launched, working and surplus threads', () => {
  it('the grid is the smallest that covers the problem', () => {
    const c = checkLaunch(BOOK_LAUNCH, 3, 40, { x: 32, y: 2 });
    expect(c.grid).toEqual({ x: 2, y: 2 });
    expect([c.blocks, c.threadsPerBlock, c.launched, c.working, c.surplus]).toEqual([4, 64, 256, 120, 136]);
  });
  it('the suggested block at one row launches no idle rows', () => {
    const c = checkLaunch(BOOK_LAUNCH, 1, 4096, { x: 64, y: 1 });
    expect([c.grid.x, c.grid.y, c.launched, c.surplus]).toEqual([64, 1, 4096, 0]);
  });
  it('nonsense inputs are made whole numbers of at least 1', () => {
    const c = checkLaunch(BOOK_LAUNCH, 0, NaN, { x: 0.4, y: -3 });
    expect([c.rows, c.cols, c.block.x, c.block.y]).toEqual([1, 1, 1, 1]);
  });
});

describe('limits from a device report', () => {
  const cuda = {
    name: 'some CUDA device', warpSize: '32', maxThreadsPerBlock: '1024', 'maxThreadsDim.x': '1024', 'maxThreadsDim.y': '1024',
    'maxThreadsDim.z': '64', 'maxGridSize.x': '2147483647', 'maxGridSize.y': '65535', 'maxGridSize.z': '65535',
    maxThreadsPerMultiProcessor: '1536', maxBlocksPerMultiProcessor: '16',
  };
  it('reads a CUDA report', () => {
    expect(launchLimitsFromReport(cuda, 'cuda')).toEqual(synthetic(32, 1024, [1024, 1024, 64], 1536, 16, [2147483647, 65535, 65535]));
  });
  it('a HIP report: block slots marked as derived, a field of 0 not reported', () => {
    const l = launchLimitsFromReport({ ...cuda, warpSize: '64', maxThreadsPerMultiProcessor: '0', maxBlocksPerMultiProcessor: '32' }, 'hip')!;
    expect(l.blockSlotsDerived).toBe(true);
    expect(l.threadSlots).toBeNull();
    const c = checkLaunch(l, 1, 4096, { x: 64, y: 1 });
    expect(c.skipped).toEqual(['divides', 'block-slots']);
    expect(c.ok).toBe(true);
  });
  it('derived block slots can never fail the block-slot rule for whole waves', () => {
    const l = launchLimitsFromReport({ ...cuda, warpSize: '64', maxThreadsPerMultiProcessor: '2048', maxBlocksPerMultiProcessor: '32' }, 'hip')!;
    for (let t = 64; t <= 1024; t += 64) expect(status(l, 1, 4096, t, 1)['block-slots']).toBe('pass');
  });
  it('returns nothing for the emulators and for a Metal device', () => {
    expect(launchLimitsFromReport({ ...cuda, note: 'synthetic values describing the CPU emulator, not a GPU' }, 'emu')).toBeNull();
    expect(launchLimitsFromReport(cuda, 'race')).toBeNull();
    expect(launchLimitsFromReport({ name: 'a Metal device', 'maxThreadsPerThreadgroup.width': '1024', maxThreadgroupMemoryLength: '32768' }, 'metal')).toBeNull();
  });
});

// Section 4.8 shows code and reads report fields by name. These tests fail if the lesson's
// copies drift from the files, the notes or the report writers they stand for.
describe("section 4.8's listings and tables are tied to their sources", () => {
  const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
  const notes = read('notes/ch04.md');
  const facts = read('notes/today/ch04-facts.md');
  const lines = AS_PRINTED.split('\n');

  it("the erratum listing's loop is the book's loop as the notes record it (p. 88)", () => {
    for (const n of [1, 2, 3, 4, 5]) expect(notes, `line ${n}`).toContain(lines[n - 1].trim());
    expect(lines[3]).toBe('for(unsigned int i = 0; i < devCount; i++) {');
    expect(lines[lines.length - 1]).toBe('}');
  });
  it('its other lines read exactly the seven fields the notes list for p. 89, in the notes\' order', () => {
    const fields = lines.slice(5, 12).map((l) => /devProp\.(\w+)/.exec(l)![1]);
    expect(fields).toEqual(['maxThreadsPerBlock', 'multiProcessorCount', 'clockRate', 'maxThreadsDim', 'maxGridSize', 'regsPerBlock', 'warpSize']);
    for (const f of fields) expect(notes).toContain('`devProp.' + f);
  });
  it('the two answers of the hunt are the clockRate and regsPerBlock lines, as facts A18 says', () => {
    expect(lines[HUNT_ANSWER.compile - 1]).toContain('devProp.clockRate');
    expect(lines[HUNT_ANSWER.describe - 1]).toContain('devProp.regsPerBlock');
    expect(lines[HUNT_ANSWER.describe - 1]).toContain('registers in each SM'); // the book's description
    expect(facts).toContain('| `clockRate` | WRONG: removed in CUDA 13.0 |');
    expect(facts).toContain('| `regsPerBlock` | PARTLY | "32-bit registers available per block"');
  });
  it('the corrected loop the lesson loads has an int index, checks both calls and reads no clock', () => {
    const loop = read('labs/ch04/snippets/device_loop.cu');
    expect(loop).toContain('for (int i = 0; i < devCount; i++)');
    expect(loop).toContain('cudaGetDeviceCount(&devCount) != cudaSuccess');
    expect(loop).toContain('cudaGetDeviceProperties(&devProp, i) != cudaSuccess');
    expect(loop).not.toContain('clockRate');
    for (const f of ['maxThreadsPerMultiProcessor', 'maxBlocksPerMultiProcessor', 'regsPerMultiprocessor', 'sharedMemPerMultiprocessor'])
      expect(loop).toContain('devProp.' + f);
  });
  it("every report key the field table reads is one the device report writes", () => {
    const cudaReport = read('labs/common/src/backend_cudalike.cpp');
    const metalReport = read('labs/common/src/backend_metal.mm');
    const written = (src: string, k: string) => src.includes(`add("${k}"`) || src.includes(`emplace_back("${k}"`);
    for (const f of FIELDS)
      for (const [family, w] of [['cuda', f.cuda], ['hip', f.hip], ['metal', f.metal]] as const) {
        if (!w || typeof w === 'string') continue;
        for (const k of w.keys ?? []) expect(written(family === 'metal' ? metalReport : cudaReport, k), `${f.id} ${family} ${k}`).toBe(true);
      }
  });
  it("every lab key the field table reads is one lab 4.1's queried row writes", () => {
    const main = read('labs/ch04/lab1_launchconfig/main.cpp');
    for (const f of FIELDS)
      for (const w of [f.cuda, f.hip, f.metal]) {
        if (!w || typeof w === 'string') continue;
        for (const k of w.lab ?? []) expect(main.includes(`.n("${k}"`) || main.includes(`.s("${k}"`), `${f.id} ${k}`).toBe(true);
      }
  });
  it("the Metal excerpt is part 1 of lab 4.1's reference launch.mm, and only that", () => {
    const file = read('labs/ch04/lab1_launchconfig/solution/launch.mm');
    const part = metalQueryExcerpt(file);
    expect(part.startsWith('// ---------- part 1: ask')).toBe(true);
    expect(part).not.toContain('part 2');
    expect(part).toContain('threadExecutionWidth');
    expect(part).toContain('maxTotalThreadsPerThreadgroup');
  });
});

describe("Apple's recipe for a threadgroup (facts C6)", () => {
  it('w is the execution width and h is the pipeline limit divided by it', () => {
    expect(appleRecipe(16, 512)).toEqual({ x: 16, y: 32 }); // 16 is the width in Apple's own example
    expect(appleRecipe(24, 1000)).toEqual({ x: 24, y: 41 });
  });
  it('refuses values a pipeline cannot report', () => {
    expect(appleRecipe(0, 1024)).toBeNull();
    expect(appleRecipe(64, 32)).toBeNull();
  });
});
