import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { functionSource } from './toolchain';
import {
  FIG_11_6,
  HALF_CASES,
  PORT_ISSUES,
  barriersPerBlockScan,
  blockScanFig119,
  groupedScan,
  halfCase,
  inclusiveScan,
  issueLines,
  laneValues,
  maskVerdict,
  oneWarpOfSums,
  portIssues,
  scanScanAdd,
  scorePort,
  shuffleUp,
  warpScanTrace,
} from './warpscan';

const repo = path.resolve(__dirname, '../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const PRINTED = 'labs/ch11/snippets/s114_blockscan_as_printed.cu';
const printed = read(PRINTED);
const lines = printed.split('\n');
const range = (n: number, from = 0) => Array.from({ length: n }, (_, i) => from + i);
const laneValue0 = (w: number) => laneValues(1, 32 * w)[0];

describe('scan, scan, add (Fig. 11.6, pp. 261–262)', () => {
  const r = scanScanAdd(FIG_11_6, 4);
  it('reproduces every row of Fig. 11.6', () => {
    expect(r.stage1).toEqual([2, 3, 6, 7, 0, 4, 5, 7, 0, 3, 4, 6, 5, 8, 9, 11]);
    expect(r.totals).toEqual([7, 7, 6, 11]);
    expect(r.stage2).toEqual([7, 14, 20, 31]);
    expect(r.added).toEqual([0, 7, 14, 20]);
    expect(r.final).toEqual([2, 3, 6, 7, 7, 11, 12, 14, 14, 17, 18, 20, 25, 28, 29, 31]);
  });
  it('the scanned totals are the final values at the segment ends', () => {
    expect([3, 7, 11, 15].map((i) => r.final[i])).toEqual(r.stage2);
  });
  it('gives the inclusive scan for every segment size', () => {
    for (const s of [1, 2, 4, 8, 16]) expect(scanScanAdd(FIG_11_6, s).final).toEqual(inclusiveScan(FIG_11_6));
  });
  it('question q-sss: position 9 ends at 15', () => {
    const q = scanScanAdd([1, 2, 0, 3, 2, 2, 1, 0, 4, 0, 1, 1, 0, 3, 2, 2], 4);
    expect(q.stage1.slice(8, 12)).toEqual([4, 4, 5, 6]);
    expect(q.stage2).toEqual([6, 11, 17, 24]);
    expect(q.final[9]).toBe(15);
    expect(q.final[9]).toBe(q.stage1[9] + q.stage2[1]);
  });
});

describe('shuffle up and Fig. 11.8', () => {
  const v32 = laneValues(32);
  it('lanes below the delta get their own value back (infra A3, B3, C2)', () => {
    const s = shuffleUp(range(32), 4);
    for (let l = 0; l < 4; l++) expect(s[l]).toEqual({ lane: l, src: null, value: l });
    for (let l = 4; l < 32; l++) expect(s[l]).toEqual({ lane: l, src: l - 4, value: l - 4 });
  });
  it('with a width, the rule applies per group', () => {
    const s = shuffleUp(range(32), 2, 8);
    expect(s[9]).toEqual({ lane: 9, src: null, value: 9 });
    expect(s[10]).toEqual({ lane: 10, src: 8, value: 8 });
  });
  it('NVIDIA\'s width-8 example (ch11-facts A3) is a scan of each group of 8', () => {
    const t = warpScanTrace(v32, { loopBound: 8, laneWidth: 8, shuffleWidth: 8 });
    expect(t.steps.map((s) => s.stride)).toEqual([1, 2, 4]);
    expect(t.final).toEqual(groupedScan(v32, 8));
    expect(t.wrong).toEqual([]);
  });
  it('as printed: five strides, the inclusive scan of the warp, 129 additions', () => {
    const t = warpScanTrace(v32);
    expect(t.steps.map((s) => s.stride)).toEqual([1, 2, 4, 8, 16]);
    expect(t.final).toEqual(inclusiveScan(v32));
    expect(t.additions).toBe(31 + 30 + 28 + 24 + 16);
    expect(t.doubled).toEqual([]);
  });
  it('question q-below-stride: with no test, lane 0 ends at 32 times its input, and only the top lane is right', () => {
    const t = warpScanTrace(v32, { test: 'none' });
    expect(t.final[0]).toBe(32 * v32[0]);
    expect(t.doubled).toEqual(range(16));
    expect(t.wrong).toEqual(range(31));
    expect(warpScanTrace(new Array(8).fill(1), { test: 'none' }).final).toEqual(new Array(8).fill(8));
    expect(warpScanTrace(laneValues(8), { test: 'none' }).final[0]).toBe(8 * laneValues(8)[0]);
  });
  it('question q-laneidx: threadIdx.x as the test is right in warp 0; in every later warp lanes 0 to 15 double themselves and all but lane 31 end wrong', () => {
    expect(warpScanTrace(v32, { test: 'threadIdx', warp: 0 }).wrong).toEqual([]);
    for (let w = 1; w < 8; w++) {
      const t = warpScanTrace(laneValues(32, 32 * w), { test: 'threadIdx', warp: w });
      expect(t.doubled).toEqual(range(16));
      expect(t.wrong).toEqual(range(31));
      expect(t.final[0]).toBe(32 * laneValue0(w));
    }
  });
  it('question q-half-wave: on 64 lanes, bound 32 with lanes counted in 64s leaves lanes 32 to 63 wrong (ch11-facts B1)', () => {
    const v = laneValues(64);
    expect(halfCase(v, 'both32').wrong).toEqual([]);
    expect(halfCase(v, 'both32').final).toEqual(groupedScan(v, 32));
    expect(halfCase(v, 'mixed').wrong).toEqual(range(32, 32));
    expect(halfCase(v, 'both64').wrong).toEqual([]);
    expect(halfCase(v, 'both64').final).toEqual(inclusiveScan(v));
    expect(HALF_CASES.map((c) => halfCase(v, c.key).steps.length)).toEqual([5, 5, 6]);
  });
});
describe('Fig. 11.9, the block scan', () => {
  it('is right at 1024 threads for warps of 32 and wavefronts of 64 (ch11-facts B1)', () => {
    for (const w of [32, 64]) {
      const r = blockScanFig119(laneValues(1024), w);
      expect(r.fits).toBe(true);
      expect(r.wrong).toEqual([]);
      expect(r.barriers).toBe(2);
    }
    expect(blockScanFig119(laneValues(1024), 32).numWarps).toBe(32);
    expect(blockScanFig119(laneValues(1024), 64).numWarps).toBe(16);
  });
  it('question q-metal-width: 16-wide SIMD-groups at 1024 threads give 64 totals for 16 lanes, and go wrong', () => {
    const r = blockScanFig119(laneValues(1024), 16);
    expect(r.numWarps).toBe(64);
    expect(r.fits).toBe(false);
    expect(oneWarpOfSums(1024, 16)).toBe(false);
    expect(oneWarpOfSums(1024, 32)).toBe(true);
    expect(r.wrong[0]).toBe(17 * 16);
  });
  it('question q-barriers: 20, 10 and 2 barriers per block scan at 1024 threads', () => {
    expect(barriersPerBlockScan('fig11.3', 1024)).toBe(20);
    expect(barriersPerBlockScan('fig11.5', 1024)).toBe(10);
    expect(barriersPerBlockScan('fig11.9', 1024)).toBe(2);
  });
});

describe('masks per backend (question q-hip-mask)', () => {
  it('the literal compiles for CUDA, not for HIP; a typed mask does; Metal has none', () => {
    expect(maskVerdict('cuda', 'literal').compiles).toBe(true);
    expect(maskVerdict('hip', 'literal').compiles).toBe(false);
    expect(maskVerdict('hip', 'typed').compiles).toBe(true);
    expect(maskVerdict('metal', 'literal').maskType).toBe('none');
  });
});

describe('the "as printed" file and the Port it hunt', () => {
  it('holds the book\'s lines', () => {
    expect(printed).toContain('for(unsigned int stride = 1; stride < WARP_SIZE; stride *= 2) {');
    expect(printed).toContain('float leftVal = __shfl_up_sync(0xffffffff, val, stride);');
    expect(printed).toContain('if(laneIdx() >= stride) {');
    expect(printed).toContain('float warpSum = (threadIdx.x<NUM_WARPS)?warpSums_s[threadIdx.x]:0.0f;');
    expect(printed).toContain('val += warpSums_s[warpIdx() - 1];');
    expect(printed).toContain('float val = (i < N)?input[i]:0.0f;');
  });
  it('the three figures have the book\'s line counts (9, 24 and 8)', () => {
    expect(functionSource(printed, 'warpScan')!.lastLine - functionSource(printed, 'warpScan')!.firstLine + 1).toBe(9);
    expect(functionSource(printed, 'blockScan')!.lastLine - functionSource(printed, 'blockScan')!.firstLine + 1).toBe(24);
    expect(functionSource(printed, 'scan_kernel')!.lastLine - functionSource(printed, 'scan_kernel')!.firstLine + 1).toBe(8);
  });
  it('every issue points at the line that holds its text', () => {
    for (const p of PORT_ISSUES) expect(lines[p.line - 1], p.id).toContain(p.text);
  });
  it('"Fig. 11.x, line nn" in an issue is that line of the figure', () => {
    const fig = { '11.8': 'warpScan', '11.9': 'blockScan', '11.10': 'scan_kernel' } as Record<string, string>;
    for (const p of PORT_ISSUES) {
      const m = /^Fig\. (11\.\d+), lines? (\d+)/.exec(p.where);
      if (!m) continue;
      expect(functionSource(printed, fig[m[1]])!.firstLine + Number(m[2]) - 1, p.id).toBe(p.line);
    }
  });
  it('the alternate lines of an issue are inside the same construct', () => {
    expect(lines[13]).toContain('laneIdx()');
    for (const l of [41, 42, 43, 44]) expect(lines[l - 1]).toMatch(/NUM_WARPS\)\?warpSums_s|warpScan\(warpSum\)|threadIdx\.x < NUM_WARPS|warpSums_s\[threadIdx\.x\] = warpSum/);
  });
  it('lists two issues for HIP on gfx10, three on gfx9 and four on Metal', () => {
    expect(portIssues('hip32').map((p) => p.id)).toEqual(['mask', 'shared-in-helper']);
    expect(portIssues('hip64').map((p) => p.id)).toEqual(['warp-size', 'mask', 'shared-in-helper']);
    expect(portIssues('metal').map((p) => p.id)).toEqual(['warp-size', 'helpers', 'mask', 'shared-in-helper', 'one-warp-of-sums']);
    expect(PORT_ISSUES.flatMap(issueLines).length).toBe(new Set(PORT_ISSUES.flatMap(issueLines)).size);
  });
  it('scores clicks by issue', () => {
    expect(scorePort('hip32', [20, 34])).toEqual({ found: ['mask', 'shared-in-helper'], missed: [], extra: [] });
    expect(scorePort('metal', [14, 42, 59])).toEqual({ found: ['helpers', 'one-warp-of-sums'], missed: ['warp-size', 'mask', 'shared-in-helper'], extra: [59] });
  });
  it('mentions the literal 32 only where the book assumes it, with a comment saying so', () => {
    const code = (s: string) => s.replace(/\/\/.*$/, '');
    expect(lines.filter((l) => /\b32\b/.test(code(l)))).toEqual(['#define WARP_SIZE 32  // the book\'s assumption: every warp has 32 lanes']);
  });
});

// ---------------------------------------------------------------- on the emulator
// Builds the three "as printed" files and labs/ch11/snippets/check/s114_check.cu for the
// course's CPU emulator (labs/tools/cu2emu.py, pmpp/emu/cuda_emu.h, AddressSanitizer on), as
// the lab runner builds the emu backend, and compares what it prints with this model. The
// three files each define scan_kernel, so they are renamed on the way in. Skipped when no
// C++ compiler or python3 is found.
const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
const cxx = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
const python = ['python3'].find((c) => has(c, ['--version']));
const INC = path.join(repo, 'labs/common/include');

function buildCheck(blockDim: number, factor: number): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch11-s114-'));
  const rename = (file: string, to: string) => read(file).replace('void scan_kernel(', `void ${to}(`);
  const all = [
    rename(PRINTED, 'scan_kernel_fig_11_10'),
    rename('labs/ch11/snippets/s116_coarsened_as_printed.cu', 'scan_kernel_fig_11_12'),
    rename('labs/ch11/snippets/s117_regtiled_as_printed.cu', 'scan_kernel_fig_11_13'),
    read('labs/ch11/snippets/check/s114_check.cu'),
  ].join('\n');
  const cu = path.join(dir, 'all.cu');
  const cpp = path.join(dir, 'all.cpp');
  fs.writeFileSync(cu, all);
  const t = spawnSync(python!, [path.join(repo, 'labs/tools/cu2emu.py'), cu, cpp], { encoding: 'utf8' });
  if (t.status !== 0) throw new Error('cu2emu failed:\n' + t.stdout + t.stderr);
  const exe = path.join(dir, 'check');
  const c = spawnSync(cxx!, ['-std=c++17', '-O1', '-g', '-fsanitize=address', '-fno-omit-frame-pointer', '-pthread', '-DPMPP_BACKEND_EMU',
    '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h',
    `-DBLOCK_DIM=${blockDim}`, `-DCOARSE_FACTOR=${factor}`, cpp, '-o', exe], { encoding: 'utf8' });
  if (c.status !== 0) throw new Error('compile failed:\n' + c.stderr);
  return exe;
}

describe.skipIf(!cxx || !python)('the printed listings on the course emulator (labs/ch11/snippets/check/s114_check.cu)', () => {
  for (const [B, F] of [[256, 4], [1024, 2]]) {
    it(`BLOCK_DIM ${B}, COARSE_FACTOR ${F}: Figs. 11.10, 11.12 and 11.13 scan right in three thread orders; the lane results are the model's`, () => {
      const r = spawnSync(buildCheck(B, F), [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 120000 });
      expect(r.status, r.stderr).toBe(0);
      expect(r.stdout).toContain('DONE');
      const out = r.stdout.split('\n');
      for (const fig of ['fig11.10', 'fig11.12', 'fig11.13']) {
        const rows = out.filter((l) => l.startsWith(fig + ' '));
        expect(rows.length, fig).toBe(3);
        for (const l of rows) expect(l, fig).toMatch(/: 0 of \d+$/);
      }
      const lanes = (variant: string) => out.filter((l) => l.startsWith(`lanes ${variant} `)).map((l) => l.split(':')[1].trim().split(' ').map(Number));
      const model = (test: 'laneIdx' | 'threadIdx' | 'none') =>
        [0, 1].flatMap((w) => warpScanTrace(laneValues(32, 32 * w), { test, warp: w }).final);
      for (const [variant, test] of [['printed', 'laneIdx'], ['threadIdx', 'threadIdx'], ['none', 'none']] as const) {
        const got = lanes(variant);
        expect(got.length, variant).toBe(3);
        for (const g of got) expect(g, variant).toEqual(model(test));
      }
    }, 180000);
  }
});
