import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  ACCESS_LINES,
  bankMap,
  coarsenedSteps,
  coarsenedWork,
  conflictDegree,
  demoValues,
  elementOf,
  gcd,
  koggeStoneSteps,
  koggeStoneWork,
  loadTransactions,
  loaderOf,
  ownerOf,
  sceneData,
  sharedAccesses,
  threePhases,
  tripCount,
  warpLoad,
  worstDegree,
  type AccessLine,
  type CoarseKernel,
} from './coarsescan';
import { functionSource } from './toolchain';
import { FIG_11_6, inclusiveScan, scanScanAdd } from './warpscan';

const repo = path.resolve(__dirname, '../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const FILES: Record<CoarseKernel, string> = {
  'fig11.12': 'labs/ch11/snippets/s116_coarsened_as_printed.cu',
  'fig11.13': 'labs/ch11/snippets/s117_regtiled_as_printed.cu',
};

describe('the three phases (Fig. 11.11)', () => {
  it('with 4 threads of 4 elements, the phases are the rows of Fig. 11.6', () => {
    const p = threePhases(FIG_11_6, 4);
    const s = scanScanAdd(FIG_11_6, 4);
    expect(p.runs.flat()).toEqual(s.stage1);
    expect(p.totals).toEqual([7, 7, 6, 11]);
    expect(p.scannedTotals).toEqual([7, 14, 20, 31]);
    expect(p.added).toEqual([0, 7, 14, 20]);
    expect(p.final).toEqual(s.final);
    expect(p.correct).toBe(true);
  });
  it('is the inclusive scan for every shape', () => {
    for (const P of [1, 2, 4, 8]) for (const F of [1, 2, 3, 4, 8]) {
      const v = demoValues(P * F);
      expect(threePhases(v, P).final).toEqual(inclusiveScan(v));
    }
  });
});

describe('work and steps (pp. 266, 269–270)', () => {
  it('question q-work: N = 1024, P = 256 does 3836 additions; Kogge-Stone on 1024 does 9217', () => {
    expect(coarsenedWork(1024, 256).total).toBe(3836);
    expect(coarsenedWork(1024, 256).parts.map((p) => p.value)).toEqual([768, 2048, 1020]);
    expect(koggeStoneWork(1024)).toBe(9217);
  });
  it('question q-steps: N = 1024, P = 32 takes 68 steps against 320 for Kogge-Stone on 32 units (pp. 266–267, 270)', () => {
    expect(coarsenedSteps(1024, 32).total).toBe(68);
    expect(koggeStoneSteps(1024, 32)).toBe(320);
  });
  it('P = N gives Kogge-Stone back: log2N + 1 steps, O(N log N) work', () => {
    for (const N of [16, 1024]) {
      expect(coarsenedSteps(N, N).total).toBe(Math.log2(N) + 1);
      expect(coarsenedWork(N, N).total).toBe(N * Math.log2(N) + N - 1);
      expect(koggeStoneSteps(N, N)).toBe(Math.log2(N));
    }
  });
  it('P much smaller than N: work close to 2N, steps close to 2N/P', () => {
    const N = 2 ** 20;
    // 2 − 1/P + (P·log2P − P)/N: within 1/P of 2
    expect(coarsenedWork(N, 32).total / N).toBeGreaterThan(2 - 1 / 32);
    expect(coarsenedWork(N, 32).total / N).toBeLessThan(2);
    expect(coarsenedSteps(N, 32).total / ((2 * N) / 32)).toBeLessThan(1.001);
  });
});

describe('the two index mappings (p. 269)', () => {
  it('question q-owner: COARSE_FACTOR 4, BLOCK_DIM 256, c = 2, thread 5 copies element 517, which thread 129 scans', () => {
    expect(elementOf('staged', 5, 2, 256, 4)).toBe(517);
    expect(ownerOf(517, 4)).toBe(129);
    expect(loaderOf('staged', 517, 256, 4)).toEqual({ thread: 5, iter: 2 });
    expect(loaderOf('direct', 517, 256, 4)).toEqual({ thread: 129, iter: 1 });
  });
  it('both mappings visit every element of the segment exactly once', () => {
    for (const mode of ['staged', 'direct'] as const) {
      const seen = new Set<number>();
      for (let c = 0; c < 4; c++) for (let t = 0; t < 32; t++) seen.add(elementOf(mode, t, c, 32, 4));
      expect(seen.size).toBe(128);
      for (let e = 0; e < 128; e++) {
        const { thread, iter } = loaderOf(mode, e, 32, 4);
        expect(elementOf(mode, thread, iter, 32, 4)).toBe(e);
      }
    }
  });
  it('question q-sectors: factor 4, direct loads touch 16 sectors of 32 bytes, a chunk touches 4', () => {
    const d = warpLoad('direct', 0, 4, 256, 32, 32);
    expect(d.elements.slice(0, 4)).toEqual([0, 4, 8, 12]);
    expect(d.access.transactions).toBe(16);
    expect(warpLoad('staged', 0, 4, 256, 32, 32).access.transactions).toBe(4);
    expect(warpLoad('staged', 0, 4, 256, 32, 32).runs).toBe(1);
    expect(d.runs).toBe(32);
  });
  it('F times as many sectors until every lane has a sector of its own', () => {
    expect([1, 2, 4, 8, 16].map((F) => warpLoad('direct', 0, F, 256, 32, 32).access.transactions)).toEqual([4, 8, 16, 32, 32]);
    expect([1, 2, 4, 8, 16].map((F) => warpLoad('staged', 1, F, 256, 32, 32).access.transactions)).toEqual([4, 4, 4, 4, 4]);
    expect(loadTransactions('direct', 4, 256, 32, 32)).toBe(64);
    expect(loadTransactions('staged', 4, 256, 32, 32)).toBe(16);
  });
});

// ---------------------------------------------------------------- the listings, line by line

/** Accesses of one statement, read from its text: what the model says per execution. */
function parseStatement(text: string) {
  const s = text.replace(/\s+/g, ' ').trim();
  const m = /^(.*?)\s(\+?=)\s(.*);$/.exec(s);
  if (!m) throw new Error('no assignment in: ' + s);
  const [, lhsRaw, op, rhs] = m;
  const lhs = lhsRaw.replace(/^float /, '');
  const count = (str: string, re: RegExp) => (str.match(re) ?? []).length;
  const shared = /\b(buffer_s|threadSums)\[/g;
  const lhsShared = count(lhs, shared);
  return {
    sharedRead: count(rhs, shared) + (op === '+=' ? lhsShared : 0),
    sharedWrite: lhsShared,
    regRead: count(rhs, /\bbuffer_r\[/g) + (op === '+=' ? count(lhs, /\bbuffer_r\[/g) : 0),
    regWrite: count(lhs, /\bbuffer_r\[/g),
    globalRead: count(rhs, /\binput\[/g),
    globalWrite: count(lhs, /\boutput\[/g),
  };
}

describe('every counted line is the book\'s line, and its counts are what the line does', () => {
  for (const kernel of ['fig11.12', 'fig11.13'] as CoarseKernel[]) {
    const src = read(FILES[kernel]);
    const fn = functionSource(src, 'scan_kernel')!;
    const book = fn.text.split('\n'); // book line k is book[k - 1]
    it(`${kernel}: the listing has the book's ${kernel === 'fig11.12' ? 35 : 37} lines`, () => {
      expect(book.length).toBe(kernel === 'fig11.12' ? 35 : 37);
      expect(book[0]).toBe('__global__ void scan_kernel(float* input, float* output, unsigned int N) {');
    });
    for (const l of ACCESS_LINES[kernel]) {
      it(`${kernel} line ${l.line}: ${l.code}`, () => {
        expect(book[l.line - 1].trim()).toBe(l.code);
        // the whole statement: this line and the next when it ends in '='
        const stmt = book[l.line - 1].trim().endsWith('=') ? book[l.line - 1] + ' ' + book[l.line] : book[l.line - 1];
        expect(parseStatement(stmt)).toEqual(l.per);
        // the loop around it decides the trip count; an enclosing if(threadIdx.x > 0) or the ternary, thread 0
        const indent = (s: string) => s.length - s.trimStart().length;
        let trip: AccessLine['trip'] = '1';
        let thread0 = !/\(threadIdx\.x>0\)\?/.test(book[l.line - 1]);
        let level = indent(book[l.line - 1]);
        for (let k = l.line - 2; k >= 1; k--) {
          if (book[k].trim() === '' || indent(book[k]) >= level) continue;
          level = indent(book[k]);
          if (/for\(unsigned int c = 0;/.test(book[k]) && trip === '1') trip = 'F';
          if (/for\(unsigned int c = 1;/.test(book[k]) && trip === '1') trip = 'F − 1';
          if (/if\(threadIdx\.x > 0\)/.test(book[k])) thread0 = false;
        }
        expect(trip).toBe(l.trip);
        expect(thread0).toBe(l.thread0);
      });
    }
    it(`${kernel}: no other line of the kernel touches buffer_s, threadSums or buffer_r`, () => {
      const counted = new Set(ACCESS_LINES[kernel].flatMap((l) => [l.line, l.line + 1]));
      const touching = book.flatMap((t, i) => (/\b(buffer_s|threadSums|buffer_r)\[/.test(t) && !/__shared__|float buffer_r\[/.test(t) ? [i + 1] : []));
      for (const n of touching) expect(counted.has(n), `line ${n}: ${book[n - 1]}`).toBe(true);
    });
  }

  it('question q-shared: Fig. 11.13 makes 4 accesses to buffer_s per element, whatever the factor', () => {
    for (const F of [1, 2, 4, 8, 16]) {
      expect(sharedAccesses('fig11.13', F).perElement).toBe(4);
      expect(sharedAccesses('fig11.13', F, 'first').perElement).toBe(4);
    }
    const a = sharedAccesses('fig11.13', 4);
    expect([a.bufferS.write, a.bufferS.read]).toEqual([8, 8]);
    expect(a.threadSums).toEqual({ read: 1, write: 1 });
    expect(a.global).toEqual({ read: 4, write: 4 });
  });
  it('Fig. 11.12 makes 7 − 2/F per element (6.5 at F = 4); thread 0 skips the add', () => {
    for (const F of [1, 2, 4, 8, 16]) expect(sharedAccesses('fig11.12', F).perElement).toBeCloseTo(7 - 2 / F, 12);
    expect(sharedAccesses('fig11.12', 4).perElement).toBe(6.5);
    expect(sharedAccesses('fig11.12', 4, 'first').bufferS.total).toBe(5 * 4 - 2);
    expect(sharedAccesses('fig11.12', 4).global).toEqual(sharedAccesses('fig11.13', 4).global);
  });
  it('trip counts', () => {
    expect([tripCount('F', 4), tripCount('F − 1', 4), tripCount('1', 4)]).toEqual([4, 3, 1]);
  });
});

// The counts above, read off a whole kernel: every statement that touches buffer_s, times
// the trip count of the loop around it. Run on the book's listings and on lab 11.2's
// reference solution, so the counter of section 11.7 describes the code the reader runs.
function bufferSPerThread(kernel: string, F: number, thread0 = false): number {
  const lines = kernel.split('\n');
  const indent = (s: string) => s.length - s.trimStart().length;
  let total = 0;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i];
    if (i > 0 && /=\s*$/.test(lines[i - 1])) continue; // the second half of a two-line statement
    const stmt = /=\s*$/.test(t) ? t + ' ' + lines[i + 1] : t;
    if (!/\bbuffer_s\[/.test(stmt) || /__shared__|extern|threadgroup float\*/.test(stmt) || !/;\s*(\/\/.*)?$/.test(stmt)) continue;
    const c = parseStatement(stmt.replace(/\/\/.*$/, ''));
    let trip = 1;
    let level = indent(t);
    let skip = false;
    for (let k = i - 1; k >= 0; k--) {
      if (lines[k].trim() === '' || indent(lines[k]) >= level) continue;
      level = indent(lines[k]);
      if (/for \(?(unsigned int|uint) c = 0;|for\(unsigned int c = 0;/.test(lines[k]) && trip === 1) trip = F;
      else if (/for \(?(unsigned int|uint) c = 1;|for\(unsigned int c = 1;/.test(lines[k]) && trip === 1) trip = F - 1;
      if (/if ?\((threadIdx\.x|t) > 0\)/.test(lines[k]) && thread0) skip = true;
    }
    if (!skip) total += (c.sharedRead + c.sharedWrite) * trip;
  }
  return total;
}

describe('the counter agrees with whole kernels, the book\'s and lab 11.2\'s', () => {
  for (const F of [1, 2, 4, 8]) {
    it(`factor ${F}: the printed Figs. 11.12 and 11.13 give the model's per-thread buffer_s counts`, () => {
      for (const k of ['fig11.12', 'fig11.13'] as CoarseKernel[]) {
        const text = functionSource(read(FILES[k]), 'scan_kernel')!.text;
        expect(bufferSPerThread(text, F), k).toBe(sharedAccesses(k, F).bufferS.total);
        expect(bufferSPerThread(text, F, true), k + ' thread 0').toBe(sharedAccesses(k, F, 'first').bufferS.total);
      }
    });
  }
  const LAB = 'labs/ch11/lab2_coarsened/solution/';
  it.skipIf(!fs.existsSync(path.join(repo, LAB + 'kernel.cu')))('lab 11.2 (CUDA and Metal) makes the same accesses as the figures', () => {
    const cu = read(LAB + 'kernel.cu');
    const metal = read(LAB + 'kernel.metal');
    const pairs: [string, string, CoarseKernel][] = [
      [cu, 'CoarsenedScanKernel', 'fig11.12'],
      [cu, 'RegisterTiledScanKernel', 'fig11.13'],
      [metal, 'coarsened_scan', 'fig11.12'],
      [metal, 'register_tiled_scan', 'fig11.13'],
    ];
    for (const [src, name, fig] of pairs) {
      const text = functionSource(src, name)!.text;
      for (const F of [2, 4, 8]) expect(bufferSPerThread(text, F), `${name} at ${F}`).toBe(sharedAccesses(fig, F).bufferS.total);
    }
  });
});

describe('banks (exercise 4, p. 286; infra-facts A10, B10)', () => {
  it('question q-bank: F = 8 on 32 banks puts 8 lanes on a bank; padding brings it to 1', () => {
    const m = bankMap(8, 32, false);
    expect(m.degree).toBe(8);
    expect(m.lanes[0].bank).toBe(m.lanes[4].bank);
    expect(bankMap(8, 32, true).degree).toBe(1);
  });
  it('question q-odd: without padding the degree is gcd(F, 32), so odd factors are conflict-free', () => {
    for (let F = 1; F <= 16; F++) {
      expect(bankMap(F, 32, false).degree).toBe(gcd(F, 32));
      expect(conflictDegree(F, 32)).toBe(gcd(F, 32));
      if (F % 2 === 1) expect(bankMap(F, 32, false).degree).toBe(1);
    }
  });
  it('padding removes the conflict for every power-of-two factor, on 32 and on 64 banks, in every iteration', () => {
    for (const banks of [32, 64]) for (const F of [1, 2, 4, 8, 16]) for (let c = 0; c < F; c++) expect(bankMap(F, banks, true, { c }).degree).toBe(1);
  });
  it('padding is not free: on 32 banks it gives every factor that is not a power of two a 2-way conflict somewhere', () => {
    const worst = (pad: boolean) => Array.from({ length: 16 }, (_, i) => worstDegree(i + 1, 32, pad));
    expect(worst(false)).toEqual([1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16]);
    expect(worst(true)).toEqual([1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1]);
  });
  it('32 lanes on 64 banks: max(1, 32·gcd(F, 64)/64), half the conflict of 32 banks at F = 8', () => {
    expect(bankMap(8, 64, false).degree).toBe(4);
    for (let F = 1; F <= 16; F++) for (const banks of [32, 64]) expect(bankMap(F, banks, false).degree).toBe(conflictDegree(F, banks));
  });
  it('the load side, buffer_s[c·BLOCK_DIM + t], is conflict-free padded or not', () => {
    for (const pad of [false, true]) for (let c = 0; c < 4; c++) expect(bankMap(4, 32, pad, { side: 'load', c, blockDim: 256 }).degree).toBe(1);
  });
});

// ---------------------------------------------------------------- the scene and its narration

const script = (n: number) => read(`content/ch11/narration/coarse-phases-${n}.txt`).trim();
const words = (s: string) => s.replace(/\[[^\]]*\]/g, '').split(/\s+/).filter(Boolean).length;

describe('the "Three phases of a coarsened scan" narration agrees with the model', () => {
  const d = sceneData();
  it('every script is 35 to 60 words and reads no symbols aloud', () => {
    for (let n = 1; n <= 6; n++) {
      expect(words(script(n)), `part ${n}`).toBeGreaterThanOrEqual(35);
      expect(words(script(n)), `part ${n}`).toBeLessThanOrEqual(60);
      expect(script(n), `part ${n}`).not.toMatch(/[=×*+/<>%·]/);
    }
  });
  it('part 1: 4 threads, 16 elements, 4 each', () => {
    expect([d.P, d.N, d.F]).toEqual([4, 16, 4]);
    expect(script(1)).toContain('A block of 4 threads owns 16 elements, 4 for each thread.');
  });
  it('part 2: 4 load instructions, each one run; loading runs directly splits an instruction into 4', () => {
    expect(d.staged.map((s) => s.runs)).toEqual([1, 1, 1, 1]);
    expect(d.direct.map((s) => s.runs)).toEqual([4, 4, 4, 4]);
    expect(d.staged[1].elements).toEqual([4, 5, 6, 7]);
    expect(script(2)).toContain('4 load instructions');
    expect(script(2)).toContain('4 separate pieces');
  });
  it('part 3: thread 1 scans 0, 4, 1, 2 into 0, 4, 5, 7', () => {
    expect(d.values.slice(4, 8)).toEqual([0, 4, 1, 2]);
    expect(d.phases.runs[1]).toEqual([0, 4, 5, 7]);
    expect(script(3)).toContain('0, 4, 5 and 7');
  });
  it('part 4: the run totals 7, 7, 6 and 11 scan to 7, 14, 20 and 31', () => {
    expect(d.phases.totals).toEqual([7, 7, 6, 11]);
    expect(d.phases.scannedTotals).toEqual([7, 14, 20, 31]);
    expect(script(4)).toContain('7, 7, 6 and 11');
    expect(script(4)).toContain('7, 14, 20 and 31');
  });
  it('part 5: thread 2 adds 14; the last element becomes 31', () => {
    expect(d.phases.added).toEqual([0, 7, 14, 20]);
    expect(d.phases.final[15]).toBe(31);
    expect(script(5)).toContain('Thread 2 adds 14');
  });
  it('part 6: 4 shared-memory accesses per element, 32 additions for 16 elements', () => {
    expect(d.perElement).toBe(4);
    expect(d.work).toBe(32);
    expect(script(6)).toContain('32 additions for 16 elements');
    expect(koggeStoneWork(16)).toBe(49);
    expect(script(6)).toContain('Kogge-Stone on 16 elements does 49.');
  });
});
