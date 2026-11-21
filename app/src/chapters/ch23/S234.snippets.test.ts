import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { DEFAULTS, edges, program, simulate } from '../../sim/haloloop';
import { lengthLimit, longestRight, shortestRight } from './questionText';
import { WINDOW_NCCL, WINDOW_STREAMS, questionsS234 } from './S234';

// Page 23.3–23.5 (id 23.4) shows Fig. 23.14 (corrected) with its setup, and lines 27-44 of Fig. 23.20.
// They are copied into S234.tsx from labs/ch23/snippets/overlap_loop.cu and nccl_exchange.cu
// (the marked regions, dedented). This test fails if a copy drifts, checks that every line the
// model (src/sim/haloloop.ts) and the puzzle cite is the call they say it is, and runs the
// CUDA front-end check on both files, as ./verify.sh does, plus once on line 19 as the page
// prints it, which must fail. The clang part is skipped when no clang++ is found.
const repo = path.resolve(__dirname, '../../../..');
const SNIP = path.join(repo, 'labs/ch23/snippets');
const lesson = (constant: string) => {
  const src = fs.readFileSync(path.join(__dirname, 'S234.tsx'), 'utf8');
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in S234.tsx`);
  return m[1].trim();
};
const region = (file: string, name: string) => {
  const text = fs.readFileSync(path.join(SNIP, file), 'utf8');
  const m = new RegExp(`// \\[${name}\\]\\n([\\s\\S]*?)\\n[ \\t]*// \\[/${name}\\]`).exec(text);
  if (!m) throw new Error(`no region ${name} in ${file}`);
  const lines = m[1].split('\n');
  const ind = Math.min(...lines.filter((l) => l.trim()).map((l) => (/^ */.exec(l) as RegExpExecArray)[0].length));
  return lines.map((l) => l.slice(ind)).join('\n').trim();
};

const LOOP = region('overlap_loop.cu', 'loop').split('\n'); // LOOP[n - 1] is the book's line n
const NCCL = region('nccl_exchange.cu', 'exchange').split('\n'); // NCCL[n - 27] is line n
const line23_14 = (n: number) => LOOP[n - 1];
const line23_20 = (n: number) => (n >= 27 ? NCCL[n - 27] : LOOP[n - 1]); // lines 01-26 are Fig. 23.14's
const first = (spec: string) => Number(/\d+/.exec(spec)![0]);
const cited = (spec: string) =>
  spec.split(',').flatMap((p) => {
    const m = /(\d+)(?:–(\d+))?/.exec(p.trim());
    if (!m) return [];
    const a = Number(m[1]);
    const b = m[2] ? Number(m[2]) : a;
    return Array.from({ length: b - a + 1 }, (_, k) => a + k);
  });

describe("the lesson's copies match the checked files", () => {
  it('Fig. 23.14, corrected', () => expect(lesson('LOOP')).toBe(LOOP.join('\n')));
  it('its setup', () => expect(lesson('SETUP')).toBe(region('overlap_loop.cu', 'setup')));
  it('Fig. 23.20, lines 27-44', () => expect(lesson('NCCL_EXCHANGE')).toBe(NCCL.join('\n')));
});

describe("the book's line numbers are kept", () => {
  it('Fig. 23.14 has 43 lines, and the comment lines are where the book has its comments', () => {
    expect(LOOP).toHaveLength(43);
    for (const n of [3, 9, 18, 24, 36]) expect(line23_14(n).trim().startsWith('//')).toBe(true);
    for (const n of [2, 8, 17, 23, 35, 40, 42]) expect(line23_14(n).trim()).toBe('');
  });
  it('line 19 has the closing parenthesis the page gets wrong', () => {
    expect(line23_14(19).trim()).toBe('cudaStreamWaitEvent(internalStream, computeTop, 0);');
  });
  it('Fig. 23.20 lines 27-44: groups at 27-30 and 32-35, records at 31 and 36, the sync at 39, waits at 43-44', () => {
    expect(NCCL).toHaveLength(18);
    expect(line23_20(27).trim()).toBe('ncclGroupStart();');
    expect(line23_20(30).trim()).toBe('ncclGroupEnd();');
    expect(line23_20(31)).toContain('cudaEventRecord(exchangeTop, topStream)');
    expect(line23_20(36)).toContain('cudaEventRecord(exchangeBottom, bottomStream)');
    expect(line23_20(39)).toContain('cudaStreamSynchronize(internalStream)');
    expect(line23_20(43)).toContain('cudaStreamWaitEvent(internalStream, exchangeTop, 0)');
    expect(line23_20(44)).toContain('cudaStreamWaitEvent(internalStream, exchangeBottom, 0)');
  });
});

// What each call of the model's host programs must find on the line it cites.
const TOKEN: Record<string, string> = {
  cudaMemsetAsync: 'cudaMemsetAsync(l2norm_d',
  'cudaEventRecord(resetL2)': 'cudaEventRecord(resetL2, internalStream)',
  'cudaStreamWaitEvent(topStream, resetL2)': 'cudaStreamWaitEvent(topStream, resetL2, 0)',
  'cudaStreamWaitEvent(bottomStream, resetL2)': 'cudaStreamWaitEvent(bottomStream, resetL2, 0)',
  'launch top kernel': 'launch_jacobi_kernel(output, input, l2norm_d, 3, nx, topStream)',
  'cudaEventRecord(computeTop)': 'cudaEventRecord(computeTop, topStream)',
  'launch bottom kernel': 'launch_jacobi_kernel(output + (ny - 3)*nx',
  'cudaEventRecord(computeBottom)': 'cudaEventRecord(computeBottom, bottomStream)',
  'launch internal kernel': 'launch_jacobi_kernel(output + nx, input + nx, l2norm_d, ny - 2',
  'cudaStreamWaitEvent(internalStream, computeTop)': 'cudaStreamWaitEvent(internalStream, computeTop, 0)',
  'cudaStreamWaitEvent(internalStream, computeBottom)': 'cudaStreamWaitEvent(internalStream, computeBottom, 0)',
  cudaMemcpyAsync: 'cudaMemcpyAsync(l2norm_h',
  'cudaStreamSynchronize(topStream)': 'cudaStreamSynchronize(topStream)',
  'MPI_Sendrecv (top)': 'MPI_Sendrecv(output + nx',
  'cudaStreamSynchronize(bottomStream)': 'cudaStreamSynchronize(bottomStream)',
  'MPI_Sendrecv (bottom)': 'MPI_Sendrecv(output + (ny - 2)',
  'cudaStreamSynchronize(internalStream)': 'cudaStreamSynchronize(internalStream)',
  MPI_Allreduce: 'MPI_Allreduce(l2norm_h',
  'ncclGroupStart, ncclSend, ncclRecv, ncclGroupEnd (top)': 'ncclGroupStart()',
  'cudaEventRecord(exchangeTop)': 'cudaEventRecord(exchangeTop, topStream)',
  'ncclGroupStart, ncclSend, ncclRecv, ncclGroupEnd (bottom)': 'ncclGroupStart()',
  'cudaEventRecord(exchangeBottom)': 'cudaEventRecord(exchangeBottom, bottomStream)',
  'cudaStreamWaitEvent(internalStream, exchangeTop)': 'cudaStreamWaitEvent(internalStream, exchangeTop, 0)',
  'cudaStreamWaitEvent(internalStream, exchangeBottom)': 'cudaStreamWaitEvent(internalStream, exchangeBottom, 0)',
};

describe('the lines the model and the puzzle cite are the calls they name', () => {
  it("Fig. 23.14: every call of the model's host program, in both launch orders", () => {
    for (const issue of ['boundary-first', 'internal-first'] as const)
      for (const c of program('streams', issue)) {
        expect(TOKEN[c.label], c.label).toBeDefined();
        expect(line23_14(first(c.line)), `line ${c.line}`).toContain(TOKEN[c.label]);
      }
  });
  it("Fig. 23.20: every call of the model's host program", () => {
    for (const c of program('nccl')) expect(line23_20(first(c.line)), `line ${c.line}`).toContain(TOKEN[c.label]);
  });
  it('a group call spans four lines, all in the same stream', () => {
    for (const n of [28, 29]) expect(line23_20(n)).toContain('topStream');
    for (const n of [33, 34]) expect(line23_20(n)).toContain('bottomStream');
  });
  it('every line the puzzle cites for Figs. 23.14 and 23.20 holds code', () => {
    for (const e of edges('streams')) for (const n of cited(e.lines)) expect(line23_14(n).trim(), `${e.id}: line ${n}`).toMatch(/^[a-zA-Z]/);
    for (const e of edges('nccl')) for (const n of cited(e.lines)) expect(line23_20(n).trim(), `${e.id}: line ${n}`).toMatch(/^[a-zA-Z]|^\w/);
  });
});

describe('the setup', () => {
  const setup = region('overlap_loop.cu', 'setup');
  it('creates the internal stream at the least priority and the boundary streams at the greatest (A1)', () => {
    expect(setup).toContain('cudaDeviceGetStreamPriorityRange(&low, &high)');
    expect(setup).toContain('cudaStreamCreateWithPriority(&internalStream, cudaStreamDefault, low)');
    expect(setup).toContain('cudaStreamCreateWithPriority(&topStream, cudaStreamDefault, high)');
    expect(setup).toContain('cudaStreamCreateWithPriority(&bottomStream, cudaStreamDefault, high)');
  });
  it('creates all three events without timing, and pins l2norm_h', () => {
    for (const ev of ['resetL2', 'computeTop', 'computeBottom'])
      expect(setup).toContain(`cudaEventCreateWithFlags(&${ev}, cudaEventDisableTiming)`);
    expect(setup).toContain('cudaMallocHost(&l2norm_h, sizeof(float))');
  });
  it("the NCCL file's setup comment sets the device before ncclCommInitRank (A12)", () => {
    const head = fs.readFileSync(path.join(SNIP, 'nccl_exchange.cu'), 'utf8').split('[exchange]')[0];
    expect(head.indexOf('cudaSetDevice')).toBeGreaterThan(0);
    expect(head.indexOf('cudaSetDevice')).toBeLessThan(head.indexOf('ncclCommInitRank'));
  });
});

const clang = process.env.PMPP_CLANG || ['/usr/bin/clang++', '/opt/homebrew/opt/llvm/bin/clang++', '/usr/local/bin/clang++'].find((p) => fs.existsSync(p));
const inc = path.join(repo, 'labs/common/include');
const frontEnd = (file: string) =>
  spawnSync(clang as string, ['-x', 'cuda', '--cuda-host-only', '-nocudainc', '-nocudalib', '-fsyntax-only', '-std=c++17',
    '-I', inc, '-I', path.join(inc, 'pmpp', 'emu', 'shim'), '-include', 'pmpp/check/cuda_stub.h', '-DTILE_WIDTH=16', file], { encoding: 'utf8' });

describe.skipIf(!clang)('the CUDA front-end check (clang, the course stub headers)', () => {
  it('passes on both snippets', () => {
    for (const f of ['overlap_loop.cu', 'nccl_exchange.cu']) {
      const r = frontEnd(path.join(SNIP, f));
      expect(r.status, `${f}: ${r.stderr}`).toBe(0);
    }
  });
  it("fails on line 19 as the page prints it, with the extra ')'", () => {
    const src = fs.readFileSync(path.join(SNIP, 'overlap_loop.cu'), 'utf8');
    const bad = src.replace('cudaStreamWaitEvent(internalStream, computeTop, 0);', 'cudaStreamWaitEvent(internalStream, computeTop, 0));');
    expect(bad).not.toBe(src);
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pmpp-ch23-'));
    fs.mkdirSync(path.join(dir, 'check'));
    fs.copyFileSync(path.join(SNIP, 'check', 'mpi_nccl_decls.h'), path.join(dir, 'check', 'mpi_nccl_decls.h'));
    fs.writeFileSync(path.join(dir, 'as_printed.cu'), bad);
    const r = frontEnd(path.join(dir, 'as_printed.cu'));
    fs.rmSync(dir, { recursive: true, force: true });
    expect(r.status).not.toBe(0);
    expect(r.stderr).toMatch(/expected|error/);
  });
});

describe("the page's questions and its model claims", () => {
  it('the right answer is rarely strictly the longest or the shortest option, JSX labels included', () => {
    expect(longestRight(questionsS234).length).toBeLessThanOrEqual(lengthLimit(questionsS234));
    expect(shortestRight(questionsS234).length).toBeLessThanOrEqual(lengthLimit(questionsS234));
  });
  it('NCCL and NVSHMEM specifics stay out of the review pool (the chapter is condensed)', () => {
    expect(Object.keys(questionsS234).filter((id) => /nccl|nvshmem/.test(id))).toEqual([]);
  });
  it(`p-window: Fig. 23.14 hides two equal exchanges up to ${WINDOW_STREAMS.toFixed(2)}, NCCL up to ${WINDOW_NCCL.toFixed(2)}`, () => {
    expect(WINDOW_STREAMS.toFixed(2)).toBe('2.99');
    expect(WINDOW_NCCL.toFixed(2)).toBe('6.00');
    const hidden = (v: 'streams' | 'nccl' | 'mpi', e: number) => simulate({ ...DEFAULTS, exTop: e, exBottom: e, version: v }).hidden;
    expect(hidden('streams', WINDOW_STREAMS - 0.01)).toBe(true);
    expect(hidden('streams', WINDOW_STREAMS + 0.01)).toBe(false);
    expect(hidden('streams', 2.5)).toBe(true);
    expect(hidden('streams', 3)).toBe(false);
    expect(hidden('nccl', WINDOW_NCCL)).toBe(true);
    expect(hidden('nccl', WINDOW_NCCL + 0.01)).toBe(false);
    for (const e of [0.5, 1, 2, 4]) expect(hidden('mpi', e)).toBe(false);
  });
  it('dropping the reset waits never fires in Fig. 23.14 at any slider setting, and does in the NCCL loop', () => {
    for (let i = 2; i <= 12; i += 2) for (let b = 0.5; b <= 4; b += 0.75) for (let t = 0.5; t <= 12; t += 1.5) for (let u = 0.5; u <= 12; u += 1.5)
      for (const priority of ['equal', 'higher'] as const) for (const issue of ['boundary-first', 'internal-first'] as const) {
        const r = simulate({ internal: i, boundary: b, exTop: t, exBottom: u, version: 'streams', defect: 'reset-waits', priority, issue });
        expect(r.violations.some((v) => v.kind === 'reset')).toBe(false);
      }
    const nccl = simulate({ ...DEFAULTS, exTop: 2, exBottom: 10, version: 'nccl', defect: 'reset-waits' });
    expect(nccl.violations.some((v) => v.kind === 'reset')).toBe(true);
  });
});
