import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { BOOK_BLOCK_LIMIT, cubeAI, slabAI, threadsCube, threadsSlab } from '../../sim/stencilslab';
import { CORRECT, localZ, makeProblem, planeTimeline } from '../../sim/zmarch';

// Code shown in section 8.5 must be code that is checked. The stepper's kernels and the
// indexed-window example are copies of labs/ch08/snippets/*.cu, which ./verify.sh
// front-end-checks; this test fails if a copy drifts from its file. It also ties the
// lines the stepper highlights, and the numbers the narration speaks, to the model.
const repo = path.resolve(__dirname, '../../../..');
const source = fs.readFileSync(path.join(__dirname, 'S85.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(source);
  if (!m) throw new Error(`no ${constant} in S85.tsx`);
  return m[1].trim();
};
const snippet = (name: string) => fs.readFileSync(path.join(repo, 'labs/ch08/snippets', name), 'utf8').trim();
const script = (n: number) => fs.readFileSync(path.join(repo, 'content/ch08/narration', `z-march-${n}.txt`), 'utf8');
const words = (s: string) => s.trim().split(/\s+/).length;

describe('section 8.5 kernels match their checked files', () => {
  it('stepper, three shared planes', () => expect(lesson('PLANES_BODY')).toBe(snippet('zmarch_planes.cu')));
  it('stepper, registers', () => expect(lesson('REGISTERS_BODY')).toBe(snippet('zmarch_registers.cu')));
  it('keep the window in registers', () => expect(lesson('INDEXED_WINDOW')).toBe(snippet('rolling_window_indexed.cu')));
});

describe('the lines the stepper highlights exist exactly once', () => {
  const once = (code: string, needle: string) => code.split('\n').filter((l) => l.includes(needle)).length;
  it('three shared planes', () => {
    const code = snippet('zmarch_planes.cu');
    for (const n of ['inPrev_s[ty][tx] = in[', 'inCurr_s[ty][tx] = in[', 'inNext_s[ty][tx] = in[', '// barrier 1', '// barrier 2',
      'out[i*N*N', 'bool inGrid', 'if (inGrid && inner)', 'inPrev_s[ty][tx] = inCurr_s[ty][tx]', 'inCurr_s[ty][tx] = inNext_s[ty][tx]'])
      expect(once(code, n), n).toBe(1);
  });
  it('registers', () => {
    const code = snippet('zmarch_registers.cu');
    for (const n of ['inPrev = in[', 'inCurr = in[', 'inNext = in[', '// barrier 1', '// barrier 2', 'out[i*N*N', 'bool inGrid', 'if (inGrid && inner)',
      'inPrev = inCurr;', 'inCurr = inNext;', 'inCurr_s[ty][tx] = inNext;', 'float inPrev = 0.0f'])
      expect(once(code, n), n).toBe(1);
  });
  it('each load is a guard line followed by its assignment', () => {
    for (const [f, needles] of [
      ['zmarch_planes.cu', ['inPrev_s[ty][tx] = in[', 'inCurr_s[ty][tx] = in[', 'inNext_s[ty][tx] = in[']],
      ['zmarch_registers.cu', ['inPrev = in[', 'inCurr = in[', 'inNext = in[']],
    ] as const) {
      const lines = snippet(f).split('\n');
      for (const n of needles) expect(lines[lines.findIndex((l) => l.includes(n)) - 1].trim(), n).toMatch(/^if \(jk/);
    }
    const r = snippet('zmarch_registers.cu').split('\n');
    expect(r[r.findIndex((l) => l.includes('inCurr = in[')) + 1].trim()).toBe('inCurr_s[ty][tx] = inCurr;');
  });
  it('the compute is the if and the four lines of the sum; the guard is the two bools and the if', () => {
    for (const f of ['zmarch_planes.cu', 'zmarch_registers.cu']) {
      const lines = snippet(f).split('\n');
      const a = lines.findIndex((l) => l.includes('if (inGrid && inner)'));
      expect(lines[a + 4].trim().endsWith(';')).toBe(true);
      const b = lines.findIndex((l) => l.includes('bool inGrid'));
      expect(lines[b + 4]).toContain('if (inGrid && inner)');
    }
  });
  it('the c3/c4 line the "as printed" option swaps is present in both', () => {
    const pair = '+ c3*inCurr_s[ty - 1][tx] + c4*inCurr_s[ty + 1][tx]';
    expect(once(snippet('zmarch_planes.cu'), pair)).toBe(1);
    expect(once(snippet('zmarch_registers.cu'), pair)).toBe(1);
  });
});

describe('the stepper shows the tile width of the problem it runs', () => {
  it('each stepper snippet defines the tile once, as the line the page replaces', () => {
    expect(source).toContain("const TILE_DEFINE = '#define IN_TILE_DIM 6';");
    for (const f of ['zmarch_planes.cu', 'zmarch_registers.cu'])
      expect(snippet(f).split('\n').filter((l) => l.trim() === '#define IN_TILE_DIM 6').length, f).toBe(1);
  });
});

/**
 * A model of rolling_window_indexed.cu: the grid of blocks, each block's threads run one
 * at a time between barriers in a given order (as the emulator does). Every statement of
 * the kernel is here with the file's own slot arithmetic: plane z lives in w[z % 3]. The
 * slot the compute reads for plane i - 1 is a parameter, so that a wrong index can be
 * shown to fail. Returns 'ok' or the first wrong or missing output.
 */
function runIndexedWindow(order: (n: number) => number[], prevSlot = (i: number) => (i + 2) % 3): string {
  const T = 6, OUT = T - 2, N = 10;
  const c = [0.4, 0.11, 0.13, 0.07, 0.17, 0.05, 0.19];
  const at = (i: number, j: number, k: number) => i * N * N + j * N + k;
  const input = Array.from({ length: N * N * N }, (_, q) => Math.sin(q * 1.3) + (q % 7));
  const want = (i: number, j: number, k: number) =>
    c[0] * input[at(i, j, k)] + c[1] * input[at(i, j, k - 1)] + c[2] * input[at(i, j, k + 1)] +
    c[3] * input[at(i, j - 1, k)] + c[4] * input[at(i, j + 1, k)] + c[5] * input[at(i - 1, j, k)] + c[6] * input[at(i + 1, j, k)];
  const out = new Map<number, number>();
  const blocks = Math.ceil(N / OUT);
  const threads = T * T;
  for (let bz = 0; bz < blocks; bz++)
    for (let by = 0; by < blocks; by++)
      for (let bx = 0; bx < blocks; bx++) {
        const iStart = bz * OUT;
        const inCurr_s = new Array(threads).fill(NaN);
        const w = Array.from({ length: threads }, () => [0, 0, 0]);
        const pos = (t: number) => ({ ty: Math.floor(t / T), tx: t % T, j: by * OUT + Math.floor(t / T) - 1, k: bx * OUT + (t % T) - 1 });
        const jk = (j: number, k: number) => j >= 0 && j < N && k >= 0 && k < N;
        // Up to the first barrier: the two loads before the loop, then iteration 0's load.
        for (const t of order(threads)) {
          const { ty, tx, j, k } = pos(t);
          if (jk(j, k) && iStart >= 1 && iStart - 1 < N) w[t][(iStart + 2) % 3] = input[at(iStart - 1, j, k)];
          if (jk(j, k) && iStart < N) { w[t][iStart % 3] = input[at(iStart, j, k)]; inCurr_s[ty * T + tx] = w[t][iStart % 3]; }
          if (jk(j, k) && iStart + 1 < N) w[t][(iStart + 1) % 3] = input[at(iStart + 1, j, k)];
        }
        for (let i = iStart; i < iStart + OUT; ++i) {
          // Barrier 1 to barrier 2: the compute.
          for (const t of order(threads)) {
            const { ty, tx, j, k } = pos(t);
            if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1 && ty >= 1 && ty < T - 1 && tx >= 1 && tx < T - 1) {
              const s = (y: number, x: number) => inCurr_s[y * T + x];
              const v = c[0] * w[t][i % 3] + c[1] * s(ty, tx - 1) + c[2] * s(ty, tx + 1) + c[3] * s(ty - 1, tx) + c[4] * s(ty + 1, tx) +
                c[5] * w[t][prevSlot(i)] + c[6] * w[t][(i + 1) % 3];
              if (out.has(at(i, j, k))) return `computed twice at (${i}, ${j}, ${k})`;
              out.set(at(i, j, k), v);
            }
          }
          // Barrier 2 to the next barrier 1: store the new current plane, then the next iteration's load.
          for (const t of order(threads)) {
            const { ty, tx, j, k } = pos(t);
            inCurr_s[ty * T + tx] = w[t][(i + 1) % 3];
            if (i + 1 < iStart + OUT && jk(j, k) && i + 2 < N) w[t][(i + 2) % 3] = input[at(i + 2, j, k)];
          }
        }
      }
  for (let i = 1; i < N - 1; i++)
    for (let j = 1; j < N - 1; j++)
      for (let k = 1; k < N - 1; k++) {
        const v = out.get(at(i, j, k));
        if (v === undefined) return `missing at (${i}, ${j}, ${k})`;
        if (!(Math.abs(v - want(i, j, k)) < 1e-9)) return `wrong at (${i}, ${j}, ${k})`;
      }
  return 'ok';
}
const inOrder = (n: number) => Array.from({ length: n }, (_, t) => t);
const reversed = (n: number) => inOrder(n).reverse();
const shuffled = (n: number) => inOrder(n).sort((a, b) => ((a * 7919) % 13) - ((b * 7919) % 13) || a - b);

describe('"Keep the window in registers": the indexed window computes the same sweep', () => {
  const file = snippet('rolling_window_indexed.cu');
  it('the model uses the file\'s own slot arithmetic', () => {
    for (const needle of [
      'w[(iStart + 2) % 3] = in[(iStart - 1)*N*N + j*N + k]',
      'w[iStart % 3] = in[iStart*N*N + j*N + k]; inCurr_s[ty][tx] = w[iStart % 3];',
      'w[(i + 1) % 3] = in[(i + 1)*N*N + j*N + k]',
      'c0*w[i % 3]',
      'c5*w[(i + 2) % 3] + c6*w[(i + 1) % 3]',
      'inCurr_s[ty][tx] = w[(i + 1) % 3];',
    ])
      expect(file, needle).toContain(needle);
    expect(file.match(/__syncthreads\(\)/g)?.length).toBe(2);
  });
  it('every interior point of a 10³ grid, once each, under three thread orders', () => {
    for (const o of [inOrder, reversed, shuffled]) expect(runIndexedWindow(o)).toBe('ok');
  });
  it('the test has teeth: reading plane i - 1 from the wrong slot fails', () => {
    expect(runIndexedWindow(inOrder, (i) => (i + 1) % 3)).not.toBe('ok');
  });
});

describe('the narration of "Marching through z" agrees with the model', () => {
  const prob = makeProblem(10);
  const tl = planeTimeline({ N: 10, inTile: 6, kernel: 'planes', ...CORRECT }, prob);
  const L = (z: number | null) => (z === null ? null : localZ(tl.iStart, z));
  const f = tl.frames;
  const last = f[f.length - 1];

  it('every script is 35 to 60 words and reads no symbols', () => {
    for (let n = 1; n <= 6; n++) {
      const s = script(n);
      expect(words(s), `z-march-${n}`).toBeGreaterThanOrEqual(35);
      expect(words(s), `z-march-${n}`).toBeLessThanOrEqual(60);
      expect(s, `z-march-${n}`).not.toMatch(/[=×÷^<>/_*]|FLOP/);
    }
  });
  it('part 1: 216 threads for the tile, 32,768 for a 32-wide cube, at most 1,024', () => {
    const s = script(1);
    expect(s).toContain(String(threadsCube(6)));
    expect(s).toContain(threadsCube(32).toLocaleString('en-US'));
    expect(s).toContain(BOOK_BLOCK_LIMIT.toLocaleString('en-US'));
  });
  it('part 2: 36 threads, 16 compute, planes 0 and 1 loaded before the loop', () => {
    const s = script(2);
    expect(s).toContain(`${threadsSlab(6)} threads`);
    expect(s).toContain(`${f[2].active.filter(Boolean).length} threads compute`);
    expect([L(f[0].hold.prev), L(f[0].hold.curr)]).toEqual([0, 1]);
    expect(s).toMatch(/plane 0 into the previous buffer, and of plane 1 into the current/);
  });
  it('part 3: plane 2 into next, output plane 1', () => {
    const s = script(3);
    expect(L(f[1].hold.next)).toBe(2);
    expect(f[2].outputPlanes.map(L)).toEqual([1]);
    expect(s).toContain('plane 2 into the next buffer');
    expect(s).toContain('output plane 1 is done');
  });
  it('part 5: 6 planes for 4, 1,024 threads, 1.47 against 0.96', () => {
    const s = script(5);
    expect(s).toContain(`loaded ${last.planesLoaded} planes to produce ${last.outputPlanes.length}`);
    expect(s).toContain(`${threadsSlab(32).toLocaleString('en-US')} threads`);
    expect(s).toContain(String(Math.round(slabAI(32, 30) * 100) / 100));
    expect(s).toContain(String(Math.round(cubeAI(8) * 100) / 100));
  });
  it('part 6: the registers kernel loads exactly the same points', () => {
    const r = planeTimeline({ N: 10, inTile: 6, kernel: 'registers', ...CORRECT }, prob);
    expect(r.frames.map((x) => x.loads)).toEqual(f.map((x) => x.loads));
    expect(script(6)).toContain('loads exactly the same points');
  });
});
