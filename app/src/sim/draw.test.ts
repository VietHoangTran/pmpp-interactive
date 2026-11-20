import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  GE_GT,
  P_DRAW,
  RISK,
  SAMPLER_PROBS,
  STEPPER_CFG,
  STEP_ROWS,
  atRiskBound,
  barrierHunt,
  cdfIn,
  chooseIn,
  compareSamplers,
  disagreement,
  drawKernel,
  flipPoints,
  huntDraw,
  inverseCdf,
  kernelTotal,
  lastPositive,
  marginOf,
  perturb,
  sampleLogits,
  uniformsFor,
  weightsFrom,
  SCENE_U,
  SCENE_WEIGHTS,
  sceneBoundary,
  sceneDraw,
  FALLBACK_WEIGHTS,
  TOP_U,
  fallbackDraw,
  SCENE_CFG,
  type KernelCfg,
} from './draw';
import { m5U, m5Uniform, philox } from './philox';

const f32 = Math.fround;

describe('the rule on one row', () => {
  it('p-draw: four equal weights and u = 0.6 draw index 2 (target 2.4, sums 1, 2, 3, 4)', () => {
    const d = inverseCdf(P_DRAW.weights, P_DRAW.u);
    expect(d.target).toBeCloseTo(2.4, 12);
    expect(d.inclusive).toEqual([1, 2, 3, 4]);
    expect(d.index).toBe(2);
    expect(drawKernel(P_DRAW.weights, P_DRAW.u).index).toBe(2);
  });
  it('q-ge-gt: at u = 0, >= picks the zero-weight token 0 and > picks token 1', () => {
    expect(inverseCdf(GE_GT.weights, GE_GT.u, { rule: 'ge' }).index).toBe(0);
    expect(inverseCdf(GE_GT.weights, GE_GT.u, { rule: 'gt' }).index).toBe(1);
    expect(drawKernel(GE_GT.weights, GE_GT.u).index).toBe(1);
    // M5's u can be exactly 0 (E2): Philox words below 2^8 give it
    expect(m5Uniform(0xff)).toBe(0);
  });
  it('a zero weight is never drawn by the strict rule, for any u M5 can produce', () => {
    const w = [0, 2, 0, 0, 1, 0];
    for (let k = 0; k <= 64; k++) {
      const u = k === 64 ? 1 - 2 ** -24 : k / 64;
      const i = inverseCdf(w, u).index;
      expect(w[i], `u = ${u}`).toBeGreaterThan(0);
      expect(w[drawKernel(w, u).index]).toBeGreaterThan(0);
    }
  });
  it("q-u1: curand_uniform's u = 1.0 against a float row summing to 0.99999994 finds nothing without a fallback", () => {
    const probs = [0.5, 0.25, f32(0.25 - 2 ** -24), 0];
    const sum = probs.reduce((s, p) => f32(s + p), 0);
    expect(sum).toBe(f32(0.99999994));
    expect(sum).toBe(1 - 2 ** -24);
    const bare = inverseCdf(probs, 1, { rule: 'ge', scale: false, fallback: false, round: f32 });
    expect(bare.invalid).toBe(true);
    expect(bare.index).toBe(probs.length);
    const safe = inverseCdf(probs, 1, { rule: 'ge', scale: false, fallback: true, round: f32 });
    expect(safe.fallback).toBe(true);
    expect(safe.index).toBe(2); // the last positive weight, not the zero at the end
    expect(lastPositive(probs)).toBe(2);
  });
  it('u x total keeps the strict rule safe at the top: with M5\'s largest u the left-to-right float32 target stays below the total', () => {
    const u = 1 - 2 ** -24;
    for (let s = 0; s < 300; s++) {
      const w = weightsFrom(sampleLogits(5 + (s % 50), s), 0.5 + (s % 7) * 0.3);
      const d = inverseCdf(w, u, { round: f32 });
      expect(d.fallback, `seed ${s}`).toBe(false);
    }
  });
  it('the fallback is reached when the kernel\'s total and its scan round differently (a case exists and is handled)', () => {
    const u = 1 - 2 ** -24;
    let found: { w: number[]; seed: number } | null = null;
    for (let s = 0; s < 4000 && !found; s++) {
      const n = 30 + (s % 170);
      const w = weightsFrom(sampleLogits(n, s), 0.4 + (s % 11) * 0.25);
      const k = drawKernel(w, u);
      if (k.fallback) found = { w, seed: s };
    }
    expect(found, 'some row where u x total rounds to or above the last scanned sum').not.toBeNull();
    const k = drawKernel(found!.w, u);
    expect(k.index).toBe(lastPositive(found!.w));
    const { cum } = cdfIn(found!.w, 'f32-kernel');
    expect(cum[cum.length - 1]).toBeLessThanOrEqual(k.target);
  });
});

describe('the kernel, one block', () => {
  it("phase 1's total is the kernel's order of additions", () => {
    const w = weightsFrom(sampleLogits(200), 0.7);
    const t = kernelTotal(w, STEPPER_CFG);
    expect(t.partials.length).toBe(16);
    expect(t.warpSums.length).toBe(2);
    expect(t.total).toBeCloseTo(w.reduce((s, x) => s + x, 0), 4);
  });
  it('chunkedSearch walks chunks, skips those at or below the target, scans one, and its winner is the first sum above the target', () => {
    const w = weightsFrom(sampleLogits(200), 0.7);
    const k = drawKernel(w, 0.85);
    const crossing = k.chunks.findIndex((c) => c.crosses);
    expect(crossing).toBe(k.chunks.length - 1);
    expect(k.chunks.slice(0, crossing).every((c) => c.aggAfter <= k.target)).toBe(true);
    const { cum } = cdfIn(w, 'f32-kernel');
    expect(k.index).toBe(cum.findIndex((c) => c > k.target));
    expect(k.winner).not.toBeNull();
    const ch = k.chunks[crossing];
    expect(ch.ballots.some((b) => b > 0)).toBe(true);
  });
  it('on 10,000 seeded cases chunkedSearch equals the double inverse CDF whenever the draw is not within 2^-16 of a boundary, and always equals its own float32 scan', () => {
    let decisive = 0;
    let all = 0;
    for (let s = 0; s < 10000; s++) {
      const n = 1 + (s % 200);
      const w = weightsFrom(sampleLogits(n, s + 11), 0.3 + (s % 9) * 0.35);
      const u = m5U(BigInt(s), 1n, 0n);
      const cfg: KernelCfg = s % 3 === 0 ? STEPPER_CFG : s % 3 === 1 ? { threads: 64, warp: 32, vec: 4 } : { threads: 32, warp: 16, vec: 2 };
      const k = drawKernel(w, u, cfg);
      const { cum } = cdfIn(w, 'f32-kernel', cfg);
      const own = cum.findIndex((c) => c > k.target);
      expect(k.index, `seed ${s}`).toBe(own >= 0 ? own : lastPositive(w));
      const ref = inverseCdf(w, u).index;
      all++;
      if (marginOf(w, u) > 2 ** -16) {
        decisive++;
        expect(k.index, `seed ${s}`).toBe(ref);
      }
    }
    expect(decisive / all).toBeGreaterThan(0.95);
  }, 60000);
});

describe('the barrier hunt', () => {
  const w = weightsFrom(sampleLogits(40), 1);
  it('with both barriers every thread reads the total, under every order', () => {
    for (const o of ['in-order', 'reversed', 'warp-ahead'] as const) expect(barrierHunt(w, 0.5, STEPPER_CFG, 'none', o).wrong).toBe(0);
  });
  it('barrier A removed: in order the 8 threads of warp 1 read warp 0\'s partial sum; reversed all but thread 0 do', () => {
    const a = barrierHunt(w, 0.5, STEPPER_CFG, 'written', 'in-order');
    expect(a.wrong).toBe(8);
    expect(a.reads.filter((r) => !r.ok).every((r) => r.warp === 1 && r.what === "warp 0's partial sum")).toBe(true);
    const b = barrierHunt(w, 0.5, STEPPER_CFG, 'written', 'reversed');
    expect(b.wrong).toBe(15);
    expect(b.reads.find((r) => r.thread === 0)!.ok).toBe(true);
    expect(barrierHunt(w, 0.5, STEPPER_CFG, 'written', 'warp-ahead').wrong).toBe(0);
  });
  it('barrier B removed: neither emulator order sees it; a warp that runs ahead overwrites the total before warp 1 reads it', () => {
    expect(barrierHunt(w, 0.5, STEPPER_CFG, 'read', 'in-order').wrong).toBe(0);
    expect(barrierHunt(w, 0.5, STEPPER_CFG, 'read', 'reversed').wrong).toBe(0);
    const g = barrierHunt(w, 0.5, STEPPER_CFG, 'read', 'warp-ahead');
    expect(g.wrong).toBe(8);
    expect(g.reads.filter((r) => !r.ok).every((r) => r.warp === 1 && r.what === "warp 0's chunk-0 total")).toBe(true);
  });
  it("the stepper's 200-survivor row: chunks 0 and 1 skipped, chunk 2 scanned, survivor 133", () => {
    const r = STEP_ROWS.find((x) => x.id === 'long')!;
    const k = drawKernel(weightsFrom(sampleLogits(r.n), r.T), f32(r.u));
    expect(k.chunks.map((c) => c.crosses)).toEqual([false, false, true]);
    expect(k.index).toBe(133);
  });
  it("the stepper's hunt row: barrier A removed changes the token in order and reversed; barrier B only when a warp runs ahead", () => {
    const r = STEP_ROWS.find((x) => x.id === 'hunt')!;
    const hw = weightsFrom(sampleLogits(r.n), r.T);
    const u = f32(r.u);
    const at = (m: 'written' | 'read', o: 'in-order' | 'reversed' | 'warp-ahead') => huntDraw(hw, u, STEPPER_CFG, m, o);
    expect(at('written', 'in-order').index).not.toBe(at('written', 'in-order').correct);
    expect(at('written', 'reversed').index).not.toBe(at('written', 'reversed').correct);
    expect(at('written', 'warp-ahead').index).toBe(at('written', 'warp-ahead').correct);
    expect(at('read', 'in-order').index).toBe(at('read', 'in-order').correct);
    expect(at('read', 'reversed').index).toBe(at('read', 'reversed').correct);
    expect(at('read', 'warp-ahead').index).not.toBe(at('read', 'warp-ahead').correct);
    // the wrong token comes earlier: a smaller total makes a smaller target
    expect(at('written', 'in-order').index).toBeLessThan(at('written', 'in-order').correct);
    for (const row of STEP_ROWS.filter((x) => x.id !== 'hunt')) {
      const ww = weightsFrom(sampleLogits(row.n), row.T);
      expect(huntDraw(ww, f32(row.u), STEPPER_CFG, 'none', 'in-order').index).toBe(drawKernel(ww, f32(row.u)).index);
    }
  }, 60000);
});

describe('near a boundary', () => {
  const w = weightsFrom(sampleLogits(40), 1);
  it('q-risk: 40 survivors and a margin of 2^-20 put at most about 74 draws per million at risk', () => {
    expect(atRiskBound(RISK.n, RISK.margin) * 1e6).toBeCloseTo(74.39, 1);
    expect(Math.round(atRiskBound(RISK.n, RISK.margin) * 1e6)).toBe(74);
  });
  it('the three orders disagree only in narrow bands, within the bound for a margin of a few float32 steps', () => {
    const rows = (['double', 'f32-loop', 'f32-kernel'] as const).map((order) => ({ weights: w, order }));
    const d = disagreement(rows);
    expect(d.bands.length).toBeGreaterThan(0);
    for (const b of d.bands) expect(b.hi - b.lo).toBeLessThan(2 ** -20);
    expect(d.measure).toBeLessThan(atRiskBound(40, 2 ** -21));
  }, 60000);
  it('a perturbation of the weights (another backend\'s exp) widens the bands roughly in proportion', () => {
    const base = disagreement([{ weights: w, order: 'double' }, { weights: w, order: 'f32-kernel' }]).measure;
    const p = perturb(w, 2 ** -20);
    const wide = disagreement([{ weights: w, order: 'double' }, { weights: p, order: 'f32-kernel' }]).measure;
    expect(wide).toBeGreaterThan(base);
    expect(wide).toBeLessThan(atRiskBound(40, 2 ** -18));
  });
  it('chooseIn agrees with the flip points: just below a flip the earlier token, just above it the later one', () => {
    for (const order of ['double', 'f32-loop', 'f32-kernel'] as const) {
      const fp = flipPoints(w, order);
      for (const i of [0, 5, 20, 38]) {
        expect(chooseIn(w, fp[i] - 2 ** -20, order)).toBe(i);
        expect(chooseIn(w, fp[i] + 2 ** -20, order)).toBe(i + 1);
      }
    }
  });
  it('a draw far from every boundary has a large margin, one on a boundary a margin near 0', () => {
    const fp = flipPoints(w, 'double');
    expect(marginOf(w, fp[7])).toBeLessThan(1e-12);
    expect(marginOf(w, (fp[7] + fp[8]) / 2)).toBeGreaterThan(1e-4);
  });
});

describe('same bits, three samplers', () => {
  it('uniform 0 of each draw is M5\'s u for (seed, stream, step = draw)', () => {
    for (const d of [0, 1, 17]) expect(uniformsFor(5, 2, d, 6)[0]).toBe(m5U(5n, 2n, BigInt(d)));
    expect(uniformsFor(5, 2, 3, 6)[4]).toBe(m5Uniform(philox([3, 1, 2, 0], [5, 0])[0]));
  });
  it('at 100,000 draws all three frequencies lie within 4 standard errors of p, while they agree draw by draw far less often', () => {
    const N = 100000;
    const c = compareSamplers(SAMPLER_PROBS, N);
    for (const m of ['inverse-cdf', 'race', 'gumbel'] as const)
      SAMPLER_PROBS.forEach((p, i) => {
        const se = Math.sqrt((p * (1 - p)) / N);
        expect(Math.abs(c.counts[m][i] / N - p), `${m} token ${i}`).toBeLessThan(4 * se);
      });
    expect(c.agree.invRace).toBeLessThan(0.6);
    expect(c.agree.invGumbel).toBeLessThan(0.6);
    expect(c.agree.raceGumbel).toBeLessThan(0.9);
    expect(c.agree.invRace).toBeGreaterThan(SAMPLER_PROBS.reduce((s, p) => s + p * p, 0) * 0.9);
  }, 60000);
  it('with q = -ln u the race is Gumbel-max: the same token on every draw', () => {
    const c = compareSamplers(SAMPLER_PROBS, 20000, 5, 0, 'log-u');
    expect(c.agree.raceGumbel).toBe(1);
  });
});

// ---------------------------------------------------------------- the snippet on the emulator
const repo = path.resolve(__dirname, '../../..');
const cxx = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
const SNIP = path.join(repo, 'labs/x1/snippets');
const INC = path.join(repo, 'labs/common/include');
const BLOCK = 64;
const EMU_CFG: KernelCfg = { threads: BLOCK, warp: 32, vec: 4 };
const bitsOf = (v: number) => {
  const b = new DataView(new ArrayBuffer(4));
  b.setFloat32(0, v);
  return b.getUint32(0).toString(16);
};
interface Row {
  w: number[];
  u: number;
}
const ROWS: Row[] = [];
for (let s = 0; s < 24; s++) {
  const n = [1, 3, 40, 64, 255, 256, 257, 600][s % 8];
  ROWS.push({ w: weightsFrom(sampleLogits(n, 100 + s), [0.5, 1, 2][s % 3]), u: m5U(BigInt(s), 9n, 2n) });
}
ROWS.push({ w: [0, 0.5, 0.5], u: 0 }, { w: [1, 1, 1, 1], u: f32(0.6) }, { w: [0, 2, 0, 0, 1, 0], u: 1 - 2 ** -24 });
const STRIDE = Math.max(...ROWS.map((r) => r.w.length));

/** The snippet with one barrier's line removed (the line that names it), driven by check/draw_check.cu. */
function build(dir: string, missing: 'none' | 'A' | 'B', race: boolean): string | null {
  const src = fs
    .readFileSync(path.join(SNIP, 'draw_search.cu'), 'utf8')
    .split('\n')
    .filter((l) => missing === 'none' || !l.includes(`// barrier ${missing}:`))
    .join('\n');
  const driver = fs.readFileSync(path.join(SNIP, 'check/draw_check.cu'), 'utf8');
  expect(driver).toContain('#include "../draw_search.cu"');
  const tag = `${missing}-${race ? 'race' : 'emu'}`;
  const cu = path.join(dir, `all-${tag}.cu`);
  fs.writeFileSync(cu, '#line 1 "draw_search.cu"\n' + src + '\n#line 1 "draw_check.cu"\n' + driver.replace('#include "../draw_search.cu"', ''));
  const cpp = cu.replace(/\.cu$/, '.cpp');
  const t = spawnSync('python3', [path.join(repo, 'labs/tools/cu2emu.py'), cu, cpp], { encoding: 'utf8' });
  expect(t.status, t.stdout + t.stderr).toBe(0);
  const exe = path.join(dir, tag);
  const flags = ['-std=c++17', '-O1', '-g', '-fno-omit-frame-pointer', '-pthread', `-DDRAW_BLOCK=${BLOCK}`, '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h', cpp];
  const r = spawnSync(cxx!, race ? ['-fsanitize=thread', '-DPMPP_BACKEND_RACE', ...flags, '-o', exe] : ['-fsanitize=address', '-DPMPP_BACKEND_EMU', ...flags, '-o', exe], { encoding: 'utf8' });
  if (race) return r.status === 0 ? exe : null; // ThreadSanitizer is not available everywhere
  expect(r.status, r.stderr).toBe(0);
  return exe;
}
function runRows(exe: string, dir: string, rows: Row[]) {
  const file = path.join(dir, 'rows.txt');
  fs.writeFileSync(file, `${rows.length} ${STRIDE}\n` + rows.map((r) => `${r.w.length} ${bitsOf(r.u)} ${r.w.map(bitsOf).join(' ')}`).join('\n') + '\n');
  const out = spawnSync(exe, [file], {
    encoding: 'utf8',
    timeout: 120000,
    env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0', TSAN_OPTIONS: 'abort_on_error=0:exitcode=66:halt_on_error=0:report_thread_leaks=0' },
  });
  const byOrder: Record<string, number[]> = {};
  let cur = '';
  for (const l of out.stdout.split('\n')) {
    if (l.startsWith('order ')) (cur = l.slice(6)), (byOrder[cur] = []);
    const m = /^token (\d+) (-?\d+)/.exec(l);
    if (m) byOrder[cur][Number(m[1])] = Number(m[2]);
  }
  return { status: out.status, stdout: out.stdout, stderr: out.stderr, byOrder };
}

describe.skipIf(!cxx)('labs/x1/snippets/draw_search.cu on the course emulator', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'x1-draw-'));
  it('the snippet draws the model\'s token on every row, under all three emulator orders', () => {
    const r = runRows(build(dir, 'none', false)!, dir, ROWS);
    expect(r.status, r.stderr.slice(0, 2000)).toBe(0);
    const want = ROWS.map((x) => drawKernel(x.w, x.u, EMU_CFG).index);
    for (const o of ['in order', 'reversed', 'shuffled']) expect(r.byOrder[o], o).toEqual(want);
  }, 240000);
  it('barrier A removed: in order and reversed give the tokens the model predicts, and some are wrong', () => {
    const r = runRows(build(dir, 'A', false)!, dir, ROWS);
    expect(r.status, r.stderr.slice(0, 2000)).toBe(0);
    let wrong = 0;
    for (const [o, mo] of [['in order', 'in-order'], ['reversed', 'reversed']] as const)
      ROWS.forEach((x, i) => {
        const h = huntDraw(x.w, x.u, EMU_CFG, 'written', mo);
        if (h.search.voteSplit) return; // a full-mask ballot not every lane reaches: not modelled
        expect(r.byOrder[o][i], `${o} row ${i}`).toBe(h.index);
        if (h.index !== h.correct) wrong++;
      });
    expect(wrong).toBeGreaterThan(0);
  }, 240000);
  it('barrier B removed: every emulator order still gives the right token', () => {
    const r = runRows(build(dir, 'B', false)!, dir, ROWS);
    expect(r.status, r.stderr.slice(0, 2000)).toBe(0);
    const want = ROWS.map((x) => drawKernel(x.w, x.u, EMU_CFG).index);
    for (const o of ['in order', 'reversed', 'shuffled']) expect(r.byOrder[o], o).toEqual(want);
  }, 240000);
  it('the race flavour names the read of the total and the write that overwrites it when barrier B is removed, and reports nothing with both', (ctx) => {
    const exeB = build(dir, 'B', true);
    const exeOk = build(dir, 'none', true);
    if (!exeB || !exeOk) return ctx.skip();
    const src = fs.readFileSync(path.join(SNIP, 'draw_search.cu'), 'utf8').split('\n');
    const lineOf = (re: RegExp) => src.findIndex((l) => re.test(l)) + 1;
    // barrier B's line sits between the two, so the write moves one line up once it is gone
    const read = lineOf(/const float target = u\[blockIdx\.x\] \* scratch_s\[0\]/);
    const write = lineOf(/scratch_s\[warpIdx\(\)\] = incl/) - 1;
    expect(lineOf(/barrier B:/)).toBe(read + 1);
    const few = ROWS.slice(0, 6);
    const ok = runRows(exeOk, dir, few);
    expect(ok.status, ok.stderr.slice(0, 1500)).toBe(0);
    const bad = runRows(exeB, dir, few);
    expect(bad.status).toBe(66);
    expect(bad.stderr).toContain(`draw_search.cu:${read}`);
    expect(bad.stderr).toContain(`draw_search.cu:${write}`);
  }, 300000);
});

describe('the find-u scene', () => {
  it('chunk 0 is skipped, chunk 1 crosses, warp 1 lane 1 wins with survivor 26', () => {
    const k = sceneDraw();
    expect(k.chunks.length).toBe(2);
    expect(k.chunks[0].crosses).toBe(false);
    expect(k.chunks[1].crosses).toBe(true);
    expect(k.index).toBe(26);
    expect(k.winner).toEqual({ chunk: 1, warp: 1, lane: 1, thread: 5, j: 0 });
    expect(k.index).toBe(inverseCdf(SCENE_WEIGHTS, SCENE_U).index);
  });
  it('a u at which the two float32 orders differ in the last bit of one sum and pick neighbouring tokens', () => {
    const b = sceneBoundary()!;
    expect(b.boundary).toBe(6);
    expect(b.loop.index).toBe(7);
    expect(b.kernel.index).toBe(6);
    expect(b.loop.cum).not.toBe(b.kernel.cum);
    expect(Math.abs(b.loop.cum - b.kernel.cum)).toBeLessThanOrEqual(2 ** -21); // one float32 step at 4.27
    expect(b.loop.target).toBe(b.kernel.target);
    expect(Number.isInteger(b.u * 2 ** 24)).toBe(true);
  });
});

describe('the fallback row', () => {
  it('at the largest u the kernel scans to the end below the target and takes the last positive weight', () => {
    const k = fallbackDraw();
    expect(k.fallback).toBe(true);
    expect(k.index).toBe(FALLBACK_WEIGHTS.length - 1);
    const { cum } = cdfIn(FALLBACK_WEIGHTS, 'f32-kernel', SCENE_CFG);
    expect(cum[cum.length - 1]).toBeLessThanOrEqual(k.target);
    expect(k.target).toBeLessThan(k.total);
    expect(TOP_U).toBe(m5Uniform(0xffffffff));
    // left to right in one order, the same row needs no fallback
    expect(inverseCdf(FALLBACK_WEIGHTS, TOP_U, { round: Math.fround }).fallback).toBe(false);
  });
  it('at the largest u the fallback is common for the stepper\'s block: more than one row in ten', () => {
    let c = 0;
    for (let s = 0; s < 400; s++) if (drawKernel(weightsFrom(sampleLogits(1 + (s % 200), s), 0.5 + (s % 5) * 0.5), TOP_U).fallback) c++;
    expect(c / 400).toBeGreaterThan(0.1);
  }, 60000);
});
