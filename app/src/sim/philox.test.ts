import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  DIFFER_REGIONS,
  E3,
  FIRST_ONE,
  KATS,
  MPS_ROUNDS,
  avalanche,
  binadeProfile,
  countInputs,
  curandAfterInit,
  curandUniform,
  differingInputs,
  f32bits,
  hex,
  hipUniformAMD,
  m5Input,
  m5U,
  m5Uniform,
  m5Words,
  mpsPhilox,
  mulhilo,
  onesCount,
  parse64,
  philox,
  philoxTrace,
  roundsUntilAllWordsDiffer,
  sameWords,
  sampleConversions,
  split64,
} from './philox';

describe('Philox4x32-10 (x1-facts A4, E1)', () => {
  it('mulhilo is the exact 64-bit product', () => {
    const vals = [0, 1, 2, 0xffff, 0x10000, 0xd2511f53, 0xcd9e8d57, 0xffffffff, 0x80000000, 0x12345678];
    for (const a of vals)
      for (const b of vals) {
        const p = BigInt(a) * BigInt(b);
        expect(mulhilo(a, b)).toEqual([Number(p >> 32n), Number(p & 0xffffffffn)]);
      }
    let s = 1;
    for (let i = 0; i < 2000; i++) {
      s = (Math.imul(s, 1103515245) + 12345) >>> 0;
      const a = s;
      s = (Math.imul(s, 1103515245) + 12345) >>> 0;
      const p = BigInt(a) * BigInt(s);
      expect(mulhilo(a, s)).toEqual([Number(p >> 32n), Number(p & 0xffffffffn)]);
    }
  });
  it('reproduces Random123\'s three 10-round known-answer vectors', () => {
    for (const k of KATS) expect(philox(k.ctr, k.key).map(hex)).toEqual(k.expected.map(hex));
    expect(hex(philox([0, 0, 0, 0], [0, 0])[0])).toBe('6627e8d5');
  });
  it('the trace has ten rounds, the first with the unbumped key, the last ending on the vector', () => {
    const t = philoxTrace([0, 0, 0, 0], [0, 0]);
    expect(t.length).toBe(10);
    expect(t[0].key).toEqual([0, 0]);
    expect(t[1].key).toEqual([0x9e3779b9, 0xbb67ae85]);
    expect(t[9].out).toEqual(KATS[0].expected);
    // round 1 on a zero counter: both products are zero, the output is the key's XOR: zero
    expect(t[0].out).toEqual([0, 0, 0, 0]);
    // round 2: the bumped key enters through the XORs
    expect(t[1].out).toEqual([0x9e3779b9, 0, 0xbb67ae85, 0]);
  });
});

describe("M5's mapping and cuRAND's init (ours, A3)", () => {
  it('counter (step lo, step hi, stream lo, stream hi), key (seed lo, seed hi)', () => {
    const { ctr, key } = m5Input(0x0000000100000002n, 0x0000000300000004n, 0x0000000500000006n);
    expect(ctr).toEqual([6, 5, 4, 3]);
    expect(key).toEqual([2, 1]);
  });
  it('seed, stream and step 0 give the zero vector, so u = 0x6627e8 x 2^-24', () => {
    expect(m5Words(0n, 0n, 0n)).toEqual(KATS[0].expected);
    expect(m5U(0n, 0n, 0n)).toBe(0x6627e8 / 2 ** 24);
  });
  it('by our derivation, the first curand() after curand_init(seed, stream, 4 x step) is word 0', () => {
    const cases: [bigint, bigint, bigint][] = [[0n, 0n, 0n], [7n, 3n, 5n], [123456789123n, 2n ** 40n + 9n, 2n ** 33n + 1n], [2n ** 64n - 1n, 0n, 1n]];
    for (const [seed, stream, step] of cases) {
      const w = m5Words(seed, stream, step);
      expect(curandAfterInit(seed, stream, 4n * step, 4)).toEqual([...w]);
    }
    // the fifth value comes from the next counter
    expect(curandAfterInit(0n, 0n, 0n, 5)[4]).toBe(m5Words(0n, 0n, 1n)[0]);
    // an offset that is not a multiple of four starts inside the four words
    expect(curandAfterInit(9n, 1n, 4n * 3n + 2n, 1)[0]).toBe(m5Words(9n, 1n, 3n)[2]);
  });
  it('64-bit inputs: split and parse', () => {
    expect(split64(2n ** 64n - 1n)).toEqual([0xffffffff, 0xffffffff]);
    expect(parse64('0x10')).toBe(16n);
    expect(parse64('18446744073709551615')).toBe(2n ** 64n - 1n);
    expect(parse64('18446744073709551616')).toBeNull();
    expect(parse64('-1')).toBeNull();
    expect(parse64('12abc')).toBeNull();
  });
});

describe("PyTorch's Metal Philox as transcribed (D6, E3)", () => {
  it('for seed 0 and index 0 the MPS words are round 9 of the all-zero 10-round trace (the lesson says so)', () => {
    expect(philoxTrace([0, 0, 0, 0], [0, 0])[8].out.map(hex)).toEqual(['7028b1c8', 'd6594c40', '4f2051bb', '76d6628e']);
    expect(philoxTrace([0, 0, 0, 0], [0, 0])[8].out).toEqual(E3[0].mps);
  });
  it('nine rounds, high-low word order: reproduces both of E3\'s vectors', () => {
    expect(MPS_ROUNDS).toBe(9);
    for (const e of E3) {
      expect(mpsPhilox(e.seed, e.index).map(hex)).toEqual(e.mps.map(hex));
      expect(m5Words(e.seed, e.index, 0n).map(hex)).toEqual(e.ten.map(hex));
      expect(sameWords(mpsPhilox(e.seed, e.index), e.ten)).toBe(false);
    }
  });
});

describe('avalanche', () => {
  it('rows 0 and 1 at seed 0, step 0: one word after rounds 1 and 2, two after 3, three after 4, four after 5', () => {
    const A = m5Input(0n, 0n, 0n);
    const B = m5Input(0n, 1n, 0n);
    const a = avalanche(A.ctr, B.ctr, A.key);
    expect(a.differing.slice(0, 5)).toEqual([1, 1, 2, 3, 4]);
    expect(a.differing.slice(5)).toEqual([4, 4, 4, 4, 4]);
    expect(a.allBy).toBe(5);
    expect(roundsUntilAllWordsDiffer(0n, 0n, 0n)).toBe(5);
    // which word: word 1 (the low product of word 2) after round 1, word 0 after round 2
    expect(a.masks[0]).toEqual([false, true, false, false]);
    expect(a.masks[1]).toEqual([true, false, false, false]);
  });
  it('a one-bit change of the stream reaches all four words within five rounds for many seeds and steps', () => {
    for (let s = 0n; s < 40n; s++) expect(roundsUntilAllWordsDiffer(s * 977n, s, s * 3n)).toBeLessThanOrEqual(6);
  });
});

describe('bits to u (A2, B2, E2)', () => {
  it('the three conversions at the explorer\'s jumps', () => {
    expect(curandUniform(0)).toBe(Math.fround(2 ** -33));
    expect(hipUniformAMD(0)).toBe(Math.fround(2 ** -32));
    expect(f32bits(curandUniform(0))).toBe(0x2f000000);
    expect(m5Uniform(0)).toBe(0);
    // E2's examples: x = 0 gives 1.1641532e-10 and 2.3283064e-10; x = 0xFF gives 5.9488229e-08 and 5.9604645e-08
    expect(curandUniform(0).toPrecision(8)).toBe('1.1641532e-10');
    expect(hipUniformAMD(0).toPrecision(8)).toBe('2.3283064e-10');
    expect(curandUniform(0xff).toPrecision(8)).toBe('5.9488229e-8');
    expect(hipUniformAMD(0xff).toPrecision(8)).toBe('5.9604645e-8');
    expect(curandUniform(FIRST_ONE)).toBe(1);
    expect(hipUniformAMD(FIRST_ONE)).toBe(1);
    expect(curandUniform(FIRST_ONE - 1)).toBeLessThan(1);
    expect(hipUniformAMD(FIRST_ONE - 1)).toBeLessThan(1);
    expect(m5Uniform(0xffffffff)).toBe(1 - 2 ** -24);
  });
  it('the regions: every x below 2^23, even x in [2^23, 2^24), x with float32 value 2 mod 4 in [2^24, 2^25)', () => {
    const differ = (x: number) => curandUniform(x) !== hipUniformAMD(x);
    for (let x = 0; x < 3000; x++) expect(differ(x)).toBe(true);
    for (let x = 2 ** 23; x < 2 ** 23 + 3000; x++) expect(differ(x)).toBe(x % 2 === 0);
    for (let x = 2 ** 24; x < 2 ** 24 + 3000; x++) expect(differ(x)).toBe(Math.fround(x) % 4 === 2);
    for (let x = 2 ** 25 - 3000; x < 2 ** 25 + 3000; x++) expect(differ(x)).toBe(x < 2 ** 25 && Math.fround(x) % 4 === 2);
    expect(DIFFER_REGIONS.reduce((s, r) => s + r.count, 0)).toBe(2 ** 24);
  });
  it('the binade analysis: exactly 2^24 inputs differ (1 in 256), 128 give 1.0 for each vendor', () => {
    expect(differingInputs()).toBe(2 ** 24);
    expect(2 ** 32 / differingInputs()).toBe(256);
    expect(onesCount('curand')).toBe(128);
    expect(onesCount('hip')).toBe(128);
    const rows = binadeProfile();
    expect(rows.reduce((s, r) => s + r.size, 0)).toBe(2 ** 32);
    expect(rows.filter((r) => r.k >= 0 && r.k <= 22).every((r) => r.differ === r.size)).toBe(true);
    expect(rows.find((r) => r.k === 23)!.differ).toBe(2 ** 22);
    expect(rows.find((r) => r.k === 24)!.differ).toBe(2 ** 22);
    expect(rows.filter((r) => r.k >= 25).every((r) => r.differ === 0)).toBe(true);
    expect(rows.find((r) => r.k === 31)!.onesCurand).toBe(128);
    expect(countInputs((x) => x >= FIRST_ONE)).toBe(128);
  });
  it('the analysis agrees with a count over every distinct float32 value of x (2^24 + 2^26 + 1 of them)', () => {
    // Each conversion depends on x only through float32(x): count the preimages of each value.
    let differ = 0;
    let ones = 0;
    let all = 0;
    const add = (xf: number, pre: number) => {
      all += pre;
      if (curandUniform(xf) !== hipUniformAMD(xf)) differ += pre;
      if (curandUniform(xf) === 1) ones += pre;
    };
    for (let x = 0; x < 2 ** 24; x++) add(x, 1);
    for (let k = 24; k < 32; k++) {
      const step = 2 ** (k - 23);
      for (let xf = 2 ** k; xf < 2 ** (k + 1); xf += step) {
        // integers that round to xf: [xf - step/2, xf + step/2], ties to even
        const even = (xf / step) % 2 === 0;
        let lo = xf - step / 2 + (even ? 0 : 1);
        const hi = xf + step / 2 - (even ? 0 : 1);
        if (xf === 2 ** k) lo = Math.ceil(xf - step / 4); // below 2^k the spacing is half; the tie goes to 2^k
        add(xf, hi - lo + 1);
      }
    }
    // x in [2^32 - 128, 2^32) round to 2^32 as a float32; 0xFFFFFF80 stands for them (2^32
    // itself would wrap to 0 as a 32-bit word)
    expect(Math.fround(FIRST_ONE)).toBe(2 ** 32);
    add(FIRST_ONE, 128);
    expect(all).toBe(2 ** 32);
    expect(differ).toBe(2 ** 24);
    expect(ones).toBe(128);
  }, 60000);
  it('a seeded sample of 200,000 inputs agrees with the analysis', () => {
    const s = sampleConversions(200000, 1);
    expect(Math.abs(s.differ / s.n - 1 / 256)).toBeLessThan(4 * Math.sqrt((1 / 256) * (255 / 256) / s.n));
    expect(s.m5AtOne).toBe(0);
  });
});

// ---------------------------------------------------------------- the snippets themselves
const repo = path.resolve(__dirname, '../../..');
const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
const cxx = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
const python = ['python3'].find((c) => has(c, ['--version']));
const hasMetal = process.platform === 'darwin' && spawnSync('xcrun', ['-sdk', 'macosx', 'metal', '--version']).status === 0;
const INC = path.join(repo, 'labs/common/include');
const SNIP = path.join(repo, 'labs/x1/snippets');

interface Printed {
  kat: Map<string, number[]>;
  host: Map<string, number[]>;
  u: { seed: bigint; stream: bigint; step: bigint; bits?: number; ubits: number }[];
}
function parse(out: string, withBits: boolean): Printed {
  const p: Printed = { kat: new Map(), host: new Map(), u: [] };
  for (const l of out.split('\n')) {
    const f = l.trim().split(/\s+/);
    if (f[0] === 'kat' || f[0] === 'host') p[f[0] as 'kat' | 'host'].set(f[1], f.slice(2).map((h) => parseInt(h, 16) >>> 0));
    if (f[0] === 'u')
      p.u.push(withBits
        ? { seed: BigInt(f[1]), stream: BigInt(f[2]), step: BigInt(f[3]), bits: parseInt(f[4], 16) >>> 0, ubits: parseInt(f[5], 16) >>> 0 }
        : { seed: BigInt(f[1]), stream: BigInt(f[2]), step: BigInt(f[3]), ubits: parseInt(f[4], 16) >>> 0 });
  }
  return p;
}
function checkAgainstModel(p: Printed) {
  for (const k of KATS) expect(p.kat.get(k.id), k.id).toEqual(k.expected);
  expect(p.u.length).toBe(64);
  expect(p.u[0]).toMatchObject({ seed: 0n, stream: 0n, step: 0n });
  for (const r of p.u) {
    if (r.bits !== undefined) expect(r.bits).toBe(m5Words(r.seed, r.stream, r.step)[0]);
    expect(r.ubits).toBe(f32bits(m5U(r.seed, r.stream, r.step)));
  }
}

describe.skipIf(!cxx || !python)('labs/x1/snippets/philox.cu on the course emulator and the host', () => {
  it('prints the known-answer vectors and M5\'s u for 64 rows exactly as the model computes them', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'x1-philox-'));
    const driver = fs.readFileSync(path.join(SNIP, 'check/philox_kat.cu'), 'utf8');
    const snippet = fs.readFileSync(path.join(SNIP, 'philox.cu'), 'utf8');
    expect(driver).toContain('#include "../philox.cu"');
    fs.writeFileSync(path.join(dir, 'combined.cu'), driver.replace('#include "../philox.cu"', snippet));
    const t = spawnSync(python!, [path.join(repo, 'labs/tools/cu2emu.py'), path.join(dir, 'combined.cu'), path.join(dir, 'combined.cpp')], { encoding: 'utf8' });
    expect(t.status, t.stderr).toBe(0);
    const flags = ['-std=c++17', '-O1', '-g', '-fsanitize=address', '-fno-omit-frame-pointer', '-pthread', '-DPMPP_BACKEND_EMU',
      '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h', '-Wall', '-Wextra'];
    const c = spawnSync(cxx!, [...flags, '-c', path.join(dir, 'combined.cpp'), '-o', path.join(dir, 'k.o')], { encoding: 'utf8' });
    expect(c.status, c.stderr).toBe(0);
    const l = spawnSync(cxx!, ['-fsanitize=address', '-pthread', path.join(dir, 'k.o'), '-o', path.join(dir, 'kat')], { encoding: 'utf8' });
    expect(l.status, l.stderr).toBe(0);
    const r = spawnSync(path.join(dir, 'kat'), [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 120000 });
    fs.rmSync(dir, { recursive: true, force: true });
    expect(r.status, r.stderr).toBe(0);
    expect(r.stdout).toContain('DONE');
    const p = parse(r.stdout, true);
    checkAgainstModel(p);
    for (const k of KATS) expect(p.host.get(k.id), 'host ' + k.id).toEqual(k.expected);
  }, 180000);
});

describe.skipIf(!hasMetal || !cxx)('labs/x1/snippets/philox.metal on this Mac\'s GPU', () => {
  it('compiles at metal3.2 and metal4.1', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'x1-metal-'));
    for (const std of ['metal3.2', 'metal4.1']) {
      const r = spawnSync('xcrun', ['-sdk', 'macosx', 'metal', `-std=${std}`, '-c', path.join(SNIP, 'philox.metal'), '-o', path.join(dir, std + '.air')], { encoding: 'utf8' });
      expect(r.status, std + ': ' + r.stderr).toBe(0);
    }
    fs.rmSync(dir, { recursive: true, force: true });
  });
  it('prints the known-answer vectors and the same 64 floats as the model', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'x1-metalrun-'));
    const exe = path.join(dir, 'run');
    const b = spawnSync(cxx!, ['-std=c++17', '-fobjc-arc', path.join(SNIP, 'check/philox_metal_run.mm'), '-framework', 'Metal', '-framework', 'Foundation', '-o', exe], { encoding: 'utf8' });
    expect(b.status, b.stderr).toBe(0);
    const r = spawnSync(exe, [path.join(SNIP, 'philox.metal')], { encoding: 'utf8', timeout: 60000 });
    fs.rmSync(dir, { recursive: true, force: true });
    if (r.status === 3) return; // no Metal device (a virtual machine): nothing to run on
    expect(r.status, r.stderr).toBe(0);
    expect(r.stdout).toContain('DONE');
    checkAgainstModel(parse(r.stdout, false));
  }, 120000);
});
