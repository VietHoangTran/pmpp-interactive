import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { conv, decompose } from '../../sim/decompose';

// Section 22.2 shows two kernels of its own, "This scatter passes on the emulator". They are
// copied into S222.tsx (the ch05 pattern) from labs/ch22/snippets/, where ./verify.sh
// front-end-checks them. This test fails if a copy drifts, checks each kernel's shape against
// the model, and runs both on the course's CPU emulators through
// labs/ch22/snippets/check/conv_check.cu: the plain scatter must pass on `emu` in all three
// thread orders (the question's premise) and be reported by ThreadSanitizer on `race`, while
// the gather and an atomicAdd copy pass on both with no report. The emulator part is skipped
// when no C++ compiler or python3 is found; the race part when ThreadSanitizer does not build.
const repo = path.resolve(__dirname, '../../../..');
const lesson = (constant: string) => {
  const src = fs.readFileSync(path.join(__dirname, 'S222.tsx'), 'utf8');
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in S222.tsx`);
  return m[1].trim();
};
const SNIP = 'labs/ch22/snippets';
const snippet = (name: string) => fs.readFileSync(path.join(repo, SNIP, name), 'utf8').trim();
const code = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');

describe('the lesson\'s copies match the checked files', () => {
  it('scatter_conv_plain.cu', () => expect(lesson('SCATTER')).toBe(snippet('scatter_conv_plain.cu')));
  it('gather_conv.cu', () => expect(lesson('GATHER')).toBe(snippet('gather_conv.cu')));
});

describe('the kernels\' shape', () => {
  const scatter = code(snippet('scatter_conv_plain.cu'));
  const gather = code(snippet('gather_conv.cu'));
  it('the scatter: one input per thread, a plain += into out, no atomic', () => {
    expect(scatter).toMatch(/int i = blockIdx\.x \* blockDim\.x \+ threadIdx\.x;/);
    expect(scatter).toMatch(/int o = i - t \+ r;/);
    expect(scatter).toMatch(/out\[o\] \+= filter\[t\] \* v;/);
    expect(scatter).not.toMatch(/atomic/);
    // The outputs start at zero because the threads add into them.
    expect(scatter).toMatch(/cudaMemset\(out, 0, n \* sizeof\(float\)\);/);
  });
  it('the gather: one output per thread, a register sum, one plain store', () => {
    expect(gather).toMatch(/int o = blockIdx\.x \* blockDim\.x \+ threadIdx\.x;/);
    expect(gather).toMatch(/int i = o \+ t - r;/);
    expect(gather).toMatch(/acc \+= filter\[t\] \* in\[i\];/);
    expect(gather.match(/out\[o\]/g)?.length).toBe(1);
    expect(gather).toMatch(/out\[o\] = acc;/);
  });
  it('both index the same contributions as the model\'s convolution (zero-padded, width 2r + 1)', () => {
    // Enumerate the scatter's (i, o) pairs and the gather's (o, i) pairs as the kernels do.
    const n = 12, r = 2;
    const pairs = (who: 'scatter' | 'gather') => {
      const out: string[] = [];
      for (let a = 0; a < n; a++)
        for (let t = 0; t <= 2 * r; t++) {
          const b = who === 'scatter' ? a - t + r : a + t - r;
          if (b >= 0 && b < n) out.push(who === 'scatter' ? `${a}>${b}` : `${b}>${a}`);
        }
      return out.sort();
    };
    const model = conv(n, 2 * r + 1).edges.map((e) => `${e.i}>${e.o}`).sort();
    expect(pairs('scatter')).toEqual(model);
    expect(pairs('gather')).toEqual(model);
    // The question the snippet comes with: five writers on an inner output.
    expect(decompose(conv(n, 5), 'input', { warpWidth: 1, S: 1 }).writers[6]).toBe(5);
  });
});

// ---------------------------------------------------------------- on the emulators
const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
const cxx = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
const python = has('python3', ['--version']);
const INC = path.join(repo, 'labs/common/include');
const ENV = {
  ...process.env,
  ASAN_OPTIONS: 'detect_leaks=0',
  TSAN_OPTIONS: 'abort_on_error=0:exitcode=66:halt_on_error=0:report_thread_leaks=0',
};

describe.skipIf(!cxx || !python)('on the course emulators (labs/ch22/snippets/check/conv_check.cu)', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch22-s222-'));
  const exe = { emu: path.join(dir, 'emu'), race: path.join(dir, 'race') };
  let raceBuilt = false;
  const plusLine = snippet('scatter_conv_plain.cu').split('\n').findIndex((l) => l.includes('out[o] += filter[t] * v;')) + 1;

  it('builds as the lab runner builds emu and race', () => {
    const all =
      '#line 1 "scatter_conv_plain.cu"\n' + snippet('scatter_conv_plain.cu') + '\n' +
      '#line 1 "gather_conv.cu"\n' + snippet('gather_conv.cu') + '\n' +
      '#line 1 "conv_check.cu"\n' + fs.readFileSync(path.join(repo, SNIP, 'check/conv_check.cu'), 'utf8');
    fs.writeFileSync(path.join(dir, 'all.cu'), all);
    const t = spawnSync('python3', [path.join(repo, 'labs/tools/cu2emu.py'), path.join(dir, 'all.cu'), path.join(dir, 'all.cpp')], { encoding: 'utf8' });
    expect(t.status, t.stdout + t.stderr).toBe(0);
    const common = ['-std=c++17', '-O1', '-g', '-fno-omit-frame-pointer', '-pthread', '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'),
      '-include', 'pmpp/emu/cuda_emu.h', path.join(dir, 'all.cpp')];
    const c = spawnSync(cxx!, ['-fsanitize=address', '-DPMPP_BACKEND_EMU', ...common, '-o', exe.emu], { encoding: 'utf8' });
    expect(c.status, c.stderr).toBe(0);
    const r = spawnSync(cxx!, ['-fsanitize=thread', '-DPMPP_BACKEND_RACE', ...common, '-o', exe.race], { encoding: 'utf8' });
    raceBuilt = r.status === 0; // ThreadSanitizer is not available everywhere
  }, 240_000);

  it('emu: all three kernels are right in all three thread orders, the plain scatter included', () => {
    const r = spawnSync(exe.emu, ['all'], { encoding: 'utf8', env: ENV, timeout: 120_000 });
    expect(r.status, r.stderr).toBe(0);
    const lines = r.stdout.trim().split('\n');
    expect(lines[lines.length - 1]).toBe('DONE');
    const rows = lines.slice(0, -1);
    expect(rows).toEqual(
      ['gather', 'plain', 'atomic'].flatMap((k) => ['forward', 'reverse', 'shuffle'].map((o) => `${k} ${o}: wrong 0 of 48`)),
    );
  }, 120_000);

  // ThreadSanitizer's reports, as unordered pairs of lines in scatter_conv_plain.cu.
  const reported = (stderr: string) => {
    const pairs = new Set<string>();
    for (const rep of stderr.split('WARNING: ThreadSanitizer: data race').slice(1)) {
      const at = [...rep.matchAll(/scatter_conv_plain\.cu:(\d+)/g)].map((m) => Number(m[1]));
      if (at.length >= 2) pairs.add(at.slice(0, 2).sort((a, b) => a - b).join('-'));
    }
    return pairs;
  };

  it('race: ThreadSanitizer reports the plain scatter\'s += racing with itself', (ctx) => {
    if (!raceBuilt) return ctx.skip();
    // A clean run does not prove there is no race (labs/README.md); allow a few tries.
    let pairs = new Set<string>();
    for (let k = 0; k < 5 && !pairs.size; k++) {
      const r = spawnSync(exe.race, ['plain'], { encoding: 'utf8', env: ENV, timeout: 120_000 });
      expect([0, 66]).toContain(r.status);
      pairs = reported(r.stderr);
    }
    expect([...pairs]).toEqual([`${plusLine}-${plusLine}`]);
  }, 300_000);

  for (const k of ['gather', 'atomic'])
    it(`race: the ${k} kernel is right, with no report`, (ctx) => {
      if (!raceBuilt) return ctx.skip();
      const r = spawnSync(exe.race, [k], { encoding: 'utf8', env: ENV, timeout: 120_000 });
      expect(r.status, r.stderr.slice(0, 2000)).toBe(0);
      expect(r.stderr).not.toContain('ThreadSanitizer');
      expect(r.stdout).toContain(`${k} concurrent: wrong 0 of 48`);
    }, 120_000);
});
