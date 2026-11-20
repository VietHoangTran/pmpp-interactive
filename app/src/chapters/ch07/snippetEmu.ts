import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// For chapter 7's tests only (never imported by a page): build a driver from
// labs/ch07/snippets/check/ together with the snippet it includes, as the lab runner builds
// the `emu` backend (labs/tools/cu2emu.py, pmpp/emu/cuda_emu.h force-included,
// AddressSanitizer on), run it, and hand back what it printed. The snippet is inlined in
// place of the driver's `#include "../<snippet>.cu"`, with #line markers, so that
// cu2emu.py translates both files and any report names the original lines.

export const repo = path.resolve(__dirname, '../../../..');
const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
export const cxx = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
export const python = has('python3', ['--version']) ? 'python3' : undefined;

const SNIPPETS = path.join(repo, 'labs/ch07/snippets');
const INC = path.join(repo, 'labs/common/include');
const FLAGS = ['-std=c++17', '-O1', '-g', '-fsanitize=address', '-fno-omit-frame-pointer', '-pthread', '-DPMPP_BACKEND_EMU',
  '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h'];

export function runOnEmulator(driver: string): { status: number | null; stdout: string; stderr: string } {
  const drv = path.join(SNIPPETS, 'check', driver);
  const lines = fs.readFileSync(drv, 'utf8').split('\n');
  const at = lines.findIndex((l) => /^#include "\.\.\/[\w.]+\.cu"$/.test(l));
  if (at < 0) throw new Error(`${driver} includes no snippet`);
  const snip = path.join(SNIPPETS, /"\.\.\/([\w.]+\.cu)"/.exec(lines[at])![1]);
  const merged = [
    `#line 1 "${drv}"`,
    ...lines.slice(0, at),
    `#line 1 "${snip}"`,
    fs.readFileSync(snip, 'utf8'),
    `#line ${at + 2} "${drv}"`,
    ...lines.slice(at + 1),
  ].join('\n');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch07-snippet-'));
  const stem = path.basename(driver, '.cu');
  const cu = path.join(dir, stem + '.cu');
  const cpp = path.join(dir, stem + '.cpp');
  const exe = path.join(dir, stem);
  fs.writeFileSync(cu, merged);
  const t = spawnSync(python!, [path.join(repo, 'labs/tools/cu2emu.py'), cu, cpp], { encoding: 'utf8' });
  if (t.status !== 0) throw new Error('cu2emu failed on ' + driver + ':\n' + t.stdout + t.stderr);
  const c = spawnSync(cxx!, [...FLAGS, '-I', SNIPPETS, cpp, '-o', exe], { encoding: 'utf8' });
  if (c.status !== 0) throw new Error('compile failed for ' + driver + ':\n' + c.stderr);
  const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 300000 });
  return { status: r.status, stdout: r.stdout ?? '', stderr: r.stderr ?? '' };
}
