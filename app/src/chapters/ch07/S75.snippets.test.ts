import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { cxx, python, runOnEmulator } from './snippetEmu';

// Section 7.5's stepper shows labs/ch07/snippets/s75_halo_toy.cu (the constant
// HALO_TOY in S75.tsx) and highlights its lines by number (the constant LINE).
// This fails if the copy drifts from the checked file, or if a line number stops
// pointing at the statement it names.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'S75.tsx'), 'utf8');
const lesson = (() => {
  const m = /const HALO_TOY = `([\s\S]*?)`;/.exec(src);
  if (!m) throw new Error('no HALO_TOY in S75.tsx');
  return m[1].trim();
})();
const file = fs.readFileSync(path.join(repo, 'labs/ch07/snippets/s75_halo_toy.cu'), 'utf8').trim();
const nums = (() => {
  const m = /export const LINE = \{([^}]*)\}/.exec(src);
  if (!m) throw new Error('no LINE in S75.tsx');
  return Object.fromEntries(m[1].split(',').map((kv) => kv.split(':').map((x) => x.trim())).map(([k, v]) => [k, Number(v)]));
})();
const kernelEnd = Number(/export const KERNEL_END = (\d+);/.exec(src)?.[1]);
const line = (n: number) => file.split('\n')[n - 1];

describe('section 7.5 halo-kernel snippet', () => {
  it('the lesson copy is the checked file', () => {
    expect(lesson).toBe(file);
  });
  it('the highlighted lines are the statements the stepper names', () => {
    expect(line(nums.test)).toContain('if (row >= 0 && row < height && col >= 0 && col < width)');
    expect(line(nums.loadIn)).toContain('N_s[ty][tx] = N[row*width + col];');
    expect(line(nums.ghost)).toContain('N_s[ty][tx] = 0.0f;');
    expect(line(nums.barrier)).toContain('__syncthreads();');
    expect(line(nums.outside)).toContain('if (col >= 0 && col < width && row >= 0 && row < height)');
    expect(line(nums.ring)).toContain('if (tileCol >= 0 && tileCol < OUT_TILE_DIM &&');
    expect(line(nums.ring + 1)).toContain('tileRow >= 0 && tileRow < OUT_TILE_DIM');
    expect(line(nums.tap)).toContain('Pvalue += F_c[fRow][fCol]');
    expect(line(nums.tap + 1)).toContain('N_s[tileRow+fRow][tileCol+fCol]');
    expect(line(nums.store)).toContain('P[row*width + col] = Pvalue;');
  });
  it('the stepper shows the kernel through its closing brace and stops before the launch code', () => {
    expect(line(kernelEnd)).toBe('}');
    expect(file.split('\n').slice(0, kernelEnd).join('\n')).not.toContain('launchHaloToy');
  });
  it('the corrections the lesson claims are in the file', () => {
    expect(file).toContain('__shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];');
    expect(file).toContain('__constant__ float F_c[');
    expect(file).not.toMatch(/[^_]F\[/);
    expect(file).not.toMatch(/= 0\.0;/);
    expect(file).toContain('int tx = threadIdx.x, ty = threadIdx.y;');
  });
});

// The snippet's "emulator-checked" status rests on this run: labs/ch07/snippets/check/
// s75_halo_toy_test.cu builds the file as the lab runner builds the emu backend
// (AddressSanitizer on) and runs it on six image shapes in the three thread orders.
// Skipped when no C++ compiler or python3 is found.
describe.skipIf(!cxx || !python)('section 7.5 halo-kernel snippet on the course emulator', () => {
  it('every output of every shape is right in all three thread orders, and nothing is reported', () => {
    const r = runOnEmulator('s75_halo_toy_test.cu');
    expect(r.status, r.stderr).toBe(0);
    const lines = r.stdout.split('\n').filter((l) => l.startsWith('halo '));
    expect(lines.length).toBe(6 * 3);
    for (const l of lines) {
      const m = /^halo (\d+)x(\d+) (in order|reversed|shuffled): right (\d+) of (\d+)$/.exec(l);
      expect(m, l).not.toBeNull();
      expect(Number(m![4]), l).toBe(Number(m![1]) * Number(m![2]));
      expect(Number(m![5])).toBe(Number(m![1]) * Number(m![2]));
    }
    expect(r.stderr).not.toMatch(/AddressSanitizer|undefined/);
  }, 300000);
});
