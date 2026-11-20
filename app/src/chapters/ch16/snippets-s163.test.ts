import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { INF } from '../../sim/floydwarshall';

// Section 16.3 shows FW_bottomup and its host loop. The checked file is
// labs/ch16/snippets/fw_bottomup.cu, which ./verify.sh front-end-checks and
// labs/ch16/snippets/check.py runs on the emulator. This test fails if a lesson copy
// drifts from it, and pins every line number the stepper highlights.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'S163.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in S163.tsx`);
  return m[1].trim();
};
const file = fs.readFileSync(path.join(repo, 'labs/ch16/snippets/fw_bottomup.cu'), 'utf8');
const KERNEL = lesson('FW_KERNEL');
const HOST = lesson('FW_HOST');
const line = (code: string, n: number) => code.split('\n')[n - 1];
const lines = (() => {
  const m = /export const FW_LINES = (\{[^}]*\});/.exec(src);
  if (!m) throw new Error('no FW_LINES in S163.tsx');
  return JSON.parse(m[1].replace(/(\w+):/g, '"$1":')) as Record<string, number>;
})();

describe('section 16.3 shows the checked file', () => {
  it('the kernel and the host loop each appear verbatim in the file', () => {
    expect(file).toContain(KERNEL);
    expect(file).toContain(HOST);
  });
  it('the file is those two listings, one include and comments, nothing else', () => {
    const rest = file.replace(KERNEL, '').replace(HOST, '');
    for (const l of rest.split('\n')) expect(l.trim() === '' || l.trim().startsWith('//') || l.trim() === '#include <climits>').toBe(true);
  });
});

describe('the corrections the page describes', () => {
  it('the sentinel is the integer INT_MAX / 2, the same value as the model\'s', () => {
    expect(line(KERNEL, 1)).toContain('#define FW_INF (INT_MAX / 2)');
    expect(INF).toBe(Math.floor(2147483647 / 2));
    expect(KERNEL).not.toContain('INFINITY');
  });
  it('no return before the barrier: the only return comes after __syncthreads()', () => {
    const k = KERNEL.split('\n').map((l) => l.replace(/\/\/.*$/, '')); // code only, not comments
    const barrier = k.findIndex((l) => l.includes('__syncthreads();'));
    const returns = k.flatMap((l, i) => (/\breturn\b/.test(l) ? [i] : []));
    expect(barrier).toBeGreaterThan(0);
    expect(returns.length).toBe(1);
    expect(returns[0]).toBeGreaterThan(barrier);
    expect(KERNEL).toContain('bool inRange = col < n_vertices;');
  });
  it('the host loop does not wait between launches', () => {
    expect(HOST).not.toContain('cudaDeviceSynchronize');
    expect(HOST).toContain('FW_bottomup<<<dimGrid, dimBlock>>>(k, dist_d, n_vertices);');
  });
});

describe('the lines the stepper highlights', () => {
  it('point at the statements the stepper describes', () => {
    expect(line(KERNEL, lines.read)).toContain('dist_k_col = dist[n_vertices * row + k];');
    expect(line(KERNEL, lines.barrier)).toContain('__syncthreads();');
    expect(line(KERNEL, lines.retire)).toContain('if (dist_k_col == FW_INF) return;');
    expect(line(KERNEL, lines.rowK)).toContain('int dist_k_row = dist[k * n_vertices + col];');
    expect(line(KERNEL, lines.compare)).toContain('if (new_distance < dist[distIndex])');
    expect(line(KERNEL, lines.write)).toContain('dist[distIndex] = new_distance;');
  });
});
