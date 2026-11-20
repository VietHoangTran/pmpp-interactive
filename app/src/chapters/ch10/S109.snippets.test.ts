import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { SCENE, coarsenSteps, sceneData, segments, sweepC, CH6_DEVICE_R } from '../../sim/multiblock';
import { functionSource } from '../../sim/toolchain';
import { coarseningLines } from './S1010';

// Sections 10.9 and 10.10 show lab 10.1's files, loaded from disk at run time, so there
// is no copy to drift. What can drift is what the lessons say about those files, the
// benchmark rows their reveals read, the numbers in the commit-order narration, and the
// worked numbers in the prose. This test ties each to its source.
const repo = path.resolve(__dirname, '../../../..');
const read = (rel: string) => fs.readFileSync(path.join(repo, rel), 'utf8');
const LAB = 'labs/ch10/lab1_sum';
const lesson109 = fs.readFileSync(path.join(__dirname, 'S109.tsx'), 'utf8');
const lesson1010 = fs.readFileSync(path.join(__dirname, 'S1010.tsx'), 'utf8');
const flat = (s: string) => s.replace(/\s+/g, ' ');

describe('lab 10.1 kernel.cu, as sections 10.9 and 10.10 show it', () => {
  const cu = read(`${LAB}/solution/kernel.cu`);
  const k = functionSource(cu, 'CoarsenedSumReductionKernel');
  const host = functionSource(cu, 'launchCoarsenedSum');
  it('the kernel and its host function are found', () => {
    expect(k?.quals).toBe('global');
    expect(host).not.toBeNull();
  });
  it('commits with the book\'s spelling, and with the legacy call under HIP (the three lines 10.9 marks)', () => {
    expect(k!.text).toContain('cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);');
    expect(k!.text).toContain('output_ref.fetch_add(partialSum, cuda::memory_order_relaxed);');
    expect(k!.text).toMatch(/#if defined\(__HIPCC__\)\s*atomicAdd\(output, partialSum\);\s*#else\s*cuda::atomic_ref/);
    const marked = k!.text.split('\n').filter((l) => ['atomic_ref', 'fetch_add', 'atomicAdd'].some((n) => l.includes(n)));
    expect(marked.map((l) => l.trim())).toEqual([
      'atomicAdd(output, partialSum);',
      'cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);',
      'output_ref.fetch_add(partialSum, cuda::memory_order_relaxed);',
    ]);
    // <cuda/atomic> is included only where the commit uses it: not under HIP.
    expect(cu).toMatch(/#if !defined\(__HIPCC__\)\s*#include <cuda\/atomic>\s*#endif/);
    expect(read(`${LAB}/starter/kernel.cu`)).toMatch(/#if !defined\(__HIPCC__\)\s*#include <cuda\/atomic>\s*#endif/);
    expect(read('labs/ch10/lab2_reducehunt/solution/kernel.cu')).toMatch(
      /#if !defined\(__HIPCC__\)\s*#include <cuda\/atomic>\s*#endif[\s\S]*#if defined\(__HIPCC__\)\s*atomicAdd\(out, blockSum\);\s*#else/,
    );
  });
  it('the host zeroes the total before every launch (q-twice) and launches ceil(n / (2·C·block)) blocks', () => {
    expect(host!.text.split('\n').filter((l) => l.includes('emset'))).toHaveLength(1);
    expect(host!.text).toContain('cudaMemset(out, 0, sizeof(float));');
    expect(host!.text).toContain('((unsigned)n + perBlock - 1) / perBlock');
  });
  it('differs from Fig. 10.20 as 10.10 says: void, N with guarded loads, a run-time factor, blockDim.x, the reduction written out, a typed mask', () => {
    expect(flat(k!.text)).toContain('__global__ void CoarsenedSumReductionKernel(const float* input, float* output, unsigned int N, unsigned int coarse)');
    expect(k!.text).toContain('unsigned int segment = coarse * 2 * blockDim.x * blockIdx.x;');
    expect(k!.text).toContain('float partialSum = i < N ? input[i] : 0.0f;');
    expect(k!.text).toContain('for (unsigned int c = 1; c < coarse * 2; ++c) {');
    expect(k!.text).toContain('unsigned int idx = i + c * blockDim.x;');
    expect(k!.text).toContain('if (idx < N) {');
    expect(k!.text).toContain('partialSum = warp_reduce(partialSum);');
    expect(k!.text).not.toMatch(/\bBLOCK_DIM\b/);
    expect(cu).toMatch(/#if defined\(__HIPCC__\)\s*typedef unsigned long long lane_mask_t;/);
    expect(flat(lesson1010)).toContain('A lane mask typed per vendor, and the legacy <code>atomicAdd</code> in place of <code>cuda::atomic_ref</code> under HIP');
  });
  it('the lines 10.10 marks are the segment, the first load and the loop, brace to brace', () => {
    const lines = k!.text.split('\n');
    const marked = coarseningLines(k!.text).map((n) => lines[n - 1].trim());
    expect(marked[0]).toBe('unsigned int segment = coarse * 2 * blockDim.x * blockIdx.x;');
    expect(marked[1]).toBe('float partialSum = i < N ? input[i] : 0.0f;');
    expect(marked[2]).toBe('for (unsigned int c = 1; c < coarse * 2; ++c) {');
    expect(marked[marked.length - 1]).toBe('}');
    expect(marked).toHaveLength(8);
  });
});

describe('lab 10.1 kernel.metal, the Metal tab of 10.9', () => {
  const metal = read(`${LAB}/solution/kernel.metal`);
  const f = functionSource(metal, 'coarsened_sum');
  it('one kernel commits with atomic_fetch_add_explicit on a device atomic_float*, relaxed', () => {
    const withAdd = [...metal.matchAll(/\bkernel\s+void\s+(\w+)\s*\(/g)].filter((m) => functionSource(metal, m[1])!.text.includes('atomic_fetch_add_explicit'));
    expect(withAdd.map((m) => m[1])).toEqual(['coarsened_sum']);
    expect(f!.text).toContain('device atomic_float* output');
    expect(f!.text).toContain('atomic_fetch_add_explicit(output, s, memory_order_relaxed);');
  });
});

describe('lab 10.1 main.cpp: the benchmark rows the two reveals read', () => {
  const main = flat(read(`${LAB}/main.cpp`));
  it('determinism: 20 runs on 2^24 floats, rows named "determinism" with runs, distinct and n', () => {
    expect(main).toContain('size_t n = (size_t)1 << 24;');
    expect(main).toContain('const int runs = 20;');
    expect(main).toContain('pmpp::bench_row(a, "determinism").n("n", (double)n).n("runs", runs).n("distinct", (double)bits.size())');
    expect(lesson109).toContain('20 times on the same 2<sup>24</sup> floats');
  });
  it('the sweep: powers of two from 1 to 128, starting at 2^26 floats, 256-thread blocks, rows named "any" with coarse and gb_per_s', () => {
    expect(main).toContain('size_t n = (size_t)1 << 26;');
    expect(main).toContain('for (int coarse = 1; coarse <= 128; coarse *= 2)');
    expect(main).toContain('const int block = benchBlock(256, false);');
    expect(main).toContain('pmpp::bench_row(a, "any").n("n", (double)n).n("block", block).n("coarse", coarse)');
    expect(main).toContain('.n("gb_per_s", gbps)');
    expect(flat(lesson1010)).toContain('for the factors 1, 2, 4, and so on up to 128');
  });
});

describe('the commit-order narration agrees with the model', () => {
  const script = (n: number) => read(`content/ch10/narration/commit-order-${n}.txt`).trim();
  const words = (s: string) => s.replace(/\[[^\]]*\]/g, '').split(/\s+/).filter(Boolean).length;
  it('five scripts of 35 to 60 words, with no symbols read aloud', () => {
    for (let n = 1; n <= 5; n++) {
      expect(words(script(n))).toBeGreaterThanOrEqual(35);
      expect(words(script(n))).toBeLessThanOrEqual(60);
      expect(script(n)).not.toMatch(/[=×*+/<>%]/);
    }
  });
  it('part 1: 4 threads, segments of 8, 60 logits, 8 blocks', () => {
    const s = sceneData().seg;
    expect([SCENE.block, s.perBlock, SCENE.N, s.blocks]).toEqual([4, 8, 60, 8]);
    expect(script(1)).toContain('With 4 threads, a segment holds 8 logits. 60 logits need 8 blocks');
  });
  it('part 5: a million logits with 1024-thread blocks need 489 floats', () => {
    expect(segments(1_000_000, 1024).blocks).toBe(489);
    expect(script(5)).toContain('with blocks of 1024 threads, it is 489');
    expect(script(5)).toContain('that is 8 floats');
  });
});

describe('worked numbers in 10.10\'s prose', () => {
  it('a million floats on 264 slots: C = 7 launches 293 blocks, 29 in the second wave; C = 8 launches 256, 8 slots idle', () => {
    const sw = sweepC({ N: 1 << 20, block: 256, R: CH6_DEVICE_R }, 16);
    expect(sw[6]).toMatchObject({ blocks: 293, waves: 2, lastWave: 29 });
    expect(sw[7]).toMatchObject({ blocks: 256, waves: 1 });
    expect(CH6_DEVICE_R - sw[7].blocks).toBe(8);
    const text = flat(lesson1010);
    expect(text).toContain('it picks C = 7, which launches 293 blocks: one full wave and a second wave of 29. C = 8 launches 256 and finishes in one wave, with 8 slots idle.');
  });
  it('Fig. 10.21 in the prose: 8 steps (6 underused) against 6', () => {
    expect(coarsenSteps({ N: 32, block: 8, C: 1, R: 1 })).toMatchObject({ totalSteps: 8, underSteps: 6 });
    expect(coarsenSteps({ N: 32, block: 8, C: 2, R: 1 }).totalSteps).toBe(6);
    expect(flat(lesson1010)).toContain('That is 8 steps, 6 of them underused. One block coarsened by two does the same work in 6');
  });
});
