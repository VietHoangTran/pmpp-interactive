import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { blockCount, ex9, launchWith } from '../../sim/launch';

// Section 2.6 shows three pieces of code. Two are lab files loaded from disk at run
// time (labs/ch02/lab1_vecadd/solution/kernel.cu and launch.mm), so there is no
// copy to drift; what can drift is the text the lesson searches them for. The third,
// the book's exercise 9, is copied into S26.tsx from labs/ch02/snippets/ex9_foo.cu,
// which the CUDA front-end check parses. This test ties the three to the lesson.
const repo = path.resolve(__dirname, '../../../..');
const read = (rel: string) => fs.readFileSync(path.join(repo, rel), 'utf8');
const lesson = fs.readFileSync(path.join(__dirname, 'S26.tsx'), 'utf8');
const constant = (name: string) => {
  const m = new RegExp('const ' + name + ' = `([\\s\\S]*?)`;').exec(lesson);
  if (!m) throw new Error(`no ${name} in S26.tsx`);
  return m[1];
};

describe('book exercise 9 as the lesson shows it', () => {
  const file = read('labs/ch02/snippets/ex9_foo.cu').trim();
  it('is labs/ch02/snippets/ex9_foo.cu, character for character', () => {
    expect(constant('EX9').trim()).toBe(file);
  });
  it('has the index on line 2, the assignment on line 4 and the launch on line 9, as the predictions say', () => {
    const l = file.split('\n');
    expect(l[1]).toContain('unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;');
    expect(l[3]).toContain('b[i] = 2.7f * a[i] - 4.3f;');
    expect(l[7]).toContain('unsigned int N = 200000;');
    expect(l[8]).toContain('foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d, N);');
  });
  it('is what the model counts: N = 200000 in blocks of 128', () => {
    const e = ex9();
    expect([e.N, e.threadsPerBlock]).toEqual([200000, 128]);
    expect(blockCount('int-ceil', e.N, e.threadsPerBlock).steps.map((s) => s.text)).toEqual(['200000 + 128 - 1', '200127 / 128']);
  });
});

describe('lab 2.1 as the lesson reads it', () => {
  const cu = read('labs/ch02/lab1_vecadd/solution/kernel.cu').split('\n');
  const at = cu.findIndex((l) => l.includes('vecAddKernel<<<'));
  it('kernel.cu: the launch is followed by the two lines the lesson marks', () => {
    expect(cu.filter((l) => l.includes('vecAddKernel<<<'))).toHaveLength(1);
    expect(cu[at]).toContain('vecAddKernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, n);');
    expect(cu[at - 1]).toContain('int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;');
    expect(cu[at + 1]).toContain('CHECK(cudaGetLastError());');
    expect(cu[at + 2]).toContain('CHECK(cudaDeviceSynchronize());');
  });
  it('kernel.cu: the differences from the page that the lesson lists are in the file', () => {
    const src = cu.join('\n');
    expect(src).toContain('int vecAdd(const float* A_h, const float* B_h, float* C_h, int n, int threadsPerBlock)');
    expect(src).toContain('size_t size = (size_t)n * sizeof(float);');
    expect(src).toContain('return 1;');
    for (const call of ['cudaMalloc', 'cudaMemcpy', 'cudaFree', 'cudaGetLastError', 'cudaDeviceSynchronize']) {
      const uses = cu.filter((l) => l.includes(call + '(') && !l.trimStart().startsWith('//'));
      expect(uses.length).toBeGreaterThan(0);
      expect(uses.every((l) => l.includes('CHECK(' + call + '('))).toBe(true);
    }
  });
  it('launch.mm: every step of the table is found, once per call', () => {
    const mm = read('labs/ch02/lab1_vecadd/solution/launch.mm').split('\n');
    const count = (needle: string) => mm.filter((l) => l.includes(needle)).length;
    expect(count('id<MTLCommandQueue> queue')).toBe(1); // the queue is a parameter of vecAdd
    expect(mm.find((l) => l.includes('id<MTLCommandQueue> queue'))!.startsWith('int vecAdd(')).toBe(true);
    expect(count('[queue commandBuffer]')).toBe(1);
    expect(count('computeCommandEncoder]')).toBe(1);
    expect(count('setComputePipelineState:')).toBe(1);
    expect(count('setBuffer:')).toBe(3);
    expect(count('dispatchThreads:')).toBe(1);
    expect(count('endEncoding]')).toBe(1);
    expect(count('commandBuffer commit]')).toBe(1);
    expect(count('waitUntilCompleted]')).toBe(1);
    expect(count('commandBuffer.status')).toBe(1);
    expect(count('setBytes:')).toBe(0); // the lesson says this kernel needs no scalar
    expect(mm.some((l) => l.startsWith('int vecAdd('))).toBe(true);
    // The needles the lesson searches for are these same strings.
    for (const needle of ['id<MTLCommandQueue> queue', '[queue commandBuffer]', 'computeCommandEncoder]', 'setComputePipelineState:', 'setBuffer:', 'dispatchThreads:', 'endEncoding]', 'commandBuffer commit]', 'waitUntilCompleted]', 'commandBuffer.status'])
      expect(lesson).toContain(`'${needle}'`);
  });
  it('main.cpp tests the lengths the bug hunt names', () => {
    expect(read('labs/ch02/lab1_vecadd/main.cpp')).toContain('std::vector<int> lengths = {1, 31, 32, 33, 255, 256, 257, 1000, 4097};');
    expect(lesson).toContain('n = 1, 31, 32, 33, 255, 256, 257,');
    // Each of the three wrong counts fails, or wastes a block, at one of those lengths with the book's block size.
    for (const key of ['trunc', 'ceil-trunc'] as const) expect(launchWith(key, 255, 256).correct).toBe(false);
    expect(launchWith('plus-one', 256, 256).emptyBlocks).toBe(1);
  });
});
