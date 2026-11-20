import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { ceilDiv, launch } from '../../sim/launch';

// The summary page shows two kernels of its own, each for one question. Their checked
// copies are in labs/ch02/snippets/, which the CUDA front-end check parses. This test
// fails if a copy on the page and its file drift apart, and pins the numbers the
// page's questions state.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'Wrap.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in Wrap.tsx`);
  return m[1].trim();
};
const snippet = (name: string) => fs.readFileSync(path.join(repo, 'labs/ch02/snippets', name), 'utf8').trim();

describe('summary page: the listings are the checked snippets', () => {
  it('saxpy is labs/ch02/snippets/saxpy_stub_defect.cu', () => {
    expect(lesson('SAXPY')).toBe(snippet('saxpy_stub_defect.cu'));
  });
  it('countAbove is labs/ch02/snippets/count_above_race.cu', () => {
    expect(lesson('COUNT')).toBe(snippet('count_above_race.cu'));
  });

  it('the saxpy stub reads y in the kernel and never copies y_h to the device', () => {
    const f = snippet('saxpy_stub_defect.cu');
    expect(f).toContain('y[i] = a * x[i] + y[i];');
    expect(f).toContain('cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);');
    expect(f).not.toContain('cudaMemcpy(y_d, y_h');
    // Everything else is right: the integer ceiling, the guard, the copy back, the byte count.
    expect(f).toContain('saxpyKernel<<<(n + 255) / 256, 256>>>');
    expect(f).toContain('if (i < n)');
    expect(f).toContain('cudaMemcpy(y_h, y_d, size, cudaMemcpyDeviceToHost);');
    expect(f).toContain('size_t size = (size_t)n * sizeof(float);');
  });

  it('countAbove updates one int from every passing thread, with no atomic', () => {
    const f = snippet('count_above_race.cu');
    const lines = f.split('\n');
    // The page says ThreadSanitizer named "the line of the update": line 5 of the file.
    expect(lines[4].trim()).toBe('*count = *count + 1;');
    expect(f).not.toMatch(/atomic/i);
  });
});

describe('summary page: the numbers its questions state', () => {
  it('50,257 logits in blocks of 256: 197 blocks, 50,432 threads, 175 stop at the guard', () => {
    const l = launch(50257, 256, ceilDiv(50257, 256));
    expect([l.blocks, l.threads, l.working, l.surplus]).toEqual([197, 50432, 50257, 175]);
    expect(l.correct).toBe(true);
    // The page's prompt and answer come from the same model.
    expect(src).toContain('const LOGITS = launch(VOCAB, 256, ceilDiv(VOCAB, 256));');
    expect(src).toContain('answer: LOGITS.surplus');
  });

  it('a 150,000 by 4,096 float table: the index fits in an int, the byte count does not', () => {
    const n = 150000 * 4096;
    const bytes = n * 4;
    const INT_MAX = 2 ** 31 - 1;
    expect(n).toBe(614_400_000);
    expect(n - 1).toBeLessThan(INT_MAX);
    expect(bytes).toBe(2_457_600_000);
    expect(bytes).toBeGreaterThan(INT_MAX);
    // C++20: conversion to a 32-bit int is modulo 2^32.
    expect(new Int32Array([bytes])[0]).toBe(-1_837_367_296);
    expect(bytes - 2 ** 32).toBe(-1_837_367_296);
    // Lab 2.1's solution says an int byte count overflows once n reaches 2^29 floats.
    expect(2 ** 29 * 4).toBeGreaterThan(INT_MAX);
    expect((2 ** 29 - 1) * 4).toBeLessThanOrEqual(INT_MAX);
    expect(fs.readFileSync(path.join(repo, 'labs/ch02/lab1_vecadd/solution/kernel.cu'), 'utf8')).toContain('an int overflows\n//     once n reaches 2^29 floats');
  });

  it('every question has its own id under ch02/wrap/ and a source', () => {
    const ids = [...src.matchAll(/^ {2}'(ch02\/wrap\/q-[a-z-]+)': \{\n {4}kind: '(choice|numeric)',\n {4}id: '([^']+)'/gm)];
    expect(ids.map((m) => m[1])).toEqual(['ch02/wrap/q-inout', 'ch02/wrap/q-logits', 'ch02/wrap/q-count', 'ch02/wrap/q-bytes-int', 'ch02/wrap/q-overlap']);
    expect(ids.every((m) => m[1] === m[3])).toBe(true);
    expect((src.match(/^ {4}source: '/gm) ?? []).length).toBe(5);
  });
});
