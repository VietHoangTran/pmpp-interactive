import fs from 'node:fs';
import path from 'node:path';
import type { ReactElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { ChoiceQ } from '../../components/Ask';
import { DEFECTS, program, run } from '../../sim/offload';
import { HUNT_ROUNDS, checkMacro, questionsS24, stubLines } from './S24';

// Section 2.4 shows the stub (labs/ch02/snippets/vecadd_stub.cu) and the CHECK
// macro of lab 2.1's solution, both loaded from disk at run time. This test ties
// what the lesson says about them, and the answers its questions give, to the
// files and to the model the stepper and the bug hunt run on.
const repo = path.resolve(__dirname, '../../../..');
const read = (rel: string) => fs.readFileSync(path.join(repo, rel), 'utf8');
const html = (node: unknown) => renderToStaticMarkup(node as ReactElement);
const right = (id: string) => html((questionsS24[id] as ChoiceQ).options.find((o) => o.correct)!.label);

describe('the stub and the macro section 2.4 shows', () => {
  const stub = read('labs/ch02/snippets/vecadd_stub.cu');
  const lines = stub.split('\n');
  it('finds Parts 1 and 3 and the launch in the stub file', () => {
    const w = stubLines(stub)!;
    expect(w).toEqual({ part1: [13, 21], launch: 24, part3: [27, 30] });
    expect(lines[w.part1[0] - 1]).toContain('int size = n * sizeof(float);');
    expect(lines[w.part1[1] - 1]).toContain('cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);');
    expect(lines[w.launch - 1]).toContain('vecAddKernel<<<');
    expect(lines[w.part3[0] - 1]).toContain('cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);');
    expect(lines[w.part3[1] - 1]).toContain('cudaFree(C_d);');
  });
  it('gives up rather than point at the wrong lines when the file changes shape', () => {
    expect(stubLines('int main() { return 0; }')).toBeNull();
  });
  it('the CHECK macro of lab 2.1 tests err and returns 1, as the caption says', () => {
    const src = read('labs/ch02/lab1_vecadd/solution/kernel.cu');
    const m = checkMacro(src)!;
    expect(m.code.startsWith('#define CHECK(call)')).toBe(true);
    expect(src.split('\n')[m.firstLine - 1]).toMatch(/^#define CHECK\(/);
    expect(m.code).toContain('err != cudaSuccess');
    expect(m.code).toContain('cudaGetErrorString(err)');
    expect(m.code).toMatch(/return 1;/);
    expect(m.code).not.toContain('exit(');
    expect(m.code.trim().endsWith('while (0)')).toBe(true);
  });
});

describe('questions of section 2.4', () => {
  it('has the agreed ids, each with exactly one right answer and a source', () => {
    expect(Object.keys(questionsS24)).toEqual([
      'ch02/2.4/q-malloc-size',
      'ch02/2.4/q-malloc-ptr',
      'ch02/2.4/q-memcpy',
      'ch02/2.4/q-out-param',
      'ch02/2.4/q-not-cleared',
      'ch02/2.4/q-shared-wait',
    ]);
    for (const [id, q] of Object.entries(questionsS24)) {
      expect(q.id).toBe(id);
      expect((q as ChoiceQ).options.filter((o) => o.correct)).toHaveLength(1);
      expect(q.source).toBeTruthy();
    }
  });
  it('the book exercises have the answers of the notes', () => {
    expect(right('ch02/2.4/q-malloc-size')).toBe('<code>v * sizeof(int)</code>');
    expect(right('ch02/2.4/q-malloc-ptr')).toBe('<code>(void **) &amp;A_d</code>');
    expect(right('ch02/2.4/q-memcpy')).toBe('<code>cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);</code>');
  });
  it('q-not-cleared and q-shared-wait agree with the model', () => {
    // fresh memory: uncleared on CUDA, zeros on Metal
    expect(run(program('cuda'), 3).device.C!.cells.every((c) => c === 'uncleared')).toBe(true);
    expect(run(program('metal'), 3).device.C!.cells.every((c) => c === 0)).toBe(true);
    expect(right('ch02/2.4/q-not-cleared')).toContain('On CUDA whatever that device memory held before. On Metal zeros.');
    // a read after commit and before the wait is possible and not the result
    const p = program('metal');
    const commit = p.calls.findIndex((c) => c.id === 'launch') + 1;
    expect(run(p, commit).hostRead.status).toBe('stale');
    expect(right('ch02/2.4/q-shared-wait')).toContain('wait for the command buffer to complete');
  });
  it('p-copies: 0 copies and 0 frees on Metal with shared buffers, against 3 and 3', () => {
    const m = run(program('metal', 'buffers'));
    const c = run(program('cuda'));
    expect([m.copies, m.frees, c.copies, c.frees]).toEqual([0, 0, 3, 3]);
  });
});

describe('the bug hunt', () => {
  it('plays six rounds drawn from the seven defects', () => {
    expect(HUNT_ROUNDS).toBe(6);
    expect(DEFECTS.length).toBeGreaterThanOrEqual(HUNT_ROUNDS);
  });
});
