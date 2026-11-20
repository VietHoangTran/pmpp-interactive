import fs from 'node:fs';
import path from 'node:path';
import type { ReactElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { ChoiceQ } from '../../components/Ask';
import { FLOAT_BYTES, N, boundaryTraffic, program, run } from '../../sim/offload';
import { bytesText, hostFunction, questionsS23 } from './S23';

// Section 2.3 shows two lab files, loaded from disk at run time, so there is no
// copy to drift. What can drift is what the lesson searches them for and what it
// says about them, and the numbers the narration speaks. This test ties those to
// the files and to the model (sim/offload.ts) the animation draws from.
const repo = path.resolve(__dirname, '../../../..');
const read = (rel: string) => fs.readFileSync(path.join(repo, rel), 'utf8');
const html = (node: unknown) => renderToStaticMarkup(node as ReactElement);
const correct = (q: ChoiceQ) => q.options.filter((o) => o.correct);

describe('the code section 2.3 shows', () => {
  it('the sequential version is host code only, with the loop of Fig. 2.4', () => {
    const src = read('labs/ch02/snippets/vecadd_sequential.cu');
    expect(src).toContain('C_h[i] = A_h[i] + B_h[i];');
    expect(src).toMatch(/for \(int i = 0; i < n; \+\+i\)/);
    expect(src).not.toMatch(/__global__|__device__|cuda[A-Z]/);
  });
  it("finds the host function of lab 2.1's starter, with its three parts still to write", () => {
    const src = read('labs/ch02/lab1_vecadd/starter/kernel.cu');
    const f = hostFunction(src)!;
    expect(f).not.toBeNull();
    expect(src.split('\n')[f.firstLine - 1]).toMatch(/^int vecAdd\(/);
    for (const part of ['Part 1.', 'Part 2.', 'Part 3.']) expect(f.code).toContain(part);
    expect(f.code).not.toContain('__global__'); // the kernel above it is left out
  });
  it("the corrected Fig. 2.5 declaration is the tested stub's", () => {
    const decl = program('cuda').head[2].trim();
    expect(decl).toBe('float *A_d, *B_d, *C_d;');
    expect(read('labs/ch02/snippets/vecadd_stub.cu')).toContain(decl);
  });
});

describe('the explorer', () => {
  it('prints sizes in binary units', () => {
    expect(bytesText(0)).toBe('0 bytes');
    expect(bytesText(2 ** 20 * FLOAT_BYTES)).toBe('4 MiB');
    expect(bytesText(9 * 4 * 2 ** 20)).toBe('36 MiB');
    expect(bytesText(2 ** 28 * FLOAT_BYTES)).toBe('1 GiB');
  });
  it('the default chain moves 36 MiB as round trips and 20 MiB resident, for the same 36 MiB of kernel traffic', () => {
    const n = 2 ** 20;
    expect(bytesText(boundaryTraffic(3, 'transparent', 'separate', n).bytesAcross)).toBe('36 MiB');
    expect(bytesText(boundaryTraffic(3, 'resident', 'separate', n).bytesAcross)).toBe('20 MiB');
    expect(bytesText(boundaryTraffic(3, 'resident', 'separate', n).kernelBytes)).toBe('36 MiB');
  });
});

describe('questions of section 2.3', () => {
  it('has the agreed ids, each choice question with exactly one right answer', () => {
    expect(Object.keys(questionsS23)).toEqual(['ch02/2.3/q-output-only', 'ch02/2.3/q-pays', 'ch02/2.3/q-interface']);
    for (const [id, q] of Object.entries(questionsS23)) {
      expect(q.id).toBe(id);
      expect(q.kind).toBe('choice');
      expect(correct(q as ChoiceQ)).toHaveLength(1);
      expect(q.source).toBeTruthy();
    }
  });
  it('q-output-only: A_d and B_d, the two arrays the stub copies in', () => {
    expect(html(correct(questionsS23['ch02/2.3/q-output-only'] as ChoiceQ)[0].label)).toContain('A_d</code> and <code>B_d');
    expect(program('cuda').calls.filter((c) => c.group === 'copy in').map((c) => c.id)).toEqual(['copy-in-A', 'copy-in-B']);
  });
  it('q-pays: 12 bytes moved per addition, as the explanation says', () => {
    expect(boundaryTraffic(1, 'transparent', 'separate', 1000).bytesAcross / 1000).toBe(12);
  });
});

describe('the narration of "Three parts, two memories"', () => {
  const script = (k: number) => read(`content/ch02/narration/round-trip-${k}.txt`).replace(/\[[^\]]*\]\s*/g, '').trim();
  it('has six scripts of 35 to 60 words', () => {
    for (let k = 1; k <= 6; k++) {
      const words = script(k).split(/\s+/).length;
      expect(words, `round-trip-${k}`).toBeGreaterThanOrEqual(35);
      expect(words, `round-trip-${k}`).toBeLessThanOrEqual(60);
    }
  });
  it('speaks the numbers the model computes', () => {
    const p = program('cuda');
    expect(script(3)).toContain(`${N} floats of ${FLOAT_BYTES} bytes, so ${N * FLOAT_BYTES} bytes per copy`);
    expect(script(3)).toContain(`${run(p, 5).bytesMoved} bytes have crossed before any arithmetic`);
    expect(script(5)).toContain(`${run(p, 7).bytesMoved - run(p, 6).bytesMoved} more bytes`);
    const end = run(p);
    expect(script(6)).toContain(`To do ${N} additions, ${end.bytesMoved} bytes crossed the boundary`);
    expect(script(6)).toContain(`${end.bytesMoved / N} bytes for every element`);
    const shared = run(program('metal', 'buffers'));
    expect([shared.bytesMoved, shared.frees]).toEqual([0, 0]);
    expect(script(6)).toMatch(/Nothing crosses\. Nothing is freed\./);
  });
});
