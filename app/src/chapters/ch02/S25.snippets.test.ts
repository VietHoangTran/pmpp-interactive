import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { counts, coverage, mapping, span } from '../../sim/grid1d';
import { functionSource } from '../../sim/toolchain';

// Section 2.5 shows lab files loaded from disk at run time, so there is no copy of
// them to drift; what can drift is the text the lesson finds in them and the claims
// it makes about them. It also quotes two lines of AMD's example kernel and three
// lines of llama.cpp from notes/today/ch02-facts.md, and states numbers in its
// narration scripts. This test ties all of these to their sources.
const repo = path.resolve(__dirname, '../../../..');
const read = (rel: string) => fs.readFileSync(path.join(repo, rel), 'utf8');
const lesson = fs.readFileSync(path.join(__dirname, 'S25.tsx'), 'utf8');
const facts = read('notes/today/ch02-facts.md');
const constant = (name: string) => {
  const m = new RegExp('const ' + name + ' = `([\\s\\S]*?)`;').exec(lesson);
  if (!m) throw new Error(`no ${name} in S25.tsx`);
  return m[1];
};
/** The facts file's text for one item, from its heading to the next. */
const item = (id: string) => {
  const at = facts.indexOf(`### ${id}.`);
  if (at < 0) throw new Error(`no item ${id} in ch02-facts.md`);
  const next = facts.indexOf('\n### ', at + 4);
  return facts.slice(at, next < 0 ? undefined : next);
};

describe('lab 2.1 kernel.cu, as the stepper and the listing read it', () => {
  const cu = read('labs/ch02/lab1_vecadd/solution/kernel.cu');
  const k = functionSource(cu, 'vecAddKernel');
  const marks = [...(/const LINE_MARKS = \[(.*)\];/.exec(lesson)?.[1] ?? '').matchAll(/'([^']*)'/g)].map((m) => m[1]);
  it('the kernel is found, and it is a __global__ function', () => {
    expect(k).not.toBeNull();
    expect(k!.quals).toBe('global');
  });
  it('the three lines the stepper marks each occur exactly once, in order', () => {
    expect(marks).toEqual(['int i =', 'if (i < n)', 'C[i] = A[i] + B[i];']);
    const lines = k!.text.split('\n');
    const at = marks.map((s) => {
      const hits = lines.flatMap((l, i) => (l.includes(s) ? [i] : []));
      expect(hits).toHaveLength(1);
      return hits[0];
    });
    expect(at[0]).toBeLessThan(at[1]);
    expect(at[1]).toBeLessThan(at[2]);
  });
  it('differs from Fig. 2.10 as the lesson says: const inputs, the figure\'s order for the index', () => {
    expect(k!.text).toContain('vecAddKernel(const float* A, const float* B, float* C, int n)');
    expect(k!.text).toContain('int i = threadIdx.x + blockDim.x * blockIdx.x;');
  });
});

describe('lab 2.1 kernel.metal, the Metal tab of the listing', () => {
  const metal = read('labs/ch02/lab1_vecadd/solution/kernel.metal');
  const v = functionSource(metal, 'vec_add');
  it('has the three differences the lesson names: kernel void, device pointers in buffer slots, the index as an argument', () => {
    expect(v).not.toBeNull();
    expect(v!.text).toContain('kernel void vec_add(');
    expect(v!.text).toMatch(/device const float\* A\s+\[\[buffer\(0\)\]\]/);
    expect(v!.text).toContain('uint i [[thread_position_in_grid]]');
  });
  it('has no guard, and the host side dispatches exactly n threads, which is why', () => {
    const code = v!.text.split('\n').filter((l) => !l.trim().startsWith('//')).join('\n');
    expect(code).not.toMatch(/\bif\s*\(/);
    expect(read('labs/ch02/lab1_vecadd/solution/launch.mm')).toContain('dispatchThreads:MTLSizeMake(n, 1, 1)');
  });
});

describe('quotations from notes/today/ch02-facts.md', () => {
  it('the three llama.cpp lines are quoted verbatim in item D1', () => {
    const d1 = item('D1');
    for (const line of constant('LLAMA_SCALE').split('\n')) expect(d1).toContain('`' + line);
  });
  it("AMD's index line and guard are quoted verbatim in item B4", () => {
    const b4 = item('B4');
    expect(b4).toContain(constant('AMD_INDEX'));
    expect(b4).toContain(constant('AMD_GUARD'));
  });
});

describe('numbers the lesson states, from the model', () => {
  it('q-own: thread 5 of block 3 with blocks of 256 owns 773', () => {
    expect(mapping('one').first(3, 5, 256)).toBe(773);
  });
  it('p-idle: 100 elements in blocks of 32 is 4 blocks, 128 threads, 28 idle', () => {
    expect(counts({ n: 100, blockDim: 32 })).toMatchObject({ blocks: 4, launched: 128, idle: 28 });
  });
  it('q-noguard: n = 1000, blocks of 256, no guard: 24 threads touch 1000 to 1023', () => {
    const c = coverage({ n: 1000, blockDim: 256, guard: false });
    expect([c.blocks, c.launched, c.idle, span(c.outOfRange)]).toEqual([4, 1024, 24, '1000 to 1023']);
  });
  it('q-two-adjacent: with n = 101 and blocks of 32, one thread has its second element out of range', () => {
    expect(counts({ n: 101, blockDim: 32, mapping: 'two-adjacent' }).partial).toBe(1);
  });
});

describe('the narration of "Where the loop went"', () => {
  const script = (k: number) => read(`content/ch02/narration/loop-to-grid-${k}.txt`).replace(/\[[^\]]*\]\s*/g, '').trim();
  const scene = fs.readFileSync(path.join(__dirname, 'LoopToGridScene.tsx'), 'utf8');
  it('six scripts of 35 to 60 words, no symbols read aloud', () => {
    for (let k = 1; k <= 6; k++) {
      const words = script(k).split(/\s+/).length;
      expect(words).toBeGreaterThanOrEqual(35);
      expect(words).toBeLessThanOrEqual(60);
      expect(script(k)).not.toMatch(/[<>=*+[\]{}%#&]|\bi\s*<\s*n\b/);
    }
    expect(scene).toContain('export const SCENE_N = 10;');
    expect(scene).toContain('export const SCENE_B = 4;');
  });
  it('says what the model computes for 10 elements in blocks of 4', () => {
    const c = coverage({ n: 10, blockDim: 4 });
    expect(script(1)).toContain('from 0 up to 9');
    expect(script(3)).toContain(`10 threads need ${c.blocks} blocks`);
    expect(script(3)).toContain(`${c.blocks} blocks of 4 is ${c.launched} threads`);
    expect(script(3)).toContain(`That is ${c.idle} more`);
    const m = mapping('one');
    expect(script(4)).toContain(`Block 0, thread 1 gives ${m.first(0, 1, 4)}`);
    expect(script(4)).toContain(`Block 1, thread 2 gives ${m.first(1, 2, 4)}`);
    expect(script(4)).toContain(`Block 2, thread 3 gives ${m.first(2, 3, 4)}`);
    expect(script(5)).toContain(`Threads ${c.idleThreads.map((t) => t.i).join(' and ')} fail it`);
    expect(script(6)).toContain('a grid of exactly 10 threads');
    expect(counts({ n: 10, blockDim: 10 }).idle).toBe(0);
  });
});
