import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { goodBeforeFailure } from '../../sim/ptrchase';
import { REGISTERED, hex4 } from '../../sim/uvacopy';

// Section C.3 shows labs/appC/snippets/c3_chase.cu, which ./verify.sh front-end-checks.
// This test fails if the lesson's copy drifts from the file, and ties the page's
// quotations and numbers to their sources.
const repo = path.resolve(__dirname, '../../../..');
const read = (rel: string) => fs.readFileSync(path.join(repo, rel), 'utf8');
const lesson = fs.readFileSync(path.join(__dirname, 'SC3.tsx'), 'utf8');
const constant = (name: string) => {
  const m = new RegExp('const ' + name + ' = `([\\s\\S]*?)`;').exec(lesson);
  if (!m) throw new Error(`no ${name} in SC3.tsx`);
  return m[1].trim();
};
const facts = read('notes/today/appC-facts.md');
const ch02 = read('notes/today/ch02-facts.md');
const item = (text: string, id: string) => {
  const at = text.indexOf(`### ${id}.`);
  if (at < 0) throw new Error(`no item ${id}`);
  const next = text.indexOf('\n### ', at + 4);
  return text.slice(at, next < 0 ? undefined : next);
};
const flat = (s: string) => s.replace(/\s+/g, ' ');

describe('the chase kernel', () => {
  it('matches labs/appC/snippets/c3_chase.cu', () => {
    expect(constant('CHASE_CU')).toBe(read('labs/appC/snippets/c3_chase.cu').trim());
  });
  it('has the struct and the one-thread loop the page describes', () => {
    const k = constant('CHASE_CU');
    expect(k).toContain('struct Node {\n    float value;\n    Node* next;\n};');
    expect(k).toContain('for (const Node* p = head; p != nullptr; p = p->next) s += p->value;');
    expect(k).toContain('sumList<<<1, 1>>>(head, sum);');
  });
});

describe('quotations on the page are in the facts files', () => {
  it('A5: the guide\'s definition of the unified address space', () => {
    expect(flat(item(facts, 'A5'))).toContain('A single virtual address space is used for all host memory and all global memory on all GPUs');
    expect(flat(lesson)).toContain('A single virtual address space is used for all host memory and all global memory on all GPUs');
  });
  it('A12: hardware coherence does not open cudaMalloc memory to the host', () => {
    const q = 'Hardware coherency does not enable access to GPU-only allocations such as those made with cudaMalloc from the host.';
    expect(item(facts, 'A12')).toContain(q);
    expect(flat(lesson)).toContain(q);
  });
  it('A14: prefer cudaMemcpyDefault', () => {
    expect(item(facts, 'A14')).toContain('Prefer cudaMemcpyDefault over an inaccurate cudaMemcpyKind hint.');
  });
  it('B3: hipMalloc host access, and the "Cuda only" attribute', () => {
    const b3 = item(facts, 'B3');
    expect(b3).toContain('CPUs can access memory allocated via hipMalloc()');
    expect(b3).toContain('Cuda only');
  });
  it('ch02 A4: cudaMemcpyDefault requires unified virtual addressing', () => {
    expect(item(ch02, 'A4')).toContain('cudaMemcpyDefault is only allowed on systems that support unified virtual addressing');
  });
  it('C3: gpuAddress documented, relation to contents not', () => {
    const c3 = item(facts, 'C3');
    expect(c3).toContain('COULD NOT VERIFY');
    expect(c3).toContain('gpuAddress');
  });
});

describe('numbers the page states', () => {
  it('q-chase: two values before the failure (from the model)', () => {
    expect(goodBeforeFailure(['cudaHostAlloc', 'cudaHostAlloc', 'malloc', 'cudaHostAlloc'], 'uva')).toBe(2);
  });
  it('the default list of the stepper fails on its third hop under CUDA 4', () => {
    const m = /useState<Alloc\[\]>\((\[[^\]]*\])\)/.exec(lesson);
    const list = JSON.parse(m![1].replace(/'/g, '"'));
    expect(goodBeforeFailure(list, 'uva')).toBe(2);
  });
  it('the registered buffer\'s two addresses on the map', () => {
    expect(hex4(REGISTERED.host[0])).toBe('0x2000');
    expect(hex4(REGISTERED.device[0])).toBe('0x7400');
  });
});
