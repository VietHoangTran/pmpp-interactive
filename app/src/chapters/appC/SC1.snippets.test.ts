import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { CARDS, judge } from '../../sim/addrspace';

// Section C.1 shows two snippets kept in labs/appC/snippets, where the CUDA one is
// front-end-checked by ./verify.sh. This test fails if the lesson's copies drift
// from the files, and ties the cards and quotes on the page to their sources.
const repo = path.resolve(__dirname, '../../../..');
const read = (rel: string) => fs.readFileSync(path.join(repo, rel), 'utf8');
const lesson = fs.readFileSync(path.join(__dirname, 'SC1.tsx'), 'utf8');
const constant = (name: string) => {
  const m = new RegExp('const ' + name + ' = `([\\s\\S]*?)`;').exec(lesson);
  if (!m) throw new Error(`no ${name} in SC1.tsx`);
  return m[1].trim();
};
const facts = read('notes/today/appC-facts.md');
const item = (id: string) => {
  const at = facts.indexOf(`### ${id}.`);
  if (at < 0) throw new Error(`no item ${id}`);
  const next = facts.indexOf('\n### ', at + 4);
  return facts.slice(at, next < 0 ? undefined : next);
};

describe('lesson snippets match their checked files', () => {
  it('the CUDA helper with its two call sites', () => {
    expect(constant('GENERIC_CU')).toBe(read('labs/appC/snippets/c1_helper_generic.cu').trim());
  });
  it('the MSL pair of overloads', () => {
    expect(constant('OVERLOADS_METAL')).toBe(read('labs/appC/snippets/c1_helper_overloads.metal').trim());
  });
  it('the CUDA helper takes a plain pointer and is called on a shared tile and a global row', () => {
    const cu = constant('GENERIC_CU');
    expect(cu).toContain('__device__ float rowMax(const float* p, int n)');
    expect(cu).toContain('rowMax(row, n)');
    expect(cu).toContain('rowMax(tile, n)');
    expect(cu).toContain('__shared__ float tile[TILE];');
  });
  it('the MSL file has one definition per address space, and both are called', () => {
    const m = constant('OVERLOADS_METAL');
    expect(m).toContain('float rowMax(device const float* p, uint n)');
    expect(m).toContain('float rowMax(threadgroup const float* p, uint n)');
    expect(m).toContain('rowMax(r, n)');
    expect(m).toContain('rowMax(tile, n)');
  });
});

describe('the page and its sources', () => {
  it('the C1 quotes the page relies on are in the facts file', () => {
    const c1 = item('C1');
    expect(c1).toContain('If an address space attribute is missing on a pointer or reference type declaration, a compilation error occurs.');
    expect(c1).toContain('Metal extends the function overloading rules to include the address space attribute of an argument.');
    expect(c1).toContain('only if foo is called by a kernel function');
    expect(lesson).toContain('If an address space attribute is missing on a pointer or reference type declaration, a\n        compilation error occurs.');
  });
  it('A2 has the PTX windows, cvta and isspacep, and the March 2010 date the Today note states', () => {
    const a2 = item('A2');
    for (const s of ['modeled as windows within the generic address space', 'cvta', 'isspacep', 'March 2010', 'sm_20']) expect(a2).toContain(s);
  });
  it('B1 has the flat address space, the apertures and the 64-bit condition', () => {
    const b1 = item('B1');
    for (const s of ['"flat"', 'apertures', 'Only 64-bit process address spaces are supported for the amdgcn target.']) expect(b1).toContain(s);
  });
  it('the Metal tab\'s "verified on Metal" rests on the checked-in harness, which runs that file', () => {
    expect(lesson).toContain("status: 'verified on Metal'");
    expect(lesson).toContain('python3 labs/appC/snippets/check.py');
    const check = read('labs/appC/snippets/check.py');
    expect(check).toContain('"c1_helper_overloads.metal"');
    expect(read('labs/appC/snippets/check/c1_run.mm')).toContain('"row_max_twice"');
  });
  it('the MSL 4.1 observation is recorded in C1 as observed, not documented, with the compiler and the commands', () => {
    const c1 = item('C1');
    for (const s of ['**Observed, not documented**', 'metal version 32023.921', '-std=metal4.0 -c probe.metal', 'pointer type must have explicit address space qualifier', 'labs/appC/snippets/check.py'])
      expect(c1).toContain(s);
    expect(lesson).not.toContain('has been requested');
  });
  it('the card for each language lands in the bin the page text describes', () => {
    const bin = (id: string) => judge(CARDS.find((c) => c.id === id)!).bin;
    expect([bin('cuda-shared'), bin('hip-shared'), bin('msl-none'), bin('msl-device-only')]).toEqual(['shared', 'shared', 'error', 'error']);
  });
});
