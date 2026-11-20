import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { PROGRAM, functionSource } from '../../sim/toolchain';

// Section 2.7 shows functions of lab files, cut out of them at run time with
// functionSource(), so there is no copy to drift. What can drift is the files
// themselves: a renamed function would leave the lesson with nothing to show, and
// a changed body would make its sentences wrong. This test ties the two together.
const repo = path.resolve(__dirname, '../../../..');
const read = (rel: string) => fs.readFileSync(path.join(repo, rel), 'utf8');
const lesson = fs.readFileSync(path.join(__dirname, 'S27.tsx'), 'utf8');
const names = (constName: string) =>
  [...(new RegExp('const ' + constName + ' = \\[(.*)\\] as const;').exec(lesson)?.[1] ?? '').matchAll(/'([^']*)'/g)].map((m) => m[1]);
/** The code of a function without its comment lines. */
const code = (text: string) => text.split('\n').filter((l) => !l.trim().startsWith('//')).join('\n');

describe("the shared helper, from lab 2.3's solution", () => {
  const cu = read('labs/ch02/lab3_elementwise/solution/kernel.cu');
  const metal = read('labs/ch02/lab3_elementwise/solution/kernel.metal');
  it('the lesson asks for three CUDA functions and two Metal ones', () => {
    expect(names('HELPER_CU')).toEqual(['scaleShift', 'scaleShiftOnHost', 'scaleShiftKernel']);
    expect(names('HELPER_METAL')).toEqual(['scale_shift', 'scale_shift_k']);
  });
  it('kernel.cu: one __host__ __device__ definition, called by a host function and by the kernel', () => {
    const helper = functionSource(cu, 'scaleShift')!;
    const host = functionSource(cu, 'scaleShiftOnHost')!;
    const kernel = functionSource(cu, 'scaleShiftKernel')!;
    expect(helper.quals).toBe('host-device');
    expect(code(helper.text)).toContain('return alpha * x + beta;');
    expect(host.quals).toBe('none');
    expect(code(host.text)).toContain('scaleShift(x, alpha, beta)');
    expect(kernel.quals).toBe('global');
    expect(code(kernel.text)).toContain('scaleShift(X[i], alpha, beta)');
    // Defined once only.
    expect(cu.match(/__host__ __device__ inline float scaleShift\(/g)).toHaveLength(1);
  });
  it('kernel.metal: the same arithmetic written again, as a plain function the kernel calls', () => {
    const helper = functionSource(metal, 'scale_shift')!;
    const kernel = functionSource(metal, 'scale_shift_k')!;
    expect(code(helper.text)).toContain('return alpha * x + beta;');
    expect(code(helper.text)).not.toMatch(/\bkernel\b/);
    expect(code(kernel.text)).toContain('kernel void scale_shift_k(');
    expect(code(kernel.text)).toContain('scale_shift(X[i], alpha, beta)');
  });
  it("the tests compare the kernel with the host compilation of the helper, with a tolerance", () => {
    const main = read('labs/ch02/lab3_elementwise/main.cpp');
    expect(main).toContain('ref[i] = scaleShiftOnHost(D.x[i], kAlpha, kBeta)');
    expect(main).toContain('Not compared');
  });
});

describe("makeVecAddPipeline, from lab 2.1's launch.mm", () => {
  const mm = read('labs/ch02/lab1_vecadd/solution/launch.mm');
  const f = functionSource(mm, 'makeVecAddPipeline')!;
  it('is found, and has the three steps in order: library, function, pipeline state', () => {
    expect(f).not.toBeNull();
    const body = code(f.text);
    const steps = ['newLibraryWithSource:', 'newFunctionWithName:', 'newComputePipelineStateWithFunction:'].map((s) => body.indexOf(s));
    expect(steps.every((x) => x >= 0)).toBe(true);
    expect(steps).toEqual([...steps].sort((a, b) => a - b));
  });
  it('the two steps that compile report through NSError; a missing name is a plain nil', () => {
    const body = code(f.text);
    expect(body.match(/error:error\]/g)).toHaveLength(2);
    expect(body).toMatch(/newFunctionWithName:@"vec_add"\];/);
  });
  it("the lab's driver creates the pipeline once and keeps it", () => {
    const main = read('labs/ch02/lab1_vecadd/main.cpp');
    const calls = main.split('\n').filter((l) => l.includes('makeVecAddPipeline(s.device'));
    expect(calls).toHaveLength(1);
    expect(main).toContain('static MetalSetup s;');
    expect(main).toContain('if (done) return s;');
  });
});

describe("the model's program is the lab's", () => {
  it('vecAdd and vecAddKernel are in lab 2.1 with the qualifiers the model gives them', () => {
    const cu = read('labs/ch02/lab1_vecadd/solution/kernel.cu');
    for (const name of ['vecAdd', 'vecAddKernel']) {
      const fn = PROGRAM.find((p) => p.name === name)!;
      expect(functionSource(cu, name)!.quals).toBe(fn.quals);
    }
  });
});
