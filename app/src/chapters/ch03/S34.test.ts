import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

// Section 3.4 shows the book's kernel (Fig. 3.11) and our launch as two listings
// and steps through the kernel by line number. Both are one checked file,
// labs/ch03/snippets/matmul_kernel.cu. This test fails if the lesson's copy and
// that file drift apart, or if the kernel's lines stop matching the book's numbering.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'S34.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in S34.tsx`);
  return m[1].trim();
};
const file = fs.readFileSync(path.join(repo, 'labs/ch03/snippets/matmul_kernel.cu'), 'utf8').trim();

describe('section 3.4 listings match the checked snippet', () => {
  it('kernel, blank line, launch is the whole file', () => {
    expect(lesson('KERNEL') + '\n\n' + lesson('LAUNCH')).toBe(file);
  });
  it("the kernel keeps the book's twelve lines, with the loop body on line 08 and the store on line 10", () => {
    const lines = lesson('KERNEL').split('\n');
    expect(lines).toHaveLength(12);
    expect(lines[2].trim()).toBe('int row = blockIdx.y*blockDim.y+threadIdx.y;');
    expect(lines[3].trim()).toBe('int col = blockIdx.x*blockDim.x+threadIdx.x;');
    expect(lines[4].trim()).toBe('if ((row < Width) && (col < Width)) {');
    expect(lines[5].trim()).toBe('float Pvalue = 0;');
    expect(lines[7].trim()).toBe('Pvalue += M[row*Width+k]*N[k*Width+col];');
    expect(lines[9].trim()).toBe('P[row*Width+col] = Pvalue;');
  });
  it('the launch listing starts on line 14 of the file', () => {
    expect(file.split('\n')[13]).toBe(lesson('LAUNCH').split('\n')[0]);
  });
});

// Book exercise 2's question shows four kernel bodies. Each is the body of one kernel in
// labs/ch03/snippets/matvec_options.cu, which the CUDA front-end check parses and
// labs/ch03/snippets/check.py runs on the emulator.
const matvec = fs.readFileSync(path.join(repo, 'labs/ch03/snippets/matvec_options.cu'), 'utf8');
const body = (n: number) => {
  const lines = matvec.split('\n');
  const start = lines.findIndex((l) => l.startsWith(`__global__ void matvecOption${n}(`));
  if (start < 0) throw new Error(`no matvecOption${n} in matvec_options.cu`);
  const end = lines.indexOf('}', start);
  return lines
    .slice(start + 1, end)
    .map((l) => l.replace(/^ {4}/, ''))
    .join('\n');
};

describe("exercise 2's options match the checked snippet", () => {
  it.each([1, 2, 3, 4])('option %i is the body of matvecOption%i', (n) => {
    expect(lesson(`MATVEC_${n}`)).toBe(body(n));
  });
  it('the options are offered in file order and only the second is marked correct', () => {
    const q = /'ch03\/3\.4\/q-ex2': \{([\s\S]*?)\n {2}\},\n/.exec(src)?.[1] ?? '';
    const order = [...q.matchAll(/\{MATVEC_(\d)\}/g)].map((m) => Number(m[1]));
    expect(order).toEqual([1, 2, 3, 4]);
    expect(/\{MATVEC_2\}<\/code>,\s*correct: true/.test(q)).toBe(true);
    expect((q.match(/correct: true/g) ?? []).length).toBe(1);
  });
});
