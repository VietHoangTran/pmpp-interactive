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
