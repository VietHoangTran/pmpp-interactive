import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

// Code shown in section 21.2 must be code that is checked. The kernels live in
// labs/ch21/snippets, where the CUDA front-end check parses them and check.py runs them on
// the CPU emulator (the printed Fig. 21.8 to show that it fails). This test fails if a copy
// in S212.tsx drifts from the part of its file below the marker line, or if a line the page
// refers to is no longer where the page says.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'S212.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in S212.tsx`);
  return m[1].trim();
};
const shown = (name: string, marker = '// ---- shown in the lesson ----\n') => {
  const parts = fs.readFileSync(path.join(repo, 'labs/ch21/snippets', name), 'utf8').split(marker);
  if (parts.length !== 2) throw new Error(`${name}: expected one marker line`);
  return parts[1].trim();
};

describe('section 21.2 code matches its checked files', () => {
  it('scatter (Fig. 21.5 with an atom bound)', () => expect(lesson('SCATTER')).toBe(shown('scatter.cu')));
  it('gather (Fig. 21.6 with bound checks)', () => expect(lesson('GATHER')).toBe(shown('gather.cu')));
  it('Fig. 21.8 as printed', () => expect(lesson('PRINTED')).toBe(shown('coarsen_printed.cu', '// ---- as printed ----\n')));
  it('Fig. 21.8 corrected', () => expect(lesson('ADJACENT')).toBe(shown('coarsen_adjacent.cu')));
  it('Fig. 21.10 with bound checks', () => expect(lesson('STRIDED')).toBe(shown('coarsen_strided.cu')));
});

describe('the lines the page refers to', () => {
  const printed = lesson('PRINTED').split('\n');
  it('the printed listing keeps the figure\'s 34 lines', () => expect(printed.length).toBe(34));
  it('line 05 adds an unscaled threadIdx.x', () => {
    expect(printed[4]).toContain('int i = blockIdx.x * blockDim.x*COARSEN_FACTOR + threadIdx.x;');
  });
  it('lines 30 to 33 write i, i+1, i+2, i+3', () => {
    ['i  ]', 'i+1]', 'i+2]', 'i+3]'].forEach((t, k) => expect(printed[29 + k]).toContain(t));
  });
  it('the highlight constant names those lines', () => {
    expect(src).toContain('export const PRINTED_LINES = { index: 5, writes: [30, 31, 32, 33] };');
  });
  it('the comment on line 15 and the loop on line 16', () => {
    expect(printed[14].trim().startsWith('//')).toBe(true);
    expect(printed[15]).toContain('for (int n=0; n<atomarrdim; n+=4) {');
  });
  it('the corrected Fig. 21.8 scales the thread index; Fig. 21.10 keeps line 05 and strides by blockDim.x', () => {
    expect(lesson('ADJACENT')).toContain('int i = (blockIdx.x * blockDim.x + threadIdx.x) * COARSEN_FACTOR;');
    expect(lesson('ADJACENT')).toContain('float dx1 = dx0 + gridspacing;');
    expect(lesson('STRIDED')).toContain('int i = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x;');
    expect(lesson('STRIDED')).toContain('float dx1 = dx0 + blockDim.x * gridspacing;');
    expect(lesson('STRIDED')).toContain('row[i + B] += energy1;');
  });
  it('scatter keeps the legacy atomicAdd and adds the bound; gather writes with a plain +=', () => {
    expect(lesson('SCATTER')).toContain('atomicAdd(&energygrid[row + i], charge / sqrtf(dx * dx + dy2 + dz2));');
    expect(lesson('SCATTER')).toContain('if (n >= numatoms * 4) return;');
    expect(lesson('GATHER')).toContain('energygrid[grid.x * grid.y * k + grid.x * j + i] += energy;');
    expect(lesson('GATHER')).not.toContain('atomic');
  });
});
