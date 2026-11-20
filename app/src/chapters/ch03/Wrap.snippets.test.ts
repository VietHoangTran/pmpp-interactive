import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { span } from '../../sim/blur';
import { layout, strides } from '../../sim/mapping';

// The summary page shows one kernel of its own, for a question. Its checked copy is
// labs/ch03/snippets/scale_rows_defect.cu, which the CUDA front-end check parses.
// This test fails if the page's copy and that file drift apart, and pins the numbers
// the page's questions take from the models.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'Wrap.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in Wrap.tsx`);
  return m[1].trim();
};
const file = fs.readFileSync(path.join(repo, 'labs/ch03/snippets/scale_rows_defect.cu'), 'utf8').trim();

describe('summary page', () => {
  it('the row-scaling kernel is the checked snippet', () => {
    expect(lesson('SCALE_ROWS')).toBe(file);
  });

  it('the kernel reads the element that the thread of column 0 writes', () => {
    const lines = file.split('\n');
    expect(lines[5].trim()).toBe('X[row*cols + col] = X[row*cols + col] / X[row*cols];');
    // The launch sizes x from the columns and y from the rows, as p. 50 does: the mapping is not the defect.
    expect(file).toContain('dim3 dimGrid((cols + 15)/16, (rows + 15)/16, 1);');
  });

  it('one token through a 1024 to 4096 layer under 16 by 16 blocks: 61,440 of 65,536 threads are surplus', () => {
    const l = layout({ height: 1, width: 4096, blockW: 16, blockH: 16, style: 'blocks' });
    expect([l.gridX, l.gridY, l.blocks]).toEqual([256, 1, 256]);
    expect(l.threads).toBe(65536);
    expect(l.working).toBe(4096);
    expect(l.surplus).toBe(61440);
    // 256 by 1 blocks cover the same row exactly.
    expect(layout({ height: 1, width: 4096, blockW: 256, blockH: 1, style: 'blocks' }).surplus).toBe(0);
  });

  it('a corner of a volume with radius 2 averages 27 of 125 values; an edge 45, a face 75', () => {
    const end = span(0, 64, 2);
    const mid = span(32, 64, 2);
    expect([end, mid]).toEqual([3, 5]);
    expect(end ** 3).toBe(27);
    expect(mid ** 3).toBe(125);
    expect(end * end * mid).toBe(45);
    expect(end * mid * mid).toBe(75);
  });

  it('strides of a row-major [batch, j, k] tensor are (j*k, k, 1)', () => {
    expect(strides([8, 3, 5])).toEqual([15, 5, 1]);
    // (b*j + row)*k + i is the same offset as index times stride, summed.
    const [b, row, i, j, k] = [2, 1, 4, 3, 5];
    expect((b * j + row) * k + i).toBe(b * 15 + row * 5 + i);
  });

  it('B[i*k + col] and B[i*l + col] agree exactly when k = l or k = 1', () => {
    const agrees = (k: number, l: number) => {
      for (let i = 0; i < k; i++) for (let col = 0; col < l; col++) if (i * k + col !== i * l + col) return false;
      return true;
    };
    // The four options of the question, as (j, k, l).
    expect(agrees(7, 3)).toBe(false);
    expect(agrees(40, 33)).toBe(false);
    expect(agrees(1, 35)).toBe(true);
    expect(agrees(20, 1)).toBe(false);
    expect(agrees(33, 33)).toBe(true);
  });
});
