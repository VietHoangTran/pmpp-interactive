import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

// The blur kernel shown in section 3.3 is a copy of labs/ch03/snippets/blur_kernel.cu,
// which the CUDA front-end check parses. This fails if the two drift apart, or if the
// lines the lesson refers to by the book's numbers move.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'S33.tsx'), 'utf8');
const m = /const KERNEL = `([\s\S]*?)`;/.exec(src);
if (!m) throw new Error('no KERNEL in S33.tsx');
const lesson = m[1].trim();
const file = fs.readFileSync(path.join(repo, 'labs/ch03/snippets/blur_kernel.cu'), 'utf8').trim();

describe('section 3.3 blur kernel', () => {
  it('matches the checked file', () => {
    expect(lesson).toBe(file);
  });
  it("keeps the book's line numbers when the #define is shown as line 0", () => {
    const line = lesson.split('\n');
    expect(line.length).toBe(25); // line 0 and the book's 01 to 24
    expect(line[0]).toContain('#define BLUR_SIZE 1');
    expect(line[5]).toContain('if (col < w && row < h)');
    expect(line[8].trim()).toBe(''); // the number the book's listing skips
    expect(line[10]).toContain('for (int blurRow');
    expect(line[11]).toContain('for (int blurCol');
    expect(line[12]).toContain('int curRow = row + blurRow');
    expect(line[13]).toContain('int curCol = col + blurCol');
    expect(line[15]).toContain('if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w)');
    expect(line[16]).toContain('pixVal += in[curRow*w + curCol]');
    expect(line[17]).toContain('++pixels');
    expect(line[22]).toContain('out[row*w + col] = (unsigned char)((float)pixVal/pixels)');
  });
});
