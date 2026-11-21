import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

// Code shown in a lesson must be code that is checked. Lab files are loaded from
// disk at run time; the few kernels written only for a lesson are duplicated in
// labs/ch05/snippets, where ./verify.sh front-end-checks them. This test fails
// if a lesson's copy and the checked file drift apart.
const repo = path.resolve(__dirname, '../../../..');
const lesson = (file: string, constant: string) => {
  const src = fs.readFileSync(path.join(__dirname, file), 'utf8');
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in ${file}`);
  return m[1].trim();
};
const snippet = (name: string) => fs.readFileSync(path.join(repo, 'labs/ch05/snippets', name), 'utf8').trim();

describe('lesson kernels match their checked files', () => {
  it('section 5.2 "where does it live?" kernel', () => {
    expect(lesson('S52.tsx', 'SMOOTH')).toBe(snippet('where_does_it_live.cu'));
  });
  it('summary page tile-transpose kernel', () => {
    expect(lesson('Wrap.tsx', 'TRANSPOSE')).toBe(snippet('transpose_tiles_defect.cu'));
  });
  it('start page naive kernel is the labs\' shared baseline', () => {
    const file = fs.readFileSync(path.join(repo, 'labs/ch05/shared/naive_matmul.cu'), 'utf8');
    expect(file).toContain(lesson('S50.tsx', 'NAIVE'));
  });
  it('section 5.4 stepper body matches the lab 5.2 solution, line by line', () => {
    // The stepper shows the body without comments, with tighter spacing and without the
    // braces of the inner loop. Squash each line (comments, whitespace and braces removed;
    // parentheses kept, except the redundant pair the solution puts around
    // ph*TILE_WIDTH + tx) and require the stepper's lines to be a contiguous run of the
    // solution's lines: same lines, same order, none left out.
    const squash = (t: string) =>
      t.replace(/\/\/.*$/, '').replace(/[\s{}]/g, '').replace('[Row*Width+(ph*TILE_WIDTH+tx)]', '[Row*Width+ph*TILE_WIDTH+tx]');
    const fileLines = fs
      .readFileSync(path.join(repo, 'labs/ch05/lab2_tiled/solution/kernel.cu'), 'utf8')
      .split('\n').map(squash).filter(Boolean);
    const lines = lesson('S54.tsx', 'BODY').split('\n').map(squash).filter(Boolean);
    expect(lines.length).toBeGreaterThan(10);
    const start = fileLines.indexOf(lines[0]);
    expect(start).toBeGreaterThanOrEqual(0);
    expect(fileLines.slice(start, start + lines.length)).toEqual(lines);
  });
});
