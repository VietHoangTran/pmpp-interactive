import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

// Section 6.5 shows labs/ch06/snippets/s65_coarsen_toy.cu as the constant TOY and
// highlights the coarsening loop by line number. This fails if the copy drifts from
// the checked file, or the highlighted lines stop being the loop.
const repo = path.resolve(__dirname, '../../../..');
const lesson = (constant: string) => {
  const src = fs.readFileSync(path.join(__dirname, 'S65.tsx'), 'utf8');
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in S65.tsx`);
  return m[1].trim();
};
const file = fs.readFileSync(path.join(repo, 'labs/ch06/snippets/s65_coarsen_toy.cu'), 'utf8').trim();

describe('section 6.5 coarsening toy', () => {
  it('the lesson copy is the checked file', () => {
    expect(lesson('TOY')).toBe(file);
  });
  it('lines 15 to 19 are iStart and the coarsening loop', () => {
    const lines = file.split('\n');
    expect(lines[14]).toContain('unsigned int iStart = COARSEN*(blockIdx.x*blockDim.x + threadIdx.x);');
    expect(lines[15]).toContain('// the coarsening loop');
    expect(lines[18].trim()).toBe('}');
    expect(fs.readFileSync(path.join(__dirname, 'S65.tsx'), 'utf8')).toContain('highlight={[15, 16, 17, 18, 19]}');
  });
});
