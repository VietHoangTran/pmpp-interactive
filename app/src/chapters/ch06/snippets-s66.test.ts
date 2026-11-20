import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

// Section 6.6 shows labs/ch06/snippets/s66_unroll.cu as the constant UNROLL and
// highlights the unrolled, reordered loop by line number. The five "where does x
// live?" kernels are tied to s66_where_x_lives.cu by src/sim/unroll.test.ts.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'S66.tsx'), 'utf8');
const m = /const UNROLL = `([\s\S]*?)`;/.exec(src);
const file = fs.readFileSync(path.join(repo, 'labs/ch06/snippets/s66_unroll.cu'), 'utf8').trim();

describe('section 6.6 unrolling snippet', () => {
  it('the lesson copy is the checked file', () => {
    expect(m).not.toBeNull();
    expect(m![1].trim()).toBe(file);
  });
  it('lines 18 to 26 are the unrolled, reordered loop: four A calls, then four B calls', () => {
    const lines = file.split('\n');
    expect(lines[17]).toContain('i += 4');
    expect(lines.slice(18, 22).every((l) => l.includes('= A('))).toBe(true);
    expect(lines.slice(22, 26).every((l) => l.trim().startsWith('B('))).toBe(true);
    expect(src).toContain('highlight={[18, 19, 20, 21, 22, 23, 24, 25, 26]}');
  });
});
