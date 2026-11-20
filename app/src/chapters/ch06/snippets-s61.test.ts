import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { exercise1 } from '../../sim/coalesce';

// Section 6.1 shows exercise 1's kernel (book pp. 154–155). The checked copy is
// labs/ch06/snippets/s61_foo_kernel.cu, which ./verify.sh front-end-checks; this
// test fails if the lesson's copy drifts from it, or if the access model's nine
// expressions stop matching the lines they claim to come from.
const repo = path.resolve(__dirname, '../../../..');
const lesson = (file: string, constant: string) => {
  const src = fs.readFileSync(path.join(__dirname, file), 'utf8');
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in ${file}`);
  return m[1].trim();
};
const snippet = fs.readFileSync(path.join(repo, 'labs/ch06/snippets/s61_foo_kernel.cu'), 'utf8').trim();

describe('section 6.1, exercise 1', () => {
  it('the lesson shows the checked file', () => {
    expect(lesson('S61.tsx', 'FOO')).toBe(snippet);
  });
  it('the file has the book\'s twelve numbered lines', () => {
    expect(snippet.split('\n')).toHaveLength(12);
  });
  it('every access the drill classifies is on the line it names', () => {
    const lines = snippet.split('\n');
    for (const x of exercise1()) expect(lines[x.line - 1]).toContain(x.code);
  });
  it('the drill names nine accesses, a to i', () => {
    expect(exercise1().map((x) => x.id).join('')).toBe('abcdefghi');
  });
});
