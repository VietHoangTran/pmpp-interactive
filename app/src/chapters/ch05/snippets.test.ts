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
});
