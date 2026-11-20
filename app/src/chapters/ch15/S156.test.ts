import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

// Section 15.6's erratum hunt shows lines of labs/ch15/snippets/s156_padded_offset.cu
// (front-end-checked) as the constant PADDED, starting at PADDED_FIRST_LINE, and refers
// to three of its lines by number. This test fails if the lesson's copy or its line
// numbers drift from the file, and, once lab 15.4 exists, if its solution does not
// have the corrected form: every A_s index uses the padded leading dimension.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'S156.tsx'), 'utf8');
const file = fs.readFileSync(path.join(repo, 'labs/ch15/snippets/s156_padded_offset.cu'), 'utf8').split('\n');
const constant = (name: string) => {
  const m = new RegExp('const ' + name + ' = (\\d+);').exec(src);
  if (!m) throw new Error('no ' + name + ' in S156.tsx');
  return Number(m[1]);
};

describe('section 15.6 erratum hunt snippet', () => {
  const m = /const PADDED = `([\s\S]*?)`;/.exec(src);
  const first = constant('PADDED_FIRST_LINE');
  it('the lesson copy is lines of the checked file', () => {
    expect(m).not.toBeNull();
    const lines = m![1].split('\n');
    expect(lines.length).toBeGreaterThan(10);
    expect(file.slice(first - 1, first - 1 + lines.length)).toEqual(lines);
  });
  it('the line numbers the hunt names are the declaration, the A load and the mm call', () => {
    expect(file[constant('LINE_DECL') - 1].trim()).toBe('__shared__ float A_s[bM*(bK + 1)];');
    expect(file[constant('LINE_LOAD_A') - 1]).toContain('loadTile(&A[');
    expect(file[constant('LINE_LOAD_A') - 1]).toContain('&A_s[0], bK + 1, bM, bK);');
    expect(file[constant('LINE_MM') - 1].trim()).toBe('mm(tM, tN, bK, &A_s[tRow*bK], bK + 1, &B_s[tCol], bN, C_r);');
  });
  it('the question’s options name those line numbers', () => {
    expect(src).toContain(`Line ${constant('LINE_LOAD_A')}: the last argument of the A loadTile call`);
    expect(src).toContain(`Line ${constant('LINE_MM')}: the start offset &A_s[tRow*bK]`);
    expect(file[26 - 1].trim()).toBe('__shared__ float B_s[bK*bN];');
    expect(src).toContain('Line 26: B_s needs a padding column too.');
  });
});

// Lab 15.4 is written in parallel with this section; the check runs once its solution exists.
const lab4 = (() => {
  const dir = path.join(repo, 'labs/ch15');
  const d = fs.existsSync(dir) ? fs.readdirSync(dir).find((x) => x.startsWith('lab4')) : undefined;
  const f = d ? path.join(dir, d, 'solution/kernel.cu') : '';
  return f && fs.existsSync(f) ? fs.readFileSync(f, 'utf8') : null;
})();

describe.skipIf(lab4 === null)('lab 15.4 solution has the corrected form', () => {
  it('declares a padded A_s and never offsets into A_s with the unpadded row length', () => {
    // Code only: the solution's comments quote the book's line on purpose.
    const code = lab4!.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
    expect(code).toMatch(/A_s\s*\[[^\]]*\+\s*1\s*\)?\s*\]|A_s\s*\[[^\]]*(LDA|ldA|LDAS|A_LD|lda_s)[^\]]*\]/);
    // The lesson's caption names the constant the solution uses for the padded leading dimension.
    if (/LDA_S/.test(src)) expect(code).toMatch(/#define\s+LDA_S\s+\(bK \+ 1\)/);
    expect(code).not.toContain('A_s[tRow*bK]');
    expect(code).not.toMatch(/&A_s\[\s*\w+\s*\*\s*bK\s*\]/);
  });
});
