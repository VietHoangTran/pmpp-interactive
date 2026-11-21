import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { FRAGMENTS, FRAGMENT_MODEL, parseFragments, runFragment } from './qdot';

// Section X2.6's bug spot shows four fragments loaded at run time from
// labs/x2/snippets/accumulators.md. This test ties what the page shows to what runFragment
// models: the same four ids in the same order, each fragment's accumulator declaration and the
// operation that adds into it, and the type the model sums in.
const repo = path.resolve(__dirname, '../../..');
const md = fs.readFileSync(path.join(repo, 'labs/x2/snippets/accumulators.md'), 'utf8');
const frags = parseFragments(md);
const byId = Object.fromEntries(frags.map((f) => [f.id, f]));
/** Code lines without comments, trimmed. */
const lines = (code: string) => code.replace(/\/\/.*$/gm, '').split('\n').map((l) => l.trim()).filter(Boolean);

describe('labs/x2/snippets/accumulators.md against qdot.ts', () => {
  it('has the four fragments of the model, in its order, each with a title, a language and code', () => {
    expect(frags.map((f) => f.id)).toEqual(FRAGMENTS);
    for (const f of frags) {
      expect(f.title.length).toBeGreaterThan(5);
      expect(['MSL', 'CUDA']).toContain(f.lang);
      expect(f.code.split('\n').length).toBeGreaterThanOrEqual(4);
    }
  });

  it('declares each accumulator as the model says, and adds into it with the modelled operation', () => {
    for (const id of FRAGMENTS) {
      const { decl, op } = FRAGMENT_MODEL[id];
      const code = byId[id].code;
      expect(lines(code)[0].startsWith(decl + ' = '), id).toBe(true);
      expect(code, id).toContain(op);
    }
  });

  it('fragment 1: half4 operands into dot, so four products round in half before the float sum', () => {
    const c = byId['metal-half-dot'].code;
    expect(byId['metal-half-dot'].lang).toBe('MSL');
    expect(c).toMatch(/half4 w = half4\(d\) \* \(half4\(q\[i\/4\]\) - 8\.0h\);/);
    expect(c).toContain('sumf += dot(w, half4(x4[i/4]));');
    expect(c).toContain('i += 4');
    expect(FRAGMENT_MODEL['metal-half-dot'].inner).toBe('half-dot4');
  });

  it('fragment 2: one __half2 accumulator over the whole row, its halves added in float at the end', () => {
    const c = byId['cuda-hfma2'].code;
    expect(c).toContain('acc = __hfma2(w2[i], x2[i], acc);');
    expect(c).toContain('i < n / 2');
    expect(c).toContain('__low2float(acc) + __high2float(acc)');
    expect(FRAGMENT_MODEL['cuda-hfma2'].inner).toBe('half2-row');
  });

  it('fragments 3 and 4 differ only in the type of sumi, and the model sums them in int16 and int32', () => {
    const a = lines(byId['short-sum'].code);
    const b = lines(byId['int-then-float'].code);
    expect(a[0]).toBe('short sumi = 0;');
    expect(b[0]).toBe('int sumi = 0;');
    expect(a.slice(1)).toEqual(b.slice(1));
    // eight dp4a steps per block: a loop of 4 with two dot4 calls each, the step the model uses
    expect(byId['short-sum'].code.match(/dot4\(/g)!.length).toBe(2);
    expect(a.some((l) => l.startsWith('for (int i = 0; i < 4; ++i)'))).toBe(true);
    expect(a.at(-1)).toBe('sumf += d4 * (sumi * d8 - 8.0f * s8);');
    expect(FRAGMENT_MODEL['short-sum'].inner).toBe('int16');
    expect(FRAGMENT_MODEL['int-then-float'].inner).toBe('int32');
  });

  it('the model gives the verdicts the page grades against: only the int-then-float fragment is safe', () => {
    const runs = FRAGMENTS.map((id) => runFragment(id));
    expect(runs.map((r) => r.safe)).toEqual([false, false, false, true]);
    const short = runs[2];
    const int = runs[3];
    // same on realistic data, so a first test would pass; wraps on the worst case
    expect(short.realistic).toBe(int.realistic);
    expect(short.worst.event).toBe('wrap');
    expect(int.worst.event).toBe('none');
    expect(int.worst.result).toBe(int.worst.reference);
  });
});
