import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  ALL_PRESETS,
  BOOK_EXCLUSIVE,
  BOOK_INCLUSIVE,
  BOOK_ORDERS,
  CPPREF_EXCLUSIVE,
  CPPREF_INCLUSIVE,
  CPPREF_INPUT,
  FLIP,
  PRESETS,
  RULES,
  SAUSAGE_INCHES,
  TOKEN_COUNTS,
  band,
  exclusive,
  exclusiveIn,
  f32,
  f32hex,
  findOrderFlip,
  gamma,
  hundredthsRow,
  inclusive,
  inclusiveIn,
  naiveAdditions,
  offsets,
  orderAdditions,
  sequentialAdditions,
  toExclusive,
  toInclusive,
  topP,
  ulpSigned,
  type Order,
  type Rule,
} from './scan';

const ORDERS: Order[] = ['loop', 'kogge-stone', 'brent-kung'];
const exact = (v: number) => v;
const kept = (probs: number[], p: number, rule: Rule, order?: Order) =>
  topP(probs, p, rule, order ? { kind: 'f32', order } : { kind: 'exact' }).kept;

describe('the book\'s rows (pp. 252-253)', () => {
  it('inclusive scan of the sausage orders is 3 4 11 11 15 16 22 25', () => {
    expect(inclusive(BOOK_ORDERS)).toEqual(BOOK_INCLUSIVE);
  });
  it('exclusive scan is 0 3 4 11 11 15 16 22: where each piece starts', () => {
    expect(exclusive(BOOK_ORDERS)).toEqual(BOOK_EXCLUSIVE);
  });
  it('25 of the 40 inches are served and 15 remain', () => {
    expect(offsets(BOOK_ORDERS).total).toBe(25);
    expect(SAUSAGE_INCHES - offsets(BOOK_ORDERS).total).toBe(15);
  });
  it('the conversions of p. 253 go both ways', () => {
    expect(toExclusive(BOOK_INCLUSIVE)).toEqual(BOOK_EXCLUSIVE);
    expect(toInclusive(BOOK_EXCLUSIVE, BOOK_ORDERS)).toEqual(BOOK_INCLUSIVE);
  });
  it('the loop of Fig. 11.1 does N - 1 additions; the naive parallel scan N(N-1)/2 (p. 254)', () => {
    expect(sequentialAdditions(8)).toBe(7);
    expect(orderAdditions(8, 'loop')).toBe(7);
    expect(naiveAdditions(1024)).toBe(523_776);
    expect(sequentialAdditions(1024)).toBe(1023);
  });
});

describe('cppreference\'s example (ch11-facts A8)', () => {
  it('3 1 4 1 5 9 2 6: exclusive 0 3 4 8 9 14 23 25, inclusive 3 4 8 9 14 23 25 31', () => {
    expect(exclusive(CPPREF_INPUT)).toEqual(CPPREF_EXCLUSIVE);
    expect(inclusive(CPPREF_INPUT)).toEqual(CPPREF_INCLUSIVE);
  });
});

describe('allocation offsets (question q-offsets)', () => {
  it('six sequences of 5 3 0 7 2 4 tokens start at 0 5 8 8 15 17 and need 21 slots', () => {
    const o = offsets(TOKEN_COUNTS);
    expect(o.start).toEqual([0, 5, 8, 8, 15, 17]);
    expect(o.end).toEqual([5, 8, 8, 15, 17, 21]);
    expect(o.total).toBe(21);
    expect(o.start[4]).toBe(15);
  });
  it('the total is the last exclusive value plus the last input (question q-total)', () => {
    const e = exclusive(TOKEN_COUNTS);
    expect(e[e.length - 1] + TOKEN_COUNTS[TOKEN_COUNTS.length - 1]).toBe(21);
  });
});

describe('three association orders', () => {
  it('on integers every order gives the exact scan, for every length up to 40', () => {
    for (let n = 1; n <= 40; n++) {
      const xs = Array.from({ length: n }, (_, i) => ((i * 7) % 11) + 1);
      for (const o of ORDERS) expect(inclusiveIn(xs, o, exact), `${o} at ${n}`).toEqual(inclusive(xs));
    }
  });
  it('Kogge-Stone does N log2 N - (N - 1) additions: 49 at 16 (Fig. 11.2), 4097 at 512, 20,481 at 2048 (p. 266)', () => {
    expect(orderAdditions(16, 'kogge-stone')).toBe(49);
    expect(orderAdditions(512, 'kogge-stone')).toBe(512 * 9 - 511);
    expect(orderAdditions(2048, 'kogge-stone')).toBe(20_481);
    // "between 8 and 9 times" the sequential work for 512 elements (p. 266)
    const ratio = orderAdditions(512, 'kogge-stone') / sequentialAdditions(512);
    expect(ratio).toBeGreaterThan(8);
    expect(ratio).toBeLessThan(9);
  });
  it('Brent-Kung does 2N - 2 - log2 N additions: 26 at 16 (p. 284), 11 at 8', () => {
    expect(orderAdditions(16, 'brent-kung')).toBe(26);
    expect(orderAdditions(8, 'brent-kung')).toBe(11);
    for (const n of [4, 32, 1024]) expect(orderAdditions(n, 'brent-kung')).toBe(2 * n - 2 - Math.log2(n));
  });
  it('float32 in loop order: the exclusive scan is the inclusive one shifted, bit for bit', () => {
    const x = FLIP.probs;
    const inc = inclusiveIn(x, 'loop');
    const exc = exclusiveIn(x, 'loop');
    let run = 0;
    const direct = x.map((v) => {
      const out = run;
      run = f32(run + f32(v));
      return out;
    });
    expect(exc.map(f32hex)).toEqual(direct.map(f32hex));
    expect(exc.slice(1).map(f32hex)).toEqual(inc.slice(0, -1).map(f32hex));
  });
  it('dyadic rows are exact in float32 in every order', () => {
    const d = [0.5, 0.25, 0.125, 0.0625, 0.0625];
    for (const o of ORDERS) expect(inclusiveIn(d, o)).toEqual(inclusive(d));
  });
});

describe('top-p rules (ch11-facts D2 to D4)', () => {
  it('p-crossing: 0.5 0.25 0.125 0.125 at p = 0.75: two tokens under every project\'s rule, in any arithmetic', () => {
    const d = [0.5, 0.25, 0.125, 0.125];
    for (const order of [undefined, ...ORDERS]) {
      expect(kept(d, 0.75, 'exclusive', order)).toBe(2);
      expect(kept(d, 0.75, 'mlx', order)).toBe(2);
      expect(kept(d, 0.75, 'ascending', order)).toBe(2);
      expect(kept(d, 0.75, 'inclusive', order)).toBe(1);
    }
    expect(topP(d, 0.75, 'exclusive').before).toEqual([0, 0.5, 0.75, 0.875]);
  });
  it('q-rule: 0.4 0.3 0.2 0.1 at p = 0.6: the off-by-one keeps 1, the projects keep 2', () => {
    const r = [0.4, 0.3, 0.2, 0.1];
    expect(kept(r, 0.6, 'inclusive')).toBe(1);
    expect(kept(r, 0.6, 'exclusive')).toBe(2);
    expect(kept(r, 0.6, 'ascending')).toBe(2);
    expect(kept(r, 0.6, 'mlx')).toBe(2);
  });
  it('the off-by-one can keep nothing at all', () => {
    expect(kept([0.5, 0.3, 0.2], 0.3, 'inclusive')).toBe(0);
    expect(kept([0.5, 0.3, 0.2], 0.3, 'exclusive')).toBe(1);
  });
  it('ties: mlx-lm keeps every token tied with the boundary token, the others cut by position', () => {
    const t = [0.25, 0.25, 0.25, 0.25];
    expect(kept(t, 0.5, 'exclusive')).toBe(2);
    expect(kept(t, 0.5, 'ascending')).toBe(2);
    expect(kept(t, 0.5, 'mlx')).toBe(4);
    const b = [0.4, 0.2, 0.2, 0.2];
    expect(kept(b, 0.5, 'exclusive')).toBe(2);
    expect(kept(b, 0.5, 'mlx')).toBe(4);
  });
  it('with probabilities that sum to 1 exactly, the exclusive and ascending rules agree on every preset and p', () => {
    for (const pr of PRESETS)
      for (let k = 1; k < 100; k++) {
        const p = k / 100;
        expect(topP(pr.probs, p, 'ascending').keep, `${pr.id} at ${p}`).toEqual(topP(pr.probs, p, 'exclusive').keep);
      }
  });
  it('every rule but the off-by-one keeps the top token', () => {
    for (const pr of ALL_PRESETS)
      for (const rule of RULES.filter((r) => r !== 'inclusive'))
        for (const k of [1, 30, 99]) expect(topP(pr.probs, k / 100, rule).keep[0]).toBe(true);
  });
  it('zero-probability tokens are never kept by the exclusive rule below p = 1', () => {
    const z = ALL_PRESETS.find((p) => p.id === 'zero-tail')!;
    expect(topP(z.probs, 0.99, 'exclusive').keep).toEqual([true, true, true, false, false, false]);
  });
});

describe('the row on which two orders disagree', () => {
  it('findOrderFlip(1) returns the pinned row', () => {
    const f = findOrderFlip(1)!;
    expect(f.probs).toEqual(FLIP.probs);
    expect(f.p).toBe(FLIP.p);
    expect(f.token).toBe(FLIP.token);
    expect(f.loop).toBe(FLIP.loop);
    expect(f.ks).toBe(FLIP.ks);
  });
  it('the row is hundredths summing to 1.00, sorted, and the mass before token 5 is exactly 0.85', () => {
    const h = FLIP.probs.map((v) => Math.round(v * 100));
    expect(h.reduce((a, b) => a + b, 0)).toBe(100);
    expect([...h].sort((a, b) => b - a)).toEqual(h);
    expect(exclusive(h)[FLIP.token]).toBe(85);
  });
  it('loop order lands one float below p and keeps token 5; Kogge-Stone lands on p and drops it', () => {
    expect(ulpSigned(FLIP.loop, FLIP.p)).toBe(-1);
    expect(ulpSigned(FLIP.ks, FLIP.p)).toBe(0);
    expect(f32hex(FLIP.loop)).toBe('0x3f599999');
    expect(f32hex(FLIP.ks)).toBe('0x3f59999a');
    expect(kept(FLIP.probs, FLIP.p, 'exclusive', 'loop')).toBe(6);
    expect(kept(FLIP.probs, FLIP.p, 'exclusive', 'kogge-stone')).toBe(5);
    expect(kept(FLIP.probs, FLIP.p, 'exclusive')).toBe(5);
  });
  it('the band marks token 5 "either" and every other token firmly', () => {
    const b = band(FLIP.probs, FLIP.p);
    expect(b.verdict[FLIP.token]).toBe('either');
    expect(b.verdict.filter((v) => v === 'either')).toHaveLength(1);
    expect(b.verdict.slice(0, 5)).toEqual(['keep', 'keep', 'keep', 'keep', 'keep']);
  });
});

describe('the rounding band', () => {
  it('gamma(k) is about k times 2^-24', () => {
    expect(gamma(10) / (10 * 2 ** -24)).toBeCloseTo(1, 5);
  });
  it('on 300 random rows and every p in hundredths, no order contradicts a firm verdict', () => {
    let firm = 0;
    let either = 0;
    for (let s = 1; s <= 300; s++) {
      let a = s;
      const rand = () => ((a = (a * 1664525 + 1013904223) >>> 0) / 4294967296);
      const probs = hundredthsRow(8 + (s % 5), rand);
      for (let k = 1; k < 100; k++) {
        const p = k / 100;
        const b = band(probs, p);
        for (const o of ORDERS) {
          const keep = topP(probs, p, 'exclusive', { kind: 'f32', order: o }).keep;
          b.verdict.forEach((v, i) => {
            if (v === 'either') either++;
            else {
              firm++;
              expect(keep[i], `seed ${s}, p ${p}, token ${i}, ${o}`).toBe(v === 'keep');
            }
          });
        }
      }
    }
    expect(firm).toBeGreaterThan(100 * either);
  });
});

// The C++ shown in section 11.1 (labs/ch11/snippets/s111_sequential.h), compiled and run,
// must give the model's floats bit for bit. Skipped when there is no C++ compiler.
const repo = path.resolve(__dirname, '../../..');
const header = () => fs.readFileSync(path.join(repo, 'labs/ch11/snippets/s111_sequential.h'), 'utf8');
describe('labs/ch11/snippets/s111_sequential.h as section 11.1 shows it', () => {
  it('holds the loop of Fig. 11.1', () => {
    expect(header()).toContain('output[0] = input[0];');
    expect(header()).toContain('output[i] = output[i - 1] + input[i];');
  });
  it('lines 11 to 13, which section 11.1 highlights, are the loop of Fig. 11.1', () => {
    const l = header().split('\n');
    expect(l[10]).toContain('output[0] = input[0];');
    expect(l[11]).toContain('for (unsigned int i = 1; i < N; ++i)');
    expect(l[12]).toContain('output[i] = output[i - 1] + input[i];');
    expect(fs.readFileSync(path.join(__dirname, '../chapters/ch11/S111.tsx'), 'utf8')).toContain('highlight={[11, 12, 13]}');
  });
  it('section 11.1 loads it from the snippet file', () => {
    expect(fs.readFileSync(path.join(__dirname, '../chapters/ch11/S111.tsx'), 'utf8')).toContain("useLabFile('ch11/snippets/s111_sequential.h')");
  });
});
const compiler = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
describe.skipIf(!compiler)('labs/ch11/snippets/s111_sequential.h, compiled and run', () => {
  const exe = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'ch11-seq-')), 's111_sequential_test');
  const src = path.join(repo, 'labs/ch11/snippets/check/s111_sequential_test.cpp');
  const run = (xs: number[]) => {
    const r = spawnSync(exe, xs.map(String), { encoding: 'utf8' });
    expect(r.status).toBe(0);
    return Object.fromEntries(r.stdout.trim().split('\n').map((l) => {
      const [name, ...hex] = l.split(' ');
      return [name, hex];
    }));
  };
  const hex = (v: number[]) => v.map((x) => f32hex(x).slice(2));
  it('builds without warnings', () => {
    const r = spawnSync(compiler!, ['-std=c++17', '-Wall', '-Wextra', '-O2', src, '-o', exe], { encoding: 'utf8' });
    expect(r.status, r.stderr).toBe(0);
    expect(r.stderr).toBe('');
  });
  const rows: number[][] = [BOOK_ORDERS, TOKEN_COUNTS, FLIP.probs, ...PRESETS.map((p) => p.probs), [0.1, 0.2, 0.3, 0.7, 1e-8, 3.3, 1e7, -2.5]];
  it('every row: inclusive and exclusive are the model\'s loop order, bit for bit, and the conversions agree', () => {
    for (const xs of rows) {
      const x = xs.map(f32);
      const inc = inclusiveIn(x, 'loop');
      const exc = exclusiveIn(x, 'loop');
      const out = run(xs);
      expect(out.inclusive).toEqual(hex(inc));
      expect(out.exclusive).toEqual(hex(exc));
      expect(out['to-excl']).toEqual(hex(toExclusive(inc)));
      expect(out['to-incl']).toEqual(hex(toInclusive(exc.slice(0, -1).concat(exc[exc.length - 1]), x).map((v, i) => (i === x.length - 1 ? f32(v) : v))));
      expect(out['to-incl']).toEqual(out.inclusive);
    }
  });
});
