import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { stripRead } from './gemmbanks';
import { BOOK, ownerOf as mapOwner, threadTile } from './gemmmap';
import { BK, BM, BN, THREADS, TM, TN, coarseThread, ownerOf as remapOwner, remapGeometry, remapThread, type Width } from './gemmremap';

// Chapter 15's thread mappings live in three models built by different hands: section
// 15.3's in gemmmap.ts (sections 15.3 and 15.4) and again in gemmremap.ts (sections 15.5
// and 15.6, and gemmbanks.ts on top of it), and section 15.5's in gemmremap.ts and in the
// labs' kernels. This file checks that they agree with each other and with the code the
// labs run, so a page cannot teach one mapping while a lab tests another.

const repo = path.resolve(__dirname, '../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');

// ---------------------------------------------------------------- a small integer evaluator
// Evaluates the mapping lines of the labs' kernels (+, -, *, /, %, parentheses, names and
// numbers) with C's integer division, for non-negative operands.
function evalInt(expr: string, env: Record<string, number>): number {
  const toks = expr.match(/\d+|[A-Za-z_]\w*|[-+*/%()]/g) ?? [];
  let i = 0;
  const atom = (): number => {
    const t = toks[i++];
    if (t === '(') {
      const v = sum();
      if (toks[i++] !== ')') throw new Error('unbalanced: ' + expr);
      return v;
    }
    if (/^\d+$/.test(t)) return Number(t);
    if (!(t in env)) throw new Error(`unknown name ${t} in ${expr}`);
    return env[t];
  };
  const product = (): number => {
    let v = atom();
    while (toks[i] === '*' || toks[i] === '/' || toks[i] === '%') {
      const op = toks[i++];
      const r = atom();
      v = op === '*' ? v * r : op === '/' ? Math.trunc(v / r) : v % r;
    }
    return v;
  };
  const sum = (): number => {
    let v = product();
    while (toks[i] === '+' || toks[i] === '-') v = toks[i++] === '+' ? v + product() : v - product();
    return v;
  };
  const v = sum();
  if (i !== toks.length) throw new Error('trailing tokens in ' + expr);
  return v;
}

/** The right-hand side of `<decl> NAME = ...;` in a source file. */
const rhs = (src: string, name: string): string => {
  const m = new RegExp(`(?:constexpr int|const int)\\s+${name}\\s*=\\s*([^;]+);`).exec(src);
  if (!m) throw new Error(`no ${name}`);
  return m[1].trim();
};

describe('section 15.3 mapping: gemmmap.ts and gemmremap.ts agree', () => {
  it('the same book constants', () => {
    expect(BOOK).toEqual({ bM: BM, bN: BN, bK: BK, tM: TM, tN: TN, threads: THREADS });
  });
  it('every thread owns the same 8 x 8 patch in both models', () => {
    for (let t = 0; t < THREADS; t++) {
      const a = threadTile(t);
      const b = coarseThread(t);
      expect([a.tRow, a.tCol, a.ty, a.tx]).toEqual([b.tRow, b.tCol, b.ty, b.tx]);
    }
  });
  it('the inverse mappings name the same owner for every element of the block tile', () => {
    for (let r = 0; r < BM; r++)
      for (let c = 0; c < BN; c++) {
        const a = mapOwner(r, c);
        const b = remapOwner('coarse', 32, r, c);
        expect([a.t, a.row, a.col]).toEqual([b.t, b.r, b.c]);
      }
  });
  it("gemmbanks reads the A strip of the coarse layout at gemmmap's rows", () => {
    for (const warp of [0, 3, 7])
      for (const p of [0, 5]) {
        const acc = stripRead({ layout: 'coarse', strip: 'A', ld: 8, banks: 32, W: 32, warp, i: 2, p });
        for (const w of acc.words) expect(w.row).toBe(threadTile(w.t).tRow + p);
      }
  });
});

describe("section 15.5 mapping: gemmremap.ts and the labs' kernels agree", () => {
  // Labs 15.3 to 15.5 call the width W; labs 15.6 and 15.7 call it WARP (W is the weight
  // there); the Metal kernels compile it in as SIMD_WIDTH. Each source is evaluated along
  // its own chain of names down to tRow and tCol, the thread's top-left sub-tile.
  const labs = ['lab3_remap', 'lab4_padded', 'lab5_pipelined', 'lab6_linear', 'lab7_bughunt'];
  const CHAIN = ['warp', 'lane', 'lRow', 'lCol', 'wRow', 'wCol', 'tRow', 'tCol'];
  const has = (src: string, n: string) => new RegExp(`(?:constexpr int|const int)\\s+${n}\\s*=`).test(src);
  /** tRow and tCol of thread t, as the source computes them. */
  const position = (src: string, env0: Record<string, number>, t: number) => {
    const e: Record<string, number> = { ...env0, t };
    for (const n of CHAIN) if (has(src, n)) e[n] = evalInt(rhs(src, n).replace(/threadIdx\.x/g, 't'), e);
    return [e.tRow, e.tCol];
  };
  const check = (src: string, constants: (W: Width) => Record<string, number>) => {
    for (const W of [32, 64] as Width[]) {
      const env = constants(W);
      const g = remapGeometry(W);
      expect([env.LC, env.QR, env.QC, env.WC]).toEqual([g.LC, g.QR, g.QC, g.WC]);
      for (let t = 0; t < THREADS; t++) {
        const want = remapThread(t, W).subTiles[0];
        expect(position(src, env, t)).toEqual([want.row, want.col]);
      }
    }
  };

  for (const lab of labs) {
    for (const variant of ['solution', 'starter'] as const) {
      const file = `labs/ch15/${lab}/${variant}/kernel.cu`;
      const src = read(file);
      const width = /template <int (\w+)>\s*__global__/.exec(src)?.[1];
      // The starters of labs 15.3 and 15.6 leave the mapping to the reader.
      if (!width || !has(src, 'tRow') || !has(src, 'LC')) {
        it.skip(`${file}: the mapping is the reader's to write`, () => {});
        continue;
      }
      it(`${file}: every thread at widths 32 and 64 starts its sub-tiles where gemmremap says`, () => {
        check(src, (W) => {
          const env: Record<string, number> = { [width]: W, bN: BN };
          for (const n of ['LC', 'QR', 'QC', 'WC']) env[n] = evalInt(rhs(src, n), env);
          return env;
        });
      });
    }
    const mfile = `labs/ch15/${lab}/solution/kernel.metal`;
    it(`${mfile}: the same layout from SIMD_WIDTH`, () => {
      const src = read(mfile);
      check(src, (W) => {
        const env: Record<string, number> = { SIMD_WIDTH: W, bN: BN };
        for (const n of ['LC', 'QR', 'QC', 'WC']) {
          const m = new RegExp(`#define ${n} \\(([^\\n]+?)\\)\\s*(?://.*)?$`, 'm').exec(src);
          if (!m) throw new Error(`${mfile}: no #define ${n}`);
          env[n] = evalInt(m[1], env);
        }
        return env;
      });
    });
  }

  it("gemmbanks reads the A strip of the remapped layout at gemmremap's rows", () => {
    for (const W of [32, 64] as Width[])
      for (const p of [0, 3, 4, 7]) {
        const acc = stripRead({ layout: 'remap', strip: 'A', ld: 9, banks: 32, W, warp: 1, i: 5, p });
        const g = remapGeometry(W);
        for (const w of acc.words) {
          const rt = remapThread(w.t, W);
          expect(w.row).toBe(rt.wRow + Math.floor(p / 4) * g.QR + rt.lRow + (p % 4));
          expect(w.word).toBe(w.row * 9 + 5);
        }
      }
  });
});
