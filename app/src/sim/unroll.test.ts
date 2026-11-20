import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { DEFAULTS, FACTORS, WHERE_SNIPPETS, bodyText, schedule, whereLives } from './unroll';

const at = (unroll: number, reorder = false, over = {}) => schedule({ ...DEFAULTS, unroll, reorder, ...over });

describe('branches (p. 144)', () => {
  it('16 iterations: 16 branches; unrolled by 4: 4; fully unrolled: none', () => {
    expect(at(1).branches).toBe(16);
    expect(at(2).branches).toBe(8);
    expect(at(4).branches).toBe(4);
    expect(at(8).branches).toBe(2);
    expect(at(16).branches).toBe(0);
    expect(at(16).cells.some((c) => c.kind === 'branch')).toBe(false);
  });
  it('rejects a factor that does not divide the trip count', () => {
    expect(() => at(3)).toThrow();
  });
});

describe('the default sliders: A-to-B latency 3, branch stall 2', () => {
  it('pinned totals the lesson quotes', () => {
    expect(at(1)).toMatchObject({ total: 128, stallDep: 48, stallBranch: 32 });
    expect(at(4)).toMatchObject({ total: 92, stallDep: 48, stallBranch: 8 });
    expect(at(4, true)).toMatchObject({ total: 44, stallDep: 0, stallBranch: 8 });
    expect(at(2, true)).toMatchObject({ total: 72, stallDep: 16, stallBranch: 16 });
    expect(at(16)).toMatchObject({ total: 80, stallDep: 48, stallBranch: 0 });
    expect(at(16, true)).toMatchObject({ total: 32, stallDep: 0, stallBranch: 0 });
  });
  it('unrolling alone removes branch stalls but not dependence stalls', () => {
    for (const u of FACTORS) expect(at(u).stallDep).toBe(48);
    expect(FACTORS.map((u) => at(u).stallBranch)).toEqual([32, 16, 8, 4, 0]);
  });
  it('every instruction issues exactly once', () => {
    for (const u of FACTORS)
      for (const r of [false, true]) {
        const t = at(u, r);
        expect(t.cells.filter((c) => c.kind === 'issueA').map((c) => c.i)).toHaveLength(16);
        expect(t.cells.filter((c) => c.kind === 'issueB').map((c) => c.i).sort((a, b) => a! - b!)).toEqual([...Array(16).keys()]);
        expect(t.total).toBe(32 + t.branches + t.stalls);
      }
  });
});

describe('reordering (pp. 144–145)', () => {
  it('with latency at least 1 and unroll at least 2, reordering strictly lowers stall cycles', () => {
    for (let latencyAB = 1; latencyAB <= 8; latencyAB++)
      for (const branchStall of [0, 2, 5])
        for (const u of FACTORS.filter((f) => f >= 2))
          expect(at(u, true, { latencyAB, branchStall }).stalls).toBeLessThan(at(u, false, { latencyAB, branchStall }).stalls);
  });
  it('changes nothing without unrolling, or with no latency to hide', () => {
    expect(at(1, true).total).toBe(at(1).total);
    expect(at(4, true, { latencyAB: 0 }).total).toBe(at(4, false, { latencyAB: 0 }).total);
  });
  it('B never issues before A(i)’s result is ready', () => {
    const t = at(2, true, { latencyAB: 5 });
    const issued = new Map<string, number>();
    t.cells.forEach((c, k) => c.i !== undefined && c.kind.startsWith('issue') && issued.set(c.kind + c.i, k));
    for (let i = 0; i < 16; i++) expect(issued.get('issueB' + i)! - issued.get('issueA' + i)!).toBeGreaterThanOrEqual(6);
  });
  it('body text in the book’s notation', () => {
    expect(bodyText(4, false)).toBe('A(i) B(i) A(i+1) B(i+1) A(i+2) B(i+2) A(i+3) B(i+3)');
    expect(bodyText(4, true)).toBe('A(i) A(i+1) A(i+2) A(i+3) B(i) B(i+1) B(i+2) B(i+3)');
  });
});

describe('where does x live? (pp. 145–146; ch06-facts A13)', () => {
  it('answers', () => {
    expect(Object.fromEntries(WHERE_SNIPPETS.map((s) => [s.id, whereLives(s).place]))).toEqual({
      scalar: 'registers',
      'runtime-bound': 'local memory',
      'by-hand': 'registers',
      'unroll-1': 'local memory',
      'runtime-index': 'local memory',
    });
  });
  it('a constant-bound loop the compiler is free to unroll gives registers', () => {
    expect(whereLives({ id: 'x', label: '', code: '', array: true, index: 'loop counter', loop: 'constant bound' }).place).toBe('registers');
  });
  it('every snippet shown is in labs/ch06/snippets/s66_where_x_lives.cu, word for word', () => {
    const file = fs.readFileSync(path.resolve(__dirname, '../../../labs/ch06/snippets/s66_where_x_lives.cu'), 'utf8');
    for (const s of WHERE_SNIPPETS) expect(file).toContain(s.code);
  });
});
