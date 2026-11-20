import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  CATEGORIES,
  ROWS,
  SCENARIOS,
  SYMPTOMS,
  categoryCounts,
  countLine,
  rankRows,
  rowById,
  sortScore,
  type Placement,
} from './checklist';

const facts = fs.readFileSync(path.resolve(__dirname, '../../../notes/today/ch06-facts.md'), 'utf8');
/** The text of one facts item, e.g. "D3", up to the next heading. */
function factItem(id: string): string {
  const start = facts.indexOf(`### ${id}.`);
  if (start < 0) throw new Error('no facts item ' + id);
  const next = facts.indexOf('\n### ', start + 5);
  return facts.slice(start, next < 0 ? undefined : next);
}

describe('the table (Fig. 6.13, p. 148)', () => {
  it('has twelve rows: 3 compute + 6 memory + 2 synchronisation + 1 general', () => {
    expect(ROWS).toHaveLength(12);
    const c = categoryCounts();
    expect(c).toEqual({ compute: 3, memory: 6, sync: 2, general: 1 });
    expect(countLine(c)).toBe('3 + 6 + 2 + 1');
  });
  it('keeps the book order of rows', () => {
    expect(ROWS.map((r) => r.id)).toEqual([
      'occupancy', 'unroll', 'divergence', 'coalesce', 'smem-tiling', 'reg-tiling', 'vector', 'banks', 'privatise', 'warp',
      'double-buffer', 'coarsen',
    ]);
  });
  it('has unique ids, pages, strategies and commentary on every row', () => {
    expect(new Set(ROWS.map((r) => r.id)).size).toBe(12);
    for (const r of ROWS) {
      expect(r.pages).toMatch(/\d/);
      expect(r.pages).toMatch(/148|149|139/);
      expect(r.strategies.length).toBeGreaterThan(0);
      expect(r.commentary.length).toBeGreaterThan(40);
      expect(r.why.length).toBeGreaterThan(10);
    }
  });
  it('lists no memory benefit for exactly two rows: divergence and double buffering', () => {
    expect(ROWS.filter((r) => r.benefitMemory === null).map((r) => r.id)).toEqual(['divergence', 'double-buffer']);
  });
  it('gives coarsening the same "depends" benefit in both columns', () => {
    const c = rowById('coarsen');
    expect(c.category).toBe('general');
    expect(c.benefitCompute).toBe(c.benefitMemory);
    expect(c.benefitCompute).toMatch(/[Dd]epends/);
  });
  it('fixes the vector-load strategy at 16-byte chunks (p. 148)', () => {
    expect(rowById('vector').strategies.join(' ')).toMatch(/16-byte/);
  });
});

describe('costs', () => {
  it('cites a page or a facts item, or says "not stated in the book"', () => {
    for (const r of ROWS) {
      if (r.cost.source) expect(r.cost.source).toMatch(/^(p\.|pp\.)|facts A\d+/);
      else expect(r.cost.text).toMatch(/^Not stated in the book\./);
    }
  });
  it('states the costs the chapter gives, with their pages', () => {
    expect(rowById('smem-tiling').cost.source).toBe('p. 153');
    expect(rowById('banks').cost.source).toBe('pp. 140–141');
    expect(rowById('coarsen').cost.source).toBe('p. 143');
    expect(rowById('double-buffer').cost.source).toBe('p. 147');
    expect(rowById('vector').cost.source).toMatch(/facts A11/);
  });
  it('the vector-load cost matches facts A11 (register pressure, alignment)', () => {
    const a11 = factItem('A11');
    expect(a11).toMatch(/increase register pressure/);
    expect(a11).toMatch(/require aligned data/);
    expect(rowById('vector').cost.text).toMatch(/register pressure/);
    expect(rowById('vector').cost.text).toMatch(/aligned/);
  });
});

describe('returns in later chapters (pp. 149 to 153)', () => {
  it('every row but occupancy names chapters 7 to 18', () => {
    for (const r of ROWS) {
      expect(r.returnsIn.length).toBeGreaterThan(0);
      for (const x of r.returnsIn) {
        expect(x.ch).toBeGreaterThanOrEqual(7);
        expect(x.ch).toBeLessThanOrEqual(18);
      }
    }
  });
  it('matches the forward references in the notes', () => {
    const chs = (id: Parameters<typeof rowById>[0]) => rowById(id).returnsIn.map((x) => x.ch);
    expect(chs('double-buffer')).toEqual([11, 15]);
    expect(chs('privatise')).toEqual([9, 12, 18]);
    expect(chs('warp')).toEqual([10, 11, 12, 15]);
    expect(chs('coarsen')).toEqual([8, 9, 10, 11, 12, 14, 15]);
    expect(chs('banks')).toEqual([11, 15]);
  });
});

describe('seen in real backends: only from facts D1 to D4', () => {
  it('every entry cites D1 to D4 and its token appears in that facts item', () => {
    let n = 0;
    for (const r of ROWS)
      for (const s of r.seenIn) {
        n++;
        expect(['D1', 'D2', 'D3', 'D4']).toContain(s.fact);
        expect(factItem(s.fact)).toContain(s.token);
      }
    expect(n).toBe(10);
  });
  it('fills exactly the rows the facts cover, and marks the absence in MLX', () => {
    const filled = ROWS.filter((r) => r.seenIn.length).map((r) => r.id);
    expect(filled).toEqual(['unroll', 'coalesce', 'smem-tiling', 'vector', 'banks', 'double-buffer']);
    const db = rowById('double-buffer').seenIn;
    expect(db.find((s) => s.fact === 'D3')?.absent).toBe(true);
    expect(db.find((s) => s.fact === 'D1')?.absent).toBeUndefined();
    expect(rowById('vector').seenIn.map((s) => s.fact)).toEqual(['D4', 'D1']);
    expect(rowById('banks').seenIn.map((s) => s.fact)).toEqual(['D3', 'D1']);
    expect(rowById('unroll').seenIn.map((s) => s.fact)).toEqual(['D3', 'D2']);
  });
});

describe('symptoms', () => {
  it('there are eight, each relieved by at least one row', () => {
    expect(SYMPTOMS).toHaveLength(8);
    for (const s of SYMPTOMS) {
      expect(s.relief.length).toBeGreaterThan(0);
      for (const r of [...s.relief, ...s.spends]) {
        expect(() => rowById(r.row)).not.toThrow();
        expect(r.page).toMatch(/^pp?\. \d/);
      }
      // a row cannot both relieve a symptom and spend what it is short of
      const rel = new Set(s.relief.map((r) => r.row));
      for (const r of s.spends) expect(rel.has(r.row)).toBe(false);
    }
  });
  it('ranks relieving rows first, in order, then the ones that spend, then the rest in book order', () => {
    const r = rankRows('global-stall');
    expect(r).toHaveLength(12);
    expect(r.slice(0, 4).map((x) => [x.row.id, x.role])).toEqual([
      ['occupancy', 'relieves'], ['vector', 'relieves'], ['unroll', 'relieves'], ['coalesce', 'relieves'],
    ]);
    expect(r.slice(4, 6).map((x) => [x.row.id, x.role])).toEqual([['smem-tiling', 'spends'], ['coarsen', 'spends']]);
    expect(r.slice(6).every((x) => x.role === 'other')).toBe(true);
    expect(r.slice(6).map((x) => x.row.id)).toEqual(['divergence', 'reg-tiling', 'banks', 'privatise', 'warp', 'double-buffer']);
  });
  it('book order when no symptom is chosen', () => {
    expect(rankRows(null).map((x) => x.row.id)).toEqual(ROWS.map((r) => r.id));
  });
  it('the claims the lesson makes about the tool hold', () => {
    // "Half the rows relieve more than one symptom"
    const many = ROWS.filter((r) => SYMPTOMS.filter((s) => s.relief.some((x) => x.row === r.id)).length > 1);
    expect(many).toHaveLength(6);
    // "Coarsening relieves six of the eight symptoms"
    expect(SYMPTOMS.filter((s) => s.relief.some((r) => r.row === 'coarsen'))).toHaveLength(6);
    // "for three symptoms a row is marked in red"
    expect(SYMPTOMS.filter((s) => s.spends.length).map((s) => s.id)).toEqual(['global-stall', 'shared-serial', 'instructions']);
    // the bank-conflict symptom puts padding first and flags corner turning
    const b = rankRows('shared-serial');
    expect(b[0].row.id).toBe('banks');
    expect(b.find((x) => x.role === 'spends')?.row.id).toBe('coalesce');
    // too much global traffic: tiling first
    expect(rankRows('global-traffic')[0].row.id).toBe('smem-tiling');
    expect(rankRows('barrier')[0].row.id).toBe('double-buffer');
    expect(rankRows('atomics')[0].row.id).toBe('privatise');
    expect(rankRows('divergence')[0].row.id).toBe('divergence');
  });
});

describe('sorting drill', () => {
  it('scores the book answer 12 of 12 with the 3 + 6 + 2 + 1 tally', () => {
    const p: Placement = Object.fromEntries(ROWS.map((r) => [r.id, r.category]));
    const s = sortScore(p);
    expect(s.done).toBe(true);
    expect(s.right).toBe(12);
    for (const c of CATEGORIES) expect(s.tally[c].yours).toBe(s.tally[c].book);
    expect(CATEGORIES.map((c) => s.tally[c].book)).toEqual([3, 6, 2, 1]);
  });
  it('counts a misfiled card and partial placements', () => {
    const s = sortScore({ coarsen: 'memory', occupancy: 'compute' });
    expect(s.placed).toBe(2);
    expect(s.right).toBe(1);
    expect(s.done).toBe(false);
    expect(s.tally.memory.yours).toBe(1);
    expect(s.tally.general.yours).toBe(0);
  });
});

describe('scenarios', () => {
  it('cover all three outcomes, each tied to a row and a page', () => {
    expect(new Set(SCENARIOS.map((s) => s.expect))).toEqual(new Set(['faster', 'same', 'slower']));
    for (const s of SCENARIOS) {
      expect(() => rowById(s.row)).not.toThrow();
      expect(s.page).toMatch(/^pp?\. \d/);
      expect(s.why.length).toBeGreaterThan(40);
    }
    expect(new Set(SCENARIOS.map((s) => s.id)).size).toBe(SCENARIOS.length);
  });
  it('the coarsening card uses the book’s wave arithmetic (p. 143)', () => {
    // 1056 blocks on 264 slots: 4 waves; coarsened by 8: 132 blocks, half a wave.
    expect(1056 / 264).toBe(4);
    expect(1056 / 8 / 264).toBe(0.5);
    expect(SCENARIOS.find((s) => s.id === 'coarsen-small-grid')?.why).toMatch(/1056 blocks on 264 slots coarsened by 8 is half a wave/);
  });
});
