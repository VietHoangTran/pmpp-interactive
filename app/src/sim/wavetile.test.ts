import { describe, expect, it } from 'vitest';
import {
  activeThreads,
  blocksAt,
  busyPerStep,
  cellAt,
  cpuTable,
  fill,
  launchPlan,
  launches,
  loadsOf,
  makeSequences,
  producerOf,
  runTile,
  schedule,
  slotCell,
  summary,
  tilesPerSide,
  tileOutline,
  tileWork,
  type Geometry,
  type Scheme,
} from './wavetile';

const tilesOf = (g: Geometry, d: number) => launchPlan(g, d).tiles.map((t) => `(${t.row},${t.col})`).join(' ');

describe('schedule counts (ch16-facts E2)', () => {
  const cases: [number, [number, number, number], [number, number, number]][] = [
    [3, [5, 3, 9], [8, 2, 12]],
    [4, [7, 4, 16], [11, 3, 20]],
    [8, [15, 8, 64], [23, 5, 72]],
  ];
  for (const [T, sq, sh] of cases)
    it(`T = ${T}: square ${sq.join('/')}, sheared ${sh.join('/')} (launches / most tiles / tile computations)`, () => {
      for (const W of [1, 3, 4]) {
        const s = summary({ Lseq: T * W, W, scheme: 'square' });
        const h = summary({ Lseq: T * W, W, scheme: 'sheared' });
        expect([s.launches, s.mostTiles, s.blocks]).toEqual(sq);
        expect([h.launches, h.mostTiles, h.blocks]).toEqual(sh);
      }
    });

  it('launches are 2T - 1 and 3T - 1; sheared fronts hold at most floor(T/2) + 1 tiles', () => {
    for (let T = 1; T <= 20; T++) {
      expect(launches({ Lseq: 5 * T, W: 5, scheme: 'square' })).toBe(2 * T - 1);
      expect(launches({ Lseq: 5 * T, W: 5, scheme: 'sheared' })).toBe(3 * T - 1);
      expect(summary({ Lseq: 5 * T, W: 5, scheme: 'sheared' }).mostTiles).toBe(Math.floor(T / 2) + 1);
      expect(summary({ Lseq: 5 * T, W: 5, scheme: 'square' }).mostTiles).toBe(T);
    }
  });

  it('the sheared T = 3 schedule is the notes\' list, each tile row two tile columns behind the one above', () => {
    const g: Geometry = { Lseq: 9, W: 3, scheme: 'sheared' };
    expect(schedule(g).map((p) => tilesOf(g, p.d))).toEqual([
      '(0,0)', '(0,1)', '(0,2) (1,0)', '(0,3) (1,1)', '(1,2) (2,0)', '(1,3) (2,1)', '(2,2)', '(2,3)',
    ]);
  });

  it('the square T = 3 schedule is Fig. 16.7(b): 1, 2, 3, 2, 1 tiles', () => {
    expect(summary({ Lseq: 9, W: 3, scheme: 'square' }).perLaunch).toEqual([1, 2, 3, 2, 1]);
  });

  it('T = 5, d = 6 (the in-text exercise of p. 386): 3 blocks, block 0 takes tile row 2', () => {
    const g: Geometry = { Lseq: 50, W: 10, scheme: 'square' };
    const p = launchPlan(g, 6);
    expect(p.firstRow).toBe(2);
    expect(tilesOf(g, 6)).toBe('(2,4) (3,3) (4,2)');
    // the closed form
    const T = 5;
    for (let d = 0; d < 2 * T - 1; d++) expect(blocksAt(g, d)).toBe(Math.min(d, T - 1) - Math.max(0, d - T + 1) + 1);
  });

  it('q-launches: L_seq = 1000, W = 32 gives 63 square, 95 sheared, and 1,999 for the baseline', () => {
    expect(tilesPerSide(1000, 32)).toBe(32);
    expect(launches({ Lseq: 1000, W: 32, scheme: 'square' })).toBe(63);
    expect(launches({ Lseq: 1000, W: 32, scheme: 'sheared' })).toBe(95);
    expect(launches({ Lseq: 1000, W: 1, scheme: 'square' })).toBe(1999);
  });

  it('p-hyper-launches: a 9 x 9 interior in 3 x 3 tiles takes 8 sheared launches against 5 square', () => {
    expect(launches({ Lseq: 9, W: 3, scheme: 'sheared' })).toBe(8);
    expect(launches({ Lseq: 9, W: 3, scheme: 'square' })).toBe(5);
  });
});

describe('W = 1 square tiles are the baseline', () => {
  it('2 L_seq - 1 launches, launch d computing exactly the anti-diagonal r + q = d + 2', () => {
    for (const Lseq of [1, 2, 5, 12]) {
      const g: Geometry = { Lseq, W: 1, scheme: 'square' };
      expect(launches(g)).toBe(2 * Lseq - 1);
      for (const p of schedule(g)) {
        const cells = p.tiles.map((t) => cellAt(g, t, 0, 0)).filter((c) => c.active);
        expect(cells.every((c) => c.r + c.q === p.d + 2)).toBe(true);
        const want = Math.min(p.d + 1, Lseq) - Math.max(1, p.d + 2 - Lseq) + 1;
        expect(cells.length).toBe(want);
      }
    }
  });
});

describe('every cell once, after its inputs, from the right memory', () => {
  for (const scheme of ['square', 'sheared'] as Scheme[])
    it(`${scheme}: L_seq 1..40, W 1..9`, () => {
      for (let Lseq = 1; Lseq <= 40; Lseq++) {
        const { a, b } = makeSequences(Lseq, Lseq);
        const ref = cpuTable(a, b);
        for (let W = 1; W <= 9; W++) {
          const f = fill({ Lseq, W, scheme }, a, b);
          expect(f.problems).toEqual([]);
          expect(f.neverComputed).toBe(0);
          expect(f.missing).toBe(0);
          for (let r = 1; r <= Lseq; r++)
            for (let q = 1; q <= Lseq; q++) {
              const i = r * f.L + q;
              expect(f.computed[i]).toBe(1);
              expect(f.stored[i]).toBe(1);
              expect(f.H[i]).toBe(ref[i]);
            }
        }
      }
    });

  it('a cell\'s producer is the thread, step and slot that compute it (round trip)', () => {
    for (const scheme of ['square', 'sheared'] as Scheme[])
      for (const W of [1, 2, 3, 5]) {
        const g: Geometry = { Lseq: 4 * W, W, scheme };
        for (const p of schedule(g))
          for (const tile of p.tiles)
            for (let k = 0; k < (scheme === 'square' ? 2 * W - 1 : W); k++)
              for (const t of activeThreads(g, tile, k)) {
                const c = cellAt(g, tile, k, t);
                const pr = producerOf(g, c.r, c.q)!;
                expect([pr.tile.row, pr.tile.col, pr.d, pr.dTile, pr.thread, ...pr.slot]).toEqual([tile.row, tile.col, p.d, k, t, c.rTile, c.qTile]);
                expect(slotCell(g, tile, c.rTile, c.qTile)).toEqual({ r: c.r, q: c.q });
              }
      }
  });
});

describe('the sheared load offsets (Fig. 16.17)', () => {
  it('north and west are one stored column left, north-west two', () => {
    const g: Geometry = { Lseq: 24, W: 8, scheme: 'sheared' };
    const c = cellAt(g, { row: 1, col: 1 }, 5, 3);
    const [n, w, nw] = loadsOf(g, c);
    expect([n.from, n.slot]).toEqual(['tile', [2, 4]]);
    expect([w.from, w.slot]).toEqual(['tile', [3, 4]]);
    expect([nw.from, nw.slot]).toEqual(['tile', [2, 3]]);
    // and each slot holds the cell the load stands for
    for (const ld of [n, w, nw]) expect(slotCell(g, { row: 1, col: 1 }, ...ld.slot!)).toEqual({ r: ld.r, q: ld.q });
  });
});

describe('work per step (Eqs. 16.3, 16.4)', () => {
  it('W = 8: 64 cells in 15 steps square, in 8 sheared; W = 32: 16.25 against 32', () => {
    expect(tileWork(8, 'square')).toEqual({ cells: 64, steps: 15, perStep: 64 / 15 });
    expect(tileWork(8, 'sheared')).toEqual({ cells: 64, steps: 8, perStep: 8 });
    expect(tileWork(32, 'square')).toEqual({ cells: 1024, steps: 63, perStep: 1024 / 63 }); // 16.25 to two places, as the facts file rounds it
    expect(tileWork(32, 'square').perStep).toBeCloseTo(16.25, 1);
    expect(tileWork(32, 'sheared').perStep).toBe(32);
  });
  it('q-idle: a 16 x 16 square tile keeps 256 / 31 = 8.26 threads busy per step', () => {
    expect(tileWork(16, 'square').perStep).toBeCloseTo(8.258, 3);
    expect(busyPerStep(3, 'square')).toEqual([1, 2, 3, 2, 1]);
    expect(busyPerStep(3, 'sheared')).toEqual([3, 3, 3]);
  });
  it('a full interior tile has those busy counts', () => {
    for (const scheme of ['square', 'sheared'] as Scheme[]) {
      const g: Geometry = { Lseq: 24, W: 8, scheme };
      const busy = busyPerStep(8, scheme).map((_, k) => activeThreads(g, { row: 1, col: 1 }, k).length);
      expect(busy).toEqual(busyPerStep(8, scheme));
    }
  });
});

describe('the barrier between steps (one tile, any thread order)', () => {
  for (const scheme of ['square', 'sheared'] as Scheme[]) {
    const g: Geometry = { Lseq: 18, W: 6, scheme };
    const tile = { row: 1, col: 1 };
    it(`${scheme}: with the barrier every order is correct`, () => {
      for (const order of ['lockstep', 'first', 'last'] as const) {
        const run = runTile(g, tile, { order, barrier: true });
        expect(run.correct).toBe(true);
        expect(run.firstBad).toBeNull();
        expect(run.cells).toBe(36);
      }
    });
    it(`${scheme}: without it, last thread first reads a slot before it is written`, () => {
      const run = runTile(g, tile, { order: 'last', barrier: false });
      expect(run.correct).toBe(false);
      const bad = run.ticks[run.firstBad!].parts.flatMap((p) => p.reads).find((r) => r.problem);
      expect(bad?.problem).toBe('read-before-write');
      expect(run.ticks[run.firstBad!].parts[0].thread).toBe(5);
    });
    it(`${scheme}: without it, thread 0 first computes right but stores too early`, () => {
      const run = runTile(g, tile, { order: 'first', barrier: false });
      expect(run.correct).toBe(false);
      expect(run.ticks[run.firstBad!].kind).toBe('store');
      expect(run.ticks[run.firstBad!].parts[0].thread).toBe(0);
    });
    it(`${scheme}: lockstep hides the missing barrier`, () => {
      expect(runTile(g, tile, { order: 'lockstep', barrier: false }).correct).toBe(true);
    });
  }

  it('a left-edge sheared tile relies on its zeroed slots; without initialize_tile it reads garbage', () => {
    const g: Geometry = { Lseq: 18, W: 6, scheme: 'sheared' };
    const ok = runTile(g, { row: 1, col: 0 }, { order: 'lockstep', barrier: true, zeroInit: true });
    expect(ok.correct).toBe(true);
    // some reads go to zeroed slots that stand for column 0
    const zeroReads = ok.ticks.flatMap((t) => t.parts.flatMap((p) => p.reads)).filter((r) => r.from === 'tile' && r.q === 0);
    expect(zeroReads.length).toBeGreaterThan(0);
    expect(zeroReads.every((r) => r.value === 0 && r.problem === null)).toBe(true);
    const bad = runTile(g, { row: 1, col: 0 }, { order: 'lockstep', barrier: true, zeroInit: false });
    expect(bad.correct).toBe(false);
    expect(bad.ticks[bad.firstBad!].parts.flatMap((p) => p.reads).some((r) => r.problem === 'uninitialised')).toBe(true);
    // an interior sheared tile reads no slot outside the table, so it does not need the zeroing
    expect(runTile(g, { row: 1, col: 1 }, { order: 'lockstep', barrier: true, zeroInit: false }).correct).toBe(true);
  });
});

describe('the "<" defect in the sheared kernel', () => {
  it('skips the last tile column, so right-edge cells are never computed', () => {
    for (const [Lseq, W] of [[9, 3], [24, 8], [20, 6]]) {
      const g: Geometry = { Lseq, W, scheme: 'sheared', lastColLess: true };
      const { a, b } = makeSequences(Lseq);
      const f = fill(g, a, b);
      expect(f.problems).toEqual([]);
      expect(f.neverComputed).toBeGreaterThan(0);
      expect(f.missing).toBeGreaterThanOrEqual(f.neverComputed);
      const T = tilesPerSide(Lseq, W);
      for (let r = 1; r <= Lseq; r++)
        for (let q = 1; q <= Lseq; q++)
          if (f.computed[r * f.L + q] === 0) expect(producerOf(g, r, q)!.tile.col).toBe(T);
      // the launch count is unchanged; the last launch is empty
      expect(launches(g)).toBe(3 * T - 1);
      expect(launchPlan(g, 3 * T - 2).tiles).toEqual([]);
    }
  });
  it('9 x 9 in 3 x 3 tiles: 3 tiles skipped, 9 cells never computed', () => {
    const g: Geometry = { Lseq: 9, W: 3, scheme: 'sheared', lastColLess: true };
    const { a, b } = makeSequences(9);
    const f = fill(g, a, b);
    expect(f.blocks).toBe(9);
    expect(f.neverComputed).toBe(9);
  });
});

describe('tile outlines', () => {
  it('a square tile is a square; a sheared tile leans one column left per row', () => {
    const sq = tileOutline({ W: 3, scheme: 'square' }, { row: 1, col: 1 });
    const xs = sq.map((p) => p[0]);
    const ys = sq.map((p) => p[1]);
    expect([Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]).toEqual([4, 7, 4, 7]);
    const sh = tileOutline({ W: 3, scheme: 'sheared' }, { row: 1, col: 1 });
    // row 4 spans columns 4..6, row 5 columns 3..5, row 6 columns 2..4 (Fig. 16.14(c))
    expect(sh.slice(0, 6)).toEqual([[7, 4], [7, 5], [6, 5], [6, 6], [5, 6], [5, 7]]);
    expect(sh.slice(6)).toEqual([[2, 7], [2, 6], [3, 6], [3, 5], [4, 5], [4, 4]]);
  });
});
