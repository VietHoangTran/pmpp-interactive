import { describe, expect, it } from 'vitest';
import {
  BOOK_PUZZLE,
  FIG_KEEP,
  SCENE,
  blockDependencies,
  chainTimeline,
  dedupFlags,
  dedupMoves,
  dependencePath,
  exclusiveScan,
  keepOf,
  moveByIndex,
  orderingTable,
  orderingVerdict,
  overlap,
  sceneRuns,
} from './inplace';

describe('overlap: the book\'s Fig. 12.10 example, two blocks of eight keys (p. 300)', () => {
  const o = overlap(FIG_KEEP, 8);
  const w = (k: number) => o.writes.find((x) => x.key === k)!;

  it('keeps eight keys and compacts them in order', () => {
    expect(o.count).toBe(8);
    expect(o.after.slice(0, 8)).toEqual([1, 3, 4, 6, 8, 10, 11, 14]);
  });
  it('k3 lands in slot 1, where k1 lived: same block, a barrier orders it', () => {
    expect(w(3)).toMatchObject({ from: 3, to: 1, victim: 1, writer: 0, reader: 0, kind: 'same' });
  });
  it('k1 lands in slot 0, over k0 (dropped, but still read first)', () => {
    expect(w(1)).toMatchObject({ to: 0, victim: 0, kind: 'same' });
  });
  it('k8 lands in slot 4, where k4 lived, read by the other block', () => {
    expect(w(8)).toMatchObject({ from: 8, to: 4, victim: 4, writer: 1, reader: 0, kind: 'earlier' });
  });
  it('q-crossblock: all four of block 1\'s writes land in block 0\'s range', () => {
    const b1 = o.writes.filter((x) => x.writer === 1);
    expect(b1.map((x) => x.to)).toEqual([4, 5, 6, 7]);
    expect(b1.filter((x) => x.reader === 0)).toHaveLength(4);
    expect(o.pairs).toEqual([{ reader: 0, writer: 1, writes: 4 }]);
  });
  it('removal never writes into a later block', () => {
    for (const per of [4, 6, 8]) for (const keep of [FIG_KEEP, keepOf(32, [0, 5, 6, 7, 9, 12, 20, 21, 30, 31])]) {
      const r = overlap(keep, per);
      expect(r.counts.later).toBe(0);
      expect(r.writes.every((x) => x.to <= x.from)).toBe(true);
    }
  });
  it('four blocks of four (Fig. 12.9 layout): cross-block pairs only run forwards', () => {
    const r = overlap(FIG_KEEP, 4);
    expect(r.blocks).toBe(4);
    expect(r.pairs.every((p) => p.reader < p.writer)).toBe(true);
    expect(r.counts).toEqual({ own: 0, same: 2, earlier: 6, later: 0, fresh: 0 });
  });
  it('a block that keeps everything before it writes onto its own slots', () => {
    const r = overlap(keepOf(8, [0, 1, 2, 5]), 4);
    expect(r.writes.slice(0, 3).map((x) => x.kind)).toEqual(['own', 'own', 'own']);
    expect(r.writes[3]).toMatchObject({ key: 5, to: 3, reader: 0, writer: 1, kind: 'earlier' });
  });
});

describe('overlap: the reverse movement (insertion, §12.8)', () => {
  const o = overlap(FIG_KEEP, 8, 'insert');
  it('starts from the compacted list and puts each key back where it came from', () => {
    expect(o.before.slice(0, 8)).toEqual([1, 3, 4, 6, 8, 10, 11, 14]);
    expect(o.after.filter((v) => v != null)).toEqual([1, 3, 4, 6, 8, 10, 11, 14]);
  });
  it('keys move outwards, and the overlaps point to later blocks', () => {
    expect(o.counts.earlier).toBe(0);
    expect(o.writes.every((x) => x.to >= x.from)).toBe(true);
    // k8 goes back from slot 4 to slot 8; slot 8 is past the eight kept keys, so nothing lived there.
    expect(o.writes.find((x) => x.key === 8)).toMatchObject({ from: 4, to: 8, kind: 'fresh', reader: null });
    // k6 goes from slot 3 to slot 6, over k11 (compacted slot 6), read by block 0 itself.
    expect(o.writes.find((x) => x.key === 6)).toMatchObject({ from: 3, to: 6, victim: 11, kind: 'same' });
  });
  it('with four blocks of four, block 1 must read before block 0 writes', () => {
    const r = overlap(FIG_KEEP, 4, 'insert');
    expect(r.counts.later).toBeGreaterThan(0);
    expect(r.pairs.every((p) => p.reader > p.writer)).toBe(true);
    // k4: from slot 2 (block 0) to slot 4, over k8 (compacted slot 4), read by block 1.
    expect(r.writes.find((x) => x.key === 4)).toMatchObject({ from: 2, to: 4, victim: 8, writer: 0, reader: 1, kind: 'later' });
  });
});

describe('dependencePath (Fig. 12.11)', () => {
  it('read(0) -> scan(0) -> scan(1) -> write(1)', () => {
    expect(dependencePath(0, 1)).toEqual([
      { op: 'read', block: 0 },
      { op: 'scan', block: 0 },
      { op: 'scan', block: 1 },
      { op: 'write', block: 1 },
    ]);
  });
  it('the chain passes through every block in between', () => {
    expect(dependencePath(1, 3)!.map((x) => `${x.op}${x.block}`)).toEqual(['read1', 'scan1', 'scan2', 'scan3', 'write3']);
  });
  it('a forward chain cannot order a later read before an earlier write', () => {
    expect(dependencePath(2, 0)).toBeNull();
  });
  it('insertion runs the chain from the last block to the first', () => {
    expect(dependencePath(2, 0, 'insert')!.map((x) => `${x.op}${x.block}`)).toEqual(['read2', 'scan2', 'scan1', 'scan0', 'write0']);
    expect(dependencePath(0, 2, 'insert')).toBeNull();
  });
  it('inside one block it is a barrier', () => {
    expect(dependencePath(1, 1)!.map((x) => x.op)).toEqual(['read', 'barrier', 'write']);
  });
});

describe('orderingVerdict: the ordering puzzle', () => {
  it('the book\'s kernel: barriers on both sides, release and acquire: every pair is ordered', () => {
    expect(orderingTable(BOOK_PUZZLE).every((x) => x.verdict.ordered)).toBe(true);
    const v = orderingVerdict(BOOK_PUZZLE, 'other', 'other');
    expect(v.links.map((l) => l.ok)).toEqual([true, true, true]);
    expect(v.links[1].why).toMatch(/synchronizes with/);
  });
  it('a relaxed store on either side breaks every pair', () => {
    for (const cfg of [
      { ...BOOK_PUZZLE, publish: 'relaxed' as const },
      { ...BOOK_PUZZLE, observe: 'relaxed' as const },
      { ...BOOK_PUZZLE, publish: 'relaxed' as const, observe: 'relaxed' as const },
    ]) {
      expect(orderingTable(cfg).some((x) => x.verdict.ordered)).toBe(false);
    }
    expect(orderingVerdict({ ...BOOK_PUZZLE, publish: 'relaxed' }, 'leader', 'leader').missing).toMatch(/relaxed store/);
    expect(orderingVerdict({ ...BOOK_PUZZLE, observe: 'relaxed' }, 'leader', 'leader').missing).toMatch(/relaxed load/);
  });
  it('without the barrier in block i only the leader\'s own reads are ordered', () => {
    const t = orderingTable({ ...BOOK_PUZZLE, barrierI: false });
    expect(t.filter((x) => x.verdict.ordered).map((x) => x.reader)).toEqual(['leader', 'leader']);
  });
  it('without the barrier in block j only the leader\'s own writes are ordered', () => {
    const t = orderingTable({ ...BOOK_PUZZLE, barrierJ: false });
    expect(t.filter((x) => x.verdict.ordered).map((x) => x.writer)).toEqual(['leader', 'leader']);
  });
});

describe('chainTimeline: the scene\'s runs', () => {
  const { overlap: ov, dynamic, blockIdx } = sceneRuns();

  it('beat 1: twelve keys, three blocks of four; five arrows land in an earlier block\'s range', () => {
    expect(ov.blocks).toBe(3);
    expect(ov.counts.earlier).toBe(5);
    expect(ov.writes.filter((w) => w.kind === 'earlier').map((w) => w.key)).toEqual([4, 6, 8, 10, 11]);
  });
  it('dynamic numbering: blockIdx 2 starts first and takes number 0, blockIdx 0 starts last and takes 2', () => {
    const first = dynamic.frames[1].blocks;
    expect(first.find((b) => b.hw === 2)!.num).toBe(0);
    expect(first.find((b) => b.hw === 1)!.num).toBe(1);
    expect(first.find((b) => b.hw === 0)!.phase).toBe('queued');
    const last = dynamic.frames[dynamic.frames.length - 1].blocks;
    expect(last.find((b) => b.hw === 0)!.num).toBe(2);
  });
  it('dynamic numbering finishes, with no write before the read it overwrites, and the right result', () => {
    expect(dynamic.outcome).toBe('finished');
    expect(dynamic.hazards).toBe(0);
    expect(dynamic.correct).toBe(true);
    expect(dynamic.frames[dynamic.frames.length - 1].array.slice(0, 7)).toEqual([1, 3, 4, 6, 8, 10, 11]);
  });
  it('the order of events: number 1 waits while number 0 publishes, then publishes one step later', () => {
    const f = dynamic.frames;
    const ph = (k: number, num: number) => f[k].blocks.find((b) => b.num === num)?.phase;
    expect([ph(2, 0), ph(2, 1)]).toEqual(['read', 'read']);
    expect([ph(3, 0), ph(3, 1)]).toEqual(['scan', 'scan']);
    expect([ph(4, 0), ph(4, 1)]).toEqual(['publish', 'wait']);
    expect([ph(5, 0), ph(5, 1)]).toEqual(['write', 'publish']);
    expect(f[4].flagAt[0]).toBe(4);
    expect(f[5].flagAt[1]).toBe(5);
    expect(dynamic.frames).toHaveLength(12);
  });
  it('blockIdx numbering with the same start order: 2 waits on 1, 1 waits on 0, and 0 never gets a slot', () => {
    expect(blockIdx.outcome).toBe('starved');
    expect(blockIdx.stuck).toEqual([
      { num: 1, waitsOn: 0, started: false },
      { num: 2, waitsOn: 1, started: true },
    ]);
    const last = blockIdx.frames[blockIdx.frames.length - 1];
    expect(last.blocks.find((b) => b.hw === 0)!.phase).toBe('queued');
    expect(last.flagAt.every((x) => x == null)).toBe(true);
  });
  it('blockIdx numbering is fine when the hardware happens to start the blocks in order', () => {
    const r = chainTimeline({ ...SCENE, order: [0, 1, 2], numbering: 'blockIdx' });
    expect(r.outcome).toBe('finished');
    expect(r.correct).toBe(true);
  });
  it('with enough resident slots blockIdx numbering also finishes: the wait ends because block 0 is running', () => {
    const r = chainTimeline({ ...SCENE, slots: 3, numbering: 'blockIdx' });
    expect(r.outcome).toBe('finished');
  });
  it('without the chain, a later block can overwrite keys an earlier block has not read', () => {
    const r = chainTimeline({ ...SCENE, order: [2, 1, 0], slots: 1, numbering: 'blockIdx', chain: false });
    expect(r.outcome).toBe('finished');
    expect(r.hazards).toBeGreaterThan(0);
    expect(r.correct).toBe(false);
  });
});

describe('§12.8: related patterns', () => {
  it('q-dedup: [3, 3, 5, 7, 7, 7, 9] keeps 4', () => {
    const d = dedupFlags([3, 3, 5, 7, 7, 7, 9]);
    expect(d.keep).toEqual([1, 0, 1, 1, 0, 0, 1]);
    expect(d.offset).toEqual([0, 1, 1, 2, 3, 3, 3]);
    expect(d.out).toEqual([3, 5, 7, 9]);
    expect(d.count).toBe(4);
  });
  it('dedup in place is a filter: its dependencies run forwards', () => {
    const keys = [1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5, 6];
    const deps = blockDependencies(dedupMoves(keys), keys.length, 4);
    expect(deps.direction).toBe('forward');
  });
  it('p-col: a 4 x 6 matrix loses column 2; (2, 4) moves to flat index 13', () => {
    const m = moveByIndex({ kind: 'removeCols', rows: 4, cols: 6, remove: [2] });
    expect(m.newCols).toBe(5);
    const e = m.moves.find((x) => x.r === 2 && x.c === 4)!;
    expect(e.src).toBe(16);
    expect(e.dest).toBe(13);
    expect(e.formula).toBe('2 × 5 + (4 − 1) = 13');
  });
  it('removing rows: destinations from the index, dependencies forwards', () => {
    const m = moveByIndex({ kind: 'removeRows', rows: 6, cols: 4, remove: [1, 4] });
    expect(m.newRows).toBe(4);
    expect(m.moves.find((x) => x.r === 5 && x.c === 3)!.dest).toBe(3 * 4 + 3);
    expect(m.moves.filter((x) => x.r === 1).every((x) => x.dest == null)).toBe(true);
    expect(blockDependencies(m.moves, 24, 2 * 4).direction).toBe('forward');
  });
  it('removing the last row moves nothing', () => {
    const m = moveByIndex({ kind: 'removeRows', rows: 4, cols: 4, remove: [3] });
    expect(blockDependencies(m.moves, 16, 4).direction).toBe('none');
  });
  it('q-pad: padding rows from width 6 to 8 moves keys outwards; the chain runs backwards', () => {
    const m = moveByIndex({ kind: 'padCols', rows: 4, cols: 6, pad: 2 });
    expect(m.newCols).toBe(8);
    expect(m.moves.find((x) => x.r === 3 && x.c === 5)!.dest).toBe(29);
    const deps = blockDependencies(m.moves, 24, 6); // one row per block
    expect(deps.direction).toBe('backward');
    expect(deps.edges.every((e) => e.reader > e.writer)).toBe(true);
    expect(deps.fresh).toBeGreaterThan(0);
  });
  it('removing columns in place: forwards', () => {
    const m = moveByIndex({ kind: 'removeCols', rows: 4, cols: 6, remove: [0, 3] });
    expect(blockDependencies(m.moves, 24, 6).direction).toBe('forward');
  });
  it('exclusive scan', () => {
    expect(exclusiveScan([0, 1, 1, 0, 1, 0, 0, 1])).toEqual([0, 0, 1, 2, 2, 3, 3, 3]);
  });
});
