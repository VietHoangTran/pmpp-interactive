import { describe, expect, it } from 'vitest';
import {
  FORMS,
  LINE_PAIRS,
  binary,
  ctz64,
  ffs,
  halves,
  hex,
  maskOf,
  passFrom,
  popc,
  preset,
  presetsFor,
  vote,
  type Form,
  type Width,
} from './warpvote';

const lanesOfResult = (r: ReturnType<typeof vote>) => r.lanes.filter((L) => L.writes).map((L) => [L.lane, L.rank, L.slot]);
const slotsOf = (r: ReturnType<typeof vote>) => r.lanes.filter((L) => L.writes).map((L) => L.slot);
const randomLanes = (width: number, seed: number) => {
  let a = seed;
  const r = () => ((a = (a * 1103515245 + 12345) % 2147483648) / 2147483648);
  return Array.from({ length: width }, (_, l) => l).filter(() => r() < 0.45);
};

describe('bit helpers', () => {
  it('ffs is 1-based with 0 for no bit; ctz on a 64-bit vote gives 64 for no lane', () => {
    expect(ffs(BigInt(0))).toBe(0);
    expect(ffs(BigInt(0b110110))).toBe(2);
    expect(ffs(BigInt(1) << BigInt(63))).toBe(64);
    expect(ctz64(BigInt(0))).toBe(64);
    expect(ctz64(maskOf([3, 9]))).toBe(3);
    expect(popc(maskOf([1, 2, 4, 5]))).toBe(4);
  });
  it('the book\'s mask, lanes 1, 2, 4 and 5, in binary and hex', () => {
    const m = maskOf([1, 2, 4, 5]);
    expect(binary(m, 8)).toBe('0011 0110');
    expect(binary(m, 32)).toBe('0000 0000 0000 0000 0000 0000 0011 0110');
    expect(hex(m, 32)).toBe('0x00000036');
    expect(binary(maskOf([2, 4, 5, 7]), 8)).toBe('1011 0100');
  });
});

describe('Fig. 12.3 as printed', () => {
  it('the book\'s example (p. 292): mask 0b110110, leader 1, count 4, ranks 0 to 3, one atomic', () => {
    const r = vote({ width: 32, pass: passFrom(32, [1, 2, 4, 5]), counter: 40, form: 'book' });
    const g = r.groups[0];
    expect(r.groups.length).toBe(1);
    expect(g.mask).toBe(BigInt(0b110110));
    expect(g.ffs).toBe(2);
    expect(g.leader).toBe(1);
    expect(g.count).toBe(4);
    expect(g.atomic).toEqual({ lane: 1, add: 4, before: 40, after: 44 });
    expect(g.start).toBe(40);
    expect(lanesOfResult(r)).toEqual([[1, 0, 40], [2, 1, 41], [4, 2, 42], [5, 3, 43]]);
    expect(r.counterAfter).toBe(44);
    expect(r.atomics).toBe(1);
    expect(r.correct).toBe(true);
    expect(r.flags).toEqual([]);
  });
  it('lanes that fail the test are outside the branch and take no part', () => {
    const r = vote({ width: 8, pass: passFrom(8, [1, 2, 4, 5]), counter: 0, form: 'book' });
    expect(r.lanes.map((L) => L.votes)).toEqual([false, true, true, false, true, true, false, false]);
    expect(r.lanes[0].rank).toBeNull();
  });
  it('question q-rank: lanes 2, 4, 5 and 7 pass, fetch_add returns 40: lane 7 writes slot 43, leader 2, counter ends at 44', () => {
    const r = vote({ width: 32, pass: passFrom(32, [2, 4, 5, 7]), counter: 40, form: 'book' });
    expect(r.groups[0].ffs).toBe(3);
    expect(r.groups[0].leader).toBe(2);
    expect(r.lanes[7].rank).toBe(3);
    expect(r.lanes[7].slot).toBe(43);
    expect(r.counterAfter).toBe(44);
  });
  it('question q-snapshot: lanes {1, 2} then {4, 5} as two snapshots: two atomics, two contiguous ranges, all four keys right', () => {
    const one = vote({ width: 32, pass: passFrom(32, [1, 2, 4, 5]), counter: 40, form: 'book' });
    const two = vote({ width: 32, pass: passFrom(32, [1, 2, 4, 5]), counter: 40, form: 'book', snapshots: halves(passFrom(32, [1, 2, 4, 5])) });
    expect(halves(passFrom(32, [1, 2, 4, 5]))).toEqual([[1, 2], [4, 5]]);
    expect(two.atomics).toBe(2);
    expect(two.groups.map((g) => g.atomic)).toEqual([
      { lane: 1, add: 2, before: 40, after: 42 },
      { lane: 4, add: 2, before: 42, after: 44 },
    ]);
    expect(two.correct).toBe(true);
    expect([...slotsOf(two)].sort()).toEqual([...slotsOf(one)].sort());
  });
  it('one atomic per snapshot, and the same set of slots, for any split', () => {
    for (let seed = 1; seed < 60; seed++) {
      const lanes = randomLanes(32, seed);
      if (lanes.length < 2) continue;
      const pass = passFrom(32, lanes);
      const cut = 1 + (seed % (lanes.length - 1));
      const r = vote({ width: 32, pass, counter: 7, form: 'book', snapshots: [lanes.slice(cut), lanes.slice(0, cut)] });
      expect(r.atomics).toBe(2);
      expect(r.correct).toBe(true);
      expect(new Set(slotsOf(r))).toEqual(new Set(r.expected));
    }
  });
  it('snapshots that do not split the passing lanes are refused', () => {
    expect(() => vote({ width: 8, pass: passFrom(8, [1, 2]), counter: 0, form: 'book', snapshots: [[1]] })).toThrow();
    expect(() => vote({ width: 8, pass: passFrom(8, [1, 2]), counter: 0, form: 'book', snapshots: [[1, 3], [2]] })).toThrow();
  });
  it('does not compile for a 64-wide wavefront', () => {
    const r = vote({ width: 64, pass: passFrom(64, [1]), counter: 0, form: 'book' });
    expect(r.flags).toEqual(['compile-error']);
    expect(r.correct).toBe(false);
    expect(r.compileError).toContain('64-bit');
  });
});

describe('every correct form fills the same slots', () => {
  const correctForms: { form: Form; width: Width }[] = [
    { form: 'book', width: 32 },
    { form: 'ballot', width: 32 },
    { form: 'ballot', width: 64 },
    { form: 'metal', width: 32 },
    { form: 'metal', width: 64 },
    { form: 'hip-naive', width: 32 },
  ];
  it('random masks: each lane gets counter + (passing lanes below it), whichever form', () => {
    for (let seed = 1; seed < 80; seed++)
      for (const { form, width } of correctForms) {
        const lanes = randomLanes(width, seed);
        const r = vote({ width, pass: passFrom(width, lanes), counter: 100, form });
        expect(r.correct).toBe(true);
        expect(lanesOfResult(r)).toEqual(lanes.map((l, k) => [l, k, 100 + k]));
        expect(r.counterAfter).toBe(100 + lanes.length);
        expect(r.atomics).toBe(form === 'metal' || lanes.length ? 1 : 0);
      }
  });
  it('a warp with no passing lane: the book form never enters the branch, the ballot form skips', () => {
    for (const form of ['book', 'ballot', 'hip-naive'] as Form[]) {
      const r = vote({ width: 32, pass: passFrom(32, []), counter: 5, form });
      expect(r.atomics).toBe(0);
      expect(r.correct).toBe(true);
    }
    expect(vote({ width: 32, pass: passFrom(32, []), counter: 5, form: 'ballot' }).groups[0].skipped).toBe(true);
  });
});

describe('Metal (question q-metal)', () => {
  it('lane 0 fails, lanes 3 and 9 keep: lane 0 does the atomic, adding 2, and the output is right', () => {
    const r = vote({ width: 32, pass: passFrom(32, [3, 9]), counter: 10, form: 'metal' });
    expect(r.groups[0].leader).toBe(0);
    expect(r.groups[0].atomic).toEqual({ lane: 0, add: 2, before: 10, after: 12 });
    expect(r.lanes[0].writes).toBe(false);
    expect(lanesOfResult(r)).toEqual([[3, 0, 10], [9, 1, 11]]);
    expect(r.groups[0].ctz).toBe(3);
    expect(r.correct).toBe(true);
  });
  it('nobody keeps: lane 0 adds 0 unless the code checks the count', () => {
    const open = vote({ width: 32, pass: passFrom(32, []), counter: 10, form: 'metal' });
    expect(open.atomics).toBe(1);
    expect(open.groups[0].atomic?.add).toBe(0);
    expect(open.flags).toContain('zero-add');
    expect(open.correct).toBe(true);
    expect(open.groups[0].ctz).toBe(64);
    const guarded = vote({ width: 32, pass: passFrom(32, []), counter: 10, form: 'metal', guard: true });
    expect(guarded.atomics).toBe(0);
    expect(guarded.flags).toEqual([]);
  });
});

describe('the naive HIP port (question q-hip64, ch12-facts B1)', () => {
  it('on a 64-wide wavefront with only lanes 32 to 63 passing: leader -1, no atomic, undefined slots', () => {
    const r = vote({ width: 64, pass: preset('upper', 64), counter: 0, form: 'hip-naive' });
    expect(r.groups[0].mask).toBe(BigInt(0));
    expect(r.groups[0].leader).toBe(-1);
    expect(r.atomics).toBe(0);
    expect(r.flags).toEqual(expect.arrayContaining(['leader-invalid', 'lane-dropped', 'slot-undefined']));
    expect(r.correct).toBe(false);
    expect(vote({ width: 64, pass: passFrom(64, [40]), counter: 0, form: 'hip-naive' }).groups[0].leader).toBe(-1);
  });
  it('lanes on both sides: the count and the ranks ignore the upper half', () => {
    const r = vote({ width: 64, pass: passFrom(64, [1, 40]), counter: 0, form: 'hip-naive' });
    expect(r.groups[0].count).toBe(1);
    expect(r.counterAfter).toBe(1);
    expect(r.lanes[40].rank).toBeNull();
    expect(r.flags).toEqual(expect.arrayContaining(['lane-dropped', 'slot-undefined']));
    expect(r.correct).toBe(false);
  });
  it('passes every test that only uses lanes below 32, and is right on 32-wide wavefronts', () => {
    expect(vote({ width: 64, pass: passFrom(64, [1, 2, 4, 5]), counter: 0, form: 'hip-naive' }).correct).toBe(true);
    for (let seed = 1; seed < 40; seed++) expect(vote({ width: 32, pass: passFrom(32, randomLanes(32, seed)), counter: 3, form: 'hip-naive' }).correct).toBe(true);
    expect(vote({ width: 64, pass: preset('all', 64), counter: 0, form: 'hip-naive' }).correct).toBe(false);
  });
  it('the AMD active mask is exact: a requested split is ignored', () => {
    const r = vote({ width: 64, pass: passFrom(64, [1, 2, 4, 5]), counter: 0, form: 'hip-naive', snapshots: [[1, 2], [4, 5]] });
    expect(r.atomics).toBe(1);
  });
});

describe('planted bug: no - 1 after __ffs (question q-ffs)', () => {
  it('right exactly when the lane above the lowest passing lane also passes (every 8-lane mask)', () => {
    for (let m = 1; m < 256; m++) {
      const lanes = Array.from({ length: 8 }, (_, l) => l).filter((l) => (m >> l) & 1);
      const pass = passFrom(8, lanes);
      const r = vote({ width: 8, pass, counter: 0, form: 'bug-ffs' });
      const lowest = lanes[0];
      expect(r.correct).toBe(lanes.includes(lowest + 1));
      if (r.correct) expect(slotsOf(r)).toEqual(slotsOf(vote({ width: 8, pass, counter: 0, form: 'book' })));
      else expect(r.flags).toEqual(expect.arrayContaining(['leader-invalid', 'shuffle-outside']));
    }
  });
  it('dense test inputs never show it; only lane 31 passing gives leader 32, a lane the warp does not have', () => {
    expect(vote({ width: 32, pass: preset('all', 32), counter: 0, form: 'bug-ffs' }).correct).toBe(true);
    const r = vote({ width: 32, pass: preset('lane31', 32), counter: 0, form: 'bug-ffs' });
    expect(r.groups[0].leader).toBe(32);
    expect(r.atomics).toBe(0);
    expect(r.correct).toBe(false);
    expect(vote({ width: 32, pass: passFrom(32, [1, 2, 4, 5]), counter: 0, form: 'bug-ffs' }).correct).toBe(true);
  });
});

describe('planted bug: the lanes-below mask includes the lane', () => {
  it('every rank is one too high: slot counter is a hole and the last key lands past the range', () => {
    const r = vote({ width: 32, pass: passFrom(32, [1, 2, 4, 5]), counter: 40, form: 'bug-below' });
    expect(slotsOf(r)).toEqual([41, 42, 43, 44]);
    expect(r.flags).toEqual(expect.arrayContaining(['gap', 'out-of-range']));
    expect(r.correct).toBe(false);
  });
  it('lane 31 included: (2u << 31) - 1 sets all 32 bits', () => {
    const r = vote({ width: 32, pass: preset('all', 32), counter: 0, form: 'bug-below' });
    expect(r.lanes[31].rank).toBe(32);
  });
  it('split into snapshots, the first snapshot writes into the range the second one reserved', () => {
    const r = vote({ width: 32, pass: passFrom(32, [1, 2, 4, 5]), counter: 40, form: 'bug-below', snapshots: [[1, 2], [4, 5]] });
    expect(r.groups[0].atomic).toEqual({ lane: 1, add: 2, before: 40, after: 42 });
    expect(r.lanes[2].slot).toBe(42);
    expect(r.flags).toEqual(expect.arrayContaining(['gap', 'out-of-range']));
  });
  it('the per-group check: a lane outside its own group\'s range is flagged even inside the warp\'s range', () => {
    // Lane 2 writes 42, inside 40..43 overall but outside its snapshot's 40..41.
    const r = vote({ width: 32, pass: passFrom(32, [1, 2, 4, 5, 6]), counter: 40, form: 'bug-below', snapshots: [[1, 2], [4, 5, 6]] });
    expect(r.lanes[2].slot).toBe(42);
    expect(r.flags).toContain('out-of-range');
  });
});

describe('presets', () => {
  it('offer only what fits the width', () => {
    expect(presetsFor(8)).toEqual(['book', 'none', 'all', 'alternate']);
    expect(presetsFor(32)).toEqual(['book', 'none', 'all', 'lane31', 'alternate']);
    expect(presetsFor(64)).toEqual(['book', 'none', 'all', 'lane31', 'upper', 'alternate']);
    expect(preset('alternate', 8)).toEqual([true, false, true, false, true, false, true, false]);
  });
  it('every form runs on every preset that fits', () => {
    for (const width of [8, 32, 64] as Width[])
      for (const p of presetsFor(width)) for (const form of FORMS) expect(() => vote({ width, pass: preset(p, width), counter: 0, form })).not.toThrow();
  });
});

describe('the Fig. 12.3 / Fig. 12.4 pairs', () => {
  it('cover the four steps and the rank, in the book\'s order', () => {
    expect(LINE_PAIRS.map((p) => p.id)).toEqual(['mask', 'leader', 'count', 'broadcast', 'rank']);
    expect(LINE_PAIRS.map((p) => p.fig4)).toEqual([7, 10, 12, 19, 21]);
  });
});
