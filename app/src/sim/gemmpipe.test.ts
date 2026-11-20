import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  SCHEDULES,
  VARIANTS,
  baselineBarriers,
  baselineStream,
  firstFailure,
  hazardsOf,
  loopBound,
  loopSource,
  numTilesFor,
  program,
  referenceSum,
  roles,
  run,
  speedup,
  speedupLimit,
  stateAt,
  timeline,
  type Variant,
} from './gemmpipe';

const repo = path.resolve(__dirname, '../../..');

describe('Fig. 15.12(a): the barriers', () => {
  it('four pairs: seven barriers, four true then three false, alternating', () => {
    const b = baselineBarriers(4);
    expect(b.length).toBe(7);
    expect(b.map((x) => x.dep).join(' ')).toBe('true false true false true false true');
    expect(baselineStream(4).at(-1)!.label).toBe('Compute C3');
  });
});

describe('Figs. 15.13: schedules', () => {
  it('four pairs, L = C = 1: 8 units with two barriers, 5 pipelined (the book draws about 8 and 5)', () => {
    expect(timeline('two-barriers', 4, 1, 1).total).toBe(8);
    expect(timeline('double-buffered', 4, 1, 1).total).toBe(8);
    expect(timeline('pipelined', 4, 1, 1).total).toBe(5);
    expect(speedup(4, 1, 1)).toBe(1.6);
  });
  it('eight pairs, L = 3, C = 1: 32 against 25, a speed-up of 1.28', () => {
    expect(timeline('two-barriers', 8, 3, 1).total).toBe(32);
    expect(timeline('pipelined', 8, 3, 1).total).toBe(25);
    expect(speedup(8, 3, 1)).toBe(1.28);
  });
  it('four pairs, L = 3, C = 1 (the scene): 16 against 13', () => {
    expect(timeline('two-barriers', 4, 3, 1).total).toBe(16);
    expect(timeline('pipelined', 4, 3, 1).total).toBe(13);
  });
  it('barriers: 2n - 1, n and n; with a cost s each they add to the total', () => {
    expect(timeline('two-barriers', 4, 1, 1).barriers.length).toBe(7);
    expect(timeline('double-buffered', 4, 1, 1).barriers.length).toBe(4);
    expect(timeline('pipelined', 4, 1, 1).barriers.length).toBe(4);
    expect(timeline('two-barriers', 4, 1, 1, 0.5).total).toBe(11.5);
    expect(timeline('double-buffered', 4, 1, 1, 0.5).total).toBe(10);
    expect(timeline('pipelined', 4, 1, 1, 0.5).total).toBe(7);
  });
  it('in the pipelined loop the lanes overlap; in the others they never do', () => {
    const overlap = (t: ReturnType<typeof timeline>) =>
      t.boxes.some((a) => a.lane === 'mem' && t.boxes.some((b) => b.lane === 'comp' && a.start < b.end && b.start < a.end));
    expect(overlap(timeline('two-barriers', 4, 1, 1))).toBe(false);
    expect(overlap(timeline('double-buffered', 4, 1, 1))).toBe(false);
    expect(overlap(timeline('pipelined', 4, 1, 1))).toBe(true);
  });
  it('utilisation: each lane is busy n x its time', () => {
    const t = timeline('pipelined', 4, 1, 1);
    expect(t.memUtil).toBe(0.8);
    expect(t.compUtil).toBe(0.8);
    expect(timeline('two-barriers', 4, 1, 1).memUtil).toBe(0.5);
  });
  it('the speed-up tends to (L + C) / max(L, C): at most 2, and 1 when one phase dominates', () => {
    expect(speedupLimit(1, 1)).toBe(2);
    expect(speedupLimit(3, 1)).toBeCloseTo(4 / 3);
    expect(speedup(1000, 1, 1)).toBeCloseTo(2, 2);
    expect(speedup(4, 100, 1)).toBeLessThan(1.01);
  });
  it('the pipelined barriers: the first guards arrival only; the loop ones also guard reuse except the last', () => {
    const b = timeline('pipelined', 4, 1, 1).barriers;
    expect(b.map((x) => [x.arrival, x.reuse])).toEqual([[true, false], [true, true], [true, true], [true, false]]);
    const two = timeline('two-barriers', 4, 1, 1).barriers;
    expect(two.filter((x) => x.reuse && !x.arrival).length).toBe(3);
  });
  it('pairs alternate buffers when there are two', () => {
    const t = timeline('pipelined', 4, 1, 1);
    expect(t.boxes.filter((b) => b.lane === 'mem').map((b) => b.buf)).toEqual([0, 1, 0, 1]);
    expect(timeline('two-barriers', 4, 1, 1).boxes.every((b) => b.buf === 0)).toBe(true);
  });
});

describe('Fig. 15.14: curr and next', () => {
  it('swap every iteration, starting with buffer 0 as curr', () => {
    expect([0, 1, 2, 3].map((i) => roles(i))).toEqual([
      { curr: 0, next: 1 }, { curr: 1, next: 0 }, { curr: 0, next: 1 }, { curr: 1, next: 0 },
    ]);
  });
});

describe('the thread-level model', () => {
  const cfg = (variant: Variant) => ({ threads: 4, tiles: 4, variant });
  const all = (v: Variant) => SCHEDULES.map((s) => run(cfg(v), s));

  it('every thread should end with 4444: four slots of each of four tiles', () => {
    expect(referenceSum(4, 4)).toBe(4444);
  });
  it('the correct loop is right under every order, a run-ahead order included', () => {
    for (const t of all('correct')) {
      expect(t.correct).toBe(true);
      expect(t.violations.length).toBe(0);
      expect(t.sums).toEqual([4444, 4444, 4444, 4444]);
    }
    for (let seed = 1; seed <= 20; seed++) expect(run(cfg('correct'), 'shuffled', seed).correct).toBe(true);
    for (let P = 2; P <= 8; P++)
      for (let n = 1; n <= 6; n++) for (const s of SCHEDULES) expect(run({ threads: P, tiles: n, variant: 'correct' }, s).correct).toBe(true);
  });
  it('a removed loop barrier fails when threads run ahead and passes in lockstep; both hazards appear', () => {
    const [lock, inOrder, rev] = all('no-loop-barrier');
    expect(lock.correct).toBe(true);
    expect(inOrder.correct).toBe(false);
    expect(rev.correct).toBe(false);
    expect(hazardsOf(inOrder)).toEqual(['read-before-write', 'overwritten-before-read']);
  });
  it('a barrier moved before loadTile fails only with read before write, and passes in lockstep', () => {
    const [lock, inOrder] = all('barrier-before-load');
    expect(lock.correct).toBe(true);
    expect(inOrder.correct).toBe(false);
    expect(hazardsOf(inOrder)).toEqual(['read-before-write']);
  });
  it('a missing epilogue fails under every order, with no hazard: 0444', () => {
    for (const t of all('no-epilogue')) {
      expect(t.correct).toBe(false);
      expect(t.violations.length).toBe(0);
      expect(t.sums).toEqual([444, 444, 444, 444]);
    }
  });
  it('loading tile instead of tile + 1 fails under every order, with no hazard: tile 0 twice, tile 3 never', () => {
    for (const t of all('wrong-tile')) {
      expect(t.correct).toBe(false);
      expect(t.violations.length).toBe(0);
      expect(t.sums[0]).toBe(448);
    }
  });
  it('one buffer with one barrier fails under some order, by overwriting before the read', () => {
    const ts = all('single-buffer');
    expect(ts[0].correct).toBe(true);
    expect(ts.some((t) => !t.correct)).toBe(true);
    expect(hazardsOf(ts[1])).toEqual(['overwritten-before-read']);
  });
  it('the first failure of the removed barrier, in order, is thread 0 reading slot 1 of buffer 1 too early', () => {
    const t = run(cfg('no-loop-barrier'), 'in-order');
    const v = t.violations[0];
    expect(v.thread).toBe(0);
    expect(v.kind).toBe('read-before-write');
    expect(v.message).toContain('thread 0 added up buffer 1 in iteration 1 before thread 1 had loaded tile 1 into slot 1');
    expect(firstFailure(t)).toBe(v.step + 1);
  });
  it('the program of the correct loop has 12 operations for four tiles, three of them in the loop per iteration', () => {
    const p = program('correct', 4);
    expect(p.length).toBe(12);
    expect(p.map((o) => o.kind[0]).join('')).toBe('lbmlbmlbmlbm');
    expect(p.filter((o) => o.kind === 'load').map((o) => [o.buf, o.tile])).toEqual([[0, 0], [1, 1], [0, 2], [1, 3]]);
  });
  it('stateAt replays a trace to the same buffers and sums', () => {
    const t = run(cfg('correct'), 'shuffled', 7);
    const st = stateAt(t, t.steps.length);
    expect(st.threads.map((x) => x.sum)).toEqual(t.sums);
    expect(st.threads.every((x) => x.done)).toBe(true);
    expect(st.buffers[1].map((s) => s.tile)).toEqual([3, 3, 3, 3]);
  });
});

describe('the K = 0 trap (ours)', () => {
  it('unsigned numTiles - 1 wraps for K = 0', () => {
    expect(numTilesFor(0, 8)).toBe(0);
    expect(loopBound(numTilesFor(0, 8))).toBe(4294967295);
    expect(loopBound(numTilesFor(24, 8))).toBe(2);
  });
});

describe('the stepper shows the snippet file', () => {
  // Every listing the stepper shows is the loop of a snippet file that the CUDA front-end
  // check parses. 'load-first' is the summary page's, tied to its file by Wrap.snippets.test.ts.
  const FILES: Record<Exclude<Variant, 'load-first'>, string> = {
    correct: 's158_pipeline_toy.cu',
    'no-loop-barrier': 's158_no_loop_barrier.cu',
    'barrier-before-load': 's158_barrier_before_load.cu',
    'no-epilogue': 's158_no_epilogue.cu',
    'wrong-tile': 's158_wrong_tile.cu',
    'single-buffer': 's158_single_buffer.cu',
  };
  it('the stepper offers exactly the variants that have a snippet file', () => {
    expect([...VARIANTS].sort()).toEqual(Object.keys(FILES).sort());
  });
  for (const [v, name] of Object.entries(FILES) as [Exclude<Variant, 'load-first'>, string][]) {
    it(`the ${v} variant's loop is the loop of labs/ch15/snippets/${name}, line for line`, () => {
      const file = fs.readFileSync(path.join(repo, 'labs/ch15/snippets', name), 'utf8').split('\n');
      const shown = loopSource(v).code.split('\n');
      const start = file.findIndex((l) => l.trim() === shown[0].trim());
      expect(start).toBeGreaterThan(0);
      expect(file.slice(start, start + shown.length).map((l) => l.replace(/^ {4}/, ''))).toEqual(shown);
      // ...and the loop is followed by the kernel's last statement, so no line was left out.
      expect(file[start + shown.length].trim()).toBe('sums[threadIdx.x] = sum;');
    });
  }
  it('every variant marks a line for each of its operations', () => {
    for (const v of VARIANTS) {
      const lines = loopSource(v).code.split('\n');
      for (const op of program(v, 4)) {
        expect(op.line).toBeGreaterThan(0);
        expect(lines[op.line - 1]).toBeDefined();
      }
    }
  });
});
