import { describe, expect, it } from 'vitest';
import {
  BOOK_PICTURE,
  CALL_TICKS,
  HANDOFF_TICKS,
  LAUNCH_TICKS,
  gridStateAt,
  hostCanRead,
  overlap,
  program,
  readOutcomes,
  report,
  run,
  stmtText,
  timeAfter,
  waitCovers,
  type Options,
  type Stmt,
} from './timeline';

const opts = (over: Partial<Options> = {}): Options => ({ ...BOOK_PICTURE, ...over });
const all: Options[] = [];
for (const waitAfterLaunch of [true, false])
  for (const backToBack of [true, false])
    for (const gridTicks of [1, 2, 6, 12])
      for (const hostTicks of [1, 3, 9, 20]) all.push({ waitAfterLaunch, backToBack, gridTicks, hostTicks });

describe('a launch does not wait', () => {
  it('a launch statement takes the same host time whatever the grid and whatever the device is doing', () => {
    for (const o of all) {
      const tl = run(program(o));
      for (const h of tl.host.filter((x) => x.kind === 'launch')) {
        expect(h.to - h.from).toBe(LAUNCH_TICKS);
        expect(h.state).toBe('running');
      }
    }
    // A very long grid ahead in the queue changes nothing for the host.
    const tl = run([{ kind: 'launch', grid: 'A', ticks: 1000 }, { kind: 'launch', grid: 'B', ticks: 1000 }, { kind: 'host', ticks: 2 }]);
    expect(tl.host.map((h) => [h.from, h.to])).toEqual([[0, 1], [1, 2], [2, 4]]);
    expect(tl.hostEnd).toBe(4);
    expect(tl.end).toBe(2002);
  });
  it('when the launch returns, the grid is queued and has not started', () => {
    for (const o of all) {
      const tl = run(program(o));
      for (const g of tl.grids) {
        expect(gridStateAt(tl, g.grid, g.enqueued)).toBe('queued');
        expect(g.start).toBeGreaterThanOrEqual(g.enqueued + HANDOFF_TICKS);
        expect(gridStateAt(tl, g.grid, g.enqueued - 0.5)).toBe('not launched');
      }
    }
  });
  it('the host is blocked only inside a blocking call', () => {
    for (const o of all) {
      const tl = run(program(o));
      for (const h of tl.host) expect(h.state === 'blocked').toBe(h.kind === 'wait');
    }
    expect(run(program(opts({ waitAfterLaunch: false }))).host.some((h) => h.state === 'blocked')).toBe(false);
  });
});

describe('a blocking call waits for everything launched before it', () => {
  it('it returns no earlier than the end of every grid launched before it', () => {
    for (const o of all) {
      const tl = run(program(o));
      tl.host
        .filter((h) => h.kind === 'wait')
        .forEach((w) => {
          for (const g of tl.grids.filter((x) => x.stmt < w.stmt)) expect(w.to).toBeGreaterThanOrEqual(g.end);
          expect(w.to - w.from).toBeGreaterThanOrEqual(CALL_TICKS);
        });
    }
  });
  it('with nothing left to wait for it returns at once', () => {
    const tl = run([{ kind: 'launch', grid: 'A', ticks: 2 }, { kind: 'host', ticks: 50 }, { kind: 'wait' }]);
    const w = tl.host[2];
    expect(w.to - w.from).toBe(CALL_TICKS);
  });
});

describe('the device runs grids one at a time, in launch order', () => {
  it('two launches with no wait between them run back to back, A then B', () => {
    for (const waitAfterLaunch of [true, false]) {
      const tl = run(program(opts({ backToBack: true, waitAfterLaunch })));
      const [a, b] = tl.grids;
      expect([a.grid, b.grid]).toEqual(['A', 'B']);
      expect(b.enqueued).toBeLessThan(a.end); // B was launched before A finished
      expect(b.start).toBe(a.end);
      expect(gridStateAt(tl, 'B', b.enqueued)).toBe('queued');
      expect(gridStateAt(tl, 'A', b.enqueued)).toBe('running'); // it started while B was being launched
    }
  });
  it('no two grids overlap and none starts before the one launched before it has finished', () => {
    for (const o of all) {
      const tl = run(program(o));
      for (let i = 1; i < tl.grids.length; i++) expect(tl.grids[i].start).toBeGreaterThanOrEqual(tl.grids[i - 1].end);
      // The device lane covers the whole run without gaps.
      expect(tl.device[0].from).toBe(0);
      expect(tl.device[tl.device.length - 1].to).toBe(tl.end);
      for (let i = 1; i < tl.device.length; i++) expect(tl.device[i].from).toBe(tl.device[i - 1].to);
    }
  });
  it('a launch inside a host loop makes one grid per iteration (question q-grid)', () => {
    const prog: Stmt[] = [{ kind: 'launch', grid: 'A', ticks: 3 }];
    for (let i = 0; i < 10; i++) prog.push({ kind: 'launch', grid: 'B' + i, ticks: 3 });
    prog.push({ kind: 'wait' });
    const tl = run(prog);
    expect(tl.grids.length).toBe(11);
    expect(tl.host[tl.host.length - 1].to).toBe(tl.grids[10].end);
  });
});

describe("the book's picture (Fig. 2.3): a blocking call after each launch", () => {
  const tl = run(program(BOOK_PICTURE));
  it('is seven statements: work, launch, wait, work, launch, wait, read', () => {
    expect(tl.program.map((s) => s.kind)).toEqual(['host', 'launch', 'wait', 'host', 'launch', 'wait', 'read']);
  });
  it('the lanes strictly alternate: the host never works while a grid runs', () => {
    expect(overlap(tl)).toBe(0);
    for (const o of all.filter((x) => x.waitAfterLaunch && !x.backToBack)) expect(overlap(run(program(o)))).toBe(0);
    // Every moment a grid is running falls inside a blocking call.
    for (const g of tl.grids) {
      const w = tl.host.find((h) => h.kind === 'wait' && h.from <= g.start && h.to >= g.end);
      expect(w, g.grid).toBeDefined();
    }
  });
  it('the numbers for the default sliders', () => {
    expect(tl.host.map((h) => [h.kind, h.from, h.to])).toEqual([
      ['host', 0, 3],
      ['launch', 3, 4],
      ['wait', 4, 11],
      ['host', 11, 14],
      ['launch', 14, 15],
      ['wait', 15, 22],
      ['read', 22, 23],
    ]);
    expect(tl.grids.map((g) => [g.grid, g.enqueued, g.start, g.end])).toEqual([['A', 4, 5, 11], ['B', 15, 16, 22]]);
    expect(tl.end).toBe(23);
  });
  it('the read is guaranteed to see both grids', () => {
    expect(readOutcomes(tl, 6)).toEqual([{ grid: 'A', outcome: 'guaranteed' }, { grid: 'B', outcome: 'guaranteed' }]);
  });
});

describe('without a blocking call the host runs ahead', () => {
  const tl = run(program(opts({ waitAfterLaunch: false })));
  it('host work overlaps the grids', () => {
    expect(tl.program.map((s) => s.kind)).toEqual(['host', 'launch', 'host', 'launch', 'host', 'read']);
    expect(overlap(tl)).toBeGreaterThan(0);
    expect(tl.host.map((h) => [h.kind, h.from, h.to])).toEqual([
      ['host', 0, 3],
      ['launch', 3, 4],
      ['host', 4, 7],
      ['launch', 7, 8],
      ['host', 8, 11],
      ['read', 11, 12],
    ]);
    expect(tl.grids.map((g) => [g.grid, g.enqueued, g.start, g.end])).toEqual([['A', 4, 5, 11], ['B', 8, 11, 17]]);
    expect(tl.hostEnd).toBe(12);
    expect(tl.end).toBe(17);
  });
  it('step by step: what each grid is doing when each statement returns', () => {
    const states = [0, 1, 2, 3, 4, 5, 6].map((n) => report(tl, n).grids.map((g) => g.state));
    expect(states).toEqual([
      ['not launched', 'not launched'],
      ['not launched', 'not launched'],
      ['queued', 'not launched'],
      ['running', 'not launched'],
      ['running', 'queued'],
      ['finished', 'running'],
      ['finished', 'running'],
    ]);
    expect(report(tl, 2).time).toBe(4);
    expect(report(tl, 2).host?.kind).toBe('launch');
    expect(report(tl, 0).host).toBeNull();
    expect(timeAfter(tl, 6)).toBe(12);
  });
  it('the read sees A finished only by timing, and B still running', () => {
    expect(readOutcomes(tl, 5)).toEqual([
      { grid: 'A', outcome: 'finished by timing only' },
      { grid: 'B', outcome: 'not finished' },
    ]);
    expect(hostCanRead(tl, 'A', 5)).toBe(false);
  });
  it('however long the host work is, nothing in the program guarantees the read', () => {
    const slow = run(program(opts({ waitAfterLaunch: false, hostTicks: 20, gridTicks: 1 })));
    expect(readOutcomes(slow, 5).map((r) => r.outcome)).toEqual(['finished by timing only', 'finished by timing only']);
    expect(hostCanRead(slow, 'B', 5)).toBe(false);
  });
});

describe('reading results', () => {
  it('is guaranteed only after a blocking call that follows the launch', () => {
    const prog: Stmt[] = [
      { kind: 'launch', grid: 'A', ticks: 4 },
      { kind: 'wait' },
      { kind: 'launch', grid: 'B', ticks: 4 },
      { kind: 'read' },
    ];
    const tl = run(prog);
    expect(hostCanRead(tl, 'A', 3)).toBe(true);
    expect(hostCanRead(tl, 'B', 3)).toBe(false); // the wait came before B's launch
    expect(hostCanRead(tl, 'A', 1)).toBe(false); // before the wait has returned
    expect(hostCanRead(tl, 'C', 3)).toBe(false);
  });
  it('one blocking call after two launches covers both (question q-inorder)', () => {
    const tl = run(program(opts({ backToBack: true })));
    expect(tl.program.map((s) => s.kind)).toEqual(['host', 'launch', 'launch', 'wait', 'read']);
    const [a, b] = tl.grids;
    expect(b.start).toBeGreaterThanOrEqual(a.end); // B runs after A, so it sees A's writes
    expect(tl.host[3].to).toBeGreaterThanOrEqual(b.end); // the blocking call returns after both
    expect(readOutcomes(tl, 4).map((r) => r.outcome)).toEqual(['guaranteed', 'guaranteed']);
    expect(waitCovers(tl.program, 3)).toEqual(['A', 'B']);
  });
});

describe('statement text per backend', () => {
  const prog = program(BOOK_PICTURE);
  it('CUDA and HIP share the launch syntax and differ in the name of the wait', () => {
    expect(stmtText(prog[1], 'cuda')).toBe('KernelA<<<nBlk, nTid>>>(args);');
    expect(stmtText(prog[1], 'hip')).toBe('KernelA<<<nBlk, nTid>>>(args);');
    expect(stmtText(prog[2], 'cuda')).toBe('cudaDeviceSynchronize();');
    expect(stmtText(prog[2], 'hip')).toBe('hipDeviceSynchronize();');
  });
  it('Metal commits a command buffer and waits on it', () => {
    expect(stmtText(prog[4], 'metal')).toBe('[cmdB commit];');
    expect(stmtText(prog[5], 'metal', waitCovers(prog, 5))).toBe('[cmdB waitUntilCompleted];');
    const b2b = program({ ...BOOK_PICTURE, backToBack: true });
    expect(stmtText(b2b[3], 'metal', waitCovers(b2b, 3))).toBe('[cmdA waitUntilCompleted]; [cmdB waitUntilCompleted];');
  });
  it('the timeline does not depend on the backend at all', () => {
    // run() takes no backend: the same program gives the same lanes.
    expect(run(prog)).toEqual(run(program(BOOK_PICTURE)));
  });
});
