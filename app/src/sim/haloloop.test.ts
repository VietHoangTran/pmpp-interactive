import { describe, expect, it } from 'vitest';
import {
  ALLREDUCE, BOUNDARY_BLOCKS, DEFAULTS, HOST_CALL, INTERNAL_BLOCKS, SLOTS,
  changedEdges, defectsFor, edges, launchRows, program, simulate,
  type Defect, type Durations, type Issue, type Priority, type Version,
} from './haloloop';

const near = (a: number, b: number, tol = 1e-6) => Math.abs(a - b) <= tol;
const run = (version: Version, d: Partial<Durations> = {}, more: { priority?: Priority; issue?: Issue; defect?: Defect } = {}) =>
  simulate({ ...DEFAULTS, ...d, version, ...more });
const kinds = (r: ReturnType<typeof simulate>) => r.violations.map((v) => v.kind);

describe('the three launches of Fig. 23.14 (lines 10-16, with the guard of Fig. 23.1)', () => {
  it('ny = 10: the bottom launch writes row 8, the top launch row 1, the internal launch rows 2 to 7', () => {
    const r = launchRows(10);
    expect(r.top).toEqual([1]);
    expect(r.bottom).toEqual([8]); // q-threerows
    expect(r.internal).toEqual([2, 3, 4, 5, 6, 7]);
  });
  it('for every ny from 5 to 64 they cover rows 1 to ny - 2 exactly once', () => {
    for (let ny = 5; ny <= 64; ny++) {
      const r = launchRows(ny);
      const all = [...r.top, ...r.bottom, ...r.internal].sort((a, b) => a - b);
      expect(all).toEqual(Array.from({ length: ny - 2 }, (_, i) => i + 1));
    }
  });
});

describe('the GPU model', () => {
  it('the internal kernel alone is three full waves and takes `internal`', () => {
    expect(INTERNAL_BLOCKS / SLOTS).toBe(3);
    const r = run('streams', { internal: 6 }, { issue: 'internal-first', priority: 'equal' });
    const it1 = r.iters[0];
    expect(near(it1.internal.end - it1.internal.start, 6)).toBe(true);
    expect(r.blocks.filter((b) => b.iter === 1 && b.part === 'top')).toHaveLength(BOUNDARY_BLOCKS);
  });
  it('nothing is preempted: every block runs its full length in one slot', () => {
    for (const v of ['mpi', 'streams', 'nccl', 'nvshmem'] as Version[]) {
      const r = run(v);
      for (const b of r.blocks) {
        const len = b.part === 'internal' ? (DEFAULTS.internal * SLOTS) / INTERNAL_BLOCKS : DEFAULTS.boundary;
        expect(near(b.end - b.start, len)).toBe(true);
      }
      // never more than SLOTS blocks at once
      const times = r.blocks.map((b) => b.start);
      for (const t of times) expect(r.blocks.filter((b) => b.start <= t + 1e-9 && b.end > t + 1e-9).length).toBeLessThanOrEqual(SLOTS);
    }
  });
});

describe("Fig. 23.13 (p. 555): the exchange is hidden exactly when the internal computation outlasts it", () => {
  it('NCCL: the kernels do not depend on the exchange time; hidden iff each exchange fits before the internal kernel ends', () => {
    const base = run('nccl', { exTop: 0.5, exBottom: 0.5 }).iters[0];
    const window = base.internal.end - Math.max(base.top.end, base.bottom.end);
    expect(near(window, 6)).toBe(true); // at the defaults: the internal kernel's own time
    for (let X = 0.5; X <= 12; X += 0.25) {
      const r = run('nccl', { exTop: X, exBottom: X });
      const it1 = r.iters[0];
      expect(near(it1.internal.end, base.internal.end)).toBe(true);
      expect(r.iters[0].hidden).toBe(X <= window + 1e-9);
    }
  });
  it('NCCL: while hidden the iteration time does not change; far beyond, it grows one for one', () => {
    const t0 = run('nccl', { exTop: 0.5, exBottom: 0.5 }).iterTime;
    for (const X of [1, 2, 4, 6]) expect(near(run('nccl', { exTop: X, exBottom: X }).iterTime, t0)).toBe(true);
    const t8 = run('nccl', { exTop: 8, exBottom: 8 }).iterTime;
    const t10 = run('nccl', { exTop: 10, exBottom: 10 }).iterTime;
    expect(t8).toBeGreaterThan(t0);
    expect(near(t10 - t8, 2)).toBe(true);
  });
  it('MPI without overlap: never hidden, and both exchanges add to every iteration', () => {
    const a = run('mpi', { exTop: 2, exBottom: 2 });
    const b = run('mpi', { exTop: 3, exBottom: 3 });
    expect(a.hidden).toBe(false);
    expect(near(b.iterTime - a.iterTime, 2)).toBe(true);
  });
});

describe('who waits at the default durations (pp. 561-562, 567, 573)', () => {
  const mpi = run('mpi');
  const streams = run('streams');
  const nccl = run('nccl');
  const nvshmem = run('nvshmem');
  it('iteration times: 15.48, 9.48, 7.54, 7.46; only NCCL and NVSHMEM hide the exchanges', () => {
    expect([mpi, streams, nccl, nvshmem].map((r) => Math.round(r.iterTime * 100) / 100)).toEqual([15.48, 9.48, 7.54, 7.46]);
    expect([mpi, streams, nccl, nvshmem].map((r) => r.hidden)).toEqual([false, false, true, true]);
  });
  it('MPI with streams: the bottom exchange starts when the blocking top exchange returns, 4.02 after the bottom kernel ended', () => {
    const it1 = streams.iters[0];
    expect(near(it1.bottom.end, 1.14)).toBe(true);
    expect(near(it1.exTop.end, 5.12)).toBe(true);
    expect(near(it1.exBottom.start, it1.exTop.end + 2 * HOST_CALL)).toBe(true); // line 31 returns at once, then line 32
    expect(near(streams.bottomGap, 4.02)).toBe(true);
  });
  it('NCCL and NVSHMEM: each exchange starts the moment its rows are computed; the bottom gap is 0', () => {
    expect(nccl.bottomGap).toBe(0);
    expect(nvshmem.bottomGap).toBe(0);
    const it1 = nccl.iters[0];
    expect(near(it1.exTop.start, it1.top.end)).toBe(true);
    expect(near(it1.exBottom.start, it1.bottom.end)).toBe(true);
  });
  it("NCCL: the host's only waits are the sync on the internal stream (line 39) and MPI_Allreduce (line 40)", () => {
    const blocked = nccl.segs.filter((s) => s.lane === 'host' && s.kind === 'blocked' && s.iter === 1).map((s) => s.line);
    expect(blocked).toEqual(['39', '40']);
    const mpiBlocked = streams.segs.filter((s) => s.lane === 'host' && s.kind === 'blocked' && s.iter === 1).map((s) => s.line);
    expect(mpiBlocked).toEqual(['27', '28–30', '31', '32–34', '37', '38']);
  });
  it('host time blocked: 9.24 (streams), 7.06 (NCCL), 7.38 (NVSHMEM); MPI_Allreduce is part of each', () => {
    expect([streams, nccl, nvshmem].map((r) => Math.round(r.hostBlocked * 100) / 100)).toEqual([9.24, 7.06, 7.38]);
    for (const r of [streams, nccl, nvshmem]) expect(r.hostBlocked).toBeGreaterThan(ALLREDUCE);
  });
  it('insertion takes a sliver: Fig. 23.14 queues lines 04 to 22 (eleven calls) in 0.22, before any kernel has finished', () => {
    const ins = streams.insertions.filter((x) => x.iter === 1);
    const head = ins.filter((x) => ['04', '05', '06', '07', '10', '11', '12–13', '14', '15–16', '19', '20', '21–22'].includes(x.line));
    expect(head).toHaveLength(12);
    expect(near(Math.max(...head.map((x) => x.at)), 12 * HOST_CALL)).toBe(true);
    expect(Math.max(...head.map((x) => x.at))).toBeLessThan(streams.iters[0].top.end);
  });
  it('NVSHMEM: the barrier holds the next memset until both exchanges have delivered', () => {
    const r = run('nvshmem', { exTop: 9, exBottom: 9 });
    const it1 = r.iters[0];
    const it2 = r.iters[1];
    expect(it2.memset.start).toBeGreaterThanOrEqual(Math.max(it1.exTop.end, it1.exBottom.end));
    expect(r.violations).toEqual([]);
  });
});

describe('priorities (p. 558; ch23-facts A2: a hint for pending kernels, no preemption)', () => {
  it('boundary kernels issued first start before the internal kernel whatever the priority', () => {
    for (const p of ['equal', 'higher'] as Priority[]) {
      const it1 = run('streams', {}, { priority: p }).iters[0];
      expect(it1.top.start).toBeLessThan(it1.internal.start);
    }
  });
  it('internal kernel issued first: with higher priority the boundary kernels start at the first block retirement', () => {
    const r = run('streams', {}, { issue: 'internal-first', priority: 'higher' });
    const it1 = r.iters[0];
    const firstRetire = Math.min(...r.blocks.filter((b) => b.iter === 1 && b.part === 'internal').map((b) => b.end));
    expect(near(it1.top.start, firstRetire)).toBe(true);
    expect(near(it1.top.start, 2.1)).toBe(true);
  });
  it('internal kernel issued first: with equal priority they start only after every internal block was dispatched', () => {
    const r = run('streams', {}, { issue: 'internal-first', priority: 'equal' });
    const it1 = r.iters[0];
    const lastDispatch = Math.max(...r.blocks.filter((b) => b.iter === 1 && b.part === 'internal').map((b) => b.start));
    expect(it1.top.start).toBeGreaterThan(lastDispatch);
    expect(near(it1.top.start, it1.internal.end)).toBe(true); // 6.10: the boundary rows wait for the whole internal kernel
  });
  it('neither priority preempts: the boundary kernels, queued at 0.12, do not start while all slots are busy', () => {
    for (const p of ['equal', 'higher'] as Priority[]) {
      const r = run('streams', {}, { issue: 'internal-first', priority: p });
      const pend = r.segs.find((s) => s.iter === 1 && s.lane === 'top' && s.kind === 'pending');
      expect(pend).toBeDefined();
      expect(near(pend!.start, 0.12)).toBe(true);
      expect(pend!.end).toBeGreaterThan(2);
    }
  });
  it('the cost: issuing the internal kernel first lengthens the iteration, much more so with equal priority', () => {
    const book = run('streams').iterTime;
    const higher = run('streams', {}, { issue: 'internal-first', priority: 'higher' }).iterTime;
    const equal = run('streams', {}, { issue: 'internal-first', priority: 'equal' }).iterTime;
    expect(book).toBeLessThan(higher);
    expect(higher).toBeLessThan(equal);
  });
});

describe('the bug hunt: each removed ordering, and the book\'s code', () => {
  it("the book's four loops violate no ordering, at any durations, priorities and issue orders", () => {
    for (const v of ['mpi', 'streams', 'nccl', 'nvshmem'] as Version[])
      for (const priority of ['equal', 'higher'] as Priority[])
        for (const issue of ['boundary-first', 'internal-first'] as Issue[])
          for (const internal of [2, 6, 12])
            for (const boundary of [0.5, 2, 4])
              for (const exTop of [0.5, 4, 12])
                for (const exBottom of [0.5, 5, 12])
                  expect(simulate({ internal, boundary, exTop, exBottom, version: v, priority, issue }).violations).toEqual([]);
  });
  it('defects apply only where the code has the line', () => {
    expect(defectsFor('mpi')).toEqual(['none']);
    expect(defectsFor('streams')).toEqual(['none', 'reset-waits', 'compute-waits', 'sync-top', 'pageable']);
    expect(defectsFor('nccl')).toEqual(['none', 'reset-waits', 'compute-waits', 'exchange-waits', 'pageable']);
    expect(defectsFor('nvshmem')).toEqual(['none', 'barrier', 'pageable']);
    expect(program('streams', 'boundary-first', 'reset-waits').map((c) => c.line)).not.toContain('06');
    expect(program('nccl', 'boundary-first', 'exchange-waits').map((c) => c.line)).not.toContain('43');
  });
  it('lines 06-07: in the NCCL loop a boundary kernel can add to the norm before it is zeroed (a race: only for some durations)', () => {
    const bad = run('nccl', { exTop: 2, exBottom: 10 }, { defect: 'reset-waits' });
    expect(kinds(bad)).toContain('reset');
    expect(bad.violations.find((v) => v.kind === 'reset')!.edge).toBe('reset-top');
    expect(run('nccl', { exTop: 2, exBottom: 10 }).violations).toEqual([]);
    expect(run('nccl', {}, { defect: 'reset-waits' }).violations).toEqual([]); // the defaults hide it
  });
  it('lines 06-07: in the NCCL loop they also carry the protection of lines 43-44 to the next iteration', () => {
    const bad = run('nccl', { exTop: 2, exBottom: 10 }, { defect: 'reset-waits' });
    expect(kinds(bad)).toContain('halo');
  });
  it('lines 06-07 in Fig. 23.14: latent; the reset always wins here, because line 37 leaves the internal stream empty', () => {
    for (const internal of [2, 6, 12]) for (const boundary of [0.5, 1, 4]) for (const ex of [0.5, 4, 12])
      for (const issue of ['boundary-first', 'internal-first'] as Issue[])
        expect(run('streams', { internal, boundary, exTop: ex, exBottom: ex }, { defect: 'reset-waits', issue }).violations).toEqual([]);
  });
  it('lines 19-20: the norm is copied before a boundary kernel added its part, when a boundary kernel outlasts the internal one', () => {
    const bad = run('streams', {}, { defect: 'compute-waits', issue: 'internal-first', priority: 'equal' });
    expect(bad.violations.filter((v) => v.iter === 1).map((v) => v.edge)).toEqual(['top-copy', 'bottom-copy']);
    expect(run('streams', {}, { issue: 'internal-first', priority: 'equal' }).violations).toEqual([]);
    expect(run('streams', {}, { defect: 'compute-waits' }).violations).toEqual([]); // the book's issue order hides it
    expect(kinds(run('streams', { internal: 1, boundary: 4 }, { defect: 'compute-waits' }))).toContain('copy');
    expect(kinds(run('nccl', { internal: 1, boundary: 4 }, { defect: 'compute-waits' }))).toContain('copy');
  });
  it('line 27: the top rows are sent before they are computed', () => {
    const bad = run('streams', {}, { defect: 'sync-top' });
    const v = bad.violations.find((x) => x.kind === 'send')!;
    expect(v.edge).toBe('top-exchange');
    expect(v.at).toBeLessThan(bad.iters[0].top.end);
    expect(run('streams').violations).toEqual([]);
  });
  it('lines 43-44: the next iteration reads a halo before it arrives, when an exchange outlasts the iteration', () => {
    const bad = run('nccl', { exTop: 2, exBottom: 10 }, { defect: 'exchange-waits' });
    const v = bad.violations.find((x) => x.kind === 'halo')!;
    expect(v.iter).toBe(2);
    expect(v.edge).toBe('exchange-next');
    expect(v.text).toContain('top halo');
    expect(run('nccl', {}, { defect: 'exchange-waits' }).violations).toEqual([]); // hidden exchanges: no harm at these durations
  });
  it('Fig. 23.25 line 18: without the barrier the next kernel reads a halo whose puts are still in flight', () => {
    const bad = run('nvshmem', { exTop: 2, exBottom: 10 }, { defect: 'barrier' });
    expect(kinds(bad)).toEqual(['halo']);
    expect(run('nvshmem', { exTop: 2, exBottom: 10 }).violations).toEqual([]);
    expect(run('nvshmem', {}, { defect: 'barrier' }).violations).toEqual([]);
  });
  it('pageable l2norm_h: the copy turns synchronous and the host stalls; it costs the overlap in Fig. 23.14 and NCCL, nothing in NVSHMEM', () => {
    for (const v of ['streams', 'nccl', 'nvshmem'] as Version[]) {
      const bad = run(v, {}, { defect: 'pageable' });
      expect(kinds(bad)).toEqual(['stall', 'stall']);
    }
    expect(run('streams', {}, { defect: 'pageable' }).iterTime).toBeGreaterThan(run('streams').iterTime + 5);
    expect(run('nccl', {}, { defect: 'pageable' }).hidden).toBe(false);
    expect(run('nvshmem', {}, { defect: 'pageable' }).iterTime - run('nvshmem').iterTime).toBeLessThan(0.05);
  });
});

describe('events (p. 560; ch23-facts A5: a re-recordable snapshot, not a counter)', () => {
  it("each iteration's waits on resetL2 wait for that iteration's reset: re-recording the event every iteration is safe", () => {
    for (const v of ['streams', 'nccl'] as Version[]) {
      const r = run(v);
      const resets = r.waits.filter((w) => w.event === 'resetL2');
      expect(resets).toHaveLength(4); // two streams, two iterations
      for (const w of resets) {
        expect(near(w.from!, r.iters[w.iter - 1].memset.end)).toBe(true);
        expect(w.fromLane).toBe('internal');
      }
    }
  });
  it('the copy waits for computeTop and computeBottom recorded after the boundary kernels, in the same iteration', () => {
    const r = run('streams');
    for (const w of r.waits.filter((x) => x.event === 'computeTop'))
      expect(near(w.from!, r.iters[w.iter - 1].top.end)).toBe(true);
  });
});

describe('edges(version) against Figs. 23.14, 23.20 and 23.25', () => {
  const by = (v: Version) => Object.fromEntries(edges(v).map((e) => [e.id, [e.mechanism, e.lines]]));
  it('Fig. 23.14: events for the cross-stream edges, the host for the exchanges and the reduction, MPI_Sendrecv returning for the next iteration', () => {
    expect(by('streams')).toEqual({
      'reset-top': ['event', '05, 06'],
      'reset-bottom': ['event', '05, 07'],
      'reset-internal': ['stream', '04, 15'],
      'top-copy': ['event', '11, 19'],
      'bottom-copy': ['event', '14, 20'],
      'internal-copy': ['stream', '15, 21'],
      'top-exchange': ['host', '27'],
      'bottom-exchange': ['host', '31'],
      'copy-allreduce': ['host', '37'],
      'exchange-next': ['blocking', '28–34'],
    });
    expect(new Set(edges('streams').map((e) => e.mechanism))).toEqual(new Set(['stream', 'event', 'host', 'blocking']));
  });
  it('Fig. 23.20: the exchanges move into the boundary streams and the next-iteration edge becomes an event wait', () => {
    expect(changedEdges('streams', 'nccl')).toEqual(['top-exchange', 'bottom-exchange', 'exchange-next']);
    expect(by('nccl')['top-exchange']).toEqual(['stream', '10, 27–30']);
    expect(by('nccl')['exchange-next']).toEqual(['event', '31, 36, 43–44']);
  });
  it('Fig. 23.25: one kernel in one stream; puts inside the kernel; a barrier on the stream at line 18', () => {
    const n = by('nvshmem');
    expect(n['exchange-next']).toEqual(['barrier', '18']);
    expect(n['top-exchange'][0]).toBe('kernel');
    expect(changedEdges('streams', 'nvshmem')).toEqual(['reset-top', 'reset-bottom', 'top-copy', 'bottom-copy', 'top-exchange', 'bottom-exchange', 'exchange-next']);
  });
  it('every violation the simulator reports names an ordering that edges() lists', () => {
    const ids = new Set(edges('streams').map((e) => e.id));
    const bad = [
      run('nccl', { exTop: 2, exBottom: 10 }, { defect: 'reset-waits' }),
      run('streams', {}, { defect: 'sync-top' }),
      run('streams', {}, { defect: 'compute-waits', issue: 'internal-first', priority: 'equal' }),
    ];
    for (const r of bad) for (const v of r.violations) expect(ids.has(v.edge!)).toBe(true);
  });
});
