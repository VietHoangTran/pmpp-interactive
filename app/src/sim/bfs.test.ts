import { describe, expect, it } from 'vitest';
import {
  BOOK,
  EX1,
  MAZE,
  MAZE_BLOCKED,
  MAZE_GOAL,
  MAZE_ROOT,
  PRESETS,
  UNVISITED,
  backtrace,
  chain,
  depth,
  grid,
  inNeighbours,
  levels,
  outNeighbours,
  parentChoices,
  predecessors,
  runBFS,
  star,
  storage,
  toCOO,
  toCSC,
  toCSR,
  toggleEdge,
  treeEdges,
  workBounds,
  type Run,
  type Variant,
} from './bfs';

const U = UNVISITED;
const run = (g = BOOK, variant: Variant = 'push', root = g.root, blockSize = 4, warpWidth = 4, switchCall?: number) =>
  runBFS(g, { variant, root, blockSize, warpWidth, switchCall });
const roles = (r: Run, call: number) => r.calls[call - 1].threads.filter((t) => t.role !== 'surplus').map((t) => t.role);
const perCall = (r: Run, f: (c: Run['calls'][number]) => number) => r.calls.map(f);

describe("Fig. 18.3: the book's graph in CSR, CSC and COO", () => {
  it('CSR: srcPtrs and dst exactly as printed', () => {
    expect(toCSR(BOOK)).toEqual({ srcPtrs: [0, 2, 4, 7, 9, 11, 12, 13, 15, 15], dst: [1, 2, 3, 4, 5, 6, 7, 4, 8, 5, 8, 6, 8, 0, 6] });
  });
  it('CSC: dstPtrs and src exactly as printed', () => {
    expect(toCSC(BOOK)).toEqual({ dstPtrs: [0, 1, 2, 3, 4, 6, 8, 11, 12, 15], src: [7, 0, 0, 1, 1, 3, 2, 4, 2, 5, 7, 2, 3, 4, 6] });
  });
  it('COO: src and dst exactly as printed', () => {
    expect(toCOO(BOOK)).toEqual({ src: [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7], dst: [1, 2, 3, 4, 5, 6, 7, 4, 8, 5, 8, 6, 8, 0, 6] });
  });
  it("p. 427's worked reading: srcPtrs[3] = 7 and srcPtrs[4] = 9 give edges 3→4 and 3→8", () => {
    const { srcPtrs } = toCSR(BOOK);
    expect([srcPtrs[3], srcPtrs[4]]).toEqual([7, 9]);
    expect(outNeighbours(toCSR(BOOK), 3)).toEqual([4, 8]);
  });
  it('the start page questions: 4 → 5 and 4 → 8; nothing leaves 8; 2, 5 and 7 point to 6', () => {
    expect(outNeighbours(toCSR(BOOK), 4)).toEqual([5, 8]);
    expect(outNeighbours(toCSR(BOOK), 8)).toEqual([]);
    const csc = toCSC(BOOK);
    expect([csc.dstPtrs[6], csc.dstPtrs[7]]).toEqual([8, 11]);
    expect(inNeighbours(csc, 6)).toEqual([2, 5, 7]);
  });
  it('p. 428: 25 storage locations without the data array, against 81 for the dense matrix; COO takes 30', () => {
    expect(storage(BOOK)).toEqual({ dense: 81, csr: 25, csc: 25, coo: 30 });
  });
});

describe('Exercise 1 (our reading of p. 450)', () => {
  it('1a: 15 edges; rows of the adjacency matrix as the notes give them', () => {
    expect(EX1.edges.length).toBe(15);
    const rows = Array.from({ length: 8 }, (_, v) => outNeighbours(toCSR(EX1), v));
    expect(rows).toEqual([[2, 5], [0, 4, 7], [3], [0, 6], [3], [1, 7], [4], [2, 4, 6]]);
  });
  it('1b: CSR with sorted neighbour lists', () => {
    expect(toCSR(EX1)).toEqual({ srcPtrs: [0, 2, 5, 6, 8, 9, 11, 12, 15], dst: [2, 5, 0, 4, 7, 3, 0, 6, 3, 1, 7, 4, 2, 4, 6] });
  });
  it('1c levels from 0: L1 {2, 5}, L2 {1, 3, 7}, L3 {4, 6}; four calls, the fourth finds nothing', () => {
    expect(levels(EX1, 0)).toEqual([0, 2, 1, 2, 3, 1, 3, 2]);
    for (const v of ['push', 'pull', 'edge', 'frontier'] as Variant[]) {
      const r = run(EX1, v, 0);
      expect(r.calls.length).toBe(4);
      expect(r.calls[3].newVertices).toEqual([]);
    }
  });
  it('1c push: launched 8/8/8/8 at four threads a block; iterating 1/2/3/2', () => {
    const r = run(EX1, 'push', 0);
    expect(perCall(r, (c) => c.launched)).toEqual([8, 8, 8, 8]);
    expect(perCall(r, (c) => c.pastCheck)).toEqual([1, 2, 3, 2]);
  });
  it('1c pull: iterating (unvisited) 7/5/2/0 and labelling 2/3/2/0', () => {
    const r = run(EX1, 'pull', 0);
    expect(perCall(r, (c) => c.pastCheck)).toEqual([7, 5, 2, 0]);
    expect(perCall(r, (c) => c.labelsMin)).toEqual([2, 3, 2, 0]);
    expect(perCall(r, (c) => c.labelsMax)).toEqual([2, 3, 2, 0]);
  });
  it('1c pull, call 2: the five that loop are 1, 3, 4, 6, 7, and 1, 3, 7 label', () => {
    const t = run(EX1, 'pull', 0).calls[1].threads;
    expect(t.filter((x) => x.role === 'labels' || x.role === 'iterates').map((x) => x.vertex)).toEqual([1, 3, 4, 6, 7]);
    expect(t.filter((x) => x.role === 'labels').map((x) => x.vertex)).toEqual([1, 3, 7]);
  });
  it('1c edge-centric: launched 15 per call; source check passed 2/3/8/2; may label 2/3/4/0', () => {
    const r = run(EX1, 'edge', 0, 15);
    expect(perCall(r, (c) => c.items)).toEqual([15, 15, 15, 15]);
    expect(perCall(r, (c) => c.pastCheck)).toEqual([2, 3, 8, 2]);
    expect(perCall(r, (c) => c.labelsMax)).toEqual([2, 3, 4, 0]);
  });
  it('1c edge-centric, call 3: the four that may label are 1→4, 7→4, 3→6, 7→6, two writers per destination', () => {
    const c = run(EX1, 'edge', 0).calls[2];
    const coo = toCOO(EX1);
    const writers = c.threads.filter((t) => t.role === 'labels').map((t) => `${coo.src[t.item!]}→${coo.dst[t.item!]}`);
    expect(writers.sort()).toEqual(['1→4', '3→6', '7→4', '7→6']);
    expect(c.newVertices).toEqual([4, 6]);
    expect([c.labelsMin, c.labelsMax]).toEqual([2, 4]);
  });
  it('1c frontier: launched 1/2/3/2, and every launched thread iterates', () => {
    const r = run(EX1, 'frontier', 0, 1);
    expect(perCall(r, (c) => c.items)).toEqual([1, 2, 3, 2]);
    expect(perCall(r, (c) => c.pastCheck)).toEqual([1, 2, 3, 2]);
  });
});

describe('Fig. 18.4: levels, and moving the root', () => {
  it('(a) root 0: 1, 2 at level 1; 3 to 7 at level 2; 8 at level 3', () => {
    expect(levels(BOOK, 0)).toEqual([0, 1, 1, 2, 2, 2, 2, 2, 3]);
  });
  it('(b) root 2: 5, 6, 7 at 1; 0, 8 at 2; 1 at 3; 3, 4 at 4', () => {
    expect(levels(BOOK, 2)).toEqual([2, 3, 0, 4, 4, 1, 1, 1, 2]);
  });
  it('one edge\'s move of the root relabels every vertex here; vertex 4 goes from 2 to 4', () => {
    const a = levels(BOOK, 0);
    const b = levels(BOOK, 2);
    expect(a.filter((x, v) => x !== b[v]).length).toBe(9);
    expect([a[4], b[4]]).toEqual([2, 4]);
  });
  it('p. 430: back-tracing from 1 with root 2 gives 2 → 7 → 0 → 1; from 4, 2 → 7 → 0 → 1 → 4', () => {
    expect(backtrace(BOOK, 2, 1)).toEqual([2, 7, 0, 1]);
    expect(backtrace(BOOK, 2, 4)).toEqual([2, 7, 0, 1, 4]);
  });
  it('the tie rule takes the lowest-numbered parent: 8 has 3, 4 and 6 at level 2 with root 0 and gets 3', () => {
    expect(parentChoices(BOOK, 0)[8]).toBe(3);
    expect(predecessors(BOOK, 0)[8]).toBe(3);
  });
  it('the BFS tree has one edge into every reached vertex but the root', () => {
    expect(treeEdges(BOOK, 0)).toEqual([[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6], [2, 7], [3, 8]]);
  });
  it('an unreachable vertex has no level and no path', () => {
    const g = toggleEdge(toggleEdge(BOOK, 0, 1), 0, 2); // the root loses its out-edges
    expect(levels(g, 0).filter((x) => x === U).length).toBe(8);
    expect(backtrace(g, 0, 8)).toBeNull();
  });
});

describe('thread roles at level 1 → 2, root 0 (Figs. 18.7, 18.9, 18.11, 18.13)', () => {
  it('Fig. 18.7, push: only the threads on 1 and 2 iterate, and both label', () => {
    expect(roles(run(BOOK, 'push', 0), 2)).toEqual(['idle', 'labels', 'labels', 'idle', 'idle', 'idle', 'idle', 'idle', 'idle']);
  });
  it('Fig. 18.9, pull: 0, 1, 2 do not iterate; 3 to 7 iterate and label; 8 iterates and labels nothing', () => {
    expect(roles(run(BOOK, 'pull', 0), 2)).toEqual(['idle', 'idle', 'idle', 'labels', 'labels', 'labels', 'labels', 'labels', 'iterates']);
    const t8 = run(BOOK, 'pull', 0).calls[1].threads[8];
    expect(t8.trips).toBe(3); // scans 3, 4 and 6
  });
  it('Fig. 18.11, edge-centric: the five edges leaving 1 and 2 are traversed; the other ten are not', () => {
    const c = run(BOOK, 'edge', 0).calls[1];
    const coo = toCOO(BOOK);
    const on = c.threads.filter((t) => t.role === 'labels' || t.role === 'iterates').map((t) => `${coo.src[t.item!]}→${coo.dst[t.item!]}`);
    expect(on).toEqual(['1→3', '1→4', '2→5', '2→6', '2→7']);
    expect(c.threads.filter((t) => t.role === 'idle').length).toBe(10);
  });
  it('Fig. 18.13, frontier: two threads on 1 and 2; the new frontier is 3, 4, 5, 6, 7', () => {
    const c = run(BOOK, 'frontier', 0).calls[1];
    expect(c.items).toBe(2);
    expect(c.prevFrontier).toEqual([1, 2]);
    expect(c.nextFrontier).toEqual([3, 4, 5, 6, 7]);
  });
});

describe('launch geometry and warps', () => {
  it('blocks round up and the last block has surplus threads', () => {
    const c = run(BOOK, 'push', 0, 4).calls[0];
    expect([c.blocks, c.launched, c.surplus]).toEqual([3, 12, 3]);
    expect(c.threads.slice(9).every((t) => t.role === 'surplus' && t.item === null)).toBe(true);
  });
  it('a warp never spans two blocks: blocks of 2 with warps of 4 make partial warps of 2', () => {
    const c = run(BOOK, 'push', 0, 2, 4).calls[0];
    expect(c.warps.map((w) => w.size)).toEqual([2, 2, 2, 2, 2]);
  });
  it('a warp makes as many passes as its longest-looping lane', () => {
    // The star from leaf 1: in call 2 the hub's push thread walks 10 edges alone in its warp.
    const r = run(star(10), 'push', 1, 4, 4);
    const c = r.calls[1];
    expect(c.threads[0].trips).toBe(10);
    expect(c.warps[0]).toMatchObject({ passes: 10, useful: 10, size: 4 });
    expect(c.laneSlots).toBe(40);
    // Edge-centric spreads the same work one edge per thread: one pass for every warp in range.
    const e = run(star(10), 'edge', 1, 4, 4).calls[1];
    expect(e.warps.every((w) => w.passes === 1)).toBe(true);
  });
  it('direction-optimised pushes before the switch call and pulls from it on, and labels the same levels', () => {
    const r = run(BOOK, 'diropt', 0, 4, 4, 3);
    expect(r.calls.map((c) => c.kernel)).toEqual(['push', 'push', 'pull', 'pull']);
    expect(r.final).toEqual(levels(BOOK, 0));
  });
  it('m_f and m_u: edges out of the frontier and out of unvisited vertices, before each call', () => {
    const r = run(BOOK, 'push', 0);
    expect(perCall(r, (c) => c.mf)).toEqual([2, 5, 8, 0]);
    expect(perCall(r, (c) => c.mu)).toEqual([13, 8, 0, 0]);
  });
});

describe('work efficiency (pp. 438-439)', () => {
  const graphs = [...PRESETS, MAZE];
  for (const g of graphs)
    it(`${g.name}: every variant labels the CPU levels, and its totals sit inside the table's bounds`, () => {
      const lv = levels(g, g.root);
      const d = depth(lv);
      const b = workBounds(g, d);
      for (const v of ['push', 'pull', 'edge', 'frontier', 'diropt'] as Variant[]) {
        const r = run(g, v, g.root, 4, 4, 2);
        expect(r.final, v).toEqual(lv);
        expect(r.calls.length, v).toBe(d + 1);
      }
      expect(run(g, 'push').totals.work).toBeLessThanOrEqual(b.push);
      expect(run(g, 'pull').totals.work).toBeLessThanOrEqual(b.pull);
      expect(run(g, 'edge').totals.work).toBe(b.edge);
      expect(run(g, 'frontier').totals.work).toBeLessThanOrEqual(b.frontier);
      // With every vertex reachable, push and frontier hit their bounds exactly.
      if (lv.every((x) => x !== U)) {
        expect(run(g, 'push').totals.work).toBe(b.push);
        expect(run(g, 'frontier').totals.work).toBe(b.frontier);
      }
    });
  it('a chain of n searched from one end: about n² launches for push against n for the frontier', () => {
    const g = chain(10);
    const push = run(g, 'push', 0, 1, 1).totals;
    const fr = run(g, 'frontier', 0, 1, 1).totals;
    expect(depth(levels(g, 0))).toBe(9);
    expect(push.launched).toBe(100); // (d + 1) · n
    expect(fr.launched).toBe(10); // one thread per vertex, once
  });
  it('a W × W grid searched from a corner has 2(W − 1) levels', () => {
    for (const W of [2, 3, 4, 6]) expect(depth(levels(grid(W), 0))).toBe(2 * (W - 1));
  });
  it('a hub with 1,000 out-edges that is never in the frontier is read 10 times by push and 10,000 by edge-centric over 10 kernel calls', () => {
    // A chain of 10 vertices: 9 levels after the root, so the host makes 10 calls, the last
    // one labelling nothing. Vertex 10 is the hub, unreachable from the chain. Every call counts.
    const hub = 10;
    const edges: [number, number][] = [];
    for (let i = 0; i < 9; i++) edges.push([i, i + 1]);
    for (let k = 0; k < 1000; k++) edges.push([hub, 11 + k]);
    const g = { id: 'hub', name: 'hub', n: 1011, edges, pos: [], root: 0 };
    const readsOf = (v: Variant) => {
      const calls = run(g, v, 0, 64, 32).calls;
      expect(calls.length).toBe(10);
      return calls.reduce((s, c) => s + c.threads.reduce((s2, t) => s2 + t.reads.filter((x) => x.array === 'level' && x.index === hub).length, 0), 0);
    };
    expect(readsOf('push')).toBe(10);
    expect(readsOf('edge')).toBe(10000);
  });
});

describe('the maze (scene)', () => {
  const lv = levels(MAZE, MAZE_ROOT);
  it('15 × 15 with 28 blocked cells: 197 vertices, every one reachable', () => {
    expect(MAZE_BLOCKED.length).toBe(28);
    expect(MAZE.n).toBe(197);
    expect(lv.every((x) => x !== U)).toBe(true);
  });
  it('the J has 21 blocks, as the book counts its own', () => {
    expect(MAZE_BLOCKED.filter(([r]) => r >= 8).length).toBe(21);
  });
  it('in open space the first wavefronts hold 4, 8 and 12 cells, as on p. 431', () => {
    expect([1, 2, 3].map((L) => lv.filter((x) => x === L).length)).toEqual([4, 8, 12]);
  });
  it('the numbers the narration says: 30 levels, the target at level 26, the largest wavefront 15 cells', () => {
    const d = depth(lv);
    const sizes = Array.from({ length: d + 1 }, (_, L) => lv.filter((x) => x === L).length);
    expect(d).toBe(30);
    expect(lv[MAZE_GOAL]).toBe(26);
    expect(Math.max(...sizes)).toBe(15);
    expect(MAZE.edges.length).toBe(672);
    const push = run(MAZE, 'push', MAZE_ROOT, 32, 32).totals;
    const fr = run(MAZE, 'frontier', MAZE_ROOT, 32, 32).totals;
    expect([push.items, push.work]).toEqual([31 * 197, 31 * 197 + 672]);
    expect([fr.items, fr.work]).toEqual([197, 197 + 672]);
  });
  it('the back-trace is a shortest wire: 27 cells, each one level below the next, joined by edges', () => {
    const p = backtrace(MAZE, MAZE_ROOT, MAZE_GOAL)!;
    expect(p.length).toBe(27);
    expect(p.map((v) => lv[v])).toEqual(Array.from({ length: 27 }, (_, i) => i));
    for (let i = 1; i < p.length; i++) expect(MAZE.edges.some(([a, b]) => a === p[i - 1] && b === p[i])).toBe(true);
    const ch = parentChoices(MAZE, MAZE_ROOT);
    expect(p.filter((v) => ch[v] > 1).length).toBe(9);
  });
});

describe('editing', () => {
  it('toggling an edge adds or removes exactly that direction', () => {
    const g = toggleEdge(BOOK, 8, 0);
    expect(g.edges.length).toBe(16);
    expect(toggleEdge(g, 8, 0).edges).toEqual(BOOK.edges);
    expect(toggleEdge(BOOK, 3, 3)).toBe(BOOK);
  });
  it('removing 0 → 2 from the Exercise 1 graph changes the levels', () => {
    const g = toggleEdge(EX1, 0, 2);
    expect(levels(g, 0)).not.toEqual(levels(EX1, 0));
  });
});
