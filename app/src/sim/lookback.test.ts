import { describe, expect, it } from 'vitest';
import {
  FINISH,
  JITTER,
  LOCAL,
  METAL_NOTES,
  MP_TABLE,
  SCENE_G,
  SCENE_R,
  flagsAt,
  localRounds,
  messagePassing,
  phaseAt,
  rows,
  sceneRuns,
  simulate,
  startOrder,
  type LookbackConfig,
} from './lookback';

const cfg = (c: Partial<LookbackConfig>): LookbackConfig => ({
  G: 8, R: 4, order: 'index', seed: 1, index: 'counter', scheme: 'single', progress: 'started', ...c,
});

describe('seeded helpers', () => {
  it('start orders: index, reversed, and a seeded shuffle that is a permutation and repeatable', () => {
    expect(startOrder(5, 'index', 1)).toEqual([0, 1, 2, 3, 4]);
    expect(startOrder(5, 'reversed', 1)).toEqual([4, 3, 2, 1, 0]);
    const s = startOrder(16, 'shuffle', 42);
    expect([...s].sort((a, b) => a - b)).toEqual(Array.from({ length: 16 }, (_, i) => i));
    expect(startOrder(16, 'shuffle', 42)).toEqual(s);
    expect(startOrder(16, 'shuffle', 43)).not.toEqual(s);
  });
  it('each segment takes LOCAL plus 0 to JITTER - 1 rounds; "even" turns the jitter off', () => {
    const l = localRounds(32, 7);
    expect(l.every((x) => x >= LOCAL && x < LOCAL + JITTER)).toBe(true);
    expect(new Set(l).size).toBeGreaterThan(1);
    expect(localRounds(8, 7, true)).toEqual(Array(8).fill(LOCAL));
  });
});

describe('waiting on blockIdx.x - 1 (pp. 277, 280; facts A6)', () => {
  it('deadlocks when later blocks start first and fill the device', () => {
    const r = simulate(cfg({ order: 'reversed', index: 'hardware', G: 8, R: 4 }));
    expect(r.outcome).toBe('deadlock');
    // blocks 7..4 are resident and spin; block 4 waits on 3, which never started
    expect(r.stuck.map((s) => s.hw).sort()).toEqual([4, 5, 6, 7]);
    expect(r.stuck.find((s) => s.hw === 4)).toEqual({ hw: 4, idx: 4, on: 3, onStarted: false });
    expect(r.blocks.filter((b) => b.start == null).map((b) => b.hw)).toEqual([0, 1, 2, 3]);
  });
  it('the same order with every block resident at once finishes', () => {
    expect(simulate(cfg({ order: 'reversed', index: 'hardware', G: 8, R: 8 })).outcome).toBe('done');
  });
  it('in index order it happens to finish: the awaited block always started first', () => {
    expect(simulate(cfg({ order: 'index', index: 'hardware', G: 16, R: 4 })).outcome).toBe('done');
  });
  it('decoupled lookback on blockIdx.x deadlocks the same way', () => {
    expect(simulate(cfg({ order: 'reversed', index: 'hardware', scheme: 'decoupled' })).outcome).toBe('deadlock');
  });
});

describe('the claimed index under CUDA\'s rule (a started block keeps progressing)', () => {
  it('never deadlocks: every order, size and seed, single and decoupled', () => {
    for (const scheme of ['single', 'decoupled'] as const)
      for (const order of ['index', 'reversed', 'shuffle'] as const)
        for (const [G, R] of [[2, 1], [6, 3], [8, 4], [16, 2], [32, 5], [32, 32]])
          for (let seed = 1; seed <= 20; seed++) {
            const r = simulate(cfg({ G, R, order, seed, scheme }));
            expect(r.outcome, `${scheme} ${order} G${G} R${R} seed ${seed}`).toBe('done');
            expect(r.blocks.every((b) => b.end != null)).toBe(true);
          }
  });
  it('claimed indices follow the start order', () => {
    const r = simulate(cfg({ order: 'reversed', G: 8, R: 4 }));
    for (let k = 0; k < 8; k++) expect(r.blocks[r.dispatch[k]].idx).toBe(k);
  });
  it('single lookback: G - 1 additions, a chain through every block', () => {
    const r = simulate(cfg({ G: 32, R: 32 }));
    expect(r.additions).toBe(31);
    expect(r.repeated).toBe(0);
    expect(r.criticalPath).toBe(31);
    for (const b of r.blocks) expect(b.dep).toBe(b.idx === 0 ? null : b.idx! - 1);
  });
});

describe('no progress guarantee (Metal\'s case, facts C5): one adversarial schedule', () => {
  it('stalls even with the claimed index once a block has to wait', () => {
    const r = simulate(cfg({ G: 6, R: 3, seed: 6, progress: 'none' }));
    expect(r.outcome).toBe('stall');
    expect(r.stuck.every((s) => s.onStarted)).toBe(true); // what it waits for did start: it is starved
    expect(r.blocks.some((b) => b.spans.some((s) => s.phase === 'starved'))).toBe(true);
  });
  it('with one slot nobody ever waits, so it finishes', () => {
    expect(simulate(cfg({ G: 6, R: 1, progress: 'none' })).outcome).toBe('done');
  });
  it('and has no grid barrier', () => {
    const r = simulate(cfg({ scheme: 'barrier', G: 4, R: 8, progress: 'none' }));
    expect(r.outcome).toBe('refused');
    expect(r.refused).toMatch(/no grid barrier/);
  });
});

describe('decoupled lookback (pp. 279-280)', () => {
  it('never takes longer and never waits longer than single lookback', () => {
    for (const order of ['index', 'reversed', 'shuffle'] as const)
      for (const [G, R] of [[6, 3], [8, 8], [16, 4], [32, 8], [32, 32]])
        for (let seed = 1; seed <= 30; seed++) {
          const s = simulate(cfg({ G, R, order, seed, scheme: 'single' }));
          const d = simulate(cfg({ G, R, order, seed, scheme: 'decoupled' }));
          expect(d.time, `${order} G${G} R${R} seed ${seed}`).toBeLessThanOrEqual(s.time);
          expect(d.waitTotal).toBeLessThanOrEqual(s.waitTotal);
          expect(d.criticalPath).toBeLessThanOrEqual(s.criticalPath);
          expect(d.additions).toBeGreaterThanOrEqual(s.additions);
        }
  });
  it('at its extreme (Fig. 11.16c), with every block resident and equal segments, each block adds all its predecessors', () => {
    const r = simulate(cfg({ G: 8, R: 8, even: true, scheme: 'decoupled' }));
    expect(r.criticalPath).toBe(1);
    expect(r.additions).toBe((8 * 7) / 2);
    expect(r.repeated).toBe(28 - 7);
    expect(r.time).toBe(LOCAL + 1 + FINISH);
    const s = simulate(cfg({ G: 8, R: 8, even: true, scheme: 'single' }));
    expect(s.time).toBe(LOCAL + 7 + FINISH);
  });
});

describe('grid-wide barrier (pp. 276-277; infra A4)', () => {
  it('is refused when the grid does not fit', () => {
    const r = simulate(cfg({ scheme: 'barrier', G: 8, R: 4 }));
    expect(r.outcome).toBe('refused');
    expect(r.refused).toMatch(/8 blocks do not fit in 4 slots/);
  });
  it('runs when every block is resident: two barriers, G - 1 additions by one block', () => {
    const r = simulate(cfg({ scheme: 'barrier', G: 8, R: 8 }));
    expect(r.outcome).toBe('done');
    expect(r.criticalPath).toBe(2);
    expect(r.additions).toBe(7);
    expect(r.blocks.every((b) => b.end === r.time)).toBe(true); // everyone leaves the second barrier together
  });
});

describe('reading a run', () => {
  it('phases and flags at a moment', () => {
    const r = sceneRuns().single;
    const b1 = r.blocks.find((b) => b.idx === 1)!;
    expect(phaseAt(b1, 0)).toBe('local');
    expect(phaseAt(b1, b1.spans.find((s) => s.phase === 'spin')!.from)).toBe('spin');
    expect(phaseAt(b1, b1.end!)).toBe('done');
    expect(flagsAt(r, 0)).toEqual(Array(6).fill('none'));
    expect(flagsAt(r, r.time).every((f) => f === 'inclusive')).toBe(true);
  });
  it('rows are drawn by the segment a block scans, blocks that never started last', () => {
    expect(rows(sceneRuns().single).map((b) => b.idx)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(rows(sceneRuns().metal).slice(3).every((b) => b.start == null)).toBe(true);
  });
});

describe('the scene\'s runs (six blocks, three slots, seed 6)', () => {
  const S = sceneRuns();
  it('part 2: indices 0, 1, 2 claimed in start order; block 1 spins on block 0, then reads its sum', () => {
    expect([SCENE_G, SCENE_R]).toEqual([6, 3]);
    expect(S.single.dispatch.slice(0, 3).map((hw) => S.single.blocks[hw].idx)).toEqual([0, 1, 2]);
    const b1 = S.single.blocks.find((b) => b.idx === 1)!;
    expect(b1.waitedOn).toEqual([0]);
    expect(b1.wait).toBeGreaterThan(0);
    expect(b1.dep).toBe(0);
  });
  it('part 3: a chain of 5 hops for 6 blocks, 31 for 32 resident blocks', () => {
    expect(S.single.outcome).toBe('done');
    expect(S.single.criticalPath).toBe(5);
    expect(simulate({ G: 32, R: 32, order: 'index', seed: 6, index: 'counter', scheme: 'single', progress: 'started' }).criticalPath).toBe(31);
  });
  it('part 4: blocks 3, 4, 5 start first on blockIdx.x; each spins; no slot frees', () => {
    expect(S.deadlock.dispatch.slice(0, 3)).toEqual([3, 4, 5]);
    expect(S.deadlock.outcome).toBe('deadlock');
    expect(S.deadlock.stuck.map((s) => [s.idx, s.on])).toEqual([[3, 2], [4, 3], [5, 4]]);
  });
  it('part 5: decoupled: shorter waits and chain, finished sooner, 4 repeated additions', () => {
    expect(S.decoupled.outcome).toBe('done');
    expect(S.decoupled.waitTotal).toBeLessThan(S.single.waitTotal);
    expect(S.decoupled.time).toBeLessThan(S.single.time);
    expect(S.decoupled.criticalPath).toBeLessThan(S.single.criticalPath);
    expect(S.decoupled.repeated).toBe(4);
    expect(S.decoupled.events.some((e) => e.kind === 'own')).toBe(true);
  });
  it('part 6: the same claimed-index run without a progress guarantee stalls', () => {
    expect(S.metal.outcome).toBe('stall');
  });
});

describe('the flag protocol (Fig. 11.17; facts A5)', () => {
  it('only release + acquire at device scope is guaranteed', () => {
    expect(MP_TABLE.filter((r) => r.verdict === 'guaranteed')).toEqual([{ store: 'release', load: 'acquire', scope: 'device', verdict: 'guaranteed' }]);
  });
  it('a relaxed side at device scope is a data race on partialSums', () => {
    expect(MP_TABLE.filter((r) => r.verdict === 'race')).toHaveLength(3);
    expect(messagePassing('release', 'relaxed', 'device').why).toMatch(/load of the sum be moved before the flag check/);
    expect(messagePassing('relaxed', 'acquire', 'device').why).toMatch(/flag become visible before the sum/);
  });
  it('block scope is undefined whatever the orders', () => {
    expect(MP_TABLE.filter((r) => r.scope === 'block').every((r) => r.verdict === 'scope')).toBe(true);
    expect(messagePassing('release', 'acquire', 'block').why).toContain('UB: data race');
  });
  it('the code shown uses load and store, NVIDIA\'s spelling, with the chosen orders and scope', () => {
    const m = messagePassing('release', 'acquire', 'device');
    expect(m.writer.join('\n')).toContain('f.store(1, cuda::memory_order_release);');
    expect(m.reader.join('\n')).toContain('while (f.load(cuda::memory_order_acquire) == 0) {}');
    expect(m.reader[0]).toContain('cuda::thread_scope_device');
  });
  it('block scope: the text claims only what NVIDIA marks (block scope on the store)', () => {
    const why = messagePassing('release', 'acquire', 'block').why;
    expect(why).toContain('block scope on the store "UB: data race"');
    expect(why).not.toMatch(/exact variant/);
  });
  it('Metal notes: coherent(device); still unsafe (the MSL spellings are files, see S119.snippets.test.ts)', () => {
    expect(METAL_NOTES.join(' ')).toContain('coherent(device)');
    expect(METAL_NOTES.join(' ')).toContain('unsafe on Metal');
  });
});
