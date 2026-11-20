import { describe, expect, it } from 'vitest';
import { firstReuse, isScattered, makeRequests, normalise, peakLive, physicalSlot, simulate, type PagedConfig } from './paged';

const CONFIGS: PagedConfig[] = [
  { pool: 2048, block: 16, maxSeq: 512, requests: 24, seed: 1 },
  { pool: 2048, block: 64, maxSeq: 512, requests: 24, seed: 2 },
  { pool: 1024, block: 32, maxSeq: 512, requests: 30, seed: 3 },
  { pool: 512, block: 32, maxSeq: 256, requests: 8, seed: 5 },
  { pool: 4096, block: 256, maxSeq: 1024, requests: 40, seed: 7 },
  { pool: 600, block: 100, maxSeq: 590, requests: 12, seed: 9 },
];

describe('the request stream', () => {
  it('is deterministic and stays within max_seq', () => {
    const cfg = CONFIGS[0];
    expect(makeRequests(cfg)).toEqual(makeRequests(cfg));
    for (const q of makeRequests(cfg)) {
      expect(q.prompt).toBeGreaterThanOrEqual(1);
      expect(q.final).toBeGreaterThan(q.prompt);
      expect(q.final).toBeLessThanOrEqual(cfg.maxSeq);
    }
  });
  it('normalise makes every request fit both pools', () => {
    const n = normalise({ pool: 100, block: 64, maxSeq: 300, requests: 3, seed: 1 });
    expect(n.pool).toBeGreaterThanOrEqual(320);
  });
});

for (const cfg of CONFIGS)
  describe(`pool ${cfg.pool}, block ${cfg.block}, max_seq ${cfg.maxSeq}, ${cfg.requests} requests, seed ${cfg.seed}`, () => {
    const run = simulate(cfg);
    const B = run.cfg.block;
    it('both sides finish every request', () => {
      expect(run.finishedAt.contiguous).not.toBeNull();
      expect(run.finishedAt.paged).not.toBeNull();
      const last = run.frames[run.frames.length - 1];
      expect(last.contiguous.done).toBe(cfg.requests);
      expect(last.paged.done).toBe(cfg.requests);
    });
    it('contiguous: reserved = live x max_seq, wasted = reserved - used', () => {
      for (const f of run.frames) {
        const c = f.contiguous;
        expect(c.reserved).toBe(c.live.length * run.cfg.maxSeq);
        expect(c.wasted).toBe(c.reserved - c.used);
        expect(c.live.length).toBeLessThanOrEqual(run.slots);
      }
    });
    it('paged: each request holds ceil(len / B) blocks, so its waste is below one block', () => {
      for (const f of run.frames) {
        const p = f.paged;
        for (const r of p.live) expect(r.blocks!.length).toBe(Math.ceil(r.len / B));
        expect(p.wasted).toBe(p.reserved - p.used);
        expect(p.wasted).toBeLessThanOrEqual(p.live.length * (B - 1));
      }
    });
    it('paged: every token maps to exactly one physical slot, and no block is in two places', () => {
      const bad: string[] = [];
      for (const f of run.frames) {
        const p = f.paged;
        const seen = new Set<number>();
        for (const r of p.live)
          for (let t = 0; t < r.len; t++) {
            const s = physicalSlot(r.blocks!, t, B);
            if (s < 0 || s >= run.blocks * B || seen.has(s)) bad.push(`step ${f.step}: r${r.id} token ${t} -> ${s}`);
            seen.add(s);
          }
        const all = [...p.free, ...p.live.flatMap((r) => r.blocks!)];
        if (new Set(all).size !== all.length || all.length !== run.blocks) bad.push(`step ${f.step}: block accounting`);
      }
      expect(bad.slice(0, 5)).toEqual([]);
    });
    it('the same configuration gives the same run', () => {
      expect(simulate(cfg).frames.length).toBe(run.frames.length);
    });
  });

describe('the paged-blocks scene’s run (what its narration says)', () => {
  const run = simulate({ pool: 256, block: 32, maxSeq: 128, requests: 10, seed: 12 });
  it('only 2 requests fit the contiguous pool at once (paged-blocks-1); 8 blocks of 32 (paged-blocks-2)', () => {
    expect(run.slots).toBe(2);
    expect(peakLive(run).contiguous).toBe(2);
    expect(run.blocks).toBe(8);
  });
  it('after that, growing requests take recycled blocks and their tables scatter (paged-blocks-4)', () => {
    const r = firstReuse(run);
    const later = run.frames.slice(r + 1, r + 27);
    expect(later.some((f) => f.paged.live.filter((q) => isScattered(q.blocks!)).length >= 3)).toBe(true);
    expect(run.frames.slice(0, r + 1).every((f) => f.paged.live.every((q) => !isScattered(q.blocks!)))).toBe(true);
  });
  it('a request finishes and a waiting one is admitted into the freed blocks in the same step (paged-blocks-3)', () => {
    const i = firstReuse(run);
    expect(i).toBe(12);
    const before = run.frames[i - 1].paged;
    expect(before.waiting).toEqual([9]);
    // It could not start before: the free list was too short for its prompt.
    expect(before.free.length).toBeLessThan(Math.ceil(run.requests[9].prompt / 32));
    expect(run.frames[i].paged.events).toEqual(['r8 finished', 'r9 admitted']);
  });
  it('the stream finishes in well under half the steps when paged (paged-blocks-5), with no swaps', () => {
    expect(run.finishedAt.paged! * 2).toBeLessThan(run.finishedAt.contiguous!);
    expect(run.frames[run.frames.length - 1].paged.evictions).toBe(0);
  });
});

describe('what paging buys', () => {
  const run = simulate(CONFIGS[0]);
  it('freed blocks are reused by later requests', () => {
    const owner = new Map<number, number>();
    let reused = false;
    for (const f of run.frames)
      for (const r of f.paged.live)
        for (const b of r.blocks!) {
          const prev = owner.get(b);
          if (prev !== undefined && prev !== r.id) reused = true;
          owner.set(b, r.id);
        }
    expect(reused).toBe(true);
  });
  it('more requests run at once, and the stream finishes sooner, than with max_seq reservations', () => {
    const peak = peakLive(run);
    expect(peak.paged).toBeGreaterThan(peak.contiguous);
    expect(run.finishedAt.paged!).toBeLessThan(run.finishedAt.contiguous!);
  });
  it('a full pool swaps the newest request out rather than stopping', () => {
    const tight = simulate({ pool: 1024, block: 32, maxSeq: 512, requests: 30, seed: 3 });
    const last = tight.frames[tight.frames.length - 1];
    expect(last.paged.evictions).toBeGreaterThan(0);
    expect(tight.frames.some((f) => f.paged.events.some((e) => e.endsWith('swapped back in')))).toBe(true);
  });
});
