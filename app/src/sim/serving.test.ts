import { describe, expect, it } from 'vitest';
import { BOOK_DEVICE, M8_DEFAULT, precisionFor, relativeStep, stepAt } from './batching';
import {
  MIXES,
  SCENE_REQUESTS,
  SCENE_SHORT_MAX,
  SCENE_SLOTS,
  SHORT_MAX,
  makeRequests,
  rng,
  serve,
  slotsAt,
  waitingAt,
  type Mix,
  type Policy,
} from './serving';

const f32 = precisionFor(4);
const unit = (_: number) => 1;
// Step times from the decode-step model, as the explorer uses them.
const table: number[] = [];
const bookStep = (a: number) => (table[a] ??= stepAt(M8_DEFAULT, a, f32, 1, BOOK_DEVICE).seconds);
// Relative units, as the scene uses them: one batch-1 step = 1.
const sceneStep = (a: number) => relativeStep(M8_DEFAULT, a, f32, 1, 20).step / relativeStep(M8_DEFAULT, 1, f32, 1, 20).step;

const both = (requests: Parameters<typeof serve>[0]['requests'], slots: number, stepTime: (a: number) => number) =>
  (['static', 'continuous'] as Policy[]).map((policy) => serve({ requests, slots, policy, stepTime }));

const CASES: { mix: Mix; load: number; slots: number; seed: number }[] = [];
for (const mix of ['short', 'long'] as Mix[])
  for (const load of [0.5, 8, 128])
    for (const slots of [1, 4, 32, 256])
      for (const seed of [1, 2]) CASES.push({ mix, load, slots, seed });
const requestsFor = (c: (typeof CASES)[number]) =>
  makeRequests({ count: 300, rate: c.load / (bookStep(1) * MIXES[c.mix].mean), mix: c.mix, seed: c.seed });

describe('arrivals and lengths', () => {
  it('are repeatable for a seed and differ between seeds', () => {
    const a = makeRequests({ count: 50, rate: 10, mix: 'long', seed: 7 });
    expect(makeRequests({ count: 50, rate: 10, mix: 'long', seed: 7 })).toEqual(a);
    expect(makeRequests({ count: 50, rate: 10, mix: 'long', seed: 8 })).not.toEqual(a);
    expect(a[0].arrival).toBe(0);
    for (let i = 1; i < a.length; i++) expect(a[i].arrival).toBeGreaterThanOrEqual(a[i - 1].arrival);
  });
  it('the mixes keep to their ranges and roughly to their stated means', () => {
    const r = rng(3);
    const short = Array.from({ length: 20000 }, () => MIXES.short.draw(r));
    const long = Array.from({ length: 20000 }, () => MIXES.long.draw(r));
    expect(Math.min(...short)).toBe(8);
    expect(Math.max(...short)).toBe(40);
    expect(Math.min(...long)).toBe(4);
    expect(Math.max(...long)).toBeLessThanOrEqual(512);
    const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / xs.length;
    expect(Math.abs(mean(short) - MIXES.short.mean) / MIXES.short.mean).toBeLessThan(0.02);
    expect(Math.abs(mean(long) - MIXES.long.mean) / MIXES.long.mean).toBeLessThan(0.05);
    expect(long.filter((x) => x <= SHORT_MAX).length / long.length).toBeCloseTo(0.8, 1);
  });
});

describe("the scene's six requests in four slots", () => {
  const [st, co] = (['static', 'continuous'] as Policy[]).map((policy) =>
    serve({ requests: SCENE_REQUESTS, slots: SCENE_SLOTS, policy, stepTime: sceneStep, shortMax: SCENE_SHORT_MAX }),
  );
  it('static: A, B, C, D start together; C and A finish early and hold their slots; E and F wait for B, the longest', () => {
    expect(st.served.filter((r) => r.admitStep === 0).map((r) => r.id)).toEqual([0, 1, 2, 3]);
    expect(slotsAt(st, 3)).toEqual([{ id: 0, state: 'done' }, { id: 1, state: 'gen' }, { id: 2, state: 'done' }, { id: 3, state: 'gen' }]);
    expect(waitingAt(st, 3)).toEqual([4, 5]);
    expect(st.served.find((r) => r.id === 4)!.admitStep).toBe(8);
    expect(st.steps.length).toBe(12);
  });
  it('continuous: E takes C\'s slot at step 2, F takes A\'s at step 3, and the run takes 8 steps', () => {
    expect(slotsAt(co, 2)).toEqual([{ id: 0, state: 'gen' }, { id: 1, state: 'gen' }, { id: 4, state: 'gen' }, { id: 3, state: 'gen' }]);
    expect(slotsAt(co, 3).map((s) => s?.id)).toEqual([5, 1, 4, 3]);
    expect(co.steps.length).toBe(8);
    expect(co.tokens).toBe(26);
    expect(st.tokens).toBe(26);
  });
  it('in relative units continuous makes more tokens per unit time and returns short requests sooner', () => {
    expect(co.tokensPerSecond).toBeGreaterThan(1.45 * st.tokensPerSecond);
    expect(co.meanFinishShort!).toBeLessThan(st.meanFinishShort!);
    const short = co.served.filter((x) => x.length <= SCENE_SHORT_MAX);
    expect(short.map((x) => x.id)).toEqual([0, 2, 4, 5]);
    for (const r of short) expect(r.returned).toBeLessThan(st.served.find((x) => x.id === r.id)!.returned);
    // B, the longest, comes back at about the same time either way (a little later: its steps carry more rows).
    const b = (r: typeof co) => r.served.find((x) => x.id === 1)!.returned;
    expect(b(co) / b(st)).toBeCloseTo(1, 2);
  });
  it('below the knee each user\'s time per token is about one batch-1 step either way', () => {
    expect(co.meanTimePerToken).toBeGreaterThan(1);
    expect(co.meanTimePerToken).toBeLessThan(1.02);
    expect(st.meanTimePerToken).toBeLessThan(1.02);
  });
});

describe('static batching', () => {
  it('waits for the longest request in a batch: everyone in a batch returns when its longest finishes', () => {
    for (const c of CASES.filter((x) => x.slots > 1)) {
      const r = serve({ requests: requestsFor(c), slots: c.slots, policy: 'static', stepTime: bookStep });
      const batches = new Map<number, typeof r.served>();
      for (const s of r.served) batches.set(s.admitStep, [...(batches.get(s.admitStep) ?? []), s]);
      for (const b of batches.values()) {
        const longest = Math.max(...b.map((s) => s.lastToken));
        for (const s of b) expect(s.returned).toBe(longest);
      }
    }
  });
  it('admits nobody while a batch is running', () => {
    const r = serve({ requests: SCENE_REQUESTS, slots: 4, policy: 'static', stepTime: unit });
    const starts = new Set(r.served.map((s) => s.admitStep));
    for (const k of starts) if (k > 0) expect(r.served.filter((s) => s.admitStep < k).every((s) => s.returnStep <= k)).toBe(true);
  });
});

describe('static against continuous', () => {
  it('continuous finishes short requests no later than static, on average', () => {
    for (const c of CASES) {
      const [st, co] = both(requestsFor(c), c.slots, bookStep);
      if (st.meanFinishShort == null) continue;
      expect(co.meanFinishShort!).toBeLessThanOrEqual(st.meanFinishShort * (1 + 1e-9));
    }
  });
  it('with a single request both policies are identical', () => {
    const one = [{ id: 0, arrival: 0.5, length: 17 }];
    for (const slots of [1, 4, 64]) {
      const [st, co] = both(one, slots, bookStep);
      expect(co.served).toEqual(st.served);
      expect(co.steps).toEqual(st.steps);
      expect(co.tokensPerSecond).toBe(st.tokensPerSecond);
    }
  });
  it('with one slot both policies are identical', () => {
    const reqs = makeRequests({ count: 40, rate: 2000, mix: 'long', seed: 5 });
    const [st, co] = both(reqs, 1, bookStep);
    expect(co.served).toEqual(st.served);
  });
  it('at equal slots, continuous throughput is at least static\'s', () => {
    for (const c of CASES) {
      const [st, co] = both(requestsFor(c), c.slots, bookStep);
      expect(co.tokensPerSecond).toBeGreaterThanOrEqual(st.tokensPerSecond * (1 - 1e-9));
    }
  });
  it('under heavy load and a long-tailed mix continuous wins by a wide margin', () => {
    const [st, co] = both(requestsFor({ mix: 'long', load: 128, slots: 32, seed: 1 }), 32, bookStep);
    expect(co.tokensPerSecond / st.tokensPerSecond).toBeGreaterThan(2);
    expect(co.utilisation).toBeGreaterThan(2 * st.utilisation);
  });
  it('under light load the slot limit hardly matters: the slots are never full', () => {
    const reqs = requestsFor({ mix: 'short', load: 0.5, slots: 8, seed: 1 });
    const r8 = serve({ requests: reqs, slots: 8, policy: 'continuous', stepTime: bookStep });
    const r256 = serve({ requests: reqs, slots: 256, policy: 'continuous', stepTime: bookStep });
    expect(r256.tokensPerSecond / r8.tokensPerSecond).toBeCloseTo(1, 3);
    expect(r8.meanBatch).toBeLessThan(2);
  });
  it('under heavy load, raising the slot limit raises throughput until the step passes the knee, then mostly raises time per token', () => {
    const reqs = makeRequests({ count: 1000, rate: 256 / (bookStep(1) * MIXES.short.mean), mix: 'short', seed: 1 });
    const run = (slots: number) => serve({ requests: reqs, slots, policy: 'continuous', stepTime: bookStep });
    const [r8, r32, r64, r256] = [8, 32, 64, 256].map(run);
    expect(r32.tokensPerSecond / r8.tokensPerSecond).toBeGreaterThan(3);
    expect(r256.tokensPerSecond / r64.tokensPerSecond).toBeLessThan(1.25);
    expect(r256.meanTimePerToken / r64.meanTimePerToken).toBeGreaterThan(2);
  });
});

describe('bookkeeping', () => {
  it('utilisation never exceeds 1, and every request is admitted after it arrives and returned after its last token', () => {
    for (const c of CASES)
      for (const r of both(requestsFor(c), c.slots, bookStep)) {
        expect(r.utilisation).toBeLessThanOrEqual(1);
        expect(r.utilisation).toBeGreaterThan(0);
        for (const s of r.served) {
          expect(s.admit).toBeGreaterThanOrEqual(s.arrival);
          expect(s.lastToken).toBeGreaterThan(s.admit);
          expect(s.returned).toBeGreaterThanOrEqual(s.lastToken);
          expect(s.lastStep - s.admitStep + 1).toBe(s.length); // one token per step while it is in
        }
        for (const st of r.steps) expect(st.active + st.held).toBeLessThanOrEqual(c.slots);
      }
  });
  it('no slot holds two requests at once', () => {
    for (const r of both(requestsFor({ mix: 'long', load: 8, slots: 4, seed: 1 }), 4, bookStep))
      for (let k = 0; k < r.steps.length; k += 7) {
        const ids = r.served.filter((s) => s.admitStep <= k && k < s.returnStep);
        expect(new Set(ids.map((s) => s.slot)).size).toBe(ids.length);
        expect(slotsAt(r, k).filter(Boolean).length).toBe(ids.length);
      }
  });
  it('tokens per second is all tokens over the time from first arrival to last return', () => {
    const [, co] = both(SCENE_REQUESTS, 4, unit);
    expect(co.elapsed).toBe(8);
    expect(co.tokensPerSecond).toBe(26 / 8);
  });
});
