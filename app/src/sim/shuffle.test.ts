import { describe, expect, it } from 'vitest';
import {
  BLOCK_SIZES,
  HANDOFFS,
  KERNELS,
  blockCosts,
  callText,
  hybrid,
  log2,
  maskHex,
  ones,
  shuffle,
  smallValues,
  twoStage,
  warpReduceTrace,
} from './shuffle';

const lanes = (n: number) => Array.from({ length: n }, (_, i) => i);
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

describe('one shuffle, by the documented rules (ch10 facts A3, infra A3, B3, C2)', () => {
  it('q-top-lanes: down 16 in a 32-lane warp, lane 20 keeps 20, lane 3 gets 19', () => {
    const r = shuffle('down', lanes(32), 16);
    expect(r[20]).toMatchObject({ outcome: 'own', value: 20, target: 36 });
    expect(r[3]).toMatchObject({ outcome: 'moved', value: 19, src: 19 });
    expect(r.filter((x) => x.outcome === 'own').map((x) => x.lane)).toEqual(lanes(16).map((i) => i + 16));
  });
  it('q-width8: width 8, delta 2: lane 14 keeps 14, lane 13 gets 15', () => {
    const r = shuffle('down', lanes(32), 2, { width: 8 });
    expect(r[14]).toMatchObject({ outcome: 'own', value: 14 });
    expect(r[13]).toMatchObject({ outcome: 'moved', value: 15 });
    // four independent groups of eight: in every group, its top two lanes keep their own
    expect(r.filter((x) => x.outcome === 'own').map((x) => x.lane)).toEqual([6, 7, 14, 15, 22, 23, 30, 31]);
  });
  it('up: the lower delta lanes keep their own value', () => {
    const r = shuffle('up', lanes(32), 4);
    expect(r.slice(0, 4).every((x) => x.outcome === 'own')).toBe(true);
    expect(r[4].value).toBe(0);
  });
  it('xor with a smaller width reaches earlier groups, not later ones', () => {
    const r = shuffle('xor', lanes(32), 8, { width: 8 });
    expect(r[3]).toMatchObject({ outcome: 'own', value: 3, target: 11 }); // 11 is in a later group
    expect(r[11]).toMatchObject({ outcome: 'moved', value: 3 }); // 3 is in an earlier group
    const full = shuffle('xor', lanes(32), 16);
    expect(full.every((x) => x.outcome === 'moved' && x.value === (x.lane ^ 16))).toBe(true);
  });
  it('by source lane: srcLane outside the width wraps within the caller\'s own group', () => {
    const r = shuffle('idx', lanes(32), 10, { width: 8 });
    expect(r[3].value).toBe(2);
    expect(r[9].value).toBe(10);
    expect(r[30].value).toBe(26);
    expect(shuffle('idx', lanes(32), 5).every((x) => x.value === 5)).toBe(true); // a broadcast
  });
  it('reading an exited lane or one outside the mask is undefined; such lanes do not call', () => {
    const exited = lanes(32).map((i) => i >= 20);
    const r = shuffle('down', lanes(32), 16, { exited });
    expect(r[2]).toMatchObject({ outcome: 'moved', value: 18 });
    expect(r[4]).toMatchObject({ outcome: 'undefined', value: null, src: 20 });
    expect(r[25]).toMatchObject({ outcome: 'absent', value: null });
    const mask = lanes(32).map((i) => i !== 9);
    const m = shuffle('down', lanes(32), 1, { mask });
    expect(m[8]).toMatchObject({ outcome: 'undefined', value: null });
    expect(m[9].outcome).toBe('absent');
  });
  it('rejects a width that is not a power of two up to the warp', () => {
    expect(() => shuffle('down', lanes(32), 1, { width: 12 })).toThrow();
    expect(() => shuffle('down', lanes(32), 1, { width: 64 })).toThrow();
  });
  it('writes the mask as CUDA and HIP need it: 32 bits, or 64 bits with a suffix', () => {
    expect(maskHex(new Array(32).fill(true))).toBe('0xffffffff');
    expect(maskHex(new Array(64).fill(true))).toBe('0xffffffffffffffffull');
    expect(maskHex(new Array(8).fill(true))).toBe('0xff');
    expect(maskHex(lanes(32).map((i) => i < 16))).toBe('0x0000ffff');
    expect(callText('down', 2, 8, new Array(32).fill(true))).toBe('v = __shfl_down_sync(0xffffffff, v, 2, 8);');
    expect(callText('xor', 16, 32, new Array(32).fill(true))).toBe('v = __shfl_xor_sync(0xffffffff, v, 16);');
  });
});

describe('warp_reduce, Fig. 10.13 (p. 240)', () => {
  it('five steps for 32 lanes; lane 0 holds the total', () => {
    const v = smallValues(32);
    const t = warpReduceTrace(v, { pattern: 'down' });
    expect(t.steps.map((s) => s.stride)).toEqual([16, 8, 4, 2, 1]);
    expect(t.lane0).toBe(sum(v));
    expect(t.total).toBe(sum(v));
    expect(t.holders).toEqual([0]);
  });
  it('160 lane-operations, of which 31 reach lane 0 and 129 are wasted', () => {
    const t = warpReduceTrace(ones(32), { pattern: 'down' });
    expect([t.laneOps, t.useful, t.wasted]).toEqual([160, 31, 129]);
    expect(t.steps.map((s) => s.lanes.filter((l) => l.useful).length)).toEqual([16, 8, 4, 2, 1]);
    // the cone: at stride 16, lanes 0 to 15 feed lane 0
    expect(t.steps[0].lanes.filter((l) => l.inCone).map((l) => l.lane)).toEqual(lanes(16));
  });
  it('the top lanes add their own value to themselves: lane 31 ends with 31 times 32', () => {
    const t = warpReduceTrace(lanes(32), { pattern: 'down' });
    expect(t.final[31]).toBe(31 * 32);
    expect(t.final[0]).toBe(496);
  });
  it('the xor butterfly costs the same and leaves the total in every lane', () => {
    const v = smallValues(32);
    const t = warpReduceTrace(v, { pattern: 'xor' });
    expect(t.final.every((x) => x === sum(v))).toBe(true);
    expect(t.holders).toEqual(lanes(32));
    expect([t.laneOps, t.useful, t.wasted]).toEqual([160, 160, 0]);
  });
  it('q-wave64: the loop written for 32 lanes on a 64-wide wavefront leaves 32 in lane 0 and 32 in lane 32', () => {
    const t = warpReduceTrace(ones(64), { pattern: 'down', loopWidth: 32 });
    expect(t.steps.length).toBe(5);
    expect(t.final[0]).toBe(32);
    expect(t.final[32]).toBe(32);
    expect(t.total).toBe(64);
    expect(t.holders).toEqual([]);
    const fixed = warpReduceTrace(ones(64), { pattern: 'down' });
    expect(fixed.steps.length).toBe(6);
    expect(fixed.lane0).toBe(64);
    // lane 0's cone starts in lanes 0 to 31 only: the other half never reaches it
    expect(t.initialCone.map((c, i) => c === i < 32).every(Boolean)).toBe(true);
    expect(fixed.initialCone.every(Boolean)).toBe(true);
  });
  it('lanes that returned early make lane 0 undefined; staying and adding 0 fixes it', () => {
    const v = smallValues(32);
    const gone = warpReduceTrace(v, { pattern: 'down', tail: { from: 20, mode: 'return' } });
    expect(gone.lane0).toBeNull();
    expect(gone.steps[0].lanes[4]).toMatchObject({ outcome: 'undefined', received: null });
    expect(gone.steps[0].lanes[20].live).toBe(false);
    const fixed = warpReduceTrace(v, { pattern: 'down', tail: { from: 20, mode: 'zero' } });
    expect(fixed.lane0).toBe(sum(v.slice(0, 20)));
  });
  it('q-early-return: n = 50, warp 1 holds threads 32 to 63; lane 2 (thread 34) reads lane 18 (thread 50), which returned', () => {
    const t = warpReduceTrace(ones(32), { pattern: 'down', tail: { from: 50 - 32, mode: 'return' } });
    expect(t.steps[0].lanes[2]).toMatchObject({ outcome: 'undefined', src: 18 });
    expect(t.lane0).toBeNull();
  });
  it('a full warp reduces in log2(width) steps for every width', () => {
    for (const w of [8, 16, 32, 64]) {
      const t = warpReduceTrace(ones(w), { pattern: 'down' });
      expect(t.steps.length).toBe(log2(w));
      expect(t.lane0).toBe(w);
      expect(t.useful).toBe(w - 1);
    }
  });
});

describe('the hybrid kernel, Fig. 10.14 (pp. 240–241)', () => {
  it('correct at every block size and width, with log2(block) - log2(width) + 1 barriers', () => {
    for (const W of [32, 64])
      for (const B of BLOCK_SIZES) {
        const h = hybrid(smallValues(2 * B), B, W);
        expect(h.correct).toBe(true);
        expect(h.barriers).toBe(log2(B) - log2(W) + 1);
        expect(h.races).toEqual([]);
      }
    const h = hybrid(ones(2048), 1024, 32);
    expect(h.strides).toEqual([512, 256, 128, 64, 32]);
    expect(h.barriers).toBe(6);
    expect(h.shuffleSteps).toBe(5);
    expect(h.handoff.live.filter(Boolean).length).toBe(32);
  });
  it('q-line11: the barrier after the loop is not needed as indexed, at every size (each lane reads the slot it wrote)', () => {
    for (const W of [32, 64])
      for (const B of BLOCK_SIZES) {
        const h = hybrid(smallValues(2 * B), B, W, 'printed', { postLoopBarrier: false });
        expect(h.races).toEqual([]);
        expect(h.correct).toBe(true);
        expect(h.barriers).toBe(log2(B) - log2(W));
      }
  });
  it('handing off one level early silently drops half of the sum', () => {
    const h = hybrid(ones(2048), 1024, 32, 'early');
    expect(h.result).toBe(1024);
    expect(h.expected).toBe(2048);
    expect(h.handoff.live.filter(Boolean).length).toBe(64);
    expect(h.races).toEqual([]);
  });
  it('running the shared loop to stride 1 and warp_reduce as well counts stale partial sums again', () => {
    const h = hybrid(ones(2048), 1024, 32, 'double');
    expect(h.result).toBeGreaterThan(h.expected);
    expect(h.handoff.live.filter(Boolean).length).toBe(1);
  });
  it('one level early with two slots per lane is correct with the barrier and a race without it', () => {
    const ok = hybrid(smallValues(2048), 1024, 32, 'early2');
    expect(ok.correct).toBe(true);
    const bad = hybrid(smallValues(2048), 1024, 32, 'early2', { postLoopBarrier: false });
    expect(bad.result).toBeNull();
    expect(bad.races[0]).toMatchObject({ kind: 'read-after-write', slot: 32, thread: 0, other: 32 });
  });
  it('every variant at every size is decided by the model, with no exception', () => {
    for (const v of HANDOFFS)
      for (const B of BLOCK_SIZES)
        for (const W of [32, 64]) expect(() => hybrid(ones(2 * B), B, W, v)).not.toThrow();
    expect(hybrid(ones(128), 64, 64, 'early2').pastEnd).toBe(64);
  });
});

describe('the two-stage kernel, Fig. 10.16 (pp. 242–243)', () => {
  it('as printed, it is correct only when blockDim / width equals width', () => {
    for (const W of [32, 64])
      for (const B of BLOCK_SIZES) {
        const r = twoStage(smallValues(2 * B), B, W, false);
        expect(r.correct).toBe(B / W === W);
      }
    expect(twoStage(ones(2048), 1024, 32, false).result).toBe(2048);
  });
  it('guarded and padded with 0, it is correct everywhere, with one barrier', () => {
    for (const W of [32, 64])
      for (const B of BLOCK_SIZES) {
        const r = twoStage(smallValues(2 * B), B, W, true);
        expect(r.correct).toBe(true);
        expect(r.barriers).toBe(1);
      }
  });
  it('q-guard-256: 256 threads read 8 real sums and 24 entries past the array', () => {
    const r = twoStage(ones(512), 256, 32, false);
    expect(r.arraySize).toBe(8);
    expect(r.slots.filter((s) => s.state === 'written').length).toBe(8);
    expect(r.slots.filter((s) => s.state === 'past').length).toBe(24);
    expect(r.result).toBeNull();
  });
  it('q-w64-warps: 1024 threads in 64-wide warps give 16 partial sums, and the printed read runs 48 slots past them', () => {
    const r = twoStage(ones(2048), 1024, 64, false);
    expect(r.warps).toBe(16);
    expect(r.slots.filter((s) => s.state === 'past').length).toBe(48);
    const g = twoStage(ones(2048), 1024, 64, true);
    expect(g.arraySize).toBe(32); // sized for the narrowest warp
    expect(g.lanes.filter((l) => l.read == null).length).toBe(48);
  });
});

describe('what each kernel costs (pp. 241–244)', () => {
  it('q-barriers: 10, 6 and 1 barriers for a 1024-thread block of 32-wide warps', () => {
    expect(KERNELS.map((k) => blockCosts(k, 1024, 32).barriers)).toEqual([10, 6, 1]);
  });
  it('every kernel does exactly 2B - 1 useful additions; they differ in the lane slots they spend', () => {
    for (const W of [32, 64])
      for (const B of BLOCK_SIZES)
        for (const k of KERNELS) {
          const c = blockCosts(k, B, W);
          expect(c.correct).toBe(true);
          expect(c.useful).toBe(2 * B - 1);
        }
    expect(KERNELS.map((k) => blockCosts(k, 1024, 32).slots)).toEqual([2176, 2176, 6304]);
    expect(KERNELS.map((k) => blockCosts(k, 1024, 32).wasted)).toEqual([129, 129, 4257]);
  });
  it('shared-memory traffic and shuffles at 1024 threads, width 32', () => {
    const [s, h, t] = KERNELS.map((k) => blockCosts(k, 1024, 32));
    expect([s.sharedWrites, s.sharedReads]).toEqual([2047, 2047]);
    expect([h.sharedWrites, h.sharedReads]).toEqual([2016, 2016]);
    expect([t.sharedWrites, t.sharedReads]).toEqual([32, 32]);
    expect([s.warpShuffles, h.warpShuffles, t.warpShuffles]).toEqual([0, 5, 165]);
    expect([h.shuffleRounds, t.shuffleRounds]).toEqual([5, 10]);
  });
  it('the barriers fall where the book puts them', () => {
    const t = blockCosts('twostage', 256, 32);
    expect(t.steps.filter((s) => s.barrierBefore).length).toBe(1);
    expect(t.steps.findIndex((s) => s.barrierBefore)).toBe(1 + 5);
    expect(t.steps[1].warps).toBe(8);
    const h = blockCosts('hybrid', 256, 32);
    expect(h.steps.filter((s) => s.what === 'shared').map((s) => s.warps)).toEqual([4, 2, 1]);
  });
});
