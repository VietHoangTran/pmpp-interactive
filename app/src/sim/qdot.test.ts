import { describe, expect, it } from 'vitest';
import { FP16, minSubnormal } from './minifloat';
import { dequantQ4_0, quantizeQ4_0 } from './quant';
import * as Q from './qdot';

// Pins every number sections X2.5 and X2.6 state, and the claims their interactives let the
// reader discover. Sources: notes/today/x2-facts.md (A2, C1, D1, D3, D4, E3, E4) and
// notes/today/appA-facts.md A2 (FP16's limits).

const SEEDS = Array.from({ length: 40 }, (_, i) => i + 1);
const worst = (code: number) => ({ qw: new Array(32).fill(code), qx: new Array(32).fill(127) });

describe('blocks come from quant.ts unchanged', () => {
  it("codes and dequantised weights are quant.ts's", () => {
    for (const s of SEEDS.slice(0, 10)) {
      const w = Q.normals(32, s);
      const ref = quantizeQ4_0(w);
      const [b] = Q.q4Blocks(w);
      expect(Q.codesOf(b)).toEqual(ref.codes);
      expect(Q.dequant(b)).toEqual(dequantQ4_0(ref));
    }
  });
  it('weight j sits in the low nibble of byte j and weight j + 16 in its high nibble (D1)', () => {
    const { block } = Q.makeCase();
    const c = Q.codesOf(block);
    for (let j = 0; j < 16; j++) expect(block.qs[j]).toBe(c[j] | (c[j + 16] << 4));
  });
});

describe('dp4a and the Q8_1 activation quantiser', () => {
  it('dot4 adds four signed byte products to a 32-bit integer (A2)', () => {
    expect(Q.dot4(0x04030201, 0x01010101, 10)).toBe(20);
    expect(Q.dot4(0x04030201, -1, 10)).toBe(0); // 0xFFFFFFFF is four bytes of -1
    expect(Q.dot4(0x80, 0x01, 0)).toBe(-128); // byte 0x80 is -128
    expect(Q.dot4(0x7f7f7f7f, 0x7f7f7f7f, 0)).toBe(4 * 16129);
  });
  it('Q8_1: d = amax / 127, q = roundf(x / d), s = d · sum q, both stored as FP16 (D3)', () => {
    const { x } = Q.makeCase();
    const q = Q.quantizeQ8_1(x);
    const amax = Math.max(...x.map(Math.abs));
    expect(q.amax).toBe(amax);
    expect(q.dFloat).toBe(Math.fround(amax / 127));
    expect(Math.max(...q.qs.map(Math.abs))).toBe(127);
    expect(q.sum).toBe(q.qs.reduce((a, b) => a + b, 0));
    expect(q.d).toBe(Q.f16(q.dFloat));
    expect(q.s).toBe(Q.f16(Math.fround(q.sum * q.dFloat)));
    expect(Q.quantizeQ8_1(new Array(32).fill(0)).qs.every((v) => v === 0)).toBe(true);
    expect(Q.roundf(2.5)).toBe(3);
    expect(Q.roundf(-2.5)).toBe(-3);
    expect(Object.is(Q.roundf(-0.3), 0)).toBe(true);
  });
});

describe('three paths through one block', () => {
  it('path B equals path A exactly in exact arithmetic (E4), with and without the large channel', () => {
    for (const s of SEEDS)
      for (const o of [false, true]) {
        const c = Q.makeCase(o, s);
        const e = Q.exactPaths(c.block, c.x);
        expect(e.same).toBe(true);
        expect(e.A).toBe(Q.exactDequantDot(c.block, c.x));
      }
  });
  it('in float32, B is not always bit for bit A, but always within float rounding of the exact value', () => {
    let differ = 0;
    for (const s of SEEDS) {
      const c = Q.makeCase(false, s);
      const a = Q.tracePath('A', c).result;
      const b = Q.tracePath('B', c).result;
      const ex = Q.exactDequantDot(c.block, c.x);
      const scale = Q.dequant(c.block).reduce((t, w, j) => t + Math.abs(w * c.x[j]), 0);
      if (a !== b) differ++;
      expect(Math.abs(a - ex)).toBeLessThan(1e-6 * scale);
      expect(Math.abs(b - ex)).toBeLessThan(1e-6 * scale);
    }
    expect(differ).toBeGreaterThan(10);
    // The stepper's default block is one of them.
    const c = Q.makeCase();
    expect(Q.tracePath('A', c).result).not.toBe(Q.tracePath('B', c).result);
  });
  it('dividing an activation by 16, 256 or 4096 is exact in float32, until the bottom of the range (E4)', () => {
    for (const s of SEEDS) for (const v of Q.normals(32, s)) for (const k of [4, 8, 12]) expect(Q.preDivisionExact(v, k)).toBe(true);
    expect(Q.preDivisionExact(1.2345e-41, 12)).toBe(false); // a float32 subnormal loses bits
    // So B's masked nibble times its pre-divided activation is the code times the activation, exactly.
    const c = Q.makeCase();
    const run = Q.tracePath('B', c);
    for (const st of run.steps)
      for (const l of st.lanes ?? []) expect(l.wv * l.xv).toBe(l.code * c.x[l.j]);
  });
  it('the traces compute what the fast row functions compute', () => {
    for (const s of SEEDS.slice(0, 12))
      for (const o of [false, true]) {
        const c = Q.makeCase(o, s);
        expect(Q.tracePath('A', c).result).toBe(Q.blockA(c.block, c.x));
        expect(Q.tracePath('B', c).result).toBe(Q.blockB(c.block, c.x));
        expect(Q.tracePath('C', c).result).toBe(Q.blockC(c.block, Q.quantizeQ8_1(c.x)));
      }
  });
  it("path C's integer sum is the unsigned codes times the int8 activations, corrected once by 8 s (D3)", () => {
    const c = Q.makeCase();
    const run = Q.tracePath('C', c);
    const q8 = run.q8!;
    const codes = Q.codesOf(c.block);
    const sumi = codes.reduce((t, q, j) => t + q * q8.qs[j], 0);
    const dp = run.steps.filter((s) => s.kind === 'dp4a');
    expect(dp.length).toBe(8);
    expect(dp[7].acc).toBe(sumi);
    for (const s of dp) for (const l of s.lanes!) expect(l.wv).toBe(l.code); // unsigned nibbles, 0..15
    const f = Math.fround;
    expect(run.result).toBe(f(c.block.d * f(f(sumi * q8.d) - f(8 * q8.s))));
  });
  it('operation counts per block (source level)', () => {
    const c = Q.makeCase();
    const [A, B, C] = (['A', 'B', 'C'] as const).map((p) => Q.tracePath(p, c));
    expect(A.perRow).toEqual({ mul: 64, add: 32, cmp: 0, int: 64, conv: 33 });
    expect(A.perToken).toEqual({ mul: 0, add: 0, cmp: 0, int: 0, conv: 0 });
    // B: 32 multiply-adds, then the offset as one multiply-add and the scale as one multiply
    expect(B.perRow).toEqual({ mul: 34, add: 36, cmp: 0, int: 32, conv: 33 });
    expect(B.perToken).toEqual({ mul: 24, add: 31, cmp: 0, int: 0, conv: 0 });
    // C: eight dp4a and twelve masks and shifts; three float multiplies per block
    expect(C.perRow).toEqual({ mul: 3, add: 1, cmp: 0, int: 20, conv: 4 });
    expect(C.perToken).toEqual({ mul: 34, add: 0, cmp: 31, int: 31, conv: 35 });
  });
  it("a large channel inflates the block's activation scale and path C's error", () => {
    let worse = 0;
    for (const s of SEEDS) {
      const err = (o: boolean) => {
        const c = Q.makeCase(o, s);
        return Math.abs(Q.tracePath('C', c).result - Q.tracePath('B', c).result);
      };
      if (err(true) > err(false)) worse++;
    }
    expect(worse).toBeGreaterThan(30);
    const c = Q.makeCase(true);
    expect(Q.quantizeQ8_1(c.x).qs[Q.OUTLIER_J]).toBe(127);
  });
});

describe('what quantising the activations costs, over many rows', () => {
  it('large channels raise the median and the tail of C against B, at every row length', () => {
    for (const n of Q.ROW_LENGTHS) {
      const a = Q.rowErrors(n, 'normal');
      const b = Q.rowErrors(n, 'outliers');
      expect(a.actErr.length).toBe(Q.ROWS);
      expect(b.median).toBeGreaterThan(a.median * 1.5);
      expect(b.p95).toBeGreaterThan(a.p95);
      // and it stays well below what quantising the weights to 4 bits already cost
      expect(a.median).toBeLessThan(a.weightMedian / 3);
      expect(b.median).toBeLessThan(b.weightMedian);
    }
  });
  it('the histogram counts every row', () => {
    const e = Q.rowErrors(256, 'normal', 50);
    expect(Q.histogram(e.actErr).reduce((t, b) => t + b.count, 0)).toBe(50);
  });
});

describe('bounds (E3)', () => {
  it('one Q8_0 × Q8_1 block reaches 516,128: past FP16, inside int32, exact in float', () => {
    expect(Q.BOUND_Q8Q8).toBe(516128);
    expect(Q.BOUND_Q8Q8).toBeGreaterThan(Q.FP16_MAX);
    expect(Q.BOUND_Q8Q8).toBeGreaterThan(Q.FP16_EXACT_INT);
    expect(Q.BOUND_Q8Q8).toBeLessThan(2 ** 24);
    expect(Math.fround(Q.BOUND_Q8Q8)).toBe(Q.BOUND_Q8Q8);
  });
  it('Q4_0 against int8: 60,960 unsigned (more than int16 holds), 32,512 signed (255 to spare)', () => {
    expect(Q.BOUND_Q4_UNSIGNED).toBe(60960);
    expect(Q.BOUND_Q4_UNSIGNED).toBeGreaterThan(Q.INT16_MAX);
    expect(Q.BOUND_Q4_SIGNED).toBe(32512);
    expect(Q.INT16_MAX - Q.BOUND_Q4_SIGNED).toBe(255);
    expect(Q.DP4A_STEP_Q4_UNSIGNED).toBe(7620);
  });
  it('an int32 holds 133,144 worst-case int8 products; the 133,145th overflows', () => {
    expect(Q.MAX_Q8_PRODUCT).toBe(16129);
    expect(Q.int32Terms(16129)).toBe(133144);
    expect(133144 * 16129).toBeLessThanOrEqual(Q.INT32_MAX);
    expect(133145 * 16129).toBeGreaterThan(Q.INT32_MAX);
  });
});

describe('accumulators', () => {
  it('int16 wraps on the fifth of eight dp4a steps of the worst-case Q4_0 block (18th product one at a time)', () => {
    const { qw, qx } = worst(15);
    const t = Q.innerSum(qw, qx, 'int16', 4);
    expect(t.values.length).toBe(8);
    expect(t.first.wrap).toBe(5);
    expect(t.exact[3]).toBe(30480);
    expect(t.exact[4]).toBe(38100);
    expect(Q.innerSum(qw, qx, 'int16', 1).first.wrap).toBe(18);
    // signed codes -8 × 127 fit
    expect(Q.innerSum(new Array(32).fill(-8), qx, 'int16', 4).first.wrap).toBeUndefined();
  });
  it('the worst-case Q8 block: int16 wraps on the third product, FP16 rounds the first to a multiple of 8 and reaches infinity on the fifth', () => {
    const { qw, qx } = worst(127);
    const i16 = Q.innerSum(qw, qx, 'int16');
    expect(i16.first.wrap).toBe(3);
    expect(i16.values[2]).toBeLessThan(0);
    const h = Q.innerSum(qw, qx, 'fp16');
    expect(h.values[0]).toBe(16128);
    expect(h.first.inexact).toBe(1);
    expect(h.values[3]).toBe(64512);
    expect(h.first.inf).toBe(5);
    expect(h.values[4]).toBe(Infinity);
    for (const t of ['int32', 'fp32'] as const) {
      const r = Q.innerSum(qw, qx, t);
      expect(r.values[31]).toBe(516128);
      expect(r.first).toEqual({});
    }
  });
  it('a pre-divided activation of 1e-4 / 4096 rounds to 0 in FP16 and survives in float', () => {
    expect(Q.FP16_MIN_SUBNORMAL).toBe(minSubnormal(FP16));
    expect(1e-4 / 4096).toBeLessThan(Q.FP16_MIN_SUBNORMAL / 2);
    expect(Q.halfPrescaled(1e-4, 12)).toBe(0);
    expect(Math.fround(1e-4 / 4096)).toBeGreaterThan(0);
    expect(Q.halfPrescaled(1e-4, 4)).toBeGreaterThan(0);
  });
  it('the ladder: integer inside a block and float across blocks stays at float rounding; FP16 does not', () => {
    const base = { data: 'realistic', n: 4096, pairing: 'q8q8' } as const;
    const good = Q.accumulate({ ...base, inner: 'int32', outer: 'fp32' });
    expect(good.relErr).toBeLessThan(1e-6);
    const halfOuter = Q.accumulate({ ...base, inner: 'int32', outer: 'fp16' });
    expect(halfOuter.relErr).toBeGreaterThan(1000 * good.relErr);
    const halfInner = Q.accumulate({ ...base, inner: 'fp16', outer: 'fp32' });
    expect(halfInner.rounded).toBeGreaterThan(64);
    expect(halfInner.relErr).toBeGreaterThan(100 * good.relErr);
    // int16 wraps even on realistic Q8 data, in a few blocks
    const short = Q.accumulate({ ...base, inner: 'int16', outer: 'fp32' });
    expect(short.wrapped).toBeGreaterThan(0);
    expect(short.relErr).toBeGreaterThan(0.01);
    // and on the worst case FP16 overflows every block
    const w = Q.accumulate({ ...base, data: 'worst', inner: 'fp16', outer: 'fp32' });
    expect(w.overflowed).toBe(128);
    expect(w.relErr).toBe(Infinity);
  });
  it('an FP16 outer sum drifts as the row grows; an FP32 one does not', () => {
    const at = (n: number, outer: 'fp16' | 'fp32') => Q.accumulate({ inner: 'int32', outer, data: 'realistic', n, pairing: 'q8q8' });
    expect(at(16384, 'fp16').absErr).toBeGreaterThan(10 * at(256, 'fp16').absErr);
    expect(at(16384, 'fp16').absErr).toBeGreaterThan(1000 * at(16384, 'fp32').absErr);
    expect(at(16384, 'fp32').relErr).toBeLessThan(1e-6);
    expect(Q.LADDER_LENGTHS[0]).toBe(32);
    expect(Q.LADDER_LENGTHS[Q.LADDER_LENGTHS.length - 1]).toBe(16384);
  });
});

describe('the bug spot', () => {
  const R = Object.fromEntries(Q.FRAGMENTS.map((id) => [id, Q.runFragment(id)])) as Record<Q.FragmentId, Q.FragmentRun>;
  it('only the int-then-float fragment is safe', () => {
    expect(Q.FRAGMENTS.filter((id) => R[id].safe)).toEqual(['int-then-float']);
  });
  it('the short sum passes realistic data and wraps on the fifth dp4a step of the worst case', () => {
    expect(R['short-sum'].realistic).toBe(R['int-then-float'].realistic);
    expect(R['short-sum'].worst.event).toBe('wrap');
    expect(R['short-sum'].worst.at).toBe(5);
    expect(R['short-sum'].worst.relErr).toBeGreaterThan(1);
  });
  it('the __hfma2 row reaches infinity on the worst case and drifts on realistic data', () => {
    expect(R['cuda-hfma2'].worst.event).toBe('inf');
    expect(R['cuda-hfma2'].realistic).toBeGreaterThan(1000 * R['int-then-float'].realistic);
  });
  it('dot(half4) does not overflow here but rounds every four products to half', () => {
    expect(R['metal-half-dot'].worst.event).toBe('none');
    expect(R['metal-half-dot'].realistic).toBeGreaterThan(10 * R['int-then-float'].realistic);
  });
  it('int then float is exact on the worst case: 128 blocks of 28,448', () => {
    expect(R['int-then-float'].worst.result).toBe(128 * 28448);
    expect(R['int-then-float'].worst.relErr).toBe(0);
    expect(R['int-then-float'].realistic).toBeLessThan(1e-6);
  });
});

describe('the scenes', () => {
  it('integer path: eight blocks over four lanes; the gap from the float path is the activation rounding', () => {
    const s = Q.integerSceneRow();
    expect(s.c.length).toBe(Q.SCENE_BLOCKS);
    expect(s.outC).not.toBe(s.outB);
    expect(Math.abs(s.outB - s.exact)).toBeLessThan(Math.abs(s.outC - s.exact) / 10);
    expect(s.case0.block).toEqual(Q.makeCase().block);
  });
  it('narrow accumulator: the four traces and the 128-block drift', () => {
    const s = Q.narrowScene();
    expect(s.traces.int16.first.wrap).toBe(3);
    expect(s.traces.fp16.first.inf).toBe(5);
    expect(s.traces.int32.values[31]).toBe(516128);
    expect(s.outerHalf.length).toBe(128);
    expect(s.outerHalf[127]).toBeGreaterThan(1000 * s.outerFloat[127]);
  });
});
