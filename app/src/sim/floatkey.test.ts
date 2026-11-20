import { describe, expect, it } from 'vitest';
import {
  CUB_OPTIONS,
  E1,
  NEG_QNAN,
  POS_QNAN,
  PRESETS,
  SIGN,
  bitsOf,
  explore,
  fromBits,
  hex,
  isNaNBits,
  items,
  keyAsc,
  keyDesc,
  rawAscending,
  referenceDesc,
  show,
  signFlip,
  stableBy,
  twiddle,
  zeroStep,
} from './floatkey';

const preset = (id: string) => PRESETS.find((p) => p.id === id)!;

describe("E1's table, bit for bit (notes/today/ch14-facts.md)", () => {
  for (const e of E1)
    it(`${e.label}: bits ${hex(e.bits)}, key ${hex(e.key)}, canon ${hex(e.canon)}, desc ${hex(e.desc)}`, () => {
      expect(twiddle(e.bits)).toBe(e.key >>> 0);
      expect(keyAsc(e.bits)).toBe(e.canon >>> 0);
      expect(keyDesc(e.bits)).toBe(e.desc >>> 0);
    });
  it('the bits are those of the values the labels name', () => {
    expect(bitsOf(-Infinity)).toBe(0xff800000);
    expect(bitsOf(-1)).toBe(0xbf800000);
    expect(bitsOf(-0)).toBe(0x80000000);
    expect(bitsOf(0)).toBe(0);
    expect(bitsOf(1)).toBe(0x3f800000);
    expect(bitsOf(Infinity)).toBe(0x7f800000);
    expect(fromBits(0x00000001)).toBe(2 ** -149);
    expect(fromBits(0x7f7fffff)).toBe(3.4028234663852886e38);
  });
  it('the plain keys are strictly increasing down the table', () => {
    for (let k = 0; k + 1 < E1.length; k++) expect(twiddle(E1[k].bits)).toBeLessThan(twiddle(E1[k + 1].bits));
  });
});

describe('the transform orders every float', () => {
  // A sweep over the whole float32 line: bit patterns at a fixed stride plus the special values.
  const pats: number[] = [];
  for (let u = 0; u < 2 ** 32; u += 0x00fedcb7) pats.push(u >>> 0);
  pats.push(0, SIGN, 1, (SIGN | 1) >>> 0, 0x7f800000, 0xff800000, 0x7f7fffff, 0xff7fffff, 0x00800000, 0x807fffff);
  const finite = [...new Set(pats)].filter((u) => !isNaNBits(u));
  // Numeric order, with -0.0 before +0.0.
  const byValue = finite.slice().sort((a, b) => {
    const x = fromBits(a);
    const y = fromBits(b);
    if (x !== y) return x < y ? -1 : 1;
    return a === SIGN ? -1 : b === SIGN ? 1 : 0;
  });
  it(`plain keys strictly increase with the value over ${byValue.length} floats`, () => {
    for (let k = 0; k + 1 < byValue.length; k++) expect(twiddle(byValue[k])).toBeLessThan(twiddle(byValue[k + 1]));
  });
  it('with the -0.0 step, keys never decrease, and the only tie is -0.0 with +0.0', () => {
    for (let k = 0; k + 1 < byValue.length; k++) {
      const a = keyAsc(byValue[k]);
      const b = keyAsc(byValue[k + 1]);
      if (a === b) expect([byValue[k], byValue[k + 1]]).toEqual([SIGN, 0]);
      else expect(a).toBeLessThan(b);
    }
  });
  it('the descending key reverses the order', () => {
    for (let k = 0; k + 1 < byValue.length; k++) expect(keyDesc(byValue[k])).toBeGreaterThanOrEqual(keyDesc(byValue[k + 1]));
  });
});

describe('NaN, by bits (CUB: "sorted according to their bit representations")', () => {
  it('+qNaN maps above +inf and -qNaN below -inf', () => {
    expect(keyAsc(POS_QNAN)).toBe(0xffc00000);
    expect(keyAsc(NEG_QNAN)).toBe(0x003fffff);
    expect(keyAsc(POS_QNAN)).toBeGreaterThan(keyAsc(0x7f800000));
    expect(keyAsc(NEG_QNAN)).toBeLessThan(keyAsc(0xff800000));
  });
  it("the edges preset sorts descending as CUB's rule says: +NaN first, -NaN last, -0.0 tied with +0.0", () => {
    const r = referenceDesc(items(preset('edges').bits));
    expect(r.rule).toBe("CUB's bits rule");
    expect(r.order.map((x) => show(x.bits))).toEqual([
      '+NaN', '+inf', 'FLT_MAX', '1.0', '1.4e−45', '−0.0', '+0.0', '−1.4e−45', '−1.0', '−FLT_MAX', '−inf', '−NaN',
    ]);
  });
});

describe('the -0.0 step and ties', () => {
  it('without the step +0.0 and -0.0 have different keys; with it they tie', () => {
    expect(twiddle(0)).toBe(0x80000000);
    expect(twiddle(SIGN)).toBe(0x7fffffff);
    expect(zeroStep(SIGN)).toBe(0);
    expect(keyAsc(0)).toBe(keyAsc(SIGN));
  });
  it('question q-zero: -0.0 at token 2 and +0.0 at token 9, sorted descending', () => {
    const row = Array(10).fill(bitsOf(1));
    row[2] = SIGN;
    row[9] = 0;
    for (let t = 0; t < 10; t++) if (t !== 2 && t !== 9) row[t] = bitsOf(t < 5 ? 3 - t : -t);
    const zeroOrder = (o: typeof CUB_OPTIONS) => explore(items(row), o).order.filter((t) => t === 2 || t === 9);
    expect(zeroOrder({ ...CUB_OPTIONS, zero: false })).toEqual([9, 2]);
    expect(zeroOrder(CUB_OPTIONS)).toEqual([2, 9]);
    expect(referenceDesc(items(row)).order.map((x) => x.token).filter((t) => t === 2 || t === 9)).toEqual([2, 9]);
  });
  it('signs and zeros: without the step the zeros come out swapped, and only they', () => {
    const r = explore(items(preset('zeros').bits), { ...CUB_OPTIONS, zero: false });
    expect(r.order).toEqual([5, 4, 3, 2, 1, 0]);
    expect(r.reference).toEqual([5, 4, 2, 3, 1, 0]);
    expect(r.match).toEqual([true, true, false, false, true, true]);
    expect(explore(items(preset('zeros').bits), CUB_OPTIONS).firstMismatch).toBe(-1);
  });
  it('question q-ties-desc: reversing an ascending sort reverses ties; inverting does not', () => {
    const xs = items(preset('ties').bits);
    expect(explore(xs, CUB_OPTIONS).order).toEqual([1, 4, 3, 7, 6, 0, 2, 5]);
    expect(explore(xs, { ...CUB_OPTIONS, descBy: 'reverse' }).order).toEqual([4, 1, 7, 3, 6, 0, 2, 5]);
    const rev = explore(xs, { ...CUB_OPTIONS, descBy: 'reverse' });
    expect(rev.firstMismatch).toBe(0);
    expect(rev.match).toEqual([false, false, false, false, true, true, true, true]);
  });
});

describe('the partial transforms the explorer offers', () => {
  it('raw bits put the negatives after the positives, in reverse order (the prediction p-raw-order)', () => {
    expect(rawAscending([-2, -1, 0.5, 1])).toEqual([0.5, 1, -1, -2]);
    expect(rawAscending([-2, -1, 0.5, 1]).indexOf(-1)).toBe(2);
  });
  it('question q-negatives: flipping only the sign bit leaves -2.0 above -1.0', () => {
    expect(signFlip(bitsOf(-1))).toBe(0x3f800000);
    expect(signFlip(bitsOf(-2))).toBe(0x40000000);
    expect(signFlip(bitsOf(-2))).toBeGreaterThan(signFlip(bitsOf(-1)));
    expect(twiddle(bitsOf(-2))).toBeLessThan(twiddle(bitsOf(-1)));
  });
  it('on the ties preset: no transform and sign-only both fail, the full transform matches', () => {
    const xs = items(preset('ties').bits);
    const none = explore(xs, { ...CUB_OPTIONS, transform: 'none' });
    const sign = explore(xs, { ...CUB_OPTIONS, transform: 'sign' });
    // raw bits, descending: negatives first (largest bits), most negative first
    expect(none.order).toEqual([5, 2, 0, 1, 4, 3, 7, 6]);
    // sign flip only: positives above negatives, but the negatives in reverse
    expect(sign.order).toEqual([1, 4, 3, 7, 6, 5, 2, 0]);
    expect(none.firstMismatch).toBe(0);
    expect(sign.firstMismatch).toBe(5);
    expect(explore(xs, CUB_OPTIONS).firstMismatch).toBe(-1);
  });
});

describe('references agree where both apply', () => {
  it('on rows without NaN, CUB\'s rule equals std::stable_sort with > (many random rows with ties and zeros)', () => {
    let s = 7;
    const rnd = () => (s = (Math.imul(s, 1103515245) + 12345) >>> 0) / 2 ** 32;
    const pool = [-3, -1.5, -1, -0, 0, 0.25, 1, 1, 2.5, -Infinity, Infinity, 2 ** -149, -(2 ** -149)];
    for (let r = 0; r < 300; r++) {
      const row = Array.from({ length: 1 + Math.floor(rnd() * 12) }, () => bitsOf(pool[Math.floor(rnd() * pool.length)]));
      const xs = items(row);
      const cub = stableBy(xs, (x) => keyDesc(x.bits)).map((x) => x.token);
      expect(cub).toEqual(referenceDesc(xs).order.map((x) => x.token));
      expect(explore(xs, CUB_OPTIONS).firstMismatch).toBe(-1);
    }
  });
});
