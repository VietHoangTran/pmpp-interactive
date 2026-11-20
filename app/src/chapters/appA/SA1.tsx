import { useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BF16, E2M1, E4M3, E5M2, FP16, FP32, TOY6,
  anatomy, bitString, bitsOfValue, classify, encode, excessBias, excessCode, excessTable, fields, flipBit,
  maxFinite, minNormal, nextDown, nextUp, precisionBits, show, showBits, totalBits, twosComplement, ulp,
  type Anatomy, type Format,
} from '../../sim/minifloat';

// ---------------------------------------------------------------- small pieces
const minus = (n: number | string) => String(n).replace(/-/g, '−');
const Pow = ({ k }: { k: number | string }) => (
  <>
    2<sup>{minus(k)}</sup>
  </>
);
const hex = (f: Format, bits: number) => '0x' + (bits >>> 0).toString(16).toUpperCase().padStart(Math.ceil(totalBits(f) / 4), '0');
/** Parse what a reader types: decimals, exponents, inf and nan. Returns null for nothing usable. */
function parseValue(t: string): number | null {
  const s = t.trim().toLowerCase().replace(/[−–]/g, '-').replace(/[,_\s]/g, '');
  if (s === '') return null;
  if (s === 'inf' || s === '+inf' || s === 'infinity' || s === '+infinity') return Infinity;
  if (s === '-inf' || s === '-infinity') return -Infinity;
  if (s === 'nan') return NaN;
  const v = Number(s);
  return Number.isNaN(v) ? null : v;
}

// ---------------------------------------------------------------- explorer: excess codes (Fig. A.1)
type Coding = 'excess' | 'twos';
function ExcessBench() {
  const [e, setE] = useState(3);
  const bias = excessBias(e);
  const [a, setA] = useState(-2);
  const [b, setB] = useState(1);
  const [pred, setPred] = useState<Partial<Record<Coding, 'A' | 'B'>>>({});
  const clamp = (v: number) => Math.max(-bias, Math.min(bias, v));
  const A = clamp(a);
  const B = clamp(b);
  const rows = excessTable(e);
  const code = (k: Coding, v: number) => (k === 'excess' ? excessCode(e, v) : twosComplement(e, v));
  const exps: number[] = [];
  for (let v = -bias; v <= bias; v++) exps.push(v);
  const byTwos = rows.slice().sort((x, y) => x.twos - y.twos);
  const setWidth = (w: number) => {
    setE(w);
    setPred({});
  };

  const verdict = (k: Coding) => {
    const ca = code(k, A);
    const cb = code(k, B);
    const larger = ca > cb ? 'A' : 'B';
    const sameOrder = (ca > cb) === (A > B);
    const p = pred[k];
    return (
      <div data-coding={k} data-same-order={String(sameOrder)} style={{ marginBottom: '0.7rem' }}>
        <p style={{ margin: '0 0 0.35rem' }}>
          <b>{k === 'excess' ? `Excess-${bias} codes` : "Two's-complement codes"}.</b> Read as unsigned integers, which code is larger?
        </p>
        <div className="row" style={{ marginBottom: '0.35rem' }}>
          {(['A', 'B'] as const).map((w) => (
            <button key={w} aria-pressed={p === w} disabled={A === B} onClick={() => setPred({ ...pred, [k]: w })}>
              {w}'s code
            </button>
          ))}
        </div>
        {p && A !== B ? (
          <p style={{ margin: 0 }} data-part="reveal">
            A = {minus(A)} is stored as <span className="mono">{bitString(ca, e)}</span> ({ca}); B = {minus(B)} as{' '}
            <span className="mono">{bitString(cb, e)}</span> ({cb}). {larger}'s code is larger.{' '}
            {p === larger ? <span className="chip ok">you said {p}: right</span> : <span className="chip fail">you said {p}: wrong</span>}{' '}
            {sameOrder ? 'The codes are in the same order as the exponents.' : 'The codes are in the opposite order to the exponents.'}
          </p>
        ) : null}
      </div>
    );
  };

  return (
    <Bench
      title="Excess codes"
      task="Pick two exponents, say which code is larger as an unsigned integer, then change the width."
      foot={<>Fig. A.1 for any width from 2 to 5 bits. The all-ones excess code is the reserved pattern; its two's-complement column shows the bits it shares with the most negative two's-complement number.</>}
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Exponent bits e = {e}, bias {bias}
          <input type="range" min="2" max="5" value={e} aria-label="exponent bits" onChange={(ev) => setWidth(+ev.target.value)} />
        </label>
        <label className="field">
          Exponent A
          <select value={A} aria-label="exponent A" onChange={(ev) => { setA(+ev.target.value); setPred({}); }}>
            {exps.map((v) => <option key={v} value={v}>{minus(v)}</option>)}
          </select>
        </label>
        <label className="field">
          Exponent B
          <select value={B} aria-label="exponent B" onChange={(ev) => { setB(+ev.target.value); setPred({}); }}>
            {exps.map((v) => <option key={v} value={v}>{minus(v)}</option>)}
          </select>
        </label>
      </div>
      <div className="cols">
        <div className="panel">
          {A === B ? <p style={{ margin: '0 0 0.7rem' }}>Pick two different exponents.</p> : null}
          {verdict('excess')}
          {verdict('twos')}
          <p className="muted" style={{ margin: 0 }} data-part="twos-order">
            Sorted by two's-complement code, the exponents run{' '}
            {byTwos.map((r) => (r.value === null ? '(reserved)' : minus(r.value))).join(', ')}: every negative after every positive.
          </p>
        </div>
        <div className="panel">
          <table className="data" style={{ margin: 0 }} aria-label="Excess codes sorted by excess code">
            <thead>
              <tr>
                <th>Two's complement</th>
                <th className="num">Exponent</th>
                <th>Excess-{bias}</th>
                <th>Picked</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const tag = r.value === null ? '' : [r.value === A ? 'A' : '', r.value === B ? 'B' : ''].filter(Boolean).join(', ');
                return (
                  <tr key={r.excess} data-excess={r.excess} style={tag ? { fontWeight: 650, background: 'var(--rule-soft)' } : undefined}>
                    <td className="mono">{bitString(r.twos, e)}</td>
                    <td className="num">{r.value === null ? 'reserved' : minus(r.value)}</td>
                    <td className="mono">{bitString(r.excess, e)}</td>
                    <td>{tag}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: bits and value (Eq. A.1)
const BIT_FORMATS: Format[] = [TOY6, FP16, BF16, FP32, E4M3, E5M2, E2M1];
const SHORT: Record<string, string> = {
  toy6: "Book's 6-bit",
  fp16: 'FP16',
  bf16: 'BF16',
  fp32: 'FP32',
  e4m3: 'E4M3',
  e5m2: 'E5M2',
  e2m1: 'E2M1',
};

function classWords(a: Anatomy): string {
  switch (a.cls) {
    case 'normal':
      return 'normal';
    case 'subnormal':
      return 'subnormal (denormal)';
    case 'zero':
      return Object.is(a.value, -0) ? 'zero (negative zero)' : 'zero';
    case 'inf':
      return 'infinity';
    case 'qnan':
      return 'NaN (quiet)';
    case 'snan':
      return 'NaN (signaling)';
    case 'nan':
      return 'NaN';
    case 'reserved':
      return 'reserved pattern';
  }
}

function Formula({ f, a }: { f: Format; a: Anatomy }) {
  const mBits = bitString(a.M, f.m);
  if (a.rule === 'special') {
    if (a.cls === 'reserved')
      return <>The book sets this exponent code aside (p. 585). IEEE spends it on infinity and NaN (§A.3).</>;
    if (a.cls === 'inf') return <>Exponent all ones, mantissa zero: (−1)<sup>{a.s}</sup> × infinity (§A.3).</>;
    if (f.id === 'e4m3') return <>S.1111.111 is E4M3's only NaN pattern; it has no infinity.</>;
    return <>Exponent all ones, mantissa not zero: NaN (§A.3).</>;
  }
  if (a.rule === 'flushed-pattern') return <>Exponent code 0 under abrupt underflow: zero.</>;
  const sig = (a.s ? '−' : '') + show(a.significand!, 9);
  if (a.rule === 'denormal')
    return (
      <span className="mono" data-part="formula">
        (−1)<sup>{a.s}</sup> × 0.{mBits}<sub>B</sub> × 2<sup>1 − {f.bias}</sup> = {sig} × <Pow k={a.scale!} /> = {show(a.value, 9)}
      </span>
    );
  return (
    <span className="mono" data-part="formula">
      (−1)<sup>{a.s}</sup> × 1.{mBits}<sub>B</sub> × 2<sup>{a.E} − {f.bias}</sup> = {sig} × <Pow k={a.scale!} /> = {show(a.value, 9)}
    </span>
  );
}
function BitsBench() {
  const [fid, setFid] = useState('toy6');
  const f = BIT_FORMATS.find((g) => g.id === fid)!;
  const [bits, setBits] = useState(() => encode(TOY6, 0.5));
  const [text, setText] = useState('');
  const [typed, setTyped] = useState<number | null>(null);
  const [note, setNote] = useState<ReactNode>(null);
  const a = anatomy(f, bits);
  const n = totalBits(f);
  const x = fields(f, bits);

  const choose = (g: Format) => {
    if (g.id === f.id) return;
    if (totalBits(g) === n) {
      setNote(
        <>
          Same {n} bits as before. Read as {f.name} they were <b>{show(a.value)}</b>.
        </>,
      );
      setTyped(null);
      setText('');
    } else {
      const v = typed ?? a.value;
      const nb = encode(g, v);
      setBits(nb >= 0 ? nb : 0);
      setNote(Number.isFinite(v) || Number.isNaN(v) ? <>Re-encoded {show(v)} as the nearest {g.name} value.</> : null);
    }
    setFid(g.id);
  };
  const flip = (i: number) => {
    setBits(flipBit(bits, i));
    setTyped(null);
    setText('');
    setNote(null);
  };
  const onType = (t: string) => {
    setText(t);
    const v = parseValue(t);
    setNote(null);
    if (v === null) return;
    const nb = encode(f, v);
    setTyped(v);
    if (nb >= 0) setBits(nb);
  };
  const go = (to: number | null) => {
    if (to === null) return;
    setBits(bitsOfValue(f, to));
    setTyped(null);
    setText('');
    setNote(null);
  };
  const up = Number.isNaN(a.value) ? null : nextUp(f, a.value);
  const down = Number.isNaN(a.value) ? null : nextDown(f, a.value);
  const c = typed !== null ? classify(f, typed) : null;

  // Bit indices of each field, most significant first.
  const idx = (lo: number, w: number) => Array.from({ length: w }, (_, k) => lo + w - 1 - k);
  const groups = [
    ...(f.signed ? [{ key: 'S', label: 'S', idx: idx(f.e + f.m, 1) }] : []),
    { key: 'E', label: `E: exponent, ${f.e} bits`, idx: idx(f.m, f.e) },
    { key: 'M', label: `M: mantissa, ${f.m} bit${f.m === 1 ? '' : 's'}`, idx: idx(0, f.m) },
  ];
  const bitOf = (i: number) => Math.floor((bits >>> 0) / 2 ** i) % 2;

  return (
    <Bench
      title="Bits and value"
      task="Flip bits, or type a number. Then switch between FP16 and BF16 and watch the same bits change meaning."
      foot={<>Definitions: {f.source}. Values are computed exactly and shown to 9 significant digits.</>}
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="seg" role="group" aria-label="Format" style={{ flexWrap: 'wrap', rowGap: 4 }}>
          {BIT_FORMATS.map((g) => (
            <button key={g.id} aria-pressed={g.id === fid} onClick={() => choose(g)}>
              {SHORT[g.id]}
            </button>
          ))}
        </span>
        {f.notInBook ? (
          <span className="chip" style={{ border: '1px solid var(--rule)' }} data-part="badge">
            not in the book
          </span>
        ) : null}
      </div>

      <div className="row" style={{ alignItems: 'flex-start', gap: '0.5rem 0.9rem', marginBottom: '0.9rem' }} data-part="bits">
        {groups.map((g) => (
          <div key={g.key} data-field={g.key} style={{ display: 'inline-flex', flexDirection: 'column', gap: 3, maxWidth: '100%' }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              {g.idx.map((i) => (
                <button
                  key={i}
                  data-bit={i}
                  aria-label={`bit ${i}, ${g.key}`}
                  aria-pressed={bitOf(i) === 1}
                  onClick={() => flip(i)}
                  className="mono"
                  style={{ width: '1.45rem', padding: '0.3rem 0', fontSize: '0.8rem', borderColor: 'var(--ink-3)' }}
                >
                  {bitOf(i)}
                </button>
              ))}
            </div>
            <div aria-hidden="true" style={{ height: 6, border: '1px solid var(--ink-3)', borderTop: 0 }} />
            <div className="muted" style={{ fontSize: '0.74rem', textAlign: 'center' }}>
              {g.label}
            </div>
          </div>
        ))}
      </div>

      <div className="cols">
        <div className="panel">
          <table className="data" style={{ margin: 0 }} aria-label="How the bits become a value">
            <tbody>
              <tr>
                <th>Pattern</th>
                <td>
                  <span className="mono" data-part="pattern">{showBits(f, bits)}</span> <span className="muted">({hex(f, bits)})</span>
                </td>
              </tr>
              <tr>
                <th>Exponent</th>
                <td data-part="exponent">
                  E = <span className="mono">{x.eBits}</span> = {x.E}, bias {f.bias}.{' '}
                  {a.rule === 'normal' || a.rule === 'nozero-low' ? (
                    <>
                      Power of two: E − bias = <b data-part="unbiased">{minus(a.scale!)}</b>.
                    </>
                  ) : a.rule === 'denormal' ? (
                    <>
                      E = 0: the power is fixed at 1 − bias = <b data-part="unbiased">{minus(a.scale!)}</b> (§A.2).
                    </>
                  ) : (
                    <>All ones: a special pattern.</>
                  )}
                </td>
              </tr>
              <tr>
                <th>Significand</th>
                <td data-part="significand">
                  {a.hidden === null ? (
                    '–'
                  ) : (
                    <>
                      hidden bit <b data-part="hidden">{a.hidden}</b> (not stored), then M:{' '}
                      <span className="mono">
                        {a.hidden}.{bitString(a.M, f.m)}
                        <sub>B</sub>
                      </span>{' '}
                      = {show(a.significand!, 9)}
                    </>
                  )}
                </td>
              </tr>
              <tr>
                <th>Formula</th>
                <td>
                  <Formula f={f} a={a} />
                </td>
              </tr>
              <tr>
                <th>Value</th>
                <td>
                  <b style={{ fontSize: '1.15rem' }} data-part="value">{show(a.value, 9)}</b>{' '}
                  <span className="muted" data-part="class">{classWords(a)}</span>
                </td>
              </tr>
            </tbody>
          </table>
          {note ? <p style={{ margin: '0.6rem 0 0' }} data-part="note">{note}</p> : null}
        </div>

        <div className="panel">
          <label className="field" style={{ marginBottom: '0.6rem' }}>
            Type a value (decimal, 1e-5, inf, nan)
            <input type="text" inputMode="decimal" value={text} aria-label="value to encode" onChange={(ev) => onType(ev.target.value)} />
          </label>
          {c ? (
            <p style={{ margin: '0 0 0.7rem' }} data-part="typed">
              {c.status === 'inf' && Number.isFinite(c.input) ? (
                <>
                  {show(c.input)} is beyond {f.name}'s largest value, {show(maxFinite(f))}: it overflows to <b>{show(c.value)}</b>.
                </>
              ) : c.status === 'nan' && Number.isFinite(c.input) ? (
                <>
                  {show(c.input)} is beyond {f.name}'s largest value, {show(maxFinite(f))}, and {f.name} has no infinity: the
                  non-saturating conversion gives <b>NaN</b>.
                </>
              ) : c.status === 'saturated' ? (
                <>
                  {show(c.input)} is beyond {f.name}'s largest value; with no infinity or NaN to give, it is clamped to{' '}
                  <b>{show(c.value)}</b>.
                </>
              ) : Number.isFinite(c.input) ? (
                <>
                  Nearest {f.name} value (round to nearest, ties to even): <b>{show(c.value, 9)}</b>.{' '}
                  {c.exact ? (
                    <>Exact: no rounding error.</>
                  ) : (
                    <>
                      Error {show(c.error, 3)}, which is {show(Math.abs(c.error) / ulp(f, c.input), 3)} of the spacing between neighbouring values there.
                    </>
                  )}
                </>
              ) : (
                <>Stored as {show(c.value)}.</>
              )}
            </p>
          ) : null}
          <div className="row" style={{ marginBottom: '0.8rem' }}>
            <button onClick={() => go(down)} disabled={down === null}>
              Next value down
            </button>
            <button onClick={() => go(up)} disabled={up === null}>
              Next value up
            </button>
          </div>
          <p className="muted" style={{ margin: 0 }} data-part="facts">
            {f.name}: {f.e} exponent bits give{' '}
            {f.subnormals === 'nozero' ? (
              <>
                values from {show(minNormal(f))} to {show(maxFinite(f))}, and no zero: §A.1 reads E = {bitString(0, f.e)} with
                the same formula as every other code (§A.2 shows the fixes).
              </>
            ) : (
              <>
                normal values from {show(minNormal(f))} to {show(maxFinite(f))}.
              </>
            )}{' '}
            {f.m} mantissa
            bits plus the hidden one give {precisionBits(f)} significant bits: {2 ** f.m} values between each power of two and
            the next.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the dtype table
function DtypeTable() {
  const rows = [
    { f: FP32, torch: 'torch.float32' },
    { f: FP16, torch: 'torch.float16' },
    { f: BF16, torch: 'torch.bfloat16' },
  ];
  return (
    <table className="data" aria-label="The three formats behind float32, float16 and bfloat16">
      <thead>
        <tr>
          <th>Format</th>
          <th>PyTorch dtype</th>
          <th>S-E-M</th>
          <th className="num">Bias</th>
          <th className="num">Significant bits</th>
          <th className="num">Largest finite</th>
          <th className="num">Smallest normal</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({ f, torch }) => (
          <tr key={f.id} data-format={f.id}>
            <td>{f.name}</td>
            <td>
              <code>{torch}</code>
            </td>
            <td>
              1-{f.e}-{f.m}
            </td>
            <td className="num">{f.bias}</td>
            <td className="num">{precisionBits(f)}</td>
            <td className="num">{show(maxFinite(f), 4)}</td>
            <td className="num">
              <Pow k={Math.log2(minNormal(f))} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSA1: Record<string, Question> = {
  'appA/A.1/q-encode': {
    kind: 'choice',
    id: 'appA/A.1/q-encode',
    prompt: (
      <>
        In the book's 6-bit format (1 sign bit, 3 exponent bits in excess-3, 2 mantissa bits), which pattern stores 0.75?
      </>
    ),
    options: [
      { label: <span className="mono">001010</span>, correct: true },
      { label: <span className="mono">001100</span>, why: 'That is 1.0: exponent code 011 is exponent 0 in excess-3, and M = 00.' },
      { label: <span className="mono">010010</span>, why: 'That is 3.0: exponent code 100 is exponent +1, so 1.10B × 2^1.' },
      { label: <span className="mono">000110</span>, why: 'That is 0.375: exponent code 001 is exponent −2.' },
    ],
    explain: (
      <p>
        0.75 is 0.11 in binary, which normalised is 1.1B × 2<sup>−1</sup>. Sign 0. Exponent −1, plus the bias 3, is 2:
        code 010. The significand 1.10B loses its leading 1, leaving M = 10. Together: 0 010 10.
      </p>
    ),
    source: '§A.1, p. 586 (the method of the worked example for 0.5)',
  },
  'appA/A.1/q-max-half': {
    kind: 'numeric',
    id: 'appA/A.1/q-max-half',
    prompt: <>FP16 has 1 sign bit, 5 exponent bits with bias 15, and 10 mantissa bits. What is its largest finite value?</>,
    answer: 65504,
    tol: 0,
    explain: (
      <p>
        The all-ones exponent code, 31, is reserved, so the largest usable code is 30, and 30 − 15 = 15. The largest
        mantissa is all ones, so the significand is 2 − 2<sup>−10</sup>. (2 − 2<sup>−10</sup>) × 2<sup>15</sup> = 65536 −
        32 = 65504. An attention score or an activation above that has nowhere to go in FP16; §A.2 shows what happens to it.
      </p>
    ),
    source: 'our arithmetic on the §A.1 formula; NVIDIA guide Table 51 prints the same value (appA-facts A2)',
  },
  'appA/A.1/q-unsigned-compare': {
    kind: 'choice',
    id: 'appA/A.1/q-unsigned-compare',
    prompt: (
      <>
        A sort reads the bits of non-negative FP32 values as 32-bit unsigned integers and sorts those integers. Does that put
        the floats in the right order?
      </>
    ),
    options: [
      {
        label: 'Yes: the sign bit is 0 for all of them, and the excess-coded exponent sits above the mantissa.',
        correct: true,
      },
      {
        label: 'No: the exponent is stored with a bias, so integer order and value order differ.',
        why: 'The bias is what makes it work: excess codes rise with the exponents they stand for.',
      },
      {
        label: 'Only if none of the values is subnormal.',
        why: 'Subnormals have exponent code 0, below every normal value, and their mantissas order them among themselves. Still correct.',
      },
      {
        label: 'Only among values that share an exponent.',
        why: 'The exponent field is more significant than the mantissa field, so different exponents are compared first, which is the right order.',
      },
    ],
    explain: (
      <>
        <p>
          With the sign bit 0, a larger exponent gives a larger integer (excess codes are ordered), and for equal exponents
          a larger mantissa does. So integer order is value order.
        </p>
        <p>
          With negative values it fails. The sign is a separate bit, so every negative value reads as a larger integer than
          every positive one, and among negatives a larger magnitude reads larger. The same goes for −0.0, whose sign bit is
          set. Chapter 14's float-key transform fixes both (<a href={href('ch14', '14.4')}>§14.4</a>).
        </p>
      </>
    ),
    source: '§A.1, p. 585 (excess codes and the unsigned comparator); the bridge to §14.4 is ours',
  },
  'appA/A.1/q-bf16-top': {
    kind: 'choice',
    id: 'appA/A.1/q-bf16-top',
    prompt: <>You keep the top 16 bits of an FP32 value and drop the low 16. What do you have?</>,
    options: [
      { label: 'A BF16 value rounded toward zero: off by up to almost one BF16 step.', correct: true },
      {
        label: 'A BF16 value rounded to nearest: off by at most half a BF16 step.',
        why: 'Dropping bits never rounds up. FP32 1.0078124 keeps 1.0, though 1.0078125 is nearer.',
      },
      {
        label: 'An FP16 value.',
        why: 'FP16 has 5 exponent bits. The top 16 bits of FP32 keep all 8 of its exponent bits: that is the BF16 layout.',
      },
      {
        label: 'Nothing meaningful: the exponent field is cut in half.',
        why: 'The sign and the 8 exponent bits are the top 9 bits, so they survive whole.',
      },
    ],
    explain: (
      <p>
        BF16 is FP32's sign, its 8 exponent bits and the top 7 of its 23 mantissa bits. Discarding the low 16 bits always
        moves the value toward zero, by anything up to just under one BF16 step. llama.cpp converts with round to nearest
        even instead, by adding 0x7FFF plus the lowest kept bit before the shift (<a href={href('appA', 'A.4')}>§A.4</a>{' '}
        explains rounding), and it forces NaNs to stay quiet: a NaN whose set bits are all in the low half would otherwise
        truncate to infinity.
      </p>
    ),
    source: 'appA-facts A2 (BF16 layout) and D3 (llama.cpp ggml-impl.h); our arithmetic',
  },
};

// ---------------------------------------------------------------- the section
export function SA1() {
  return (
    <>
      <p>
        A float32 tensor in PyTorch is a run of 32-bit patterns, and a float16 or bfloat16 tensor a run of 16-bit ones.
        Every kernel you write for them either reads those bits as numbers or, for a sort or a conversion, works on the bits
        directly. This section is what the bits mean.
      </p>

      <h2>Three fields and one formula</h2>
      <p>
        The hardware splits each pattern into three fields: a sign bit S, an exponent field E and a mantissa field M.
        Leaving aside a few reserved patterns that §A.3 covers, the fields name one value:
      </p>
      <p className="mono" style={{ textAlign: 'center', fontSize: '1.05rem' }}>
        value = (−1)<sup>S</sup> · 1.M · 2<sup>E − bias</sup>
      </p>
      <Source section="A.1" pages="583–584" figs="Eq. (A.1)" />
      <p>
        S picks the sign: 0 for positive, 1 for negative. E, after a fixed offset called the bias is taken off, picks a power
        of two. M picks a point between that power of two and the next one. How E and M are stored is not arbitrary, and
        both choices show up in kernels: the hidden bit sets how many significant bits you get, and the biased exponent is
        what lets a GPU sort treat floats as integers.
      </p>
      <Today checked="2026-11-20 against the IEEE SA pages (notes/today/appA-facts.md, A1)">
        <p>
          IEEE 754-2019, the revision the book cites, is still the active standard. A revision project and a separate draft
          for machine-learning formats (P3109) are open; neither is a standard yet.
        </p>
      </Today>

      <h2>The bit you do not store</h2>
      <p>
        0.5 can be written in binary as 0.1B × 2<sup>0</sup>, as 10.0B × 2<sup>−2</sup> or as 1.0B × 2<sup>−1</sup>. The
        format allows only the last shape, with exactly one 1 before the binary point; the book calls numbers in that shape
        normalised. Two things follow. Every value has one pattern, so there is no wasted encoding. And since the digit
        before the point is always 1, it is never stored: an m-bit mantissa field holds only the digits after the point, and
        the significand has m + 1 bits. FP16 stores 10 mantissa bits and has 11 significant bits; FP32 stores 23 and has 24.
      </p>
      <Source section="A.1" pages="584" />

      <h2>Excess encoding</h2>
      <p>
        The exponent has to be signed. Two's complement, the integer encoding you already know, is the obvious choice. IEEE
        uses an excess code instead: add a fixed bias of 2<sup>e−1</sup> − 1 for an e-bit field, and store the sum as an
        unsigned number. With 3 exponent bits the bias is 3, so the exponent −1 is stored as 010 and +1 as 100.
      </p>
      <Source section="A.1" pages="585" figs="Fig. A.1" />
      <p>
        The reason is order. Excess codes rise with the exponents they stand for, so the cheapest circuit there is, an
        unsigned comparator, can compare two exponents directly. Two's-complement codes do not rise: read as unsigned
        numbers, every negative exponent looks larger than every positive one. Check both claims before reading on.
      </p>
      <ExcessBench />
      <p>
        The table has one code left over. Zero plus equally many positive and negative exponents is an odd count, and 2
        <sup>e</sup> codes is an even number, so whichever side took the last code would be one larger than the other. The
        book reserves the all-ones code; IEEE uses it for infinity and NaN, which is §A.3.
      </p>
      <Source section="A.1" pages="585" />

      <h3>Floats that sort as integers</h3>
      <p>
        The ordering survives at the scale of the whole pattern. For a non-negative float the sign bit is 0, the exponent
        field sits above the mantissa field, and both are ordered unsigned codes. So the bit patterns of non-negative floats,
        read as unsigned integers, are in the same order as the values. GPU radix sorts, which sort logits for top-k and
        top-p sampling, depend on this to treat floats as integer keys.
      </p>
      <Beyond>
        <p>
          The book stops at comparing exponents. Sorting whole floats as integers, and the fix for negative values, are in
          chapter 14 (<a href={href('ch14', '14.4')}>§14.4</a>), from CUB's documented float-key rule.
        </p>
      </Beyond>
      <Ask q={questionsSA1['appA/A.1/q-unsigned-compare']} />

      <h2>Encoding a number by hand</h2>
      <p>
        The book's running example is a 6-bit format: 1 sign bit, 3 exponent bits in excess-3, 2 mantissa bits. Take 0.5. It
        is positive, so S = 0. Normalised, 0.5 = 1.0B × 2<sup>−1</sup>, so the exponent is −1, stored as −1 + 3 = 2, which is
        010. The significand 1.00B loses its leading 1, leaving M = 00. The pattern is 0 010 00.
      </p>
      <Source section="A.1" pages="586" />
      <Ask q={questionsSA1['appA/A.1/q-encode']} />
      <p>
        Nothing in the method depends on the widths. With an n-bit exponent field the bias is 2<sup>n−1</sup> − 1, and a
        normalised pattern means (−1)<sup>S</sup> · 1.M · 2<sup>E − (2<sup>n−1</sup> − 1)</sup>. FP32 is n = 8, bias 127;
        FP16 is n = 5, bias 15.
      </p>
      <Source section="A.1" pages="586" />

      <h2>Bits and value</h2>
      <p>
        Now go the other way, from bits to a value, in the book's format and in the ones you use. Flip bits by clicking them,
        or type a number to see the pattern that stores it.
      </p>
      <BitsBench />
      <p>
        Things to find. In FP16, flip each exponent bit and the value jumps by a power of two; flip mantissa bits and it
        moves between one power of two and the next. Type 1 in FP16, then switch to BF16: the explorer keeps the same 16
        bits, and they now mean something else, because the two formats cut the fields in different places. Type 0.1 in
        each format and compare the errors: none of them can store it exactly.
      </p>
      <Simplification>
        <p>
          Values are exact, but the display rounds them to 9 significant digits. The explorer models encodings only; NaN
          payloads and how hardware treats them are §A.3.
        </p>
      </Simplification>

      <h2>The formats behind your dtypes</h2>
      <p>The three floating-point dtypes you use most are Eq. (A.1) with different field widths:</p>
      <DtypeTable />
      <p>
        BF16 is the top half of FP32: the same sign bit, the same 8 exponent bits and so the same range, with the mantissa
        cut to 7 bits, 8 significant. FP16 moves three of those exponent bits into the mantissa: 3 more significant bits and
        far less range. Which trade suits a tensor depends on whether its values need range or resolution; §A.2 makes that
        concrete.
      </p>
      <Beyond>
        <p>
          The book gives only single and double precision. FP16 and BF16 come from NVIDIA's guide (Tables 50 and 51), OCP's
          FP8 specification and PyTorch's dtype table (appA-facts A2). BF16 is not an IEEE format.
        </p>
      </Beyond>
      <Ask q={questionsSA1['appA/A.1/q-max-half']} />
      <Ask q={questionsSA1['appA/A.1/q-bf16-top']} />

      <h2>Their names on each backend</h2>
      <p>
        In kernel code these formats have different names on each backend, and the differences are where a single .cu file
        stops compiling for HIP unchanged. A file that includes <code>&lt;cuda_fp16.h&gt;</code> needs an <code>#if</code>{' '}
        to include <code>&lt;hip/hip_fp16.h&gt;</code> instead; one that names <code>__nv_bfloat16</code> also needs a
        typedef, which is what llama.cpp writes in its HIP header: <code>typedef __hip_bfloat16 nv_bfloat16;</code>. To ask
        a format for its limits in CUDA code, NVIDIA points to <code>cuda::std::numeric_limits</code> in{' '}
        <code>&lt;cuda/std/limits&gt;</code>.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 guide, ROCm 10.2 HIP docs, the MSL 4.1 specification and llama.cpp master (appA-facts A2, A16, B1, C1, D3)">
        <p>Not in the book. Each name below is from the vendor's current documentation.</p>
      </Today>
      <table className="data" aria-label="Half, bfloat16 and double on each backend">
        <thead>
          <tr>
            <th>Backend</th>
            <th>FP16</th>
            <th>BF16</th>
            <th>64-bit double</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA</td>
            <td>
              <code>__half</code>, <code>&lt;cuda_fp16.h&gt;</code>
            </td>
            <td>
              <code>__nv_bfloat16</code>, <code>&lt;cuda_bf16.h&gt;</code>; compute capability 8.0 or higher
            </td>
            <td>
              <code>double</code>
            </td>
          </tr>
          <tr>
            <td>HIP</td>
            <td>
              <code>__half</code>, <code>&lt;hip/hip_fp16.h&gt;</code>
            </td>
            <td>
              <code>__hip_bfloat16</code>, <code>&lt;hip/hip_bf16.h&gt;</code>
            </td>
            <td>
              <code>double</code>
            </td>
          </tr>
          <tr>
            <td>Metal</td>
            <td>
              <code>half</code>, built in (IEEE binary16 storage)
            </td>
            <td>
              <code>bfloat</code>, Metal 3.1 and Apple6 or later
            </td>
            <td>none: Metal has no <code>double</code></td>
          </tr>
        </tbody>
      </table>
      <p>
        The explorer also holds three formats the book mentions only by size, as 8-bit and 4-bit floating point: FP8 E4M3
        and E5M2, and FP4 E2M1. They are the same formula with fewer bits and different rules for the top exponent code;{' '}
        <a href={href('appA', 'A.5')}>§A.5</a> explains them.
      </p>
    </>
  );
}
