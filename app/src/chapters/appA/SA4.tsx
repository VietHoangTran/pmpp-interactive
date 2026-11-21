import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  ALIGN_PRESETS,
  BF16_PRESETS,
  MODE_LABEL,
  MODES,
  alignTrace,
  bf16FromBits,
  bitsFromF32,
  parseOperand,
  type AlignTrace,
  type Step,
} from '../../sim/fpadd';
import { BF16, FP16, FP32, TOY5, show, type Format, type Mode } from '../../sim/minifloat';
import { DisappearingAddendScene } from './DisappearingAddendScene';

const FACTS = 'notes/today/appA-facts.md';
const CHECKED = (items: string) => `2026-11-20 (${FACTS}, ${items})`;

// ---------------------------------------------------------------- the alignment stepper
const STEP_FORMATS: Format[] = [TOY5, FP16, BF16, FP32];
const FORMAT_LABEL: Record<string, string> = { toy5: "the book's 5-bit", fp16: 'FP16', bf16: 'BF16', fp32: 'FP32' };

const dotted = (w: string, carry = false) => (carry ? `${w.slice(0, 2)}.${w.slice(2)}` : `${w[0]}.${w.slice(1)}`);
const ulpsText = (e: number, belowOne = true) =>
  !Number.isFinite(e) ? '∞' : e >= 0.9995 && e < 1 + 1e-12 && belowOne && e !== 0 ? 'just under 1' : String(Number(e.toPrecision(4)));

function Tail({ bits }: { bits: string }) {
  if (!bits) return <span className="muted">(none)</span>;
  const shown = bits.length > 30 ? bits.slice(0, 28) : bits;
  return (
    <span className="mono" data-tail={bits}>
      {[...shown].map((b, i) => (
        <span key={i} style={i < 2 ? { textDecoration: 'underline', textUnderlineOffset: '3px' } : { color: 'var(--ink-2)' }}>{b}</span>
      ))}
      {bits.length > 30 ? <span className="muted">… ({bits.length - 28} more)</span> : null}
    </span>
  );
}

function GRSChips({ g, r, s }: { g: number; r: number; s: number }) {
  return (
    <span className="mono" data-grs={`${g}${r}${s}`}>
      guard {g} · round {r} · sticky {s}
    </span>
  );
}

function describe(st: Step, t: AlignTrace): string {
  switch (st.kind) {
    case 'compare': {
      const same = st.diff === 0;
      return (
        `Compare the exponents: ${show(st.big.value)} is 1.${st.big.sig.slice(1)} × 2^${st.big.exp} and ${show(st.small.value)} is 1.${st.small.sig.slice(1)} × 2^${st.small.exp}` +
        (st.small.stored === 'subnormal' ? ' (stored as a denormal, written here with its leading 1)' : '') +
        (same ? '. The exponents are equal: nothing to align.' : `. They differ by ${st.diff}, so the smaller significand moves ${st.diff} place${st.diff > 1 ? 's' : ''} right.`) +
        (st.subtract ? ' The signs differ, so the magnitudes are subtracted.' : '')
      );
    }
    case 'shift':
      return st.jump
        ? `The remaining ${st.k - (t.p + 2)} places only push zeros past the round bit, so they are done at once: ${st.k} places in all, exponent ${st.exp}. The sticky bit stays ${st.s}.`
        : `Shift ${st.k}: one place right, exponent up to ${st.exp}. ` +
            (st.k === 1 ? 'The bit that leaves the window becomes the guard bit.' : st.k === 2 ? 'The guard bit moves to the round position and a new guard bit arrives.' : 'Everything past the round bit is folded into sticky: it is 1 if any of those bits is 1.');
    case 'add':
      return `${st.subtract ? 'Subtract' : 'Add'} exactly. The exact result is ${st.exactText}${st.carry ? ', and it carried into a new leading bit' : ''}. It needs more bits than the window holds unless the part below the window is all zeros.`;
    case 'normalise':
      return st.subnormal
        ? `Normalise: the result is below the smallest normal value, so it keeps the smallest exponent, ${st.exp}, and stays denormal.`
        : st.moved > 0
          ? `Normalise: the carry makes the result 1x.xx, so it shifts right one place and the exponent goes up to ${st.exp}. One more bit leaves the window.`
          : st.moved < 0
            ? `Normalise: the subtraction cancelled the leading bit${st.moved < -1 ? 's' : ''}. Shift left ${-st.moved} place${st.moved < -1 ? 's' : ''}; the exponent drops to ${st.exp}.`
            : `Normalise: the result already has the form 1.xx × 2^${st.exp}.`;
    case 'round': {
      const half = st.reason === 'tie-even' || st.reason === 'tie-odd';
      const lead =
        st.reason === 'exact'
          ? 'Nothing was cut off: the result is exact.'
          : st.reason === 'below-half'
            ? 'Guard 0: less than half a unit was cut off, so to nearest keeps the window as it is.'
            : st.reason === 'above-half'
              ? 'Guard 1 and something after it: more than half a unit was cut off, so to nearest rounds up.'
              : half
                ? `Guard 1, round 0, sticky 0: exactly half a unit, a tie. The window's last bit is ${st.last}, so nearest-even ${st.reason === 'tie-even' ? 'keeps it (already even)' : 'rounds up to make it even'}.`
                : st.reason === 'truncate'
                  ? 'This mode points toward zero here, so the bits below the window are simply dropped.'
                  : 'This mode points away from zero here, and something was cut off, so it rounds up.';
      const err = st.reason === 'exact' ? '' : ` Error: ${ulpsText(st.errorUlps, st.errorBelowOne)} ULP.`;
      const over = st.overflow ? ` That is past the largest ${t.f.id === 'toy5' ? 'value of the format, which has no infinity, so it stays at the largest value' : 'finite value'}: the format stores ${show(st.result)}.` : '';
      return `${lead} Result: ${show(st.result, 10)}.${err}${over}`;
    }
    case 'zero-operand':
      return `One operand is zero, so nothing needs aligning: the result is ${show(st.result, 10)}.`;
    case 'exact-zero':
      return `The magnitudes cancel exactly. IEEE gives +0 here (−0 when rounding down): ${show(st.result)}.`;
    case 'special':
      return st.why === 'nan-in' ? 'A NaN goes in, a NaN comes out.' : st.why === 'inf-minus-inf' ? '∞ − ∞ is undefined: NaN.' : `An infinite operand: the result is ${show(st.result)}.`;
  }
}

function RoundLine({ st, exact }: { st: Extract<Step, { kind: 'round' }>; exact: number }) {
  const lo = Math.min(st.below, st.above);
  const hi = Math.max(st.below, st.above);
  const u = hi - lo;
  const a = lo - 0.35 * u;
  const b = hi + 0.35 * u;
  // A narrow canvas, so the labels stay readable when a phone scales it down.
  const sx = (v: number) => 20 + ((v - a) / (b - a)) * 400;
  const chosen = st.pick === 'above' ? st.above : st.below;
  return (
    <svg viewBox="0 0 440 74" style={{ width: '100%', maxWidth: 460, display: 'block' }} role="img" aria-label="The exact result between its two neighbours" data-chosen={chosen}>
      <line x1={10} x2={430} y1={36} y2={36} stroke="var(--ink-2)" />
      {[st.below, st.above].map((v) => (
        <g key={v}>
          <line x1={sx(v)} x2={sx(v)} y1={26} y2={46} stroke={v === chosen ? 'var(--ok)' : 'var(--ink)'} strokeWidth={v === chosen ? 3 : 1.4} />
          <text x={sx(v)} y={64} fontSize="14" textAnchor="middle" style={{ fill: v === chosen ? 'var(--ok)' : 'var(--ink)' }} fontWeight={v === chosen ? 700 : 400}>
            {show(v, 10)}
          </text>
        </g>
      ))}
      {Number.isFinite(exact) && exact >= a && exact <= b ? (
        <g>
          <circle cx={sx(exact)} cy={36} r={4.5} fill="var(--ink)" />
          <text x={sx(exact)} y={18} fontSize="12.5" textAnchor="middle">exact</text>
        </g>
      ) : null}
    </svg>
  );
}

function Stepper() {
  const [fmtId, setFmtId] = useState('toy5');
  const [aText, setAText] = useState('2');
  const [bText, setBText] = useState('0.25');
  const [mode, setMode] = useState<Mode>('rne');
  const [n, setN] = useState(0);
  const f = STEP_FORMATS.find((x) => x.id === fmtId)!;
  const a = parseOperand(aText);
  const b = parseOperand(bText);
  const trace = useMemo(() => (a === null || b === null ? null : alignTrace(f, a, b, mode)), [f, a, b, mode]);
  // Changing the mode keeps your place: it only changes the last step.
  useEffect(() => setN(0), [fmtId, aText, bText]);
  const total = trace?.steps.length ?? 0;
  const at = Math.min(n, total);
  const steps = trace?.steps.slice(0, at) ?? [];
  const cur = at > 0 ? trace!.steps[at - 1] : undefined;
  const compare = steps.find((s): s is Extract<Step, { kind: 'compare' }> => s.kind === 'compare');
  const shift = [...steps].reverse().find((s): s is Extract<Step, { kind: 'shift' }> => s.kind === 'shift');
  const add = steps.find((s): s is Extract<Step, { kind: 'add' }> => s.kind === 'add');
  const norm = steps.find((s): s is Extract<Step, { kind: 'normalise' }> => s.kind === 'normalise');
  const rnd = steps.find((s): s is Extract<Step, { kind: 'round' }> => s.kind === 'round');
  const firstAfterShifts = trace ? trace.steps.findIndex((s) => s.kind === 'add') + 1 : 0;

  const preset = (id: string) => {
    const p = ALIGN_PRESETS.find((x) => x.id === id)!;
    setFmtId(p.fmt.id);
    setAText(p.aText);
    setBText(p.bText);
  };

  return (
    <Bench
      title="Alignment and rounding"
      task="Step through one addition. Then change the rounding mode at the last step and watch which neighbour wins."
      foot={
        <>
          Every step comes from the exact sum of the two stored values; the result is checked against an independent
          rounding of the same sum in all four modes. Underlined: the guard and round bits.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Format">
          {STEP_FORMATS.map((x) => (
            <button key={x.id} aria-pressed={x.id === fmtId} onClick={() => setFmtId(x.id)}>{FORMAT_LABEL[x.id]}</button>
          ))}
        </span>
        <label className="field">
          a
          <input type="text" value={aText} onChange={(e) => setAText(e.target.value)} aria-label="operand a" style={{ width: '8rem' }} />
        </label>
        <label className="field">
          b
          <input type="text" value={bText} onChange={(e) => setBText(e.target.value)} aria-label="operand b" style={{ width: '8rem' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted">Presets:</span>
        {ALIGN_PRESETS.map((p) => (
          <button key={p.id} className="quiet" onClick={() => preset(p.id)} data-preset={p.id}>{p.label}</button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="muted">Rounding:</span>
        <span className="seg" role="group" aria-label="Rounding mode">
          {MODES.map((m) => (
            <button key={m} aria-pressed={m === mode} onClick={() => setMode(m)}>{MODE_LABEL[m]}</button>
          ))}
        </span>
      </div>
      {trace ? (
        <>
          <p className="muted" style={{ margin: '0 0 0.6rem' }}>
            Stored in {f.id === 'toy5' ? "the book's 5-bit format" : f.name}: a = {show(trace.a, 12)}, b = {show(trace.b, 12)}
            {trace.a !== a || trace.b !== b ? ' (your input was rounded to the nearest value the format holds)' : ''}.
          </p>
          <div className="row" style={{ marginBottom: '0.8rem' }}>
            <button onClick={() => setN(0)} disabled={at === 0}>Restart</button>
            <button onClick={() => setN(at - 1)} disabled={at === 0}>Back</button>
            <button className="primary" onClick={() => setN(at + 1)} disabled={at >= total}>Step</button>
            <button onClick={() => setN(Math.max(at, firstAfterShifts - 1))} disabled={at >= firstAfterShifts - 1}>Finish the shifts</button>
            <button onClick={() => setN(total)} disabled={at >= total}>Run to the end</button>
            <span className="muted" aria-live="polite">step {at} of {total}</span>
          </div>
          <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>
            {cur ? describe(cur, trace) : <>Nothing has happened yet. Press Step.</>}
          </div>
          {compare ? (
            // A 24-bit window is wider than a phone: the table scrolls on its own, the rest of the bench does not.
            <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
            <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.6rem' }} aria-label="The operands and the sum">
              <thead>
                <tr><th style={{ paddingLeft: '0.6rem' }}></th><th>sign</th><th>significand (the window)</th><th>bits that left the window</th><th>scale</th></tr>
              </thead>
              <tbody>
                <tr data-row="big">
                  <td style={{ paddingLeft: '0.6rem' }}>{show(compare.big.value, 10)}</td>
                  <td className="mono">{compare.big.sign ? '−' : '+'}</td>
                  <td className="mono">{dotted(compare.big.sig)}</td>
                  <td></td>
                  <td className="mono nowrap">× 2^{compare.big.exp}</td>
                </tr>
                <tr data-row="small">
                  <td style={{ paddingLeft: '0.6rem' }}>{show(compare.small.value, 10)}</td>
                  <td className="mono">{compare.small.sign ? '−' : '+'}</td>
                  <td className="mono" data-window={shift ? shift.window : compare.small.sig}>{dotted(shift ? shift.window : compare.small.sig)}</td>
                  <td>{shift ? <><Tail bits={shift.tail} /><br /><GRSChips g={shift.g} r={shift.r} s={shift.s} /></> : null}</td>
                  <td className="mono nowrap">× 2^{shift ? shift.exp : compare.small.exp}</td>
                </tr>
                {add ? (
                  <tr data-row="sum">
                    <td style={{ paddingLeft: '0.6rem' }}>exact {add.subtract ? 'difference' : 'sum'}</td>
                    <td className="mono">{compare.big.sign ? '−' : '+'}</td>
                    <td className="mono">{dotted(add.window, add.carry)}</td>
                    <td><Tail bits={add.tail} /></td>
                    <td className="mono nowrap">× 2^{compare.big.exp}</td>
                  </tr>
                ) : null}
                {norm ? (
                  <tr data-row="normalised">
                    <td style={{ paddingLeft: '0.6rem' }}>normalised</td>
                    <td className="mono">{compare.big.sign ? '−' : '+'}</td>
                    <td className="mono">{norm.subnormal ? `0.${norm.window.slice(1)}` : dotted(norm.window)}</td>
                    <td><Tail bits={norm.tail} />{norm.tail ? <><br /><GRSChips g={norm.g} r={norm.r} s={norm.s} /></> : null}</td>
                    <td className="mono nowrap">× 2^{norm.exp}</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
            </div>
          ) : null}
          {rnd && add ? (
            <div data-result={rnd.result} data-err={rnd.errorUlps} data-reason={rnd.reason}>
              <RoundLine st={rnd} exact={add.exact} />
              <p style={{ margin: '0.3rem 0 0' }}>
                <span className={'chip ' + (rnd.overflow && !Number.isFinite(rnd.result) ? 'fail' : 'ok')}>
                  {MODE_LABEL[mode]}: {show(rnd.result, 10)}
                </span>{' '}
                exact {add.exactText}; error {ulpsText(rnd.errorUlps, rnd.errorBelowOne)} ULP{rnd.carryOut ? '; rounding up carried into the exponent' : ''}.
              </p>
            </div>
          ) : null}
        </>
      ) : (
        <p className="t-fail">Type two numbers (for example 1000, -0.3, 2^-11 or 0.75×2^-23).</p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- your device (lab A.1's probe)
const LAB = 'appA/lab1_sumerror';
const yes = (v: unknown) => v === true || v === 1 || v === 'true' || v === 'yes';

function YourDevice() {
  const { reports, loading } = useReports();
  const found = Object.keys(reports).flatMap((b) => {
    const rows = [...benchRows(reports[b], LAB), ...benchRows(reports[b], LAB, 'work')];
    const probe = rows.find((r) => r.name === 'probe');
    return probe ? [{ b, device: String(reports[b].device?.name ?? b), probe: probe as Record<string, unknown> }] : [];
  });
  return (
    <Bench title="Your device" task="What lab A.1's probe found on the machines you have run it on." foot={<>The probe reads its inputs from a buffer, so the compiler cannot fold the arithmetic away at build time.</>}>
      {loading ? (
        <p className="muted">Loading results…</p>
      ) : !found.length ? (
        <p data-empty>
          No run on this machine yet. Run <code>python3 labs/pmpp.py run appA/lab1 --bench</code> and reload this page:
          the probe adds 1 and three quarters of an ULP in float, halves the smallest normal float, and computes a·b + c
          where fusing shows.
        </p>
      ) : (
        <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
          <thead>
            <tr><th style={{ paddingLeft: '0.6rem' }}>backend</th><th>device</th><th>1 + 0.75 ULP in float</th><th>FP32 denormals</th><th>a·b + c</th></tr>
          </thead>
          <tbody>
            {found.map(({ b, device, probe }) => (
              <tr key={b} data-backend={b}>
                <td style={{ paddingLeft: '0.6rem' }}>{b}</td>
                <td>{b === 'emu' || b === 'race' ? <>the CPU (emulator): the host's arithmetic, not a GPU's</> : device}</td>
                <td>{probe.rounding === 'toward-zero' ? 'rounds down to 1: toward zero' : probe.rounding === 'nearest' ? 'rounds up: not toward zero' : String(probe.rounding ?? 'not reported')}</td>
                <td>{probe.keeps_denormals === undefined ? 'not reported' : yes(probe.keeps_denormals) ? 'kept' : 'flushed to zero'}</td>
                <td>{probe.fuses_mul_add === undefined ? 'not reported' : yes(probe.fuses_mul_add) ? 'fused (one rounding)' : 'two roundings'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- FP32 to BF16 on the bits
const hex = (v: number, w: number) => '0x' + (v >>> 0).toString(16).padStart(w, '0');
const bits32 = (u: number) => (u >>> 0).toString(2).padStart(32, '0');
const groups = (b: string, cuts: number[]) => {
  const out: string[] = [];
  let i = 0;
  for (const c of cuts) {
    out.push(b.slice(i, i + c));
    i += c;
  }
  return out.join(' ');
};

function parseF32(t: string): number | null {
  const s = t.trim();
  if (/^0x[0-9a-f]{1,8}$/i.test(s)) return Number.parseInt(s.slice(2), 16) >>> 0;
  if (/^-?nan$/i.test(s)) return 0x7fc00000;
  if (/^-?(inf|infinity)$/i.test(s)) return s.startsWith('-') ? 0xff800000 : 0x7f800000;
  if (s === '' || !/^[-+]?[\d.]+(e[-+]?\d+)?$/i.test(s)) return null;
  return bitsFromF32(Number(s));
}

function Bf16Explorer() {
  const [text, setText] = useState('3.14159274');
  const u = parseF32(text);
  const c = u === null ? null : bf16FromBits(u);
  const err = (e: number) => (Number.isNaN(e) ? 'n/a' : Number.isFinite(e) ? String(Number(e.toPrecision(3))) : '∞');
  return (
    <Bench
      title="FP32 to BF16 on the bits"
      task="Try the two ties first: same rule, different outcome. Then the carry, the largest float, and the NaN."
      foot={<>BF16 is the top half of an FP32: same sign and exponent, 7 of the 23 mantissa bits.</>}
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          an FP32 value (a number, or bits as 0x…)
          <input type="text" value={text} onChange={(e) => setText(e.target.value)} aria-label="FP32 value" style={{ width: '14rem' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        {BF16_PRESETS.map((p) => (
          <button key={p.id} className="quiet" data-preset={p.id} onClick={() => setText(hex(p.bits, 8))}>{p.label}</button>
        ))}
      </div>
      {c ? (
        <div data-rounded={c.rounded} data-truncated={c.truncated} data-carry={c.carryIntoExponent ? 1 : 0} data-tie={c.tie ? 1 : 0}>
          <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.6rem' }}>
            <tbody>
              <tr>
                <td style={{ paddingLeft: '0.6rem' }}>FP32 bits</td>
                <td className="mono">
                  {groups(bits32(c.input).slice(0, 16), [1, 8, 7])} <span className="muted">| {bits32(c.input).slice(16)}</span>
                </td>
                <td className="mono">{hex(c.input, 8)} = {show(c.x, 9)}</td>
              </tr>
              <tr>
                <td style={{ paddingLeft: '0.6rem' }}>kept, cut</td>
                <td className="mono">{hex(c.kept, 4)}, {hex(c.cut, 4)}</td>
                <td>{c.nan ? 'a NaN: the rounding is skipped' : c.tie ? 'the cut half is exactly 0x8000: a tie' : c.cut > 0x8000 ? 'more than half a unit cut' : c.cut === 0 ? 'nothing cut' : 'less than half a unit cut'}</td>
              </tr>
              {!c.nan ? (
                <tr>
                  <td style={{ paddingLeft: '0.6rem' }}>add 0x7fff + lsb</td>
                  <td className="mono">{hex(c.input, 8)} + {hex(c.increment, 4)} = {hex(c.sum, 8)}</td>
                  <td>lsb (the kept half's last bit) = {c.lsb}{c.roundedUp ? '; the sum carries into the kept half' : '; no carry into the kept half'}{c.carryIntoExponent ? ', and on into the exponent' : ''}</td>
                </tr>
              ) : null}
              <tr>
                <td style={{ paddingLeft: '0.6rem' }}>rounded (llama.cpp)</td>
                <td className="mono">{groups((c.rounded >>> 0).toString(2).padStart(16, '0'), [1, 8, 7])} = {hex(c.rounded, 4)}</td>
                <td>
                  <b>{show(c.roundedValue, 9)}</b>
                  {c.nan ? ' (quiet NaN: bit 6 of the half, the top mantissa bit, forced on)' : `, error ${err(c.roundedErr)} BF16 ULP`}
                </td>
              </tr>
              <tr>
                <td style={{ paddingLeft: '0.6rem' }}>truncated</td>
                <td className="mono">{groups((c.truncated >>> 0).toString(2).padStart(16, '0'), [1, 8, 7])} = {hex(c.truncated, 4)}</td>
                <td>
                  {show(c.truncatedValue, 9)}
                  {c.nan ? (Number.isFinite(c.truncatedValue) || Number.isNaN(c.truncatedValue) ? '' : <span className="t-fail"> : the NaN became infinity</span>) : `, error ${err(c.truncatedErr)} BF16 ULP`}
                </td>
              </tr>
            </tbody>
          </table>
          {!c.nan && !Number.isFinite(c.roundedValue) && Number.isFinite(c.x) ? (
            <p className="t-fail" style={{ margin: 0 }}>Rounded to nearest, this value is past BF16's largest value plus half a unit: it becomes infinity.</p>
          ) : null}
        </div>
      ) : (
        <p className="t-fail">Type a number, or 32 bits in hex such as 0x3f808000.</p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSA4: Record<string, Question> = {
  'appA/A.4/q-tie': {
    kind: 'choice',
    id: 'appA/A.4/q-tie',
    prompt: (
      <>
        An FP16 accumulator holds 2048, and a loop keeps adding 1 under round to nearest even. What does it hold after
        the next addition, and why does it never move again?
      </>
    ),
    options: [
      {
        label: '2048. 2049 is a tie between 2048 and 2050, and nearest-even picks 2048 every time.',
        correct: true,
      },
      { label: '2050. A tie rounds up to the larger neighbour, and from there the sum keeps climbing by 2.', why: 'Ties do not round up under this mode; they go to the neighbour whose last mantissa bit is 0.' },
      { label: '2049. FP16 holds every integer up to its largest value, 65,504, so adding 1 is exact.', why: 'Above 2048 the spacing of FP16 values is 2: 11 significant bits cover 2048 to 4094 in steps of 2.' },
      { label: 'Infinity: 2049 needs more than the 11 significant bits FP16 has, so it overflows.', why: 'FP16 reaches 65,504. Nothing overflows here; the sum stops growing long before.' },
    ],
    explain: (
      <p>
        Between 2048 and 4096 FP16 values are 2 apart, so 2049 lies exactly halfway between 2048 and 2050. Nearest-even
        picks the neighbour with an even mantissa, which is 2048 (mantissa all zeros). The next addition starts from the
        same 2048 and ends the same way. Toward zero also gives 2048; rounding up gives 2050, and then 2052, and so on.
        Check it in the stepper with the preset "2048 + 1 in FP16" and all four modes.
      </p>
    ),
    source: 'book §A.4, pp. 591–592 (alignment and rounding); the tie rule is IEEE 754\'s, appA-facts A9',
  },
  'appA/A.4/q-ulp-reference': {
    kind: 'choice',
    id: 'appA/A.4/q-ulp-reference',
    prompt: (
      <>
        NVIDIA lists float division with "a maximum ULP error of zero". The book says correct rounding means at most 0.5
        ULP. Metal's specification gives float <code>x / y</code> under fast math as "&lt;= 2.5 ulp". Which of these are
        correctly rounded?
      </>
    ),
    options: [
      {
        label: "NVIDIA's and the book's: the same guarantee, measured from different references. Metal's fast-math division is not.",
        correct: true,
      },
      { label: "Only the book's: 0 ULP is impossible, so NVIDIA must mean something weaker.", why: 'NVIDIA counts from the correctly rounded result, not from the exact one. Zero ULP from the correctly rounded value is 0.5 ULP from the exact value.' },
      { label: 'All three: each is a bound on the same error, and 2.5 ulp is within the usual tolerance.', why: 'Correctly rounded means the nearest representable value, which is at most half a unit from the exact result. 2.5 ulp allows the result to be two values away.' },
      { label: 'None: GPU division is built from an approximation, so none of the three is correctly rounded.', why: 'NVIDIA guarantees it for +, −, ×, / and 1/x under its default flags, and Metal for +, − and × in every mode.' },
    ],
    explain: (
      <p>
        Before comparing two vendors' numbers, find what each counts from. NVIDIA: "the difference in ULPs between the
        value returned by the function and a correctly rounded result". Metal: "the infinitely precise result". The book:
        the exact result. Metal's float division is correctly rounded only with fast math off. AMD documents
        measured differences against the C++ library instead (<code>__fadd_rn</code>: 0 over [−1000, 1000]), and nothing
        about the plain operators.
      </p>
    ),
    source: `book §A.4, p. 592; ${FACTS} A8 (NVIDIA 5.6.6, 5.6.7), B4 (AMD) and C2 (MSL spec 8.4)`,
  },
  'appA/A.4/q-ex3': {
    kind: 'choice',
    id: 'appA/A.4/q-ex3',
    prompt: <>Exercise 3: an adder can only round toward zero (truncate), and keeps enough bits that rounding is its only error. What is its largest error?</>,
    options: [
      { label: 'Just under 1 ULP.', correct: true },
      { label: '0.5 ULP, like any other correct rounding.', why: 'That is to nearest: the farthest the exact value can be from the nearer neighbour is half the gap. Truncation always takes the neighbour toward zero, however far away it is.' },
      { label: 'Exactly 1 ULP.', why: 'If a whole unit were cut off, the exact value would be the next representable number, and truncation would return it unchanged.' },
      { label: '2 ULP.', why: 'Truncation chooses between the two neighbours of the exact value; it cannot land two values away.' },
    ],
    fixedOrder: true,
    explain: (
      <p>
        Truncation drops everything below the last kept bit. What is dropped is less than one unit of that place and can
        come as close to one unit as the extra bits allow: try 1 + 0.75 ULP, or any sum whose guard, round and sticky bits
        are all 1, in the stepper with "toward zero". Real hardware does this on demand: CUDA's <code>__fadd_rz</code>
        is this adder (it is not available on AMD without a macro).
      </p>
    ),
    source: `book §A.9 Exercise 3, p. 599; our answer (notes/appA.md); ${FACTS} A9, B4`,
  },
  'appA/A.4/q-ex5': {
    kind: 'numeric',
    id: 'appA/A.4/q-ex5',
    prompt: (
      <>
        Exercise 5: a <code>sin()</code> unit produces two more accurate mantissa bits per clock cycle and runs for 9
        cycles; the remaining mantissa bits are set to 0. The hidden leading 1 must be produced too. What is the largest
        error, in ULPs, of its single-precision result?
      </>
    ),
    answer: 63,
    unit: 'ULP',
    explain: (
      <p>
        9 cycles × 2 bits = 18 accurate bits. A single-precision significand has 24 bits, the hidden one included, so
        the lowest 6 are forced to 0. In the worst case all 6 should have been 1: 2⁶ − 1 = 63 units of the last place
        (strictly, just under 64 measured from the exact value).
      </p>
    ),
    source: 'book §A.9 Exercise 5, p. 600; our answer (notes/appA.md)',
  },
  'appA/A.4/q-fma': {
    kind: 'choice',
    id: 'appA/A.4/q-fma',
    prompt: (
      <>
        Your CUDA kernel computes <code>a*b + c</code>. Your CPU reference, built with Clang's defaults, computes{' '}
        <code>float p = a*b; float r = p + c;</code> on the same floats. The two differ in the last bit, and nothing else
        in the kernel is wrong. What is the most likely cause?
      </>
    ),
    options: [
      { label: 'nvcc fused the multiply and add into one FMA, rounding once; the reference rounds the product first.', correct: true },
      { label: 'The GPU rounds every float operation toward zero by default, and the CPU rounds to nearest even.', why: 'CUDA rounds to nearest even by default; other modes need the _rz, _ru, _rd intrinsics.' },
      { label: 'The GPU flushed a denormal product to zero, which the CPU reference keeps.', why: 'Possible only with -ftz=true or --use_fast_math, which are off by default, and only if a value is tiny.' },
      { label: 'A race condition between threads that write neighbouring outputs.', why: 'A race gives wrong or varying results, not a consistent one-bit difference in an expression with no shared data.' },
    ],
    explain: (
      <>
        <p>
          nvcc's <code>--fmad</code> is true by default, so the kernel rounds once. Your reference stores the product in{' '}
          <code>p</code> first, and Clang on the host fuses only within one statement, so it rounds twice. (HIP-Clang
          fuses across statements by default, so the same split would not stop it there.) Had the reference been the one
          statement <code>a*b + c</code>, a host with FMA instructions could fuse it too, and the two might agree. Neither
          result is wrong: the fused one is in fact the more accurate. Compare with a tolerance, or, where you need
          bit-for-bit agreement, write <code>__fadd_rn(__fmul_rn(a, b), c)</code>: those two intrinsics are never merged
          into an FMA.
        </p>
        <p>The probe of lab A.1 reports what your own compiler did.</p>
      </>
    ),
    source: `${FACTS} A9 (NVIDIA 5.6.1.5, 5.6.9.1, nvcc --fmad) and B5 (Clang -ffp-contract)`,
  },
};

// ---------------------------------------------------------------- the page
export function SA4() {
  const bf16 = useLabFile('appA/snippets/bf16_round.h');
  return (
    <>
      <p>
        Section A.2 ended with precision: how close the format's nearest value can be to a number you want to store. It
        is a property of the format, set by its mantissa bits. <b>Accuracy</b> is a property of an operation: the largest
        error an addition, a division or an exp introduces into its result. A format can be precise and an operation on
        it inaccurate. This section is about where that error comes from and how big it is allowed to be.
      </p>
      <Source section="A.4" pages="591" />

      <h2>Why a result needs rounding</h2>
      <p>
        The main source of error is simple: the exact result of an operation often has more significant bits than the
        format can hold. A product of two 24-bit significands has up to 48. An addition first has to bring the two
        operands to the same exponent. The adder shifts the smaller operand's significand right one place at a time and
        raises its exponent by one each time, until the exponents match. Bits pushed off the end of the window do not
        vanish from the exact sum, so the sum can need more bits than the window has.
      </p>
      <Source section="A.4" pages="591–592" />
      <p>
        The book's example uses its 5-bit format from section A.2: add 0.25 to 2.0. The stepper below starts there. It
        computes the exact sum and shows it the way an adder works on it, keeping three bits beyond the window: guard,
        round and sticky. Before you step through it, commit to the result.
      </p>
      <Simplification>
        <p>
          No real adder's pipeline or internal width. An adder needs only the guard, round and sticky bits to round
          correctly, which is why those three are shown. No fused multiply-add, no flush-to-zero.
        </p>
      </Simplification>
      <Predict
        id="appA/A.4/p-book-sum"
        prompt={<>2.0 + 0.25 = 2.25 exactly, but the 5-bit format has no 2.25: its values near it are 2.0 and 2.5. Which will the adder return?</>}
      >
        {(v) => (
          <p>
            {v === 2 ? 'Right, but for a reason the book does not give. ' : v === 2.5 ? 'A reasonable guess: both neighbours are equally close. ' : 'It has to be one of the two neighbours. '}
            2.25 is exactly halfway: either answer is 0.5 ULP off, and the book stops there. IEEE breaks the tie toward the
            neighbour whose last mantissa bit is 0, which is 2.0 (mantissa 00; 2.5 is 01). Step through it below.
          </p>
        )}
      </Predict>
      <Stepper />
      <p>
        In the book's example, 0.25 is a denormal in the 5-bit format: stored as 0.01 × 2⁰. The stepper writes it with its
        leading 1, as 1.00 × 2⁻², which is how the book counts its three shifts. After them, 0.25's 1 sits in the guard
        position: exactly half a unit of the last place, a tie.
      </p>
      <Aside>
        <p>
          The tie rule is ours to add. The book says only that 2.0 and 2.5 are both 0.5 ULP away. Round to nearest, ties
          to even, is IEEE's default and CUDA's (appA-facts A9).
        </p>
      </Aside>
      <Ask q={questionsSA4['appA/A.4/q-tie']} />

      <h2>ULP, and the 0.5 ULP ideal</h2>
      <p>
        The book's unit for error is the <b>ULP</b>, a unit in the last place: the place value of the last mantissa bit,
        which is also the gap between two neighbouring representable numbers. In the example the error is 0.001 × 2¹,
        half of the last place's 0.01 × 2¹: 0.5 ULP. An exact result that is not representable lies between two
        neighbours, so the best any hardware can do is the nearer one, at most half a ULP away. The book says that is
        what CUDA devices achieve.
      </p>
      <Source section="A.4" pages="592" />
      <Ask q={questionsSA4['appA/A.4/q-ulp-reference']} />

      <h2>Four rounding modes</h2>
      <p>
        The book shows one rounding. IEEE 754 defines four: to nearest with ties to even, toward zero, toward +∞ (up)
        and toward −∞ (down). Set the stepper to the FP32 preset "1 + 0.75 ULP" and run it to the end, then switch modes:
        nearest and up give 1 + 2⁻²³, toward zero and down give 1. That one addition tells two of the modes apart, which
        is what lab A.1's probe uses it for.
      </p>
      <Beyond>
        <p>Rounding modes beyond the book's single example, from appA-facts A9, B4 and C2.</p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>Backend</th><th>Default</th><th>Other modes</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA</td>
            <td>to nearest, ties to even</td>
            <td>chosen per operation by intrinsic: <code>__fadd_rz</code>, <code>__fmul_ru</code>, <code>__fdiv_rd</code> and so on (<code>_rn _rz _ru _rd</code>), each "maximum ULP error of 0"; no mode you can set for a whole kernel</td>
          </tr>
          <tr>
            <td>HIP</td>
            <td>to nearest, ties to even</td>
            <td>"Only the nearest-even rounding mode is supported by default on AMD GPUs"; the <code>_rz _ru _rd</code> intrinsics exist only if <code>OCML_BASIC_ROUNDED_OPERATIONS</code> is defined</td>
          </tr>
          <tr>
            <td>Metal</td>
            <td>"Either round ties to even or round toward zero rounding mode may be supported"</td>
            <td>none you can choose; Apple does not say which mode a given GPU uses</td>
          </tr>
        </tbody>
      </table>
      <YourDevice />

      <h2>A truncating adder</h2>
      <p>
        Exercise 3 asks about an adder that can only round toward zero. Set the stepper to "toward zero" and try the
        presets: whatever lies below the window is dropped, however close it is to a whole unit.
      </p>
      <Ask q={questionsSA4['appA/A.4/q-ex3']} />

      <h2>Units that approximate</h2>
      <p>
        Division and transcendental functions are usually built from polynomial approximations. With too few terms the
        result can be off by more than half a ULP. The book's example: an inversion whose exact result is 1.00 × 2¹
        returns 1.10 × 2¹, which is off by 0.10, twice the last place's 0.01: 2 ULP. Some early devices' hardware
        inversion was this inaccurate; the book says recent CUDA devices are IEEE-conformant.
      </p>
      <Source section="A.4" pages="592" />
      <p>
        The inaccurate division is still available today, by choice. In CUDA, <code>-prec-div=false</code>, which{' '}
        <code>--use_fast_math</code> implies, turns <code>x / y</code> into <code>__fdividef</code>, within 2 ULP, and{' '}
        <code>1/x</code> into a 1-ULP reciprocal; the default is the correctly rounded division.
      </p>
      <Today checked={CHECKED('A8, A14')}>
        <p>
          The book's "special function units" (p. 599) do not appear in NVIDIA's current text, which says intrinsics "map
          to fewer native instructions".
        </p>
      </Today>
      <p>
        Exponentials make the same trade. CUDA's <code>expf</code> is within 2 ULP and <code>__expf</code> within 2 +
        ⌊|1.173 x|⌋; Metal's <code>exp</code> is within 4 ulp and <code>fast::exp</code> within 3 + ⌊|2x|⌋. Both fast
        exponentials get worse as |x| grows, which is one more reason softmax keeps the argument of exp at or below 0
        (section <a href={href('appA', 'A.3')}>A.3</a>).
      </p>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md, A11, C4)">
        <p>
          Bounds from NVIDIA's guide (sections 5.6.7.2 and 5.6.9.2) and the Metal Shading Language specification
          (Tables 8.1 and 8.2).
        </p>
      </Today>
      <Ask q={questionsSA4['appA/A.4/q-ex5']} label="Work it out" />

      <h2>One rounding instead of two</h2>
      <p>
        A <b>fused multiply-add</b> computes <code>a*b + c</code> with a single rounding at the end, where a multiply
        followed by an add rounds twice. The book does not mention it, but every GPU compiler produces it by default, and
        it is the most common reason a correct kernel and a correct CPU reference disagree in the last bit.
      </p>
      <Beyond>
        <p>
          Defaults: nvcc <code>--fmad=true</code>; HIP-Clang <code>-ffp-contract=fast-honor-pragmas</code> (fuses across
          statements); Metal allows contraction; host Clang <code>-ffp-contract=on</code> (within one statement).{' '}
          <code>__fadd_rn</code> and <code>__fmul_rn</code> are never fused (appA-facts A9, B5).
        </p>
      </Beyond>
      <Ask q={questionsSA4['appA/A.4/q-fma']} />

      <h2>FP32 to BF16 on the bits</h2>
      <p>
        Rounding is something you will also write yourself. Converting FP32 weights or activations to BF16 is the
        everyday case: BF16 is the top 16 bits of an FP32, so truncation is one shift. llama.cpp rounds instead, on the
        integer bits: it adds <code>0x7fff</code> plus the kept half's last bit and then shifts. If the cut half is below
        0x8000 nothing carries; above it, it carries into the kept half; exactly 0x8000, a tie, carries only when the
        kept half's last bit is 1, so the result's last bit ends even. That is round to nearest even in two integer
        additions. A NaN is handled first: its top mantissa bit is forced on, so a NaN whose payload lived only in the
        cut half cannot turn into infinity.
      </p>
      <Today checked={CHECKED('D3, C2, C3')}>
        <p>
          llama.cpp (<code>ggml-impl.h</code>): "Floats shall round to nearest even, and NANs shall be quiet. Subnormals
          aren't flushed to zero". Metal's own float-to-bfloat conversion also rounds ties to even; Metal 4.1 adds{' '}
          <code>-fmetal-rtz-fp-conversion</code> to make float-to-float conversions round toward zero instead.
        </p>
      </Today>
      <Code
        tabs={[{ label: 'C++', code: bf16 ?? '// loading labs/appA/snippets/bf16_round.h' }]}
        highlight={[16]}
        caption={
          <>
            labs/appA/snippets/bf16_round.h. Line 16 is llama.cpp's rounding; the NaN test above it and the{' '}
            <code>memcpy</code> are ours. The unit tests compile and run it on the CPU and compare every result, bit for
            bit, with the explorer below.
          </>
        }
      />
      <Bf16Explorer />
      <p>
        Truncation is not only less accurate, it is biased: every value moves toward zero, by half a BF16 unit on
        average, so the error of a sum of many truncated values grows in one direction. Rounding to nearest errs both ways
        and has no such drift. That is what the extra integer addition buys.
      </p>
      <Aside>
        <p>
          A fifth way to round: AMD's conversion APIs for gfx950 offer <b>stochastic rounding</b>; such an API "will
          expect an input as seed" (appA-facts B2). In the usual meaning, which is ours to state, it rounds up or down at
          random, more often toward the nearer neighbour, so the errors cancel on average. The book does not cover it.
        </p>
      </Aside>

      <h2>The disappearing addend</h2>
      <p>
        One addition loses at most half a ULP. A loop of them can lose everything it adds. Watch an FP16 accumulator
        receive 0.3 again and again, starting from 1000.
      </p>
      <DisappearingAddendScene />
      <p>
        The order of additions and the width of the accumulator are the subject of section{' '}
        <a href={href('appA', 'A.6')}>A.6</a>, and lab A.1 measures both on your GPU (<a href={href('appA', 'labs')}>labs</a>).
      </p>
      <Simplification>
        <p>
          The scene's model: 0.3 is first stored as the nearest FP16 value, and every addition rounds once to nearest
          even, with no fused multiply-add and no compiler in between. The FP32 line uses the same model.
        </p>
      </Simplification>
      <Today checked={CHECKED('D1, D2')}>
        <p>
          This is why frameworks choose the accumulator. PyTorch accumulates half and bfloat16 in float on every device.
          MLX's default <code>softmax</code> on half keeps its sum in half; <code>precise=True</code> accumulates in
          float32 instead, because otherwise it "can lose precision over long reduction axes".
        </p>
      </Today>
    </>
  );
}
