import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import { Q4_0, Q4_1, Q4_K, bitsPerWeight, mlxAffine } from '../../sim/qformats';
import {
  DIST_LABEL,
  GRANS,
  MIRROR_AT,
  OUTLIER_AT,
  RULES,
  RULE_FACT,
  RULE_LABEL,
  SCHEME_LABEL,
  SWEEP_BITS,
  bestUnder,
  demoBlock,
  errorSweep,
  fp16,
  granLabel,
  quantTrace,
  scaleMinSources,
  type Dist,
  type Gran,
  type OutlierSign,
  type Rule,
  type Scheme,
  type SweepPoint,
  type Trace,
} from '../../sim/quant';

const FACTS = 'notes/today/x2-facts.md';
const CHECKED = (items: string) => `2026-11-20 in the facts pass (${FACTS}, ${items})`;
const SNIPPET = 'x2/snippets/q4_0_ref.h';
const SNIPPET_STATUS = 'host C++; compiled and compared bit for bit with the stepper’s model by a unit test; not a GPU kernel';
const n3 = (v: number) => (Object.is(v, -0) ? '-0' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));
const n4 = (v: number) => (Object.is(v, -0) ? '-0' : v.toPrecision(4));

// ---------------------------------------------------------------- questions
export const questionsSX22: Record<string, Question> = {
  'x2/X2.2/q-q40-clip': {
    kind: 'numeric',
    id: 'x2/X2.2/q-q40-clip',
    prompt: (
      <>
        A Q4_0 block's value of largest magnitude is <code>-0.8</code>. Another weight in the same block is{' '}
        <code>+0.8</code>. What does it dequantise to? (Two decimals are enough.)
      </>
    ),
    answer: 0.7,
    tol: 0.01,
    explain: (
      <p>
        d = -0.8 / -8 = 0.1, so the extreme becomes code 0 (value -8 × 0.1). For +0.8: 0.8 × 10 + 8.5 = 16.5, truncated
        to 16, clamped to 15. (15 - 8) × 0.1 = 0.7, or 0.6998 once d has been stored as FP16 (0.0999756). The block's
        16 levels run from -8d to +7d: the side opposite the extreme loses one step.
      </p>
    ),
    source: `${FACTS} D1 (quantize_row_q4_0_ref)`,
  },
  'x2/X2.2/q-q40-zero': {
    kind: 'choice',
    id: 'x2/X2.2/q-q40-zero',
    prompt: <>A block of 32 zeros goes through llama.cpp's Q4_0 reference rule. What happens?</>,
    options: [
      { label: 'd is zero, the inverse is guarded to 0, every code is 8, every weight dequantises to 0.', correct: true },
      { label: 'Division by zero: d is NaN and the block decodes to NaN.', why: 'The code writes id = d ? 1.0f/d : 0.0f, so it never divides by a zero d.' },
      { label: 'Every code is 0, the lowest level.', why: 'x × 0 + 8.5 truncates to 8, the middle code, which is (8 - 8) × d = 0.' },
      { label: 'The rule skips empty blocks and writes no bytes.', why: 'Every block is 18 bytes, whatever its contents.' },
    ],
    explain: (
      <p>
        amax and max start at 0 and stay there, so d = 0 / -8, which is minus zero in IEEE arithmetic (stored as the
        FP16 pattern 0x8000). The guard makes the inverse 0, each weight becomes 0 + 8.5, truncated to 8, and (8 - 8)
        × d is zero. Nothing breaks, which is why llama.cpp can write the guard instead of a special case.
      </p>
    ),
    source: `${FACTS} D1; the sign of the zero is our reading of IEEE division`,
  },
  'x2/X2.2/q-bits': {
    kind: 'numeric',
    id: 'x2/X2.2/q-bits',
    prompt: (
      <>
        MLX's default affine quantisation: 4 bits, groups of 64, with a scale and a bias per group stored in FP16. How
        many bits per weight is that?
      </>
    ),
    answer: bitsPerWeight(mlxAffine(4, 64, 'f16')),
    tol: 0.001,
    unit: 'bits',
    explain: (
      <p>
        4 + (16 + 16) / 64 = 4.5. The scale and bias have the input's type, so float32 weights give float32 scales and
        biases: 4 + 64 / 64 = 5.0, as many bits as group 32 in FP16.
      </p>
    ),
    source: `${FACTS} D5, E1`,
  },
  'x2/X2.2/q-convert': {
    kind: 'choice',
    id: 'x2/X2.2/q-convert',
    prompt: (
      <>
        You have one MLX group, value = s·q + bias, and want MPSGraph's form, value = scale·(q - zeroPoint), where the
        zero point must have the input's integer type. What are scale and zeroPoint, and when does it work?
      </>
    ),
    options: [
      {
        label: 'scale = s, zeroPoint = -bias / s; it works when that is an integer, which MLX’s coded rule (bias = q0·s at the edge) arranges and the documented min/max rule generally does not.',
        correct: true,
      },
      { label: 'scale = s, zeroPoint = bias; always works.', why: 'Expand scale·(q - zp) = s·q - s·zp: you need -s·zp = bias, so zp = -bias / s, not bias.' },
      { label: 'scale = 1 / s, zeroPoint = bias·s.', why: 'The scale multiplies the codes in both forms; it is the same s.' },
      { label: 'It always works, because zeroPoint may be any float.', why: 'The header says the zero-point tensor must match the input tensor’s type: Int4, UInt4, Int8 or UInt8.' },
    ],
    explain: (
      <p>
        s·q + bias = s·(q - (-bias / s)). With the documented rule bias = min and s = (max - min) / 15, so -min / s is
        whatever it happens to be. MLX's code instead picks the edge of larger magnitude, rounds edge / s to an integer
        q0, and sets s = edge / q0, bias = edge; then -bias / s = -q0, an integer. The stepper shows that code (it
        marks "code on zero").
      </p>
    ),
    source: `${FACTS} C4, D5 (our algebra)`,
  },
  'x2/X2.2/q-outlier': {
    kind: 'choice',
    id: 'x2/X2.2/q-outlier',
    prompt: (
      <>
        A weight row of 4096 has one value 12 standard deviations out. At 4 bits, how does the error of the other 4095
        weights compare between one scale per row and one scale per 32?
      </>
    ),
    options: [
      { label: 'Per row, every weight’s step grows with the outlier; per 32, only its own 31 neighbours pay.', correct: true },
      { label: 'The same either way: only the outlier itself is badly rounded.', why: 'The outlier sets the scale, and the scale is every weight’s step.' },
      { label: 'Per 32 is worse, because more scales add more rounding.', why: 'Each FP16 scale adds a relative error near 2^-11, small next to a 4-bit step.' },
    ],
    explain: (
      <p>
        With absmax-style scaling the step is the extreme divided by the number of levels on its side. The largest of
        32 typical weights is usually 2 to 3σ, so one weight at 12σ makes the step four to six times larger, and every
        other weight's rounding error grows with it. Confine the scale to 32 weights and 31 weights pay instead of 4095. Move
        the outlier slider in the stepper, then compare the curves for "block 32" and "one per row" in the explorer
        with heavy tails.
      </p>
    ),
    source: `ours; ${FACTS} E2 for the measured trend`,
  },
  'x2/X2.2/q-q4k': {
    kind: 'numeric',
    id: 'x2/X2.2/q-q4k',
    prompt: (
      <>
        llama.cpp's Q4_K stores 256 weights in 144 bytes, with a scale and a min for every 32 weights. How many bits per
        weight?
      </>
    ),
    answer: bitsPerWeight(Q4_K),
    tol: 0.001,
    unit: 'bits',
    explain: (
      <p>
        144 × 8 / 256 = 4.5. Q4_1 has the same per-32 scale and min and costs 5.0, because each is an FP16 number. Q4_K
        stores the eight scales and eight mins as 6-bit codes (12 bytes in all) under two FP16 super-block numbers, d
        and dmin: 2 + 2 + 12 + 128 bytes.
      </p>
    ),
    source: `${FACTS} D2, E1`,
  },
};

// ---------------------------------------------------------------- stepper: one block, six rules
const STEPS = ['the weights', 'find the extreme', 'derive the scale', 'divide', 'round', 'clamp', 'store'] as const;

function stepText(t: Trace, k: number, fp16Scale: boolean): string {
  const e = t.extreme;
  const sym = t.rule === 'q8_0' || t.rule === 'sym4';
  const qmax = t.rule === 'q8_0' ? 127 : 7;
  const clips = t.clipped.flatMap((c, i) => (c ? [i] : []));
  switch (k) {
    case 0:
      return `32 float32 weights of one block. Weight ${OUTLIER_AT} is the outlier you control.`;
    case 1:
      if (t.rule === 'q4_0') return `Q4_0 looks for the largest magnitude and keeps its sign: weight ${e.index}, ${n3(e.value)}.`;
      if (sym) return `The largest magnitude, sign dropped: |${n3(e.value)}| at weight ${e.index}.`;
      return `Both ends: min ${n3(e.min)} at weight ${e.minIndex}, max ${n3(e.max)} at weight ${e.maxIndex}.`;
    case 2:
      if (t.rule === 'q4_0') return `d = ${n3(e.value)} / -8 = ${n4(t.scale)}. The extreme will sit exactly on code 0 (value -8d); the codes run 0 to 15 for -8d to +7d.`;
      if (sym) return `d = ${n3(Math.abs(e.value))} / ${qmax} = ${n4(t.scale)}. Codes run -${qmax} to +${qmax}: ${2 * qmax + 1} levels, symmetric around zero.`;
      if (t.rule === 'mlx-code')
        return `s from (max - min) / 15, signed toward the edge of larger magnitude (${n3(t.offset)}), then nudged to edge / q0 = ${n4(t.scale)}; bias = ${n3(t.offset)}.${t.zeroCode != null ? ` Code ${t.zeroCode} now lands on zero.` : ''}`;
      return `${t.rule === 'q4_1' ? 'd' : 's'} = (max - min) / 15 = ${n4(t.scale)}, offset = min = ${n3(t.offset)}. Sixteen levels from min to max, none wasted outside them.`;
    case 3:
      return t.rule === 'q4_0' || sym
        ? 'Each weight times 1 / d.'
        : t.rule === 'q4_1'
          ? 'Each weight minus m, times 1 / d.'
          : 'Each weight minus the bias, divided by s.';
    case 4:
      return t.rule === 'q4_0'
        ? 'Add 8.5 and truncate (the (int8_t) cast): round half up, and shifted so the codes start at 0.'
        : t.rule === 'q4_1'
          ? 'Add 0.5 and truncate.'
          : 'Round to the nearest integer, halves away from zero.';
    case 5:
      return clips.length
        ? `Clamp to ${t.codeMin}..${t.codeMax}: weight${clips.length > 1 ? 's' : ''} ${clips.join(', ')} clipped (shown in red).`
        : `Clamp to ${t.codeMin}..${t.codeMax}: nothing needed clamping.`;
    default:
      return `Store the codes and ${fp16Scale ? 'the scale rounded to FP16' : 'the scale as a float (FP16 rounding off)'}${t.offset !== 0 || t.rule === 'q4_1' ? ' and offset' : ''}: ${n4(t.scaleStored)}${t.rule === 'q4_0' || sym ? '' : `, ${n4(t.offsetStored)}`}. Relative RMS error of this block: ${(100 * t.rel).toFixed(2)}%.`;
  }
}

function NumberLine({ t, k }: { t: Trace; k: number }) {
  const W = 720;
  const H = 150;
  const R = Math.max(...t.values.map(Math.abs), ...(k >= 2 ? t.levels.map(Math.abs) : [0])) * 1.08 || 1;
  const X = (v: number) => 20 + ((v + R) / (2 * R)) * (W - 40);
  const showLevels = k >= 2 && t.levels.length <= 32;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Weights, the grid and the dequantised values on a number line" style={{ width: '100%', display: 'block' }}>
      <line x1={20} x2={W - 20} y1={70} y2={70} stroke="var(--rule)" />
      <line x1={X(0)} x2={X(0)} y1={60} y2={80} stroke="var(--ink-3)" />
      <text x={X(0)} y={95} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>0</text>
      {showLevels
        ? t.levels.map((v, i) => (
            <line key={i} data-level={v} x1={X(v)} x2={X(v)} y1={62} y2={78} stroke="var(--global)" strokeWidth={1.4} />
          ))
        : k >= 2 ? <text x={20} y={56} fontSize="10" style={{ fill: 'var(--global)' }}>{t.levels.length} levels, too dense to draw</text> : null}
      {t.values.map((v, i) => {
        const hot = k >= 1 && (i === t.extreme.index || ((t.rule === 'q4_1' || t.rule.startsWith('mlx')) && (i === t.extreme.minIndex || i === t.extreme.maxIndex)));
        const bad = k >= 5 && t.clipped[i];
        return (
          <g key={i}>
            <circle cx={X(v)} cy={40} r={hot ? 5 : 3.4} fill={bad ? 'var(--fail)' : hot ? 'var(--ink)' : 'var(--page)'} stroke={bad ? 'var(--fail)' : 'var(--ink)'} />
            {k >= 6 ? (
              <>
                <line x1={X(v)} y1={44} x2={X(t.dequant[i])} y2={100} stroke={bad ? 'var(--fail)' : 'var(--ink-3)'} strokeWidth={bad ? 1.6 : 0.8} />
                <circle cx={X(t.dequant[i])} cy={104} r={3.4} fill={bad ? 'var(--fail)' : 'var(--global)'} />
              </>
            ) : null}
          </g>
        );
      })}
      <text x={W - 20} y={14} fontSize="11" textAnchor="end">weights (float32)</text>
      {k >= 2 ? <text x={20} y={130} fontSize="11" style={{ fill: 'var(--global)' }}>{k >= 6 ? 'dequantised values, on the grid' : 'the grid: every value the block can hold'}</text> : null}
      {k >= 1 ? <text x={X(t.extreme.value)} y={27} fontSize="10" textAnchor="middle">{t.rule === 'q4_0' ? 'signed extreme' : 'extreme'}</text> : null}
    </svg>
  );
}

function OneBlock() {
  const [dist, setDist] = useState<Dist>('normal');
  const [outlier, setOutlier] = useState(0.8);
  const [sign, setSign] = useState<OutlierSign>('±');
  const [fp16Scale, setFp16] = useState(true);
  const [rule, setRule] = useState<Rule>('q4_0');
  const [k, setK] = useState(0);
  const values = useMemo(() => demoBlock(dist, outlier, sign), [dist, outlier, sign]);
  const t = useMemo(() => quantTrace(rule, values, fp16Scale), [rule, values, fp16Scale]);
  const last = STEPS.length - 1;
  const codeShown = (i: number) => (rule === 'q4_0' ? `${t.codes[i]} (${t.codes[i] - 8})` : String(t.codes[i]));
  return (
    <Bench
      title="One block, six rules"
      task="Step Q4_0 to the end with the outlier mirrored, then try the other rules on the same block."
      foot={
        <>
          32 seeded weights, standard deviation 0.25. Weight {OUTLIER_AT} is the outlier; with ± its mirror image is at
          weight {MIRROR_AT}, the other half of the same Q4_0 byte. Red: a weight the clamp changed. Codes for Q4_0 are
          shown as stored, with code - 8 in brackets. Rules: {RULES.map((r) => `${RULE_LABEL[r]} (${RULE_FACT[r]})`).join('; ')}.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Rule
          <select aria-label="rule" value={rule} onChange={(e) => setRule(e.target.value as Rule)}>
            {RULES.map((r) => <option key={r} value={r}>{RULE_LABEL[r]}</option>)}
          </select>
        </label>
        <label className="field">
          Weights
          <span className="seg" role="group" aria-label="distribution">
            {(['normal', 't3'] as Dist[]).map((d) => (
              <button key={d} aria-pressed={dist === d} onClick={() => setDist(d)}>{d === 'normal' ? 'normal' : 'Student t (3)'}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Outlier at weight {OUTLIER_AT}: {outlier.toFixed(2)}
          <input type="range" aria-label="outlier magnitude" min="0" max="3" step="0.05" value={outlier} onChange={(e) => setOutlier(+e.target.value)} />
        </label>
        <label className="field">
          Sign
          <span className="seg" role="group" aria-label="outlier sign">
            {(['-', '+', '±'] as OutlierSign[]).map((s) => (
              <button key={s} aria-pressed={sign === s} onClick={() => setSign(s)}>{s === '±' ? '− and its mirror' : s === '-' ? 'negative' : 'positive'}</button>
            ))}
          </span>
        </label>
        <label className="check"><input type="checkbox" checked={fp16Scale} onChange={(e) => setFp16(e.target.checked)} /> store the scale as FP16</label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setK(0)} disabled={k === 0}>Restart</button>
        <button onClick={() => setK(k - 1)} disabled={k === 0}>Back</button>
        <button className="primary" onClick={() => setK(k + 1)} disabled={k >= last}>Step</button>
        <button onClick={() => setK(last)} disabled={k >= last}>To the end</button>
        <span className="muted" aria-live="polite">step {k} of {last}: {STEPS[k]}</span>
      </div>
      <div className={'narrate' + (k >= 5 && t.clipped.some(Boolean) ? ' bad' : '')} role="status" data-step={k} style={{ marginBottom: '0.8rem' }}>
        {stepText(t, k, fp16Scale)}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <div style={{ minWidth: 560 }}><NumberLine t={t} k={k} /></div>
      </div>
      <div data-part="cells" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(5.4rem, 1fr))', gap: '4px', marginTop: '0.6rem' }}>
        {t.values.map((v, i) => (
          <div key={i} data-w={i} data-clipped={k >= 5 && t.clipped[i] ? 'yes' : 'no'} className="panel"
            style={{ padding: '0.25rem 0.4rem', fontVariantNumeric: 'tabular-nums', lineHeight: 1.3, borderColor: k >= 5 && t.clipped[i] ? 'var(--fail)' : undefined }}>
            <div className="muted">w{i}{k >= 1 && i === t.extreme.index ? ' ★' : ''}</div>
            <div>{n3(v)}</div>
            {k >= 3 ? <div className="muted">÷ {n3(t.divided[i])}</div> : null}
            {k >= 5 ? <div className={t.clipped[i] ? 't-fail' : 't-global'}>q {codeShown(i)}</div> : k >= 4 ? <div className="t-global">→ {t.rounded[i]}</div> : null}
            {k >= 6 ? <div data-deq={t.dequant[i]}>{n3(t.dequant[i])}</div> : null}
            {k >= 6 ? <div className={t.clipped[i] ? 't-fail' : 'muted'}>err {n3(t.err[i])}</div> : null}
          </div>
        ))}
      </div>
      {k >= last ? (
        <p style={{ margin: '0.8rem 0 0' }} data-rel={t.rel}>
          <span className={'chip ' + (t.clipped.some(Boolean) ? 'fail' : 'ok')}>
            {t.clipped.filter(Boolean).length} clipped
          </span>{' '}
          relative RMS error {(100 * t.rel).toFixed(2)}%; {t.levels.length} levels
          {t.zeroCode != null && t.zeroValue != null ? <>; code {t.zeroCode} on zero ({t.zeroValue.toExponential(1)} after storage)</> : null}.
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: error against block size and bit width
const DASH: Record<string, string> = { '16': '', '32': '6 3', '64': '2 3', '128': '8 3 2 3', '256': '1 5', row: '10 4', tensor: '3 6' };
const gkey = (g: Gran) => String(g);

function SweepPlot({ pts }: { pts: SweepPoint[] }) {
  const W = 720;
  const H = 330;
  const pad = { l: 52, r: 14, t: 12, b: 38 };
  const xs = pts.map((p) => p.bpw);
  const x0 = 2;
  const x1 = Math.max(...xs);
  const lo = Math.min(...pts.map((p) => p.rel)) / 1.3;
  const hi = Math.min(2, Math.max(...pts.map((p) => p.rel)) * 1.3);
  const X = (v: number) => pad.l + ((v - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(Math.min(v, hi)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (H - pad.t - pad.b);
  const ticks: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= 0; e++) for (const m of [1, 2, 5]) if (m * 10 ** e >= lo && m * 10 ** e <= hi) ticks.push(m * 10 ** e);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Relative error against bits per weight, one curve per block size" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {ticks.map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 4} y={Y(v) + 3.5} fontSize="10" textAnchor="end">{v >= 0.01 ? `${+(v * 100).toFixed(1)}%` : `${+(v * 100).toPrecision(1)}%`}</text>
        </g>
      ))}
      {Array.from({ length: Math.floor(x1) - 1 }, (_, i) => i + 2).map((b) => (
        <g key={b}>
          <line x1={X(b)} x2={X(b)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(b)} y={H - pad.b + 13} fontSize="10" textAnchor="middle">{b}</text>
        </g>
      ))}
      <text x={(W - pad.r + pad.l) / 2} y={H - 6} fontSize="11" textAnchor="middle">bits per weight, scales included</text>
      <text x={12} y={(H - pad.b) / 2} fontSize="11" textAnchor="middle" transform={`rotate(-90 12 ${(H - pad.b) / 2})`}>relative RMS error (log)</text>
      {GRANS.map((g) => {
        const line = pts.filter((p) => p.gran === g).sort((a, b) => a.bits - b.bits);
        return (
          <g key={gkey(g)} data-curve={gkey(g)}>
            <polyline points={line.map((p) => `${X(p.bpw).toFixed(1)},${Y(p.rel).toFixed(1)}`).join(' ')} fill="none" stroke="var(--ink)" strokeWidth={1.6} strokeDasharray={DASH[gkey(g)]} />
            {line.map((p) => <circle key={p.bits} cx={X(p.bpw)} cy={Y(p.rel)} r={2.4} fill="var(--ink)" />)}
          </g>
        );
      })}
      <g transform={`translate(${W - pad.r - 128}, ${pad.t + 8})`}>
        <rect x={-6} y={-6} width={130} height={GRANS.length * 15 + 6} fill="var(--page)" stroke="var(--rule)" />
        {GRANS.map((g, i) => (
          <g key={gkey(g)} transform={`translate(0, ${i * 15})`}>
            <line x1={0} x2={34} y1={5} y2={5} stroke="var(--ink)" strokeWidth={1.6} strokeDasharray={DASH[gkey(g)]} />
            <text x={40} y={9} fontSize="10">{granLabel(g)}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

const BUDGETS = [4.5, 5, 5.5, 6, 8.5];

function Sweep() {
  const [dist, setDist] = useState<Dist>('normal');
  const [scheme, setScheme] = useState<Scheme>('signed-extreme');
  const [budget, setBudget] = useState(5);
  const pts = useMemo(() => errorSweep({ dist, scheme }), [dist, scheme]);
  const best = bestUnder(pts, budget);
  const at = (g: Gran, b: number) => pts.find((p) => p.gran === g && p.bits === b)!;
  return (
    <Bench
      title="Error against block size and bit width"
      task="Switch to heavy tails and watch which curves move. Then pick a budget and see where it is best spent."
      foot={
        <>
          One seeded 16 × 4096 matrix per distribution, quantised along each row with one FP16 scale (and, for affine,
          one FP16 bias) per block. Bits per weight count those: 16 or 32 bits per block. {SCHEME_LABEL[scheme]}.
          Arithmetic in double precision apart from the FP16 scales: an error model, not a kernel.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Weights
          <span className="seg" role="group" aria-label="sweep distribution">
            {(['normal', 't3'] as Dist[]).map((d) => (
              <button key={d} aria-pressed={dist === d} onClick={() => setDist(d)}>{d === 'normal' ? 'normal' : 'Student t (3)'}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Scheme
          <select aria-label="scheme" value={scheme} onChange={(e) => setScheme(e.target.value as Scheme)}>
            {(Object.keys(SCHEME_LABEL) as Scheme[]).map((s) => <option key={s} value={s}>{SCHEME_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="field">
          Budget, bits per weight
          <span className="seg" role="group" aria-label="budget">
            {BUDGETS.map((b) => <button key={b} aria-pressed={budget === b} onClick={() => setBudget(b)}>{b}</button>)}
          </span>
        </label>
      </div>
      <div>
        <SweepPlot pts={pts} />
        <div style={{ marginTop: '0.8rem' }}>
          <p data-part="best" data-bits={best?.bits} data-gran={best ? gkey(best.gran) : ''} style={{ marginTop: 0 }}>
            {best ? (
              <>
                Within {budget} bits per weight, the lowest error on {DIST_LABEL[dist]} weights is{' '}
                <b>{best.bits} bits, {granLabel(best.gran)}</b>: {(100 * best.rel).toFixed(2)}% at {best.bpw.toFixed(3)} bits.
              </>
            ) : (
              <>Nothing fits {budget} bits per weight with this scheme.</>
            )}
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table className="data" data-part="sweep" style={{ margin: 0 }}>
              <thead>
                <tr><th>one scale per</th>{SWEEP_BITS.map((b) => <th key={b} className="num" style={{ whiteSpace: 'nowrap' }}>{b} bits</th>)}</tr>
              </thead>
              <tbody>
                {GRANS.map((g) => (
                  <tr key={gkey(g)}>
                    <td style={{ whiteSpace: 'nowrap' }}>{granLabel(g)}</td>
                    {SWEEP_BITS.map((b) => (
                      <td key={b} className="num" data-cell={`${gkey(g)}:${b}`} data-rel={at(g, b).rel}>{(100 * at(g, b).rel).toFixed(1)}%</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- predict: where to spend a bit
const SPEND_ID = 'x2/X2.2/p-spend';
const SPEND = ['halve the group to 32 (5.0 bits per weight)', 'keep group 64, use 5 bits (5.5 bits per weight)'];
function SpendPredict() {
  const { progress, answer } = useProgress();
  const prior = progress.answers[SPEND_ID]?.value;
  const committed = typeof prior === 'number' ? prior : null;
  const pts = useMemo(() => errorSweep({ dist: 't3', scheme: 'affine', bits: [4, 5], grans: [32, 64] }), []);
  const get = (b: number, g: number) => pts.find((p) => p.bits === b && p.gran === g)!;
  const base = get(4, 64);
  const opts = [get(4, 32), get(5, 64)];
  const lower = opts[0].rel < opts[1].rel ? 0 : 1;
  const perBit = opts.map((o) => (base.rel - o.rel) / (o.bpw - base.bpw));
  return (
    <div className="q" data-part="spend">
      <div className="kind">Predict first</div>
      <div className="prompt">
        Heavy-tailed (Student t) weights, MLX-style affine, 4 bits, group 64, FP16 scale and bias: 4.5 bits per weight.
        You may spend more. Which lowers the error more?
      </div>
      {committed == null ? (
        <div className="row">
          {SPEND.map((s, i) => (
            <button key={i} onClick={() => answer(SPEND_ID, i)}>{s}</button>
          ))}
        </div>
      ) : (
        <div className="verdict info" data-lower={lower}>
          <p><b>You chose: {SPEND[committed]}.</b></p>
          <p>
            From the model: {(100 * base.rel).toFixed(2)}% now; group 32 gives {(100 * opts[0].rel).toFixed(2)}%; 5 bits at
            group 64 gives {(100 * opts[1].rel).toFixed(2)}%. The {lower === 0 ? 'smaller group' : 'extra bit'} lowers the error
            more{committed === lower ? ', as you predicted' : ''}. Per bit spent: {(100 * perBit[0]).toFixed(2)} points for the
            group, {(100 * perBit[1]).toFixed(2)} for the bit.
          </p>
          <p className="muted">
            Both are real options in MLX (groups 32, 64, 128; bits 1 to 6 and 8). The bit halves every step; the smaller
            group only helps the weights whose block holds an outlier, and on these tails that is a minority.
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the page
export function SX22() {
  const q = questionsSX22;
  const src = useLabFile(SNIPPET);
  const d32 = 0.1;
  return (
    <>
      <p>
        Section X2.1 showed that fewer bytes per weight make decode faster. This one is about what you give up for them.
        A 4-bit code can hold 16 values; a real weight can hold any float. The choices that decide how close you get are
        few: one scale or a scale and an offset, how many weights share them, and how the scale is chosen and stored.
        You will meet each one in the rules llama.cpp and MLX actually run.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          The book stops at floating-point formats. Integer quantisation with scales and zero points is not in it; every
          rule here is checked against the projects' source in the facts pass.
        </p>
      </Beyond>
      <Source section="A.5" pages="592–593" />

      <h2>The map</h2>
      <p>
        Every scheme here stores an integer code q per weight and recovers w ≈ s·(q - z): a scale s, and a zero point z
        that says which code means zero. <b>Symmetric</b> schemes fix z (so the grid is centred on zero) and store only s;{' '}
        <b>asymmetric</b> or <b>affine</b> ones store an offset too, so the 16 levels can sit wherever the weights are.
        The second axis is granularity: one scale per tensor, per row, or per <b>block</b> (MLX says group) of g
        consecutive weights along a row. In PyTorch terms, a block scale is a tensor of shape (out, in / g) beside the
        codes.
      </p>

      <h2>Q8_0: absmax over 127</h2>
      <p>
        llama.cpp's 8-bit format takes the block of 32, finds the largest magnitude, and sets d = amax / 127. Each weight
        becomes <code>roundf(x / d)</code>, a code from -127 to 127, and d is stored as FP16 beside the 32 bytes: 34 bytes
        per block. Nothing ever clips, and with 255 levels the error is tiny. Code -128 is never used.
      </p>
      <Today checked={CHECKED('D1')}>
        <p>
          The rules on this page are llama.cpp's <code>_ref</code> quantisers, the ones it calls the "reference
          implementation for deterministic creation of model files". Given an importance matrix it chooses d
          differently (<code>make_qx_quants</code>); the layout is the same.
        </p>
      </Today>

      <h2>Q4_0: the signed extreme over -8</h2>
      <p>
        The 4-bit format is subtler. It finds the value of largest magnitude <i>with its sign</i>, call it max, and sets
        d = max / -8. Then each weight becomes <code>MIN(15, (int8_t)(x/d + 8.5))</code> and dequantises as (q - 8)·d.
        So the extreme lands exactly on code 0, the most negative level, and the codes cover -8d to +7d. A weight on the
        other side with the same magnitude would need +8 and gets +7: it clips by one step. That one step is what buys
        the sixteenth level. People call Q4_0 symmetric; its scale is, its code range is not.
      </p>
      <p>
        Adding 8.5 and truncating is round half up, shifted so the codes start at zero. Run it in the stepper below with
        the default block, whose weight 5 is -0.8 and whose weight 21 is its mirror, +0.8: d = {d32}, and weight 21 comes
        back as 0.7.
      </p>
      {src ? (
        <Code
          tabs={[{ label: 'q4_0_ref.h', code: src, status: SNIPPET_STATUS, highlight: [49, 50, 53, 54, 55, 58, 59, 60, 64, 66] }]}
          caption={
            <>
              A CPU reference for one block, written from llama.cpp's <code>quantize_row_q4_0_ref</code>, loaded from{' '}
              <code>labs/x2/snippets/q4_0_ref.h</code>. It is the shape a lab's CPU reference takes. Lines that decide the
              rule are marked.
            </>
          }
        />
      ) : null}
      <Ask q={q['x2/X2.2/q-q40-clip']} label="Work it out" />
      <Ask q={q['x2/X2.2/q-q40-zero']} />

      <h2>Q4_1 and MLX: a scale and an offset</h2>
      <p>
        Q4_1 is llama.cpp's asymmetric twin: d = (max - min) / 15, m = min, value = q·d + m, both stored as FP16, 20 bytes
        per block. No level falls outside the weights' range, so when a block is lopsided (all its large values on one
        side) none is wasted.
      </p>
      <p>
        MLX's <code>affine</code> mode documents the same idea: s = (max - min) / (2^b - 1), bias = min, codes from 0 to
        2^b - 1, with b from 1 to 6 or 8 and groups of 32, 64 or 128. Its code does something slightly different. It takes
        the edge of larger magnitude as the bias (so the scale is negative when that edge is the max), rounds edge / s to
        an integer q0, and resets s = edge / q0. The effect: code -q0 dequantises to zero, up to the rounding of one
        division. A weight that was exactly 0 stays 0, and the far edge may clip by a fraction of a step. Pick both MLX
        rules in the stepper and compare the step "derive the scale".
      </p>
      <Today checked={CHECKED('D5')}>
        <p>
          <code>mx.quantize(w, group_size=64, bits=4)</code> returns <code>(w_q, scales, biases)</code>: codes packed into
          uint32 words, and one scale and one bias per group in the input's type. <code>mlx.nn.QuantizedLinear</code>{' '}
          uses the same defaults. MLX's docs give the min/max formula; the kernel computes the refinement above.
        </p>
      </Today>
      <Ask q={q['x2/X2.2/q-bits']} label="Work it out" />

      <h2>One block, six rules</h2>
      <p>
        Step Q4_0 through the default block and find the clipped weight. Then change only the rule. Before you press
        Step on each, predict: does anything clip, and where is the level nearest zero? Then drag the outlier to 2.5 and
        watch what happens to every other weight's error under the symmetric rules.
      </p>
      <Simplification>
        <p>
          Synthetic, independent weights: a normal distribution, or a Student t with 3 degrees of freedom for heavy
          tails, scaled to the same variance. Real weight matrices have structure (outlier channels, for one) that this
          leaves out. Error is measured on the weights, not on a model's output; section X2.7 does that.
        </p>
      </Simplification>
      <OneBlock />

      <h2>Four conventions for one idea</h2>
      <p>
        Every asymmetric format stores the same two numbers per group, but each project writes the formula its own way.
        When you move weights between them, this is the table you need.
      </p>
      <div style={{ overflowX: 'auto' }} className="wide">
        <table className="data" style={{ maxWidth: '52rem' }}>
          <thead>
            <tr><th>Project</th><th>Value of code q</th><th>What is stored</th><th>Fact</th></tr>
          </thead>
          <tbody>
            <tr><td>llama.cpp Q4_1</td><td><code>d·q + m</code></td><td>d and m as FP16, codes 0..15</td><td>D1</td></tr>
            <tr><td>MLX affine</td><td><code>s·q + bias</code></td><td>scale and bias in the input's type</td><td>D5</td></tr>
            <tr><td>MPSGraph</td><td><code>scale·(q - zeroPoint)</code></td><td>float scale, integer zero point of the input's type (Int4, UInt4, Int8, UInt8)</td><td>C4</td></tr>
            <tr><td>PyTorch int4 (tinygemm)</td><td><code>q·scale + zero</code></td><td>codes as int4 in -8..7, scale and zero in bf16; the zero is added after scaling</td><td>D6</td></tr>
          </tbody>
        </table>
      </div>
      <Ask q={q['x2/X2.2/q-convert']} />

      <h2>Granularity and error</h2>
      <p>
        The stepper is one block. The explorer below quantises a whole 16 × 4096 matrix and plots relative error
        against what you pay, bits per weight with the scales included. Four things to find: what one more bit does;
        whether small blocks help normal weights or heavy tails more; whether Q4_0's sixteenth level is worth its
        clipping; and what one scale per tensor does to heavy tails at 4 bits.
      </p>
      <Today checked={CHECKED('D6')}>
        <p>
          PyTorch's quantisation lives in torchao now. Its <code>Int4WeightOnlyConfig</code> is group-wise only, with
          group sizes 256, 128, 64 and 32 and a default of 128.
        </p>
      </Today>
      <Sweep />
      <SpendPredict />
      <Ask q={q['x2/X2.2/q-outlier']} />

      <h2>The stored scale</h2>
      <p>
        Look again at the quantiser: the codes are computed with the float d, and then the block stores FP16(d). So
        dequantisation multiplies by a slightly different number from the one the codes were chosen for. For the
        default block d = 0.1 but FP16 keeps {fp16(0.1)}, a relative change of about {(100 * Math.abs(fp16(0.1) - 0.1) / 0.1).toFixed(3)}%.
        That is small next to a 4-bit step, which is why nobody cares for accuracy. It matters for testing: a CPU
        reference that skips the FP16 round trip will not match the GPU's dequantised values bit for bit. Untick "store
        the scale as FP16" in the stepper and watch the last digit of the dequantised values change. FP16 itself is
        appendix A's subject (<a href={href('appA', 'A.2')}>section A.2</a>).
      </p>
      <Source section="A.2" pages="586–590" />

      <h2>Two-level scales: Q4_K</h2>
      <p>
        A scale and a min per 32 weights cost Q4_1 a full bit per weight. llama.cpp's Q4_K keeps both and pays half:
        a super-block of 256 weights has two FP16 numbers, d and dmin, and each of its eight 32-weight sub-blocks gets a
        6-bit scale and a 6-bit min that multiply them. Sixteen 6-bit numbers fit in 12 bytes, so the block is 2 + 2 +
        12 + 128 = 144 bytes: {bitsPerWeight(Q4_K)} bits per weight against Q4_1's {bitsPerWeight(Q4_1)}, and the same as
        Q4_0's {bitsPerWeight(Q4_0)}, which has no min at all. The value is d·sc·q - dmin·m. The bit-packing of those 12
        bytes is the kind of detail a kernel must get exactly right:
      </p>
      <Aside>
        <p>
          The table is <code>get_scale_min_k4</code> read as data; a unit test packs random scales and mins with its
          inverse and reads every one back.
        </p>
      </Aside>
      <div style={{ overflowX: 'auto' }} className="wide">
        <table className="data" data-part="q4k" style={{ maxWidth: '52rem' }}>
          <thead><tr><th>Sub-block j</th><th>6-bit scale from</th><th>6-bit min from</th></tr></thead>
          <tbody>
            {Array.from({ length: 8 }, (_, j) => (
              <tr key={j}><td>{j}</td><td>{scaleMinSources(j).sc}</td><td>{scaleMinSources(j).m}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <Ask q={q['x2/X2.2/q-q4k']} label="Work it out" />
      <p>
        Next, section <a href={href('x2', 'X2.3')}>X2.3</a> puts two codes in each byte and shows that where each
        nibble goes is chosen for the kernel that reads it.
      </p>
    </>
  );
}
