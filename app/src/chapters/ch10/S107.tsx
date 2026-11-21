import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BLOCK_SIZES,
  HANDOFFS,
  OP_PARAM,
  callText,
  hybrid,
  isPow2,
  ones,
  shuffle,
  smallValues,
  sourceOf,
  warpReduceTrace,
  type Handoff,
  type Op,
  type Outcome,
  type Trace,
} from '../../sim/shuffle';
import { functionSource } from '../../sim/toolchain';
import { widthChoices } from '../../sim/warps';
import { MIN_W, ShuffleScene, VIEW_W, WarpGrid, gridHeight } from './ShuffleScene';
import { useNarrow } from './useNarrow';
import { STATUS } from './status';

import { NoteGroup } from './NoteGroup';
const LAB = 'ch10/lab1_sum';
const AS_PRINTED = 'ch10/snippets/s107_warp_reduce_as_printed.cu';
const HYBRID = 'ch10/snippets/s107_hybrid.cu';
const missing = (s: string | null) => !s || s.startsWith('// could not load');

/** Identifiers coloured by the memory they live in. */
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', input_s: 'shared',
  partialSum: 'reg', val: 'reg', t: 'reg', stride: 'reg', full: 'reg', lane: 'reg', width: 'reg',
};

/** Warp widths to offer: the 8-lane toy of the book's figures, then the book's 32, AMD's 64 and any device reported. */
function useWarpChoices(withToy: boolean) {
  const { reports } = useReports();
  return useMemo(() => {
    const real = widthChoices(reports as Reports).filter((c) => isPow2(c.width) && c.width >= 8 && c.width <= 64);
    const toy = { key: 'toy', label: '8 lanes, the scale of Fig. 10.12', width: 8, note: 'A toy: the book draws warps of 4 threads in Fig. 10.12; no GPU has warps of 8.' };
    return [...(withToy ? [toy] : []), ...real.map((c) => ({ key: c.key, label: c.label, width: c.width, note: c.note }))];
  }, [reports, withToy]);
}

const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const count = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;

// ---------------------------------------------------------------- explorer: one shuffle
type ClickMode = 'explain' | 'exit' | 'mask';
const OP_LABEL: Record<Op, string> = { down: 'down', up: 'up', xor: 'xor', idx: 'by source lane' };

function explainLane(op: Op, lane: number, param: number, width: number, warp: number, r: { outcome: Outcome; value: number | null; src: number | null }, exited: boolean[], mask: boolean[]): string {
  if (r.outcome === 'absent')
    return exited[lane]
      ? `Lane ${lane} has returned, so it does not call. A lane that asks for its value gets an undefined one.`
      : `Lane ${lane} is not in the mask, so it takes no part in this call. A lane that asks for its value gets an undefined one.`;
  const { target, inRange } = sourceOf(op, lane, param, width);
  const base = lane - (lane % width);
  const pos = lane % width;
  const whole = width === warp;
  const where = whole ? 'the warp' : `its group, lanes ${base} to ${base + width - 1}`;
  // With the whole warp as the group, positions are lane numbers and need no introduction.
  const intro = whole ? `Lane ${lane}'s source would be lane ${op === 'down' ? pos + param : pos - param}`
    : `Lane ${lane} is position ${pos} of ${where}. Its source would be position ${op === 'down' ? pos + param : pos - param}`;
  let s: string;
  switch (op) {
    case 'down':
      s = intro + (inRange ? (whole ? '.' : `, lane ${target}.`) : `, past the end of ${where}. The source lane is not wrapped around, so lane ${lane} keeps its own value, ${lane}.`);
      break;
    case 'up':
      s = intro + (inRange ? (whole ? '.' : `, lane ${target}.`) : `, before the start of ${where}, so lane ${lane} keeps its own value, ${lane}.`);
      break;
    case 'xor':
      s = `${lane} xor ${param} is ${target}. ` +
        (!inRange ? `That lane is in a later group, which xor cannot read, so lane ${lane} gets its own value back.`
          : target < base ? `That lane is in an earlier group, which xor may read.` : `It is in the same group.`);
      break;
    case 'idx':
      s = param >= width
        ? `srcLane ${param} is outside 0 to ${width - 1}, so it is taken modulo ${width}: position ${param % width} of ${where}, lane ${target}.`
        : `srcLane ${param} is position ${param} of ${where}: lane ${target}.`;
      break;
  }
  if (r.outcome === 'undefined') s += ` But lane ${r.src} ${exited[r.src!] ? 'has returned' : 'is not in the mask'}, so the value lane ${lane} gets is undefined.`;
  else if (r.outcome === 'moved') s += ` It receives ${r.value}.`;
  return s;
}

function ShuffleExplorer() {
  const choices = useWarpChoices(true);
  const [key, setKey] = useState('book');
  const chosen = choices.find((c) => c.key === key) ?? choices[1] ?? choices[0];
  const warp = chosen.width;
  const [op, setOp] = useState<Op>('down');
  const [param, setParam] = useState(16);
  const [width, setWidth] = useState(32);
  const [mode, setMode] = useState<ClickMode>('explain');
  const [exited, setExited] = useState<boolean[]>(() => new Array(32).fill(false));
  const [mask, setMask] = useState<boolean[]>(() => new Array(32).fill(true));
  const [focus, setFocus] = useState(20);

  // A new warp width starts clean.
  useEffect(() => {
    setExited(new Array(warp).fill(false));
    setMask(new Array(warp).fill(true));
    setWidth(warp);
    setParam((p) => Math.min(Math.max(p, 1), warp - 1));
    setFocus((f) => Math.min(f, warp - 1));
  }, [warp]);

  const ready = exited.length === warp && mask.length === warp;
  const values = range(warp);
  const res = ready ? shuffle(op, values, param, { width, mask, exited }) : [];
  const tally = (o: Outcome) => res.filter((r) => r.outcome === o).length;
  const pmin = op === 'idx' ? 0 : 1;

  // On a phone the warp folds into rows of 16 lanes, drawn on a narrower canvas so the
  // lanes stay large enough to read and click; no lane is off-screen.
  const narrow = useNarrow();
  const VW = narrow ? 400 : VIEW_W;
  const perRow = Math.min(warp, narrow ? 16 : 32);
  const blocks = warp / perRow;
  const cw = Math.min((VW - 16) / perRow, 48);
  const X0 = (VW - cw * perRow) / 2;
  const CH = 24;
  const GAPY = 46;
  const BLOCK_H = 14 + CH + GAPY + CH + 22;
  const H = blocks * BLOCK_H + 4;
  const pos = (lane: number) => ({ x: X0 + (lane % perRow) * cw, y: Math.floor(lane / perRow) * BLOCK_H + 14 });
  const font = cw < 22 ? 9 : 11;

  const clickLane = (lane: number) => {
    if (mode === 'explain') setFocus(lane);
    else if (mode === 'exit') setExited((e) => e.map((x, i) => (i === lane ? !x : x)));
    else setMask((m) => m.map((x, i) => (i === lane ? !x : x)));
  };
  const reset = () => {
    setExited(new Array(warp).fill(false));
    setMask(new Array(warp).fill(true));
  };
  const f = res[focus];

  return (
    <Bench
      title="One shuffle across a warp"
      task="Set down, delta 16, and explain lane 20. Then width 8, delta 2: what do lanes 13 and 14 get?"
      foot={
        <>
          Each lane starts with its own lane number, so the value a lane receives names the lane it came from. Top row:
          what each lane passes in. Bottom row: what the call returns to it. A dashed box kept its own value; a red{' '}
          <b className="t-fail">?</b> is undefined.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Warp
          <select value={chosen.key} onChange={(e) => setKey(e.target.value)} aria-label="warp width">
            {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
        <label className="field">
          Function
          <select value={op} aria-label="shuffle function" onChange={(e) => { const o = e.target.value as Op; setOp(o); if (o !== 'idx' && param < 1) setParam(1); }}>
            {(['down', 'up', 'xor', 'idx'] as Op[]).map((o) => <option key={o} value={o}>{OP_LABEL[o]}</option>)}
          </select>
        </label>
        <label className="field">
          {OP_PARAM[op]}: <b>{param}</b>
          <input type="range" min={pmin} max={warp - 1} value={param} aria-label={OP_PARAM[op]} onChange={(e) => setParam(+e.target.value)} />
        </label>
        <label className="field">
          width
          <select value={width} aria-label="width argument" onChange={(e) => setWidth(+e.target.value)}>
            {range(Math.round(Math.log2(warp)) + 1).map((k) => 2 ** k).map((w) => <option key={w} value={w}>{w}{w === warp ? ' (default)' : ''}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="ui muted">Clicking a lane</span>
        <span className="seg" role="group" aria-label="What clicking a lane does">
          {([['explain', 'explains it'], ['exit', 'makes it return'], ['mask', 'takes it out of the mask']] as [ClickMode, string][]).map(([m, l]) => (
            <button key={m} aria-pressed={mode === m} onClick={() => setMode(m)}>{l}</button>
          ))}
        </span>
        <button className="quiet" onClick={reset} disabled={!exited.some(Boolean) && mask.every(Boolean)}>All lanes back</button>
      </div>
      <pre className="mono" style={{ margin: '0 0 0.6rem', whiteSpace: 'pre-wrap' }} data-call>{ready ? callText(op, param, width, mask) : ''}</pre>
      {ready ? (
        <svg viewBox={`0 0 ${VW} ${H}`} role="img" aria-label="Lanes before and after the shuffle" data-rows={blocks}
          style={{ width: '100%', maxWidth: VW, minWidth: narrow ? 0 : warp > 32 ? 640 : warp > 8 ? MIN_W : 0, display: 'block' }}>
          {/* groups of `width` lanes */}
          {width < warp
            ? range(warp / width).map((g) => {
                if (g % 2 === 0) return null;
                const first = g * width;
                const parts = [];
                for (let l = first; l < first + width; l += perRow - ((l % perRow))) {
                  const n = Math.min(first + width - l, perRow - (l % perRow));
                  const p = pos(l);
                  parts.push(<rect key={l} x={p.x} y={p.y - 2} width={n * cw} height={CH * 2 + GAPY + 4} fill="var(--rule-soft)" />);
                }
                return <g key={g}>{parts}</g>;
              })
            : null}
          {/* arrows */}
          {res.map((r) => {
            if (r.outcome === 'absent' || r.src == null) return null;
            const d = pos(r.lane);
            const xd = d.x + cw / 2;
            if (r.outcome === 'own') return <line key={'a' + r.lane} x1={xd} y1={d.y + CH + 2} x2={xd} y2={d.y + CH + GAPY - 2} stroke="var(--ink-3)" strokeDasharray="2 2" />;
            const s = pos(r.src);
            const hot = r.lane === focus;
            return (
              <line key={'a' + r.lane} x1={s.x + cw / 2} y1={s.y + CH + 1} x2={xd} y2={d.y + CH + GAPY - 1}
                stroke={r.outcome === 'undefined' ? 'var(--fail)' : 'var(--reg)'} strokeWidth={hot ? 2.4 : 1}
                strokeDasharray={r.outcome === 'undefined' ? '3 2' : undefined} opacity={hot ? 1 : 0.55} />
            );
          })}
          {range(warp).map((lane) => {
            const p = pos(lane);
            const r = res[lane];
            const gone = exited[lane];
            const off = !mask[lane];
            const hot = lane === focus;
            const show = lane % (perRow > 16 ? 4 : 1) === 0 || hot;
            const outFill = r.outcome === 'moved' ? 'var(--reg-wash)' : r.outcome === 'undefined' ? 'var(--fail-wash)' : 'none';
            const outStroke = r.outcome === 'moved' ? 'var(--reg)' : r.outcome === 'undefined' ? 'var(--fail)' : 'var(--ink-3)';
            return (
              <g key={lane} onClick={() => clickLane(lane)} style={{ cursor: 'pointer' }}>
                {show ? <text x={p.x + cw / 2} y={p.y - 4} fontSize="9" textAnchor="middle" style={{ fill: hot ? 'var(--ink)' : 'var(--ink-2)' }} fontWeight={hot ? 700 : 400}>{lane}</text> : null}
                <rect data-in-lane={lane} data-state={gone ? 'exited' : off ? 'unmasked' : 'calls'} x={p.x + 0.8} y={p.y} width={cw - 1.6} height={CH} rx="2"
                  fill={gone || off ? 'none' : 'var(--cell)'} stroke={hot ? 'var(--ink)' : gone || off ? 'var(--ink-3)' : 'var(--reg)'}
                  strokeWidth={hot ? 2.2 : 1} strokeDasharray={gone ? '2 2' : off ? '1 2' : undefined} />
                <text x={p.x + cw / 2} y={p.y + CH / 2 + font / 2.8} fontSize={font} textAnchor="middle" style={{ fill: gone || off ? 'var(--ink-3)' : 'var(--ink)' }}>
                  {gone ? '×' : lane}
                </text>
                <rect data-lane={lane} data-outcome={r.outcome} data-value={r.value == null ? (r.outcome === 'absent' ? '' : '?') : r.value}
                  x={p.x + 0.8} y={p.y + CH + GAPY} width={cw - 1.6} height={CH} rx="2" fill={outFill} stroke={hot ? 'var(--ink)' : outStroke}
                  strokeWidth={hot ? 2.2 : 1} strokeDasharray={r.outcome === 'own' || r.outcome === 'absent' ? '3 2' : undefined} />
                <text x={p.x + cw / 2} y={p.y + CH + GAPY + CH / 2 + font / 2.8} fontSize={font} textAnchor="middle"
                  style={{ fill: r.outcome === 'undefined' ? 'var(--fail)' : r.outcome === 'absent' ? 'var(--ink-3)' : 'var(--ink)' }} fontWeight={r.outcome === 'undefined' ? 700 : 400}>
                  {r.outcome === 'absent' ? '–' : r.value == null ? '?' : r.value}
                </text>
              </g>
            );
          })}
        </svg>
      ) : null}
      {ready ? (
        <>
          <p className="narrate" role="status" style={{ margin: '0.7rem 0 0.5rem' }} data-explain>
            {explainLane(op, focus, param, width, warp, f, exited, mask)}
          </p>
          <p className="muted" style={{ margin: 0 }} data-tally>
            {count(tally('moved'), 'lane')} received another lane's value, {count(tally('own'), 'lane')} kept {tally('own') === 1 ? 'its' : 'their'} own,{' '}
            {count(tally('undefined'), 'lane')} got an undefined value, {count(tally('absent'), 'lane')} did not call.
            {warp === 64 ? ' On HIP the mask must be a 64-bit integer, as written here.' : ''}
            {key === 'toy' ? ' ' + chosen.note : ''}
          </p>
        </>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- stepper: warp_reduce
type Scenario = 'full' | 'wave64' | 'return' | 'zero';
const SCENARIOS: { key: Scenario; label: string }[] = [
  { key: 'full', label: 'a full 32-lane warp' },
  { key: 'wave64', label: 'the loop written for 32 lanes, on a 64-wide wavefront' },
  { key: 'return', label: 'lanes from n on returned early' },
  { key: 'zero', label: 'lanes from n on stay and contribute 0' },
];

export function stepperTrace(pattern: 'down' | 'xor', data: 'small' | 'ones', scenario: Scenario, n: number): Trace {
  const hw = scenario === 'wave64' ? 64 : 32;
  const v = data === 'ones' ? ones(hw) : smallValues(hw);
  if (scenario === 'wave64') return warpReduceTrace(v, { pattern, loopWidth: 32 });
  if (scenario === 'full') return warpReduceTrace(v, { pattern });
  return warpReduceTrace(v, { pattern, tail: { from: n, mode: scenario } });
}

function stepText(t: Trace, k: number): string {
  if (k === 0)
    return `Every lane holds its starting value${t.live.some((x) => !x) ? `; lanes ${t.live.indexOf(false)} and up have returned` : ''}. ` +
      `The loop starts at stride ${t.loopWidth / 2}. Press Step.`;
  const st = t.steps[k - 1];
  const live = st.lanes.filter((l) => l.live);
  const n = (o: Outcome) => live.filter((l) => l.outcome === o).length;
  const useful = live.filter((l) => l.useful).length;
  let s = t.pattern === 'down'
    ? `Stride ${st.stride}: every lane that is still running adds the value of the lane ${st.stride} above it. ${count(n('moved'), 'lane')} received another lane's value`
    : `xor ${st.stride}: every lane that is still running trades values with the lane whose number differs from its own in the bit worth ${st.stride}. ${count(n('moved'), 'lane')} received a partner's value`;
  if (n('own')) s += `; ${count(n('own'), 'lane')} had no lane ${st.stride} above ${n('own') === 1 ? 'it' : 'them'} and added ${n('own') === 1 ? 'its' : 'their'} own value to ${n('own') === 1 ? 'itself' : 'themselves'}`;
  if (n('undefined')) s += `; ${count(n('undefined'), 'lane')} read a lane that had returned and got an undefined value`;
  s += `. ${useful} of these ${live.length} additions ${t.pattern === 'down' ? 'reach lane 0' : 'combine data every lane needs'}.`;
  return s;
}

function resultText(t: Trace, scenario: Scenario, n: number): { ok: boolean; text: string } {
  const lanes = t.hardwareWidth;
  if (scenario === 'wave64' && t.pattern === 'down')
    return { ok: false, text: `Lane 0 holds ${t.final[0]} and lane 32 holds ${t.final[32]}. The sum is ${t.total}, and no lane holds it: the loop reduced each half of the wavefront separately.` };
  if (scenario === 'wave64')
    return { ok: false, text: `Every lane holds the sum of its own half: ${t.final[0]} in lanes 0 to 31, ${t.final[32]} in lanes 32 to 63. The sum is ${t.total}.` };
  if (t.pattern === 'down') {
    if (t.lane0 == null)
      return { ok: false, text: `Lane 0's result is undefined. Lanes ${n} and up returned; lanes that read them passed the undefined value on, and it reached lane 0.` };
    return { ok: t.lane0 === t.total, text: `Lane 0 holds ${t.lane0}, the sum of ${scenario === 'zero' ? `lanes 0 to ${n - 1}` : `all ${lanes} values`}.` };
  }
  const undef = t.final.filter((v, i) => t.live[i] && v == null).length;
  if (undef) return { ok: false, text: `${count(undef, 'lane')} ended with an undefined value; ${t.holders.length} hold the sum.` };
  return { ok: t.holders.length === t.live.filter(Boolean).length, text: `Every lane holds ${t.total}.` };
}

function ReduceStepper() {
  const [pattern, setPattern] = useState<'down' | 'xor'>('down');
  const [data, setData] = useState<'small' | 'ones'>('small');
  const [scenario, setScenario] = useState<Scenario>('full');
  const [n, setN] = useState(20);
  const [k, setK] = useState(0);
  const [cone, setCone] = useState(true);
  const t = useMemo(() => stepperTrace(pattern, data, scenario, n), [pattern, data, scenario, n]);
  const total = t.steps.length;
  const at = Math.min(k, total);
  useEffect(() => setK(0), [pattern, data, scenario, n]);
  const done = t.steps.slice(0, at);
  const ops = done.reduce((a, s) => a + s.lanes.filter((l) => l.live).length, 0);
  const useful = done.reduce((a, s) => a + s.lanes.filter((l) => l.useful).length, 0);
  const res = at === total ? resultText(t, scenario, n) : null;
  const tail = scenario === 'return' || scenario === 'zero';
  const H = gridHeight(total + 1, 20) + 6;
  const ring = at === total ? (scenario === 'wave64' && pattern === 'down' ? [0, 32] : pattern === 'down' ? [0] : []) : [];
  return (
    <Bench
      title="warp_reduce, step by step"
      task="Step a full warp. Count the additions that reach lane 0. Then break it two ways and mend it."
      foot={
        <>
          Rows run down in time: the starting values, then each lane's partial sum after each step. With the cone on, a solid
          cell is an addition whose result reaches lane 0 (for xor: one every lane needs); a pale one is work nobody reads.
          Dashed cells are lanes that returned. Arrows show the step just taken.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Pattern">
          <button aria-pressed={pattern === 'down'} onClick={() => setPattern('down')}>down (Fig. 10.13)</button>
          <button aria-pressed={pattern === 'xor'} onClick={() => setPattern('xor')}>xor butterfly</button>
        </span>
        <label className="field">
          Case
          <select value={scenario} aria-label="case" onChange={(e) => setScenario(e.target.value as Scenario)}>
            {SCENARIOS.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
          </select>
        </label>
        {tail ? (
          <label className="field">
            n = <b>{n}</b>
            <input type="range" min={1} max={31} value={n} aria-label="n" onChange={(e) => setN(+e.target.value)} />
          </label>
        ) : null}
        <label className="field">
          Values
          <select value={data} aria-label="values" onChange={(e) => setData(e.target.value as 'small' | 'ones')}>
            <option value="small">small integers</option>
            <option value="ones">all ones</option>
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={cone} onChange={(e) => setCone(e.target.checked)} /> show the cone</label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => setK(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setK(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setK(at + 1)} disabled={at >= total}>Step</button>
        <button onClick={() => setK(total)} disabled={at >= total}>Run to the end</button>
        <span className="muted" aria-live="polite">step {at} of {total}</span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.7rem' }} data-narrate>{stepText(t, at)}</div>
      <svg viewBox={`0 0 ${VIEW_W} ${H}`} role="img" aria-label="Partial sums of every lane after each step"
        style={{ width: '100%', maxWidth: VIEW_W, minWidth: t.hardwareWidth > 32 ? 640 : MIN_W, display: 'block' }} data-stepper-grid>
        <WarpGrid trace={t} upto={at} cone={cone} ring={ring} gap={20} />
      </svg>
      <p style={{ margin: '0.7rem 0 0' }} data-counts>
        <span className="muted">Lane-operations so far: </span><b>{ops}</b>
        <span className="muted">; reaching {pattern === 'down' ? 'lane 0' : 'every lane'}: </span><b>{useful}</b>
        <span className="muted">; wasted: </span><b>{ops - useful}</b>.
        {res ? <> <span className={'chip ' + (res.ok ? 'ok' : 'fail')} data-result={res.ok ? 'ok' : 'fail'}>{res.ok ? 'correct' : 'wrong'}</span> {res.text}</> : null}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: where to hand over
const HANDOFF_LABEL: Record<Handoff, string> = {
  printed: 'as printed: shared loop while stride ≥ warp width',
  early: 'one level early: stride > warp width',
  double: 'shared loop down to 1, then warp_reduce too',
  early2: 'one level early, each lane adds slots t and t + width (ours)',
};

function HandoffExplorer() {
  const choices = useWarpChoices(false);
  const [B, setB] = useState(1024);
  const [key, setKey] = useState('book');
  const [variant, setVariant] = useState<Handoff>('printed');
  const [line11, setLine11] = useState(true);
  const chosen = choices.find((c) => c.key === key) ?? choices[0];
  const W = chosen.width;
  const h = useMemo(() => hybrid(ones(2 * B), B, W, variant, { postLoopBarrier: line11 }), [B, W, variant, line11]);
  const readSet = new Set(h.handoff.reads.flat());
  const slots = h.handoff.values.length;
  const perRow = 32;
  const rowsN = Math.ceil(slots / perRow);
  const cw = (VIEW_W - 16) / perRow;
  const RH = 40;
  const svgH = rowsN * RH + 8;
  const state = (k: number) => {
    const live = h.handoff.live[k];
    const read = readSet.has(k);
    return live && read ? 'used' : live ? 'dropped' : read ? 'again' : 'idle';
  };
  const dropped = range(slots).filter((k) => state(k) === 'dropped').length;
  const again = range(slots).filter((k) => state(k) === 'again').length;
  const race = h.races[0];
  return (
    <Bench
      title="Where to hand over"
      task="Find out what goes wrong when warp 0 takes over one level too early, or one level too late."
      foot={
        <>
          Every input element is 1, so the right answer is twice the block size. The strip shows <code>input_s</code> at the
          hand-off: solid slots still hold a partial sum and are read by warp 0; red ones are either still holding a sum that
          no lane reads, or already added into a lower slot and read again.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Block size
          <select value={B} aria-label="block size" onChange={(e) => setB(+e.target.value)}>
            {BLOCK_SIZES.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <label className="field">
          Warp width
          <select value={chosen.key} aria-label="warp width" onChange={(e) => setKey(e.target.value)}>
            {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
        <label className="field">
          Hand-off
          <select value={variant} aria-label="hand-off" onChange={(e) => setVariant(e.target.value as Handoff)}>
            {HANDOFFS.map((v) => <option key={v} value={v}>{HANDOFF_LABEL[v]}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={line11} onChange={(e) => setLine11(e.target.checked)} /> barrier after the loop (line 11)</label>
      </div>
      <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.8rem' }} data-handoff-table>
        <tbody>
          <tr><td style={{ paddingLeft: '0.6rem' }}>shared-memory iterations (strides)</td><td className="num" data-k="strides">{h.strides.length ? h.strides.join(', ') : 'none'}</td></tr>
          <tr><td style={{ paddingLeft: '0.6rem' }}>__syncthreads() executed</td><td className="num" data-k="barriers">{h.barriers}</td></tr>
          <tr><td style={{ paddingLeft: '0.6rem' }}>shared-memory writes / reads</td><td className="num" data-k="shared">{h.sharedWrites} / {h.sharedReads}</td></tr>
          <tr><td style={{ paddingLeft: '0.6rem' }}>shuffle steps in warp 0</td><td className="num" data-k="shuffles">{h.shuffleSteps}</td></tr>
          <tr>
            <td style={{ paddingLeft: '0.6rem' }}>result (right answer {h.expected})</td>
            <td className="num" data-k="result">
              <span className={'chip ' + (h.correct ? 'ok' : 'fail')}>{h.result == null ? 'undefined' : h.result}</span>
            </td>
          </tr>
        </tbody>
      </table>
      <svg viewBox={`0 0 ${VIEW_W} ${svgH}`} role="img" aria-label="Shared memory at the hand-off" style={{ width: '100%', maxWidth: VIEW_W, minWidth: MIN_W, display: 'block' }}>
        {range(slots).map((k) => {
          const x = 8 + (k % perRow) * cw;
          const y = Math.floor(k / perRow) * RH + 12;
          const s = state(k);
          const fill = s === 'used' ? 'var(--shared)' : s === 'idle' ? 'var(--cell)' : 'var(--fail-wash)';
          const stroke = s === 'used' ? 'var(--shared)' : s === 'idle' ? 'var(--rule)' : 'var(--fail)';
          return (
            <g key={k} data-slot={k} data-state={s}>
              {k % 8 === 0 ? <text x={x + cw / 2} y={y - 3} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{k}</text> : null}
              <rect x={x + 0.6} y={y} width={cw - 1.2} height={22} rx="2" fill={fill} stroke={stroke} />
              <text x={x + cw / 2} y={y + 14.5} fontSize={cw < 22 ? 8 : 9.5} textAnchor="middle"
                style={{ fill: s === 'used' ? 'var(--page)' : s === 'idle' ? 'var(--ink-3)' : 'var(--fail)' }}>{h.handoff.values[k] ?? ''}</text>
            </g>
          );
        })}
      </svg>
      <p className="narrate" role="status" style={{ margin: '0.6rem 0 0' }} data-handoff-says>
        {variant === 'printed'
          ? `The last shared iteration ${h.strides.length ? `(stride ${h.strides[h.strides.length - 1]})` : '(none ran: the block is one warp)'} leaves ${W} partial sums, exactly one for each lane of warp 0.`
          : variant === 'early'
            ? dropped
              ? `${dropped} slots still hold partial sums that no lane reads: half the input is silently dropped. Nothing crashes and nothing warns.`
              : `With a block of one warp the loop never runs, so stopping early changes nothing here.`
            : variant === 'double'
              ? `Only slot 0 still holds a partial sum. Warp 0 also reads ${again} slots whose values are already inside slot 0, and counts them again.`
              : h.pastEnd
                ? `This block has only one warp's worth of slots, so lanes read past the end of input_s.`
                : race
                  ? `Without line 11, thread ${race.thread} reads input_s[${race.slot}], which thread ${race.other} wrote in the same iteration with no barrier between: a data race, and an undefined result.`
                  : `Each lane adds two slots that other threads wrote, so line 11 is what makes their writes visible. Untick it.`}
        {variant !== 'early2' && !line11 && h.races.length === 0 ? ' Line 11 is off, and nothing changed: every lane of warp 0 reads only a slot it wrote itself.' : ''}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- code
function WarpReduceListing() {
  const printed = useLabFile(AS_PRINTED);
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  const fn = !missing(cu) ? functionSource(cu!, 'warp_reduce') : null;
  const mask = !missing(cu) ? /#if defined\(__HIPCC__\)[\s\S]*?#endif/.exec(cu!)?.[0] ?? null : null;
  const mfn = !missing(metal) ? functionSource(metal!, 'warp_reduce') : null;
  if (missing(printed)) return null;
  const tabs = [
    { label: 'as printed (CUDA only)', code: printed!, tiers: TIERS, status: STATUS.printed },
    ...(fn ? [{ label: 'portable (lab 10.1)', code: (mask ? mask + '\n\n' : '') + fn.text, tiers: TIERS, status: STATUS.cu }] : []),
    ...(mfn ? [{ label: 'Metal (lab 10.1)', code: mfn.text, tiers: TIERS, status: STATUS.metal }] : []),
  ];
  return (
    <Code
      tabs={tabs}
      caption={
        <>
          First tab: Fig. 10.13 and the helpers of pp. 238–239 as printed (<code>labs/{AS_PRINTED}</code>). The others are
          lab 10.1's reference solution (<code>labs/{LAB}/solution/</code>): a mask type that is 32 bits for CUDA and 64 for
          HIP, and a loop that starts at <code>warpSize / 2</code>; on Metal, no mask and a width passed in from{' '}
          <code>[[threads_per_simdgroup]]</code>.
          {fn ? null : <> The lab's files are not on this machine yet, so only the first tab is shown.</>}
        </>
      }
    />
  );
}

function HybridListing() {
  const src = useLabFile(HYBRID);
  const k = !missing(src) ? functionSource(src!, 'WarpLevelSumReductionKernel') : null;
  if (!k) return null;
  return (
    <Code
      tabs={[{ label: 'Fig. 10.14, corrected', code: k.text, tiers: TIERS, status: STATUS.snippet }]}
      highlight={[5, 11, 12, 13, 14]}
      caption={
        <>
          The book's kernel with its line numbers, from <code>labs/{HYBRID}</code>, where the portable{' '}
          <code>warp_reduce</code> above and a <code>warpIdx()</code> on <code>warpSize</code> sit beside it. Marked: the
          loop that stops at the warp width (line 5), the barrier after it (line 11) and the hand-off (lines 12 to 14).
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
export const questionsS107: Record<string, Question> = {
  'ch10/10.7/q-top-lanes': {
    kind: 'numeric',
    id: 'ch10/10.7/q-top-lanes',
    prompt: (
      <>
        A 32-lane warp, lane <i>i</i> holding <i>i</i> in <code>v</code>. Every lane runs{' '}
        <code>v = __shfl_down_sync(0xffffffff, v, 16)</code>. What does lane 20 hold afterwards?
      </>
    ),
    answer: 20,
    explain: (
      <p>
        Lane 20's source would be lane 36, past the end of the warp. The source lane is not wrapped around, so the upper 16
        lanes keep their own value: 20. That is why, in <code>warp_reduce</code>, the top lanes add their own value to
        themselves and compute sums nobody uses.
      </p>
    ),
    source: 'CUDA guide 5.4.6 (notes/today/ch10-facts.md A3); book Fig. 10.11, p. 238',
  },
  'ch10/10.7/q-width8': {
    kind: 'numeric',
    id: 'ch10/10.7/q-width8',
    prompt: (
      <>
        Same warp, lane <i>i</i> holding <i>i</i>, but now <code>v = __shfl_down_sync(0xffffffff, v, 2, 8)</code>. What does
        lane 14 receive?
      </>
    ),
    answer: 14,
    explain: (
      <p>
        Width 8 splits the warp into four groups of eight that behave as separate warps. Lane 14 is position 6 of the group
        8 to 15; its source would be position 8, outside the group, so it keeps its own value, 14. Lane 13 (position 5)
        gets lane 15.
      </p>
    ),
    source: 'book p. 241; CUDA guide 5.4.6 (ch10-facts A3)',
  },
  'ch10/10.7/q-syncwarp': {
    kind: 'choice',
    id: 'ch10/10.7/q-syncwarp',
    prompt: (
      <>
        An older tutorial finishes a block sum with the last warp alone. The shared-memory loop has left 64 partial sums
        in <code>input_s</code>, and then, with no barrier and no <code>_sync</code> call:
        <pre className="mono" style={{ margin: '0.5rem 0', whiteSpace: 'pre-wrap' }}>
          {'if (t < 32) {\n  volatile float* s = input_s;\n  s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];\n  s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];\n}'}
        </pre>
        On a device with independent thread scheduling, which is every device in NVIDIA's current tables, what can lane 0
        end up with?
      </>
    ),
    options: [
      {
        label: 'A wrong sum: lanes may run ahead, so lane 0 can read s[16] before lane 16 adds s[48] into it, or after it has also added s[32].',
        correct: true,
      },
      { label: 'The right sum: volatile forces every read and write to go to shared memory in program order.', why: 'volatile keeps the values out of registers. It orders nothing between one lane and another.' },
      { label: 'The right sum: a warp issues one instruction for all its lanes.', why: 'That is the assumption the code was written on. NVIDIA says independent thread scheduling makes it invalid, and names this very pattern.' },
      { label: 'Nothing: without a barrier or a _sync call, the warp deadlocks at its first step.', why: 'Nothing in it waits, so nothing can hang. The reads simply race the writes.' },
    ],
    explain: (
      <p>
        NVIDIA's guide says "synchronization-free intra-warp reductions" written for GPUs before compute capability 7.0
        must be revisited, and synchronised with <code>__syncwarp()</code>. Since <code>__syncwarp()</code> guarantees only
        that the lanes meet at the call, this code would need one between each step's reads and its writes, and another
        before the next step reads (our reading of the rule). The shuffle reduction later in this section needs none of
        that: each <code>__shfl_down_sync</code> names its lanes and exchanges registers in one call.
      </p>
    ),
    source: 'CUDA guide 3.2.2.1.1, 5.4.4.2 (notes/today/ch10-facts.md A2, A5); book p. 237',
  },
  'ch10/10.7/q-early-return': {
    kind: 'choice',
    id: 'ch10/10.7/q-early-return',
    prompt: (
      <>
        A warp-level reduction over n = 50 elements, one per thread, two warps. Each thread starts with{' '}
        <code>if (threadIdx.x &gt;= n) return;</code> and then calls <code>warp_reduce</code>. What is wrong, and what is the
        fix?
      </>
    ),
    options: [
      { label: 'Nothing: the full mask ignores exited threads, so warp 1 sums its 18 live values and its lane 0 holds their total.', why: 'The mask is fine with exited lanes. Reading their values is not.' },
      {
        label: 'Lanes of warp 1 read lanes that returned, and those undefined values reach lane 0. Keep every lane in the shuffles, giving lanes with no element 0.',
        correct: true,
      },
      { label: 'The kernel deadlocks, because the lanes that returned never reach the shuffle the full mask waits for.', why: 'Exited lanes do not block a _sync call; they make the values read from them undefined.' },
      { label: 'Use a smaller mask naming only the live lanes, the low 18 bits in warp 1; then every read is defined.', why: 'A lane outside the mask is not taking part either, so reading it is just as undefined.' },
    ],
    explain: (
      <p>
        "Exited threads are ignored" for the mask, but "if the target thread is inactive, the retrieved value is
        undefined". In warp 1, threads 50 to 63 (lanes 18 to 31) have returned; at stride 16 lane 2 reads lane 18, lanes 2
        to 15 all get undefined values, and by the second step one has reached lane 0. Every lane must take part, holding the identity of the
        operator (0 for a sum, the lowest value for a max) when it has no data. Try it in the stepper: "lanes from n on
        returned early", then "stay and contribute 0".
      </p>
    ),
    source: 'CUDA guide 5.4.6.5 to 5.4.6.6 (ch10-facts A3); model: src/sim/shuffle.ts',
  },
  'ch10/10.7/q-line11': {
    kind: 'choice',
    id: 'ch10/10.7/q-line11',
    prompt: (
      <>
        In Fig. 10.14, is the <code>__syncthreads()</code> on line 11, right after the loop, needed for a correct result?
      </>
    ),
    options: [
      { label: 'Yes: without it warp 0 could load input_s before the last iteration has written it.', why: 'Which thread wrote the slot that lane t loads?' },
      { label: 'No: as indexed, each lane of warp 0 loads only the slot it wrote in the last iteration.', correct: true },
      { label: 'No: the shuffles in warp_reduce synchronise the warp, and that covers shared memory.', why: 'Shuffles do not order memory accesses at all ("these intrinsics do not provide any memory ordering").' },
      { label: 'Yes: a block barrier is needed before any warp shuffles values it loaded from shared memory.', why: 'Shuffles move registers; they do not need a block barrier.' },
    ],
    explain: (
      <>
        <p>
          The last iteration has stride 32 (the warp width): threads 0 to 31, exactly warp 0, write{' '}
          <code>input_s[t]</code>, and on line 13 thread <code>t</code> loads <code>input_s[t]</code>. A thread reading its
          own write needs no barrier. If the loop never runs (a block of one warp), each thread reads what it wrote on line
          04. The barrier inside the loop is the one that protects reads of other threads' slots.
        </p>
        <p>
          It would be needed the moment the hand-off read slots other threads wrote: choose "one level early, each lane adds
          slots t and t + width" in the explorer and untick line 11. Both claims were checked, by the model at every block
          size and width it offers, and on the course emulator under three thread orders (
          <code>labs/ch10/snippets/check/s107_hybrid_test.cu</code>). Keeping the barrier costs one barrier and is harmless.
        </p>
      </>
    ),
    source: 'our analysis of Fig. 10.14 (p. 240), checked with src/sim/shuffle.ts and the emulator',
  },
  'ch10/10.7/q-hip-mask': {
    kind: 'choice',
    id: 'ch10/10.7/q-hip-mask',
    prompt: (
      <>
        This compiles for CUDA:
        <pre className="mono" style={{ margin: '0.5rem 0', whiteSpace: 'pre-wrap' }}>
          {'__device__ float warp_sum(float v) {\n  for (unsigned s = warpSize / 2; s > 0; s /= 2)\n    v += __shfl_down_sync(0xffffffff, v, s);\n  return v;\n}'}
        </pre>
        Built for HIP, where does it fail, and what is the smallest change that keeps one source for both?
      </>
    ),
    options: [
      {
        label: 'At compile time, on every AMD target. Type the mask per vendor (64 bits under __HIPCC__) and pass the all-ones value of that type.',
        correct: true,
      },
      {
        label: 'At run time on 64-lane wavefronts, where lanes 32 to 63 are not in the mask. Write 0xffffffffffffffff instead.',
        why: 'HIP never gets as far as running it: the mask\'s type is checked at compile time. And CUDA\'s mask parameter is an unsigned, 32 bits.',
      },
      { label: 'Only on gfx10 and later, which run 32-wide wavefronts and so need a mask of their own width.', why: 'The check is on the mask\'s type, not its value, and it applies in wave32 mode too.' },
      { label: 'Nowhere: the loop already starts at warpSize / 2, so it adapts to 64-lane wavefronts.', why: 'The loop is portable; the mask literal is not.' },
    ],
    explain: (
      <p>
        Each HIP <code>_sync</code> function has a <code>static_assert</code> that the mask is a 64-bit integer ("Implicitly
        promoting a smaller integer is almost always an error"), on every AMD target, wave32 included. A type chosen per
        vendor keeps one file: lab 10.1 calls it <code>lane_mask_t</code>, the name AMD's own example uses, and passes{' '}
        <code>~lane_mask_t(0)</code>, all ones; llama.cpp calls it <code>ggml_cuda_thread_mask_t</code>. This loop already starts at <code>warpSize / 2</code>; the book's
        starts at 16, which is the next question.
      </p>
    ),
    source: 'notes/today/ch10-facts.md B1, B2; infra-facts.md A3, B3',
  },
  'ch10/10.7/q-wave64': {
    kind: 'numeric',
    id: 'ch10/10.7/q-wave64',
    prompt: (
      <>
        Suppose the mask is fixed but the book's loop (strides 16, 8, 4, 2, 1) runs on a 64-wide AMD wavefront, every lane
        holding 1. What does lane 0 return?
      </>
    ),
    answer: 32,
    explain: (
      <p>
        With strides from 16 down, lane 0's sum only ever draws on lanes 0 to 31. Lane 32 collects lanes 32 to 63 in the
        same way, so each holds 32 and no lane holds 64. The loop must start at <code>warpSize / 2</code>; the stepper's
        case "the loop written for 32 lanes, on a 64-wide wavefront" shows it.
      </p>
    ),
    source: 'ch10-facts B1 (our arithmetic from HIP\'s documented out-of-range rule); model: src/sim/shuffle.ts',
  },
  'ch10/10.7/q-xor': {
    kind: 'choice',
    id: 'ch10/10.7/q-xor',
    prompt: (
      <>
        A softmax over an attention row of 128 scores, one warp per row. Every lane needs the row's maximum to compute{' '}
        <code>exp(x - max)</code> for its own scores. Which reduction avoids a trip through shared memory to broadcast it?
      </>
    ),
    options: [
      { label: 'An xor butterfly (or simd_max on Metal): every lane ends with the maximum.', correct: true },
      { label: 'Fig. 10.13 with max in place of +: lane 0 has the maximum, and the others can read it.', why: 'They can, but only through another shuffle or shared memory: the down tree leaves it in lane 0 alone.' },
      { label: 'Either; both leave the maximum in every lane.', why: 'Run the stepper with both patterns and look at the last row.' },
    ],
    explain: (
      <p>
        In the butterfly every lane trades with a partner in both directions, so after log2(32) = 5 steps every lane holds
        the result, for the same number of shuffles. llama.cpp's <code>warp_reduce_max</code> is exactly this loop with{' '}
        <code>fmaxf</code>; on Metal, <code>simd_max</code> broadcasts the result to every active lane.
      </p>
    ),
    source: 'ch10-facts D1, C1; CUDA guide 5.4.6.5 example 3',
  },
};

// ---------------------------------------------------------------- the section
export function S107() {
  return (
    <>
      <p>
        Run the shared-memory kernel of <a href={href('ch10', '10.6')}>section 10.6</a> on a block of 1,024 threads and
        follow its last iterations. From stride 32 on, at most one warp (threads 0 to 31) has work, yet each of those six
        iterations still goes through shared memory and a <code>__syncthreads()</code> that the whole block must reach.
        Threads of the same warp have a cheaper way to cooperate. CUDA's <b>warp-level primitives</b> let them exchange data and synchronise
        without shared memory and without a block barrier.
      </p>
      <Source section="10.7" pages="237" />

      <h2>A barrier for one warp</h2>
      <p>
        The simplest of them is <code>__syncwarp()</code>: every thread of the warp reaches it before any goes on, like{' '}
        <code>__syncthreads()</code> for a single warp. The book adds that it makes threads of the warp that had diverged
        reconverge. NVIDIA's documentation promises less: <code>__syncwarp(mask)</code> is a barrier that also orders
        memory for the lanes in <code>mask</code>, and convergence is guaranteed at a <code>_sync</code> call, not after
        it. So do not rely on lanes staying in step; every exchange between lanes needs its own <code>_sync</code> call.
      </p>
      <NoteGroup>
        <Source pages="237" />
        <Today checked="2026-11-20 (ch10-facts A2)">
          <p>
            The mask "ensures proper convergence before the hardware executes the intrinsic".
          </p>
        </Today>
      </NoteGroup>

      <h2>Reading another lane's register</h2>
      <p>
        A register belongs to one thread. So far, handing a value to another thread has meant three moves: write it to
        shared memory, wait at a barrier, let the other thread read it. A <b>warp shuffle</b> does it in one: the lanes of a
        warp read each other's registers directly.
      </p>
      <Source pages="237–238" />
      <p>
        The book uses <code>T __shfl_down_sync(unsigned mask, T var, unsigned int delta)</code> (Fig. 10.11). Every lane
        named in <code>mask</code> calls it with its own <code>var</code>, and lane <i>i</i> gets back the <code>var</code>{' '}
        of lane <i>i</i> + <code>delta</code>. Lanes are numbered within their warp. The book's two helpers name the numbers:{' '}
        <code>warpIdx()</code> is <code>threadIdx.x / WARP_SIZE</code>, the warp a thread is in, and <code>laneIdx()</code>{' '}
        is <code>threadIdx.x % WARP_SIZE</code>, its place in that warp. Thread 33 is lane 1 of warp 1.
      </p>
      <Source pages="238–239" figs="Fig. 10.11" />
      <p>
        Fig. 10.11 covers the case where both lanes take part. It does not say what lane 31 gets when it asks for lane 47,
        or what a lane gets from a neighbour that has already returned. Those two answers decide whether a reduction is
        right, so try them before reading on.
      </p>
      <Today checked="2026-11-20 (ch10-facts A3; infra-facts A3)">
        <p>
          The signature now has a fourth parameter, <code>int width = warpSize</code>. The explorer follows NVIDIA's
          rules for both cases.
        </p>
      </Today>
      <p>
        In the explorer, lane <i>i</i> starts with the value <i>i</i>, so what a lane receives names the lane it came from.
        Start with down and delta 16 and read the explanation for lane 20. Then make a lane return, or take one out of the
        mask, and see who is affected. Last, set the width to 8 and watch the warp split into four groups.
      </p>
      <Simplification>
        <p>
          One warp and one call, by the rules NVIDIA documents. HIP documents the same for down, up and xor, and says
          nothing about an out-of-range <code>srcLane</code>; Metal has no mask and no width. No timing.
        </p>
      </Simplification>
      <ShuffleExplorer />
      <Ask q={questionsS107['ch10/10.7/q-top-lanes']} label="Work it out" />
      <Ask q={questionsS107['ch10/10.7/q-width8']} label="Work it out" />

      <h2>A warp reduction in five steps</h2>
      <p>
        Fig. 10.13 builds a sum from shuffles alone. Every lane passes in its value. The stride starts at 16 and halves to
        1, and at each step every lane adds the value of the lane <code>stride</code> above it. After stride 16, lanes 0 to
        15 each hold a sum of two values; after stride 8, lanes 0 to 7 hold sums of four; after stride 1, lane 0 holds the
        sum of all 32.
      </p>
      <Source pages="239–240" figs="Fig. 10.13" />
      <p>
        Two choices in it are deliberate. There is no <code>if</code>: every lane runs every shuffle and every addition,
        because a lane can only read a neighbour that takes part. And only lane 0's return value means anything. The book
        says so; the other lanes return sums with values missing or counted twice.
      </p>
      <WarpReduceListing />
      <p>
        Step through it below. Count how many of the additions reach lane 0. Then switch to the xor butterfly, and then
        break it: run the 32-lane loop on a 64-wide wavefront, and let lanes past the end of the data return early.
      </p>
      <Simplification>
        <p>
          Each step is one shuffle and one add for every lane, executed at once. The model knows nothing of timing or of how
          the hardware schedules the warp; it is exact about which value each lane holds.
        </p>
      </Simplification>
      <ReduceStepper />
      <p>
        A full warp spends 160 lane-operations (32 lanes, 5 steps) to produce a sum that needs 31 additions. The other 129
        are real work whose results nobody reads. Section 10.8 decides to pay that price on purpose. The animation goes
        through the same steps once more, with the cone drawn in.
      </p>
      <ShuffleScene />
      <Ask q={questionsS107['ch10/10.7/q-early-return']} />

      <h2>Shared memory first, then one warp</h2>
      <p>
        The hybrid kernel (Fig. 10.14) keeps the shared-memory tree, with its barriers, only while more than one warp has
        work: the loop runs while <code>stride &gt;= WARP_SIZE</code>. Its last iteration, at stride 32, leaves 32 partial
        sums in <code>input_s[0]</code> to <code>input_s[31]</code>. Warp 0 then loads them into registers, one per lane,
        and finishes with <code>warp_reduce</code>. From stride 16 on there is no shared memory and no barrier.
      </p>
      <NoteGroup>
        <Source pages="240–241" figs="Figs. 10.12, 10.14" />
        <Aside>
          <p>
            Fig. 10.12 draws a toy block of 4 warps of 4 threads; a real block of 1,024 threads has 32 warps of 32 (p. 238).
            The text cites the final write as lines 17 to 19; in the figure it is lines 15 to 17.
          </p>
        </Aside>
      </NoteGroup>
      <HybridListing />
      <p>
        Where exactly to hand over is the only design decision in this kernel. The book's choice is the point at which
        one warp holds every remaining value. Move it one level either way and see what happens to the answer.
      </p>
      <Simplification>
        <p>
          One iteration at a time, all threads before the next. A read of a slot another thread wrote, with no barrier
          between, counts as a race. Thread orders are left to the emulator check.
        </p>
      </Simplification>
      <HandoffExplorer />
      <Ask q={questionsS107['ch10/10.7/q-line11']} />

      <h2>Narrower groups, and the rest of the family</h2>
      <p>
        <code>__shfl_down_sync</code> and its relatives take an optional last argument, the width of the group of lanes
        that cooperate. It defaults to the whole warp. With width 8 the warp behaves as four independent groups of eight,
        each reducing its own values: four short rows per warp. The other shuffles are <code>__shfl_sync</code> (read a named
        lane), <code>__shfl_up_sync</code> and <code>__shfl_xor_sync</code>. Beyond shuffles the book names three families:
        vote functions (which lanes meet a condition; chapter 12 uses one), match functions (which lanes hold the same
        value) and reduce functions, which reduce across a warp "for integer values".
      </p>
      <NoteGroup>
        <Source pages="241" />
        <Today checked="2026-11-20 (ch10-facts A4)">
          <p>
            The reduce family needs compute capability 8.x and takes integers only, so a <code>float</code> warp sum on
            CUDA is still the shuffle tree.
          </p>
        </Today>
      </NoteGroup>
      <p>
        One pattern you will meet in older code belongs to none of these families. It finishes a reduction with the last
        warp unrolled over <code>volatile</code> shared memory, with no barrier and no <code>_sync</code> call, trusting
        the 32 lanes to stay in step. They no longer have to: on every device in NVIDIA's current tables, lanes can
        diverge and reconverge at any point. Do not copy it. Work out what it gets wrong.
      </p>
      <Today checked="2026-11-20 (ch10-facts A5)">
        <p>
          NVIDIA names "synchronization-free intra-warp reductions" as code to revisit for independent thread scheduling.
        </p>
      </Today>
      <Ask q={questionsS107['ch10/10.7/q-syncwarp']} />

      <h2>One source, three backends</h2>
      <p>
        Here the course's rule that one <code>.cu</code> file compiles for CUDA and HIP stops being true. It holds for
        threads, blocks, shared memory, barriers, legacy atomics and runtime calls. It does not hold for the warp{' '}
        <code>_sync</code> intrinsics.
        The book's <code>warp_reduce</code> fails under HIP in three separate ways: the 32-bit mask literal is a compile
        error, because HIP checks that every mask is 64 bits; warps are 64 lanes wide on AMD's gfx9 devices, so a loop that
        starts at 16 reduces each half separately; and HIP's <code>warpSize</code> has not been a compile-time constant since
        ROCm 7.0, so it cannot be relied on where one is needed, such as the size of an array. The fix is in the "portable (lab 10.1)" tab of the <code>warp_reduce</code> listing above: a mask type chosen per vendor, a loop
        from <code>warpSize / 2</code>, and no literal 32.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. HIP and Metal facts here are from AMD's and Apple's documentation
          (notes/today/ch10-facts.md B1, B2, C1; infra-facts.md B3, C2, C4, C5). Nothing has run on AMD hardware.
        </p>
      </Beyond>
      <p>
        Metal does it with less ceremony. <code>simd_shuffle_down(data, delta)</code> has no mask and no width. A kernel
        receives its lane and the SIMD-group width as <code>[[thread_index_in_simdgroup]]</code> and{' '}
        <code>[[threads_per_simdgroup]]</code>. And Metal has what CUDA lacks for floats: <code>simd_sum</code> and{' '}
        <code>simd_max</code> reduce a <code>float</code> across the SIMD-group in one call (Apple7 and later, so every Apple
        silicon Mac) and hand the result to every active lane. Apple does not document the order in which{' '}
        <code>simd_sum</code> adds, so its result may differ from a CPU sum in the last bits.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-backends>
        <thead>
          <tr><th style={{ width: '9rem' }}>What</th><th style={{ width: '27%' }}>CUDA</th><th style={{ width: '27%' }}>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr><td>Warp width a kernel may assume</td><td>32 on every listed device; read <code>warpSize</code></td><td>64 on gfx9, 32 on gfx10 and later; not a compile-time constant since ROCm 7.0</td><td>undocumented; <code>[[threads_per_simdgroup]]</code> at run time</td></tr>
          <tr><td>Full-warp mask</td><td><code>0xffffffff</code> (<code>unsigned</code>)</td><td>must be a 64-bit integer; the 32-bit literal fails a <code>static_assert</code></td><td>no masks</td></tr>
          <tr><td>Shuffle down</td><td><code>__shfl_down_sync(mask, v, d, width)</code>; width defaults to <code>warpSize</code></td><td>same name, 64-bit mask; also <code>__shfl_down(v, d)</code></td><td><code>simd_shuffle_down(v, d)</code>; a width only through <code>simd_shuffle_and_fill_down</code>, Apple8 and later (not M1)</td></tr>
          <tr><td>A lane whose source is past the end</td><td>keeps its own value</td><td>returns its own value</td><td>upper lanes unchanged</td></tr>
          <tr><td>Built-in float warp sum or max</td><td>none (<code>__reduce_*_sync</code> is integer only)</td><td>none by default (integer)</td><td><code>simd_sum</code>, <code>simd_max</code>, Apple7 and later</td></tr>
          <tr><td>Which lanes hold the result</td><td>lane 0 with shuffle down; all with xor</td><td>same</td><td>all active lanes</td></tr>
          <tr><td>Warp barrier</td><td><code>__syncwarp()</code></td><td><code>__syncwarp()</code>; a masked form needs 64 bits</td><td><code>simdgroup_barrier(mem_flags::mem_threadgroup)</code></td></tr>
        </tbody>
      </table>
      <Ask q={questionsS107['ch10/10.7/q-hip-mask']} />
      <Ask q={questionsS107['ch10/10.7/q-wave64']} label="Work it out" />

      <h2>Where you meet this: short attention rows</h2>
      <p>
        Attention computes a softmax over every row of scores, and the rows can be short. PyTorch's persistent softmax
        gives each row one warp, and packs several rows into a warp when they are short enough; its comment notes that
        "only __shfl_ instructions are required for reductions". Several independent rows in one warp is the case the
        width argument exists for. Softmax needs the row's maximum in every lane, though, not only in lane 0, so production code
        uses the butterfly: llama.cpp's <code>warp_reduce_max</code> is an xor loop around <code>fmaxf</code>, and on Metal{' '}
        <code>simd_max</code> broadcasts by itself. Section 10.8 turns the warp reduction into a block-wide one, the shape a
        softmax over a longer row needs.
      </p>
      <Beyond>
        <p>
          PyTorch's <code>PersistentSoftmax.cuh</code> and llama.cpp's <code>common.cuh</code> as read on 2026-11-20
          (notes/today/ch10-facts.md D1, D4).
        </p>
      </Beyond>
      <Ask q={questionsS107['ch10/10.7/q-xor']} />
    </>
  );
}
