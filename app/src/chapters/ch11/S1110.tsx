import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  EX6,
  FIG116,
  FIG_11_20_PRINTED,
  additions,
  brentKung,
  closedForm,
  finalAfter,
  koggeStone,
  needsChain,
  printedDiffers,
  rangeLabel,
  ranges,
  readWriteOverlap,
  reverseLevels,
  run,
  span,
  upSteps,
  warpView,
  type Network,
  type Range,
} from '../../sim/prefixnet';
import { widthChoices } from '../../sim/warps';
import { BkSweepsScene } from './BkSweepsScene';

const SNIPPET = 'ch11/snippets/s1110_brent_kung.cu';
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const WRAP: CSSProperties = { flexWrap: 'wrap' };
const SHADE = [0, 0.2, 0.45, 0.75];
const list = (xs: (string | number)[]) =>
  xs.length <= 1 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`;
/** "2 to 13" for a run of consecutive positions, else a list. */
const span2 = (xs: number[]) =>
  xs.length > 2 && xs[xs.length - 1] - xs[0] === xs.length - 1 ? `${xs[0]} to ${xs[xs.length - 1]}` : list(xs);
const TIERS: Record<string, Tier> = { input: 'global', output: 'global', buffer_s: 'shared', i: 'reg', idx: 'reg', stride: 'reg' };

const INPUTS = [
  { key: 8, label: '8: exercise 6', values: EX6 },
  { key: 16, label: '16: Fig. 11.6\'s values', values: FIG116 },
];

// ---------------------------------------------------------------- a network, drawn
function NetSvg({
  net, k, rows, labels, finals, name, onCell, picked,
}: {
  net: Network;
  /** Steps done. */
  k: number;
  /** Row slots to reserve, so two networks line up step for step. */
  rows: number;
  labels: string[][];
  finals: Set<number>;
  name: string;
  onCell?: (i: number) => void;
  picked?: number | null;
}) {
  const n = net.n;
  const cw = n > 8 ? 32 : 52;
  const X0 = 56;
  const x = (i: number) => X0 + i * cw + cw / 2;
  const DY = 26;
  const y = (j: number) => 46 + j * DY;
  const cellY = y(rows) - 4;
  const width = X0 + n * cw + 8;
  const height = cellY + 38;
  const changed = new Set(k > 0 ? net.steps[k - 1].adders.map((a) => a.dst) : []);
  const two = labels.some((l) => l.length > 1);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${name} network`} data-part="net" data-net={net.id} data-k={k}
      style={{ width: '100%', maxWidth: width * 1.45, minWidth: n > 8 ? 500 : 340, display: 'block' }}>
      {Array.from({ length: n }, (_, i) => (
        <g key={i}>
          <text x={x(i)} y={16} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{i}</text>
          <line x1={x(i)} y1={22} x2={x(i)} y2={cellY - 4} stroke="var(--rule)" />
        </g>
      ))}
      {Array.from({ length: rows }, (_, j) => {
        const st = net.steps[j];
        const label = st ? (st.phase === 'up' ? `up ${j + 1}` : st.phase === 'down' ? `down ${st.level}` : `stride ${st.stride}`) : '';
        const op = !st ? 0 : j < k - 1 ? 0.5 : j === k - 1 ? 1 : 0.14;
        return (
          <g key={j} opacity={st ? 1 : 0}>
            <text x={4} y={y(j) + 4} fontSize="10" style={{ fill: j === k - 1 ? 'var(--ink)' : 'var(--ink-3)', fontWeight: j === k - 1 ? 700 : 400 }}>{label}</text>
            <g opacity={op} data-part="row" data-step={j + 1} data-state={j < k - 1 ? 'done' : j === k - 1 ? 'current' : 'future'}>
              {st?.adders.map((a) => (
                <g key={a.dst}>
                  <line x1={x(a.src)} y1={y(j) - 10} x2={x(a.dst) - 5} y2={y(j) - 3} stroke="var(--ink)" strokeWidth={j === k - 1 ? 1.6 : 1} />
                  <circle cx={x(a.src)} cy={y(j) - 10} r={1.8} fill="var(--ink)" />
                  <circle cx={x(a.dst)} cy={y(j)} r={6.5} fill="var(--cell)" stroke="var(--ink)" strokeWidth={j === k - 1 ? 1.7 : 1} />
                  <text x={x(a.dst)} y={y(j) + 3.3} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}>+</text>
                </g>
              ))}
            </g>
          </g>
        );
      })}
      {k >= net.steps.length && net.steps.length < rows ? (
        <text x={X0 + 4} y={y(net.steps.length) + 4} fontSize="10.5" style={{ fill: 'var(--ok)' }} data-part="finished">
          finished after {net.steps.length} steps
        </text>
      ) : null}
      {labels.map((l, i) => {
        const fin = finals.has(i);
        return (
          <g key={i} data-part="cell" data-pos={i} data-label={l.join('')} data-final={fin ? 1 : 0}
            role={onCell ? 'button' : undefined} tabIndex={onCell ? 0 : undefined}
            aria-label={onCell ? `position ${i}` : undefined}
            onMouseEnter={onCell ? () => onCell(i) : undefined} onFocus={onCell ? () => onCell(i) : undefined} onClick={onCell ? () => onCell(i) : undefined}
            style={onCell ? { cursor: 'pointer' } : undefined}>
            <rect x={x(i) - cw / 2 + 2} y={cellY} width={cw - 4} height={two ? 30 : 22} rx={3}
              fill={fin ? 'var(--ok-wash)' : 'var(--cell)'}
              stroke={picked === i ? 'var(--ink)' : fin ? 'var(--ok)' : changed.has(i) ? 'var(--ink)' : 'var(--rule)'}
              strokeWidth={changed.has(i) || picked === i ? 1.8 : 1} />
            {l.map((t, r) => (
              <text key={r} x={x(i)} y={cellY + (two ? 12.5 + r * 11 : 15)} fontSize={n > 8 ? 9 : 11} textAnchor="middle" style={{ fill: 'var(--ink)' }}>{t}</text>
            ))}
          </g>
        );
      })}
    </svg>
  );
}

const twoLines = (r: Range) => (r.lo === r.hi ? [`x${r.lo}`] : [`x${r.lo}..`, `x${r.hi}`]);

// ---------------------------------------------------------------- Fig. 11.20 as a heat map
function NeedsMap({ n, k, printed, picked, onPick }: { n: number; k: number; printed: boolean; picked: { row: number; pos: number } | null; onPick: (row: number, pos: number) => void }) {
  const levels = useMemo(() => reverseLevels(n), [n]);
  const diffs = useMemo(() => (n === 16 ? printedDiffers() : []), [n]);
  const isDiff = (row: number, col: number) => diffs.some((d) => d.row === row && d.col === col);
  const cw = n > 8 ? 34 : 52;
  const X0 = 58;
  const RH = 28;
  const width = X0 + n * cw + 6;
  const height = 22 + levels.length * RH + 4;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="What each position still needs" data-part="needs-map" data-shading={printed ? 'printed' : 'corrected'}
      style={{ width: '100%', maxWidth: width * 1.2, minWidth: n > 8 ? 540 : 340, display: 'block' }}>
      {Array.from({ length: n }, (_, i) => (
        <text key={i} x={X0 + i * cw + cw / 2} y={14} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{i}</text>
      ))}
      {levels.map((lv, row) => {
        const current = k === lv.step;
        return (
          <g key={row} data-part="needs-row" data-row={row} data-label={lv.label} data-current={current ? 1 : 0}>
            <text x={4} y={22 + row * RH + 17} fontSize="10.5" style={{ fill: current ? 'var(--ink)' : 'var(--ink-2)', fontWeight: current ? 700 : 400 }}>{lv.label}</text>
            {lv.ranges.map((r, col) => {
              const v = printed && n === 16 ? FIG_11_20_PRINTED[row][col] : lv.needs[col];
              const show = row === 0 || lv.changed.includes(col);
              const dark = v >= 2;
              const sel = picked?.row === row && picked.pos === col;
              return (
                <g key={col} data-part="need-cell" data-row={row} data-pos={col} data-shade={v} data-needs={lv.needs[col]} data-differs={isDiff(row, col) ? 1 : 0}
                  role="button" tabIndex={0} aria-label={`${lv.label}, position ${col}`} style={{ cursor: 'pointer' }}
                  onMouseEnter={() => onPick(row, col)} onFocus={() => onPick(row, col)} onClick={() => onPick(row, col)}>
                  <rect x={X0 + col * cw + 1} y={22 + row * RH} width={cw - 2} height={RH - 3} fill="var(--ink)" fillOpacity={SHADE[Math.min(3, v)]}
                    stroke={sel ? 'var(--ink)' : isDiff(row, col) ? 'var(--fail)' : current ? 'var(--ink-2)' : 'var(--rule)'}
                    strokeWidth={sel ? 2 : isDiff(row, col) ? 1.6 : 1} strokeDasharray={isDiff(row, col) && !sel ? '3 2' : undefined} />
                  {show ? (
                    <text x={X0 + col * cw + cw / 2} y={22 + row * RH + 16} fontSize={n > 8 ? 8 : 10} textAnchor="middle"
                      style={{ fill: dark ? 'var(--page)' : 'var(--ink)' }}>{rangeLabel(r)}</text>
                  ) : null}
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------- interactive 1: the stepper
function Stepper() {
  const [n, setN] = useState(16);
  const [show, setShow] = useState<'values' | 'ranges'>('ranges');
  const [step, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [printed, setPrinted] = useState(false);
  const [picked, setPicked] = useState<{ row: number; pos: number } | null>(null);
  const values = (INPUTS.find((x) => x.key === n) ?? INPUTS[1]).values;
  const bk = useMemo(() => brentKung(n), [n]);
  const ks = useMemo(() => koggeStone(n), [n]);
  const bkV = useMemo(() => run(bk, values), [bk, values]);
  const ksV = useMemo(() => run(ks, values), [ks, values]);
  const bkR = useMemo(() => ranges(bk), [bk]);
  const ksR = useMemo(() => ranges(ks), [ks]);
  const levels = useMemo(() => reverseLevels(n), [n]);
  const total = bk.steps.length;
  const k = Math.min(step, total);
  const up = upSteps(bk);
  const kk = Math.min(k, ks.steps.length);

  useEffect(() => {
    if (!playing) return;
    if (k >= total) { setPlaying(false); return; }
    const id = window.setTimeout(() => setK((x) => x + 1), 900);
    return () => window.clearTimeout(id);
  }, [playing, k, total]);

  const labelsOf = (vals: number[], rs: Range[]) => (show === 'values' ? vals.map((v) => [String(v)]) : rs.map(twoLines));
  const addsSoFar = (net: Network, upto: number) => net.steps.slice(0, upto).reduce((s, x) => s + x.adders.length, 0);
  const bkStep = k > 0 ? bk.steps[k - 1] : null;
  const ksStep = k > 0 && k <= ks.steps.length ? ks.steps[k - 1] : null;
  const bkOverlap = bkStep ? readWriteOverlap(bkStep) : [];
  const ksOverlap = ksStep ? readWriteOverlap(ksStep) : [];

  const describeBK = () => {
    if (!bkStep) return `Nothing has run. Each position holds its own input. Press Step.`;
    const pairs = bkStep.adders.map((a) => `${a.src} into ${a.dst}`);
    const what = bkStep.phase === 'up'
      ? `Reduction tree, step ${k}: positions of the form ${2 * bkStep.stride}n − 1 add the position ${bkStep.stride} to their left`
      : `Reverse tree, level ${bkStep.level}: each partial sum moves ${bkStep.stride} ${bkStep.stride === 1 ? 'place' : 'places'} right`;
    const fin = finalAfter(bk, k);
    return `${what} (${bkStep.adders.length > 4 ? `${bkStep.adders.length} additions` : pairs.join(', ')}). Final now: ${span2(fin)}.`;
  };

  // The readout for a picked cell of the needs map.
  const pickedText = (() => {
    if (!picked || picked.row >= levels.length || picked.pos >= n) return null;
    const lv = levels[picked.row];
    const chain = needsChain(lv.ranges, picked.pos);
    const own = rangeLabel(lv.ranges[picked.pos]);
    const pr = n === 16 ? FIG_11_20_PRINTED[picked.row][picked.pos] : null;
    const tail = pr != null && pr !== chain.length
      ? ` The book shades this cell for ${pr}, one level more than its ranges imply.`
      : '';
    if (!chain.length) return `${lv.label}, position ${picked.pos}: holds ${own}, which is final.${tail}`;
    return `${lv.label}, position ${picked.pos}: holds ${own} and still needs ${chain.length} partial ${chain.length === 1 ? 'sum' : 'sums'}: ${list(chain.map((c) => `${c} (${rangeLabel(lv.ranges[c])})`))}.${tail}`;
  })();

  return (
    <Bench
      title="Up-sweep, down-sweep"
      task="Step both networks on the same input. Which positions are final when the reduction tree ends, and how many steps does each network take?"
      foot={
        <>
          Time runs down, one row of adders per step; a dot marks where a value is read, a circle the position that adds it.
          A green cell holds its final value. The table at the bottom is Fig. 11.20: the stronger a cell's shading, the
          more partial sums its position still needs (none, then 1, 2 and 3; the book prints white, light grey, dark grey
          and black). Point at a cell to list them.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Input
          <span className="seg" role="group" aria-label="Input" style={WRAP}>
            {INPUTS.map((x) => (
              <button key={x.key} aria-pressed={n === x.key} onClick={() => { setN(x.key); setK(0); setPicked(null); setPlaying(false); }}>{x.label}</button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Cells show
          <span className="seg" role="group" aria-label="Cells show">
            {(['ranges', 'values'] as const).map((s) => <button key={s} aria-pressed={show === s} onClick={() => setShow(s)}>{s}</button>)}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => { setPlaying(false); setK(0); }} disabled={k === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setK(k - 1); }} disabled={k === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setK(k + 1); }} disabled={k >= total}>Step</button>
        <button onClick={() => setPlaying(!playing)} disabled={k >= total}>{playing ? 'Pause' : 'Play'}</button>
        <span className="muted" aria-live="polite" data-readout="step">step {k} of {total}</span>
      </div>
      <div className="narrate" role="status" data-readout="narrate" style={{ marginBottom: '0.8rem' }}>
        <b>Brent-Kung.</b> {describeBK()}{' '}
        <b>Kogge-Stone.</b>{' '}
        {ksStep
          ? `Stride ${ksStep.stride}: every position from ${ksStep.stride} on adds the one ${ksStep.stride} to its left (${ksStep.adders.length} additions).`
          : k === 0 ? 'Not started.' : `Finished after ${ks.steps.length} steps.`}
      </div>
      <div className="cols" style={{ gridTemplateColumns: n > 8 ? 'minmax(0, 1fr)' : undefined }}>
        <div className="panel" data-panel="brent-kung">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>Brent-Kung</b>: <span data-counter="bk-adds">{addsSoFar(bk, k)}</span> additions in{' '}
            <span data-counter="bk-steps">{k}</span> steps
            {k >= total ? <span className="chip ok" style={{ marginLeft: 6 }}>done</span> : k === up ? <span className="chip" style={{ marginLeft: 6 }}>reduction tree done</span> : null}
          </p>
          <div style={{ overflowX: 'auto' }}>
          <NetSvg net={bk} k={k} rows={total} name="Brent-Kung" labels={labelsOf(bkV[k], bkR[k])} finals={new Set(finalAfter(bk, k))}
            onCell={(i) => { if (k >= up) setPicked({ row: Math.min(levels.length - 1, k - up), pos: i }); }}
            picked={picked && levels[picked.row]?.step === k ? picked.pos : null} />
          </div>
        </div>
        <div className="panel" data-panel="kogge-stone">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>Kogge-Stone</b>: <span data-counter="ks-adds">{addsSoFar(ks, kk)}</span> additions in{' '}
            <span data-counter="ks-steps">{kk}</span> steps
            {kk >= ks.steps.length ? <span className="chip ok" style={{ marginLeft: 6 }}>done</span> : null}
          </p>
          <div style={{ overflowX: 'auto' }}>
            <NetSvg net={ks} k={kk} rows={total} name="Kogge-Stone" labels={labelsOf(ksV[kk], ksR[kk])} finals={new Set(finalAfter(ks, kk))} />
          </div>
        </div>
      </div>
      <p style={{ margin: '0.8rem 0' }} data-readout="overlap" data-bk={bkOverlap.length} data-ks={ksOverlap.length}>
        Read and write in this step:{' '}
        {bkStep ? (
          <>Brent-Kung: {bkOverlap.length ? `overwrites ${span2(bkOverlap)} while another adder reads them` : <span className="chip ok">no position written by one adder is read by another</span>}{' '}</>
        ) : null}
        {ksStep ? (
          <>Kogge-Stone: {ksOverlap.length ? <span className="chip fail">writes positions {span2(ksOverlap)}, which other adders of the same step read</span> : 'writes nothing another adder of this step reads.'}</>
        ) : null}
        {!bkStep && !ksStep ? 'nothing yet.' : null}
      </p>
      <div className="panel" data-panel="needs">
        <div className="row" style={{ marginBottom: '0.5rem', justifyContent: 'space-between' }}>
          <b>What each position still needs (Fig. 11.20)</b>
          {n === 16 ? (
            <span style={FIELD}>
              Shading
              <span className="seg" role="group" aria-label="Shading">
                <button aria-pressed={!printed} onClick={() => setPrinted(false)}>from the ranges</button>
                <button aria-pressed={printed} onClick={() => setPrinted(true)}>as printed</button>
              </span>
            </span>
          ) : null}
        </div>
        <div style={{ overflowX: 'auto' }}>
          <NeedsMap n={n} k={k} printed={printed} picked={picked} onPick={(row, pos) => setPicked({ row, pos })} />
        </div>
        <p className="muted" style={{ margin: '0.5rem 0 0' }} aria-live="polite" data-readout="needs">
          {pickedText ?? (k < up
            ? `The table starts when the reduction tree ends, after step ${up}. Point at a cell.`
            : 'Point at a cell, or at a cell of the Brent-Kung network, to list the partial sums it still needs.')}
        </p>
        {n === 16 ? (
          <p className="muted" style={{ margin: '0.3rem 0 0' }}>
            Dashed red outline: a cell the book shades one level stronger than its ranges imply.
          </p>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive 2: inside one warp
function LaneGrid({ v, upto, title }: { v: ReturnType<typeof warpView>; upto: number; title: string }) {
  const L = v.width > 32 ? 8 : 13;
  const G = v.width > 32 ? 1.5 : 2;
  const LX = 92;
  const width = LX + v.width * (L + G) + 6;
  const rows = v.steps.length;
  const height = 8 + rows * (L + G) + 4;
  return (
    <div>
      <p style={{ margin: '0 0 0.3rem' }}>
        <b>{title}</b>{' '}
        {upto >= v.finishesAt ? <span className="chip ok" data-part="status">done after {v.finishesAt} steps</span> : <span className="chip" data-part="status">step {upto} of {v.finishesAt}</span>}
      </p>
      <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title}: lanes per step`} data-part="lanes" data-net={v.id}
        style={{ width: '100%', maxWidth: width * 1.3, minWidth: 460, display: 'block' }}>
        {v.steps.map((s, k) => (
          <g key={k} opacity={k < upto ? 1 : 0.15} data-part="lane-row" data-step={k + 1} data-adds={s.adds} data-idle={s.idle} data-shown={k < upto ? 1 : 0}>
            <text x={LX - 6} y={8 + k * (L + G) + L - 2} fontSize={v.width > 32 ? 8 : 9.5} textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
              {s.adds} add · {s.idle} idle
            </text>
            {Array.from({ length: v.width }, (_, l) => (
              <rect key={l} x={LX + l * (L + G)} y={8 + k * (L + G)} width={L} height={L} rx={1.2}
                fill={s.lanes.includes(l) ? 'var(--ink-2)' : 'none'} stroke="var(--rule)" />
            ))}
          </g>
        ))}
      </svg>
      </div>
    </div>
  );
}

function WarpExplorer() {
  const { reports } = useReports();
  const widths = useMemo(() => widthChoices(reports as Reports).filter((c) => [8, 16, 32, 64].includes(c.width)), [reports]);
  const [key, setKey] = useState('book');
  const W = (widths.find((c) => c.key === key) ?? widths[0]).width;
  const ks = useMemo(() => warpView('kogge-stone', W), [W]);
  const bk = useMemo(() => warpView('brent-kung', W), [W]);
  const [t, setT] = useState(0);
  useEffect(() => setT(0), [W]);
  const both = [ks, bk];
  return (
    <Bench
      title="Inside one warp"
      task="Slide time forward. When is each network done, and how many lane slots has the warp spent by then?"
      foot={
        <>
          One warp scans one value per lane. A filled square is a lane that adds in that step; an empty one idles but is part
          of the warp the step is issued for. A lane slot is one lane for one step.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          Warp width
          <select value={key} aria-label="warp width" onChange={(e) => setKey(e.target.value)}>
            {widths.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
        <label className="field">
          <span>
            After step <b data-value="t">{t}</b>
          </span>
          <input type="range" min={0} max={bk.finishesAt} value={t} aria-label="Steps elapsed" onChange={(e) => setT(Number(e.target.value))} style={{ width: '14rem' }} />
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr)' }}>
        <LaneGrid v={ks} upto={t} title="Kogge-Stone" />
        <LaneGrid v={bk} upto={t} title="Brent-Kung" />
      </div>
      <div style={{ overflowX: 'auto', marginTop: '0.8rem' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-part="warp-table">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>{W} lanes</th>
              <th className="num">additions</th>
              <th className="num">steps</th>
              <th className="num">lane slots</th>
              <th className="num">idle slots</th>
              <th className="num">slots spent by step {t}</th>
            </tr>
          </thead>
          <tbody>
            {both.map((v) => (
              <tr key={v.id} data-row={v.id} data-adds={v.additions} data-steps={v.finishesAt} data-slots={v.slots} data-idle={v.idleSlots}>
                <td style={{ paddingLeft: '0.6rem' }}>{v.id === 'kogge-stone' ? 'Kogge-Stone' : 'Brent-Kung'}</td>
                <td className="num">{v.additions}</td>
                <td className="num">{v.finishesAt}</td>
                <td className="num">{v.slots}</td>
                <td className="num">{v.idleSlots}</td>
                <td className="num" data-cell="spent">{Math.min(t, v.finishesAt) * W}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="narrate" role="status" style={{ margin: '0.8rem 0 0' }} data-readout="warp">
        Brent-Kung saves {ks.additions - bk.additions} additions on {W} lanes, but it issues {bk.finishesAt} steps where
        Kogge-Stone issues {ks.finishesAt}, so the warp spends {bk.slots} lane slots on it against {ks.slots}. Of Brent-Kung's
        slots, {bk.idleSlots} are idle lanes.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const EX6_LEVEL1 = run(brentKung(8), EX6)[upSteps(brentKung(8)) + 1];
const BK1024 = brentKung(1024);

export const questionsS1110: Record<string, Question> = {
  'ch11/11.10/q-ex6': {
    kind: 'numeric',
    id: 'ch11/11.10/q-ex6',
    prompt: (
      <>
        Book exercise 6: Brent-Kung on <code>4 6 7 1 2 8 5 2</code>. What does position 5 hold after the first level of the
        reverse tree?
      </>
    ),
    answer: EX6_LEVEL1[5],
    explain: (
      <p>
        The reduction tree gives 4 10 7 8 2 10 5 7, then 4 10 7 18 2 10 5 17, then 4 10 7 18 2 10 5 35. For 8 elements the
        reverse tree's first level works 2 apart: position 5 adds position 3, 10 + 18 = {EX6_LEVEL1[5]}. The last level adds
        1, 3, 5 into 2, 4, 6: 4 10 17 18 20 28 33 35.
      </p>
    ),
    source: 'ch. 11 exercise 6, p. 286 (the answer is ours)',
  },
  'ch11/11.10/q-ops': {
    kind: 'numeric',
    id: 'ch11/11.10/q-ops',
    prompt: <>How many additions does Brent-Kung perform on a segment of 1,024 elements?</>,
    answer: additions(BK1024),
    unit: 'additions',
    explain: (
      <p>
        2N − 2 − log₂N = 2048 − 2 − 10 = {additions(BK1024)}: 1,023 in the reduction tree and 1,013 in the reverse tree.
        Kogge-Stone would do {closedForm('kogge-stone', 1024).toLocaleString('en-US')}.
      </p>
    ),
    source: 'ch. 11 §11.10, p. 284',
  },
  'ch11/11.10/q-steps': {
    kind: 'numeric',
    id: 'ch11/11.10/q-steps',
    prompt: <>And in how many steps, with enough units for every adder of a step?</>,
    answer: BK1024.steps.length,
    unit: 'steps',
    explain: (
      <p>
        log₂N = 10 in the reduction tree and log₂N − 1 = 9 in the reverse tree: {BK1024.steps.length}, against Kogge-Stone's{' '}
        {span('kogge-stone', 1024)}. For 16 elements it is 4 + 3 = 7, the seven rows of Fig. 11.19.
      </p>
    ),
    source: 'ch. 11 §11.10, pp. 282–284 (the count 2·log₂N − 1 is ours, from Fig. 11.19)',
  },
  'ch11/11.10/q-needs': {
    kind: 'numeric',
    id: 'ch11/11.10/q-needs',
    prompt: (
      <>
        N = 16. After the reduction tree, position 13 holds x12..x13. From which position does it receive the partial sum it
        still lacks?
      </>
    ),
    answer: needsChain(reverseLevels(16)[1].ranges, 13)[0],
    unit: '',
    explain: (
      <p>
        It lacks x0..x11. After level 1 of the reverse tree position 11 holds exactly that (7 was added into it), so at level
        2 position 13 adds position 11, two places to its left. Before level 1 it would have needed two pieces, 11 (x8..x11)
        and 7 (x0..x7).
      </p>
    ),
    source: 'ch. 11 §11.10, pp. 283–284, Fig. 11.20',
  },
  'ch11/11.10/q-warp': {
    kind: 'choice',
    id: 'ch11/11.10/q-warp',
    prompt: <>After coarsening, why does the book keep Kogge-Stone for the warp-level scan, despite its extra work?</>,
    options: [
      { label: 'Brent-Kung cannot be written with shuffles.', why: 'Both networks only ever read a value a power of two to the left, which a shuffle-up delivers.' },
      {
        label: 'Inside a warp the lanes Brent-Kung leaves idle are occupied anyway, so its savings buy nothing while its extra steps cost time.',
        correct: true,
      },
      { label: 'Kogge-Stone does fewer additions within one warp.', why: 'It does more: 129 against 57 for 32 lanes.' },
      { label: 'Brent-Kung needs a barrier per step and Kogge-Stone does not.', why: 'Inside a warp neither uses a block barrier.' },
    ],
    explain: (
      <p>
        By then most of the work is each thread's sequential scan, and the parallel scan is a small part of the run time.
        In a warp every step is issued for all the lanes. The additions Brent-Kung avoids turn into lanes that sit idle but
        still use the warp's execution resources, and it needs 2·log₂N − 1 steps where Kogge-Stone needs log₂N. Use the
        lanes and finish sooner.
      </p>
    ),
    source: 'ch. 11 §11.10, pp. 284–285',
  },
  'ch11/11.10/q-ex7': {
    kind: 'choice',
    id: 'ch11/11.10/q-ex7',
    prompt: (
      <>
        Exercise 7: a block-level Brent-Kung kernel updates <code>buffer_s[idx] += buffer_s[idx - stride]</code> (or its
        reverse-tree form) for the active positions of each step, in place. Why is one barrier per iteration enough, when
        Fig. 11.3's in-place Kogge-Stone needed two?
      </>
    ),
    options: [
      { label: 'Brent-Kung has fewer steps, so it can afford to skip one.', why: 'It has more steps, and skipping a needed barrier is never affordable.' },
      {
        label: 'In each Brent-Kung step no position written by one thread is read by another, so nothing is overwritten before it is read; only the dependence between steps needs a barrier.',
        correct: true,
      },
      { label: 'Shared memory serialises writes to the same bank.', why: 'Bank behaviour is about speed, not about the order of reads and writes.' },
      { label: 'Only one thread is active in each step.', why: 'The first step of the reduction tree has N/2 active threads.' },
    ],
    explain: (
      <p>
        The second barrier of Fig. 11.3 enforces a write-after-read: a Kogge-Stone thread overwrites a position that another
        thread of the same step still has to read. In a Brent-Kung step the reduction tree writes positions 2·stride·k − 1 and
        reads those minus stride; the reverse tree writes those plus stride. The sets never meet, as the stepper's "read and
        write" line shows, so the only dependence left is the true one between consecutive steps.
      </p>
    ),
    source: 'ch. 11 §11.2, pp. 257–258; §11.3, pp. 258–259; exercise 7, p. 286 (the answer is ours)',
  },
};

// ---------------------------------------------------------------- the section
export function S1110() {
  const q = questionsS1110;
  const cu = useLabFile(SNIPPET);
  const loaded = cu && !cu.startsWith('// could not load');
  const barrierLines = loaded ? cu.split('\n').flatMap((l, i) => (l.includes('__syncthreads();') ? [i + 1] : [])) : [];
  return (
    <>
      <p>
        Every kernel in this chapter has used Kogge-Stone, and section 11.5 counted what that costs: N·log₂N − (N − 1)
        additions where the loop needs N − 1. This section presents the other classic network, Brent-Kung, which brings
        the count down to about 2N at the price of more steps. The book includes it for completeness and says so: after
        section 11.6's coarsening, Kogge-Stone's extra work stopped being a major concern.
      </p>
      <Source section="11.10" pages="282" />
      <p>
        Brent-Kung, like Kogge-Stone, began as a design for adder circuits (the book's reference [8]). It starts from a
        reduction tree, which computes the total of N values in log₂N steps and, on the way, produces sub-sums that some of
        the scan's outputs could use. Kogge-Stone runs a tree for every output and shares what it can between them.
        Brent-Kung runs one tree and then hands its sub-sums to the positions that need them.
      </p>

      <h2>Phase 1: the reduction tree</h2>
      <p>
        The first phase is <a href={href('ch10', '10.2')}>chapter 10's reduction tree</a>, computed in place. On 16
        elements it takes 4 steps. Step 1 updates the positions whose index has the form 2n − 1 (1, 3, 5, …, 15), each adding
        its left neighbour. Step 2 updates the positions 4n − 1 (3, 7, 11, 15), each adding the position 2 to its left. Step
        3 updates 8n − 1, from 4 to the left, and step 4 position 15 alone. That is 8 + 4 + 2 + 1 = 15 additions, and N − 1
        in general.
      </p>
      <Source section="11.10" pages="282" figs="Fig. 11.19, top" />
      <p>
        Every position now holds a contiguous run of inputs ending at itself, x<sub>i</sub>..x<sub>j</sub>. Positions 0, 1,
        3, 7 and 15 hold everything from x0: they are final. The others hold a piece of their answer.
      </p>

      <h2>Phase 2: the reverse tree</h2>
      <p>
        Look at what each position is missing. Position 14 holds only x14. It needs x0..x13, which the tree has left in three
        pieces: x12..x13 at position 13, x8..x11 at position 11 and x0..x7 at position 7. Two facts make the second phase
        possible. No position needs more than log₂N − 1 pieces, and the pieces sit a power of two apart: 14 to 13 is 1, 13
        to 11 is 2, 11 to 7 is 4.
      </p>
      <Source section="11.10" pages="283–284" figs="Fig. 11.20" />
      <p>
        So the reverse tree works from the largest distance to the smallest. Level 1 adds position 7 into position 11, 4
        apart, and 11 is final. Level 2 adds 3, 7 and 11 into 5, 9 and 13, 2 apart. Level 3 adds every odd position into the
        even position to its right. For 16 elements that is 1 + 3 + 7 = 11 more additions in 3 more steps.
      </p>
      <Aside>
        <p>
          Slips, pp. 283–284: the reduction phase is called the "bottom portion" of Fig. 11.19 (it is the top), and two-digit
          subscripts lower only their first digit ("x₁4" is x14).
        </p>
      </Aside>
      <p>
        Step both networks on the same input below. Exercise 6's eight values let you check the arithmetic by hand; Fig.
        11.6's sixteen give Fig. 11.19 and 11.20 exactly. The table under the networks is Fig. 11.20, recomputed from the
        ranges the cells hold.
      </p>
      <Simplification>
        <p>
          Every adder of a step reads the values from before the step, as in a circuit. The stepper counts additions and
          steps; it has no notion of time per step, memory or barriers.
        </p>
      </Simplification>
      <Stepper />
      <p>
        By the book's own legend, position 9 (holding x8..x9) and position 11 (x8..x11) each need one more partial sum
        before the reverse tree starts, the one at position 7. Fig. 11.20 shades both dark grey, the colour for two, and
        position 9 stays dark grey in the Level 1 row. Every other cell matches. Switch the shading to "as printed" to see
        the three cells.
      </p>
      <Aside>
        <p>Fig. 11.20 shading, p. 284: our reading of the printed figure against its legend on p. 283.</p>
      </Aside>
      <Ask q={q['ch11/11.10/q-ex6']} label="Work it out" />
      <Ask q={q['ch11/11.10/q-needs']} label="Work it out" />

      <h2>Counting the work</h2>
      <p>
        The reverse tree adds (2 − 1) + (4 − 1) + … + (N/4 − 1) + (N/2 − 1) = N − 1 − log₂N. With the reduction tree's N − 1
        that makes 2N − 2 − log₂N additions: 26 for 16 elements, against Kogge-Stone's 49. That is O(N), the order of the
        loop, where Kogge-Stone is O(N·log₂N). The price is steps: log₂N in the reduction tree and log₂N − 1 in the reverse
        tree, 2·log₂N − 1 in all, against Kogge-Stone's log₂N.
      </p>
      <Source section="11.10" pages="284" />
      <Ask q={q['ch11/11.10/q-ops']} label="Work it out" />
      <Ask q={q['ch11/11.10/q-steps']} label="Work it out" />
      <p>
        The animation runs both phases once on sixteen positions, then moves into a single warp, where the comparison turns
        around.
      </p>
      <BkSweepsScene />

      <h2>Why Kogge-Stone still wins inside a warp</h2>
      <p>
        Recall what coarsening did in section 11.6: each thread scans its own run of elements sequentially, and the only
        parallel scan left is across the lanes of a warp, a small part of the run time. There, the book reports, Kogge-Stone
        performs better in practice. A warp executes each step for all its lanes. The additions Brent-Kung avoids do not
        vanish; they become lanes that sit idle in a warp that is still executing and still use its execution resources.
        Better to put those lanes to work, as Kogge-Stone does, and finish in fewer steps.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A9, B7, C4; ch11-facts A4)">
        <p>
          NVIDIA and AMD both document masked lanes that still go through the warp's instruction. Apple documents no
          execution model for inactive SIMD-group threads. CUB's <code>BlockScan</code> docs describe the warp step of its
          warp-scans variant as "a shallow, but inefficient warp-synchronous Kogge-Stone style scan".
        </p>
      </Today>
      <p>Try both warp widths. The warp issues every step for all its lanes, idle or not.</p>
      <Simplification>
        <p>
          One step is one issue of the addition for the whole warp. Shuffles, comparisons and the cost of an idle lane beyond
          "it occupies the warp" are not modelled; NVIDIA warns that hardware may treat masked lanes in ways a program cannot
          see.
        </p>
      </Simplification>
      <WarpExplorer />
      <Ask q={q['ch11/11.10/q-warp']} />

      <h2>Exercise 7: a block-level kernel</h2>
      <p>
        The book leaves a Brent-Kung kernel to the reader. Here is ours. A block scans a segment of <code>SEG_SIZE</code>{' '}
        elements with <code>SEG_SIZE/2</code> threads, since the widest step, the first of the reduction tree, has that many
        additions; each thread loads and stores two elements. In both loops thread t performs the t-th addition of the step,
        at <code>idx = (t + 1)·2·stride − 1</code>, so the active threads are always the lowest-numbered ones and whole warps
        drop out together as the tree narrows (the lesson of <a href={href('ch10', '10.4')}>section 10.4</a>).
      </p>
      <Beyond>
        <p>
          The kernel is ours; the book prints no answer. Built from Fig. 11.19 and checked against the stepper's model on
          the course's emulators, including a version without the loop barriers that goes wrong.
        </p>
      </Beyond>
      {cu ? (
        <Code
          tabs={[{ label: 'CUDA', code: cu, tiers: TIERS, status: 'emulator-checked and front-end-checked; unverified on CUDA hardware' }]}
          highlight={barrierLines}
          caption={
            <>
              Loaded from <code>labs/{SNIPPET}</code>. The highlighted lines are its barriers: one at the top of each
              iteration of both loops, and one before the stores. It uses only shared memory, barriers and thread indices,
              which the HIP build spells the same way; it has not been compiled for HIP.
            </>
          }
        />
      ) : null}
      <p>
        Compare its barriers with Fig. 11.3's. That kernel needed two per iteration, because a Kogge-Stone thread overwrites a
        position that another thread of the same step has yet to read. The stepper's "read and write" line shows Kogge-Stone
        doing exactly that in every step but its last, and Brent-Kung never.
      </p>
      <Source section="11.10" pages="285" figs="exercise 7, p. 286; Fig. 11.3, pp. 256–258" />
      <Ask q={q['ch11/11.10/q-ex7']} />

      <h2>A design space, not two algorithms</h2>
      <p>
        Kogge-Stone and Brent-Kung are two points among many ways to wire a prefix sum, each trading additions against steps
        and against how evenly the work spreads over lanes. PyTorch's CUDA scan along the last dimension, the
        batch-by-vocabulary case, labels its parallel step "Sklansky method", a third network, and its comment links the
        decoupled look-back report for the diagram. On Metal the warp-level step is one call,{' '}
        <code>simd_prefix_inclusive_sum</code>, and Apple does not document which network it uses or in what order it adds.
      </p>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md D4, C1)">
        <p>
          PyTorch <code>ScanUtils.cuh</code>: "// Parallel reduction with Sklansky method", one <code>__syncthreads()</code>{' '}
          per step. Metal: the summation order of the SIMD-group prefix functions is undocumented.
        </p>
      </Today>
      <p>
        That completes the chapter's algorithms. The <a href={href('ch11', 'labs')}>labs</a> put the block scans on your own
        GPU, and the <a href={href('ch11', 'wrap')}>summary</a> maps the exercises, including 6 and 7 from this page.
      </p>
    </>
  );
}
