import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  ACC_FORMAT,
  ACC_LABEL,
  BOOK_FOUR,
  BUG_HUNT,
  DATA,
  EX4_FORMATS,
  EX4_VALUES,
  METHODS,
  OUR_FOUR,
  TOY_VALUES,
  VERDICT_LABEL,
  absSum,
  bugHunt,
  bugHuntValues,
  challengeHolds,
  chunkSweep,
  ex4Order,
  generate,
  measure,
  reference,
  runningTrace,
  show,
  sumSequential,
  toySum,
  treeTrace,
  ulpText,
  pct,
  type Acc,
  type ChunkPoint,
  type DataKind,
  type Ex4Order,
  type Method,
  type Pairing,
  type ToyRun,
  type Verdict,
} from '../../sim/fpsum';
import { maxFinite } from '../../sim/minifloat';
import { functionSource } from '../../sim/toolchain';
import { KahanScene } from './KahanScene';

const FACTS = 'notes/today/appA-facts.md';
const CHECKED = (items: string) => `2026-11-20 (${FACTS}, ${items})`;

// ================================================================ four values, two orders (p. 593)
const toyOptions = TOY_VALUES;

function AddRow({ a, k, shown }: { a: ToyRun['adds'][number]; k: number; shown: boolean }) {
  if (!shown) return <li className="muted" data-add={k}>·</li>;
  const exact = a.rounded === a.exact;
  return (
    <li data-add={k} data-rounded={a.rounded} data-lost={a.lost}>
      {show(a.left)} + {show(a.right)} = {show(a.exact)}{' '}
      {exact ? (
        <span className="muted">exactly</span>
      ) : (
        <>
          → stored <b>{show(a.rounded)}</b>{' '}
          <span className="t-fail">({a.lost > 0 ? `${show(a.lost)} lost` : `${show(-a.lost)} gained`}{a.saturated ? ', past 3.5: clamped' : ''})</span>
        </>
      )}
    </li>
  );
}

function FourValues() {
  const [vals, setVals] = useState<number[]>(BOOK_FOUR);
  const [n, setN] = useState(0);
  const [tries, setTries] = useState(0);
  const [verdict, setVerdict] = useState<null | 'yes' | 'no'>(null);
  const seq = toySum(vals, 'sequential');
  const tree = toySum(vals, 'tree');
  const set = (i: number, v: number) => {
    setVals(vals.map((x, j) => (j === i ? v : x)));
    setN(0);
    setVerdict(null);
  };
  const check = () => {
    setN(3);
    if (challengeHolds(vals)) setVerdict('yes');
    else {
      setVerdict('no');
      setTries(tries + 1);
    }
  };
  const better = seq.error === tree.error ? 'Both orders are equally accurate here.' : seq.error < tree.error ? 'The loop is more accurate here.' : 'The tree is more accurate here.';
  return (
    <Bench
      title="Four values, two orders"
      task="Step through the book's example, then change the values until the tree is the worse order."
      foot="Every addition is rounded to the book's 5-bit format (Fig. A.4, with denormals), to nearest with ties to even. Its largest value is 3.5; it has no infinity, so a sum beyond 3.5 is clamped."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        {vals.map((v, i) => (
          <label className="field" key={i}>
            {'abcd'[i]}
            <select aria-label={`value ${'abcd'[i]}`} value={v} onChange={(e) => set(i, Number(e.target.value))}>
              {toyOptions.map((o) => (
                <option key={o} value={o}>{show(o)}</option>
              ))}
            </select>
          </label>
        ))}
        <button onClick={() => { setVals(BOOK_FOUR); setN(0); setVerdict(null); }}>The book's values</button>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setN(0)} disabled={n === 0}>Restart</button>
        <button className="primary" onClick={() => setN(Math.min(3, n + 1))} disabled={n >= 3}>Step</button>
        <span className="muted" aria-live="polite">addition {n} of 3 in each order</span>
      </div>
      <div className="cols">
        <div className="panel" data-order="sequential">
          <b>In order: ((a + b) + c) + d</b>
          <ol style={{ margin: '0.4rem 0 0', paddingLeft: '1.3rem' }}>
            {seq.adds.map((a, k) => <AddRow key={k} a={a} k={k} shown={k < n} />)}
          </ol>
        </div>
        <div className="panel" data-order="tree">
          <b>As a tree: (a + b) + (c + d)</b>
          <ol style={{ margin: '0.4rem 0 0', paddingLeft: '1.3rem' }}>
            {tree.adds.map((a, k) => <AddRow key={k} a={a} k={k} shown={k < n} />)}
          </ol>
        </div>
      </div>
      {n >= 3 ? (
        <p data-summary style={{ margin: '0.8rem 0 0' }}>
          Exact sum <b>{show(seq.exact)}</b>. In order <b data-result="sequential">{show(seq.result)}</b> (off by {show(seq.error)}), as a tree{' '}
          <b data-result="tree">{show(tree.result)}</b> (off by {show(tree.error)}). {better}
        </p>
      ) : null}
      <div className="row" style={{ marginTop: '0.9rem' }}>
        <button onClick={check}>Check: is the tree worse?</button>
        {tries >= 3 && verdict !== 'yes' ? (
          <button onClick={() => { setVals(OUR_FOUR); setN(3); setVerdict('yes'); }}>Show one</button>
        ) : null}
        {verdict === 'yes' ? (
          <span className="chip ok" data-challenge="yes">Yes: the tree is the less accurate order for these values.</span>
        ) : verdict === 'no' ? (
          <span className="chip fail" data-challenge="no">
            {seq.saturated || tree.saturated ? 'A sum went past 3.5; keep every partial sum in range.' : 'Not this time: the tree is at least as accurate.'}
          </span>
        ) : null}
      </div>
    </Bench>
  );
}

// ================================================================ thread to data (Exercise 4)
const ORDERS: Ex4Order[] = ['ascending', 'descending', 'shuffled'];

function ThreadToData() {
  const [acc, setAcc] = useState<Acc>('fp16');
  const [order, setOrder] = useState<Ex4Order>('ascending');
  const [pairing, setPairing] = useState<Pairing>('convergent');
  const [base, setBase] = useState<number[]>(EX4_VALUES.fp16);
  const [text, setText] = useState(EX4_VALUES.fp16.join(', '));
  const [err, setErr] = useState('');
  const [k, setK] = useState(0);
  const values = useMemo(() => ex4Order(base, order), [base, order]);
  const trace = useMemo(() => treeTrace(values, pairing, acc), [values, pairing, acc]);
  const other = useMemo(() => treeTrace(values, pairing === 'convergent' ? 'neighbours' : 'convergent', acc), [values, pairing, acc]);
  useEffect(() => setK(0), [values, pairing, acc]);
  const total = trace.steps.length;
  const cur = k > 0 ? trace.steps[k - 1] : undefined;
  const cells = cur ? cur.cells : trace.input;
  const level = cur ? cur.level : 0;
  const active = new Set(trace.levels[level].active);
  const lostSoFar = trace.steps.slice(0, k).reduce((s, x) => s + (Number.isFinite(x.lost) ? x.lost : 0), 0);
  const pickFormat = (a: Acc) => {
    setAcc(a);
    setBase(EX4_VALUES[a as 'fp16' | 'bf16']);
    setText(EX4_VALUES[a as 'fp16' | 'bf16'].join(', '));
    setErr('');
  };
  const apply = () => {
    const parts = text.split(/[\s,]+/).filter(Boolean).map(Number);
    if (parts.length !== 16 || parts.some((x) => !Number.isFinite(x))) {
      setErr(`Give exactly 16 numbers (you gave ${parts.length}).`);
      return;
    }
    setErr('');
    setBase(parts);
  };
  const endOfLevel = () => {
    let j = k;
    while (j < total && (j === k || trace.steps[j].level === trace.steps[k].level)) j++;
    setK(Math.min(total, j));
  };
  const threadOf = (i: number) => (pairing === 'convergent' ? (i < 8 ? i : null) : i % 2 === 0 ? i / 2 : null);
  const cellCls = (i: number) => {
    if (!cur) return '';
    if (i === cur.dst) return cur.lost !== 0 ? 'bad' : 'hot';
    if (i === cur.src) return 'on';
    return '';
  };
  const describeStep = () => {
    if (!cur)
      return `${trace.input.length} values in ${ACC_LABEL[acc]}, ${order}. Eight threads; ${pairing === 'convergent' ? 'thread i owns cell i and first adds cell i + 8 (Fig. 10.8)' : 'thread i owns cell 2i and first adds cell 2i + 1 (Fig. 10.6)'}. Press Step.`;
    const lost = cur.lost === 0 ? 'nothing lost' : Number.isFinite(cur.lost) ? `${show(cur.lost)} lost` : 'overflow';
    return `Level ${cur.level + 1} (stride ${cur.stride}): thread ${cur.thread} adds cell ${cur.src} to cell ${cur.dst}: ${show(cur.a)} + ${show(cur.b)} = ${show(cur.exact)}, stored ${show(cur.rounded)}; ${lost}.`;
  };
  return (
    <Bench
      title="Thread to data, with rounding"
      task="On ascending data, step the convergent tree to the end; then switch to neighbours and compare."
      foot={
        <>
          The array lives in global memory and is reduced in place, as in chapter 10. Each addition is rounded once to the
          chosen format. At each step the cell a thread writes is solid and the cell it reads is shaded; the written cell
          turns red when the rounded sum dropped part of the exact one.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Format">
          {EX4_FORMATS.map((a) => (
            <button key={a} aria-pressed={acc === a} onClick={() => pickFormat(a)}>{ACC_LABEL[a]}</button>
          ))}
        </span>
        <label className="field">
          Order
          <select aria-label="Order" value={order} onChange={(e) => setOrder(e.target.value as Ex4Order)}>
            {ORDERS.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label="Pairing">
          <button aria-pressed={pairing === 'convergent'} onClick={() => setPairing('convergent')}>convergent (i, i + stride)</button>
          <button aria-pressed={pairing === 'neighbours'} onClick={() => setPairing('neighbours')}>neighbours (2i, 2i + stride)</button>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ flex: '1 1 18rem' }}>
          The 16 values (any order; the Order menu arranges them)
          <input
            type="text"
            aria-label="values"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && apply()}
            style={{ font: 'inherit', fontFamily: 'var(--sans)', padding: '0.3rem 0.4rem', width: '100%' }}
          />
        </label>
        <button onClick={apply}>Use these values</button>
        {err ? <span className="t-fail">{err}</span> : null}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setK(0)} disabled={k === 0}>Restart</button>
        <button onClick={() => setK(k - 1)} disabled={k === 0}>Back</button>
        <button className="primary" onClick={() => setK(k + 1)} disabled={k >= total}>Step</button>
        <button onClick={endOfLevel} disabled={k >= total}>Finish this level</button>
        <button onClick={() => setK(total)} disabled={k >= total}>Run to the end</button>
        <span className="muted" aria-live="polite">step {k} of {total}</span>
      </div>
      <div className={'narrate' + (cur && cur.lost !== 0 ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {describeStep()}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <div className="mx global" style={{ display: 'grid', gridTemplateColumns: 'repeat(8, minmax(2.2rem, 3.4rem))', rowGap: 4 }} aria-label="Cells">
          {cells.map((v, i) => {
            const t = threadOf(i);
            return (
              <span key={i} className={cellCls(i)} data-cell={i} data-value={v} style={{ height: '2.3rem' }}>
                {t != null ? (
                  <span
                    data-thread={t}
                    data-active={active.has(t) ? '1' : '0'}
                    style={{
                      position: 'absolute', left: 3, top: 1, fontSize: '0.56rem', lineHeight: 1.2,
                      opacity: active.has(t) ? 1 : 0.35,
                      fontWeight: cur?.thread === t ? 700 : 400,
                      textDecoration: cur?.thread === t ? 'underline' : 'none',
                    }}
                  >
                    t{t}
                  </span>
                ) : null}
                {show(v, 5)}
                <sub>{i}</sub>
              </span>
            );
          })}
        </div>
      </div>
      <p className="muted" style={{ margin: '0.5rem 0 0' }}>
        Top row cells 0 to 7, bottom row 8 to 15; t3 marks the cell thread 3 owns, faint when the thread is idle at this
        level. Lost so far: <b data-lost-so-far={lostSoFar}>{show(lostSoFar)}</b>.
      </p>
      {k >= total ? (
        <table className="data" style={{ marginTop: '0.8rem', background: 'var(--page)' }} data-summary>
          <thead>
            <tr><th>Pairing</th><th>Result</th><th>Exact</th><th>Off by</th></tr>
          </thead>
          <tbody>
            {[trace, other].map((t) => (
              <tr key={t.pairing} data-pairing={t.pairing}>
                <td>{t.pairing === 'convergent' ? 'convergent (Fig. 10.8)' : 'neighbours (Fig. 10.6)'}</td>
                <td data-k="result">{show(t.result)}</td>
                <td>{show(t.exact)}</td>
                <td data-k="error">{show(t.error)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </Bench>
  );
}

// ================================================================ the summation explorer
const LOGN_MIN = 4;
const LOGN_MAX = 20;
const ACCS: Acc[] = ['fp32', 'fp16', 'bf16'];

function RunningPlot({ points, acc }: { points: { i: number; acc: number; exact: number }[]; acc: Acc }) {
  const W = 360;
  const H = 210;
  const L = 52;
  const B = 28;
  const n = points[points.length - 1].i;
  const finite = points.map((p) => p.acc).filter(Number.isFinite);
  const top = Math.max(...points.map((p) => p.exact), ...finite, 1e-30);
  const bottom = Math.min(0, ...points.map((p) => p.exact), ...finite);
  const x = (i: number) => L + (i / n) * (W - L - 8);
  const y = (v: number) => 10 + (H - B - 10) * (1 - (Math.min(v, top) - bottom) / (top - bottom || 1));
  const line = (pick: (p: (typeof points)[number]) => number) =>
    points.map((p) => `${x(p.i).toFixed(1)},${y(Number.isFinite(pick(p)) ? pick(p) : top).toFixed(1)}`).join(' ');
  const last = points[points.length - 1];
  const overflowAt = points.find((p) => !Number.isFinite(p.acc));
  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Running sum of the loop against the exact running sum" style={{ width: '100%', display: 'block' }} data-final-acc={last.acc} data-final-exact={last.exact}>
        <line x1={L} x2={L} y1={10} y2={H - B} stroke="var(--rule)" />
        <line x1={L} x2={W - 8} y1={H - B} y2={H - B} stroke="var(--rule)" />
        <text x={L - 6} y={14} fontSize="10.5" textAnchor="end">{show(top, 3)}</text>
        <text x={L - 6} y={H - B} fontSize="10.5" textAnchor="end">{show(bottom, 3)}</text>
        <text x={L} y={H - 10} fontSize="10.5">0</text>
        <text x={W - 8} y={H - 10} fontSize="10.5" textAnchor="end">{n} values added</text>
        <polyline points={line((p) => p.exact)} fill="none" stroke="var(--ink-3)" strokeWidth="1.5" strokeDasharray="5 4" />
        <polyline points={line((p) => p.acc)} fill="none" stroke="var(--ink)" strokeWidth="2" />
        <g fontSize="10.5">
          <rect x={L + 4} y={8} width={170} height={38} rx="3" fill="var(--bench)" opacity="0.9" />
          <line x1={L + 10} x2={L + 30} y1={20} y2={20} stroke="var(--ink)" strokeWidth="2" />
          <text x={L + 34} y={23}>{ACC_LABEL[acc]} loop: {Number.isFinite(last.acc) ? show(last.acc, 6) : 'overflow'}</text>
          <line x1={L + 10} x2={L + 30} y1={36} y2={36} stroke="var(--ink-3)" strokeWidth="1.5" strokeDasharray="5 4" />
          <text x={L + 34} y={39} style={{ fill: 'var(--ink-2)' }}>exact: {show(last.exact, 6)}</text>
        </g>
        {overflowAt ? <text x={x(overflowAt.i) + 4} y={56} fontSize="10.5" style={{ fill: 'var(--fail)' }}>overflow here</text> : null}
      </svg>
      <figcaption className="muted" style={{ fontSize: '0.85rem' }}>The sequential loop's running sum (solid) against the exact running sum (dashed).</figcaption>
    </figure>
  );
}

function ChunkPlot({ sweep, c, active }: { sweep: ChunkPoint[] | null; c: number; active: boolean }) {
  const W = 360;
  const H = 210;
  const L = 52;
  const B = 28;
  if (!sweep) return <p className="muted" data-sweep="pending">Computing the error at every c…</p>;
  const vals = sweep.flatMap((p) => [p.plain.ulps, p.kahan.ulps]).filter((v) => Number.isFinite(v) && v > 0);
  const lo = Math.floor(Math.log10(Math.max(1e-3, Math.min(...vals, 1))));
  const hi = Math.max(lo + 1, Math.ceil(Math.log10(Math.max(...vals, 1))));
  const maxK = Math.log2(sweep[sweep.length - 1].c) || 1;
  const x = (cc: number) => L + (Math.log2(cc) / maxK) * (W - L - 12);
  const y = (u: number) => {
    if (!Number.isFinite(u)) return 6;
    const v = Math.log10(Math.max(u, 10 ** lo));
    return 10 + (H - B - 10) * (1 - (v - lo) / (hi - lo));
  };
  const line = (pick: (p: ChunkPoint) => number) => sweep.map((p) => `${x(p.c).toFixed(1)},${y(pick(p)).toFixed(1)}`).join(' ');
  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Error against the number of values per thread" style={{ width: '100%', display: 'block' }} data-sweep="ready" data-points={JSON.stringify(sweep.map((p) => [p.c, p.plain.ulps, p.kahan.ulps]))}>
        <line x1={L} x2={L} y1={10} y2={H - B} stroke="var(--rule)" />
        <line x1={L} x2={W - 12} y1={H - B} y2={H - B} stroke="var(--rule)" />
        {Array.from({ length: hi - lo + 1 }, (_, j) => lo + j).map((e) => (
          <text key={e} x={L - 6} y={y(10 ** e) + 3} fontSize="10.5" textAnchor="end">{e >= -3 && e <= 4 ? String(10 ** e) : `1e${e}`}</text>
        ))}
        <text x={4} y={9} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>ULPs</text>
        <text x={L - 6} y={H - 10} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>c =</text>
        {sweep.filter((p, j) => j % 2 === 0 || j === sweep.length - 1).map((p) => (
          <text key={p.c} x={x(p.c)} y={H - 10} fontSize="10.5" textAnchor="middle">{p.c}</text>
        ))}
        {active && c <= sweep[sweep.length - 1].c ? <line x1={x(c)} x2={x(c)} y1={10} y2={H - B} stroke="var(--ink-2)" strokeDasharray="2 3" data-c={c} /> : null}
        <polyline points={line((p) => p.plain.ulps)} fill="none" stroke="var(--ink-3)" strokeWidth="2.5" />
        <polyline points={line((p) => p.kahan.ulps)} fill="none" stroke="var(--ink)" strokeWidth="2" />
        <g fontSize="10.5">
          <rect x={L + 4} y={8} width={250} height={18} rx="3" fill="var(--bench)" opacity="0.9" />
          <line x1={L + 10} x2={L + 30} y1={18} y2={18} stroke="var(--ink-3)" strokeWidth="2.5" />
          <text x={L + 34} y={21}>plain per thread</text>
          <line x1={L + 130} x2={L + 150} y1={18} y2={18} stroke="var(--ink)" strokeWidth="2" />
          <text x={L + 154} y={21}>Kahan per thread</text>
        </g>
        {sweep.some((p) => !Number.isFinite(p.plain.ulps) || !Number.isFinite(p.kahan.ulps)) ? (
          <text x={W - 14} y={36} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--fail)' }}>top edge: overflow</text>
        ) : null}
      </svg>
      <figcaption className="muted" style={{ fontSize: '0.85rem' }}>
        Error of the chunked sum against c, the values each thread adds before the tree (log scales; c = 1 is the plain tree).
      </figcaption>
    </figure>
  );
}

function Explorer() {
  const [kind, setKind] = useState<DataKind>('uniform');
  const [logn, setLogn] = useState(16);
  const [acc, setAcc] = useState<Acc>('fp32');
  const [method, setMethod] = useState<Method>('sequential');
  const [logc, setLogc] = useState(6);
  const n = 2 ** logn;
  const c = 2 ** logc;
  const chunked = method === 'chunked' || method === 'kahan-chunked';
  const data = useMemo(() => generate(kind, n), [kind, n]);
  const ref = useMemo(() => reference(data), [data]);
  const scale = useMemo(() => absSum(data), [data]);
  const e = useMemo(() => measure(data, { method, acc, chunk: c }, ref, scale), [data, method, acc, c, ref, scale]);
  const running = useMemo(() => runningTrace(data, acc), [data, acc]);
  const sweepKey = `${kind}|${n}|${acc}`;
  const [sweep, setSweep] = useState<{ key: string; points: ChunkPoint[] } | null>(null);
  useEffect(() => {
    // The sweep runs 13 chunk lengths twice over the data; let the rest of the bench paint first.
    const id = window.setTimeout(() => setSweep({ key: sweepKey, points: chunkSweep(data, acc) }), 30);
    return () => window.clearTimeout(id);
  }, [sweepKey, data, acc]);
  const points = sweep && sweep.key === sweepKey ? sweep.points : null;
  const fmt = ACC_FORMAT[acc];
  return (
    <Bench
      title="Summation order and accumulator width"
      task="Find the FP16 stall, then compare the loop, the tree and Kahan on each data set."
      foot={
        <>
          Data are seeded and stored in FP32; the reference is a compensated sum in binary64 (Neumaier's). Relative error is
          measured against the sum of |x|. ULPs are units in the last place of the accumulator format at the reference
          value. {ACC_LABEL[acc]}'s largest value is {show(maxFinite(fmt))}.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Data
          <select aria-label="Data" value={kind} onChange={(ev) => setKind(ev.target.value as DataKind)}>
            {DATA.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
          </select>
        </label>
        <label className="field">
          n = {n.toLocaleString('en-US')}
          <input type="range" aria-label="n (log2)" min={LOGN_MIN} max={LOGN_MAX} value={logn} onChange={(ev) => setLogn(Number(ev.target.value))} />
        </label>
        <span className="seg" role="group" aria-label="Accumulator">
          {ACCS.map((a) => (
            <button key={a} aria-pressed={acc === a} onClick={() => setAcc(a)}>{ACC_LABEL[a]}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Method
          <select aria-label="Method" value={method} onChange={(ev) => setMethod(ev.target.value as Method)}>
            {METHODS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        </label>
        {chunked ? (
          <label className="field">
            c = {c} values per thread ({Math.ceil(n / c).toLocaleString('en-US')} threads)
            <input type="range" aria-label="c (log2)" min={0} max={12} value={logc} onChange={(ev) => setLogc(Number(ev.target.value))} />
          </label>
        ) : null}
        <span className="muted">{DATA.find((d) => d.id === kind)!.note}</span>
      </div>
      <div className="stats" aria-label="Result" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(9rem, 1fr))', gap: '0.5rem 1rem', marginBottom: '1rem', fontFamily: 'var(--sans)', fontSize: '0.9rem' }}>
        {[
          { k: 'result', label: `Result (${ACC_LABEL[acc]})`, v: e.result, text: Number.isFinite(e.result) ? show(e.result, 9) : e.result > 0 ? '+inf (overflow)' : '−inf (overflow)' },
          { k: 'reference', label: 'Reference', v: e.reference, text: show(e.reference, 12) },
          { k: 'abs', label: 'Absolute error', v: e.abs, text: e.overflow ? 'overflow' : show(e.abs, 4) },
          { k: 'rel', label: 'Relative error', v: e.rel, text: pct(e.rel) },
          { k: 'ulps', label: `Error in ULPs of ${ACC_LABEL[acc]}`, v: e.ulps, text: ulpText(e.ulps) },
          { k: 'scale', label: 'Sum of |x|', v: scale, text: show(scale, 9) },
        ].map((r) => (
          <div key={r.k} className="panel" style={{ padding: '0.4rem 0.6rem' }}>
            <div className="muted" style={{ fontSize: '0.78rem' }}>{r.label}</div>
            <div data-k={r.k} data-v={String(r.v)} style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>{r.text}</div>
          </div>
        ))}
      </div>
      <div className="cols">
        <RunningPlot points={running} acc={acc} />
        <ChunkPlot sweep={points} c={c} active={chunked} />
      </div>
    </Bench>
  );
}

function HalfLoopReveal({ v }: { v: number }) {
  const u = useMemo(() => generate('uniform', 65536), []);
  const got = sumSequential(u, 'fp16');
  const exact = reference(u);
  const close = Math.abs(v - got) <= 0.1 * got;
  return (
    <>
      <p>
        The model's FP16 loop returns <b data-reveal="half-loop">{show(got)}</b>; the exact sum of these 65,536 values is{' '}
        {show(exact, 7)}. {close ? 'You saw the stall coming.' : v > 10000 ? 'Most people expect something near the true sum.' : ''}
      </p>
      <p>
        Below 2048 the running sum grows: from 1024 on, FP16's values are 1 apart, so each addition rounds up by one or
        down to nothing. At 2048 the spacing becomes 2. Every value below 1 is less than half of that and rounds away; a
        value of exactly 1 is a tie that goes back to 2048, whose last mantissa bit is 0. From then on the loop adds
        nothing, about 60,000 times. Set the explorer to FP16 and look at the left plot.
      </p>
    </>
  );
}

// ================================================================ Kahan, four ways
const VERDICTS: Verdict[] = ['kahan', 'plain', 'worse'];

function BugHunt() {
  const file = useLabFile('appA/snippets/kahan.h');
  const [marks, setMarks] = useState<Record<number, Verdict | undefined>>({});
  const [ran, setRan] = useState(false);
  const results = useMemo(() => (ran ? bugHunt(bugHuntValues(), 'fp32') : null), [ran]);
  const uniform = useMemo(() => (ran ? bugHunt(generate('uniform', 65536), 'fp32') : null), [ran]);
  const all = BUG_HUNT.every((b) => marks[b.version]);
  const right = results ? BUG_HUNT.filter((b, j) => marks[b.version] === results.runs[j].verdict).length : 0;
  return (
    <Bench
      title="Kahan, four ways"
      task="Mark each version before you run them: like Kahan, like the plain sum, or worse than plain?"
      foot={
        <>
          The code is <code>labs/appA/snippets/kahan.h</code>, loaded from the repository. A unit test compiles it with the
          host compiler and checks that every version gives this model's float, bit for bit. Data: 1.0, then 65,535 values
          of 2⁻²⁵, in FP32.
        </>
      }
    >
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 24rem), 1fr))' }}>
        {BUG_HUNT.map((b, j) => {
          const src = file ? functionSource(file, `version${b.version}`) : null;
          const run = results?.runs[j];
          const mark = marks[b.version];
          return (
            <div key={b.version} className="panel" data-version={b.version}>
              <b>Version {b.version}</b>
              {src ? <Code tabs={[{ label: `version${b.version}`, code: src.text.replace(/^\/\/ Version \d+\n/, '') }]} /> : <p className="muted">Loading labs/appA/snippets/kahan.h…</p>}
              <div className="opts" role="radiogroup" aria-label={`Version ${b.version}`} style={{ display: 'grid', gap: '0.2rem', marginTop: '0.4rem' }}>
                {VERDICTS.map((v) => (
                  <label key={v} className="check">
                    <input
                      type="radio"
                      name={`kahan-v${b.version}`}
                      checked={mark === v}
                      onChange={() => {
                        setMarks({ ...marks, [b.version]: v });
                        setRan(false);
                      }}
                    />
                    {VERDICT_LABEL[v]}
                  </label>
                ))}
              </div>
              {run ? (
                <p style={{ margin: '0.5rem 0 0' }} data-verdict={run.verdict} data-mark={mark === run.verdict ? 'right' : 'wrong'}>
                  <span className={'chip ' + (mark === run.verdict ? 'ok' : 'fail')}>{mark === run.verdict ? 'right' : 'not quite'}</span>{' '}
                  {VERDICT_LABEL[run.verdict]}: result {show(run.result.result, 9)}, off by {ulpText(run.result.ulps)} ULPs
                  {uniform ? <> (on uniform data: {ulpText(uniform.runs[j].result.ulps)} ULPs)</> : null}.
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="row" style={{ marginTop: '0.9rem' }}>
        <button className="primary" disabled={!all} onClick={() => setRan(true)}>Run the four versions</button>
        {results ? (
          <span data-score={right}>
            {right} of 4 right. The plain loop: result {show(results.plain.result, 9)}, off by {ulpText(results.plain.ulps)} ULPs
            {uniform ? <> ({ulpText(uniform.plain.ulps)} on uniform data)</> : null}.
          </span>
        ) : (
          <span className="muted">{all ? 'Ready.' : 'Mark all four first.'}</span>
        )}
      </div>
    </Bench>
  );
}

// ================================================================ questions
export const questionsSA6: Record<string, Question> = {
  'appA/A.6/q-ex4': {
    kind: 'choice',
    id: 'appA/A.6/q-ex4',
    prompt: (
      <>
        Exercise 4: a student's array is always sorted from smallest to largest. To avoid divergence he reduces it with
        chapter 10's convergent kernel, and the sum is less accurate than he expected. Why?
      </>
    ),
    options: [
      {
        label: 'Its first level adds element i + n/2 to element i, so small values meet large ones and vanish.',
        correct: true,
        why: 'Sorting put similar magnitudes side by side; this pairing reaches across the array and undoes that.',
      },
      {
        label: 'Without divergence a warp adds in a different order, and sorted data rounds badly in that order.',
        why: 'Divergence decides which threads run in an iteration, not which elements are paired.',
      },
      { label: 'Sorted data makes the tree deeper, so each value goes through more roundings.', why: 'The depth is log2 n for any data.' },
      { label: 'Sorting rounds every value once more before the reduction begins.', why: 'Sorting moves values; it does not change them.' },
    ],
    explain: (
      <p>
        The book's text points to "the algorithm of Fig. A.4", a table of representable numbers. Our reading is chapter 10's
        convergent kernel (Fig. 10.8), the one written to avoid divergence. Its first level pairs element i with element i
        + n/2: on ascending data, a small value with a larger one. The neighbour pairing of Fig. 10.6 adds like with like
        and keeps the benefit of sorting; in the stepper it is exact on the same data.
      </p>
    ),
    source: 'book Exercise 4, p. 600, with ch. 10 Figs. 10.6 and 10.8 (pp. 229–233); our model (fpsum.treeTrace)',
  },
  'appA/A.6/q-sign': {
    kind: 'choice',
    id: 'appA/A.6/q-sign',
    prompt: <>The data has both signs and wildly different magnitudes. How should you sort it before a sequential or tree sum?</>,
    options: [
      { label: 'By magnitude, smallest |x| first.', correct: true },
      { label: 'By value, ascending.', why: 'Then the largest negative values come first, and the small values in the middle meet a running sum that is already large.' },
      { label: 'By value, descending.', why: 'The largest positive values come first, and every small value meets a large running sum.' },
      { label: 'Sorting cannot help once there are two signs.', why: 'It still brings values of similar size together; only the key changes.' },
    ],
    explain: (
      <p>
        The book says signs must be taken into account (p. 594). What decides whether an addend disappears is its size
        relative to what it is added to, so the key is |x|. In the model, on the "cancelling" data, the loop sorted by
        magnitude is many times more accurate than the same loop sorted by value.
      </p>
    ),
    source: 'book §A.6, p. 594; our model (fpsum.test.ts, q-sign)',
  },
  'appA/A.6/q-half-tree': {
    kind: 'choice',
    id: 'appA/A.6/q-half-tree',
    prompt: (
      <>
        An FP16 accumulator sums 262,144 values drawn uniformly from [0, 1): the true sum is about 131,000, twice FP16's
        largest value. Once as a loop, once as a pairwise tree. What comes out?
      </>
    ),
    options: [
      { label: 'The loop returns 2048; the tree returns infinity.', correct: true },
      { label: 'Both return infinity.', why: 'The loop never gets near the top of the range: it stalls at 2048, long before.' },
      { label: 'Both return about 65,504, the largest value.', why: 'Round to nearest overflows to infinity rather than clamping.' },
      { label: 'The loop returns infinity; the tree about 131,000.', why: 'No FP16 value is anywhere near 131,000.' },
    ],
    explain: (
      <p>
        The loop stalls where FP16's spacing reaches 2, at 2048, and stays there: a quiet, wrong, finite answer. The tree
        keeps each partial sum accurate, so its last additions really do produce numbers above FP16's largest value,
        65,504, and round to nearest turns them into infinity (anything from 65,520 up). The tree's failure is loud; the
        loop's is not. Try it in the explorer with n = 2¹⁸.
      </p>
    ),
    source: 'our model (fpsum.test.ts, q-half-tree); FP16 range from appA-facts A2',
  },
  'appA/A.6/q-kahan16': {
    kind: 'choice',
    id: 'appA/A.6/q-kahan16',
    prompt: <>Summing 65,536 values drawn uniformly from [0, 1): which is more accurate, Kahan's loop in FP16 or a plain loop in FP32?</>,
    options: [
      { label: 'The plain FP32 loop.', correct: true },
      { label: 'Kahan in FP16.', why: 'Kahan recovers what each addition drops, but it cannot make FP16 hold more than 11 significant bits.' },
      { label: 'They are about the same.', why: 'In the model they differ by about two orders of magnitude.' },
    ],
    explain: (
      <p>
        In the model, Kahan in FP16 ends within half an FP16 ULP of the true sum, which is as good as FP16 can store it:
        about 32,656 against 32,664. The plain FP32 loop is about 0.08 off. Compensation fixes the error of the additions;
        the width of the accumulator sets the floor. Wider accumulator first, compensation second.
      </p>
    ),
    source: 'our model (fpsum.test.ts, q-kahan16)',
  },
  'appA/A.6/q-fastmath': {
    kind: 'choice',
    id: 'appA/A.6/q-fastmath',
    prompt: <>A Kahan kernel passes on the emulator and on HIP, but on Metal its result is no better than the plain sum. Why?</>,
    options: [
      {
        label: 'The default math mode lets the compiler reassociate, and c = (t − sum) − y became 0.',
        correct: true,
      },
      {
        label: 'Metal may round toward zero, and compensation cannot correct truncated additions.',
        why: 'A rounding mode changes how each addition rounds; it does not remove the statement that computes c. Here the compensation is gone altogether.',
      },
      { label: "Metal's float subtraction is not correctly rounded, so c holds noise, not the lost part.", why: 'Metal documents + and − as correctly rounded in every mode.' },
      { label: 'Metal flushes denormals to zero, and c is almost always a denormal.', why: 'c is about the size of the last bits of the sum, far above the denormal range for sums like these.' },
    ],
    explain: (
      <>
        <p>
          Both candidate default modes, fast and relaxed, allow reassociation, and Metal's specification warns that it
          breaks code relying on rounding, such as (x + 2⁵²) − 2⁵². Put <code>#pragma METAL fp math_mode(safe)</code> at
          the top of the kernel file. Clang, which HIP uses, does not reassociate unless told to, so the HIP build kept it.
        </p>
        <p>While this chapter was designed, the course's Apple GPU did exactly this: the default mode removed the compensation and the pragma restored it.</p>
      </>
    ),
    source: `${FACTS} C2, C3 (MSL spec 1.6.3, 8.4), B5 (Clang); measured while designing this chapter`,
  },
  'appA/A.6/q-coarse': {
    kind: 'choice',
    id: 'appA/A.6/q-coarse',
    prompt: (
      <>
        A row starts with one large value followed by a million small ones (the explorer's "one big, many small"). You
        raise the coarsening factor of a plain FP32 sum from 4 to 1024 values per thread. What happens to the error?
      </>
    ),
    options: [
      {
        label: 'It grows about in proportion to c; Kahan inside each thread brings it back to about one ULP.',
        correct: true,
        why: 'The thread that holds the large value adds about a thousand small ones to it and drops each.',
      },
      {
        label: 'Nothing changes: the tree at the end, not the per-thread loops, decides how accurate it is.',
        why: 'The tree only sees the per-thread sums; what a thread dropped is already gone.',
      },
      { label: 'It shrinks, because with fewer threads fewer partial sums go through the tree.', why: 'The tree is the accurate part; it is the longer loops that lose.' },
    ],
    explain: (
      <p>
        In the model, at 2²⁰ values, the error goes from about 3 ULPs at c = 4 to about 260 at c = 1024; with Kahan inside
        each thread it stays near 1. On random data that does not start with a large value the growth is much smaller. The
        cost of compensation: four additions per element instead of one, which moves the sum from 0.25 to 1 operation per
        byte, still far below what a memory-bound kernel needs to become compute-bound on the book's example device (about
        20). Lab A.1 measures both on yours.
      </p>
    ),
    source: 'ch. 10 §10.10 (pp. 246–248); our model (fpsum.test.ts, q-coarse); ch. 5 §5.1 for the book\'s example ratio',
  },
};

// ================================================================ the page
export function SA6() {
  return (
    <>
      <p>
        Section <a href={href('appA', 'A.4')}>A.4</a> ended with one addition losing at most half a ULP and a loop of
        additions losing everything it adds. This section is about the long sums where that happens, and the two things you
        control in a kernel: the order of the additions and the accumulator they go into.
      </p>

      <h2>Long sums are everywhere</h2>
      <p>
        The book's example of a long sum is the dot product inside matrix multiplication: every element of a GEMM's output
        is one. Mathematically addition is associative, so the order should not matter. In finite precision it changes the
        result.
      </p>
      <Source section="A.6" pages="593" />
      <p>
        One decoder step adds up many more: the softmax normaliser over a row of logits, the mean of squares a norm takes
        over the hidden dimension, the running total a top-p sampler builds over sorted probabilities. Each is a long sum
        in some accumulator format.
      </p>
      <Beyond>
        <p>The inference examples are ours. The book's is the matrix-multiplication dot product.</p>
      </Beyond>

      <h2>Two orders, two answers</h2>
      <p>
        The book uses the 5-bit format of section <a href={href('appA', 'A.2')}>A.2</a>, whose values between 2 and 3.5
        are 0.5 apart. Add 1, 1, 0.25 and 0.25; the exact sum is 2.5. In order, 1 + 1 = 2, then 2 + 0.25 = 2.25 has no
        representation and rounds back to 2, and so does the last addition: the small operands disappear, and the loop
        returns 2.0. As a tree, (1 + 1) + (0.25 + 0.25) = 2 + 0.5 = 2.5, exactly: the two small values combine first into
        something large enough to count.
      </p>
      <Source section="A.6" pages="593" />
      <p>
        The book adds that a nearby case makes the tree the less accurate order, and leaves it at that. Find one: change
        the values in the stepper until the tree loses.
      </p>
      <Source section="A.6" pages="594" />
      <FourValues />
      <p>
        After three tries the stepper offers a case of ours. Whatever you found, look at where the tree loses: one of its
        pairs adds a small value to a large one before the small values have had a chance to combine.
      </p>
      <Aside>
        <p>
          Our case, 0.25, 1, 0.25, 2, was found by a brute-force search while designing this page; the unit tests confirm
          it. The tree's 0.25 + 2 = 2.25 is a tie, and ties go to the even neighbour, 2 (section{' '}
          <a href={href('appA', 'A.4')}>A.4</a>).
        </p>
      </Aside>

      <h2>Neither answer is the sum</h2>
      <p>
        The tree is chapter 10's reduction tree, which is only allowed because addition is assumed associative. A GPU
        reduction and a CPU loop over the same floats therefore return different numbers, and in general neither is the
        exact sum.
      </p>
      <Source section="A.6" pages="593" />
      <p>
        Both are legitimate floating-point results; the difference is not a bug in either. What a test should check is
        that both are within a tolerance of a more accurate reference, scaled to the sum of |x|.
      </p>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md, A9)">
        <p>
          NVIDIA's guide makes the same point with its own three-value and dot-product examples, and adds: "The presence
          of differences does not necessarily mean the GPU's result is incorrect".
        </p>
      </Today>

      <h2>Exercise 4: thread to data, with rounding</h2>
      <p>
        The book's exercise 4: a student sums a large float array that is always sorted from smallest to largest. To avoid
        branch divergence he uses "the algorithm of Fig. A.4", and the result is less accurate than he expected. Fig. A.4
        is a table of representable numbers, so the reference is broken. We take the intended algorithm to be chapter
        10's convergent kernel (Fig. 10.8), the one designed to avoid divergence.
      </p>
      <Source pages="600" figs="Exercise 4; ch. 10 Fig. 10.8, p. 233" />
      <p>
        In that kernel thread i owns element i and adds element i + stride to it, with the stride starting at half the
        array and halving at each level. Before you step through it, work out what that pairing does to sorted data.
      </p>
      <Ask q={questionsSA6['appA/A.6/q-ex4']} />
      <p>
        Now check your answer. Step through the kernel in FP16 on the preset data (eight 1s, five 8s, three 2048s), then
        switch the pairing to neighbours, chapter 10's first kernel (Fig. 10.6), and run it again.
      </p>
      <Simplification>
        <p>
          One addition per step, rounded once, no fused multiply-add. Threads run one at a time, which gives the same
          result as any real order because no two threads touch the same cell in one level. The 5-bit format is not
          offered: with 16 values its largest value, 3.5, leaves no data on which the two pairings differ without
          clamping.
        </p>
      </Simplification>
      <ThreadToData />

      <h2>Sort first, by magnitude</h2>
      <p>
        The book's remedy is to sort the data ascending before reducing it. Then elements next to each other have
        similar magnitudes, so a tree that combines neighbours adds like with like at every level. Where each thread
        first sums its own group sequentially, ascending order helps that loop as well.
      </p>
      <Source section="A.6" pages="594" />
      <p>
        The book gives this as one reason sorting is so common in massively parallel numerical code, and adds that the
        signs of the values have to be taken into account.
      </p>
      <Beyond title="Not in the book: the same effect without sorting">
        <p>
          PyTorch's CPU sum is a cascade, "chunking the sum into groups of 16 or more elements" and summing those sums
          the same way, so "only numbers of a similar order of magnitude are added together". NumPy uses partial pairwise
          summation: always when no axis is given, otherwise only along the fast axis in memory ({FACTS} D1, D4).
        </p>
      </Beyond>
      <Ask q={questionsSA6['appA/A.6/q-sign']} />

      <h2>The accumulator's width</h2>
      <p>
        Order is one lever. The one you will pull most often in inference code is the other: the format of the
        accumulator. Inputs in FP16 or BF16 are cheap to load; the question is what you add them into. Predict one number
        before you open the explorer.
      </p>
      <Beyond>
        <p>Accumulator width is not discussed in the book. The explorer and every number on it come from our model.</p>
      </Beyond>
      <Predict
        id="appA/A.6/p-half-loop"
        prompt={
          <>
            An FP16 accumulator adds 65,536 values drawn uniformly from [0, 1), in order. The true sum is about 32,768. What
            does the loop return?
          </>
        }
      >
        {(v) => <HalfLoopReveal v={v} />}
      </Predict>
      <Ask q={questionsSA6['appA/A.6/q-half-tree']} />
      <p>
        Now explore. Start with uniform data, FP32, the sequential loop, then switch the accumulator; then try the tree
        and the two Kahan methods, and each data set.
      </p>
      <Simplification>
        <p>
          Every addition rounds to nearest even in the accumulator format, after the input is converted to it. No fused
          multiply-add, no flush-to-zero, no compiler reassociation. One tree shape, chapter 10's "halves". The chunked
          order is chapter 10's coarsening: thread t adds elements t, t + T, t + 2T and so on, then the T sums go
          through the tree.
        </p>
      </Simplification>
      <Explorer />
      <p>What the explorer shows, all of it from the model:</p>
      <ul>
        <li>
          A narrow accumulator stalls. A running sum of values below 1 stops at 2048 in FP16 and at 256 in BF16, the
          points where each format's spacing reaches 2. BF16 has FP32's range but only 8 significant bits.
        </li>
        <li>
          On mixed magnitudes, ascending order beats descending for the loop, and the tree and Kahan beat the long loop.
        </li>
        <li>
          An FP16 tree keeps its partial sums accurate, so it can overflow at 65,504 where the loop merely stalls.
        </li>
        <li>
          The chunk length trades speed against error. Chapter 10 lengthens each thread's loop to cut the tree's idle
          steps; each thread's loop then loses more. On random FP32 data the effect is small, on "one big, many small" it
          grows with c, and in FP16 it is large.
        </li>
      </ul>
      <p>
        The libraries you call make this choice for you, and they do not all make the same one. When a half-precision
        model gives slightly different numbers in two frameworks, the accumulator is the first place to look.
      </p>
      <Today checked={CHECKED('D1, D2, D3')}>
        <ul>
          <li>
            PyTorch accumulates half, bfloat16 and FP8 in float everywhere, and float in double on the CPU. Half GEMMs may
            truncate a few partial sums (a flag, <code>allow_fp16_reduced_precision_reduction</code>); the BF16 flag "is
            turned on by default".
          </li>
          <li>
            MLX's <code>softmax</code> accumulates in the input type unless <code>precise=True</code>; by our reading of
            its kernels, <code>mx.sum</code> on half accumulates in half.
          </li>
          <li>
            llama.cpp asks for F32 accumulation in flash attention, and allows BF16 for the non-flash KQ product, with{' '}
            <code>ggml_prec_set_acc</code>.
          </li>
        </ul>
      </Today>

      <h2>Compensated summation</h2>
      <p>
        For something sturdier than sorting, the book points to Kahan's compensated summation, without describing it.
      </p>
      <Source section="A.6" pages="594" figs="ref. [5]" />
      <p>
        The idea is to keep a second variable, c, that holds what the last addition dropped, and to put it back into the
        next input before adding. Each step computes y = x − c, then t = sum + y. The difference t − sum is what the
        addition really added; subtracting y from it leaves the part that was lost, with its sign turned around. That
        goes into c. The cost is four additions per element instead of one.
      </p>
      <Beyond>
        <p>
          The description of the algorithm, the scene and the bug hunt are ours. The data in both are FP32 values near 1,
          where each step can be written in units of 2⁻²⁵.
        </p>
      </Beyond>
      <Simplification>
        <p>
          Every step rounds once to nearest even in FP32, with no reassociation and no fused multiply-add: what the bug
          hunt's C++ does when it is compiled without fast math.
        </p>
      </Simplification>
      <KahanScene />
      <p>
        The four statements have to be exactly these, in exactly this order. Below are four versions of the loop. Decide
        what each one does before you run them.
      </p>
      <BugHunt />
      <p>
        Version 1 subtracts the lost part instead of restoring it. On these data its corrections cancel in pairs and it
        returns exactly the plain sum; on uniform data it lands between the plain loop and Kahan's, nowhere near the
        latter. Version 3 updates sum before it computes c, so t − sum is always 0 and c is always −y: every input is
        carried into all the inputs after it, and the sum runs away.
      </p>
      <p>
        Version 4 is the one to remember. In real arithmetic (t − sum) − y is zero, so a compiler that is allowed to treat
        addition as associative may delete it, and Kahan's loop silently becomes the plain loop. Whether it may depends on
        the backend and on your flags.
      </p>
      <Today checked={CHECKED('A13, B5, C2, C3')}>
        <ul>
          <li>
            CUDA: <code>__fadd_rn</code> and <code>__fsub_rn</code> are never fused into multiply-adds. Whether nvcc
            reassociates plain float additions, NVIDIA's guide does not settle (sections 5.6.3 and 5.6.1.8 disagree).
          </li>
          <li>HIP and Clang: <code>-fno-associative-math</code> is the default; <code>-ffast-math</code> reassociates.</li>
          <li>
            Metal: both candidate default modes, fast and relaxed, allow reassociation;{' '}
            <code>#pragma METAL fp math_mode(safe)</code> forbids it.
          </li>
        </ul>
      </Today>
      <p>
        While this chapter was designed, a compensated loop on the course's Apple GPU behaved exactly as Version 4 under
        the default mode, and like Kahan's with the pragma. Spelled <code>#pragma metal fp math_mode(safe)</code>, in
        lower case, it compiled without a warning and changed nothing. Lab A.1 checks your device rather than trusting
        either result.
      </p>
      <Ask q={questionsSA6['appA/A.6/q-fastmath']} />
      <Ask q={questionsSA6['appA/A.6/q-kahan16']} />

      <h2>Back to chapter 10, and M4</h2>
      <p>
        Thread coarsening (§10.10) makes each thread add many elements in a loop before the tree. Each of those loops is
        a long sum, so a larger factor can mean a larger error. A per-thread Kahan loop removes most of it, and in a
        memory-bound kernel its three extra additions per element cost little: the sum goes from 0.25 to 1 operation per
        byte, still far below the 20 or so at which the book's example device stops being memory-bound. Lab A.1
        measures both on your GPU (<a href={href('appA', 'labs')}>labs</a>).
      </p>
      <Source pages="246–248" figs="ch. 10 §10.10; ch. 5 §5.1, p. 94" />
      <Ask q={questionsSA6['appA/A.6/q-coarse']} />
      <p>
        Milestone M4, the softmax of lab 10.3, already does the most important thing. Subtracting the row's maximum makes
        every term exp(x − max) lie in (0, 1], all positive, so there is no cancellation, and the largest term is exactly
        1. The lever left is the accumulator, and lab 10.3 keeps it in float. Each of its threads adds the terms of a row
        one block-width apart: the chunked order of the explorer, with c equal to the row length divided by the block
        size (its tree after that is warp shuffles, a different shape from the explorer's).
      </p>
      <Aside>
        <p>
          The start page's teaser was this sum: one softmax denominator over 131,072 terms. In half precision the loop
          stalled far below the true value; the tree did not.
        </p>
      </Aside>
    </>
  );
}
