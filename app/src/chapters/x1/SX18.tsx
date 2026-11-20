import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  CDF_LABEL,
  HUNT_ORDER_LABEL,
  MISSING_LABEL,
  P_DRAW,
  SAMPLER_LABEL,
  SAMPLER_PROBS,
  STEP_ROWS,
  atRiskBound,
  chooseIn,
  compareSamplers,
  disagreement,
  drawKernel,
  flipPoints,
  huntDraw,
  inverseCdf,
  kernelTotal,
  marginOf,
  perturb,
  sampleLogits,
  sampleWith,
  uniformsFor,
  weightsFrom,
  type CdfOrder,
  type Chunk,
  type HuntOrder,
  type KernelCfg,
  type Missing,
  type RaceQ,
  type Sampler,
} from '../../sim/draw';
import { f32short, m5U } from '../../sim/philox';
import { FindUScene } from './FindUScene';

// Section X1.8: the final draw. Everything the interactives show is computed by
// src/sim/draw.ts (drawKernel, huntDraw, cdfIn / chooseIn / disagreement, compareSamplers),
// pinned in draw.test.ts, which also runs labs/x1/snippets/draw_search.cu on the course's
// emulator (with each barrier removed in turn) and compares its tokens with the model.

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const MONO: CSSProperties = { fontFamily: 'var(--sans)', fontVariationSettings: "'MONO' 1" };
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);
const s = (v: number) => f32short(v);
const f32 = Math.fround;
const TOP = 1 - 2 ** -24;

// ---------------------------------------------------------------- the stepper
type View = { kind: 'total' } | { kind: 'chunk'; c: number } | { kind: 'flags'; c: number } | { kind: 'vote'; c: number } | { kind: 'done' };

function viewsOf(chunks: Chunk[]): View[] {
  const v: View[] = [{ kind: 'total' }];
  for (const ch of chunks) {
    v.push({ kind: 'chunk', c: ch.c });
    if (ch.crosses) v.push({ kind: 'flags', c: ch.c }, { kind: 'vote', c: ch.c });
  }
  v.push({ kind: 'done' });
  return v;
}
/** Runs of consecutive threads in one warp that read the same thing. */
function groupReads(reads: { thread: number; warp: number; value: number; ok: boolean; what: string }[]) {
  const out: { from: number; to: number; warp: number; value: number; ok: boolean; what: string }[] = [];
  for (const r of reads) {
    const g = out[out.length - 1];
    if (g && g.warp === r.warp && g.to === r.thread - 1 && g.what === r.what && Object.is(g.value, r.value)) g.to = r.thread;
    else out.push({ from: r.thread, to: r.thread, warp: r.warp, value: r.value, ok: r.ok, what: r.what });
  }
  return out;
}
const bits = (b: number, width: number) => Array.from({ length: width }, (_, l) => (Math.floor(b / 2 ** l) % 2 ? '1' : '0')).reverse().join('');

function FindUStepper() {
  const [rowId, setRowId] = useState('long');
  const row0 = STEP_ROWS.find((r) => r.id === rowId) ?? STEP_ROWS[1];
  const [n, setN] = useState(row0.n);
  const [T, setT] = useState(row0.T);
  const [uText, setUText] = useState(String(row0.u));
  const [fromPhilox, setFromPhilox] = useState(false);
  const [pSeed, setPSeed] = useState(7);
  const [pStream, setPStream] = useState(3);
  const [pStep, setPStep] = useState(0);
  const [vec, setVec] = useState(4);
  const [warp, setWarp] = useState(8);
  const [missing, setMissing] = useState<Missing>('none');
  const [order, setOrder] = useState<HuntOrder>('in-order');
  const [at, setAt] = useState(0);

  const uParsed = Number(uText);
  const uOk = uText.trim() !== '' && isFinite(uParsed) && uParsed >= 0 && uParsed < 1;
  const u = fromPhilox ? m5U(BigInt(pSeed), BigInt(pStream), BigInt(pStep)) : uOk ? f32(Math.min(uParsed, TOP)) : 0;
  const cfg: KernelCfg = useMemo(() => ({ threads: 16, warp, vec }), [warp, vec]);
  const weights = useMemo(() => weightsFrom(sampleLogits(n), T), [n, T]);
  const run = useMemo(() => {
    const tot = kernelTotal(weights, cfg);
    const correct = drawKernel(weights, u, cfg);
    if (missing === 'none') return { tot, correct, search: correct, targets: new Array(16).fill(correct.target) as number[], hunt: null };
    const h = huntDraw(weights, u, cfg, missing, order);
    return { tot, correct, search: h.search, targets: h.targets, hunt: h.hunt };
  }, [weights, u, cfg, missing, order]);
  const views = useMemo(() => viewsOf(run.search.chunks), [run]);
  const total = views.length;
  const pos = Math.min(at, total);
  const cur = pos > 0 ? views[pos - 1] : undefined;
  const reset = () => setAt(0);

  const pickRow = (id: string) => {
    const r = STEP_ROWS.find((x) => x.id === id)!;
    setRowId(id);
    setN(r.n);
    setT(r.T);
    setUText(String(r.u));
    setFromPhilox(false);
    reset();
  };

  const chunkOf = (c: number) => run.search.chunks.find((x) => x.c === c)!;
  const lastChunk = run.search.chunks[run.search.chunks.length - 1];
  const curChunk = cur && cur.kind !== 'total' && cur.kind !== 'done' ? chunkOf(cur.c) : cur?.kind === 'done' && lastChunk?.crosses ? lastChunk : undefined;
  const flagsShown = cur?.kind === 'flags' || cur?.kind === 'vote' || (cur?.kind === 'done' && !!lastChunk?.crosses);
  const nW = Math.ceil(cfg.threads / cfg.warp);
  const uniformTarget = run.targets.every((t) => t === run.targets[0]);
  const targetTxt = uniformTarget ? s(run.targets[0]) : 'differs by thread';

  const narrate = (() => {
    if (!cur)
      return `One block of 16 threads owns this row of ${n} survivors. Chunks of 16 × ${vec} = ${16 * vec} survivors: thread t holds survivors t·${vec} to t·${vec} + ${vec - 1} of each chunk. Press Step.`;
    if (cur.kind === 'total')
      return `Phase 1: each thread adds survivors t, t + 16, t + 32, …; each warp of ${warp} adds its threads' sums with a shuffle tree; lane 0 writes the warp sum to scratch_s; warp 0 adds those and thread 0 writes the total, ${s(run.tot.total)}, to scratch_s[0]. Every thread reads it and computes target = u × total = ${targetTxt}.${run.hunt ? ` With ${MISSING_LABEL[missing]}, ${run.hunt.wrong} of 16 threads read something else (below).` : ''}`;
    if (cur.kind === 'chunk') {
      const ch = chunkOf(cur.c);
      return `Chunk ${ch.c}, survivors ${ch.base} to ${ch.end - 1}: each warp scans its threads' totals, the last lane writes the warp total to scratch_s, and every thread adds the warp totals: chunk sum ${s(ch.sum)}. Aggregate ${s(ch.aggBefore)} + ${s(ch.sum)} = ${s(ch.aggAfter)} ${ch.crosses ? `is greater than the target: this chunk crosses.` : `is not greater than the target: skip it, no flags.`}`;
    }
    if (cur.kind === 'flags') {
      const ch = chunkOf(cur.c);
      const nf = ch.lanes.reduce((a, l) => a + l.flags.filter(Boolean).length, 0);
      return `Each thread adds the aggregate, the warps before it and the lanes before it to its own inclusive sums, then flags each survivor whose sum is greater than its target. ${nf} survivors are flagged; each thread keeps its first.`;
    }
    if (cur.kind === 'vote') {
      const ch = chunkOf(cur.c);
      return `Each warp ballots "I have a flag": ${ch.ballots.map((b, k) => `warp ${k} ${bits(b, cfg.warp)}`).join(', ')}. The lowest set bit is the warp's first flagged lane; an atomicMin over the warps' candidates keeps the lowest: survivor ${ch.winner}.`;
    }
    const sr = run.search;
    const base = sr.fallback ? `No sum was greater than the target: the fallback takes the last survivor with positive weight, ${sr.index}.` : `The block writes survivor ${sr.index}.`;
    return run.hunt
      ? `${base} With both barriers it writes ${run.correct.index}: ${sr.index === run.correct.index ? 'the same token, this time.' : 'a different token.'}`
      : base;
  })();

  return (
    <Bench
      title="Find u"
      task="Step through the 200-survivor row; then remove barrier A, then barrier B, under each order."
      foot={
        <>
          One block of 16 threads; the warp width is a display choice. Float32 sums in the kernel's own order. The orders are
          the course emulator's two fixed schedules and one a GPU may produce, not GPU timing.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Row
          <span className="seg" role="group" aria-label="Row">
            {STEP_ROWS.map((r) => (
              <button key={r.id} aria-pressed={rowId === r.id && n === r.n && T === r.T && !fromPhilox && uText === String(r.u)} onClick={() => pickRow(r.id)}>{r.label}</button>
            ))}
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 10rem' }}>
          survivors n = {n}
          <input type="range" min={1} max={200} value={n} aria-label="n" onChange={(e) => { setN(Number(e.target.value)); reset(); }} style={{ width: '100%' }} />
        </label>
        <label className="field" style={{ flex: '1 1 10rem' }}>
          temperature T = {T.toFixed(2)}
          <input type="range" min={0.3} max={3} step={0.05} value={T} aria-label="T" onChange={(e) => { setT(Number(e.target.value)); reset(); }} style={{ width: '100%' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="check">
          <input type="checkbox" checked={fromPhilox} aria-label="u from Philox" onChange={(e) => { setFromPhilox(e.target.checked); reset(); }} /> u from Philox (X1.7)
        </label>
        {fromPhilox ? (
          <>
            {([['seed', pSeed, setPSeed], ['stream', pStream, setPStream], ['step', pStep, setPStep]] as const).map(([name, v, set]) => (
              <label key={name} className="field">
                {name}
                <input type="number" min={0} step={1} value={v} aria-label={'philox ' + name} onChange={(e) => { set(Math.max(0, Math.floor(Number(e.target.value) || 0))); reset(); }} style={{ width: '6rem' }} />
              </label>
            ))}
          </>
        ) : (
          <label className="field">
            u in [0, 1)
            <input type="number" min={0} max={0.99999994} step="any" value={uText} aria-label="u" onChange={(e) => { setUText(e.target.value); reset(); }} />
          </label>
        )}
        <span className="muted" data-u={u}>u = {s(u)}</span>
        <span style={FIELD}>
          Survivors per thread
          <span className="seg" role="group" aria-label="Survivors per thread">
            {[1, 2, 4, 8].map((v) => (
              <button key={v} aria-pressed={vec === v} onClick={() => { setVec(v); reset(); }}>{v}</button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Warp width (display)
          <span className="seg" role="group" aria-label="Warp width">
            {[4, 8, 16].map((v) => (
              <button key={v} aria-pressed={warp === v} onClick={() => { setWarp(v); reset(); }}>{v}</button>
            ))}
          </span>
        </span>
      </div>
      {!fromPhilox && !uOk ? <p className="t-fail" style={{ margin: '0 0 0.6rem' }}>u must be a number from 0 up to, not including, 1.</p> : null}
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Barriers around the read of the total
          <span className="seg" role="group" aria-label="Barriers">
            {(['none', 'written', 'read'] as Missing[]).map((m) => (
              <button key={m} aria-pressed={missing === m} onClick={() => { setMissing(m); reset(); }}>
                {m === 'none' ? 'both in place' : m === 'written' ? 'remove A' : 'remove B'}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Order
          <span className="seg" role="group" aria-label="Order">
            {(['in-order', 'reversed', 'warp-ahead'] as HuntOrder[]).map((o) => (
              <button key={o} aria-pressed={order === o} disabled={missing === 'none'} onClick={() => { setOrder(o); reset(); }}>{HUNT_ORDER_LABEL[o]}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setAt(0)} disabled={pos === 0}>Restart</button>
        <button onClick={() => setAt(pos - 1)} disabled={pos === 0}>Back</button>
        <button className="primary" onClick={() => setAt(pos + 1)} disabled={pos >= total}>Step</button>
        <button onClick={() => setAt(total)} disabled={pos >= total}>Run to the end</button>
        <span className="muted" aria-live="polite" data-step={pos}>step {pos} of {total}</span>
      </div>
      <div className={'narrate' + (cur?.kind === 'done' && run.hunt && run.search.index !== run.correct.index ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {narrate}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)' }}>
        <div>
          <div className="panel" data-part="phase1" style={{ marginBottom: '0.8rem' }}>
            <div className="mx-label t-shared">Phase 1: the total (scratch_s)</div>
            {cur ? (
              <>
                <div style={{ fontSize: '0.78rem' }}>
                  warp sums: {run.tot.warpSums.map((v, k) => <span key={k} style={MONO}>{k ? ', ' : ''}{s(v)}</span>)}
                </div>
                <div>
                  total <b data-total={run.tot.total}>{s(run.tot.total)}</b>; target <b data-target={uniformTarget ? run.targets[0] : ''}>{targetTxt}</b>
                </div>
              </>
            ) : (
              <div className="muted">not computed yet</div>
            )}
            {cur && run.hunt ? (
              <div style={{ marginTop: '0.5rem' }} data-part="hunt" data-wrong={run.hunt.wrong}>
                <div style={{ fontSize: '0.8rem' }}>
                  <b className={run.hunt.wrong ? 't-fail' : 't-ok'}>{run.hunt.wrong} of 16</b> threads read something other than the total ({HUNT_ORDER_LABEL[order]}).
                </div>
                <table className="threads">
                  <thead><tr><th>threads</th><th>warp</th><th>read scratch_s[0]</th><th>target</th></tr></thead>
                  <tbody>
                    {groupReads(run.hunt.reads).map((g) => (
                      <tr key={g.from} data-read={g.from} data-to={g.to} data-ok={g.ok ? 1 : 0}>
                        <td>{g.from === g.to ? g.from : `${g.from} to ${g.to}`}</td>
                        <td>{g.warp}</td>
                        <td className={g.ok ? '' : 't-fail'} style={{ whiteSpace: 'normal' }}>{isNaN(g.value) ? '—' : s(g.value)} ({g.what})</td>
                        <td style={MONO}>{s(run.targets[g.from])}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </div>
          <div className="panel" data-part="chunks">
            <div className="mx-label">Chunks</div>
            <div data-part="chunks-scroll" style={{ overflowX: 'auto', maxWidth: '100%' }}>
            <table className="threads">
              <thead><tr><th>chunk</th><th>survivors</th><th>sum</th><th>aggregate</th><th /></tr></thead>
              <tbody>
                {run.search.chunks.map((ch) => {
                  const seen = views.slice(0, pos).some((v) => v.kind === 'chunk' && v.c === ch.c);
                  return (
                    <tr key={ch.c} data-chunk={ch.c} data-crosses={seen ? (ch.crosses ? 1 : 0) : ''} className={curChunk?.c === ch.c ? 'cur' : ''} style={{ opacity: seen ? (ch.crosses ? 1 : 0.55) : 0.3 }}>
                      <td>{ch.c}</td>
                      <td>{ch.base}–{ch.end - 1}</td>
                      <td style={MONO}>{seen ? s(ch.sum) : ''}</td>
                      <td style={MONO}>{seen ? `${s(ch.aggBefore)} → ${s(ch.aggAfter)}` : ''}</td>
                      <td>{seen ? (ch.crosses ? <b>crosses</b> : 'skipped') : ''}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            </div>
            {run.search.chunks.length * 16 * vec < n ? <div className="muted" style={{ fontSize: '0.75rem' }}>later chunks are never read</div> : null}
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          {curChunk ? (
            <table className="threads" data-part="lanes" style={{ minWidth: '26rem' }}>
              <thead>
                <tr><th>thread</th><th>survivors: weight → sum</th><th>before</th>{run.hunt ? <th>its target</th> : null}<th>first flag</th></tr>
              </thead>
              <tbody>
                {Array.from({ length: nW }, (_, k) => {
                  const ls = curChunk.lanes.filter((l) => l.warp === k);
                  return [
                    <tr key={'w' + k}>
                      <td colSpan={run.hunt ? 5 : 4} className="muted" style={{ paddingTop: '0.4rem' }}>
                        warp {k}
                        {cur?.kind === 'vote' || cur?.kind === 'done' ? (
                          <span data-ballot={k} data-bits={curChunk.ballots[k]}>
                            {' '}· ballot <span style={MONO}>{bits(curChunk.ballots[k], cfg.warp)}</span> · first flagged lane {curChunk.warpFirst[k] < 0 ? 'none' : curChunk.warpFirst[k]}
                          </span>
                        ) : null}
                      </td>
                    </tr>,
                    ...ls.map((l) => {
                      const win = (cur?.kind === 'vote' || cur?.kind === 'done') && run.search.winner && run.search.winner.thread === l.thread && run.search.winner.chunk === curChunk.c;
                      return (
                        <tr key={l.thread} data-thread={l.thread} data-hit={flagsShown && l.hit >= 0 ? l.first + l.hit : ''} className={win ? 'cur' : ''}>
                          <td>{l.thread}</td>
                          <td style={{ whiteSpace: 'normal' }}>
                            {l.w.map((w, j) => {
                              const i = l.first + j;
                              if (i >= n) return null;
                              const f = flagsShown && l.flags[j];
                              return (
                                <span key={j} data-i={i} data-flag={flagsShown ? (l.flags[j] ? 1 : 0) : ''}
                                  style={{ ...MONO, display: 'inline-block', marginRight: 4, padding: '0 0.25rem', borderRadius: 3, fontSize: '0.72rem',
                                    border: `1px solid ${f ? 'var(--ok)' : 'var(--rule)'}`, background: f ? 'var(--ok-wash)' : 'transparent' }}>
                                  #{i} {s(w)}{flagsShown && l.cum.length ? ` → ${s(l.cum[j])}` : ''}
                                </span>
                              );
                            })}
                          </td>
                          <td style={MONO}>{s(l.before)}</td>
                          {run.hunt ? <td style={MONO} className={run.targets[l.thread] === run.correct.target ? '' : 't-fail'}>{s(run.targets[l.thread])}</td> : null}
                          <td>{flagsShown ? (l.hit >= 0 ? <b>#{l.first + l.hit}</b> : '—') : ''}</td>
                        </tr>
                      );
                    }),
                  ];
                })}
              </tbody>
            </table>
          ) : (
            <p className="muted" style={{ margin: 0 }}>The lanes of the current chunk appear here.</p>
          )}
        </div>
      </div>
      {cur?.kind === 'done' ? (
        <p style={{ margin: '0.8rem 0 0' }} data-result={run.search.index} data-correct={run.correct.index} data-fallback={run.search.fallback ? 1 : 0}>
          Token: <b className={run.hunt && run.search.index !== run.correct.index ? 't-fail' : 't-ok'}>survivor {run.search.index}</b>
          {run.search.fallback ? ' (fallback)' : ''}. {run.search.chunks.filter((c) => !c.crosses).length} chunk(s) skipped, one scanned.
          {run.hunt ? ` With both barriers: survivor ${run.correct.index}.` : ` The double-precision CPU reference draws survivor ${inverseCdf(weights, u).index}.`}
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- near a boundary
const BROW = weightsFrom(sampleLogits(40), 1);
const EPS: { label: string; v: number }[] = [
  { label: 'none', v: 0 },
  { label: '2⁻²²', v: 2 ** -22 },
  { label: '2⁻²⁰', v: 2 ** -20 },
  { label: '2⁻¹⁸', v: 2 ** -18 },
];
const ORDERS: CdfOrder[] = ['double', 'f32-loop', 'f32-kernel'];
const K_RANGE = 12;

function BoundaryExplorer() {
  const [b, setB] = useState(20);
  const [eps, setEps] = useState(0);
  const [k, setK] = useState(0);
  const pert = useMemo(() => (eps ? perturb(BROW, eps) : null), [eps]);
  const flip = useMemo(() => flipPoints(BROW, 'double'), []);
  const c = Math.round(flip[b] * 2 ** 24);
  const ks = Array.from({ length: 2 * K_RANGE + 1 }, (_, i) => i - K_RANGE);
  const rows: { id: string; label: string; pick: (u: number) => number }[] = [
    ...ORDERS.map((o) => ({ id: o, label: CDF_LABEL[o], pick: (u: number) => chooseIn(BROW, u, o) })),
    ...(pert ? [{ id: 'perturbed', label: "float32, the kernel's order, weights perturbed (another backend's exp)", pick: (u: number) => chooseIn(pert, u, 'f32-kernel') }] : []),
  ];
  const grid = rows.map((r) => ks.map((d) => r.pick((c + d) / 2 ** 24)));
  const ref = grid[0];
  const u = (c + k) / 2 ** 24;
  const dis = useMemo(
    () => disagreement([{ weights: BROW, order: 'double' }, { weights: BROW, order: 'f32-loop' }, { weights: BROW, order: 'f32-kernel' }, ...(pert ? [{ weights: pert, order: 'f32-kernel' as CdfOrder }] : [])]),
    [pert],
  );
  const col = ks.indexOf(k);
  return (
    <Bench
      title="Near a boundary"
      task="Drag u across boundary 20, then switch on a perturbation of 2⁻²⁰ and watch the band widen."
      foot={<>u moves in steps of 2⁻²⁴, the steps M5's u can take. Red: a choice that differs from the double-precision CPU reference.</>}
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          boundary between survivors {b} and {b + 1}
          <input type="range" min={0} max={38} value={b} aria-label="boundary" onChange={(e) => { setB(Number(e.target.value)); setK(0); }} style={{ width: '100%' }} />
        </label>
        <span style={FIELD}>
          Relative perturbation of the weights
          <span className="seg" role="group" aria-label="Perturbation">
            {EPS.map((e) => (
              <button key={e.label} aria-pressed={eps === e.v} onClick={() => setEps(e.v)}>{e.label}</button>
            ))}
          </span>
        </span>
      </div>
      <label className="field" style={{ display: 'flex', marginBottom: '0.7rem' }}>
        u = boundary {k >= 0 ? '+' : '−'} {Math.abs(k)} × 2⁻²⁴
        <input type="range" min={-K_RANGE} max={K_RANGE} value={k} aria-label="u offset" onChange={(e) => setK(Number(e.target.value))} style={{ width: '100%' }} />
      </label>
      <div style={{ overflowX: 'auto' }}>
        <table className="threads" data-part="grid" style={{ minWidth: '40rem' }}>
          <thead>
            <tr>
              <th>order</th>
              {ks.map((d) => <th key={d} style={{ textAlign: 'center', fontWeight: d === k ? 800 : 400 }}>{d === 0 ? '0' : d % 4 === 0 ? (d > 0 ? '+' + d : d) : ''}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={r.id} data-order={r.id} data-tokens={grid[ri].join(',')}>
                <td style={{ whiteSpace: 'normal', minWidth: '9rem', maxWidth: '15rem' }}>{r.label}</td>
                {grid[ri].map((t, ci) => (
                  <td key={ci} style={{ textAlign: 'center', padding: '0.2rem 0', minWidth: '1.5rem',
                    background: t !== ref[ci] ? 'var(--fail-wash)' : ci === col ? 'var(--rule-soft)' : undefined,
                    color: t !== ref[ci] ? 'var(--fail)' : undefined, fontWeight: ci === col ? 700 : 400 }}>{t}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.7rem 0 0' }} data-part="readout" data-margin={marginOf(BROW, u)} data-choices={rows.map((r, i) => grid[i][col]).join(',')}>
        u = {u.toPrecision(9)}: {rows.map((r, i) => `${r.id === 'perturbed' ? 'perturbed' : r.id} → ${grid[i][col]}`).join(', ')}. The draw is{' '}
        {marginOf(BROW, u).toExponential(2)} of the total from the nearest boundary.
      </p>
      <p style={{ margin: '0.4rem 0 0' }} data-part="risk" data-measure={dis.measure}>
        Over all 39 boundaries of this row, the orders shown disagree on <b>{(dis.measure * 1e6).toFixed(1)} per million</b> of u.
        The bound 2 × (n − 1) × m gives {(atRiskBound(40, 2 ** -22) * 1e6).toFixed(1)} per million at a margin m of 2⁻²²
        and {(atRiskBound(40, 2 ** -20) * 1e6).toFixed(1)} at 2⁻²⁰.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- three samplers
const DRAWS = [1, 10, 100, 1000, 10000, 100000];
const METHODS: Sampler[] = ['inverse-cdf', 'race', 'gumbel'];

function SamplersExplorer() {
  const [draws, setDraws] = useState(1000);
  const [seed, setSeed] = useState(5);
  const [logU, setLogU] = useState(false);
  const q: RaceQ = logU ? 'log-u' : 'one-minus';
  const cmp = useMemo(() => compareSamplers(SAMPLER_PROBS, draws, seed, 0, q), [draws, seed, q]);
  const first = useMemo(
    () => Array.from({ length: 8 }, (_, d) => {
      const us = uniformsFor(seed, 0, d, SAMPLER_PROBS.length);
      return { d, u0: us[0], picks: METHODS.map((m) => sampleWith(SAMPLER_PROBS, us, m, q)) };
    }),
    [seed, q],
  );
  return (
    <Bench
      title="Same bits, three samplers"
      task="Run 10 draws, then 100,000: the columns converge to p, the per-draw agreement does not."
      foot={<>Double precision. Each draw takes 6 uniforms from Philox (seed, stream 0, step = draw), M5's conversion; uniform 0 is M5's u.</>}
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Draws
          <span className="seg" role="group" aria-label="Draws">
            {DRAWS.map((d) => (
              <button key={d} aria-pressed={draws === d} onClick={() => setDraws(d)}>{d.toLocaleString('en-US')}</button>
            ))}
          </span>
        </span>
        <label className="field">
          seed
          <input type="number" min={0} step={1} value={seed} aria-label="sampler seed" onChange={(e) => setSeed(Math.max(0, Math.floor(Number(e.target.value) || 0)))} style={{ width: '6rem' }} />
        </label>
        <label className="check">
          <input type="checkbox" checked={logU} aria-label="race with q = -ln u" onChange={(e) => setLogU(e.target.checked)} /> race with q = −ln u instead of −ln(1 − u)
        </label>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '34rem' }} data-part="freq">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>token</th>
              <th className="num">p</th>
              {METHODS.map((m) => <th key={m} className="num">{m === 'inverse-cdf' ? 'inverse CDF' : m === 'race' ? 'argmax(p / q)' : 'argmax(l + g)'}</th>)}
            </tr>
          </thead>
          <tbody>
            {SAMPLER_PROBS.map((p, i) => (
              <tr key={i} data-token={i}>
                <td style={{ paddingLeft: '0.6rem' }}>{i}</td>
                <td className="num">{p.toFixed(2)}</td>
                {METHODS.map((m) => (
                  <td key={m} className="num" data-method={m} data-count={cmp.counts[m][i]}>{(cmp.counts[m][i] / draws).toFixed(draws >= 1000 ? 3 : 2)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.7rem 0 0' }} data-part="agree" data-inv-race={cmp.agree.invRace} data-inv-gumbel={cmp.agree.invGumbel} data-race-gumbel={cmp.agree.raceGumbel}>
        Same token on the same draw: inverse CDF and race {(cmp.agree.invRace * 100).toFixed(1)}%, inverse CDF and Gumbel{' '}
        {(cmp.agree.invGumbel * 100).toFixed(1)}%, race and Gumbel {(cmp.agree.raceGumbel * 100).toFixed(1)}%.
      </p>
      <div style={{ overflowX: 'auto', marginTop: '0.6rem' }}>
        <table className="threads" data-part="first" style={{ minWidth: '28rem' }}>
          <thead><tr><th>draw</th><th>u (uniform 0)</th>{METHODS.map((m) => <th key={m}>{SAMPLER_LABEL[m].split(' (')[0]}</th>)}</tr></thead>
          <tbody>
            {first.map((r) => (
              <tr key={r.d} data-draw={r.d} data-picks={r.picks.join(',')}>
                <td>{r.d}</td>
                <td style={MONO}>{s(r.u0)}</td>
                {r.picks.map((t, i) => <td key={i}><b>{t}</b></td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSX18: Record<string, Question> = {
  'x1/X1.8/q-ge-gt': {
    kind: 'choice',
    id: 'x1/X1.8/q-ge-gt',
    prompt: (
      <>
        Candidates in vocabulary order with probabilities [0, 0.5, 0.5] and u = 0 exactly (possible with M5's conversion).
        Which index does "first inclusive sum ≥ u × total" pick, and which does "strictly greater"?
      </>
    ),
    options: [
      { label: '≥ picks index 0, a zero-probability token; > picks index 1', correct: true },
      { label: 'Both pick index 1', why: 'The first inclusive sum is 0, and 0 ≥ 0 holds, so ≥ stops at index 0.' },
      { label: 'Both pick index 0', why: '0 > 0 is false, so the strict rule moves on to index 1, where the sum is 0.5.' },
    ],
    explain: (
      <p>
        Target 0 × 1 = 0; inclusive sums 0, 0.5, 1. "≥" stops at the first, a token the model gave probability 0. "&gt;"
        never stops on a zero-weight token, because adding 0 cannot make a sum exceed a target the sum before it did not
        exceed. M5 uses "&gt;".
      </p>
    ),
    source: 'ours; notes/today/x1-facts.md A2, E2',
  },
  'x1/X1.8/q-u1': {
    kind: 'choice',
    id: 'x1/X1.8/q-u1',
    prompt: (
      <>
        A kernel draws u with <code>curand_uniform</code>, which can return exactly 1.0, and searches for the first
        inclusive sum ≥ u, but the row's float sum is 0.99999994. What happens?
      </>
    ),
    options: [
      { label: 'No index qualifies; without a fallback the kernel writes an invalid token or reads past the row', correct: true },
      { label: 'The last token is chosen automatically', why: 'Only if someone wrote that fallback. The search itself finds nothing.' },
      { label: 'Nothing, because probabilities sum to 1', why: 'They sum to 1 in exact arithmetic; the float sum here is 1 − 2⁻²⁴.' },
    ],
    explain: (
      <p>
        FlashInfer's kernel says so in a comment: "this would happen when u is very close to 1 and the sum of probabilities
        is smaller than u. In this case, we use the last valid index". PyTorch's binary search has a guard too: "this actually
        can happen when the random number is 1".
      </p>
    ),
    source: 'notes/today/x1-facts.md A2, A6, D5 (checked 2026-11-20)',
  },
  'x1/X1.8/q-multinomial': {
    kind: 'choice',
    id: 'x1/X1.8/q-multinomial',
    prompt: <>What does <code>torch.multinomial(probs, 1)</code>, the call transformers' <code>generate</code> makes, compute today?</>,
    options: [
      { label: 'An exponential race: q ~ Exp(1) per entry, then argmax(p / q), on CPU, CUDA, MPS and XPU', correct: true },
      { label: 'A binary search of a cumulative sum', why: 'That path is used only with replacement and more than one sample.' },
      { label: 'Gumbel-max on the logits', why: "Its comment derives the race from Gumbel-max, but it takes probabilities and divides by exponentials." },
    ],
    explain: (
      <p>
        <code>Distributions.cpp</code>: "Fast-path for no replacement or if only one sample is drawn", <code>q.exponential_(1)</code>,
        <code>p / q</code>, <code>argmax</code>. Same distribution as the inverse CDF, different token from the same bits.
      </p>
    ),
    source: 'notes/today/x1-facts.md D4, D5 (checked 2026-11-20)',
  },
  'x1/X1.8/q-mlx-batch': {
    kind: 'choice',
    id: 'x1/X1.8/q-mlx-batch',
    prompt: <>mlx-lm draws a row with <code>mx.random.categorical</code>. By our reading of MLX's source, what changes when the batch grows from 1 to 2 rows with the same key?</>,
    options: [
      { label: 'The algorithm: inverse CDF at batch 1, Gumbel-max at 2 or more, so the same key can give a different token for the same row', correct: true },
      { label: 'Nothing; only the speed', why: '`categorical_impl` takes its inverse-CDF path only for a single distribution with a sample shape.' },
      { label: 'The key is split per row, so each row gets a fresh stream but the same algorithm', why: 'The switch is in the algorithm, not only in the key.' },
    ],
    explain: (
      <p>
        <code>categorical_inverse_cdf</code> runs when the logits are one distribution and a sample shape is given; otherwise
        <code> argmax(logits + gumbel)</code>. With (batch, vocabulary) logits, batch 1 qualifies and batch 2 does not (our
        reading).
      </p>
    ),
    source: 'notes/today/x1-facts.md D7, D8 (our reading of the source, checked 2026-11-20)',
  },
  'x1/X1.8/q-risk': {
    kind: 'numeric',
    id: 'x1/X1.8/q-risk',
    prompt: (
      <>
        u is uniform on [0, 1). A draw can flip between backends only if u × total lands within a margin of 2⁻²⁰ × total of
        one of the 39 interior boundaries of 40 survivors. At most how many draws per million are at risk?
      </>
    ),
    answer: 74,
    tol: 0.02,
    unit: 'per million',
    explain: <p>Each boundary has a band of width 2 × 2⁻²⁰ on the u axis: 2 × 39 × 2⁻²⁰ × 10⁶ = 74.4.</p>,
    source: 'ours; src/sim/draw.ts atRiskBound',
  },
  'x1/X1.8/q-device-u': {
    kind: 'choice',
    id: 'x1/X1.8/q-device-u',
    prompt: <>llama.cpp's backend <code>dist</code> uploads a u drawn by the host's mt19937 every step. What does deriving u on the device from (seed, stream, step) buy?</>,
    options: [
      { label: 'No per-step upload, and every backend reproduces the same u from the same inputs', correct: true },
      { label: 'Better randomness than mt19937', why: 'Both pass the standard test batteries; quality is not the difference.' },
      { label: 'Nothing: the upload is a few bytes', why: 'The bytes are few, but it is a host step per token, and the host stream ties every backend to the host.' },
    ],
    explain: (
      <p>
        llama.cpp's choice has its own reason: host and backend paths consume one mt19937 stream, so switching paths keeps
        the numbers. M5 trades that for a draw computed where the logits already are.
      </p>
    ),
    source: 'notes/today/x1-facts.md D3 (checked 2026-11-20)',
  },
  'x1/X1.8/q-barrier': {
    kind: 'choice',
    id: 'x1/X1.8/q-barrier',
    prompt: (
      <>
        In the draw kernel every thread reads the total from <code>scratch_s[0]</code> (between barriers A and B), and the
        chunk loop then writes warp totals into the same array. Remove barrier B. Which of the course's backends is sure to
        report it?
      </>
    ),
    options: [
      { label: 'The race backend: it names the read of the total and the write of a warp total; emu passes in every order', correct: true },
      { label: 'The emu backend, under its in-order schedule', why: 'Every thread reads the total before its first shuffle, and no thread writes a warp total until its whole warp is past the scan: in both fixed orders all reads come first.' },
      { label: 'A GPU, on every run', why: 'Only when one warp runs ahead through its scan before another has read the total; it may pass every run.' },
    ],
    explain: (
      <p>
        Barrier A is the one emu catches: without it, the other warps read warp 0's partial sum. Barrier B guards a
        write-after-read across warps that the emulator's schedules never reorder, so only the race backend, which checks
        happens-before rather than outcomes, sees it. Checked by <code>draw.test.ts</code> on the snippet.
      </p>
    ),
    source: 'ours; labs/README.md (emu and race); src/sim/draw.test.ts',
  },
};

// ---------------------------------------------------------------- the section
export function SX18() {
  const cu = loaded(useLabFile('x1/snippets/draw_search.cu'));
  const barrierLines = cu ? cu.split('\n').flatMap((l, i) => (/barrier [AB]:/.test(l) ? [i + 1] : [])) : [];
  return (
    <>
      <p>
        Every stage so far has narrowed the row. Now one survivor must be picked, with probability proportional to its
        weight, using the u from the <a href={href('x1', 'X1.7')}>previous page</a>. Lay the weights end to end on a line,
        put a mark at u times their total, and take the survivor under the mark.
      </p>
      <Beyond>
        <p>
          The draw is ours. It is chapter 11's sausage with u as the knife: an inclusive scan gives the cut points. The block
          scan is §11.4, the total §10.8, the vote §12.3.
        </p>
      </Beyond>
      <Source section="11.1" pages="252–254" />

      <h2>The inverse CDF</h2>
      <p>
        The survivors arrive sorted, so l₀ is the largest logit. Their weights are w_i = exp((l_i − l₀) / T), with
        temperature folded in here, as on the X1.2 page. A block reduction gives the total. The target is u × total, and the
        token is the first i whose inclusive sum w₀ + … + w_i is strictly greater than the target.
      </p>
      <p>
        Scaling u by the total, rather than comparing u with sums of probabilities, avoids assuming the weights sum to exactly
        1: they never do in float. llama.cpp's host sampler does the same.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D3)">
        <p>
          llama.cpp <code>dist</code>: <code>const double sum_tgt = sum_cum*rnd;</code>, then the first candidate with{' '}
          <code>sum_run &gt;= sum_tgt</code>. Its backend path compares cumulative probabilities with u, strictly.
        </p>
      </Today>
      <Predict
        id="x1/X1.8/p-draw"
        prompt={<>Four survivors with equal weights 1, 1, 1, 1 and u = 0.6. Which index does M5 draw (counting from 0)?</>}
        unit="index"
      >
        {(v) => {
          const d = inverseCdf(P_DRAW.weights, P_DRAW.u);
          return (
            <p data-reveal={d.index}>
              <b>{d.index}.</b> Target 0.6 × 4 = {d.target.toFixed(1)}; inclusive sums {d.inclusive.join(', ')}; the first strictly
              greater is 3, at index {d.index}. {v === d.index ? 'Right.' : v === 3 ? 'Index 3 is where the sum is 4; the sum 3 at index 2 already exceeds 2.4.' : ''}
            </p>
          );
        }}
      </Predict>

      <h2>Both ends of the interval</h2>
      <p>
        M5's u lies in [0, 1 − 2⁻²⁴] and can be exactly 0. With "strictly greater", a token of weight 0 is never chosen, even
        at u = 0: its inclusive sum equals the one before it, which did not exceed the target. With "greater or equal" it can
        be. llama.cpp's host uses ≥ with u × total, its backend &gt; with u; they differ on an exact tie.
      </p>
      <Ask q={questionsSX18['x1/X1.8/q-ge-gt']} />
      <p>
        The other end bites a (0, 1] conversion. <code>curand_uniform</code> can return exactly 1.0, and a float sum of
        probabilities can round to just under 1, so nothing qualifies. M5's u never reaches 1, but it keeps a fallback anyway:
        the kernel's total and its scan add the same weights in different orders, and at u = 1 − 2⁻²⁴ the last scanned sum can
        end at or below u × total. The fallback is the last survivor with a positive weight.
      </p>
      <Ask q={questionsSX18['x1/X1.8/q-u1']} />

      <h2>One block per row</h2>
      <p>
        The kernel gives each row one block. Phase 1 is chapter 10's two-stage reduction: each thread sums a strided slice,
        each warp sums its threads with a shuffle tree, and warp 0 sums the warp sums. Thread 0 writes the total to shared
        memory; after barrier A every thread reads it and forms its target; barrier B lets the array be reused.
      </p>
      <Source section="10.8" pages="241–244" />
      <p>
        Phase 2 walks the row in chunks of blockDim × 4 survivors, four consecutive survivors per thread. For each chunk the
        block computes the chunk's sum (each warp scans its threads' totals with shuffles; the last lane writes the warp total
        to shared memory; every thread adds those). A chunk that leaves the running aggregate at or below the target is
        skipped. The chunk that crosses it is the only one scanned to the end: each thread adds the aggregate and everything
        before it to its own inclusive sums and flags every survivor whose sum exceeds the target.
      </p>
      <Source section="11.4" pages="261–265" />
      <p>
        Then the vote. A ballot gives each warp a mask of the lanes holding a flag; its lowest set bit is the warp's first
        flagged lane; an <code>atomicMin</code> on a shared index keeps the lowest across warps. That is chapter 12's leader
        election, choosing the lowest instead of any.
      </p>
      <Source section="12.3" pages="291–295" />
      <Today checked="2026-11-20 (notes/today/x1-facts.md A6)">
        <p>
          FlashInfer's <code>SamplingFromProbKernel</code> has this shape: one block per row, chunks of{' '}
          <code>BLOCK_THREADS * VEC_SIZE</code>, a chunk scanned only if <code>aggregate + aggregate_local &gt; u</code>, then{' '}
          <code>BlockScan</code> and the first lane past u.
        </p>
      </Today>
      <p>
        The stepper runs this kernel on one row. Step through the 200-survivor row first: two chunks are skipped on their
        sums alone and only the third is scanned. Then use the barrier switch to remove one of the two barriers around the
        read of the total, and pick the order in which the threads run.
      </p>
      <Simplification>
        <p>
          One block of 16 threads with a warp width you choose for display; float32 imitated by rounding after every add. The
          orders are the course emulator's in-order and reversed schedules (each thread runs to its next barrier or shuffle),
          plus one a GPU may produce. No timing.
        </p>
      </Simplification>
      <FindUStepper />
      <p>
        Two barriers bracket the read of the total, and they fail differently. Remove A and the threads of other warps read
        scratch_s[0] before thread 0 has written the total there; they find warp 0's partial sum, aim too low, and on the
        120-survivor row the block writes the wrong token in both emulator orders. Remove B and both emulator orders still
        pass: every thread reads the total before its first shuffle, and nobody writes a warp total until a whole warp is
        through its scan. Only a warp that runs ahead overwrites the total under the others, which a GPU may or may not do.
      </p>
      <Ask q={questionsSX18['x1/X1.8/q-barrier']} />
      <FindUScene />

      <h2>Exact agreement across backends</h2>
      <p>
        u is bit-identical everywhere. The cumulative sums are not. <code>exp</code> differs by backend within its documented
        error, a float sum depends on its order of additions, CUDA contracts multiply-adds by default, and Metal may round
        toward zero and defaults to fast math. A draw whose target lies within a few float steps of a boundary can pick the
        neighbouring token on another backend.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md C5; ch10-facts A9, A11, B4, C4; ch05-facts C7)">
        <p>
          MSL: "Either round ties to even or round toward zero rounding mode may be supported." Floating-point addition is not
          associative, in NVIDIA's own words.
        </p>
      </Today>
      <p>
        So the test that compares tokens across backends classifies before it compares. On the CPU, in double, it computes each
        draw's margin, the distance from u × total to the nearest boundary. It demands exact agreement only for decisive draws
        and accepts either neighbour for the rest. The explorer shows how wide "the rest" is: forty survivors, three orders
        of summation, and a perturbation of the weights standing in for another backend's <code>exp</code>.
      </p>
      <Simplification>
        <p>
          Sums are rounded with Math.fround in the stated orders. Backend <code>exp</code> differences are not modelled except
          as the perturbation you set, a relative change per weight drawn from Philox.
        </p>
      </Simplification>
      <BoundaryExplorer />
      <Ask q={questionsSX18['x1/X1.8/q-risk']} label="Work it out" />

      <h2>Same distribution, different tokens</h2>
      <p>
        The inverse CDF is not the only exact sampler. <code>torch.multinomial(probs, 1)</code>, which transformers'{' '}
        <code>generate</code> calls, runs an exponential race: draw q ~ Exp(1) per token and take argmax(p / q). MLX's{' '}
        <code>categorical</code> uses an inverse CDF for one distribution with a sample shape and Gumbel-max, argmax(l + g),
        otherwise. llama.cpp, FlashInfer and MLX at batch 1 use the inverse CDF.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D4, D5, D7)">
        <p>
          PyTorch: "s = argmax( p / q ) where q ~ Exp(1)", on CPU, CUDA, MPS and XPU. MLX's inverse-CDF path is commented "O(N
          + M) in memory, where the Gumbel-max trick needs O(N * M)".
        </p>
      </Today>
      <Ask q={questionsSX18['x1/X1.8/q-multinomial']} />
      <p>
        The explorer feeds the same Philox uniforms to all three methods: uniform 0 to the inverse CDF, one uniform per token
        to the race and to Gumbel-max.
      </p>
      <Simplification>
        <p>
          Double precision; q = −ln(1 − u) and g = −ln(−ln u) from M5's uniforms. PyTorch's exponential is computed differently
          on each device; that is not modelled.
        </p>
      </Simplification>
      <SamplersExplorer />
      <p>
        All three converge to the same frequencies; draw by draw they agree far less often. This is one more reason "same
        seed, same token" never holds across frameworks. Now tick the box: with q = −ln u the race picks Gumbel-max's token on
        every draw, because argmax(p / q) = argmax(ln p − ln q), and −ln q = −ln(−ln u) is a Gumbel draw. That is the identity
        PyTorch's comment derives the race from; how each one turns bits into q or g decides which token you get.
      </p>
      <Ask q={questionsSX18['x1/X1.8/q-mlx-batch']} />

      <h2>Who ships it</h2>
      <p>
        FlashInfer's kernel, which vLLM can call on CUDA, is this design in production, with{' '}
        <code>curand_init(seed, row, offset)</code> for u. llama.cpp's experimental backend path builds the draw from graph
        ops: softmax, an inclusive cumsum, a step at u, and a count of entries above it. Its u is drawn on the host and
        uploaded. By our reading, a double just below 1 that rounds to 1.0f in that upload makes it pick the last candidate,
        which can be a masked one, about once in 2²⁵ draws.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A6, D1, D3)">
        <p>
          llama.cpp: <code>ggml_soft_max</code> → <code>ggml_cumsum</code> → <code>ggml_step</code> → <code>ggml_sum</code>;
          "We sample in double precision and cast to float". Backend sampling is off by default for generation.
        </p>
      </Today>
      <Ask q={questionsSX18['x1/X1.8/q-device-u']} />

      <h2>On HIP and Metal</h2>
      <p>
        The vote is where the backends differ. HIP's ballot is 64 bits on every target, so the mask is a 64-bit type and the
        first lane comes from <code>__ffsll</code>; <code>__ffs</code> would see only the low 32 lanes. On Metal,{' '}
        <code>simd_ballot</code> returns a 64-bit vote whose top bits are undefined on a narrower SIMD-group; the first flagged
        lane is <code>ctz</code> of the vote, which never reaches those bits when some lane voted.{' '}
        <code>simd_prefix_inclusive_sum</code> scans a SIMD-group in one call, in an order the spec does not state.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md B1, C2; ch11-facts.md C1, C5; infra-facts B3)">
        <p>
          <code>ctz(x)</code>: "If x is 0, returns the size in bits of the type of x". One threadgroup per row means no
          threadgroup ever waits for another; such waiting (a look-back) is what Metal does not document as safe.
        </p>
      </Today>
      {cu ? (
        <Code
          tabs={[{ label: 'CUDA and HIP', code: cu, status: 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware', highlight: barrierLines }]}
          caption={
            <>
              <code>labs/x1/snippets/draw_search.cu</code>, the chunk loop and the vote as lab X1.3's reference has them; barriers A
              and B are highlighted. <code>draw.test.ts</code> runs it on the emulator, with each barrier removed in turn, and
              compares its tokens with the model.
            </>
          }
        />
      ) : (
        <p className="muted">The snippet (<code>labs/x1/snippets/draw_search.cu</code>) appears here once loaded.</p>
      )}
      <p>
        Lab X1.3 on the <a href={href('x1', 'labs')}>labs page</a> assembles the whole sampler, milestone M5, and tests it
        across backends with the margin rule. Lab X1.4 plants defects of this kind in a fused filter for you to find.
      </p>
    </>
  );
}
