import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { Bench } from '../../components/notes';
import {
  PRESETS,
  UNVISITED,
  VARIANT_NAME,
  runBFS,
  showLevel,
  toCOO,
  toCSC,
  toCSR,
  workBounds,
  type Graph,
  type LevelRun,
  type ThreadInfo,
  type Variant,
} from '../../sim/bfs';
import { BfsGraph, GlyphLegend, type Glyph } from './BfsGraph';

// "One graph, five ways to run a level" (section 18.3). Every number it shows comes from
// src/sim/bfs.ts; this file only lays them out and writes the sentences.

export interface ExplorerRequest {
  preset: number;
  root: number;
  variant: Variant;
  step: number;
  /** Changes on every request, so the same request can be made twice. */
  key: number;
}

const VARIANTS: Variant[] = ['push', 'pull', 'edge', 'frontier', 'diropt'];
/** Blocks and warps of 4 threads, so that a whole call of the small graphs fits on screen. */
const BLOCK_SIZE = 4;
const WARP_WIDTH = 4;
const KERNEL_SHORT: Record<string, string> = { push: 'push', pull: 'pull', edge: 'edge', frontier: 'frontier' };

function itemName(c: LevelRun, t: ThreadInfo, g: Graph): string {
  if (t.item === null) return '·';
  if (c.kernel === 'edge') {
    const [s, d] = g.edges[t.item];
    return `${s}→${d}`;
  }
  if (c.kernel === 'frontier') return `[${t.item}] ${t.vertex}`;
  return `v${t.item}`;
}

/** One sentence per thing the thread did, from the model's record of what it read. */
function describe(c: LevelRun, t: ThreadInfo, g: Graph, blockSize: number): ReactNode {
  const L = c.call;
  const idx = `thread ${t.t} = blockIdx ${t.block} × blockDim ${blockSize} + threadIdx ${t.lane}`;
  if (t.item === null)
    return (
      <>
        <b>{idx}.</b> There are only {c.items} {c.kernel === 'edge' ? 'edges' : c.kernel === 'frontier' ? 'frontier entries' : 'vertices'}, so it fails the bounds check and returns: one of the last block's {c.surplus} surplus threads.
      </>
    );
  const b = c.before;
  if (c.kernel === 'push') {
    const v = t.item;
    if (t.role === 'idle') return <><b>{idx}</b> owns vertex {v}. It reads <code>level[{v}] = {showLevel(b[v])}</code>, which is not {L - 1}, and stops.</>;
    const csr = toCSR(g);
    const outs = csr.dst.slice(csr.srcPtrs[v], csr.srcPtrs[v + 1]);
    return (
      <>
        <b>{idx}</b> owns vertex {v}, which is at level {L - 1}. It walks <code>dst[{csr.srcPtrs[v]}..{csr.srcPtrs[v + 1] - 1}]</code>:{' '}
        {outs.map((u) => `${u} (${b[u] === UNVISITED ? 'unvisited' : 'level ' + b[u]})`).join(', ')}.{' '}
        {t.labels.length ? <>It writes level {L} to {t.labels.join(' and ')}.</> : <>Every neighbour is already labelled; it writes nothing.</>}
      </>
    );
  }
  if (c.kernel === 'pull') {
    const v = t.item;
    if (t.role === 'idle') return <><b>{idx}</b> owns vertex {v}. It reads <code>level[{v}] = {showLevel(b[v])}</code>: already visited, so it stops.</>;
    const csc = toCSC(g);
    const ins = csc.src.slice(csc.dstPtrs[v], csc.dstPtrs[v + 1]);
    const seen = ins.slice(0, t.trips);
    return (
      <>
        <b>{idx}</b> owns vertex {v}, still unvisited. It scans its in-edges <code>src[{csc.dstPtrs[v]}..]</code>:{' '}
        {seen.map((u) => `${u} (${b[u] === UNVISITED ? 'unvisited' : 'level ' + b[u]})`).join(', ')}
        {t.labels.length ? <>. {seen[seen.length - 1]} is at level {L - 1}: it labels {v} with {L} and breaks after {t.trips} of {ins.length}.</> : <>. No parent at level {L - 1}: it scanned all {ins.length} and labels nothing.</>}
      </>
    );
  }
  if (c.kernel === 'edge') {
    const coo = toCOO(g);
    const e = t.item;
    const [s, d] = [coo.src[e], coo.dst[e]];
    if (t.role === 'idle') return <><b>{idx}</b> owns edge {e}, {s} → {d}. It reads <code>level[{s}] = {showLevel(b[s])}</code>, which is not {L - 1}, and stops.</>;
    return (
      <>
        <b>{idx}</b> owns edge {e}, {s} → {d}. Its source is at level {L - 1}, so it reads <code>level[{d}] = {showLevel(b[d])}</code>
        {t.labels.length ? <> and writes {L} there.</> : <>: already labelled, nothing to write.</>}
      </>
    );
  }
  const v = t.vertex!;
  const csr = toCSR(g);
  const outs = csr.dst.slice(csr.srcPtrs[v], csr.srcPtrs[v + 1]);
  return (
    <>
      <b>{idx}</b> owns frontier slot {t.item}: <code>prevFrontier[{t.item}] = {v}</code>. It walks {outs.length ? outs.join(', ') : 'no edges'}
      {outs.length ? <>, a compare-and-swap on each; {t.wins.length ? <>in the model's order it wins {t.wins.join(' and ')} and appends {t.wins.length === 1 ? 'it' : 'them'}</> : <>it wins none</>}.</> : '.'}
      {t.labels.length > t.wins.length ? <> ({t.labels.filter((u) => !t.wins.includes(u)).join(', ')} went to another thread first.)</> : null}
    </>
  );
}

const roleStyle = (role: ThreadInfo['role'], selected: boolean): CSSProperties => ({
  minWidth: '2.6rem',
  height: '2rem',
  background: role === 'surplus' ? 'transparent' : 'var(--cell)',
  border: role === 'labels' ? '2px solid var(--ink)' : role === 'iterates' ? '1.5px dashed var(--ink)' : role === 'surplus' ? '1px dashed var(--rule)' : '1px solid var(--rule)',
  color: role === 'idle' || role === 'surplus' ? 'var(--ink-3)' : 'var(--ink)',
  fontWeight: role === 'labels' ? 650 : 450,
  outline: selected ? '2px solid var(--ink)' : undefined,
  outlineOffset: selected ? 1 : undefined,
  cursor: 'pointer',
});

export function LevelExplorer({ request }: { request?: ExplorerRequest | null }) {
  const [preset, setPreset] = useState(0);
  const g: Graph = PRESETS[preset];
  const [root, setRoot] = useState(PRESETS[0].root);
  const [variant, setVariant] = useState<Variant>('push');
  const blockSize = BLOCK_SIZE;
  const warpWidth = WARP_WIDTH;
  const [switchCall, setSwitchCall] = useState(3);
  const [step, setStep] = useState(0);
  const [sel, setSel] = useState<number | null>(null);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!request) return;
    setPreset(request.preset);
    setRoot(request.root);
    setVariant(request.variant);
    setStep(request.step);
    setSel(null);
    box.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [request?.key]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = useMemo(() => runBFS(g, { variant, root, blockSize, warpWidth, switchCall }), [g, variant, root, blockSize, warpWidth, switchCall]);
  const total = run.calls.length;
  const s = Math.min(step, total);
  const cur = s > 0 ? run.calls[s - 1] : null;
  const shown = cur ? cur.after : run.calls[0].before;
  const fresh = new Set(cur?.newVertices ?? []);
  const bounds = workBounds(g, run.d);

  // Writers per vertex in this call, for the "may be written twice" ring.
  const writers = new Map<number, number>();
  if (cur && (cur.kernel === 'push' || cur.kernel === 'edge')) for (const t of cur.threads) for (const u of t.labels) writers.set(u, (writers.get(u) ?? 0) + 1);

  const glyphs: Glyph[] = cur
    ? cur.threads.flatMap((t): Glyph[] =>
        t.item === null ? [] : cur.kernel === 'edge' ? [{ edge: t.item, role: t.role, t: t.t, selected: sel === t.t }] : [{ vertex: t.vertex!, role: t.role, t: t.t, selected: sel === t.t }],
      )
    : [];
  const reset = () => {
    setStep(0);
    setSel(null);
  };
  const choosePreset = (k: number) => {
    setPreset(k);
    setRoot(PRESETS[k].root);
    reset();
  };
  const clickVertex = (v: number) => {
    const t = cur?.threads.find((x) => x.item !== null && (cur.kernel === 'edge' ? false : x.vertex === v));
    if (t) setSel(t.t);
  };
  const selected = cur && sel !== null ? cur.threads[sel] : undefined;
  const warpOf = (t: number) => cur?.warps.findIndex((w) => t >= w.first && t < w.first + w.size) ?? -1;

  const legend =
    cur?.kernel === 'edge'
      ? { idle: 'source not in the last level', iterates: 'source in it, destination visited', labels: 'source in it, may label' }
      : cur?.kernel === 'pull'
        ? { idle: 'already visited: stops', iterates: 'scans, finds no parent', labels: 'finds a parent, labels itself' }
        : cur?.kernel === 'frontier'
          ? { idle: '', iterates: 'walks its edges, labels nothing', labels: 'walks its edges, labels' }
          : { idle: 'not in the last level: stops', iterates: 'loops, labels nothing', labels: 'loops and labels' };

  return (
    <div ref={box} className="wide" style={{ scrollMarginTop: '1rem' }}>
      <Bench
        title="One graph, five ways to run a level"
        task="Step through the calls. Count the threads that do nothing, and the edges read for nothing."
        foot={<GlyphLegend {...legend} />}
      >
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          <label className="field">
            Graph
            <select value={preset} onChange={(e) => choosePreset(+e.target.value)} aria-label="Graph">
              {PRESETS.map((p, k) => <option key={p.id} value={k}>{p.name}</option>)}
            </select>
          </label>
          <label className="field">
            Root
            <select value={root} onChange={(e) => { setRoot(+e.target.value); reset(); }} aria-label="Root">
              {Array.from({ length: g.n }, (_, v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label className="field">
            Kernel
            <select value={variant} onChange={(e) => { setVariant(e.target.value as Variant); reset(); }} aria-label="Kernel">
              {VARIANTS.map((v) => <option key={v} value={v}>{VARIANT_NAME[v]}</option>)}
            </select>
          </label>
          {variant === 'diropt' ? (
            <label className="field">
              Pull from call {switchCall}
              <input type="range" min={1} max={Math.max(2, total + 1)} value={switchCall} aria-label="First call that pulls"
                onChange={(e) => { setSwitchCall(+e.target.value); }} />
            </label>
          ) : null}
        </div>
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <button onClick={() => { setStep(0); setSel(null); }} disabled={s === 0}>Restart</button>
          <button onClick={() => { setStep(s - 1); setSel(null); }} disabled={s === 0}>Back</button>
          <button className="primary" onClick={() => { setStep(s + 1); setSel(null); }} disabled={s >= total}>Step</button>
          <button onClick={() => { setStep(total); setSel(null); }} disabled={s >= total}>Run to end</button>
          <span className="muted" aria-live="polite" data-call={s}>
            {s === 0 ? `before the first call: root ${root} at level 0` : `call ${s} of ${total}: ${KERNEL_SHORT[cur!.kernel]} kernel labels level ${s}`}
          </span>
        </div>

        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 6fr)' }}>
          <div>
            <BfsGraph
              g={g}
              label={`${g.name}, ${cur ? 'call ' + s : 'before the first call'}`}
              look={(v) => (v === root ? 'root' : fresh.has(v) ? 'new' : shown[v] === UNVISITED ? 'unvisited' : 'visited')}
              tag={(v) => (shown[v] === UNVISITED ? null : String(shown[v]))}
              ring={(v) => ((writers.get(v) ?? 0) > 1 ? 'fail' : null)}
              edgeLook={(_, a, b) =>
                !cur || cur.kernel === 'edge' ? 'plain'
                : cur.kernel === 'pull' ? (cur.before[b] === UNVISITED ? 'plain' : 'dim')
                : cur.before[a] === cur.call - 1 ? 'plain' : 'dim'}
              glyphs={glyphs}
              onVertex={clickVertex}
              onGlyph={(t) => setSel(t)}
            />
            <p className="muted" style={{ margin: '0.4rem 0 0' }}>
              n = {g.n} vertices, m = {g.edges.length} edges, d = {run.d} levels after level 0. Amber: labelled, darker for this call's new labels.
              {writers.size && [...writers.values()].some((x) => x > 1) ? ' A dashed red ring: more than one thread may write it.' : ''}
            </p>
          </div>
          <div style={{ minWidth: 0 }}>
            <div className="narrate" role="status" style={{ marginBottom: '0.7rem' }} data-thread-text>
              {cur ? (selected ? describe(cur, selected, g, blockSize) : <>Click a thread (a zigzag on the graph, or a cell below) to see what it reads.</>) : <>Press Step to launch the first call.</>}
            </div>
            {cur ? (
              <div data-call-summary data-launched={cur.launched} data-past={cur.pastCheck} data-labelling={cur.threads.filter((t) => t.role === 'labels').length}
                data-looping={cur.threads.filter((t) => t.role === 'labels' || t.role === 'iterates').length}>
                <div className="mx-label">
                  {cur.blocks} block{cur.blocks === 1 ? '' : 's'} of {blockSize}: thread → {cur.kernel === 'edge' ? 'edge' : cur.kernel === 'frontier' ? 'frontier slot, then its vertex' : 'vertex'}
                </div>
                <div style={{ overflowX: 'auto' }}>
                  {Array.from({ length: cur.blocks }, (_, b) => (
                    <div key={b} className="row" style={{ gap: '0.5rem', marginBottom: '0.35rem', flexWrap: 'nowrap' }}>
                      <span className="muted" style={{ minWidth: '4.2rem' }}>block {b}</span>
                      {cur.warps.filter((w) => w.block === b).map((w) => (
                        <span key={w.first} style={{ display: 'inline-flex', flexDirection: 'column', gap: 2 }}>
                          <span className="mx" style={{ gridTemplateColumns: `repeat(${w.size}, auto)` }}>
                            {cur.threads.slice(w.first, w.first + w.size).map((t) => (
                              <span key={t.t} data-t={t.t} data-role={t.role} style={roleStyle(t.role, sel === t.t)} onClick={() => setSel(t.t)} title={`thread ${t.t}`}>
                                {itemName(cur, t, g)}
                                {t.trips ? <sub>{t.trips}</sub> : null}
                              </span>
                            ))}
                          </span>
                          <span className="muted" style={{ fontSize: '0.7rem' }} data-warp={warpOf(w.first)}>
                            {w.passes} pass{w.passes === 1 ? '' : 'es'}, {w.useful} of {w.passes * w.size} lane slots used
                          </span>
                        </span>
                      ))}
                    </div>
                  ))}
                </div>
                <p className="muted" style={{ margin: '0.3rem 0 0' }}>
                  Small numbers: loop iterations. This call: {cur.warpPasses} warp passes, {cur.useful} useful of {cur.laneSlots} lane slots. Real warps are 32 lanes on NVIDIA, 64 or 32 on AMD, and on Metal you read the width from the pipeline.
                </p>
              </div>
            ) : null}
          </div>
        </div>

        {s > 0 ? <div style={{ overflowX: 'auto', marginTop: '0.9rem' }}>
          <table className="data" style={{ background: 'var(--page)', margin: 0 }} aria-label="Per-call accounting">
            <thead>
              <tr>
                <th style={{ paddingLeft: '0.6rem' }}>call</th>
                <th>kernel</th>
                <th className="num">launched</th>
                <th className="num">past 1st check</th>
                <th className="num">edges read</th>
                <th className="num">labels written</th>
                <th>new frontier</th>
                <th className="num">m_f</th>
                <th className="num">m_u</th>
              </tr>
            </thead>
            <tbody>
              {run.calls.slice(0, s).map((c) => (
                <tr key={c.call} data-row={c.call} style={c.call === s ? { fontWeight: 600 } : undefined}>
                  <td style={{ paddingLeft: '0.6rem' }}>{c.call}</td>
                  <td>{KERNEL_SHORT[c.kernel]}</td>
                  <td className="num">{c.launched}{c.surplus ? <span className="muted"> ({c.surplus} surplus)</span> : null}</td>
                  <td className="num">{c.pastCheck}</td>
                  <td className="num">{c.edgeWork}</td>
                  <td className="num">{c.labelsMin === c.labelsMax ? c.labelsMin : `${c.labelsMin} to ${c.labelsMax}`}</td>
                  <td>{c.nextFrontier.length ? c.nextFrontier.join(', ') : 'empty: stop'}</td>
                  <td className="num">{c.mf}</td>
                  <td className="num">{c.mu}</td>
                </tr>
              ))}
              {s > 0 ? (
                <tr data-totals>
                  <td style={{ paddingLeft: '0.6rem' }} colSpan={2}><b>so far</b></td>
                  <td className="num">{run.calls.slice(0, s).reduce((a, c) => a + c.launched, 0)}</td>
                  <td className="num">{run.calls.slice(0, s).reduce((a, c) => a + c.pastCheck, 0)}</td>
                  <td className="num">{run.calls.slice(0, s).reduce((a, c) => a + c.edgeWork, 0)}</td>
                  <td className="num" colSpan={4} />
                </tr>
              ) : null}
            </tbody>
          </table>
        </div> : null}
        {s >= total ? (
          <p data-work style={{ margin: '0.7rem 0 0' }}>
            Whole search, {total} calls: <b>{run.totals.work}</b> units of work ({run.totals.vertexWork} threads that each checked a vertex, plus {run.totals.edgeWork} edges read).
            {run.options.variant === 'push' ? <>The p. 439 bound O(d·n + m), counted here with the final empty call as (d + 1)·n + m = {bounds.push}</>
              : run.options.variant === 'pull' ? <>The p. 439 bound O(d·n + d·m), counted here with the final empty call as (d + 1)·(n + m) = {bounds.pull}</>
              : run.options.variant === 'edge' ? <>The p. 439 bound O(d·m), counted here with the final empty call as (d + 1)·m = {bounds.edge}</>
              : run.options.variant === 'frontier' ? <>The p. 439 bound O(n + m), counted here as n + m = {bounds.frontier}</>
              : <>The p. 439 bounds, counted here with the final empty call, are push's {bounds.push} and pull's {bounds.pull}; which one applies depends on the switch</>}
            ; one pass over every vertex and edge would be n + m = {bounds.frontier}.
          </p>
        ) : null}
      </Bench>
    </div>
  );
}
