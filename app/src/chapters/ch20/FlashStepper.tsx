import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Bench, TierLegend } from '../../components/notes';
import {
  DEFECTS,
  LANE_LABEL,
  OUTLINE,
  TOY,
  WARP_LABEL,
  exactRow,
  fmtv,
  makeProblem,
  panels,
  rowOk,
  runBlock,
  runKernel,
  withDefects,
  type DefectId,
  type KConfig,
  type KStep,
  type LaneSchedule,
  type Op,
  type WarpSchedule,
} from '../../sim/flashkernel';
import { FlashBlockView } from './FlashBlockView';

// "Step one block": the flash kernel of Figs. 20.8–20.16 as lab 20.2 corrects it, at a toy
// size, with its barriers and the book's slips switchable. Every value comes from
// src/sim/flashkernel.ts.

const LANES: LaneSchedule[] = ['lockstep', 'apart'];
const WARPS: WarpSchedule[] = ['together', 'ahead'];
const SCHEDULES = LANES.flatMap((l) => WARPS.map((w) => [l, w] as const));

/** Which outline line a step belongs to. */
function outlineIndex(s: KStep | undefined): number {
  if (!s) return -1;
  const op: Op = s.op === 'release_syncthreads' ? 'syncthreads' : s.op === 'release_syncwarp' ? 'syncwarp' : s.op;
  if (op === 'syncthreads') return s.barrier === 'end' ? 8 : 2;
  return OUTLINE.findIndex((o) => o.op === op);
}

function Outline({ cur, cfg }: { cur: KStep | undefined; cfg: KConfig }) {
  const hl = outlineIndex(cur);
  return (
    <ol className="mono" style={{ listStyle: 'none', padding: '0.5rem 0.6rem', margin: 0, background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 4, fontSize: '0.74rem', lineHeight: 1.7 }} data-outline>
      <li className="muted">for (i = blockIdx.x; i &lt; T_r; i += gridDim.x)</li>
      {OUTLINE.map((o, k) => {
        const removed = (o.op === 'syncwarp' && !cfg.syncwarp) || (k === 8 && !cfg.barrierAfter);
        const header =
          k === 1 ? <li className="muted" style={{ paddingLeft: '1.2em' }}>for (j : tiles{cfg.order === 'rtl' ? ', right to left' : ''})</li> :
          k === 3 ? <li className="muted" style={{ paddingLeft: '2.4em' }}>for (ii = 0; ii &lt; B_r_warp; ii++)</li> : null;
        return (
          <FragmentLine key={k} header={header}>
            <li
              data-line={k}
              className={hl === k ? 'cur' : ''}
              style={{
                paddingLeft: `${1.2 * (o.depth + 1)}em`,
                background: hl === k ? 'var(--ink)' : undefined,
                color: hl === k ? 'var(--page)' : removed ? 'var(--fail)' : undefined,
                textDecoration: removed ? 'line-through' : undefined,
                borderRadius: 2,
              }}
            >
              {o.text}
              {removed ? ' (removed)' : ''}
              {k === 4 && cfg.returnsNewMax ? <span style={{ color: hl === k ? undefined : 'var(--fail)' }}> returns the new max</span> : null}
            </li>
          </FragmentLine>
        );
      })}
    </ol>
  );
}
function FragmentLine({ header, children }: { header: ReactNode; children: ReactNode }) {
  return (
    <>
      {header}
      {children}
    </>
  );
}

export function FlashStepper() {
  const [n, setN] = useState(12);
  const [brWarp, setBrWarp] = useState(2);
  const [bc, setBc] = useState(4);
  const [panel, setPanel] = useState(1);
  const [lanes, setLanes] = useState<LaneSchedule>('lockstep');
  const [warps, setWarps] = useState<WarpSchedule>('together');
  const [defects, setDefects] = useState<DefectId[]>([]);
  const [at, setAt] = useState(0);
  const [playing, setPlaying] = useState(false);

  const cfg = useMemo(() => withDefects({ ...TOY, n, brWarp, bc }, defects), [n, brWarp, bc, defects]);
  const prob = useMemo(() => makeProblem(n, TOY.d), [n]);
  const nPanels = panels(cfg);
  const p = Math.min(panel, nPanels - 1);
  const trace = useMemo(() => runBlock(cfg, prob, p, lanes, warps), [cfg, prob, p, lanes, warps]);
  const total = trace.steps.length;
  const k = Math.min(at, total);
  const cur = k > 0 ? trace.steps[k - 1] : undefined;
  const snap = trace.snaps[k];
  const firstBad = trace.violations.length ? trace.violations[0].step + 1 : null;
  const BR = cfg.warps * cfg.brWarp;

  useEffect(() => { setAt(0); setPlaying(false); }, [cfg, p, lanes, warps]);
  useEffect(() => {
    if (!playing) return;
    if (k >= total) return setPlaying(false);
    const id = window.setTimeout(() => setAt((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, k, total]);

  const jump = (pred: (s: KStep) => boolean) => {
    for (let i = k; i < total; i++) if (pred(trace.steps[i])) return setAt(i + 1);
    setAt(total);
  };
  // Just past the first load of the next tile; once no tile is left, to the end.
  const nextTile = () => {
    const t = cur?.tile;
    for (let i = k; i < total; i++) {
      const s = trace.steps[i];
      if (s.op === 'load_KT_and_V' && s.tile !== t) return setAt(i + 1);
      if (s.op === 'store_O') break;
    }
    setAt(total);
  };
  const toggle = (id: DefectId) => setDefects((d) => (d.includes(id) ? d.filter((x) => x !== id) : [...d, id]));

  const storedRows = Object.keys(snap.stored).map(Number);
  const allOk = storedRows.length > 0 && storedRows.every((r) => rowOk(prob, r, snap.stored[r]));
  const finished = k >= total;

  return (
    <Bench
      title="Step one block"
      task="Find where each barrier matters: switch on one defect at a time and try both schedules."
      foot={
        <>
          A toy configuration: d = {cfg.d}, warps of {cfg.lanes} lanes (not 32, so a warp fits on screen), {cfg.warps} warps,
          B_r = {BR}, B_c = {cfg.bc}. Hover a cell for its value. <TierLegend />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field">
          Rows N
          <select value={n} onChange={(e) => { setN(+e.target.value); setPanel(1); }} aria-label="Rows N">
            {[8, 10, 12].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          B_r_warp
          <select value={brWarp} onChange={(e) => { setBrWarp(+e.target.value); setPanel(0); }} aria-label="B_r_warp">
            {[1, 2].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          B_c
          <select value={bc} onChange={(e) => setBc(+e.target.value)} aria-label="B_c">
            {[2, 4, 6].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          Block (Q panel)
          <select value={p} onChange={(e) => setPanel(+e.target.value)} aria-label="Block">
            {Array.from({ length: nPanels }, (_, i) => (
              <option key={i} value={i}>i = {i}: rows {i * BR} to {Math.min(n, (i + 1) * BR) - 1}</option>
            ))}
          </select>
        </label>
        <label className="field">
          Lanes
          <select value={lanes} onChange={(e) => setLanes(e.target.value as LaneSchedule)} aria-label="Lanes">
            {LANES.map((l) => <option key={l} value={l}>{LANE_LABEL[l]}</option>)}
          </select>
        </label>
        <label className="field">
          Warps
          <select value={warps} onChange={(e) => setWarps(e.target.value as WarpSchedule)} aria-label="Warps">
            {WARPS.map((w) => <option key={w} value={w}>{WARP_LABEL[w]}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }} data-defects>
        {DEFECTS.map((d) => (
          <label key={d.id} className="check" data-defect={d.id}>
            <input type="checkbox" checked={defects.includes(d.id)} onChange={() => toggle(d.id)} /> {d.label}
          </label>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setAt(0); }} disabled={k === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setAt(k - 1); }} disabled={k === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setAt(k + 1); }} disabled={finished}>Step</button>
        <button onClick={() => { setPlaying(false); jump((s) => s.op === 'release_syncthreads' || s.op === 'release_syncwarp'); }} disabled={finished}>Run to next barrier</button>
        <button onClick={() => { setPlaying(false); nextTile(); }} disabled={finished}>Next tile</button>
        <button onClick={() => setPlaying(!playing)} disabled={finished}>{playing ? 'Pause' : 'Play'}</button>
        {firstBad ? <button onClick={() => { setPlaying(false); setAt(firstBad); }}>Go to the first failure</button> : null}
        <span className="muted" aria-live="polite" data-progress>step {k} of {total}</span>
      </div>

      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }} data-narrate>
        {cur ? (
          <>
            {cur.text}
            {cur.violation ? <><br /><b className="t-fail">Failure: {cur.violation.message}.</b></> : null}
          </>
        ) : (
          <>
            Nothing has run yet. Block {p} owns rows {p * BR} to {(p + 1) * BR - 1} of Q and O
            {(p + 1) * BR > n ? ` (rows from ${n} on are past N)` : ''}:{' '}
            {Array.from({ length: cfg.warps }, (_, w) => `warp ${w} rows ${p * BR + w * cfg.brWarp}${cfg.brWarp > 1 ? `–${p * BR + w * cfg.brWarp + cfg.brWarp - 1}` : ''}`).join(', ')}. Press Step.
          </>
        )}
      </div>

      <div style={{ overflowX: 'auto', marginBottom: '0.8rem' }}>
        <div style={{ minWidth: 560 }}>
          <FlashBlockView trace={trace} prob={prob} at={k} />
        </div>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 4fr) minmax(0, 8fr)' }}>
        <div style={{ minWidth: 0 }}>
          <Outline cur={cur} cfg={cfg} />
          <p className="muted" style={{ fontSize: '0.74rem', margin: '0.5rem 0 0' }}>
            The structure of lab 20.2's kernel. With a defect on, a removed barrier is struck through.
          </p>
        </div>
        <div style={{ minWidth: 0, overflowX: 'auto' }}>
          <table className="data" data-rows style={{ margin: 0, background: 'var(--page)' }}>
            <thead>
              <tr><th>row</th><th>m</th><th>D</th><th>tiles merged</th><th>Õ/D vs exact over those tiles</th><th>stored</th></tr>
            </thead>
            <tbody>
              {Array.from({ length: BR }, (_, lr) => {
                const row = p * BR + lr;
                const w = Math.floor(lr / cfg.brWarp);
                const ii = lr % cfg.brWarp;
                const per = snap.merged[lr];
                const same = per.every((t) => t.length === per[0].length && t.every((x, i) => x === per[0][i]));
                const reg0 = snap.regs[w * cfg.lanes];
                const est = Array.from({ length: cfg.d }, (_, dd) => {
                  const r = snap.regs[w * cfg.lanes + (dd % cfg.lanes)];
                  return r.O[ii][Math.floor(dd / cfg.lanes)] / r.D[ii];
                });
                const cols = per[0].flatMap((t) => Array.from({ length: cfg.bc }, (_, q) => t * cfg.bc + q));
                const ex = row < n ? exactRow(prob, row, cols) : null;
                const err = ex ? Math.max(...est.map((x, i) => Math.abs(x - ex[i]))) : null;
                const st = snap.stored[row];
                return (
                  <tr key={lr} data-row={row}>
                    <td>{row}{row >= n ? ' (past N)' : ''}</td>
                    <td className="t-reg">{fmtv(reg0.m[ii])}</td>
                    <td className="t-reg">{fmtv(reg0.D[ii])}</td>
                    <td>{same ? (per[0].length ? per[0].join(', ') : 'none') : per.map((t, L) => `lane ${L}: ${t.join(',') || '–'}`).join('; ')}</td>
                    <td>
                      {row >= n ? <span className="muted">not stored</span>
                        : !same ? <span className="muted">lanes not level yet</span>
                        : ex == null ? <span className="muted">nothing visible yet</span>
                        : Number.isNaN(err!) ? <span className="chip fail" data-live="nan">NaN</span>
                        : <span className={'chip ' + (err! < 1e-9 ? 'ok' : 'fail')} data-live={err! < 1e-9 ? 'ok' : 'off'}>{err! < 1e-9 ? 'equal' : `off by ${fmtv(err!)}`}</span>}
                    </td>
                    <td data-stored={st ? (rowOk(prob, row, st) ? 'ok' : 'wrong') : 'no'}>
                      {st ? (rowOk(prob, row, st) ? <span className="chip ok">O and L correct</span> : <span className="chip fail">wrong</span>) : <span className="muted">not yet</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
      {finished ? (
        <p style={{ margin: '0.8rem 0 0' }} data-block-result={allOk ? 'ok' : 'wrong'}>
          {allOk ? <span className="chip ok">This block's rows match exact attention.</span> : <span className="chip fail">This block's results are wrong.</span>}{' '}
          {trace.violations.length ? `${trace.violations.length} bad reads of shared memory.` : 'No bad read of shared memory.'}
        </p>
      ) : null}
    </Bench>
  );
}

/** Every defect against every schedule, for the whole 12-row kernel. */
export function FlashOutcomes() {
  const prob = useMemo(() => makeProblem(12, TOY.d), []);
  const rows: { label: string; on: DefectId[] }[] = [{ label: 'corrected kernel (lab 20.2)', on: [] }, ...DEFECTS.map((d) => ({ label: d.label, on: [d.id] }))];
  return (
    <Bench
      title="Every schedule at once"
      task="Before reading the table: which defect is wrong under every schedule, and which only under one?"
      foot="Each cell runs all three blocks of the toy kernel (N = 12) in the model above and compares O and L with exact attention."
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-outcomes>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Kernel</th>
              {SCHEDULES.map(([l, w]) => <th key={l + w}>{LANE_LABEL[l]}, {WARP_LABEL[w]}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label} data-kernel={r.on[0] ?? 'correct'}>
                <td style={{ paddingLeft: '0.6rem' }}>{r.label}</td>
                {SCHEDULES.map(([l, w]) => {
                  const res = runKernel(withDefects(TOY, r.on), prob, l, w);
                  return (
                    <td key={l + w} data-outcome={res.correct ? 'correct' : 'wrong'}>
                      {res.correct ? <span className="chip ok">correct</span> : <span className="chip fail">{res.wrongRows.length} of 12 rows wrong</span>}
                      {res.violations.length ? <div className="muted">{res.violations.length} bad reads</div> : null}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}
