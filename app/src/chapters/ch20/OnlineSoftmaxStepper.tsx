import { useEffect, useMemo, useState } from 'react';
import { Bench } from '../../components/notes';
import {
  FORM_LABEL,
  INIT_LABEL,
  ORDER_LABEL,
  PRESET_LABEL,
  ROW_N,
  V12,
  presetScores,
  runOnline,
  type Form,
  type InitMax,
  type PresetId,
  type TileOrder,
  type TileStep,
} from '../../sim/onlinesoftmax';

// Section 20.5's central interactive: one query row merged tile by tile (Eqs. 20.3–20.6).
// Every number comes from src/sim/onlinesoftmax.ts.

/** A float as the stepper prints it: 4 significant digits, the two start values by name. */
export function fv(x: number | null | undefined): string {
  if (x == null) return '—';
  if (Number.isNaN(x)) return 'NaN';
  if (x === -Infinity) return '−∞';
  if (x === Infinity) return '∞';
  if (x <= -1e37) return '−FLT_MAX/2';
  if (x !== 0 && Math.abs(x) < 1e-3) return x.toExponential(1).replace('-', '−').replace('e', '×10^');
  return Number(x.toPrecision(4)).toString().replace('-', '−');
}
const fvec = (v: number[] | null) => (v ? '(' + v.map(fv).join(', ') + ')' : '—');

const WIDTHS = [1, 2, 3, 4, 6, 12];
const PRESETS: PresetId[] = ['demo', 'random', 'late', 'offset'];
const ORDERS: TileOrder[] = ['ltr', 'rtl', 'shuffled'];
const INITS: InitMax[] = ['neg-inf', 'sentinel'];
const FORMS: Form[] = ['deferred', 'normalised'];

function Select<T extends string>({ label, value, options, labels, onChange }: {
  label: string; value: T; options: T[]; labels: Record<T, string>; onChange: (v: T) => void;
}) {
  return (
    <label className="field">
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value as T)} aria-label={label}>
        {options.map((o) => <option key={o} value={o}>{labels[o]}</option>)}
      </select>
    </label>
  );
}

function verdict(s: TileStep | undefined) {
  if (!s) return null;
  if (s.nan) return <span className="chip fail" data-check="nan">NaN</span>;
  if (s.diff == null) return <span className="muted">nothing seen yet</span>;
  return <span className={'chip ' + (s.diff < 1e-5 ? 'ok' : 'fail')} data-check={s.diff < 1e-5 ? 'ok' : 'off'}>{s.diff < 1e-5 ? 'matches' : 'differs'} ({fv(s.diff)})</span>;
}

export function OnlineSoftmaxStepper() {
  const [preset, setPreset] = useState<PresetId>('demo');
  const [bc, setBc] = useState(4);
  const [row, setRow] = useState(ROW_N - 1);
  const [order, setOrder] = useState<TileOrder>('ltr');
  const [init, setInit] = useState<InitMax>('neg-inf');
  const [form, setForm] = useState<Form>('deferred');
  const [k, setK] = useState(0);

  const scores = useMemo(() => presetScores(preset), [preset]);
  const trace = useMemo(() => runOnline({ scores, V: V12, bc, row, order, init, form }), [scores, bc, row, order, init, form]);
  useEffect(() => setK(0), [preset, bc, row, order, init, form]);

  const total = trace.steps.length;
  const cur = k > 0 ? trace.steps[k - 1] : undefined;
  const mergedTiles = new Set(trace.steps.slice(0, k).map((s) => s.tile));
  const nTiles = Math.ceil(ROW_N / bc);
  const done = k >= total;
  const startM = trace.start.m;
  const formLabel = form === 'deferred' ? 'Õ' : 'o';

  return (
    <Bench
      title="Online softmax, one tile at a time"
      task="Before each Step, say whether the running max will rise and what the factor on the old D will be."
      foot={
        <>
          One query row of {ROW_N} scores (already scaled) and a {ROW_N} × 3 V. Columns past the row index are masked by
          index, as the labs do. The check column compares the current estimate with exact attention, in double
          precision, over the visible columns merged so far.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Tile width B_c
          <span className="seg" role="group" aria-label="Tile width">
            {WIDTHS.map((w) => (
              <button key={w} aria-pressed={bc === w} onClick={() => setBc(w)}>{w}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Causal row r = {row}
          <input type="range" min={0} max={ROW_N - 1} value={row} aria-label="Causal row" onChange={(e) => setRow(+e.target.value)} />
        </label>
        <Select label="Data" value={preset} options={PRESETS} labels={PRESET_LABEL} onChange={setPreset} />
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <Select label="Tile order" value={order} options={ORDERS} labels={ORDER_LABEL} onChange={setOrder} />
        <Select label="Running max starts at" value={init} options={INITS} labels={INIT_LABEL} onChange={setInit} />
        <Select label="Form" value={form} options={FORMS} labels={FORM_LABEL} onChange={setForm} />
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setK(0)} disabled={k === 0}>Restart</button>
        <button onClick={() => setK(k - 1)} disabled={k === 0}>Back</button>
        <button className="primary" onClick={() => setK(k + 1)} disabled={done}>Step</button>
        <span className="muted" aria-live="polite" data-progress>tile {k} of {total} merged</span>
      </div>

      <div className="mx-label">The row (tiles of {bc}, merged in the order {trace.steps.map((s) => s.tile).join(', ')})</div>
      <div className="mx" data-row style={{ display: 'flex', flexWrap: 'wrap', marginBottom: '0.3rem' }}>
        {scores.map((s, j) => {
          const t = Math.floor(j / bc);
          const vis = j <= row;
          const isCur = cur?.tile === t;
          const merged = mergedTiles.has(t);
          return (
            <span
              key={j}
              className={vis ? '' : 'void'}
              data-col={j}
              data-state={isCur ? 'current' : merged ? 'merged' : 'waiting'}
              title={vis ? `column ${j}: ${s}` : `column ${j}: masked (past row ${row})`}
              style={{
                background: isCur ? 'var(--ink)' : merged && vis ? 'var(--rule-soft)' : undefined,
                color: isCur && vis ? 'var(--page)' : undefined,
                borderLeft: j % bc === 0 ? '3px solid var(--ink-2)' : undefined,
                width: '2.9rem',
              }}
            >
              {vis ? fv(s) : '×'}
            </span>
          );
        })}
      </div>
      <p className="muted" style={{ margin: '0 0 0.8rem' }}>
        × marks a masked column. Thick left edges mark tile boundaries; the highlighted tile is the one just merged, greyed ones were merged before.
        {preset === 'offset' ? <> e<sup>largest score</sup> in float32 is {fv(trace.naiveExpOfMax)}: without subtracting the max, the exponentials overflow.</> : null}
      </p>

      <div className={'narrate' + (cur?.nan ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }} data-narrate>
        {!cur ? (
          <>
            Nothing merged yet: m = {fv(startM)}, D = 0, {formLabel} = (0, 0, 0). The row has {nTiles} tile{nTiles > 1 ? 's' : ''}. Press Step.
          </>
        ) : cur.skipped ? (
          <>
            Tile {cur.tile} has no visible column. Eq. 20.5 would be 0/0 for it, so the normalised form skips it; nothing
            changes.
          </>
        ) : (
          <>
            Tile {cur.tile} (columns {cur.cols[0]}–{cur.cols[cur.cols.length - 1]}
            {cur.visible.length < cur.cols.length ? `, ${cur.visible.length ? cur.visible.length + ' visible' : 'none visible'}` : ''}):{' '}
            {cur.visible.length ? <>its own max m<sub>B</sub> = {fv(cur.mB)}. </> : <>no visible column, so its max is the start value {fv(cur.mB)}. </>}
            {cur.rose ? (
              <>The running max rises from {fv(cur.mOld)} to {fv(cur.mNew)}, so everything kept so far shrinks by e<sup>{fv(cur.mOld)} − {fv(cur.mNew)}</sup> = {fv(cur.aFactor)}.</>
            ) : Number.isNaN(cur.aFactor) ? (
              <>The running max stays {fv(cur.mOld)}, and the rescale factor e<sup>{fv(cur.mOld)} − ({fv(cur.mNew)})</sup> is NaN: (−∞) − (−∞) has no value, and {formLabel} becomes NaN for good.</>
            ) : (
              <>The running max stays {fv(cur.mNew)}: the factor on what is kept is {fv(cur.aFactor)}, and the tile's terms add on.</>
            )}
          </>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 7fr)' }}>
        <table className="data" data-detail style={{ margin: 0, background: 'var(--page)' }}>
          <caption className="muted" style={{ textAlign: 'left', captionSide: 'top', padding: '0 0 0.3rem' }}>
            This step, in the terms of Eq. 20.4 (A = merged so far, B = this tile)
          </caption>
          <tbody>
            <tr><td>m<sub>A</sub>, the running max before</td><td data-v="mA">{fv(cur?.mOld)}</td></tr>
            <tr><td>m<sub>B</sub>, the tile's max</td><td data-v="mB">{fv(cur?.mB)}</td></tr>
            <tr><td>m<sub>A∪B</sub> = max(m<sub>A</sub>, m<sub>B</sub>)</td><td data-v="mAB">{fv(cur?.mNew)}</td></tr>
            <tr><td>A term factor e<sup>m<sub>A</sub> − m<sub>A∪B</sub></sup> (on D<sub>A</sub> and {formLabel})</td><td data-v="aFactor"><b>{fv(cur?.aFactor)}</b></td></tr>
            <tr><td>B term factor e<sup>m<sub>B</sub> − m<sub>A∪B</sub></sup> (on D<sub>B</sub>)</td><td data-v="bFactor">{cur && cur.bFactor == null ? 'no visible column' : fv(cur?.bFactor)}</td></tr>
            <tr><td>D<sub>B</sub>, Eq. 20.3 over the tile</td><td data-v="DB">{fv(cur?.DB)}</td></tr>
            <tr><td>B term as the code adds it: Σ e<sup>l − m<sub>A∪B</sub></sup></td><td data-v="bTerm">{fv(cur?.bTerm)}</td></tr>
            <tr><td>D after the merge</td><td data-v="D"><b>{fv(cur ? cur.D : 0)}</b></td></tr>
            <tr><td>{formLabel} after the merge</td><td data-v="O">{fvec(cur ? cur.O : [0, 0, 0])}</td></tr>
          </tbody>
        </table>
        <div style={{ minWidth: 0, overflowX: 'auto' }}>
          <table className="data" data-history style={{ margin: 0, background: 'var(--page)' }}>
            <thead>
              <tr><th>step</th><th>tile</th><th>m</th><th>D</th><th>estimate {form === 'deferred' ? 'Õ/D' : 'o'}</th><th>exact so far</th><th>check</th></tr>
            </thead>
            <tbody>
              {trace.steps.slice(0, k).map((s) => (
                <tr key={s.k} data-step={s.k}>
                  <td>{s.k}</td>
                  <td>{s.tile}</td>
                  <td>{fv(s.mNew)}</td>
                  <td>{fv(s.D)}</td>
                  <td>{fvec(s.out)}</td>
                  <td>{fvec(s.ref)}</td>
                  <td>{verdict(s)}</td>
                </tr>
              ))}
              {k === 0 ? <tr><td colSpan={7} className="muted">No tile merged yet.</td></tr> : null}
            </tbody>
          </table>
          {done ? (
            <p style={{ margin: '0.7rem 0 0' }} data-final>
              {trace.nan ? (
                <span className="chip fail">The result is NaN.</span>
              ) : trace.diff < 1e-5 ? (
                <span className="chip ok">Equals exact attention over columns 0–{row}: float32 difference {fv(trace.diff)}.</span>
              ) : (
                <span className="chip fail">Differs from exact attention by {fv(trace.diff)}.</span>
              )}
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}
