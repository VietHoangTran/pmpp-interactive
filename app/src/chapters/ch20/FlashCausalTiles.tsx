import { useMemo, useState } from 'react';
import { Bench } from '../../components/notes';
import { tileClasses, type TileClass } from '../../sim/flashplan';

// "Which tiles does causality empty?": the (Q panel, K/V tile) pairs of a causal head,
// classified as FlashAttention-2 classifies them (facts A7). Model: src/sim/flashplan.ts.

const FILL: Record<TileClass, string> = { full: 'var(--ink-2)', mask: 'var(--ink-3)', skip: 'var(--cell)' };
const NAME: Record<TileClass, string> = {
  full: 'full: no column past any row, no mask needed',
  mask: 'crosses the diagonal: mask by index',
  skip: 'wholly above the diagonal: empty',
};
const int = (v: number) => v.toLocaleString('en-US');

export function FlashCausalTiles() {
  const [n, setN] = useState(4096);
  const [br, setBr] = useState(32);
  const [bc, setBc] = useState(32);
  const t = useMemo(() => tileClasses(n, br, bc), [n, br, bc]);
  const side = 360;
  const cell = side / Math.max(t.tr, t.tc);

  // Draw each row as runs of one class, so a 256 × 256 grid is a few hundred rectangles.
  const runs: { i: number; j0: number; j1: number; k: TileClass }[] = [];
  t.grid.forEach((row, i) => {
    let j0 = 0;
    for (let j = 1; j <= row.length; j++)
      if (j === row.length || row[j] !== row[j0]) {
        runs.push({ i, j0, j1: j, k: row[j0] });
        j0 = j;
      }
  });
  const computed = t.counts.full + t.counts.mask;

  return (
    <Bench
      title="Which tiles does causality empty?"
      task="Before moving N: about what fraction of the pairs is empty for a long sequence?"
      foot="Rows: Q panels of B_r rows. Columns: K/V tiles of B_c columns. A ragged last panel or tile is classified by its real extent."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Sequence length N = {int(n)}
          <input type="range" min={32} max={8192} step={32} value={n} aria-label="Sequence length" onChange={(e) => setN(+e.target.value)} />
        </label>
        <label className="field">
          or type N
          <input type="number" min={1} max={16384} value={n} aria-label="Sequence length exactly" onChange={(e) => setN(Math.max(1, Math.min(16384, Math.round(+e.target.value || 1))))} />
        </label>
        <label className="field">
          B_r
          <select value={br} onChange={(e) => setBr(+e.target.value)} aria-label="B_r">
            {[16, 32, 64, 128].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          B_c
          <select value={bc} onChange={(e) => setBc(+e.target.value)} aria-label="B_c">
            {[16, 32, 64, 128].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <svg viewBox={`0 0 ${side} ${side}`} shapeRendering="crispEdges" role="img" aria-label="Grid of tiles coloured by causal class" style={{ width: '100%', maxWidth: side, display: 'block', border: '1px solid var(--rule)', background: 'var(--cell)' }} data-tilegrid={`${t.tr}x${t.tc}`}>
          {runs.map((r, k) =>
            r.k === 'skip' ? null : (
              <rect key={k} x={r.j0 * cell} y={r.i * cell} width={(r.j1 - r.j0) * cell} height={cell} fill={FILL[r.k]}
                stroke={cell > 6 ? 'var(--page)' : 'none'} strokeWidth={cell > 6 ? 1 : 0} />
            ),
          )}
        </svg>
        <div>
          <table className="data" style={{ margin: 0, background: 'var(--page)' }} data-counts>
            <tbody>
              {(['full', 'mask', 'skip'] as TileClass[]).map((k) => (
                <tr key={k} data-class={k}>
                  <td><i style={{ display: 'inline-block', width: '0.8em', height: '0.8em', background: FILL[k], border: '1px solid var(--rule)', marginRight: '0.4em', verticalAlign: '-0.05em' }} />{NAME[k]}</td>
                  <td className="num">{int(t.counts[k])}</td>
                </tr>
              ))}
              <tr><td>all pairs, T_r × T_c = {int(t.tr)} × {int(t.tc)}</td><td className="num">{int(t.total)}</td></tr>
            </tbody>
          </table>
          <p data-fraction>
            Empty: <b>{int(t.counts.skip)} of {int(t.total)}</b> = {(t.counts.skip / t.total).toFixed(3)} of the pairs. The book’s
            kernel computes all {int(t.total)}; FlashAttention-2 computes {int(computed)} and masks {int(t.counts.mask)} of them.
          </p>
        </div>
      </div>
    </Bench>
  );
}
