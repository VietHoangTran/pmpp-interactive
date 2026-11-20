import { Scene, type Beat } from '../../components/Scene';
import { BOUND_Q8Q8, FP16_MAX, INT16_MAX, MAX_Q8_PRODUCT, narrowScene, sci, show, type Inner } from '../../sim/qdot';

// "Where the sum stops fitting": the worst-case Q8_0 × Q8_1 block added one product at a time
// into four accumulators, then a 4,096-wide row summed across its 128 blocks in half and in
// float. Every number is from src/sim/qdot.ts (innerSum, accumulate), FP16 rounding from
// fpsum.ts and minifloat.ts. Five beats (content/x2/narration/narrow-accumulator-N.txt).
const S = narrowScene();
const COLS: { id: Inner; name: string; limit: number; limitName: string }[] = [
  { id: 'int16', name: 'int16', limit: INT16_MAX, limitName: '32,767' },
  { id: 'fp16', name: 'FP16', limit: FP16_MAX, limitName: '65,504' },
  { id: 'int32', name: 'int32', limit: 2 ** 31 - 1, limitName: '2,147,483,647' },
  { id: 'fp32', name: 'FP32', limit: 3.4e38, limitName: 'about 3.4e38' },
];
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const W = 720;
const H = 340;
const TOP = 66;
const BOT = 280;
const YMAX = 560000; // a little above 516,128
const y = (v: number) => BOT - (Math.min(Math.abs(v), YMAX) / YMAX) * (BOT - TOP);

function Columns({ k, focus, head }: { k: number; focus?: Inner[]; head: string }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Animation of four accumulators adding the same products" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <text x={20} y={22} fontSize="13" fontWeight="600" data-head="">{head}</text>
      <text x={20} y={40} fontSize="11.5" style={{ fill: 'var(--ink-2)' }} data-k={k}>
        products added: {k} of 32, each 127 × 127 = {MAX_Q8_PRODUCT.toLocaleString('en-US')}; exact sum {(k * MAX_Q8_PRODUCT).toLocaleString('en-US')}
      </text>
      <line x1={70} x2={700} y1={y(BOUND_Q8Q8)} y2={y(BOUND_Q8Q8)} stroke="var(--ink-3)" strokeDasharray="4 3" />
      <text x={20} y={y(BOUND_Q8Q8) - 2} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>bound</text>
      <text x={20} y={y(BOUND_Q8Q8) + 10} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>516,128</text>
      {COLS.map((col, i) => {
        const tr = S.traces[col.id];
        const v = k === 0 ? 0 : tr.values[k - 1];
        const exact = k * MAX_Q8_PRODUCT;
        const wrapped = tr.first.wrap !== undefined && k >= tr.first.wrap;
        const inf = tr.first.inf !== undefined && k >= tr.first.inf;
        const dim = focus && !focus.includes(col.id) ? 0.35 : 1;
        const cx = 90 + i * 160;
        const bad = wrapped || inf;
        const status = k === 0 ? '' : inf ? 'infinity' : wrapped ? 'wrapped' : v === exact ? 'exact' : 'rounded';
        return (
          <g key={col.id} opacity={dim} data-col={col.id} data-value={String(v)} data-status={status}>
            <text x={cx + 40} y={BOT + 20} fontSize="12.5" fontWeight="600" textAnchor="middle">{col.name}</text>
            <rect x={cx} y={TOP} width={80} height={BOT - TOP} fill="var(--cell)" stroke="var(--rule)" />
            {col.limit < YMAX ? (
              <g>
                <line x1={cx - 6} x2={cx + 86} y1={y(col.limit)} y2={y(col.limit)} stroke="var(--fail)" strokeWidth="1.5" />
                <text x={cx + 84} y={y(col.limit) - 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--fail)' }}>max {col.limitName}</text>
              </g>
            ) : (
              <text x={cx + 40} y={TOP - 5} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>max {col.limitName}</text>
            )}
            {inf ? (
              <rect x={cx + 8} y={TOP} width={64} height={BOT - TOP} fill="var(--fail-wash)" stroke="var(--fail)" />
            ) : (
              <rect x={cx + 8} y={y(v)} width={64} height={BOT - y(v)}
                fill={bad ? 'var(--fail-wash)' : 'var(--reg-wash)'} stroke={bad ? 'var(--fail)' : 'var(--reg)'} />
            )}
            <text x={cx + 40} y={BOT + 38} fontSize="12" textAnchor="middle" style={{ fill: bad ? 'var(--fail)' : 'var(--ink)' }}>
              {k === 0 ? '0' : show(v)}
            </text>
            <text x={cx + 40} y={BOT + 54} fontSize="10.5" textAnchor="middle" style={{ fill: bad ? 'var(--fail)' : status === 'exact' ? 'var(--ok)' : 'var(--ink-2)' }}>
              {status}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Beat 5: the absolute error of the outer sum after each of 128 blocks, half against float (log scale). */
function Drift({ p }: { p: number }) {
  const n = Math.max(1, Math.round(ramp(p, 0.05, 0.75) * 128));
  const gx = 70;
  const gw = 600;
  const gy = 60;
  const gh = 200;
  const LO = -9;
  const HI = -1;
  const yl = (e: number) => gy + gh - ((Math.log10(Math.max(e, 10 ** LO)) - LO) / (HI - LO)) * gh;
  const path = (arr: number[]) => arr.slice(0, n).map((e, i) => `${i ? 'L' : 'M'}${gx + (i / 127) * gw},${yl(e)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Error of a half and a float accumulator across a row" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <text x={20} y={22} fontSize="13" fontWeight="600" data-head="">A 4,096-wide row: 128 blocks, int32 inside each, the outer sum in half or float</text>
      <text x={20} y={42} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>absolute error of the running sum after each block (log scale), realistic data</text>
      <rect x={gx} y={gy} width={gw} height={gh} fill="var(--cell)" stroke="var(--rule)" />
      {[-9, -7, -5, -3, -1].map((e) => (
        <g key={e}>
          <line x1={gx} x2={gx + gw} y1={yl(10 ** e)} y2={yl(10 ** e)} stroke="var(--rule-soft)" />
          <text x={gx - 6} y={yl(10 ** e) + 3} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>1e{e}</text>
        </g>
      ))}
      <path d={path(S.outerHalf)} fill="none" stroke="var(--fail)" strokeWidth="2" />
      <path d={path(S.outerFloat)} fill="none" stroke="var(--reg)" strokeWidth="2" />
      <text x={gx + gw + 4} y={yl(S.outerHalf[n - 1]) + 4} fontSize="10.5" style={{ fill: 'var(--fail)' }}>half</text>
      <text x={gx + gw + 4} y={yl(S.outerFloat[n - 1]) + 4} fontSize="10.5" style={{ fill: 'var(--reg)' }}>float</text>
      <text x={gx} y={gy + gh + 18} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>block 1</text>
      <text x={gx + gw} y={gy + gh + 18} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>block 128</text>
      <text x={20} y={306} fontSize="12" data-drift="">
        after {n} blocks: half off by {sci(S.outerHalf[n - 1])}, float off by {sci(S.outerFloat[n - 1])}
      </text>
      <text x={20} y={328} fontSize="13" fontWeight="600" opacity={ramp(p, 0.8, 0.9)}>
        Integer inside a block, float across blocks, never half for a sum.
      </text>
    </svg>
  );
}

const beats: Beat[] = [
  { clip: 'narrow-accumulator-1', label: 'The worst case', draw: (p) => <Columns k={Math.round(ramp(p, 0.4, 0.9) * 2)} head="The worst-case Q8_0 × Q8_1 block, one product at a time" /> },
  { clip: 'narrow-accumulator-2', label: 'int16 wraps', draw: (p) => <Columns k={2 + Math.round(ramp(p, 0.2, 0.6) * 2)} focus={['int16']} head="The 16-bit integer passes 32,767 on the third product" /> },
  { clip: 'narrow-accumulator-3', label: 'FP16 overflows', draw: (p) => <Columns k={4 + Math.round(ramp(p, 0.3, 0.7) * 2)} focus={['fp16']} head="FP16 rounds from the first product and overflows on the fifth" /> },
  { clip: 'narrow-accumulator-4', label: 'Wide enough', draw: (p) => <Columns k={6 + Math.round(ramp(p, 0.05, 0.7) * 26)} focus={['int32', 'fp32']} head="int32 and float carry the block to 516,128, exactly" /> },
  { clip: 'narrow-accumulator-5', label: 'Across the row', draw: (p) => <Drift p={p} /> },
];

export function NarrowAccumulatorScene() {
  return <Scene chapter="x2" title="Where the sum stops fitting" beats={beats} />;
}
