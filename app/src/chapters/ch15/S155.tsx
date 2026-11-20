import { useState, type MouseEvent, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BM,
  BN,
  DEFAULT_STORE,
  PATTERNS,
  PATTERN_ORDER,
  coalesced,
  coarseThread,
  instructionsPerThread,
  ownerOf,
  remapGeometry,
  remapThread,
  rowStrips,
  storeColumns,
  storeInstruction,
  warpTileTotal,
  type Layout,
  type StorePattern,
  type Width,
} from '../../sim/gemmremap';
import { RemapScene } from './RemapScene';

const thousands = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => `${Math.round(v * 1000) / 10} %`;

// ---------------------------------------------------------------- the store visualiser

const N_CHOICES: { N: number; label: string }[] = [
  { N: 1024, label: '1024 (a multiple of 8)' },
  { N: 1020, label: '1020 (a multiple of 4, not of 8)' },
  { N: 1001, label: '1001 (odd)' },
];
const SEGMENTS: { bytes: number; label: string }[] = [
  { bytes: 32, label: '32-byte sectors (NVIDIA)' },
  { bytes: 64, label: '64-byte lines (AMD)' },
  { bytes: 128, label: '128-byte lines (AMD)' },
];
const UNIT: Record<number, string> = { 32: 'sectors', 64: 'lines', 128: 'lines' };
const SHORT: Record<StorePattern, string> = {
  scalar8x8: '8 × 8, scalar (§15.3)',
  'float4-8x8': '8 × 8, float4 (Exercise 2)',
  remap4x4: 'four 4 × 4, float4 (§15.5)',
};

type Verdict = 'yes' | 'no';
/** Value cells of the thread table may wrap on a phone (the table style keeps cells on one line). */
const WRAP = { whiteSpace: 'normal' as const };

function StoreVisualiser() {
  const [pattern, setPattern] = useState<StorePattern>('scalar8x8');
  const [W, setW] = useState<Width>(32);
  const [N, setN] = useState(1024);
  const [seg, setSeg] = useState(32);
  const [warp, setWarp] = useState(0);
  const [row, setRow] = useState(0);
  const [col, setCol] = useState(0);
  const [said, setSaid] = useState<Partial<Record<StorePattern, Verdict>>>({});

  const cols = storeColumns(pattern, N);
  const c = cols.includes(col) ? col : col - (col % 4);
  const warps = 256 / W;
  const w = Math.min(warp, warps - 1);
  const res = storeInstruction({ pattern, W, N, warp: w, row, col: c, segmentBytes: seg });
  const strips = rowStrips(res, seg);
  const total = warpTileTotal(pattern, W, N, w, seg);
  const verdict = said[pattern];
  const truth: Verdict = coalesced(pattern) ? 'yes' : 'no';

  // Drawing: one strip per matrix row, labelled above it, at one scale for every pattern
  // so the patterns can be compared; the picture is never drawn larger than life.
  const span = Math.max(...strips.map((s) => (s.seg1 - s.seg0) * seg));
  const pxB = Math.min(1.25, 420 / Math.max(span, 1));
  const X0 = 6;
  const pitch = 40;
  const VW = Math.max(320, Math.ceil(X0 * 2 + span * pxB));
  const H = 8 + strips.length * pitch;
  return (
    <Bench
      title="One warp, one store instruction"
      task="For each pattern, decide whether the store is coalesced before the counts appear. Then change the width, N and the segment size."
      foot={
        <>
          Each strip is the stretch of C one matrix row contributes, from the first {seg}-byte segment the instruction touches
          to the last. Dark: bytes a lane writes. Pale: bytes the memory system moves but nobody asked for. C starts on a
          256-byte boundary; the block is block (0, 0).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Store pattern">
          {PATTERN_ORDER.map((p) => (
            <button key={p} aria-pressed={p === pattern} onClick={() => setPattern(p)} title={`${PATTERNS[p].label} (${PATTERNS[p].book})`}>
              {SHORT[p]}
            </button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          Warp width
          <select value={W} onChange={(e) => setW(Number(e.target.value) as Width)}>
            <option value={32}>32 lanes</option>
            <option value={64}>64 lanes (AMD CDNA)</option>
          </select>
        </label>
        <label className="field">
          N, the width of C
          <select value={N} onChange={(e) => setN(Number(e.target.value))}>
            {N_CHOICES.map((o) => <option key={o.N} value={o.N}>{o.label}</option>)}
          </select>
        </label>
        <label className="field">
          Segment
          <select value={seg} onChange={(e) => setSeg(Number(e.target.value))}>
            {SEGMENTS.map((o) => <option key={o.bytes} value={o.bytes}>{o.label}</option>)}
          </select>
        </label>
        <label className="field">
          Warp
          <select value={w} onChange={(e) => setWarp(Number(e.target.value))}>
            {Array.from({ length: warps }, (_, k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <label className="field">
          Instruction: tile row
          <select value={row} onChange={(e) => setRow(Number(e.target.value))}>
            {Array.from({ length: 8 }, (_, k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <label className="field">
          tile columns
          <select value={c} onChange={(e) => setCol(Number(e.target.value))}>
            {cols.map((k) => <option key={k} value={k}>{res.vector ? `${k}–${k + 3}` : k}</option>)}
          </select>
        </label>
      </div>

      {res.fallback ? (
        <p className="narrate" style={{ marginBottom: '0.8rem' }}>
          N = {N} is not a multiple of 4, so most rows of C do not start on a 16-byte boundary and a float4 store would be
          misaligned. The kernel takes its scalar path: one float per lane, {instructionsPerThread(pattern, N)} store
          instructions per thread.
        </p>
      ) : null}

      <svg viewBox={`0 0 ${VW} ${H}`} role="img" aria-label="Bytes of C written by one store instruction"
        data-transactions={res.transactions} data-efficiency={res.efficiency} data-rows={strips.length} data-vector={res.vector ? 'yes' : 'no'}
        style={{ width: '100%', maxWidth: VW * 1.15, display: 'block', marginBottom: '0.6rem' }}>
        {strips.map((s, k) => {
          const y = 4 + k * pitch;
          return (
            <g key={s.row} data-strip={s.row}>
              <text x={X0} y={y + 10} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
                row {s.row} <tspan style={{ fill: 'var(--ink-3)' }}>from byte {thousands(s.seg0 * seg)}</tspan>
              </text>
              {s.used.map((u, j) => (
                <rect key={j} x={X0 + j * seg * pxB} y={y + 15} width={seg * pxB - 1} height={18}
                  fill="var(--global-wash)" stroke="var(--global)" strokeWidth="0.7" />
              ))}
              {s.pieces.map((pc) => {
                const x = X0 + (pc.lo - s.seg0 * seg) * pxB;
                const wd = (pc.hi - pc.lo) * pxB;
                return (
                  <g key={pc.lane}>
                    <rect x={x} y={y + 17} width={Math.max(1, wd - 0.6)} height={14} fill="var(--global)">
                      <title>lane {pc.lane}: bytes {thousands(pc.lo)} to {thousands(pc.hi - 1)}</title>
                    </rect>
                    {wd >= 15 ? (
                      <text x={x + wd / 2} y={y + 27.5} fontSize="8" textAnchor="middle" style={{ fill: 'var(--page)' }}>{pc.lane}</text>
                    ) : null}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>

      {verdict == null ? (
        <div className="row" data-classify={pattern}>
          <span>Is this store coalesced, with 32 lanes, N = 1024 and 32-byte sectors?</span>
          <button onClick={() => setSaid({ ...said, [pattern]: 'yes' })}>Coalesced</button>
          <button onClick={() => setSaid({ ...said, [pattern]: 'no' })}>Not coalesced</button>
        </div>
      ) : (
        <div data-counts={pattern}>
          <p style={{ margin: '0 0 0.5rem' }}>
            {verdict === truth ? <span className="chip ok">Right.</span> : <span className="chip fail">Not quite.</span>}{' '}
            In the default setting this pattern is {truth === 'yes' ? 'coalesced: every byte moved is a byte asked for' : 'not coalesced: most of each segment is moved for nothing'}.
          </p>
          <p className="ui" style={{ margin: 0 }} data-readout="">
            This instruction: <b>{res.transactions} {UNIT[seg]}</b> touched, {thousands(res.bytesNeeded)} of{' '}
            {thousands(res.bytesFetched)} bytes useful: <b>{pct(res.efficiency)}</b>. {res.runs} separate{' '}
            {res.runs === 1 ? 'run' : 'runs'} of contiguous bytes, on {res.rows.length} matrix rows.
          </p>
          <p className="ui muted" style={{ margin: '0.3rem 0 0' }} data-total="">
            The warp's whole tile: {total.instructions} store instructions per thread, {thousands(total.transactions)}{' '}
            {UNIT[seg]} in all; its {thousands(total.bytes)} bytes fit in {thousands(total.minimum)}.
          </p>
        </div>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- the lane picker

function LanePicker() {
  const [W, setW] = useState<Width>(32);
  const [warp, setWarp] = useState(6);
  const [lane, setLane] = useState(5);
  const [layout, setLayout] = useState<Layout>('remap');
  const [hover, setHover] = useState<{ row: number; col: number } | null>(null);

  const g = remapGeometry(W);
  const wp = Math.min(warp, g.warps - 1);
  const ln = Math.min(lane, W - 1);
  const t = wp * W + ln;
  const rt = remapThread(t, W);
  const ct = coarseThread(t);
  const S = 4; // pixels per element
  const O = 24;
  const X = (c: number) => O + c * S;
  const Y = (r: number) => O + r * S;
  const at = (e: MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const scale = (O * 2 + BN * S) / box.width;
    const col = Math.floor(((e.clientX - box.left) * scale - O) / S);
    const row = Math.floor(((e.clientY - box.top) * scale - O) / S);
    return row >= 0 && row < BM && col >= 0 && col < BN ? { row, col } : null;
  };
  const owner = hover ? ownerOf(layout, W, hover.row, hover.col) : null;
  const laneRowMates = Array.from({ length: g.LC }, (_, k) => Math.floor(ln / g.LC) * g.LC + k);

  let shapes: ReactNode;
  if (layout === 'remap') {
    shapes = (
      <>
        {Array.from({ length: g.warps }, (_, k) => {
          const r = remapThread(k * W, W);
          return (
            <rect key={'w' + k} x={X(r.wCol)} y={Y(r.wRow)} width={g.warpCols * S} height={g.warpRows * S}
              fill={k === wp ? 'var(--bench-grid)' : 'none'} stroke="var(--ink-2)" strokeWidth={k === wp ? 2 : 1} />
          );
        })}
        {[0, 1].flatMap((qr) => [0, 1].map((qc) => (
          <rect key={`q${qr}${qc}`} x={X(rt.wCol + qc * g.QC)} y={Y(rt.wRow + qr * g.QR)} width={g.QC * S} height={g.QR * S}
            fill="none" stroke="var(--ink-3)" strokeDasharray="4 3" />
        )))}
        {laneRowMates.filter((m) => m !== ln).flatMap((m) =>
          remapThread(wp * W + m, W).subTiles.map((s) => (
            <rect key={`m${m}-${s.qr}${s.qc}`} x={X(s.col)} y={Y(s.row)} width={4 * S} height={4 * S} fill="var(--reg-wash)" stroke="var(--reg)" strokeWidth="0.6" />
          )),
        )}
        {rt.subTiles.map((s) => (
          <rect key={`s${s.qr}${s.qc}`} data-subtile={`${s.row},${s.col}`} x={X(s.col)} y={Y(s.row)} width={4 * S} height={4 * S} fill="var(--reg)" />
        ))}
      </>
    );
  } else {
    shapes = (
      <>
        {Array.from({ length: 15 }, (_, k) => (k + 1) * 8).map((v) => (
          <g key={v}>
            <line x1={X(0)} x2={X(BN)} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
            <line y1={Y(0)} y2={Y(BM)} x1={X(v)} x2={X(v)} stroke="var(--rule-soft)" />
          </g>
        ))}
        {Array.from({ length: W }, (_, l) => {
          const c = coarseThread(wp * W + l);
          return <rect key={l} x={X(c.tCol)} y={Y(c.tRow)} width={8 * S} height={8 * S} fill="var(--bench-grid)" stroke="var(--ink-3)" strokeWidth="0.5" />;
        })}
        <rect data-subtile={`${ct.tRow},${ct.tCol}`} x={X(ct.tCol)} y={Y(ct.tRow)} width={8 * S} height={8 * S} fill="var(--reg)" />
      </>
    );
  }

  return (
    <Bench
      title="Lane picker"
      task="Pick a warp and a lane and find its four squares. Hover over any element to see who owns it; click it to pick that lane."
      foot={
        <>
          The 128 × 128 block tile. Dark: the lane's tile. Pale: the other lanes of its lane row, whose stores sit beside its
          own. Dashed: the quadrants of its warp's tile. In the §15.3 view the grey patches are the rest of the warp.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span className="seg" role="group" aria-label="Mapping">
          <button aria-pressed={layout === 'remap'} onClick={() => setLayout('remap')}>§15.5: four 4 × 4</button>
          <button aria-pressed={layout === 'coarse'} onClick={() => setLayout('coarse')}>§15.3: one 8 × 8</button>
        </span>
        <label className="field">
          Warp width
          <select value={W} onChange={(e) => setW(Number(e.target.value) as Width)}>
            <option value={32}>32 lanes</option>
            <option value={64}>64 lanes (AMD CDNA)</option>
          </select>
        </label>
        <label className="field">
          Warp
          <select value={wp} onChange={(e) => setWarp(Number(e.target.value))}>
            {Array.from({ length: g.warps }, (_, k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <label className="field">
          Lane
          <select value={ln} onChange={(e) => setLane(Number(e.target.value))}>
            {Array.from({ length: W }, (_, k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 5fr)', alignItems: 'start' }}>
        <svg viewBox={`0 0 ${O * 2 + BN * S} ${O * 2 + BM * S}`} role="img" aria-label="The block tile with the chosen lane's elements"
          data-layout={layout} data-thread={t}
          style={{ width: '100%', maxWidth: 540, display: 'block', cursor: 'crosshair' }}
          onMouseMove={(e) => setHover(at(e))}
          onMouseLeave={() => setHover(null)}
          onClick={(e) => {
            const h = at(e);
            if (!h) return;
            const o = ownerOf(layout, W, h.row, h.col);
            setWarp(o.warp);
            setLane(o.lane);
          }}>
          <rect x={X(0)} y={Y(0)} width={BN * S} height={BM * S} fill="var(--cell)" stroke="var(--rule)" />
          {shapes}
          {hover ? <rect x={X(hover.col) - 1} y={Y(hover.row) - 1} width={S + 2} height={S + 2} fill="none" stroke="var(--ink)" strokeWidth="1.5" /> : null}
          {[0, 32, 64, 96, 127].map((v) => (
            <g key={v}>
              <text x={X(v) + (v === 127 ? S : 0)} y={O - 4} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{v === 127 ? 128 : v}</text>
              <text x={O - 3} y={Y(v) + (v === 127 ? S : 0) + 3} fontSize="9" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{v === 127 ? 128 : v}</text>
            </g>
          ))}
        </svg>
        <div>
          <p className="narrate" role="status" data-hover={owner ? `${owner.t}` : ''} style={{ marginTop: 0 }}>
            {owner && hover ? (
              <>
                Element (row {hover.row}, column {hover.col}) belongs to thread {owner.t}: warp {owner.warp}, lane {owner.lane}.
                It is element ({owner.r}, {owner.c}) of that thread's logical 8 × 8 tile
                {layout === 'remap' ? <>, in its sub-tile of quadrant ({owner.qr}, {owner.qc})</> : null}.
              </>
            ) : (
              <>Hover over the tile to see who owns an element.</>
            )}
          </p>
          <table className="threads" aria-label="The chosen thread" data-table={layout}>
            <tbody>
              <tr><td>thread</td><td className="mono" style={WRAP}>t = {wp}·{W} + {ln} = {t}</td></tr>
              {layout === 'remap' ? (
                <>
                  <tr><td>warp tile</td><td className="mono" style={WRAP}>rows {rt.wRow}–{rt.wRow + g.warpRows - 1}, columns {rt.wCol}–{rt.wCol + g.warpCols - 1}</td></tr>
                  <tr><td>lane row, column</td><td className="mono" style={WRAP}>{rt.lRow / 4}, {rt.lCol / 4} (of 8 × {g.LC})</td></tr>
                  {rt.subTiles.map((s) => (
                    <tr key={`${s.qr}${s.qc}`} data-row-subtile={`${s.row},${s.col}`}>
                      <td>quadrant ({s.qr}, {s.qc})</td>
                      <td className="mono t-reg" style={WRAP}>rows {s.row}–{s.row + 3}, columns {s.col}–{s.col + 3}</td>
                    </tr>
                  ))}
                  <tr><td>its lane row stores</td><td className="mono" style={WRAP}>{g.LC} lanes × 16 B = {g.laneRowBytes} contiguous bytes</td></tr>
                </>
              ) : (
                <>
                  <tr><td>ty, tx</td><td className="mono" style={WRAP}>{ct.ty}, {ct.tx}</td></tr>
                  <tr data-row-subtile={`${ct.tRow},${ct.tCol}`}><td>its 8 × 8 patch</td><td className="mono t-reg" style={WRAP}>rows {ct.tRow}–{ct.tRow + 7}, columns {ct.tCol}–{ct.tCol + 7}</td></tr>
                  <tr><td>its neighbour in the warp</td><td className="mono" style={WRAP}>8 floats (32 bytes) to the right</td></tr>
                </>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS155: Record<string, Question> = {
  'ch15/15.5/q-float4-alone': {
    kind: 'choice',
    id: 'ch15/15.5/q-float4-alone',
    prompt: (
      <>
        You do Exercise 2: <code>writeTile</code> stores each row of the 8 × 8 tile as two float4s instead of eight floats,
        and nothing else changes. What does one store instruction of a 32-lane warp cost now, with N a multiple of 8 and
        32-byte sectors?
      </>
    ),
    options: [
      { label: '8 sectors, all bytes used.', why: 'That would need the 32 lanes’ 16-byte pieces to sit end to end. They are 32 bytes apart.' },
      { label: '16 sectors.', why: 'That is the count after the remap of §15.5, where the pieces of 4 lanes form 64-byte runs.' },
      { label: '32 sectors, each half written: a quarter of the instructions, but still not coalesced.', correct: true },
    ],
    explain: (
      <p>
        A float4 covers columns 0–3 of a lane's patch; the neighbouring lane's patch starts 8 floats further on. So each
        lane's 16 bytes sit alone in a 32-byte sector, and the instruction touches 32 sectors at 50 %. The book says it
        plainly: a vector store is half the width of an 8 × 8 tile, so neighbours' stores are still four elements apart.
        Fewer instructions, same scattering.
      </p>
    ),
    source: 'ch. 15 §15.5, p. 360; sector rule from notes/today/ch06-facts.md A2',
  },
  'ch15/15.5/q-remap-sectors': {
    kind: 'numeric',
    id: 'ch15/15.5/q-remap-sectors',
    prompt: (
      <>
        After the remap, how many 32-byte sectors does one float4 store instruction of a 32-lane warp touch, with N a
        multiple of 8?
      </>
    ),
    answer: 16,
    unit: 'sectors',
    explain: (
      <p>
        32 lanes × 16 bytes = 512 bytes. The 4 lanes of each lane row write 64 contiguous bytes, which start on a sector
        boundary when N is a multiple of 8: 2 sectors per lane row, 8 lane rows, 16 sectors, and every byte of them is
        written. That is also the least any 512 bytes can occupy.
      </p>
    ),
    source: 'ch. 15 §15.5, pp. 360–361; our count with NVIDIA’s 32-byte sectors (ch06-facts A2)',
  },
  'ch15/15.5/q-registers': {
    kind: 'choice',
    id: 'ch15/15.5/q-registers',
    prompt: <>Does the remap change how many registers a thread needs for its tile and strips?</>,
    options: [
      { label: 'Yes: four sub-tiles need four sets of strips, so 4 × (4 + 4) = 32 strip values per step.', why: 'Each A piece serves two sub-tiles (the two in the same quadrant row), and each B piece two as well.' },
      { label: 'No: 64 accumulators, and per step two 4-value A pieces and two 4-value B pieces, 16 values, as before.', correct: true },
      { label: 'Fewer: a 4 × 4 sub-tile needs only 4 + 4 strip values.', why: 'Per sub-tile, yes, but the thread updates all four sub-tiles in each step.' },
    ],
    explain: (
      <p>
        The thread still owns 64 outputs, now as four 4 × 4 squares. In step i it needs column i of A for its 8 rows (two
        pieces of 4, one per quadrant row) and row i of B for its 8 columns (two pieces of 4): the same 8 + 8 values as the
        8 × 8 tile of §15.4. The register budget of §15.7 does not change.
      </p>
    ),
    source: 'ch. 15 §15.5, pp. 360–361, Fig. 15.10; the count is ours (notes [inferred])',
  },
  'ch15/15.5/q-wave64': {
    kind: 'choice',
    id: 'ch15/15.5/q-wave64',
    prompt: (
      <>
        A block of 256 threads on a CDNA GPU is 4 wavefronts of 64. Which layout keeps four 4 × 4 sub-tiles per thread and
        contiguous float4 stores?
      </>
    ),
    options: [
      { label: 'The book’s layout unchanged: 2 × 4 warp tiles of 64 × 32.', why: 'There are only 4 wavefronts, not 8, and a wavefront has 64 lanes to place, not 32.' },
      { label: '2 × 2 wavefront tiles of 64 × 64; lanes 8 × 8; quadrants 32 × 32; a lane row of 8 lanes stores 128 contiguous bytes.', correct: true },
      { label: '1 × 4 wavefront tiles of 128 × 32; lanes 16 × 4; quadrants 64 × 16.', why: 'It covers the tile, but each lane row still writes only 64 bytes, and the lanes are 16 rows tall: the extra lanes go into rows, not into longer runs.' },
    ],
    explain: (
      <p>
        Keep 8 lane rows and four 4 × 4 squares per lane; the extra 32 lanes go into the lane rows, which become 8 wide.
        A quadrant is then 32 rows by 8 × 4 = 32 columns, a wavefront tile 64 × 64, and four of them fill the block tile as
        2 × 2. Each lane row writes 8 × 16 = 128 contiguous bytes per instruction. This is our derivation, not AMD's.
      </p>
    ),
    source: 'our derivation (notes/today/ch15-facts.md B1); wavefront widths from ch05-facts B5',
  },
  'ch15/15.5/q-width-wrong': {
    kind: 'choice',
    id: 'ch15/15.5/q-width-wrong',
    prompt: <>The remapped kernel is compiled for a width of 32 and runs on a GPU with 64-lane wavefronts. What happens?</>,
    options: [
      { label: 'Wrong results: two threads write the same elements.', why: 'Every element still has exactly one owner: the mapping is a bijection over threadIdx.x whatever constant it was compiled with.' },
      { label: 'The results are still correct; only the store pattern is no longer the one the layout was designed for.', correct: true },
      { label: 'It does not compile.', why: 'Nothing in the kernel asks the hardware for its width; it uses the constant it was given.' },
    ],
    explain: (
      <>
        <p>
          The mapping is pure index arithmetic: warp and lane are derived from <code>threadIdx.x</code> with the compiled
          constant, and every element of the tile still gets one owner. Nothing in it depends on which threads really
          execute together. What does depend on that is the coalescing: a 64-lane wavefront now holds two of the 32-lane
          "warps" the layout assumed, so one store instruction writes two separate groups of runs.
        </p>
        <p>
          A width mismatch here costs speed, not correctness. That is no licence to write a literal 32: choose the constant
          from the device, as below, so the layout matches the hardware it runs on.
        </p>
      </>
    ),
    source: 'our analysis; widths from notes/today/ch15-facts.md B1 and C1',
  },
  'ch15/15.5/q-align': {
    kind: 'choice',
    id: 'ch15/15.5/q-align',
    prompt: <>C has N = 1001 columns. May the kernel store each sub-tile row with one float4?</>,
    options: [
      { label: 'Yes: float4 only needs four consecutive floats.', why: 'It needs them at an address that is a multiple of 16 bytes.' },
      { label: 'Not in general: row r starts at byte 4 × 1001 × r, 16-byte aligned only when r is a multiple of 4. The kernel stores scalars, with bounds checks, unless N % 4 == 0.', correct: true },
      { label: 'Only for the last block column.', why: 'Alignment depends on the row, not on which block column.' },
    ],
    explain: (
      <p>
        4 × 1001 = 4004 bytes per row, and 4004 mod 16 = 4, so row r starts 4r bytes past a 16-byte boundary: aligned only
        for r = 0, 4, 8, and so on. A float4 store to a misaligned address is not allowed. The simple rule the labs use:
        take the vector path when N is a multiple of 4, and bounds-checked scalar stores otherwise. A full kernel would
        also check the right edge of C, where fewer than 4 columns remain.
      </p>
    ),
    source: 'ours, from notes/today/ch06-facts.md A11 and ch15-facts.md E1 (alignment note)',
  },
  'ch15/15.5/q-epilogue': {
    kind: 'choice',
    id: 'ch15/15.5/q-epilogue',
    prompt: (
      <>
        CUTLASS keeps the thread mapping that suits the arithmetic and makes its stores coalesced in a separate epilogue,
        through shared memory. What does that cost that the book's remap does not?
      </>
    ),
    options: [
      { label: 'Nothing: it is the same work, done later.', why: 'The results take an extra trip and the threads must wait for each other.' },
      { label: 'A round trip through shared memory, a barrier, and shared-memory space; in return the mapping can be chosen for the arithmetic alone.', correct: true },
      { label: 'More registers per thread.', why: 'The accumulators are the same; the epilogue moves them through shared memory, not through more registers.' },
    ],
    explain: (
      <p>
        In the epilogue each thread writes its outputs to shared memory, the block synchronises, and the threads read them
        back in a pattern that suits global memory. That costs shared-memory traffic, a barrier and space, but it decouples
        two choices: the mapping for the multiply and the pattern for the store. The remap couples them. It gets away with
        that because four 4 × 4 squares need the same registers and the same rank-1 updates as one 8 × 8; its price shows
        up in the shared-memory reads instead, which §15.6 deals with.
      </p>
    ),
    source: 'notes/today/ch15-facts.md A12 (CUTLASS “Efficient GEMM in CUDA”, section “Epilogue”)',
  },
};

// ---------------------------------------------------------------- the page

export function S155() {
  const scalar = storeInstruction({ ...DEFAULT_STORE, pattern: 'scalar8x8' });
  const g32 = remapGeometry(32);
  const g64 = remapGeometry(64);
  const w6 = remapThread(6 * 32, 32);
  return (
    <>
      <p>
        Sections 15.3 and 15.4 gave every thread an 8 × 8 tile of C in registers. That is good for reuse: each value a
        thread loads feeds eight products. The cost appears once, at the very end, when the tile goes back to global
        memory. One store instruction writes the same element of every lane's tile, and neighbouring lanes own
        neighbouring tiles, so their addresses are 8 floats apart.
      </p>
      <Source section="15.5" pages="360" />
      <Predict
        id="ch15/15.5/p-scalar-sectors"
        prompt={
          <>
            One scalar store instruction of one warp in the §15.3 layout, N a multiple of 8. How many 32-byte sectors
            does it touch?
          </>
        }
        unit="sectors"
      >
        {(v) => (
          <p>
            It touches <b>{scalar.transactions}</b>
            {v === scalar.transactions ? ', as you said' : ''}. A warp is 32 consecutive threads: two rows of 16 tiles in the
            block tile. Each lane writes 4 bytes, and its neighbour writes 32 bytes further on, so every lane lands in a
            sector of its own: {scalar.transactions} sectors to deliver {scalar.bytesNeeded} bytes, {pct(scalar.efficiency)}{' '}
            of what moves. The visualiser below draws it.
          </p>
        )}
      </Predict>
      <p>
        The obvious repair is the book's Exercise 2: store four floats at once with a float4. It cuts the number of store
        instructions by four. But a float4 covers half a row of an 8 × 8 tile, so the neighbouring lane's store still starts
        four floats after this one ends.
      </p>
      <Source section="15.5" pages="360" />

      <h2>One warp, one store instruction</h2>
      <p>
        The visualiser shows, for one store instruction of one warp, which bytes of C each lane writes: one strip per
        matrix row the warp touches, cut into the memory system's segments. Say whether each pattern is coalesced before
        its counts appear. Then try the third pattern, which the rest of this page explains, and change the width, N and
        the segment size.
      </p>
      <Simplification>
        <p>
          One instruction at a time, counted in NVIDIA's documented 32-byte sectors (ch06-facts A2). AMD documents 64- or
          128-byte cache lines (ch06-facts B1), offered as options; Apple documents no rule (ch06-facts C1). A cache that merges partly written
          sectors from different instructions is not modelled, and neither is time.
        </p>
      </Simplification>
      <StoreVisualiser />
      <Ask q={questionsS155['ch15/15.5/q-float4-alone']} />

      <h2>Four 4 × 4 sub-tiles per thread</h2>
      <p>
        The book's fix keeps each thread's 64 outputs but changes which 64. It adds a level to the tile hierarchy: the warp.
        The 128 × 128 block tile is divided among the block's 8 warps as 2 rows of 4 <b>warp-level tiles</b>, each 64 rows by
        32 columns. Each warp tile is cut into four quadrants of 32 × 16. The warp's 32 lanes are arranged 8 × 4, and each
        lane takes the 4 × 4 square at its position in <i>every</i> quadrant. Those four squares are the thread's logical
        8 × 8 tile; the squares are its physical sub-tiles.
      </p>
      <Source section="15.5" pages="360–361" figs="Fig. 15.10" />
      <p>
        Now a store instruction writes one row of one square per lane: 4 floats, 16 bytes, one float4. The 4 lanes of a
        lane row own neighbouring squares, so their stores sit end to end as 64 contiguous bytes. The 8 lane rows are 4
        matrix rows apart. The book calls the result fully coalesced. It does not mention that one instruction still covers
        8 separate rows; in sectors that costs nothing, as long as each 64-byte run starts on a sector boundary.
      </p>
      <Beyond title="Our reading">
        <p>The per-row split is ours (notes, §15.5 [inferred]); the book stops at "fully coalesced".</p>
      </Beyond>
      <p>
        In code, every coordinate comes from the thread index by division and remainder. With W the warp width, LC = W / 8
        lanes per lane row, and t the thread index:
      </p>
      <ul>
        <li><code>warp = t / W</code>, <code>lane = t % W</code></li>
        <li><code>lRow = 4*(lane / LC)</code>, <code>lCol = 4*(lane % LC)</code>: the lane's square inside a quadrant</li>
        <li><code>QR = 32</code>, <code>QC = 4*LC</code>: the quadrant; <code>WC = 128 / (2*QC)</code> warp tiles per row</li>
        <li><code>wRow = 64*(warp / WC)</code>, <code>wCol = 2*QC*(warp % WC)</code>: the warp tile</li>
        <li>sub-tile (qr, qc) starts at <code>(wRow + qr*QR + lRow, wCol + qc*QC + lCol)</code></li>
      </ul>
      <p>
        With W = 32 these give the book's numbers: LC = {g32.LC}, QC = {g32.QC}, WC = {g32.WC}, and warp 6, the warp
        Fig. 15.10 shades, starts at row {w6.wRow}, column {w6.wCol}.
      </p>
      <Beyond>
        <p>
          The book gives the layout as a figure and leaves the code to its Exercise 3. These formulas are ours, written so
          W can be 64 too; lab 15.3 and the model behind this page's pictures use them.
        </p>
      </Beyond>
      <LanePicker />
      <p>
        The animation puts the steps in order: the bad store, the re-partition one level at a time, the same store
        instruction replayed, and what changes with 64 lanes.
      </p>
      <RemapScene />
      <Ask q={questionsS155['ch15/15.5/q-remap-sectors']} label="Work it out" />

      <h2>Registers do not change</h2>
      <p>
        The remap changes which elements a thread owns, not how many. The 64 accumulators are still 64 registers. In each
        step of the inner loop the thread needs a piece of 4 values of the A strip for each of its two quadrant rows and a
        piece of 4 values of the B strip for each of its two quadrant columns: 16 values, the same 8 + 8 as before. What
        does change is the code: <code>mm</code> and <code>writeTile</code> must walk over the four sub-tiles, and{' '}
        <code>writeTile</code> issues one float4 per sub-tile row. The book shows no code for this. It is Exercise 3, and
        it is <a href={href('ch15', 'labs')}>lab 15.3</a>.
      </p>
      <Source section="15.5" pages="361" figs="Fig. 15.10, right panel" />
      <Ask q={questionsS155['ch15/15.5/q-registers']} />

      <h2>When a warp is not 32 lanes</h2>
      <p>
        Every number so far assumed 32 lanes, as the book does. Under HIP, AMD wavefronts are 64 lanes on CDNA (gfx9) and
        32 on RDNA (gfx10 and later). On Metal the SIMD-group width is whatever the compute pipeline reports, with no
        documented value.
      </p>
      <Today checked="2026-11-20 (notes/today/ch15-facts.md B1, C1, D1; ch05-facts.md B5, C3)">
        <p>
          AMD: "Since warpSize can differ between devices, it can not be assumed to be a compile-time constant on the
          host." Apple: the width is "returned by the threadExecutionWidth of your compute pipeline state object".
        </p>
      </Today>
      <p>
        The layout carries over. With 256 threads a CDNA block is 4 wavefronts. Keep 8 lane rows and four 4 × 4 squares per
        lane: the lanes become 8 × {g64.LC}, the quadrants {g64.QR} × {g64.QC}, and the wavefront tiles {g64.warpRows} ×{' '}
        {g64.warpCols}, arranged {g64.WR} × {g64.WC}. A lane row is now {g64.LC} lanes and writes {g64.laneRowBytes}{' '}
        contiguous bytes. Switch the lane picker and the visualiser to 64 lanes to see it.
      </p>
      <Beyond>
        <p>The 64-lane layout is our derivation (ch15-facts B1). AMD documents the width, not this layout.</p>
      </Beyond>
      <Ask q={questionsS155['ch15/15.5/q-wave64']} />
      <p>
        The layout is arithmetic on constants: the quadrant width, the warp tiles per row. So the width has to be a
        compile-time constant, chosen to match the device. AMD documents two ways: query the warp size on the host and
        instantiate a kernel template for it, or, where a device-side constant is needed, define it from the target, 64
        when <code>__GFX8__</code> or <code>__GFX9__</code> is defined and 32 otherwise. llama.cpp's HIP build does the
        second. Lab 15.3 does the first: the kernel is a template over the width, and the host launches the instance for
        the width the device reports. On Metal, read <code>threadExecutionWidth</code> from the pipeline on the host and
        compile the kernel for it, which is what lab 15.3's Metal host does; MLX and llama.cpp simply assume 32.
      </p>
      <Today checked="2026-11-20 (notes/today/ch15-facts.md B1, C1, D1)">
        <p>
          AMD's recommended approach is "to determine the warp size of the GPU on host side and setup the kernel
          accordingly", or a <code>#define</code> under <code>#if defined(__GFX8__) || defined(__GFX9__)</code>. llama.cpp:{' '}
          <code>ggml_cuda_get_physical_warp_size()</code> returns 64 under that test, else 32. MLX: <code>tgp_size = WM * WN * 32</code>.
        </p>
      </Today>
      <p>
        Whatever the backend, never write a literal 32 for the width. A layout built from a named constant can be rebuilt
        for another width; a literal hides the assumption until the store pattern quietly degrades.
      </p>
      <Ask q={questionsS155['ch15/15.5/q-width-wrong']} />
      <p>
        That answer is worth keeping in mind for the labs. The mapping is correct for whatever width it was compiled with,
        because it is pure index arithmetic: each element has one owner. Only the coalescing depends on the compiled width
        matching the hardware. That is why lab 15.3 tests both widths against the CPU reference on every backend, and its
        launch picks the width your device reports.
      </p>

      <h2>Alignment</h2>
      <p>
        A float4 store writes 16 bytes and needs a 16-byte-aligned address; the book does not mention it. Row r of C starts
        4·N·r bytes after C's first element, and on CUDA that element is aligned: <code>cudaMalloc</code> returns memory
        aligned to at least 256 bytes. So when N is a multiple of 4, every row starts on a 16-byte boundary and every
        sub-tile row can be one float4. When it is not, most rows do not. The kernel decides once: the vector path when{' '}
        <code>N % 4 == 0</code>, bounds-checked scalar stores otherwise. The same <code>.cu</code> on HIP gets C from{' '}
        <code>hipMalloc</code>, whose alignment none of this course's facts files documents, so there the rule is
        unverified.
      </p>
      <Beyond>
        <p>
          Ours, from ch06-facts A3 (<code>cudaMalloc</code>: "at least 256 bytes"), A11 (NVIDIA: "These instructions
          require aligned data") and ch15-facts E1's alignment note.
        </p>
      </Beyond>
      <p>
        The store itself is the book's chapter 6 idiom: cast the <code>float*</code> to a <code>float4*</code> and assign.
        No vendor reference documents that cast; NVIDIA's developer blog shows it, with the alignment rule above. Metal has a
        documented alternative, and lab 15.3 uses it: the host binds C a second time as a <code>device float4*</code>{' '}
        buffer, and the kernel stores whole float4 elements through it.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A11, C3; infra-facts.md A6, C7)">
        <p>
          NVIDIA's blog: vector types "enable the compiler to generate these wider memory operations through simple pointer
          casting with reinterpret_cast". Apple's route is a buffer declared <code>device float4*</code> (16-byte aligned).
        </p>
      </Today>
      <p>
        Set N to 1020 in the visualiser and step through the tile rows: odd rows start 16 bytes into a sector, so each
        64-byte run spans three sectors instead of two. A multiple of 4 is enough to be correct; a multiple of 8 keeps every
        run on a sector boundary. Then set N to 1001: the stores fall back to scalars.
      </p>
      <Ask q={questionsS155['ch15/15.5/q-align']} />

      <h2>The other way: fix the stores afterwards</h2>
      <p>
        CUTLASS, NVIDIA's open-source GEMM library, solves the same problem the other way round. It keeps whichever thread
        mapping is best for the arithmetic and adds an <b>epilogue</b>: threads write their results to shared memory,
        synchronise, and read them back in a pattern chosen for coalesced global stores. llama.cpp's Metal matrix kernel does
        the same for tiles at the edge of the matrix: it stages them in threadgroup memory and copies them out with float4
        stores, while full tiles go straight to device memory. When you read a production GEMM, look for one of the two.
      </p>
      <Today checked="2026-11-20 (notes/today/ch15-facts.md A12, D2)">
        <p>
          CUTLASS: "The epilogue is a separate phase in which threads exchange data through shared memory then cooperatively
          access global memory using efficient striped access patterns." llama.cpp, <code>mul_mm.metal</code>: "if no bounds
          checks on the output are needed, we can directly write to device memory".
        </p>
      </Today>
      <Ask q={questionsS155['ch15/15.5/q-epilogue']} />

      <h2>What the remap does to shared memory</h2>
      <p>
        Changing which elements a thread owns also changes which rows of <code>A_s</code> and columns of{' '}
        <code>B_s</code> it reads in each step. In the §15.3 layout a warp's A-strip read was a 2-way bank conflict and its
        B-strip read 4-way. After the remap the A-strip read is 8-way, and B's conflict is gone. Section 15.6 removes the
        8-way conflict with a single padding column: <a href={href('ch15', '15.6')}>eliminating bank conflicts</a>.
      </p>
      <Beyond title="Our arithmetic">
        <p>The before-and-after counts are from ch15-facts E1, with NVIDIA's 32-bank rule; the book analyses only the remapped layout.</p>
      </Beyond>
    </>
  );
}
