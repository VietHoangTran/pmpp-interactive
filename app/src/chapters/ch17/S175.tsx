import { useId, useMemo, useRef, useState, type CSSProperties, type ReactNode, type RefObject } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Beyond, Bench, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { fmt, useReports, type Reports } from '../../lib/reports';
import { SEGMENT_MODELS, type SegmentModelId } from '../../sim/coalesce';
import { widthChoices } from '../../sim/warps';
import { countByRow, exclusiveScan, stableScatter } from '../../sim/sparseroute';
import {
  FIG1713_LENGTHS,
  PRESETS,
  amortiseIterations,
  colsOf,
  csr,
  ell,
  hybrid,
  laneStates,
  maxOf,
  preset,
  rowLengths,
  sliceChoices,
  sortedLengths,
  spills,
  storage,
  traffic,
  warps,
  type FormatId,
  type Preset,
  type Shape,
  type ShapeParams,
  type Store,
  type Traffic,
  type WarpStats,
  type Widths,
} from '../../sim/sparsescale';
import { RouteScene } from './RouteScene';

// Page 17.5: padding, spilling and sorting, priced on the explorer "Pad, spill or sort"
// (src/sim/sparsescale.ts, through divergence.ts loopBatch and coalesce.ts warpAccess),
// when preprocessing pays, and mixture-of-experts routing as exercise 3 (the scene
// "Routing tokens to experts", src/sim/sparseroute.ts). Every number stated comes from
// those models.

const W4: Widths = { value: 4, index: 4 };
const BOOK = rowLengths(preset('book').params);
const BOOK_CSR = csr(BOOK, W4);
const BOOK_ELL = ell(BOOK, W4);
const BOOK_RATIO = BOOK_ELL.numbers / BOOK_CSR.numbers; // 19.05
const FIG = [...FIG1713_LENGTHS];
const FIG_ELL = ell(FIG, W4);
const FIG_K2 = hybrid(FIG, 2, W4);
const FIG_SPILL = spills(FIG, 2).flatMap((s, r) => (s ? [`${s} from row ${r}`] : []));
const HK = [...Array(1000).fill(16), 400, 400];
const HK_HYB = hybrid(HK, 16, W4);
const HK_ELL = ell(HK, W4);
const AMORT = amortiseIterations(30, 0.6);
const MOE_IDS = [2, 0, 2, 1, 0, 2];
const MOE_COUNTS = countByRow(MOE_IDS, 3);
const MOE_OFFSETS = exclusiveScan(MOE_COUNTS);
const MOE_SCATTER = stableScatter(MOE_IDS, MOE_OFFSETS);
const MOE_E0 = MOE_SCATTER.order.slice(0, MOE_COUNTS[0]);
const list = (xs: readonly number[]) => '[' + xs.join(', ') + ']';
const big = (n: number) => n.toLocaleString('en-US');
const pct = (x: number) => (x * 100 >= 100 ? big(Math.round(x * 100)) : fmt(x * 100)) + '%';

export const questionsS175: Record<string, Question> = {
  'ch17/17.5/q-hybrid-k': {
    kind: 'numeric',
    id: 'ch17/17.5/q-hybrid-k',
    prompt: (
      <>
        A matrix has 1,000 rows of 16 non-zeros and 2 rows of 400. You build a hybrid whose ELL part is 16 wide. How
        many entries go to the COO part?
      </>
    ),
    answer: HK_HYB.coo,
    unit: 'entries',
    explain: (
      <p>
        Each long row keeps 16 in ELL and sends 384 to COO: 2 × 384 = {big(HK_HYB.coo)}, and no pad slots are left,
        since every row has at least 16. Pure ELL would pad all 1,002 rows to 400 and hold {big(HK_ELL.pads)} pad slots.
        The ELL kernel now runs 16 iterations in every thread, instead of two threads holding their warps for 400.
      </p>
    ),
    source: 'ch. 17 §17.5, pp. 414–415',
  },
  'ch17/17.5/q-sort-y': {
    kind: 'choice',
    id: 'ch17/17.5/q-sort-y',
    prompt: (
      <>
        JDS sorts the rows longest first before the product. Afterwards, which vector must be put back into the original
        order, and with what?
      </>
    ),
    options: [
      {
        label: (
          <>
            <code>x</code>, with the row array, since the threads read it.
          </>
        ),
        why: 'Sorting rows reorders equations, not unknowns. Every column index still names the same element of x.',
      },
      {
        label: (
          <>
            <code>y</code>, with the row array; <code>x</code> never moved.
          </>
        ),
        correct: true,
      },
      {
        label: (
          <>
            Both, each with its own permutation.
          </>
        ),
        why: 'Only rows moved. x is indexed by column, and no column moved.',
      },
      {
        label: 'Neither: sorting rows cannot change a product.',
        why: 'The values are right, but output r of the sorted product belongs to original row row[r].',
      },
    ],
    explain: (
      <p>
        Thread <code>r</code> computes the dot product of sorted row <code>r</code>, which is original row{' '}
        <code>row[r]</code>, so its result belongs in <code>y[row[r]]</code>. The column indices are untouched, so{' '}
        <code>x</code> is read in place. The book's text slips once and speaks of reordering the solution's{' '}
        <code>x</code> elements (p. 417); the notes record it, and this is the reading that keeps the arithmetic right.
      </p>
    ),
    source: 'ch. 17 §17.6, p. 417 (notes/ch17.md, "Unclear in the PDF" item 5)',
  },
  'ch17/17.5/q-jds-loop': {
    kind: 'choice',
    id: 'ch17/17.5/q-jds-loop',
    prompt: (
      <>
        Exercise 5 asks for a JDS kernel. One thread per sorted row <code>r</code>, rows longest first; iteration{' '}
        <code>t</code>'s elements start at <code>iterPtr[t]</code>. Which test lets thread <code>r</code> take an element
        in iteration <code>t</code>?
      </>
    ),
    options: [
      {
        label: <code>t &lt; iterPtr[r+1] - iterPtr[r]</code>,
        why: 'That is CSR’s shape, one pointer per row. iterPtr has one entry per iteration, not per row.',
      },
      { label: <code>r &lt; numRows</code>, why: 'Every thread passes it, and reads past the end of iteration t into iteration t + 1.' },
      {
        label: <code>colIdx[iterPtr[t] + r] &gt;= 0</code>,
        why: 'JDS has no pads to mark. Past the end of iteration t, that address belongs to iteration t + 1.',
      },
      { label: <code>r &lt; iterPtr[t+1] - iterPtr[t]</code>, correct: true },
    ],
    explain: (
      <p>
        <code>iterPtr[t+1] - iterPtr[t]</code> is the number of rows that still have a <code>t</code>-th element.
        Because rows are sorted longest first, those are exactly sorted rows 0 up to that number minus one, so thread{' '}
        <code>r</code> takes part while <code>r</code> is below it. Its element is at <code>iterPtr[t] + r</code>:
        consecutive threads read consecutive addresses, as in ELL, with no padding. The book leaves the kernel as an
        exercise; this test is our answer, not the book's.
      </p>
    ),
    source: 'ch. 17 exercise 5, p. 422; Figs. 17.14, 17.15',
  },
  'ch17/17.5/q-jds-align': {
    kind: 'choice',
    id: 'ch17/17.5/q-jds-align',
    prompt: <>ELL can make every iteration start on a segment boundary at little cost. Why can JDS not do the same?</>,
    options: [
      { label: 'JDS stores its elements row by row.', why: 'It is column-major, like ELL (Fig. 17.14).' },
      {
        label: (
          <>
            Iteration <code>t</code> starts at <code>iterPtr[t]</code>, set by how many rows reach each length.
          </>
        ),
        correct: true,
      },
      { label: 'Sorting the rows breaks coalescing.', why: 'Within an iteration, consecutive threads still read consecutive elements.' },
      { label: 'JDS needs atomics, which ignore alignment.', why: 'One thread owns each sorted row: there are no atomics.' },
    ],
    explain: (
      <p>
        ELL's iteration <code>t</code> starts at element <code>t*numRows</code>. Make <code>numRows</code> a multiple of 8
        and, with 4-byte values and an aligned base, every start falls on a 32-byte boundary (a multiple of 16 or 32 for
        AMD's 64- or 128-byte lines). JDS's starts are sums of row counts, arbitrary numbers, and only padding could align
        them, which is what JDS exists to avoid. The book's example unit, "such as 64-bytes" for unnamed older GPUs, is in
        no NVIDIA document read today (ch17-facts A3). NVIDIA's unit today is the 32-byte sector.
      </p>
    ),
    source: 'ch. 17 pp. 413, 417; ch17-facts A3, E1',
  },
  'ch17/17.5/q-amortise': {
    kind: 'numeric',
    id: 'ch17/17.5/q-amortise',
    prompt: (
      <>
        Converting a CSR matrix to a sorted, padded format takes as long as 30 CSR SpMVs. Each SpMV in the new format then
        takes 0.6 of the CSR time. After how many solver iterations has the conversion paid for itself?
      </>
    ),
    answer: Math.round(AMORT),
    tol: 0.01,
    unit: 'iterations',
    explain: (
      <p>
        Each iteration saves 0.4 of a CSR SpMV, so it takes 30 / 0.4 = {fmt(AMORT)} iterations to win back the 30. A
        solver that converges in 50 iterations should not convert; one that runs 500 should. The vendor libraries give
        the same advice about their own preprocessing (below).
      </p>
    ),
    source: 'ch. 17 §17.5 and §17.6, pp. 414–415, 417',
  },
  'ch17/17.5/q-moe-offset': {
    kind: 'numeric',
    id: 'ch17/17.5/q-moe-offset',
    prompt: (
      <>
        Six token slots are routed to experts {list(MOE_IDS)}, with three experts. After counting, an exclusive scan and a
        stable scatter, at which position of the expert-sorted order does slot 4 land?
      </>
    ),
    answer: MOE_SCATTER.pos[4],
    unit: '',
    explain: (
      <p>
        Counts {list(MOE_COUNTS)}, offsets {list(MOE_OFFSETS)}. Expert 0 holds slots {MOE_E0.join(' and ')}, in that
        order, so slot 4 lands at {MOE_OFFSETS[0]} + {MOE_SCATTER.rank[4]} = {MOE_SCATTER.pos[4]}. The scan gave expert
        0's start; the rank, the number of earlier slots routed to the same expert, gave the rest.
      </p>
    ),
    source: 'ours, after ch. 17 exercise 3, p. 422',
  },
  'ch17/17.5/q-moe-atomic': {
    kind: 'choice',
    id: 'ch17/17.5/q-moe-atomic',
    prompt: (
      <>
        llama.cpp's <code>mm_ids_helper</code> ranks each entry inside its expert's slice with a ballot and a popcount
        over the lanes, not with an atomically incremented cursor per expert. What does that buy?
      </>
    ),
    options: [
      { label: 'Fewer bytes moved per token.', why: 'Both write the same ids into the same slices; only the order inside a slice differs.' },
      { label: 'No need for the exclusive scan.', why: 'The scan still gives each slice its start; the rank only places an entry inside it.' },
      { label: 'The same order inside every slice, every run.', correct: true },
      { label: 'No need to count the entries first.', why: 'The slices’ starts still come from the counts, through the scan.' },
    ],
    explain: (
      <p>
        An atomic cursor hands out ranks in the order the threads' atomics land, which is undefined, so the order inside
        a slice can change from run to run. A ballot over the lanes below, followed by a popcount, ranks each entry by its
        place in the input, as the model's stable scatter does: tokens stay in input order inside each expert's slice.
        The price is that one block works through all the ids for its expert, chunk by chunk.
      </p>
    ),
    source: 'ch17-facts D5; ballot and popcount: ch. 12; atomics: ch. 9',
  },
};

// ---------------------------------------------------------------- explorer state
interface XState {
  preset: Preset['id'] | null;
  params: ShapeParams;
  K: number;
  sorted: boolean;
  S: number;
  widthKey: string;
  value: number;
  index: number;
  model: SegmentModelId;
  /** Iteration shown in the lane grid; -1 until the reader moves the slider. */
  t: number;
}
function fromPreset(id: Preset['id'], prev?: XState, K?: number): XState {
  const params = { ...preset(id).params };
  const len = rowLengths(params);
  return {
    preset: id,
    params,
    K: K ?? maxOf(len),
    sorted: false,
    S: len.length,
    widthKey: prev?.widthKey ?? 'book',
    value: prev?.value ?? 4,
    index: prev?.index ?? 4,
    model: prev?.model ?? 'nvidia',
    t: -1,
  };
}
const SHAPES: { id: Shape; label: string }[] = [
  { id: 'uniform', label: 'uniform' },
  { id: 'spread', label: 'spread from 1 to 2 × mean − 1' },
  { id: 'long', label: 'a few long rows, total kept' },
  { id: 'power', label: 'power law (ours)' },
  { id: 'fig1713', label: 'Fig. 17.13' },
];
const ROW_CHOICES = [256, 1000, 1024, 4096];
const VALUE_BYTES = [4, 2, 1, 0.5];
const INDEX_BYTES = [4, 2];
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const NARROW_OK: CSSProperties = { maxWidth: '100%', minWidth: 0 };
const Wide = ({ children, style }: { children: ReactNode; style?: CSSProperties }) => (
  <div style={{ overflowX: 'auto', maxWidth: '100%', ...style }}>{children}</div>
);
const byteLabel = (b: number) => (b < 1 ? '4 bits' : `${b} B`);

const FORMAT_ORDER: FormatId[] = ['dense', 'csr', 'ell', 'hybrid', 'jds', 'sell'];
function formatLabel(s: Store, x: XState): string {
  switch (s.id) {
    case 'dense':
      return 'dense';
    case 'csr':
      return 'CSR';
    case 'ell':
      return `ELL, ${s.width} wide`;
    case 'hybrid':
      return `hybrid: ELL ${s.width} wide + ${big(s.coo)} COO`;
    case 'jds':
      return 'JDS';
    case 'sell':
      return `sliced ELL, slices of ${x.S}${x.sorted ? ', sorted' : ''}`;
  }
}

// ---------------------------------------------------------------- explorer panels
function StoragePanel({ table, x, w }: { table: ReturnType<typeof storage>; x: XState; w: Widths }) {
  const dense = table.stores.dense;
  const maxBytes = Math.max(...FORMAT_ORDER.map((f) => table.stores[f].bytes));
  return (
    <div className="panel" data-panel="storage">
      <div className="mx-label">Storage: values and indices as the book counts them, and bytes at the widths chosen</div>
      <Wide>
        <table className="threads" aria-label="Storage of each format">
          <thead>
            <tr>
              <th>Format</th>
              <th>numbers</th>
              <th>bytes</th>
              <th>of dense</th>
              <th>pad slots</th>
              <th style={{ minWidth: '9rem' }}></th>
            </tr>
          </thead>
          <tbody>
            {FORMAT_ORDER.map((f) => {
              const s = table.stores[f];
              const padBytes = s.pads * (w.value + w.index);
              return (
                <tr key={f} data-format={f}>
                  <td>{formatLabel(s, x)}</td>
                  <td data-out="numbers">{big(s.numbers)}</td>
                  <td data-out="bytes">{big(Math.round(s.bytes))}</td>
                  <td data-out="pct">{pct(s.bytes / dense.bytes)}</td>
                  <td data-out="pads">{f === 'dense' ? '' : big(s.pads)}</td>
                  <td>
                    <div style={{ display: 'flex', height: 10, width: '100%', minWidth: '9rem' }} aria-hidden="true">
                      <div style={{ width: `${(100 * (s.bytes - padBytes)) / maxBytes}%`, background: 'var(--global)' }} />
                      <div style={{ width: `${(100 * padBytes) / maxBytes}%`, background: 'var(--global-wash)', borderRight: padBytes ? '1px dashed var(--global)' : undefined }} />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Wide>
      <p className="muted" style={{ margin: '0.5rem 0 0' }}>
        Solid: data, indices and pointers. Pale: padding. Pointer arrays (rowPtrs, iterPtr, the row array, slice offsets)
        take 4 bytes each.
      </p>
    </div>
  );
}

const hatchStyle = (id: string) => `url(#${id})`;
function LaneGrid({ len, width, t }: { len: number[]; width: number; t: number }) {
  const hatch = useId();
  const sorted = useMemo(() => sortedLengths(len), [len]);
  const c = width <= 32 ? 12 : 7;
  const LEFT = 150;
  const rh = 18;
  const rows = [
    { key: 'orig-0', label: 'row order, warp 0', trips: len, warp: 0 },
    { key: 'orig-1', label: 'row order, warp 1', trips: len, warp: 1 },
    { key: 'sort-0', label: 'sorted, warp 0', trips: sorted, warp: 0 },
    { key: 'sort-1', label: 'sorted, warp 1', trips: sorted, warp: 1 },
  ];
  const Wd = LEFT + width * c + 110;
  const Ht = 22 + rows.length * rh + 8;
  return (
    <svg viewBox={`0 0 ${Wd} ${Ht}`} role="img" aria-label={`Which lanes of the first two warps still work in iteration ${t}, in row order and sorted`}
      style={{ width: Wd, display: 'block' }} data-view="lanes">
      <defs>
        <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.2" />
        </pattern>
      </defs>
      <text x={LEFT - 8} y={13} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>lane</text>
      {[0, width / 2, width - 1].map((l) => (
        <text key={l} x={LEFT + l * c + c / 2} y={13} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{l}</text>
      ))}
      {rows.map((r, i) => {
        const st = laneStates(r.trips, width, r.warp, t);
        const working = st.filter((s) => s === 'work').length;
        const real = st.filter((s) => s !== 'absent').length;
        const y = 22 + i * rh + (i >= 2 ? 6 : 0);
        return (
          <g key={r.key} data-grid-row={r.key} data-work={working}>
            <text x={LEFT - 8} y={y + 11} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{r.label}</text>
            {st.map((s, l) => (
              <g key={l}>
                <rect x={LEFT + l * c} y={y} width={c - 1} height={rh - 5} rx="1.5"
                  fill={s === 'work' ? 'var(--ink)' : s === 'done' ? 'var(--cell)' : 'none'}
                  stroke={s === 'work' ? 'none' : 'var(--rule)'} strokeWidth="0.6" strokeDasharray={s === 'absent' ? '2 2' : undefined} />
                {s === 'done' ? <rect x={LEFT + l * c} y={y} width={c - 1} height={rh - 5} rx="1.5" fill={hatchStyle(hatch)} pointerEvents="none" /> : null}
              </g>
            ))}
            <text x={LEFT + width * c + 8} y={y + 11} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
              {real ? `${working} of ${real} work` : 'no rows'}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function WarpPanel({ wt, len, width, x, setT }: { wt: ReturnType<typeof warps>; len: number[]; width: number; x: XState; setT: (t: number) => void }) {
  const firstTwo = (trips: number[]) => maxOf(trips.slice(0, 2 * width));
  const tMax = Math.max(1, firstTwo(len), firstTwo(sortedLengths(len)));
  // until the reader moves the slider, show the iteration at the mean row length: where lanes start to drop out
  const auto = Math.round(len.reduce((a, b) => a + b, 0) / Math.max(1, len.length));
  const t = Math.max(0, Math.min(x.t < 0 ? auto : x.t, tMax - 1));
  const row = (id: string, name: ReactNode, s: WarpStats) => (
    <tr data-warps={id}>
      <td>{name}</td>
      <td data-out="spent">{big(s.spent)}</td>
      <td data-out="useful">{big(s.useful)}</td>
      <td data-out="efficiency">{pct(s.efficiency)}</td>
      <td data-out="longest">{big(s.longest)}</td>
    </tr>
  );
  return (
    <div className="panel" data-panel="warps">
      <div className="mx-label">Warps: one thread per row, one iteration per non-zero</div>
      <Wide>
        <table className="threads" aria-label="What each order costs a per-row kernel">
          <thead>
            <tr>
              <th>Kernel</th>
              <th>warp-iterations</th>
              <th>lane-iterations of work</th>
              <th>lane time that is work</th>
              <th>longest trip</th>
            </tr>
          </thead>
          <tbody>
            {row('original', 'CSR or ELL, row order', wt.original)}
            {row('sorted', 'JDS, longest first', wt.sorted)}
            {row('hybrid', `hybrid's ELL part, ${x.K} wide`, wt.hybrid)}
          </tbody>
        </table>
      </Wide>
      <label className="field" style={{ marginTop: '0.7rem', width: '100%' }}>
        <span>
          Iteration <b data-out="t">{t}</b> of the first two warps
        </span>
        <input type="range" min={0} max={tMax - 1} value={t} aria-label="Iteration shown in the lane grid" onChange={(e) => setT(Number(e.target.value))} />
      </label>
      <Wide style={{ marginTop: '0.4rem' }}>
        <LaneGrid len={len} width={width} t={t} />
      </Wide>
      <p className="muted" style={{ margin: '0.4rem 0 0' }}>
        Filled: the lane works this iteration. Hatched: its row is done and it waits for the warp's longest. Dashed: no row.
        The hybrid's COO part runs one thread per entry and has no loop.
      </p>
    </div>
  );
}

function TrafficPanel({ tr, model }: { tr: Traffic[] | null; model: SegmentModelId }) {
  const m = SEGMENT_MODELS[model];
  if (!tr || m.segmentBytes == null)
    return (
      <div className="panel" data-panel="traffic" data-hidden="1">
        <div className="mx-label">Requested segments</div>
        <p style={{ margin: 0 }}>
          Apple documents no segment size, line size or coalescing rule ({m.facts}), so there is nothing to count against.
          Choose NVIDIA's or AMD's unit to see the counts; on Metal, the lab measures instead.
        </p>
      </div>
    );
  return (
    <div className="panel" data-panel="traffic">
      <div className="mx-label">Requested segments of the value and index arrays, {m.label}</div>
      <Wide>
        <table className="threads" aria-label="Segments each format's kernel requests">
          <thead>
            <tr>
              <th>Kernel</th>
              <th>{m.unit}s requested, no cache</th>
              <th>distinct {m.unit}s, perfect cache</th>
              <th>bytes used</th>
              <th>used / requested</th>
            </tr>
          </thead>
          <tbody>
            {tr.map((r) => (
              <tr key={r.format} data-traffic={r.format}>
                <td>{r.format === 'csr' ? 'CSR' : r.format === 'ell' ? 'ELL' : 'JDS'}</td>
                <td data-out="requested">{big(r.requested)}</td>
                <td data-out="distinct">{big(r.distinct)}</td>
                <td data-out="needed">{big(Math.round(r.needed))}</td>
                <td data-out="efficiency">{pct(r.efficiency)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Wide>
      <p className="muted" style={{ margin: '0.5rem 0 0' }}>
        The first column counts every warp's every load as if nothing were cached; the second counts each {m.unit} once. A
        real device moves something between the two. ELL's kernel stops at each row's length (Fig. 17.12), so it requests
        no padding.
      </p>
    </div>
  );
}

function Explorer({ x, setX, anchor }: { x: XState; setX: (f: (s: XState) => XState) => void; anchor: RefObject<HTMLDivElement | null> }) {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports as Reports), [reports]);
  const chosen = choices.find((c) => c.key === x.widthKey) ?? choices[0];
  const width = chosen.width;
  const len = useMemo(() => rowLengths(x.params), [x.params]);
  const cols = colsOf(x.params);
  const w: Widths = useMemo(() => ({ value: x.value, index: x.index }), [x.value, x.index]);
  const maxLen = maxOf(len);
  const K = Math.min(x.K, maxLen);
  const S = sliceChoices(len.length).includes(x.S) ? x.S : len.length;
  const table = useMemo(() => storage(len, cols, w, K, S, x.sorted), [len, cols, w, K, S, x.sorted]);
  const wt = useMemo(() => warps(len, width, K), [len, width, K]);
  const seg = SEGMENT_MODELS[x.model].segmentBytes;
  const tr = useMemo(
    () => (seg == null ? null : (['csr', 'ell', 'jds'] as const).map((f) => traffic(f, len, width, seg, w))),
    [len, width, seg, w],
  );
  const nnz = table.nnz;
  const mean = nnz / len.length;
  const ratio = table.stores.ell.numbers / table.stores.csr.numbers;
  const hyb = table.stores.hybrid;
  const fixed = x.params.shape === 'fig1713';
  const isLong = x.params.shape === 'long';

  const edit = (p: Partial<ShapeParams>) =>
    setX((s) => {
      const params = { ...s.params, ...p };
      if (params.shape === 'fig1713') params.rows = 8;
      else if (s.params.shape === 'fig1713') params.rows = 1024;
      const before = rowLengths(s.params);
      const after = rowLengths(params);
      // a K at the old longest row means "pure ELL": keep meaning that
      const K = s.K >= maxOf(before) ? maxOf(after) : Math.min(s.K, maxOf(after));
      const S = after.length !== before.length ? after.length : s.S;
      return { ...s, preset: null, params, K, S, t: -1 };
    });

  return (
    <div ref={anchor} className="wide">
      <Bench
        title="Pad, spill or sort"
        task="Price ELL's padding, then remove it three ways."
        foot={
          <>
            Runs on src/sim/sparsescale.ts: row lengths from a seeded generator, storage counted as the book counts it,
            warps through chapter 4's loop model and segments through chapter 6's. {chosen.note}
          </>
        }
      >
        <div
          data-explorer="pad-spill-sort"
          data-preset={x.preset ?? ''}
          data-rows={len.length}
          data-cols={cols}
          data-nnz={nnz}
          data-max={maxLen}
          data-k={K}
          data-s={S}
          data-sorted={x.sorted ? 1 : 0}
          data-width={width}
          data-value={x.value}
          data-index={x.index}
          data-model={x.model}
        >
          <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
            <div style={FIELD}>
              Preset
              <span className="seg" role="group" aria-label="Presets" style={{ flexWrap: 'wrap', rowGap: 4 }}>
                {PRESETS.map((p) => (
                  <button key={p.id} aria-pressed={x.preset === p.id} onClick={() => setX((s) => fromPreset(p.id, s))} title={p.note}>
                    {p.label}
                  </button>
                ))}
              </span>
            </div>
          </div>
          <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
            <label className="field" style={NARROW_OK}>
              Row lengths
              <select value={x.params.shape} aria-label="Shape of the row lengths" onChange={(e) => edit({ shape: e.target.value as Shape })}>
                {SHAPES.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
              </select>
            </label>
            <label className="field">
              Rows (square matrix)
              <select value={len.length} disabled={fixed} aria-label="Rows" onChange={(e) => edit({ rows: Number(e.target.value) })}>
                {(fixed ? [8] : ROW_CHOICES).map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Mean per row: <b data-out="mean-param">{x.params.mean}</b></span>
              <input type="range" min={1} max={64} value={x.params.mean} disabled={fixed} aria-label="Mean non-zeros per row" onChange={(e) => edit({ mean: Number(e.target.value) })} />
            </label>
            <label className="field">
              Long rows
              <input type="number" min={0} max={64} value={x.params.longCount} disabled={!isLong} aria-label="Number of long rows"
                onChange={(e) => edit({ longCount: Math.max(0, Math.min(64, Math.floor(Number(e.target.value) || 0))) })} style={{ width: '5rem' }} />
            </label>
            <label className="field">
              <span>Long-row length: <b>{x.params.longLen}</b></span>
              <input type="range" min={1} max={cols} value={Math.min(cols, Math.max(1, x.params.longLen))} disabled={!isLong} aria-label="Length of the long rows"
                onChange={(e) => edit({ longLen: Number(e.target.value) })} />
            </label>
          </div>
          <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
            <label className="field">
              <span>Hybrid ELL width K: <b data-out="k">{K}</b></span>
              <input type="range" min={0} max={maxLen} value={K} aria-label="Hybrid ELL width K" onChange={(e) => setX((s) => ({ ...s, K: Number(e.target.value) }))} />
            </label>
            <label className="field">
              Slice size S
              <select value={S} aria-label="Slice size" onChange={(e) => setX((s) => ({ ...s, S: Number(e.target.value) }))}>
                {sliceChoices(len.length).map((v) => <option key={v} value={v}>{v === len.length ? `${v} (plain ELL)` : v}</option>)}
              </select>
            </label>
            <label className="check">
              <input type="checkbox" checked={x.sorted} aria-label="Sort rows longest first" onChange={(e) => setX((s) => ({ ...s, sorted: e.target.checked }))} /> sort rows longest first
            </label>
          </div>
          <div className="row" style={{ marginBottom: '1rem', alignItems: 'flex-end' }}>
            <label className="field" style={NARROW_OK}>
              Warp width
              <select value={chosen.key} aria-label="Warp width" onChange={(e) => setX((s) => ({ ...s, widthKey: e.target.value, t: -1 }))}>
                {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
              </select>
            </label>
            <div style={FIELD}>
              Bytes per value
              <span className="seg" role="group" aria-label="Bytes per value">
                {VALUE_BYTES.map((b) => (
                  <button key={b} aria-pressed={x.value === b} onClick={() => setX((s) => ({ ...s, value: b }))}>{byteLabel(b)}</button>
                ))}
              </span>
            </div>
            <div style={FIELD}>
              Bytes per index
              <span className="seg" role="group" aria-label="Bytes per index">
                {INDEX_BYTES.map((b) => (
                  <button key={b} aria-pressed={x.index === b} onClick={() => setX((s) => ({ ...s, index: b }))}>{byteLabel(b)}</button>
                ))}
              </span>
            </div>
            <label className="field" style={NARROW_OK}>
              Segment model
              <select value={x.model} aria-label="Segment model" onChange={(e) => setX((s) => ({ ...s, model: e.target.value as SegmentModelId }))}>
                {Object.values(SEGMENT_MODELS).map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
            </label>
          </div>

          <p className="narrate" data-readout="summary" style={{ marginTop: 0 }}>
            {big(len.length)} × {big(cols)}, <b data-out="nnz">{big(nnz)}</b> non-zeros ({pct(nnz / (len.length * cols))}), a
            mean row of <b data-out="mean">{fmt(mean)}</b> and a longest of <b data-out="max">{big(maxLen)}</b>. ELL stores{' '}
            <b data-out="ratio">{fmt(ratio, 4)}</b> times the numbers of CSR: its width is the longest row, not the mean. At K ={' '}
            {K} the hybrid keeps <b data-out="hyb-pads">{big(hyb.pads)}</b> pad slots and moves{' '}
            <b data-out="hyb-coo">{big(hyb.coo)}</b> entries to COO.
          </p>
          <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'minmax(0, 1fr)' }}>
            <StoragePanel table={table} x={{ ...x, S }} w={w} />
            <WarpPanel wt={wt} len={len} width={width} x={{ ...x, K }} setT={(t) => setX((s) => ({ ...s, t }))} />
            <TrafficPanel tr={tr} model={x.model} />
          </div>
        </div>
      </Bench>
    </div>
  );
}

// ---------------------------------------------------------------- the page
export function S175() {
  const [x, setXRaw] = useState<XState>(() => fromPreset('book'));
  const setX = (f: (s: XState) => XState) => setXRaw(f);
  const anchor = useRef<HTMLDivElement | null>(null);
  const show = (id: Preset['id'], K?: number) => {
    setXRaw((s) => fromPreset(id, s, K));
    anchor.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
  return (
    <>
      <p>
        Page <a href={href('ch17', '17.2')}>17.2</a> left ELL with two problems: one long row pads every other row, and
        the threads still diverge. This page prices both on rows of uneven length, then buys regularity three ways: spill
        the tails of long rows into COO, sort the rows by length, or pad slices instead of the whole matrix. Each costs a
        preprocessing pass, so the page ends with when that pays, and with the place you will meet all of it in LLM
        inference.
      </p>
      <Source section="17.4–17.6" pages="412–418" />

      <h2>Padding is priced by the longest row</h2>
      <p>
        ELL pads every row to the length of the longest. The book's example: a 1000 × 1000 matrix with 1% non-zeros, so
        10 per row on average, except that one row has 200. Count values and indices, as the book does, and compare ELL
        with CSR.
      </p>
      <Source section="17.4" pages="412–413" />
      <Predict
        id="ch17/17.5/p-book1000"
        prompt={<>For the book's example, how many times as many numbers does ELL store as CSR?</>}
        unit="×"
      >
        {(v) => (
          <div data-reveal="ch17/17.5/p-book1000">
            <p>
              CSR: 2 × 10,000 + 1,001 = <b>{big(BOOK_CSR.numbers)}</b> numbers, {pct(BOOK_CSR.numbers / 1e6)} of the dense
              matrix. ELL pads all 1,000 rows to 200: 2 × 200,000 = <b>{big(BOOK_ELL.numbers)}</b>,{' '}
              {pct(BOOK_ELL.numbers / 1e6)}. The ratio is <b data-out="ratio">{fmt(BOOK_RATIO, 4)}</b>; the book rounds it
              to 20. {Math.abs(v - BOOK_RATIO) < 1.5 ? 'You had it.' : 'One row decided the width of all 1,000.'}
            </p>
            <button onClick={() => show('book')}>Load the book's example in the explorer</button>
          </div>
        )}
      </Predict>
      <p>
        The book also says the other rows have "fewer than 10" non-zeros. That cannot hold together with a 1% total: 999
        rows of at most 9 plus one of 200 make at most 9,191, short of 10,000. The explorer keeps the 1% and gives the
        other rows 9 or 10. ELL's count does not change, since it depends only on the longest row.
      </p>
      <Beyond title="Our arithmetic">
        <p>
          The 19.05 is ours, checked in ch17-facts E1; the book prints "20×". The conflict is our arithmetic on the book's
          figures, 999 × 9 + 200 = 9,191; the book prints "fewer than 10" (pp. 412–413).
        </p>
      </Beyond>
      <p>
        The explorer stores the same rows six ways and runs a per-row kernel over them. Start on the book's example, then
        change one control at a time. Watch three things: ELL's size follows the longest row, not the mean; a hybrid width
        K near the typical row removes almost all the padding for a handful of COO entries; and sorting the rows, alone
        or with small slices, raises the share of lane time that does work. On the book's example the single row of 200
        hides that last effect: switch to the "Spread, 1 to 31" preset to see what the sort does to lane time.
      </p>
      <Simplification>
        <p>
          Row lengths are generated, never read from real matrices. Warps are counted, not timed. Traffic counts only the
          value and index arrays; <code>x</code>, <code>y</code> and the small per-row arrays are left out. There is no
          cache model between "every request" and "each segment once". Apple's segment size is not documented, so that
          choice hides the traffic panel.
        </p>
      </Simplification>
      <Explorer x={x} setX={setX} anchor={anchor} />

      <h2>Spill the tail: hybrid ELL-COO</h2>
      <p>
        When a few rows are much longer than the rest, cut them short. Keep the first K non-zeros of every row in ELL, move
        the rest of each long row into a COO list, then run SpMV/ELL on the ELL part and SpMV/COO on the leftovers. Using
        two formats for one product is what the book calls a hybrid method.
      </p>
      <Source section="17.5" pages="414" figs="Fig. 17.13" />
      <Predict
        id="ch17/17.5/p-fig1713"
        prompt={
          <>
            Fig. 17.13's eight rows have {FIG.join(', ')} non-zeros. Pure ELL is {FIG_ELL.width} wide and holds{' '}
            {FIG_ELL.pads} pad slots. Cut the ELL part to width 2 and send the rest of each row to COO. How many pad slots
            are left?
          </>
        }
        unit="pad slots"
      >
        {(v) => (
          <div data-reveal="ch17/17.5/p-fig1713">
            <p>
              <b data-out="pads">{FIG_K2.pads}</b>: rows 0, 4 and 7 have one non-zero each and need one pad each.{' '}
              <b data-out="coo">{FIG_K2.coo}</b> entries go to COO, {FIG_SPILL.join(' and ')}, and every thread now makes at
              most 2 iterations. The book's text says the three come from "row 2"; its figure and the counts say row 1
              (notes, "Unclear in the PDF" item 2). {v === FIG_K2.pads ? 'You had it.' : ''}
            </p>
            <button onClick={() => show('fig1713', 2)}>Show Fig. 17.13 at K = 2 in the explorer</button>
          </div>
        )}
      </Predict>
      <p>
        Both launches add into <code>y</code>, so they compose in either order: the COO pass adds a row's overflow on top
        of whatever the ELL pass wrote. That relies on the ELL kernel accumulating. The book's listing assigns{' '}
        <code>y[row] = sum</code>, which would force the ELL pass to run first; the corrected kernel on page{' '}
        <a href={href('ch17', '17.2')}>17.2</a> accumulates like the CSR one.
      </p>
      <Source section="17.5" pages="414" figs="Fig. 17.12" />
      <p>
        The price is accessibility. A row that overflowed is no longer in one place, and finding all its non-zeros means
        searching the COO part. Everything else improves on ELL alone: less space, less divergence in the ELL kernel, none
        in the COO kernel, and both still coalesce.
      </p>
      <Source section="17.5" pages="415" />
      <p>
        The vendors split on this format. NVIDIA dropped it; AMD still ships it and lets you choose the width of its ELL
        part.
      </p>
      <Today checked="2026-11-20 (ch17-facts A5, B3)">
        <p>
          NVIDIA removed the hybrid format from cuSPARSE in CUDA 11.0 (deprecated in 10.2). AMD's rocSPARSE still ships{' '}
          <code>rocsparse_hybmv</code>, with the ELL width chosen automatically, by you, or as the longest row with no COO
          part (<code>auto</code>, <code>user</code>, <code>max</code>).
        </p>
      </Today>
      <Ask q={questionsS175['ch17/17.5/q-hybrid-k']} />

      <h2>Sort by length: JDS</h2>
      <p>
        A hybrid removes the extremes; sorting removes the mixture. Sort the rows by length, longest first, and store them
        column-major with no padding: iteration <code>t</code> holds the <code>t</code>-th element of every row that has
        one, and because the rows are sorted, those are the first rows. An <code>iterPtr</code> array records where each
        iteration starts, and a <code>row</code> array records which original row each sorted row was. Neighbouring threads
        now get rows of similar length, so the threads of a warp have similar trip counts and divergence is much reduced.
        The book calls this Jagged Diagonal Storage.
      </p>
      <Source section="17.6" pages="416–417" figs="Figs. 17.14, 17.15" />
      <p>
        Two slips in the PDF sit here. The text says the rows are sorted "in increasing order", while the paragraph before
        it and Fig. 17.14 sort longest first; this course follows the figure. And the paragraph that scores JDS on p. 417
        opens with "Now we examine the ELL format": it means JDS. Both are in the notes' list of unclear places (items 3
        and 4).
      </p>
      <Source section="17.6" pages="416–417" />
      <Ask q={questionsS175['ch17/17.5/q-sort-y']} />
      <p>
        Sorting has two costs besides the sort. Flexibility: adding a non-zero changes a row's length and may force a new
        sort. Alignment: iteration <code>t</code> starts at <code>iterPtr[t]</code>, wherever the row counts put it, so
        there is no cheap way to make every iteration start on a segment boundary, as ELL can by padding its row count.
      </p>
      <Source section="17.6" pages="417–418" />
      <Ask q={questionsS175['ch17/17.5/q-jds-align']} />
      <Ask q={questionsS175['ch17/17.5/q-jds-loop']} />
      <p>
        The book's middle road is the sectioned variant: after sorting, cut the rows into sections and give each section
        its own ELL, padded only to that section's longest row. Rows in a section have similar lengths, so there is little
        padding, and each section stays regular. In the explorer, tick the sort and lower the slice size: the pad slots
        nearly vanish while every slice stays an ELL.
      </p>
      <Source section="17.6" pages="417" />
      <p>
        Both vendor libraries ship this idea in some form, and neither ships JDS itself. NVIDIA's Sliced ELL is the
        sectioned variant without the sort. AMD's LRB algorithm does the sort as bucketing: it bins rows by length.
      </p>
      <Today checked="2026-11-20 (ch17-facts A6, B4)">
        <p>
          Sliced ELL pads with column −1. LRB's bins are powers of two; that is read from AMD's source code.
        </p>
      </Today>

      <h2>When preprocessing pays</h2>
      <p>
        Splitting, sorting and slicing are all work done before the first product, and for a single product they can cost
        more than they save. In an iterative solver the same <code>A</code> comes back every iteration, with only{' '}
        <code>x</code> and <code>y</code> changing, so the cost is spread over all of them.
      </p>
      <Source section="17.5, 17.6" pages="414–415, 417" />
      <p>
        Both vendor libraries build this argument into their APIs: a preprocessing call or stage that runs once per matrix
        and speeds up the later products that share its sparsity pattern.
      </p>
      <Today checked="2026-11-20 (ch17-facts A8)">
        <p>
          cuSPARSE's <code>cusparseSpMV_preprocess</code> "may accelerate subsequent calls"; rocSPARSE's adaptive CSR
          "needs to be amortized over many subsequent sparse vector products".
        </p>
      </Today>
      <Ask q={questionsS175['ch17/17.5/q-amortise']} />
      <p>
        The same argument covers repacking model weights once, when they are loaded, into whatever layout the kernel reads
        fastest, before many forward passes reuse them.
      </p>
      <Beyond>
        <p>The weight-repacking analogy is ours.</p>
      </Beyond>

      <h2>Performance that depends on the data</h2>
      <p>
        Most earlier kernels in the book ran the same way whatever the numbers in their input. These do not. How much
        padding ELL stores, how many lanes idle in CSR and how well JDS's iterations line up all depend on where the
        non-zeros fall. The book calls sparse computation a simple example of data-dependent performance (pp. 421–422).
      </p>
      <Source section="17.8" pages="421–422" />
      <p>
        It adds that sparse code reaches far fewer FLOPS than dense code on CPUs and GPUs; on the same page it notes that
        sparse formats must read explicit indices and are hard to tile.
      </p>
      <Source section="17.8" pages="422" />
      <p>
        Treat the FLOPS claim as the book's: the <a href={href('ch17', 'labs')}>lab</a> measures your own kernels against
        the bytes they move, on rows that are uniform, spread, sorted by length and skewed. The bytes you
        can check here. Set the explorer's values to 4 bits and its indices to 2 bytes: CSR then reads 2.5 bytes per kept
        weight against 0.5 per position for the dense matrix, so it loses once more than a fifth of the weights are kept.
        Try 256 rows with a mean of 64.
      </p>
      <Aside>
        <p>ch17-facts A12: no vendor document read for this course states the FLOPS claim.</p>
      </Aside>

      <h2>Where this meets LLM inference: routing tokens to experts</h2>
      <p>
        A mixture-of-experts layer runs a small router that picks, for every token, a few experts out of many, and each
        expert has its own weights. Read the router's output as (expert, token) pairs and it is a sparse matrix in COO,
        listed in token order. To run each expert as one dense GEMM over its tokens, the pairs must be grouped by expert,
        and that is exercise 3: COO to CSR.
      </p>
      <Source section="17.9, 20.7" pages="422, 510" />
      <p>
        Four passes, each feeding the next. Count the pairs per expert, a histogram as in chapter 9. Scan the counts into
        offsets, chapter 11. Scatter each pair to its expert's offset plus its rank, and run one GEMM per contiguous slice.
        Then scatter the results back through the inverse permutation, which is JDS's row array.
      </p>
      <Beyond>
        <p>
          The mapping of MoE routing onto exercise 3 and JDS is ours; the code it rests on is in ch17-facts D5 to D7.
        </p>
      </Beyond>
      <p>
        The animation runs the four passes on 8 tokens with top-2 routing over 4 experts, one of which no token picks.
      </p>
      <Simplification>
        <p>
          Every pass is computed in one go, with no threads and no timing; the router's weights and the GEMMs' arithmetic
          are left out. Padding each slice to a whole tile of 4 is our illustration: llama.cpp's CUDA path pads the id list
          as a whole, and its Metal kernel pads each slice by one entry.
        </p>
      </Simplification>
      <RouteScene />
      <p>
        On CUDA and HIP, llama.cpp does this in <code>mm_ids_helper</code>, one block per expert, ranking entries with a
        ballot and a popcount, and allocates its id list "padded for unconditional memory access". Its Metal kernel{' '}
        <code>kernel_mul_mm_id_map0</code> counts, takes an exclusive prefix sum, and pads every slice by one entry so that
        every thread makes one store whatever its token matched; the comment says "conditional stores of the divergent
        lanes are very slow". That is llama.cpp's observation from its own code, not Apple documentation. mlx-lm sorts
        tokens by expert with an argsort, keeps the argsort of that as the inverse, and does so only once there are 64 or
        more entries, its own amortisation threshold. MLX itself has no sparse array type; it offers{' '}
        <code>gather_mm</code> with a <code>sorted_indices</code> flag instead.
      </p>
      <Today checked="2026-11-20 (ch17-facts D5, D6, D7)">
        <p>
          llama.cpp <code>ggml-cuda/mmid.cu</code>, <code>mmq.cu</code> and <code>ggml-metal/kernels/mul_mm.metal</code>;
          mlx-lm <code>mlx_lm/layers/moe.py</code>; MLX 0.32.4.
        </p>
      </Today>
      <Ask q={questionsS175['ch17/17.5/q-moe-offset']} />
      <Ask q={questionsS175['ch17/17.5/q-moe-atomic']} />

      <h2>Libraries today</h2>
      <p>
        None of the book's teaching formats survives intact everywhere. This is what each vendor ships for SpMV, and where
        the deterministic choices are.
      </p>
      <Today checked="2026-11-20 (ch17-facts A2, A4, A5, B3 to B5, C4)">
        <p>Versions as the sources reported them: cuSPARSE 13.5, rocSPARSE 5.1.0, hipSPARSE 4.8.0, macOS 27.0 SDK.</p>
      </Today>
      <table className="data wide" data-libraries>
        <thead>
          <tr>
            <th></th>
            <th>NVIDIA: cuSPARSE</th>
            <th>AMD: rocSPARSE, hipSPARSE</th>
            <th>Apple</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>GPU SpMV</td>
            <td>
              <code>cusparseSpMV</code>
            </td>
            <td>
              <code>rocsparse_spmv</code>; <code>hipsparseSpMV</code> as the portable layer
            </td>
            <td>None documented; Accelerate's sparse routines run on the CPU</td>
          </tr>
          <tr>
            <td>Formats</td>
            <td>COO, CSR, CSC, Sliced ELL, BSR, Blocked-ELL</td>
            <td>COO, CSR, CSC, ELL, HYB, Sliced ELL, BSR, Blocked ELL</td>
            <td>MPSGraph creates COO, CSC and CSR tensors, with no documented operation that consumes them</td>
          </tr>
          <tr>
            <td>The book's ELL, hybrid, JDS</td>
            <td>No plain ELL; hybrid removed in CUDA 11.0; no JDS</td>
            <td>ELL and HYB; no JDS</td>
            <td>None</td>
          </tr>
          <tr>
            <td>Deterministic SpMV</td>
            <td>
              <code>CSR_ALG2</code>, <code>COO_ALG2</code>, <code>SELL_ALG1</code>, <code>BSR_ALG1</code>
            </td>
            <td>rowsplit, coo, ell, bsr, sell (rocSPARSE); hipSPARSE documents none</td>
            <td>Not applicable</td>
          </tr>
          <tr>
            <td>Reuse across products</td>
            <td>
              <code>cusparseSpMV_preprocess</code>
            </td>
            <td>a preprocess stage</td>
            <td>Not applicable</td>
          </tr>
        </tbody>
      </table>
      <p>
        Two more facts complete the picture. CUB's <code>DeviceSpmv</code> was removed in CCCL 3.0 with the advice "Use
        cuSPARSE instead". And PyTorch's MPS backend writes its own sparse kernel: each thread finds its row's slice of the
        sorted COO arrays by binary search and loops over it, with no atomics. That is the accessibility problem of page{' '}
        <a href={href('ch17', '17.2')}>17.2</a> solved a third way, row access recovered from sorted COO without building{' '}
        <code>rowPtrs</code>, at the price of a search per thread.
      </p>
      <Today checked="2026-11-20 (ch17-facts A9, D2)">
        <p>
          PyTorch <code>aten/src/ATen/native/sparse/mps/kernels/SparseTensorMath.metal</code>, the{' '}
          <code>spmm_addmm_coo</code> kernels; the reading as a third solution is ours.
        </p>
      </Today>
    </>
  );
}
