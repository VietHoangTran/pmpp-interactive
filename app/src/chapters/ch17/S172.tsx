import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { widthChoices } from '../../sim/warps';
import {
  FIG17_1, FIG17_6_ORDER, FORMAT_NAME, ORDER_TERMS, PRESETS, ROW0_PRODUCTS,
  alignedStride, build, cooUpdates, idleLanes, interleavings, longestRow, matrix, matrixRuns, nnz, permutations,
  floatSum, rmwDone, rmwInit, rmwStep, RMW_NEXT, runTrace, shuffledOrder, spmvRef, spmvTrace, storage, summarize,
  toCSC, toCSR, toCOO, toELL, toHybrid, toJDS, toggle, xRuns,
  type ArrayName, type EllLoop, type Entry, type Format, type FormatId, type Lane, type Matrix, type RmwState, type Trace,
} from '../../sim/sparse';
import { EllScene } from './EllScene';

// ---------------------------------------------------------------- the code shown

/** The snippets this page shows, loaded at run time; snippets-s172.test.ts pins the lines below in them. */
export const SNIPPETS = {
  coo: 'ch17/snippets/spmv_coo.cu',
  csr: 'ch17/snippets/spmv_csr.cu',
  ell: 'ch17/snippets/spmv_ell.cu',
  csc: 'ch17/snippets/spmv_csc.cu',
} as const;
export const COO_ATOMIC_LINES = [
  'cuda::atomic_ref<float, cuda::thread_scope_device> y_ref(y[row]);',
  'y_ref.fetch_add(x[col]*value, cuda::memory_order_relaxed);',
] as const;
export const CSR_LOOP_LINE = 'for (unsigned int i = csrMatrix.rowPtrs[row]; i < csrMatrix.rowPtrs[row + 1]; ++i) {';
export const ELL_INDEX_LINE = 'unsigned int i = t*ellMatrix.stride + row;';
export const ELL_ACCUMULATE_LINE = 'y[row] += sum;';
export const ELL_PAD_TEST_LINE = 'if (col >= 0) sum += x[col]*ellMatrix.value[i];';
export const CSC_LOAD_LINE = 'float inValue = x[col];';

const STATUS = 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware';
const SPMV_TIERS: Record<string, Tier> = { x: 'global', y: 'global', sum: 'reg', inValue: 'reg', row: 'reg', col: 'reg', i: 'reg', t: 'reg' };

/** The file without its leading comment block (the caption carries what it says). */
export function body(src: string): string {
  const lines = src.split('\n');
  let k = 0;
  while (k < lines.length && (lines[k].startsWith('//') || lines[k].trim() === '')) k++;
  return lines.slice(k).join('\n');
}
/** 1-based numbers of the lines that contain any of the needles. */
const linesWith = (text: string, needles: readonly string[]): number[] =>
  text.split('\n').flatMap((l, i) => (needles.some((n) => l.includes(n)) ? [i + 1] : []));

function Snippet({ file, caption, marks }: { file: string; caption: ReactNode; marks: readonly string[] }) {
  const src = useLabFile(file);
  if (!src) return null;
  const code = body(src);
  return (
    <Code
      tabs={[{ label: file.split('/').pop()!, code, tiers: SPMV_TIERS, status: STATUS, highlight: linesWith(code, marks) }]}
      caption={caption}
    />
  );
}

// ---------------------------------------------------------------- explorer state

type CooOrder = 'row' | 'fig17_6' | 'shuffled';
interface ExState {
  m: Matrix;
  /** A preset's id, or 'edited'. */
  preset: string;
  tab: FormatId;
  step: number;
  K: number;
  ellLoop: EllLoop;
  aligned: boolean;
  order: CooOrder;
  seed: number;
  group: 'four' | 'rows';
}
const INITIAL: ExState = {
  m: FIG17_1, preset: 'fig17_1', tab: 'coo', step: 0, K: 2, ellLoop: 'nnzPerRow', aligned: false, order: 'row', seed: 1, group: 'four',
};
const EMPTY8 = { id: 'empty8', label: 'Empty 8 × 8', source: 'start from nothing', m: matrix(8, 8, []) };
const TABS: FormatId[] = ['coo', 'csr', 'ell', 'hyb', 'jds', 'csc'];
const kOf = (s: ExState) => Math.max(1, Math.min(s.K, Math.max(1, longestRow(s.m))));
const widthOf = (s: ExState) => (s.group === 'four' ? 4 : Math.max(1, s.m.rows));

function formatOf(s: ExState): Format {
  const order =
    s.tab !== 'coo' ? undefined
      : s.order === 'fig17_6' && s.preset === 'fig17_1' ? FIG17_6_ORDER
        : s.order === 'shuffled' ? shuffledOrder(nnz(s.m), s.seed) : undefined;
  return build(s.tab, s.m, { K: kOf(s), stride: s.aligned ? alignedStride(s.m.rows, 8) : s.m.rows, order });
}

/** The non-zero (or pad) at an index of the array a step reads. */
function entryAt(f: Format, array: ArrayName, i: number): Entry | null {
  if (f.kind === 'hyb') return array === 'ell' ? f.ell.at[i] : f.coo.at[i];
  return f.at[i] ?? null;
}

type Hover = { kind: 'entry'; r: number; c: number } | { kind: 'row'; r: number } | { kind: 'col'; c: number } | null;
interface ArrCell { text: string; entry: Entry | null; pad?: boolean; row?: number; col?: number; end?: boolean }
interface ArrRow { name: string; array: ArrayName | null; cells: ArrCell[]; part?: string; stride?: number }

const entryCells = (at: readonly Entry[], text: (e: Entry) => string): ArrCell[] => at.map((e) => ({ text: text(e), entry: e }));
const ptrCells = (ptrs: readonly number[], kind: 'row' | 'col'): ArrCell[] =>
  ptrs.map((p, k) => ({ text: String(p), entry: null, end: k === ptrs.length - 1, ...(k < ptrs.length - 1 ? { [kind]: k } : {}) }));
function ellRows(e: Extract<Format, { kind: 'ell' }>, part = ''): ArrRow[] {
  const sfx = part ? ` (${part})` : '';
  return [
    { name: 'nnzPerRow' + sfx, array: null, cells: e.nnzPerRow.map((n, r) => ({ text: String(n), entry: null, row: r })) },
    { name: 'colIdx' + sfx, array: 'ell', stride: e.stride, cells: e.at.map((x, i) => ({ text: x ? String(x.c) : String(e.colIdx[i]), entry: x, pad: !x })) },
    { name: 'value' + sfx, array: 'ell', stride: e.stride, cells: e.at.map((x) => ({ text: x ? x.label : '0', entry: x, pad: !x })) },
  ];
}
function arraysOf(f: Format): ArrRow[] {
  switch (f.kind) {
    case 'coo':
      return [
        { name: 'rowIdx', array: 'coo', cells: entryCells(f.at, (e) => String(e.r)) },
        { name: 'colIdx', array: 'coo', cells: entryCells(f.at, (e) => String(e.c)) },
        { name: 'value', array: 'coo', cells: entryCells(f.at, (e) => e.label) },
      ];
    case 'csr':
      return [
        { name: 'rowPtrs', array: null, cells: ptrCells(f.rowPtrs, 'row') },
        { name: 'colIdx', array: 'csr', cells: entryCells(f.at, (e) => String(e.c)) },
        { name: 'value', array: 'csr', cells: entryCells(f.at, (e) => e.label) },
      ];
    case 'ell':
      return ellRows(f);
    case 'hyb':
      return [
        ...ellRows(f.ell, 'ELL part'),
        { name: 'rowIdx (COO part)', array: 'coo', cells: entryCells(f.coo.at, (e) => String(e.r)) },
        { name: 'colIdx (COO part)', array: 'coo', cells: entryCells(f.coo.at, (e) => String(e.c)) },
        { name: 'value (COO part)', array: 'coo', cells: entryCells(f.coo.at, (e) => e.label) },
      ];
    case 'jds':
      return [
        { name: 'row', array: null, cells: f.row.map((r) => ({ text: String(r), entry: null, row: r })) },
        { name: 'iterPtr', array: null, cells: f.iterPtr.map((p, k) => ({ text: String(p), entry: null, end: k === f.iterPtr.length - 1 })) },
        { name: 'colIdx', array: 'jds', cells: entryCells(f.at, (e) => String(e.c)) },
        { name: 'value', array: 'jds', cells: entryCells(f.at, (e) => e.label) },
      ];
    case 'csc':
      return [
        { name: 'colPtrs', array: null, cells: ptrCells(f.colPtrs, 'col') },
        { name: 'rowIdx', array: 'csc', cells: entryCells(f.at, (e) => String(e.r)) },
        { name: 'value', array: 'csc', cells: entryCells(f.at, (e) => e.label) },
      ];
  }
}

const touches = (h: Hover, e: Entry | null, row?: number, col?: number): boolean => {
  if (!h) return false;
  if (h.kind === 'entry') return !!e && e.r === h.r && e.c === h.c;
  if (h.kind === 'row') return e ? e.r === h.r : row === h.r;
  return e ? e.c === h.c : col === h.c;
};

const MAPPING_TEXT: Record<FormatId, string> = {
  coo: 'one thread per non-zero: thread i takes the triple at position i',
  csr: 'one thread per row: thread r walks rowPtrs[r] up to rowPtrs[r + 1]',
  ell: 'one thread per row: thread r reads slot t·stride + r in iteration t',
  hyb: 'the ELL part, one thread per row, then the COO part, one thread per spilled non-zero',
  jds: 'one thread per sorted row: thread p reads slot iterPtr[t] + p in iteration t',
  csc: 'one thread per column: thread c loads x[c] once, then walks colPtrs[c] up to colPtrs[c + 1]',
};

const LABEL_CELL: CSSProperties = { border: 'none', background: 'transparent', color: 'var(--ink-3)', minWidth: '1.4rem' };
const NEUTRAL: CSSProperties = { border: '1px solid var(--rule)' };
/** The cell under the pointer, and its partner in the other view. */
const HOVER_RING: CSSProperties = { boxShadow: '0 0 0 2px var(--ink)', zIndex: 1 };

// ---------------------------------------------------------------- explorer

function MatrixGrid({ m, hover, setHover, hot, onToggle }: {
  m: Matrix; hover: Hover; setHover: (h: Hover) => void; hot: Set<Entry>; onToggle: (r: number, c: number) => void;
}) {
  const byPos = new Map(m.entries.map((e) => [`${e.r},${e.c}`, e]));
  return (
    <div>
      <div className="mx-label t-global">A, {m.rows} × {m.cols}, {nnz(m)} non-zeros</div>
      <div className="mx global" data-matrix="" style={{ gridTemplateColumns: `auto repeat(${m.cols}, auto)` }}>
        <span style={LABEL_CELL} />
        {Array.from({ length: m.cols }, (_, c) => (
          <span key={'c' + c} style={LABEL_CELL} onMouseEnter={() => setHover({ kind: 'col', c })} onMouseLeave={() => setHover(null)}>{c}</span>
        ))}
        {Array.from({ length: m.rows }, (_, r) => [
          <span key={'r' + r} style={LABEL_CELL} onMouseEnter={() => setHover({ kind: 'row', r })} onMouseLeave={() => setHover(null)}>{r}</span>,
          ...Array.from({ length: m.cols }, (_, c) => {
            const e = byPos.get(`${r},${c}`) ?? null;
            const lit = !!e && touches(hover, e, r, c);
            const cls = e && hot.has(e) ? 'hot' : lit ? 'on' : '';
            return (
              <span
                key={`${r},${c}`} data-r={r} data-c={c} role="button" tabIndex={0} className={cls}
                aria-label={e ? `row ${r}, column ${c}: ${e.label}; click to remove` : `row ${r}, column ${c}: zero; click to add a non-zero`}
                style={{ cursor: 'pointer', ...(lit ? HOVER_RING : {}) }}
                onClick={() => onToggle(r, c)}
                onKeyDown={(ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); onToggle(r, c); } }}
                onMouseEnter={() => setHover(e ? { kind: 'entry', r, c } : null)}
                onMouseLeave={() => setHover(null)}
              >
                {e ? e.label : ''}
              </span>
            );
          }),
        ])}
      </div>
      <p className="muted" style={{ margin: '0.4rem 0 0', maxWidth: '18rem' }}>
        Click a cell to add or remove a non-zero. Hover a row or column number to light up all of it.
      </p>
    </div>
  );
}

function Arrays({ rows, hover, setHover, reading }: {
  rows: ArrRow[]; hover: Hover; setHover: (h: Hover) => void; reading: { array: ArrayName; lanes: Map<number, Lane> } | null;
}) {
  return (
    <div>
      {rows.map((a) => (
        <div key={a.name} style={{ marginBottom: '0.55rem' }}>
          <div className="mx-label t-global">{a.name}</div>
          <div className="mx global" data-array={a.name} style={{ display: 'flex', flexWrap: 'wrap' }}>
            {a.cells.map((c, i) => {
              const lane = reading && a.array === reading.array ? reading.lanes.get(i) : undefined;
              const on = touches(hover, c.entry, c.row, c.col);
              const cls = c.pad || c.end ? 'void' : lane ? 'hot' : on ? 'on' : '';
              const style: CSSProperties = on ? { ...HOVER_RING } : {};
              if (a.stride && i > 0 && i % a.stride === 0) style.marginLeft = '0.45rem';
              if (lane && c.pad) style.outline = '2px solid var(--global)';
              return (
                <span
                  key={i} data-i={i} className={cls} style={style}
                  onMouseEnter={() => setHover(c.entry ? { kind: 'entry', r: c.entry.r, c: c.entry.c } : c.row != null ? { kind: 'row', r: c.row } : c.col != null ? { kind: 'col', c: c.col } : null)}
                  onMouseLeave={() => setHover(null)}
                >
                  {c.text}
                  <sub style={lane && !c.pad ? { color: 'inherit', opacity: 0.8 } : undefined}>{i}</sub>
                </span>
              );
            })}
            {a.cells.length === 0 ? <span className="void" style={{ minWidth: '5rem' }}>empty</span> : null}
          </div>
        </div>
      ))}
    </div>
  );
}

function threadName(f: Format, step: { array: ArrayName }, lane: Lane): string {
  if (step.array === 'coo') return `thread ${lane.thread}`;
  if (f.kind === 'jds') return `thread ${lane.thread} (row ${f.row[lane.thread]})`;
  if (f.kind === 'csc') return `thread ${lane.thread} (column ${lane.thread})`;
  return `thread ${lane.thread} (row ${lane.thread})`;
}

function Explorer({ st, set, setTab }: { st: ExState; set: (p: Partial<ExState>) => void; setTab: (t: FormatId) => void }) {
  const [hover, setHover] = useState<Hover>(null);
  const f = useMemo(() => formatOf(st), [st]);
  const trace: Trace = useMemo(() => spmvTrace(f, { ellLoop: st.ellLoop }), [f, st.ellLoop]);
  const width = widthOf(st);
  const sum = useMemo(() => summarize(trace, width, st.m.rows), [trace, width, st.m.rows]);
  const store = storage(f, st.m);
  const steps = trace.steps;
  const si = Math.min(st.step, Math.max(0, steps.length - 1));
  const cur = steps[si];
  const launch = cur ? trace.launches[cur.launch] : trace.launches[0];
  const lanesByIndex = new Map((cur?.lanes ?? []).filter((l) => l.index != null).map((l) => [l.index!, l]));
  const hot = new Set((cur?.lanes ?? []).flatMap((l) => (l.index != null ? [entryAt(f, cur!.array, l.index)].filter((e): e is Entry => !!e) : [])));
  const ones = Array(st.m.cols).fill(1);
  const zeros = Array(st.m.rows).fill(0);
  const y = runTrace(trace, ones, zeros);
  const ref = spmvRef(st.m, ones, zeros);
  const runsNow = cur ? matrixRuns(cur, width) : [];
  const xNow = cur ? xRuns(cur, width) : [];
  const idleNow = cur ? idleLanes(cur) : 0;
  const z = nnz(st.m);
  const K = longestRow(st.m);
  const showGroupChoice = st.m.rows !== 4;

  const loadPreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id) ?? EMPTY8;
    set({ m: p.m, preset: p.id, step: 0, order: st.order === 'fig17_6' && p.id !== 'fig17_1' ? 'row' : st.order });
  };
  const runsText = (rs: number[]) => {
    const n = (v: number) => `${v} run${v === 1 ? '' : 's'}`;
    if (rs.length <= 1) return n(rs[0] ?? 0);
    return rs.every((v) => v === rs[0])
      ? `${n(rs[0])} in each of ${rs.length} groups`
      : rs.map((v, g) => `group ${g}: ${v ? n(v) : 'nothing'}`).join(', ');
  };

  return (
    <Bench
      title="Format explorer"
      task="Pick a format, step its kernel, and edit the matrix. Hover a cell to find it in the arrays."
      foot={
        <>
          Every array and count comes from <code>src/sim/sparse.ts</code>, which its tests tie to the book's figures. x is 1
          everywhere and y starts at 0, so y ends as the row sums.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Matrices">
          {[...PRESETS, EMPTY8].map((p) => (
            <button key={p.id} data-preset={p.id} aria-pressed={st.preset === p.id} onClick={() => loadPreset(p.id)} title={p.source}>
              {p.label}
            </button>
          ))}
        </span>
        {st.preset === 'edited' ? <span className="muted">edited</span> : null}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'max-content minmax(0, 1fr)' }}>
        <MatrixGrid
          m={st.m} hover={hover} setHover={setHover} hot={hot}
          onToggle={(r, c) => set({ m: toggle(st.m, r, c), preset: 'edited', step: 0, order: st.order === 'fig17_6' ? 'row' : st.order })}
        />
        <div>
          <span className="seg" role="group" aria-label="Formats" style={{ flexWrap: 'wrap', rowGap: 4, marginBottom: '0.7rem' }}>
            {TABS.map((t) => (
              <button key={t} data-tab={t} aria-pressed={st.tab === t} onClick={() => setTab(t)}>{FORMAT_NAME[t]}</button>
            ))}
          </span>
          {st.tab === 'hyb' || st.tab === 'jds' ? (
            <p className="muted" style={{ margin: '0 0 0.6rem' }}>
              Page 17.5 explains this format. It is here so that every format can be compared on the same matrix.
            </p>
          ) : null}
          <Arrays rows={arraysOf(f)} hover={hover} setHover={setHover} reading={cur ? { array: cur.array, lanes: lanesByIndex } : null} />
          <p style={{ margin: '0.6rem 0 0' }} data-storage="">
            Stored: <b data-k="total">{store.total}</b> numbers ({store.values} values, {store.indices} indices, {store.pointers} pointers and counts),
            {store.pads ? <> including <b data-k="pads">{store.pads}</b> pad slots in <code>value</code> and as many in the column indices.</> : <> with <b data-k="pads">no</b> pads.</>}{' '}
            The dense matrix is <span data-k="dense">{store.dense}</span>.
            <br />
            <span className="muted">
              Exercise 2's count: <code>{store.formula}</code> with m = {st.m.rows}, n = {st.m.cols}, z = {z},{' '}
              {f.kind === 'hyb'
                ? `K = ${f.ell.width} (the ELL part's width; the longest row has ${K}), s = ${f.coo.value.length} spilled`
                : `K = ${K} (the longest row)`}
              {f.kind === 'ell' && f.stride !== f.numRows ? `, stride = ${f.stride}` : ''}.
            </span>
          </p>
        </div>
      </div>

      <div className="panel" style={{ marginTop: '1rem' }} data-kernel={st.tab}>
        <p style={{ margin: '0 0 0.6rem' }}>
          <b>SpMV/{FORMAT_NAME[st.tab].replace('Hybrid ELL-COO', 'hybrid')}</b>: {MAPPING_TEXT[st.tab]}.
        </p>
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          <button onClick={() => set({ step: 0 })} disabled={si === 0}>Restart</button>
          <button onClick={() => set({ step: si - 1 })} disabled={si === 0}>Back</button>
          <button className="primary" onClick={() => set({ step: si + 1 })} disabled={si >= steps.length - 1}>Step</button>
          <span className="muted" aria-live="polite" data-step={si}>
            {steps.length ? <>step {si + 1} of {steps.length}: {launch.name}, iteration t = {cur.t}</> : 'no non-zeros: nothing to run'}
          </span>
          {showGroupChoice ? (
            <label className="field">
              Threads per group
              <select value={st.group} onChange={(e) => set({ group: e.target.value as ExState['group'] })} aria-label="Threads per group">
                <option value="four">4, as in the book's figures</option>
                <option value="rows">{st.m.rows}, the row count</option>
              </select>
            </label>
          ) : null}
        </div>
        {st.tab === 'coo' ? (
          <div className="row" style={{ marginBottom: '0.7rem' }}>
            <span className="seg" role="group" aria-label="COO order">
              <button aria-pressed={st.order === 'row'} onClick={() => set({ order: 'row', step: 0 })}>row order (Fig. 17.3)</button>
              <button aria-pressed={st.order === 'fig17_6'} disabled={st.preset !== 'fig17_1'} onClick={() => set({ order: 'fig17_6', step: 0 })}>Fig. 17.6's order</button>
              <button aria-pressed={st.order === 'shuffled'} onClick={() => set({ order: 'shuffled', seed: st.order === 'shuffled' ? st.seed + 1 : st.seed, step: 0 })}>
                {st.order === 'shuffled' ? 'shuffle again' : 'shuffled'}
              </button>
            </span>
          </div>
        ) : null}
        {st.tab === 'ell' || st.tab === 'hyb' ? (
          <div className="row" style={{ marginBottom: '0.7rem' }}>
            <span className="seg" role="group" aria-label="ELL loop bound">
              <button aria-pressed={st.ellLoop === 'nnzPerRow'} onClick={() => set({ ellLoop: 'nnzPerRow', step: 0 })}>loop to nnzPerRow (the book)</button>
              <button aria-pressed={st.ellLoop === 'full'} onClick={() => set({ ellLoop: 'full', step: 0 })}>full width, test col &gt;= 0</button>
            </span>
            {st.tab === 'ell' ? (
              <span className="seg" role="group" aria-label="ELL stride">
                <button aria-pressed={!st.aligned} onClick={() => set({ aligned: false, step: 0 })}>stride = numRows</button>
                <button aria-pressed={st.aligned} onClick={() => set({ aligned: true, step: 0 })}>stride a multiple of 8</button>
              </span>
            ) : (
              <label className="field">
                K, the ELL part's width: {kOf(st)}
                <input type="range" min={1} max={Math.max(1, K)} value={kOf(st)} aria-label="K"
                  onChange={(e) => set({ K: Number(e.target.value), step: 0 })} />
              </label>
            )}
          </div>
        ) : null}

        {cur ? (
          <table className="threads" aria-label="Threads in this step" style={{ marginBottom: '0.7rem' }}>
            <thead>
              <tr><th>thread</th><th>reads</th><th>x</th><th>its y</th><th>update of y</th></tr>
            </thead>
            <tbody>
              {cur.lanes.map((l, k) => (
                <tr key={k} data-thread={l.thread} data-read={l.index ?? '-'} data-pad={l.pad ? 1 : 0}
                  style={k > 0 && k % width === 0 ? { borderTop: '2px solid var(--rule)' } : undefined}>
                  <td>{threadName(f, cur, l)}</td>
                  <td className={l.index != null ? 't-global' : 'muted'}>
                    {l.index == null ? 'idle' : `${cur.array === 'coo' && f.kind === 'hyb' ? 'COO part ' : ''}slot ${l.index}${l.pad ? ' (pad: colIdx only)' : ''}`}
                  </td>
                  <td>{l.x != null ? `x[${l.x}]` : <span className="muted">–</span>}</td>
                  <td>{l.y != null ? `y[${l.y}]` : <span className="muted">–</span>}</td>
                  <td style={{ whiteSpace: 'normal' }}>
                    {l.atomic ? <span className="chip global">atomic add</span>
                      : l.index != null || (l.y != null && !launch.atomic) ? <span className="muted">own sum, added once at the end</span> : <span className="muted">–</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}

        <div className="row" style={{ gap: '0.4rem 0.6rem' }}>
          <span className="chip" style={NEUTRAL} data-k="idle-now">{idleNow} of {cur?.lanes.length ?? 0} threads idle in this step</span>
          <span className="chip global" data-k="runs-now">matrix reads in this step: {runsText(runsNow)}</span>
          <span className="chip global" data-k="x-runs-now">x reads in this step: {runsText(xNow)}</span>
          <span className="chip global" data-k="atomics">{sum.atomicUpdates} atomic updates of y</span>
          <span className="chip" style={NEUTRAL} data-k="shared-y">
            {sum.sharedY} y element{sum.sharedY === 1 ? '' : 's'} updated by more than one thread
          </span>
        </div>
        <p style={{ margin: '0.7rem 0 0' }} data-k="divergence">
          Whole kernel, groups of {width}: the groups run <b data-k="group-iters">{sum.groupIterations}</b> iterations in all. Of the lane-iterations,{' '}
          <b data-k="useful">{sum.useful}</b> multiply a non-zero, <b data-k="waiting">{sum.idle}</b> wait for a longer row in the group
          {sum.pad ? <>, and <b data-k="pad-work">{sum.pad}</b> read a pad</> : <span data-k="pad-work" data-v="0" />}.
        </p>
        <p style={{ margin: '0.4rem 0 0' }} data-k="y">
          y when the kernel ends: [{y.join(', ')}]{' '}
          {y.every((v, i) => v === ref[i]) ? <span className="chip ok">the row sums of A</span> : <span className="chip fail">not A·x</span>}
        </p>
      </div>
    </Bench>
  );
}

function WidthsNote() {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports as Reports), [reports]);
  return <>{choices.map((c) => c.label).join('; ')}</>;
}

// ---------------------------------------------------------------- two threads, one y[0]

const ALL_ORDERS = permutations(3);
const showTerm = (v: number) => (v < 0 ? '−' : '') + Math.abs(v).toLocaleString('en-US');

function RaceStepper() {
  const [atomic, setAtomic] = useState(false);
  const [s, setS] = useState<RmwState>(() => rmwInit());
  const [found, setFound] = useState<number[]>([]);
  const [showAll, setShowAll] = useState(false);
  const [picked, setPicked] = useState<number[]>([]);
  const [seed, setSeed] = useState(5);
  const done = rmwDone(s);
  const tally = useMemo(() => {
    const t = new Map<number, number>();
    for (const sch of interleavings([3, 3])) {
      const v = cooUpdates(sch).y;
      t.set(v, (t.get(v) ?? 0) + 1);
    }
    return [...t.entries()].sort((a, b) => a[0] - b[0]);
  }, []);
  const orderTally = useMemo(() => {
    const t = new Map<number, number>();
    for (const o of ALL_ORDERS) t.set(floatSum(ORDER_TERMS, o), (t.get(floatSum(ORDER_TERMS, o)) ?? 0) + 1);
    return [...t.entries()].sort((a, b) => a[0] - b[0]);
  }, []);
  const want = ROW0_PRODUCTS[0] + ROW0_PRODUCTS[1];

  const stepThread = (t: number) => {
    const n = rmwStep(s, t, ROW0_PRODUCTS, atomic);
    setS(n);
    if (rmwDone(n) && !atomic && !found.includes(n.y)) setFound([...found, n.y].sort((a, b) => a - b));
  };
  const reset = (a = atomic) => { setAtomic(a); setS(rmwInit()); };
  const bothLoad = () => {
    setAtomic(false);
    const n = cooUpdates([0, 1, 0, 1, 0, 1]);
    setS(n);
    if (!found.includes(n.y)) setFound([...found, n.y].sort((a, b) => a - b));
  };
  const sums = picked.map((_, k) => floatSum(ORDER_TERMS, picked.slice(0, k + 1)));

  return (
    <Bench
      title="Two threads, one y[0]"
      task="Interleave the two threads so that one update is lost. Then switch the atomic on and try again."
      foot="Row 0 of Fig. 17.1 holds 1 and 7; with x all ones, the two COO threads of row 0 add 1 and 7 into y[0], which starts at 0. The second panel's three products are ours, chosen to show rounding."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="check"><input type="checkbox" checked={atomic} onChange={(e) => reset(e.target.checked)} /> atomic add</label>
        <button onClick={() => reset()}>Reset</button>
        <button onClick={bothLoad}>Both load, then both store</button>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(14rem, 1fr))' }}>
        {[0, 1].map((t) => (
          <div key={t} className="panel" data-rmw-thread={t}>
            <p style={{ margin: '0 0 0.4rem' }}>
              <b>Thread {t}</b>: adds {ROW0_PRODUCTS[t]} (value {ROW0_PRODUCTS[t]} × x[{t}])
            </p>
            <p style={{ margin: '0 0 0.5rem' }} className="muted">
              next: {atomic && s.pc[t] < 3 ? 'atomic add (load, add, store as one step)' : RMW_NEXT[Math.min(3, s.pc[t])]}; register:{' '}
              <span className="t-reg" data-k={'reg' + t}>{s.reg[t] ?? '–'}</span>
            </p>
            <button className="primary" onClick={() => stepThread(t)} disabled={s.pc[t] >= 3}>Step thread {t}</button>
          </div>
        ))}
        <div className="panel">
          <p style={{ margin: 0 }}>
            <span className="t-global">y[0] in memory</span>: <b data-k="y0">{s.y}</b>
          </p>
          {done ? (
            <p style={{ margin: '0.5rem 0 0' }} data-k="rmw-result">
              {s.y === want ? <span className="chip ok">{s.y}: both updates landed</span>
                : <span className="chip fail">{s.y} instead of {want}: thread {s.y === ROW0_PRODUCTS[0] ? 1 : 0}'s update was lost</span>}
            </p>
          ) : null}
        </div>
      </div>
      <ol className="muted" style={{ margin: '0.7rem 0', paddingLeft: '1.4rem', minHeight: '1.4em' }} data-k="rmw-log">
        {s.events.map((e, k) => (
          <li key={k}>thread {e.thread} {e.op}: register {e.reg}, y[0] = {e.y}</li>
        ))}
      </ol>
      <div className="row">
        <span>Results you have found without the atomic:</span>
        {found.length ? found.map((v) => <span key={v} className={'chip ' + (v === want ? 'ok' : 'fail')} data-found={v}>{v}</span>) : <span className="muted">none yet</span>}
        <button onClick={() => setShowAll(true)} disabled={showAll}>Show every interleaving</button>
      </div>
      {showAll ? (
        <p data-k="all-interleavings">
          The {tally.reduce((a, [, n]) => a + n, 0)} ways to interleave two threads of three steps each end at{' '}
          {tally.map(([v, n]) => `${v} (${n} of them)`).join(', ')}. With the atomic, each thread has one indivisible step, and both
          orders give {want}.
        </p>
      ) : null}

      <hr style={{ border: 0, borderTop: '1px dashed var(--rule)', margin: '1rem 0' }} />
      <p style={{ marginTop: 0 }}>
        <b>The atomic is on. Now the order.</b> Row 1 of Fig. 17.1 has three non-zeros, so under COO three threads add into y[1].
        Suppose their products are {ORDER_TERMS.map(showTerm).join(', ')}. Pick the order in which their atomic adds land.
      </p>
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        {ORDER_TERMS.map((v, k) => (
          <button key={k} data-term={k} onClick={() => setPicked([...picked, k])} disabled={picked.includes(k)}>{showTerm(v)}</button>
        ))}
        <button onClick={() => { setPicked(shuffledOrder(3, seed)); setSeed(seed + 1); }}>Shuffle</button>
        <button onClick={() => setPicked([])} disabled={!picked.length}>Clear</button>
      </div>
      <p style={{ margin: 0 }} data-k="float-sum">
        float32 running sum: 0
        {picked.map((k, i) => <span key={i}> + {showTerm(ORDER_TERMS[k])} → <b>{showTerm(sums[i])}</b></span>)}
        {picked.length === 3 ? (
          <> {' '}<span className={'chip ' + (sums[2] === 1 ? 'ok' : 'fail')} data-k="float-result">{showTerm(sums[2])}; the exact sum is 1</span></>
        ) : null}
      </p>
      {picked.length === 3 ? (
        <p className="muted" style={{ margin: '0.5rem 0 0' }} data-k="float-all">
          Over all {ALL_ORDERS.length} orders: {orderTally.map(([v, n]) => `${showTerm(v)} in ${n}`).join(', ')}. Float32 values near
          100,000,000 are 8 apart, so adding 1 to one changes nothing; the 1 survives only when the two large terms cancel first. The atomic made every
          add count; it did not choose the order. The "Today" note beside the COO section says how the libraries offer the choice.
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- comparison

const BOOK_VERDICT: Record<string, { flex: string; access: string }> = {
  coo: { flex: 'best: append, or reorder the three arrays together (p. 406)', access: 'non-zero to its row and column (p. 406)' },
  csr: { flex: 'poor: insert, shift later rows, bump later rowPtrs (p. 409)', access: 'row to its non-zeros (p. 409)' },
  ell: { flex: 'better than CSR: a pad slot can take a non-zero (p. 413)', access: 'both: row to non-zeros, slot i to row i % stride (p. 413)' },
  hyb: { flex: 'better than ELL: an overflow goes to the COO part (p. 415)', access: 'weaker than ELL for rows that spilled (p. 415)' },
  jds: { flex: 'worst: a new non-zero may force a re-sort (p. 417)', access: 'row to its non-zeros only (p. 418)' },
  csc: { flex: 'as CSR (p. 420)', access: 'column to its non-zeros (p. 420)' },
};

export function scoreRows(m: Matrix, K: number) {
  const list: { key: string; book: string; name: string; f: Format; loop: EllLoop }[] = [
    { key: 'coo', book: 'coo', name: 'COO', f: toCOO(m), loop: 'nnzPerRow' },
    { key: 'csr', book: 'csr', name: 'CSR', f: toCSR(m), loop: 'nnzPerRow' },
    { key: 'ell', book: 'ell', name: 'ELL, loop to nnzPerRow', f: toELL(m), loop: 'nnzPerRow' },
    { key: 'ell-full', book: 'ell', name: 'ELL, full width', f: toELL(m), loop: 'full' },
    { key: 'hyb', book: 'hyb', name: `Hybrid, K = ${K}`, f: toHybrid(m, K), loop: 'nnzPerRow' },
    { key: 'jds', book: 'jds', name: 'JDS', f: toJDS(m), loop: 'nnzPerRow' },
    { key: 'csc', book: 'csc', name: 'CSC', f: toCSC(m), loop: 'nnzPerRow' },
  ];
  return list.map((x) => {
    const t = spmvTrace(x.f, { ellLoop: x.loop });
    const s = summarize(t, 4, m.rows);
    const st = storage(x.f, m);
    const worst = Math.max(0, ...t.steps.flatMap((step) => matrixRuns(step, 4)));
    return { ...x, mapping: [...new Set(t.launches.map((l) => l.mapping))].join(', then '), storage: st, summary: s, worst };
  });
}

function ScoreTable({ st, set }: { st: ExState; set: (p: Partial<ExState>) => void }) {
  const K = kOf(st);
  const rows = useMemo(() => scoreRows(st.m, K), [st.m, K]);
  return (
    <Bench
      title="All six formats on one matrix"
      task="Load Fig. 17.13 and compare it with Fig. 17.1. Which columns change, and which formats keep their verdict?"
      foot={
        <>
          Groups of 4 threads, as in the book's figures. "Worst runs" is the most separate runs of consecutive addresses one group's
          matrix reads form in any one iteration: 1 is contiguous. The last two columns are the book's verdicts. The matrix and K are the
          explorer's, at the top of the page.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Matrices for the table">
          {PRESETS.map((p) => (
            <button key={p.id} aria-pressed={st.preset === p.id} onClick={() => set({ m: p.m, preset: p.id, step: 0, order: 'row' })}>{p.label}</button>
          ))}
        </span>
        {st.preset === 'edited' ? <span className="muted">your edited matrix</span> : null}
      </div>
      <table className="data" data-score="" style={{ background: 'var(--page)', margin: 0 }}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>Format</th>
            <th>One thread per</th>
            <th className="num">Stored</th>
            <th className="num">Pads</th>
            <th className="num">Atomic adds</th>
            <th className="num">Worst runs</th>
            <th className="num">Lanes waiting</th>
            <th className="num">Lanes on pads</th>
            <th>Flexibility (book)</th>
            <th>Access (book)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} data-format={r.key}>
              <td style={{ paddingLeft: '0.6rem' }}><b>{r.name}</b></td>
              <td>{r.mapping.replace(/^per /, '').replace(', then per ', ', then ')}</td>
              <td className="num" data-col="stored">{r.storage.total} <span className="muted">({Math.round((100 * r.storage.total) / Math.max(1, r.storage.dense))}%)</span></td>
              <td className="num" data-col="pads">{r.storage.pads}</td>
              <td className="num" data-col="atomics">{r.summary.atomicUpdates}{r.summary.sharedY ? <span className="muted"> ({r.summary.sharedY} y shared)</span> : null}</td>
              <td className="num" data-col="worst">{r.worst}</td>
              <td className="num" data-col="waiting">{r.summary.idle}</td>
              <td className="num" data-col="padwork">{r.summary.pad}</td>
              <td className="muted">{BOOK_VERDICT[r.book].flex}</td>
              <td className="muted">{BOOK_VERDICT[r.book].access}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

const csrFig = spmvTrace(toCSR(FIG17_1));
const ell1 = summarize(spmvTrace(toELL(FIG17_1)), 4, 4);
const ellFull1 = summarize(spmvTrace(toELL(FIG17_1), { ellLoop: 'full' }), 4, 4);

export const questionsS172: Record<string, Question> = {
  'ch17/17.2/q-coo-y1': {
    kind: 'numeric',
    id: 'ch17/17.2/q-coo-y1',
    prompt: (
      <>
        SpMV/COO on Fig. 17.1, with the triples in Fig. 17.3's order. How many threads add into <code>y[1]</code>?
      </>
    ),
    answer: 3,
    explain: (
      <p>
        Row 1 holds 5, 3 and 9, at positions 2, 3 and 4 of the COO arrays. Each position is one thread, and all three add into{' '}
        <code>y[1]</code>, so the add must be atomic. Under CSR, thread 1 owns row 1: it sums the three products in a register and
        adds the sum to <code>y[1]</code> once, with no atomic. The explorer's chips show the difference: 8 atomic adds and 3 shared
        y elements under COO, none under CSR.
      </p>
    ),
    source: 'ch. 17 §17.2, pp. 404–405; §17.3, pp. 408–409',
  },
  'ch17/17.2/q-ell-index': {
    kind: 'numeric',
    id: 'ch17/17.2/q-ell-index',
    prompt: (
      <>
        Store Fig. 17.1 as ELL with stride 8 instead of 4: four rows of pads are appended so that every iteration starts on a
        32-byte boundary. What is the flat index of row 1's element <code>t = 2</code>, the 9?
      </>
    ),
    answer: 17,
    explain: (
      <p>
        <code>i = t·stride + row = 2·8 + 1 = 17</code>. With the book's stride of 4 it would be 9. Rows 4 to 7 hold only pads, and no
        thread works on them: the bounds check is <code>row &lt; numRows</code>, not <code>row &lt; stride</code>. In the explorer's
        ELL tab, choose "stride a multiple of 8" and find the 9 in slot 17.
      </p>
    ),
    source: 'ch. 17 §17.4, pp. 411–413; the 32-byte unit from notes/today/ch17-facts.md A3',
  },
  'ch17/17.2/q-ell-assign': {
    kind: 'choice',
    id: 'ch17/17.2/q-ell-assign',
    prompt: (
      <>
        Fig. 17.12 ends with <code>y[row] = sum;</code>. You build a hybrid ELL-COO SpMV and launch the COO part first, then the ELL
        kernel exactly as printed. What happens?
      </>
    ),
    options: [
      { label: 'It is correct either way: the COO part only holds rows the ELL part does not touch.', why: 'Every row has an ELL part, even if it is all pads, and the ELL kernel writes y[row] for every row.' },
      { label: <>Every row loses what was in <code>y</code> before the ELL kernel ran: its COO contribution and the starting <code>y</code>. The fix is <code>y[row] += sum</code>, as in Fig. 17.9 and in every library.</>, correct: true },
      { label: 'Only the rows longer than K come out wrong.', why: 'Those lose their COO tails, but every row also loses the y it started with: SpMV is y = A·x + y.' },
      { label: 'The two kernels race on y.', why: 'Nothing here needs two threads at once: the ELL kernel simply writes over what the COO part left.' },
    ],
    explain: (
      <p>
        The emulator run of this page's snippets shows it on Fig. 17.1 with K = 2: COO part first, then the page's line, 4 of 4 rows
        wrong; with <code>+=</code>, none. Running the COO part after the ELL kernel hides the problem only when <code>y</code> starts
        at zero. With an accumulating ELL kernel the two parts compose in either order.
      </p>
    ),
    source: 'notes/ch17.md "Unclear" item 8; notes/today/ch17-facts.md A10; labs/ch17/snippets/check/spmv_check.cu on the emulator',
  },
  'ch17/17.2/q-pad-marker': {
    kind: 'choice',
    id: 'ch17/17.2/q-pad-marker',
    prompt: (
      <>
        An ELL kernel loops over the full padded width, and the pad slots hold column 0 and value 0 (so a <code>col &gt;= 0</code>{' '}
        test lets them through). <code>x[0]</code> is +Inf. What comes out?
      </>
    ),
    options: [
      { label: 'NaN in every row that has a pad, because 0 × Inf is NaN.', correct: true },
      { label: 'The correct result: the pads are zero, so they add nothing.', why: 'Zero times a finite number is zero; zero times infinity is NaN in IEEE arithmetic.' },
      { label: 'Only row 0 is wrong, the row that really has column 0.', why: 'Every pad slot points at column 0, whatever row it belongs to.' },
      { label: 'Inf in every row.', why: 'A non-zero times Inf is Inf, but a pad gives 0 × Inf = NaN, and NaN wins any sum it enters.' },
    ],
    explain: (
      <p>
        On Fig. 17.1, rows 0, 2 and 3 have pads and come out NaN; row 1 has none. With column −1 in the pads and the test, no pad
        touches <code>x</code>: the libraries' convention. cuSPARSE describes the same split between dense and sparse code: a dense
        dot product of [0, 1, 0] and [1, 1, NaN] is NaN, a sparse one is 1.
      </p>
    ),
    source: 'notes/today/ch17-facts.md A4, A6; the model (src/sim/sparse.ts) and the emulator run of the snippets',
  },
  'ch17/17.2/q-ell-loop': {
    kind: 'choice',
    id: 'ch17/17.2/q-ell-loop',
    prompt: (
      <>
        ELL can stop at <code>nnzPerRow[row]</code> (the book's kernel) or loop over the full width and skip the pads. Which
        statement holds?
      </>
    ),
    options: [
      { label: 'Stopping at nnzPerRow keeps CSR\'s divergence and does no pad work; the full width removes trip-count divergence, but every thread runs as many iterations as the longest row.', correct: true },
      { label: 'The full width is always faster.', why: 'It trades waiting lanes for pad work, and one long row makes every thread pay for that row.' },
      { label: 'ELL removes divergence either way.', why: 'With nnzPerRow, lanes stop at their row\'s length exactly as in CSR (pp. 413–414).' },
      { label: 'Stopping at nnzPerRow breaks coalescing.', why: 'The active lanes of an iteration still read neighbouring slots; an idle lane reads nothing.' },
    ],
    explain: (
      <p>
        On Fig. 17.1 in one group of 4, both versions run {ell1.groupIterations} iterations and do {ell1.useful} useful
        lane-iterations. Stopping at nnzPerRow, {ell1.idle} lane-iterations wait; looping the full width, {ellFull1.idle} wait and{' '}
        {ellFull1.pad} read a pad instead. The lanes are busy either way in this model, which has no timing: what the full width buys
        is a loop with no data-dependent trip count, at the price of reading <code>colIdx</code> at every pad.
      </p>
    ),
    source: 'ch. 17 §17.4, pp. 411, 413–414',
  },
  'ch17/17.2/q-csc-use': {
    kind: 'choice',
    id: 'ch17/17.2/q-csc-use',
    prompt: <>Which job is CSC the right layout for?</>,
    options: [
      { label: <>Multiplying a vector by the matrix, <code>xᵀ·A</code>, where each output is a column's dot product; and a sparse <code>x</code>, where every column whose <code>x</code> entry is zero can be skipped whole.</>, correct: true },
      { label: 'Plain SpMV, because it reads x coalesced and only once.', why: 'That is its only virtue for SpMV. It still needs atomics on y, and its matrix reads are scattered and divergent (pp. 420–421).' },
      { label: 'Matrices that change often.', why: 'Adding a non-zero shifts every later column and bumps every later colPtrs entry, as in CSR (p. 420).' },
      { label: 'The smallest storage.', why: 'It takes 2z + n + 1 numbers: the same as CSR when there are about as many columns as rows.' },
    ],
    explain: (
      <p>
        The book's two uses (p. 421): a vector–matrix product, where grouping by column is exactly the work of one output, and
        sparse-matrix–sparse-vector products, where a column whose <code>x</code> entry is zero contributes nothing and its non-zeros
        sit together, so skipping it is cheap.
      </p>
    ),
    source: 'ch. 17 §17.7, pp. 419–421',
  },
  'ch17/17.2/q-ex1-pads': {
    kind: 'numeric',
    id: 'ch17/17.2/q-ex1-pads',
    prompt: (
      <>
        Exercise 1's matrix, rows <code>[1,0,7,0]</code>, <code>[0,0,8,0]</code>, <code>[0,4,3,0]</code>, <code>[2,0,0,1]</code>, in
        ELL: how many pad slots does <code>value</code> hold?
      </>
    ),
    answer: 1,
    explain: (
      <p>
        The longest row has 2 non-zeros, so ELL holds 4 × 2 = 8 slots for 7 non-zeros, and row 1 gets the one pad. Load the
        "Exercise 1" matrix in the explorer to see the exercise's other formats; its CSR tab shows <code>rowPtrs = [0, 2, 3, 5, 7]</code>.
      </p>
    ),
    source: 'ch. 17 §17.9, exercise 1, p. 422 (answer ours: notes/ch17.md, exercises)',
  },
  'ch17/17.2/q-ex2-missing': {
    kind: 'choice',
    id: 'ch17/17.2/q-ex2-missing',
    prompt: (
      <>
        Exercise 2: an integer matrix has m rows, n columns and z non-zeros. Which formats' integer counts cannot be computed from m,
        n and z alone?
      </>
    ),
    options: [
      { label: 'ELL and JDS: both need K, the length of the longest row.', correct: true },
      { label: 'COO and CSR.', why: 'COO is 3z and CSR is 2z + m + 1.' },
      { label: 'All four, because each needs n.', why: 'None of COO, CSR, ELL or JDS stores anything per column, so n appears in none of the counts.' },
      { label: 'Only ELL.', why: 'JDS stores iterPtr, one entry per iteration plus one: K + 1 entries.' },
    ],
    explain: (
      <p>
        COO 3z; CSR 2z + m + 1; ELL 2·m·K, plus m if <code>nnzPerRow</code> is stored as Fig. 17.12 needs; JDS 2z + m + K + 1 (the{' '}
        <code>row</code> array and <code>iterPtr</code>). Only a column format such as CSC would need n. The explorer prints each
        format's count with the matrix's own m, n, z and K.
      </p>
    ),
    source: 'ch. 17 §17.9, exercise 2, p. 422 (answer ours: notes/ch17.md, exercises)',
  },
};

// ---------------------------------------------------------------- the page

function CsrReveal({ v }: { v: number }) {
  const t1 = csrFig.steps[1].lanes;
  return (
    <>
      <p>
        {v === t1[2].index ? 'Right. ' : `The answer is ${t1[2].index}. `}Row 2 starts at <code>rowPtrs[2] = {toCSR(FIG17_1).rowPtrs[2]}</code>,
        so its element 1 is at {t1[2].index}. In iteration 1 the four threads read{' '}
        {t1.map((l) => (l.index == null ? `nothing (thread ${l.thread}'s row has one element)` : `value[${l.index}]`)).join(', ')}.
      </p>
      <p>
        Step the explorer's CSR tab to iteration 1 to see it, then compare the "matrix reads" chip with the ELL tab's at the same step.
      </p>
    </>
  );
}

function BackendsTable() {
  return (
    <table className="data wide" style={{ maxWidth: '58rem' }}>
      <thead>
        <tr><th>For these kernels</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>One thread per row (CSR, ELL)</td>
          <td><code>blockIdx.x*blockDim.x + threadIdx.x</code> and a bounds check</td>
          <td>the same source</td>
          <td><code>[[thread_position_in_grid]]</code>; with <code>dispatchThreads</code> no thread falls outside the grid</td>
        </tr>
        <tr>
          <td>Float atomic add (COO, CSC)</td>
          <td><code>cuda::atomic_ref&lt;float, cuda::thread_scope_device&gt;</code>, <code>fetch_add(v, cuda::memory_order_relaxed)</code>; or the legacy <code>atomicAdd</code></td>
          <td>the same spelling through libhipcxx (ROCm 10.1 and later); or <code>atomicAdd</code>, done safely, possibly as a compare-and-swap loop, unless built with <code>-munsafe-fp-atomics</code></td>
          <td><code>atomic_fetch_add_explicit</code> on a <code>device atomic_float*</code> buffer, <code>memory_order_relaxed</code> only; Metal 3 and Apple7 or later</td>
        </tr>
        <tr>
          <td>Order of float atomic adds</td>
          <td>undefined</td>
          <td>undefined</td>
          <td>not documented</td>
        </tr>
        <tr>
          <td>Unit that coalescing is counted in</td>
          <td>32-byte sectors</td>
          <td>64- or 128-byte cache lines</td>
          <td>not documented</td>
        </tr>
        <tr>
          <td>Divergence (CSR's uneven loops)</td>
          <td>documented: masked passes per warp</td>
          <td>documented: the EXEC mask, per wavefront</td>
          <td>not documented</td>
        </tr>
        <tr>
          <td>The book's struct of device pointers as an argument</td>
          <td>by value; all parameters together at most 32,764 bytes</td>
          <td>the same syntax; no size limit found</td>
          <td>an argument buffer (Metal 2 and later)</td>
        </tr>
      </tbody>
    </table>
  );
}

export function S172() {
  const [st, setSt] = useState<ExState>(INITIAL);
  const set = (p: Partial<ExState>) => setSt((s) => ({ ...s, ...p }));
  const setTab = (tab: FormatId) => set({ tab, step: 0 });
  /** Open a tab of the explorer at a step and bring it into view. */
  const show = (tab: FormatId, step = 0, extra: Partial<ExState> = {}) => {
    set({ tab, step, ...extra });
    document.querySelector('section.bench[aria-label="Format explorer"]')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
  const Open = ({ tab, step = 0, extra, children }: { tab: FormatId; step?: number; extra?: Partial<ExState>; children: ReactNode }) => (
    <button className="quiet" data-open={`${tab}:${step}`} onClick={() => show(tab, step, extra)}>{children}</button>
  );

  return (
    <>
      <p>
        This chapter asks one question of six storage formats. When most of A is zeros, how should <code>y = A·x + y</code> be
        stored, and what does each layout make the kernel do? The format decides which thread owns which piece of work. Three
        things follow from that, and you can read all three off the arrays without running anything: whether two threads update the
        same <code>y</code> element, whether neighbouring threads read neighbouring addresses, and whether they finish together.
        This page builds four formats by hand and gives each its kernel. Page 17.5 adds the two that buy ELL's regularity with less
        padding.
      </p>
      <Source section="17.1" pages="401–404" />

      <h2>The running example</h2>
      <p>
        The book works one 4 × 4 matrix through every format (Fig. 17.1): 8 non-zeros in rows of 2, 3, 2 and 1. Each row is one
        equation that involves only a few of the unknowns. It is small enough that every array fits on a line and uneven enough to
        show every problem the chapter is about.
      </p>
      <Source section="17.1" pages="402" figs="Fig. 17.1" />
      <p>
        The explorer below is this page's instrument, and each section after it gives you a task in one of its tabs. Click a cell to
        add or remove a non-zero, or load another of the book's matrices. Hover any cell, in the matrix or in an array, to find it in
        the other. Step the kernel to see, thread by thread, which slot each one reads and where its result goes. The hybrid and JDS
        tabs belong to page 17.5; they are here so that the comparison at the end of this page covers all six.
      </p>
      <Simplification>
        <p>
          Threads run in groups of 4 (or one group per row), as the book's figures draw them. Real groups are wider; the widths this
          course knows of: <WidthsNote />.
        </p>
        <p>
          On a matrix this small, chapter 6's sector count would put four 4-byte reads in one 32-byte segment even when they are
          scattered (CSR's slots 0, 2, 5 and 7 all fall in bytes 0 to 31), so the explorer counts runs of consecutive addresses
          instead: one run is the book's picture of a coalesced read. No caches, no timing, no cost for an atomic.
        </p>
      </Simplification>
      <Explorer st={st} set={set} setTab={setTab} />

      <h2>COO: one thread per non-zero</h2>
      <p>
        The coordinate format stores each non-zero as a triple, spread over three arrays of equal length: <code>rowIdx</code>,{' '}
        <code>colIdx</code> and <code>value</code>. The natural kernel gives each thread one triple. It multiplies its value by{' '}
        <code>x[col]</code> and adds the product into <code>y[row]</code>.
      </p>
      <Source section="17.2" pages="404–405" figs="Figs. 17.3–17.5" />
      <p>
        All of a row's non-zeros target the same <code>y</code> element, so several threads add into one place: the add has to be
        atomic. That is COO's one cost. Everything else about it is good. Consecutive threads read consecutive elements of all three
        arrays, so the matrix loads coalesce. Every thread does exactly one multiply, so no thread waits for another, except at the
        bounds check. <Open tab="coo">Open the COO tab</Open>
      </p>
      <Source section="17.2" pages="405–406" figs="Fig. 17.4" />
      <Ask q={questionsS172['ch17/17.2/q-coo-y1']} label="Work it out" />
      <Snippet
        file={SNIPPETS.coo}
        marks={COO_ATOMIC_LINES}
        caption={
          <>
            Fig. 17.5 from <code>labs/ch17/snippets/spmv_coo.cu</code>. It adds the <code>#include &lt;cuda/atomic&gt;</code> the page
            leaves out and the <code>COOMatrix</code> struct the book never defines. The highlighted lines are the atomic add, with
            device scope and relaxed order. Run on the course's emulator in three thread orders; the race emulator flags a copy without
            the atomic.
          </>
        }
      />
      <p>
        COO is also the most flexible format. The triples can sit in any order, as long as the three arrays move together: in the
        explorer's COO tab, switch to <Open tab="coo" step={0} extra={{ order: 'fig17_6', m: FIG17_1, preset: 'fig17_1' }}>Fig. 17.6's order</Open>{' '}
        and compare <code>y</code> with the row order. Adding a non-zero is an append. That is why matrices are often assembled
        in COO and converted later. What COO cannot answer cheaply is "which non-zeros are in row r?".
      </p>
      <Source section="17.2" pages="405–406" figs="Fig. 17.6" />
      <p>
        The book names two costs of the atomic. One is throughput: threads queue on one address. The other is order. Atomic adds into
        one element happen one at a time, in an undefined order, and float addition is not associative, so the low bits of{' '}
        <code>y</code> can change from one run to the next. The stepper below makes both concrete.
      </p>
      <Source section="17.2" pages="406–407" />
      <p>
        The vendor libraries turn that second cost into a switch: a faster algorithm whose result may differ in the last bits from
        run to run, or a deterministic one that is slower. Which one runs is chosen in the call, not by the format.
      </p>
      <Today checked="2026-11-20 (notes/today/ch17-facts.md, A2, A7, B5)">
        <p>
          cuSPARSE: <code>COO_ALG1</code> "may produce slightly different results during different runs"; <code>COO_ALG2</code> is
          bit-wise deterministic and slower (this table, not the general reproducibility section, is the specific statement).
          rocSPARSE: <code>coo_atomic</code> or a deterministic <code>coo</code>; hipSPARSE maps both COO choices to the
          deterministic one on AMD. Library COO is assumed sorted by row; the book's kernel is not.
        </p>
      </Today>
      <RaceStepper />

      <h2>CSR: one thread per row</h2>
      <p>
        Group the non-zeros by row, and the row array can shrink to one offset per row. CSR keeps <code>colIdx</code> and{' '}
        <code>value</code> in row order and replaces <code>rowIdx</code> (z entries) with <code>rowPtrs</code> (m + 1 entries): row r
        is the slice from <code>rowPtrs[r]</code> up to <code>rowPtrs[r+1]</code>, and the extra last entry ends the last row.
      </p>
      <Source section="17.3" pages="407–408" figs="Fig. 17.7" />
      <p>
        Now one thread can own a row. It sums the row's products in a register and adds the sum to <code>y[row]</code> once. No two
        threads write the same element, so the atomic is gone.
      </p>
      <Source section="17.3" pages="408–409" figs="Figs. 17.8–17.9" />
      <Snippet
        file={SNIPPETS.csr}
        marks={[CSR_LOOP_LINE, 'y[row] += sum;']}
        caption={
          <>
            Fig. 17.9 from <code>labs/ch17/snippets/spmv_csr.cu</code>, with the <code>CSRMatrix</code> struct the book never defines.
            The loop bounds lean on the end marker; the last line accumulates. The same source builds for HIP.
          </>
        }
      />
      <p>
        The cost moves to memory and to time. In iteration t of the loop, thread r reads slot <code>rowPtrs[r] + t</code>. In
        iteration 0, threads 0 to 3 read <code>value[0]</code>, <code>value[2]</code>, <code>value[5]</code> and{' '}
        <code>value[7]</code>: four separate places, not one run. And the loop runs as many times as the row is long, so in a group
        of threads the short rows finish and wait while the long ones go on. The book expects that in most or all warps of a real
        matrix.
      </p>
      <Source section="17.3" pages="409–410" />
      <Predict
        id="ch17/17.2/p-csr-t1"
        prompt={
          <>
            In iteration 1 of SpMV/CSR on Fig. 17.1, which index of <code>value</code> does thread 2 read? Commit before you step the
            explorer's CSR tab.
          </>
        }
        unit="(an index)"
      >
        {(v) => <CsrReveal v={v} />}
      </Predict>
      <p>
        <Open tab="csr" step={1}>Show CSR at iteration 1</Open> CSR is smaller than COO and needs no atomics. It is less flexible
        (inserting a non-zero shifts every later row and bumps every later <code>rowPtrs</code> entry) and gives no access by column.
        Its two problems, scattered reads and uneven trip counts, are what the next formats work on.
      </p>
      <Source section="17.3" pages="409–410" />

      <h2>ELL: pad, then turn</h2>
      <p>
        ELL (named after the ELLPACK package) fixes the reads in two moves. Pad every row at its end to the length of the longest row,
        so that the non-zeros form a rectangle. Then store the rectangle column by column: every row's element 0, then every row's
        element 1, and so on. Element t of row r lands at <code>t·stride + row</code>, and <code>rowPtrs</code> is no longer needed.
        The book writes <code>numRows</code> where this course writes <code>stride</code>; the two are equal unless whole rows of
        pads are added for alignment, as below.
      </p>
      <Source section="17.4" pages="410–411" figs="Fig. 17.10" />
      <EllScene />
      <p>
        The kernel is CSR's with the index changed. Because <code>i</code> is a function of <code>row</code>, and <code>row</code> of{' '}
        <code>threadIdx.x</code>, consecutive threads touch consecutive slots in every iteration. (Page 413 says this index "was
        calculated in Fig. 17.9"; it is in Fig. 17.12.) <Open tab="ell">Open the ELL tab</Open>
      </p>
      <Source section="17.4" pages="411–413" figs="Figs. 17.11–17.12" />
      <Ask q={questionsS172['ch17/17.2/q-ell-index']} label="Work it out" />
      <Snippet
        file={SNIPPETS.ell}
        marks={[ELL_INDEX_LINE, ELL_ACCUMULATE_LINE, ELL_PAD_TEST_LINE]}
        caption={
          <>
            Fig. 17.12 from <code>labs/ch17/snippets/spmv_ell.cu</code>, then the same kernel looping the full width. It differs from
            the page in three ways: <code>y[row] += sum</code> where the page assigns; pads hold column −1 and value 0, so{' '}
            <code>colIdx</code> and <code>col</code> are signed; and <code>stride</code> for <code>numRows</code>. Run on the
            emulator, including with <code>stride</code> rounded up to a multiple of 8.
          </>
        }
      />
      <p>
        Two of those differences fix real problems. Fig. 17.12 ends with <code>{'y[row] = sum'}</code> where Fig. 17.9 has{' '}
        <code>+=</code>; SpMV accumulates, and with the page's line whatever was in <code>y</code> is lost. And the book says a
        kernel without <code>nnzPerRow</code> can loop over the full padded width, because pad values are zero, but never says what a
        pad's column holds, though that loop reads it (p. 411). The snippet stores −1 and tests it.
      </p>
      <Today checked="2026-11-20 (notes/today/ch17-facts.md, A6, A10)">
        <p>
          Both conventions are the libraries'. NVIDIA's Sliced ELL and AMD's ELL store column −1 in a pad, and AMD's reference loop
          tests the index before reading <code>x</code>. Every library SpMV computes y = α·A·x + β·y, an accumulate.
        </p>
      </Today>
      <Ask q={questionsS172['ch17/17.2/q-ell-assign']} />
      <Ask q={questionsS172['ch17/17.2/q-pad-marker']} />
      <p>
        So a pad that points at column 0 is harmless only while <code>x</code> stays finite: one infinite <code>x[0]</code> turns every
        padded row into NaN, which is the difference cuSPARSE describes between dense and sparse arithmetic. Now the loop bound. In
        the explorer's ELL tab, step through the three iterations with the loop stopping at <code>nnzPerRow</code>, then switch
        to the full width (<Open tab="ell" extra={{ ellLoop: 'full' }}>full width</Open>) and step again. The reads stay one run per
        iteration either way. What changes is what the short rows' lanes do: wait, or read pads. ELL fixed coalescing, not
        divergence.
      </p>
      <Source section="17.4" pages="411, 413–414" />
      <Ask q={questionsS172['ch17/17.2/q-ell-loop']} />
      <p>
        ELL gains more than coalescing. A new non-zero can take a pad slot without moving anything, so it is more flexible than CSR.
        It gives access both ways: row r's elements are at r, r + stride and so on, and slot i belongs to row{' '}
        <code>i % stride</code>. The price is space, and the longest row sets it alone. Add a few non-zeros to one row of the
        explorer's matrix and watch the pad count. The book's example: a 1000 × 1000 matrix with 1% non-zeros takes about 2% of its
        dense size as CSR; if one row has 200 non-zeros, ELL pads every row to 200 and takes 40%. Page 17.5 is about that problem.
      </p>
      <Source section="17.4" pages="412–413" />
      <p>
        The book adds an alignment trick (p. 413): append a few rows of pads before turning the rectangle, so that every iteration
        starts on an aligned address. In this course's terms, make <code>stride</code> larger than <code>numRows</code>; the ELL tab's stride
        switch does it (<Open tab="ell" extra={{ aligned: true }}>stride a multiple of 8</Open>). Fig. 17.1 then takes 24 slots instead
        of 12: iteration t starts at slot 8t, and rows 4 to 7 hold only pads, which no thread visits because the bounds check is still{' '}
        <code>row &lt; numRows</code>. The pad rows are the price; what they buy is that every iteration's first read begins a new
        aligned segment instead of straddling one.
      </p>
      <Today checked="2026-11-20 (notes/today/ch17-facts.md, A3)">
        <p>
          The book's example unit, "such as 64-bytes", is in no NVIDIA document read today. On every device current CUDA supports
          the unit is a 32-byte sector, so every ELL iteration starts aligned when <code>stride</code> is a multiple of 8 floats;
          AMD's 64- or 128-byte lines need 16 or 32. Apple documents no unit.
        </p>
      </Today>
      <p>
        If you have used a block-sparse attention mask, you have used this layout. Such a mask over query and key blocks is mostly
        empty, and the program that handles one row of query blocks has to know which key blocks to visit. Store, for every row, a
        count and a fixed-width list of block indices, padded where the row is short, and each program finds its row's work at a
        fixed offset with no search. That is <code>nnzPerRow</code> plus an ELL row, for the same reason.
      </p>
      <Beyond title="Not in the book: sparse attention masks">
        <p>
          PyTorch's FlexAttention <code>BlockMask</code> is <code>kv_num_blocks</code> plus a padded{' '}
          <code>kv_indices[ROWS, MAX]</code>, row-major because one program walks one row; its backward pass generates a transposed
          copy, <code>q_indices</code> (p. 409's column-oriented copy). llama.cpp's CUDA and HIP sparse flash attention builds
          fixed-width index lists padded with −1, plus counts. Read 2026-11-20 in <code>torch/nn/attention/flex_attention.py</code> and{' '}
          <code>ggml/src/ggml-cuda/fattn.cu</code> (facts D3, D4); the mapping to the book's formats is ours.
        </p>
      </Beyond>

      <h2>CSC: the same idea, by column</h2>
      <p>
        CSC is CSR with rows and columns swapped: <code>value</code> grouped by column, <code>rowIdx</code> giving each value's row,
        and <code>colPtrs</code> with one entry per column plus the end marker.
      </p>
      <Source section="17.7" pages="418–419" figs="Fig. 17.16" />
      <p>
        The book shows its SpMV kernel for completeness, and it teaches by being bad. One thread per column loads{' '}
        <code>x[col]</code> once, then walks its column and adds each product into <code>y[row]</code>. Different columns hit the
        same rows, so the add is atomic again. Threads 0 to 3 read <code>value[0]</code>, <code>[2]</code>, <code>[4]</code>,{' '}
        <code>[6]</code>, then <code>[1]</code>, <code>[3]</code>, <code>[5]</code>, <code>[7]</code>: scattered, as in CSR. Trip
        counts follow column lengths, so it diverges. It combines the worst of COO and of CSR.{' '}
        <Open tab="csc">Open the CSC tab</Open>
      </p>
      <Source section="17.7" pages="419–421" figs="Figs. 17.17–17.18" />
      <Snippet
        file={SNIPPETS.csc}
        marks={[CSC_LOAD_LINE, 'y_ref.fetch_add(inValue*value, cuda::memory_order_relaxed);']}
        caption={
          <>
            Fig. 17.18 from <code>labs/ch17/snippets/spmv_csc.cu</code>. The page's listing reads <code>rowIdxs</code> and{' '}
            <code>values</code> where the text and Fig. 17.16 say <code>rowIdx</code> and <code>value</code>, and its caption calls it
            an SpMV/CSR kernel; this copy uses the singular names, and adds the include and the struct.
          </>
        }
      />
      <p>
        It has one virtue. Adjacent threads read adjacent elements of <code>x</code>, each exactly once, where every other kernel here
        reads <code>x</code> wherever the non-zeros' columns point. In the CSC tab, the "x reads" chip shows one run at t = 0 and
        nothing afterwards.
      </p>
      <Source section="17.7" pages="421" />
      <p>
        CSC fits wherever a whole column is the unit of work: <code>xᵀ·A</code>, where each output is a column's dot product, and a
        sparse <code>x</code>, where every column whose <code>x</code> entry is zero can be skipped whole. It is also the "second,
        column-oriented copy" p. 409 suggests keeping beside CSR when a computation needs both directions.
      </p>
      <Source section="17.3, 17.7" pages="409, 421" />
      <Ask q={questionsS172['ch17/17.2/q-csc-use']} />

      <h2>The same kernels on other backends</h2>
      <p>
        The per-row kernels need nothing beyond a thread index, so they port everywhere. The atomic kernels need a float atomic add,
        and the table says what each backend calls it and what it promises.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. Everything here about HIP and Metal is from notes/today/ch17-facts.md, "Where the backends really
          differ", checked 2026-11-20. Nothing on this page has run on a GPU.
        </p>
      </Beyond>
      <BackendsTable />
      <p>
        The lab binds one buffer per array and passes sizes explicitly, which maps one to one onto Metal's buffer bindings and leaves
        the struct question aside. One consequence on Metal: a COO kernel's <code>y</code> is declared <code>device atomic_float*</code>,
        so it is not the same plain <code>float</code> buffer type the CSR kernel writes, though the host allocates plain floats either
        way.
      </p>

      <h2>Which format, then</h2>
      <p>
        The table below is the chapter's comparison, computed for whatever matrix the explorer holds. Read it with Fig. 17.1 loaded,
        then with Fig. 17.13, whose two long rows are page 17.5's starting point. No row wins every column. COO buys flexibility and
        balance with atomics. CSR buys space and ownership of <code>y</code> with scattered, uneven loops. ELL buys contiguous reads
        with pads, and keeps CSR's uneven loops unless it pays for the full width.
      </p>
      <Source section="17.2–17.7" pages="405–421" />
      <ScoreTable st={st} set={set} />
      <p>
        Which one is fastest depends on how the non-zeros are spread, which the book calls data-dependent performance, and sparse code
        reaches far fewer FLOPS than dense code, on CPUs and GPUs alike. The <a href={href('ch17', 'labs')}>lab</a> (17.1, CSR and ELL SpMV
        on skewed rows) measures CSR against ELL on your device. Make its prediction before you run it.
      </p>
      <Source section="17.8" pages="421–422" />
      <Ask q={questionsS172['ch17/17.2/q-ex1-pads']} label="Work it out" />
      <Ask q={questionsS172['ch17/17.2/q-ex2-missing']} />
    </>
  );
}
