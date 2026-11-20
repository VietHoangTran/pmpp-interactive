import { useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  blockDependencies,
  dedupFlags,
  dedupMoves,
  moveByIndex,
  type Dependencies,
  type MatrixPattern,
  type Move,
} from '../../sim/inplace';

// Section 12.8, p. 301. Every destination, flag and dependency shown comes from sim/inplace.ts.

export const questionsS128: Record<string, Question> = {
  'ch12/12.8/q-dedup': {
    kind: 'numeric',
    id: 'ch12/12.8/q-dedup',
    prompt: (
      <>
        Remove the duplicates from the sorted list <code>[3, 3, 5, 7, 7, 7, 9]</code>. How many keys remain?
      </>
    ),
    answer: 4,
    unit: 'keys',
    explain: (
      <p>
        The keep flags are 1 0 1 1 0 0 1: a key is kept when it differs from the key before it, and the first key is
        always kept. Their exclusive scan gives the survivors 3, 5, 7 and 9 the slots 0 to 3. It is a stable filter
        whose test looks at a neighbour.
      </p>
    ),
    source: 'ch. 12 §12.8, p. 301',
  },
  'ch12/12.8/q-noscan': {
    kind: 'choice',
    id: 'ch12/12.8/q-noscan',
    prompt: <>Why does removing columns of a matrix by index need no scan?</>,
    options: [
      { label: 'Because the columns of a matrix are contiguous in memory.', why: 'In a row-major matrix a column is strided: its elements are a whole row apart.' },
      { label: 'It does need a scan, of the keep flags of every element.', why: 'It could use one, but nothing about the data decides who survives, so the count before each key is known from the index.' },
      {
        label: 'Whether a key survives, and how many removed columns come before it, depend only on its column index, so its destination is a closed formula.',
        correct: true,
      },
    ],
    explain: (
      <p>
        A scan exists to count survivors that the data decides. When the index decides, the count of removed columns
        to the left of column <i>c</i> is a constant per column, and the new flat index is{' '}
        <code>r × (C − m) + (c − removed before c)</code>.
      </p>
    ),
    source: 'ch. 12 §12.8, p. 301',
  },
  'ch12/12.8/q-pad': {
    kind: 'choice',
    id: 'ch12/12.8/q-pad',
    prompt: (
      <>
        You pad every row of a row-major matrix from width 6 to width 8, in place, one row per thread block. Which
        ordering between blocks is required?
      </>
    ),
    options: [
      { label: 'The same order as filtering: earlier blocks read before later blocks write.', why: 'That protects keys from writes coming from behind. Here the writes come from in front.' },
      {
        label: 'Each block must finish reading before any earlier block writes, because keys move outwards into later slots: the chain runs from the last block to the first.',
        correct: true,
      },
      { label: 'None: nothing is removed, so nothing is overwritten.', why: 'Row 1 moves from slots 6 to 11 into slots 8 to 13, and slots 12 and 13 still hold row 2, which may not have been read yet.' },
    ],
    explain: (
      <p>
        Row <i>r</i> moves from <code>6r</code> to <code>8r</code>, so it lands on the slots of rows after it. Those
        rows must be read first. The book's one-directional synchronisation still works, run from the block with the
        last keys towards the block with the first.
      </p>
    ),
    source: 'ch. 12 §12.8, p. 301 (with the erratum corrected)',
  },
  'ch12/12.8/q-order-only': {
    kind: 'choice',
    id: 'ch12/12.8/q-order-only',
    prompt: <>You remove rows of a matrix by index, in place, with the single-pass scan's machinery between blocks. What does each block pass on to the next?</>,
    options: [
      { label: 'Only a flag that says "I have finished reading"; no partial sum is needed.', correct: true },
      { label: 'The number of rows it kept, as in the stable filter.', why: 'Every block can compute that from the indices of the removed rows. Nobody needs it passed on.' },
      { label: 'Nothing at all: the destinations are known.', why: 'Known destinations remove the scan, not the hazard. A later block still must not write before an earlier block has read.' },
    ],
    explain: (
      <p>
        The scan of section 12.7 did two jobs at once: it computed offsets and it ordered the blocks. When the index
        gives the offsets, only the ordering job is left, and the flag alone does it, with the same release and acquire
        as before.
      </p>
    ),
    source: 'ch. 12 §12.8, p. 301',
  },
};

// ---------------------------------------------------------------- the explorer

type Kind = 'dedup' | 'removeRows' | 'removeCols' | 'padCols';
const KINDS: { kind: Kind; label: string }[] = [
  { kind: 'dedup', label: 'remove duplicates' },
  { kind: 'removeRows', label: 'remove rows' },
  { kind: 'removeCols', label: 'remove columns' },
  { kind: 'padCols', label: 'pad columns' },
];
const DEFAULT_KEYS = '1, 1, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7';

function parseKeys(s: string): { keys: number[]; error: string | null } {
  const parts = s.split(/[\s,]+/).filter(Boolean);
  const keys = parts.map(Number);
  if (!keys.length) return { keys: [], error: 'Type at least one key.' };
  if (keys.some((k) => !Number.isFinite(k))) return { keys: [], error: 'Keys must be numbers.' };
  if (keys.length > 24) return { keys: [], error: 'At most 24 keys.' };
  if (keys.some((k, i) => i > 0 && k < keys[i - 1])) return { keys: [], error: 'The list must be sorted: each key at least the one before it.' };
  return { keys, error: null };
}

function DirectionPanel({ deps, inPlace, unit }: { deps: Dependencies; inPlace: boolean; unit: string }) {
  if (!inPlace)
    return (
      <p data-part="direction" data-direction="out-of-place">
        Out of place, every write goes to the second array, where no key waits to be read. No order between blocks is
        needed.
      </p>
    );
  const label =
    deps.direction === 'forward'
      ? 'earlier blocks must read before later blocks write: run the chain from the first block to the last'
      : deps.direction === 'backward'
        ? 'later blocks must read before earlier blocks write: run the chain from the last block to the first'
        : deps.direction === 'none'
          ? 'no block writes over another block\'s keys: no chain needed'
          : 'dependencies run both ways';
  return (
    <div data-part="direction" data-direction={deps.direction}>
      <p style={{ margin: '0 0 0.4rem' }}>
        <span className={'chip' + (deps.direction === 'none' ? ' ok' : deps.direction === 'mixed' ? ' fail' : '')}>
          {deps.direction === 'forward' ? 'forward' : deps.direction === 'backward' ? 'backward' : deps.direction}
        </span>{' '}
        {label}.
      </p>
      {deps.edges.length ? (
        <ul style={{ margin: '0 0 0.4rem', paddingLeft: '1.2rem' }} data-part="edges">
          {deps.edges.map((e) => (
            <li key={`${e.reader}-${e.writer}`} data-edge={`${e.reader}>${e.writer}`}>
              block {e.reader} reads before block {e.writer} writes ({e.writes} {unit}
              {e.writes === 1 ? ' lands' : 's land'} in block {e.reader}'s range) {e.reader < e.writer ? '→' : '←'}
            </li>
          ))}
        </ul>
      ) : null}
      <p className="muted" style={{ margin: 0 }}>
        {deps.within} write{deps.within === 1 ? '' : 's'} inside the writer's own block (a barrier orders them)
        {deps.fresh ? `; ${deps.fresh} into new memory past the original end` : ''}.
      </p>
    </div>
  );
}

function RelatedExplorer() {
  const [kind, setKind] = useState<Kind>('removeCols');
  const [rows, setRows] = useState(4);
  const [cols, setCols] = useState(6);
  const [rmRows, setRmRows] = useState<number[]>([1]);
  const [rmCols, setRmCols] = useState<number[]>([2]);
  const [pad, setPad] = useState(2);
  const [rowBlocks, setRowBlocks] = useState(4);
  const [inPlace, setInPlace] = useState(true);
  const [keysText, setKeysText] = useState(DEFAULT_KEYS);
  const [perBlock, setPerBlock] = useState(4);
  const [sel, setSel] = useState<number | null>(null);

  const parsed = parseKeys(keysText);
  const isMatrix = kind !== 'dedup';
  const pattern: MatrixPattern | null = !isMatrix
    ? null
    : kind === 'removeRows'
      ? { kind, rows, cols, remove: rmRows.filter((r) => r < rows) }
      : kind === 'removeCols'
        ? { kind, rows, cols, remove: rmCols.filter((c) => c < cols) }
        : { kind: 'padCols', rows, cols, pad };
  const moved = pattern ? moveByIndex(pattern) : null;
  const rowsPerBlock = Math.ceil(rows / Math.min(rowBlocks, rows));
  const dedup = parsed.error ? null : dedupFlags(parsed.keys);
  const moves: Move[] = isMatrix ? moved!.moves : dedup ? dedupMoves(parsed.keys) : [];
  const srcSize = isMatrix ? rows * cols : parsed.keys.length;
  const blockSize = isMatrix ? rowsPerBlock * cols : perBlock;
  const deps = blockDependencies(moves, srcSize, blockSize);
  const selMove = sel == null ? undefined : moves.find((m) => m.src === sel);

  const toggle = (list: number[], v: number) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v].sort((a, b) => a - b));
  const pick = (k: Kind) => {
    setKind(k);
    setSel(null);
  };

  return (
    <Bench
      title="Which way does the dependency run?"
      task="Predict the direction for each pattern before you switch in-place on. Then click cells to check destinations."
      foot={isMatrix ? 'Row-major storage. Each cell of the source shows the flat index it moves to; click one for the formula.' : 'Each key is kept when it differs from the key before it. The offsets are the exclusive scan of the keep flags.'}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Pattern">
          {KINDS.map((k) => (
            <button key={k.kind} aria-pressed={kind === k.kind} onClick={() => pick(k.kind)}>{k.label}</button>
          ))}
        </span>
        <label className="check"><input type="checkbox" checked={inPlace} onChange={(e) => setInPlace(e.target.checked)} /> in place</label>
      </div>

      {isMatrix ? (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <label className="field">
            Rows
            <select value={rows} onChange={(e) => { setRows(+e.target.value); setSel(null); }} aria-label="rows">
              {[2, 3, 4, 5, 6, 7, 8].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label className="field">
            Columns
            <select value={cols} onChange={(e) => { setCols(+e.target.value); setSel(null); }} aria-label="columns">
              {[2, 3, 4, 5, 6, 7, 8].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label className="field">
            Row blocks
            <select value={Math.min(rowBlocks, rows)} onChange={(e) => setRowBlocks(+e.target.value)} aria-label="row blocks">
              {Array.from({ length: rows }, (_, i) => i + 1).map((v) => (
                <option key={v} value={v}>{v} ({Math.ceil(rows / v)} row{Math.ceil(rows / v) === 1 ? '' : 's'} each)</option>
              ))}
            </select>
          </label>
          {kind === 'padCols' ? (
            <label className="field">
              Padding per row
              <select value={pad} onChange={(e) => setPad(+e.target.value)} aria-label="padding">
                {[1, 2, 3].map((v) => <option key={v} value={v}>{v} column{v === 1 ? '' : 's'}</option>)}
              </select>
            </label>
          ) : null}
          {kind === 'removeRows' ? <span className="muted">Click a row number to remove or keep it.</span> : null}
          {kind === 'removeCols' ? <span className="muted">Click a column number to remove or keep it.</span> : null}
        </div>
      ) : (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <label className="field" style={{ minWidth: 'min(22rem, 100%)' }}>
            Sorted keys
            <input type="text" value={keysText} onChange={(e) => setKeysText(e.target.value)} aria-label="sorted keys" />
          </label>
          <label className="field">
            Keys per block
            <select value={perBlock} onChange={(e) => setPerBlock(+e.target.value)} aria-label="keys per block">
              {[2, 3, 4, 6, 8].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          {parsed.error ? <span className="t-fail" data-part="error">{parsed.error}</span> : null}
        </div>
      )}

      {isMatrix && moved ? (
        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
          <div>
            <div className="mx-label t-global">source, {rows} × {cols}: where each element goes</div>
            <div className="mx global" style={{ gridTemplateColumns: `auto repeat(${cols}, auto)` }} data-part="source">
              <span className="void" />
              {Array.from({ length: cols }, (_, c) =>
                kind === 'removeCols' ? (
                  <span key={'h' + c} className={rmCols.includes(c) ? 'hot' : 'on'} role="button" tabIndex={0} style={{ cursor: 'pointer' }}
                    data-col-head={c} aria-label={`column ${c}`} onClick={() => { setRmCols(toggle(rmCols, c)); setSel(null); }}
                    onKeyDown={(e) => e.key === 'Enter' && setRmCols(toggle(rmCols, c))}>{rmCols.includes(c) ? `c${c} ✕` : `c${c}`}</span>
                ) : (
                  <span key={'h' + c} className="void">c{c}</span>
                ),
              )}
              {Array.from({ length: rows }, (_, r) => [
                kind === 'removeRows' ? (
                  <span key={'r' + r} className={rmRows.includes(r) ? 'hot' : 'on'} role="button" tabIndex={0} style={{ cursor: 'pointer' }}
                    data-row-head={r} aria-label={`row ${r}`} onClick={() => { setRmRows(toggle(rmRows, r)); setSel(null); }}
                    onKeyDown={(e) => e.key === 'Enter' && setRmRows(toggle(rmRows, r))}>{rmRows.includes(r) ? `r${r} ✕` : `r${r}`}<sub>b{Math.floor(r / rowsPerBlock)}</sub></span>
                ) : (
                  <span key={'r' + r} className="void" title={`block ${Math.floor(r / rowsPerBlock)}`}>r{r}<sub>b{Math.floor(r / rowsPerBlock)}</sub></span>
                ),
                ...Array.from({ length: cols }, (_, c) => {
                  const m = moved.moves[r * cols + c];
                  return (
                    <span key={r + '-' + c} data-src={m.src} data-dest={m.dest ?? ''}
                      className={sel === m.src ? 'hot' : m.dest == null ? 'void' : ''}
                      style={{ cursor: 'pointer' }}
                      onClick={() => setSel(m.src)}>
                      {m.dest == null ? '✕' : m.dest}
                    </span>
                  );
                }),
              ])}
            </div>
            <p className="muted" style={{ margin: '0.4rem 0 0', fontSize: '0.8rem' }}>
              Blocks of {rowsPerBlock} row{rowsPerBlock === 1 ? '' : 's'}; the small b on each row is its block.
            </p>
          </div>
          <div>
            <div className="mx-label t-global">result, {moved.newRows} × {moved.newCols}: which source element lands here</div>
            <div className="mx global" style={{ gridTemplateColumns: `repeat(${Math.max(1, moved.newCols)}, auto)` }} data-part="result">
              {Array.from({ length: moved.size }, (_, d) => {
                const m = moved.moves.find((x) => x.dest === d);
                return (
                  <span key={d} className={m && sel === m.src ? 'hot' : m ? 'on' : 'void'} data-slot={d}>
                    {m ? m.src : 'pad'}
                  </span>
                );
              })}
            </div>
          </div>
        </div>
      ) : null}

      {!isMatrix && dedup ? (
        <div style={{ overflowX: 'auto' }} data-part="dedup">
          {(
            [
              ['key', parsed.keys.map(String)],
              ['keep', dedup.keep.map(String)],
              ['offset', dedup.offset.map((o, i) => (dedup.keep[i] ? String(o) : '·'))],
              ['block', parsed.keys.map((_, i) => String(Math.floor(i / perBlock)))],
            ] as [string, string[]][]
          ).map(([name, vals]) => (
            <div key={name} className="row" style={{ gap: '0.5rem', marginBottom: '0.3rem', flexWrap: 'nowrap' }}>
              <span className="mx-label" style={{ width: '3.4rem', margin: 0 }}>{name}</span>
              <div className="mx global" style={{ gridTemplateColumns: `repeat(${vals.length}, auto)` }} data-row={name}>
                {vals.map((v, i) => (
                  <span key={i} className={sel === i ? 'hot' : name !== 'block' && dedup.keep[i] ? 'on' : name === 'block' ? 'void' : ''}
                    onClick={() => setSel(i)} style={{ cursor: 'pointer' }}>{v}</span>
                ))}
              </div>
            </div>
          ))}
          <div className="row" style={{ gap: '0.5rem', flexWrap: 'nowrap' }}>
            <span className="mx-label" style={{ width: '3.4rem', margin: 0 }}>out</span>
            <div className="mx global" style={{ gridTemplateColumns: `repeat(${Math.max(1, dedup.count)}, auto)` }} data-row="out">
              {dedup.out.map((v, i) => <span key={i} className="on">{v}</span>)}
            </div>
            <span className="muted" data-part="count">{dedup.count} kept</span>
          </div>
        </div>
      ) : null}

      <div className="narrate" role="status" style={{ margin: '0.8rem 0' }} data-part="formula">
        {selMove ? (
          isMatrix ? (
            <>
              Element (row {selMove.r}, column {selMove.c}), flat index {selMove.src}:{' '}
              {selMove.dest == null ? <b>{selMove.formula}</b> : <>moves to <b>{selMove.formula}</b>. No scan: the index decides.</>}
            </>
          ) : (
            <>Key {selMove.src}: {selMove.formula}. Here the value decides, so the offsets need a scan.</>
          )
        ) : (
          <>Click a cell to see where it goes and why.</>
        )}
      </div>

      {parsed.error && !isMatrix ? null : <DirectionPanel deps={deps} inPlace={inPlace} unit={isMatrix ? 'element' : 'key'} />}
    </Bench>
  );
}

// ---------------------------------------------------------------- the page

export function S128() {
  const q = questionsS128;
  return (
    <>
      <p>
        The stable filter is one member of a family: patterns in which every key moves in one direction. Removing keys
        moves the survivors towards the front; inserting gaps moves keys towards the back. What you learned for the
        filter carries over: staging through shared memory, coarsening, and, in place, an order between blocks.
      </p>
      <Source section="12.8" pages="301" />

      <h2>Removing duplicates from a sorted list</h2>
      <p>
        In a sorted list, equal keys sit next to each other. Removing duplicates is then a stable filter whose keep
        test is "this key differs from the one before it", with the first key always kept. Everything else is section
        12.5: keep flags, an exclusive scan, a scatter.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md, A8)">
        <p>
          CUB has it as <code>DeviceSelect::Unique</code>: "only the first key from each run is selectively copied".
        </p>
      </Today>
      <Ask q={q['ch12/12.8/q-dedup']} label="Work it out" />

      <h2>Removing rows or columns</h2>
      <p>
        Removing rows or columns of a matrix is different in one way that matters: keys go by <i>index</i>, not by
        value. Whether a key survives, and how many removed keys come before it, follow from its row and column, so
        each thread computes its key's destination with a formula. No scan.
      </p>
      <p>
        In place, the hazard of section <a href={href('ch12', '12.7')}>12.7</a> is still there: a later block's keys
        land in an earlier block's range, so earlier blocks must still read before later blocks write. The book keeps
        the single-pass scan's one-directional synchronisation for that, but with nothing to add up: the blocks pass a
        flag and no partial sum.
      </p>
      <Source section="12.8" pages="301" />
      <Predict
        id="ch12/12.8/p-col"
        prompt={
          <>
            A 4 × 6 row-major matrix loses column 2. Element (row 2, column 4) moves to which flat index?
          </>
        }
      >
        {(v) => (
          <p>
            It moves to <b>13</b>: the new width is 5, column 4 becomes column 3 because one removed column is to its
            left, and 2 × 5 + 3 = 13.{' '}
            {v === 13 ? 'You had it.' : v === 15 ? 'Did you keep the old width? 2 × 6 + 3 = 15 uses it.' : v === 14 ? 'Did you keep the old column? 2 × 5 + 4 = 14 does.' : 'Check the explorer below: select remove columns and click (2, 4).'}
          </p>
        )}
      </Predict>

      <h2>Adding rows or columns</h2>
      <p>
        Adding rows or columns, for example padding each row to a width the memory system prefers, is again decided by
        the index. But the keys now move <i>outwards</i>, towards higher slots. Out of place that changes nothing. In
        place it reverses the requirement: a block's keys land on slots of <i>later</i> blocks, so later blocks must
        finish reading before earlier blocks write, and the chain runs from the block with the last keys to the block
        with the first.
      </p>
      <Aside>
        <p>
          <b>The page says "earlier" twice.</b> It reads "the earlier thread blocks ... write to memory locations
          previously occupied by keys read by earlier thread blocks". The second "earlier" should be "later", as the
          next sentence confirms. The page also has two small typos ("in the section", "data in-place data movement").
        </p>
      </Aside>
      <p>
        The explorer has all four patterns. For each one, decide which way the arrows between blocks will point before
        you switch <i>in place</i> on.
      </p>
      <Simplification>
        <p>
          Blocks own whole rows (or a run of keys) and read all of them before writing any. The explorer shows who must
          wait for whom, not when anything happens.
        </p>
      </Simplification>
      <RelatedExplorer />
      <Ask q={q['ch12/12.8/q-noscan']} />
      <Ask q={q['ch12/12.8/q-order-only']} />
      <Ask q={q['ch12/12.8/q-pad']} />

      <h2>Harder movements</h2>
      <p>
        Some in-place data movements are more complex than these. Transposing a matrix or a tensor in place is the
        book's example, and it only points to the literature for it (reference [2], Sung and others, 2014). It is not
        covered here.
      </p>
      <Source section="12.8" pages="301–302" />
      <p>
        You will meet both related patterns close to a model. Padding a vocabulary dimension up to a multiple of a
        tile width is adding columns. Removing duplicate token ids from a sorted list, so that a repetition penalty
        counts each distinct token once, is the duplicate filter.
      </p>
      <Beyond>
        <p>Both examples are ours. The book's examples are a sorted list and matrix layouts adjusted for alignment.</p>
      </Beyond>
    </>
  );
}
