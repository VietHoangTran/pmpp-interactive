import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Code, useLabFile } from '../../components/Code';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  AXES,
  BOOK_RULE,
  CUDA_DOCUMENTED,
  FIG31,
  FORMULAS,
  LAUNCH_ERROR,
  blockCount,
  checkLaunch,
  coverage,
  describeViolation,
  dim3,
  drillTargets,
  formatBlockLabel,
  formatDim3,
  formatThreadLabel,
  inside,
  isDim3,
  limitsFromReport,
  linearThread,
  linearThreadWorking,
  misread,
  sameIndex,
  sizeGrid,
  sizeGridMinusOne,
  threadFromLabel,
  threadsPerBlock,
  totalThreads,
  workingInBlock,
  type Dim3,
  type Limits,
  type Target,
  type Unchecked,
} from '../../sim/grid';

// ---------------------------------------------------------------- small helpers
const num = (v: number | bigint) => v.toLocaleString('en-US');
const plural = (n: number | bigint, one: string, many = one + 's') => `${num(n)} ${Number(n) === 1 ? one : many}`;
const ORIGIN: Dim3 = { x: 0, y: 0, z: 0 };
/** A caption above a group of controls (a <label> would forward clicks on the caption to the first control). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };

/** A limit as the lesson writes it: 2^31 - 1 stays symbolic. */
function Cap({ v }: { v: number }) {
  return v === 2 ** 31 - 1 ? (
    <span className="nowrap">
      2<sup>31</sup> − 1
    </span>
  ) : (
    <>{num(v)}</>
  );
}

const CODE_TIERS: Record<string, Tier> = {
  out: 'global', out_d: 'global', A_d: 'global', B_d: 'global', C_d: 'global', A: 'global', B: 'global', C: 'global',
  threadsPerBlock: 'reg', block: 'reg', thread: 'reg', i: 'reg', n: 'reg', err: 'reg',
};

// ---------------------------------------------------------------- limit sets the reader can judge by
type Kind = 'book' | 'cuda-doc' | 'cuda' | 'hip' | 'emu' | 'metal' | 'other';
interface Choice {
  key: string;
  kind: Kind;
  label: string;
  limits: Limits;
  note: string;
}
const BOOK_CHOICE: Choice = {
  key: 'book',
  kind: 'book',
  label: "The book's rule (p. 47)",
  limits: BOOK_RULE,
  note: "The book's rule (section 3.1, p. 47): at most 1024 threads in a block, spread over its dimensions in any way; gridDim.x up to 2^31 − 1; gridDim.y and gridDim.z up to 65,535.",
};
const CUDA_CHOICE: Choice = {
  key: 'cuda-doc',
  kind: 'cuda-doc',
  label: "CUDA's documented limits",
  limits: CUDA_DOCUMENTED,
  note: "NVIDIA's documented limits, the same for every compute capability its table lists (checked 2026-11-20). Documented values, not read from a device.",
};

/** Limit sets from the reader's own device reports. */
function reported(reports: Reports): Choice[] {
  // Real devices first, then the emulator once (its two flavours report the same numbers).
  const emu = (b: string) => b === 'emu' || b === 'race';
  const keys = Object.keys(reports).sort((a, b) => Number(emu(a)) - Number(emu(b)) || a.localeCompare(b));
  return keys
    .filter((b) => b !== 'race' || !('emu' in reports))
    .flatMap((b): Choice[] => {
      const limits = limitsFromReport(reports[b].device);
      if (!limits) return [];
      const name = String(reports[b].device.name ?? b);
      const when = reports[b].generated.slice(0, 10);
      if (emu(b))
        return [
          {
            key: b,
            kind: 'emu',
            label: "The course's CPU emulator (not a GPU)",
            limits,
            note: 'The numbers the course emulator reports about itself. It is not a GPU; it copies the limits in NVIDIA\'s table and enforces them.',
          },
        ];
      if (b === 'metal')
        return [
          {
            key: b,
            kind: 'metal',
            label: `Yours: ${name} (metal)`,
            limits,
            note: `What this Metal device reported when ./verify.sh ran on ${when}: the most threads along each dimension of a threadgroup, which Apple says holds for a trivial kernel only.`,
          },
        ];
      return [
        {
          key: b,
          kind: b === 'cuda' ? 'cuda' : b === 'hip' ? 'hip' : 'other',
          label: `Yours: ${name} (${b})`,
          limits,
          note: `What this device reported in its device properties when ./verify.sh ran on ${when}.`,
        },
      ];
    });
}

function uncheckedNote(u: Unchecked, kind: Kind): string {
  if (kind === 'book' && u === 'block-dim') return 'The book gives no cap for a single dimension of a block, so none is tested.';
  if (kind === 'metal' && u === 'block-threads')
    return 'The total per threadgroup is not in a device record. Metal sets it for each compiled kernel (maxTotalThreadsPerThreadgroup).';
  if (kind === 'metal' && u === 'grid-dim') return 'Apple documents no limit on the size of a grid, so none is tested.';
  return `This record has no figure for ${u === 'block-threads' ? 'the threads in a block' : u === 'block-dim' ? 'a single dimension of a block' : 'the size of the grid'}, so it is not tested.`;
}

// What the course emulator itself does with a launch (labs/common/include/pmpp/emu/emu_runtime.h,
// launch_impl; checked by labs/ch03/snippets/check.py): it rejects a block of more than 1024
// threads, a block or grid dimension over NVIDIA's per-dimension caps, a zero dimension, and
// dynamic shared memory over its limit, all with cudaErrorInvalidConfiguration.
const EMU_ENFORCES =
  "The emulator enforces the limits it reports, which are NVIDIA's documented ones. It also rejects a zero dimension and too much dynamic shared memory.";

function errorNote(kind: Kind): string {
  const e = kind === 'hip' ? LAUNCH_ERROR.hip : LAUNCH_ERROR.cuda;
  switch (kind) {
    case 'book':
      return 'The book stops here. What a device does with such a launch is the subject of "What happens when you break a limit", further down this page.';
    case 'emu':
      return `The emulator rejects this launch: it sets ${e.name} (${e.value}) and prints a line. ${EMU_ENFORCES}`;
    case 'metal':
      return 'Apple documents no error for a threadgroup over its limit. Do not rely on any particular behaviour.';
    default:
      return `By the documentation, a launch like this fails and leaves ${e.name} (${e.value}) in the runtime's error state. The launch statement returns nothing, so ${e.query} is the only way to see it.`;
  }
}

// ---------------------------------------------------------------- the grid explorer
const DRAW = { blocks: { x: 8, y: 6, z: 3 }, threads: { x: 32, y: 16, z: 4 } } as const;

const cellStyle = (w: number, h: number): CSSProperties => ({
  width: w,
  height: h,
  padding: 0,
  borderColor: 'var(--rule)',
  borderRadius: 2,
  fontSize: '0.68rem',
  fontWeight: 450,
  fontVariantNumeric: 'tabular-nums',
  lineHeight: 1,
});
const HEAD: CSSProperties = {
  fontSize: '0.64rem',
  color: 'var(--ink-3)',
  textAlign: 'center',
  alignSelf: 'center',
  fontVariantNumeric: 'tabular-nums',
  whiteSpace: 'nowrap',
};

/** A table of cells with x across and y down, and the two index rulers. */
function Lattice({
  nx,
  ny,
  axis,
  label,
  cell,
}: {
  nx: number;
  ny: number;
  /** Name printed in the corner, e.g. "blockIdx" or "threadIdx". */
  axis: string;
  label: string;
  cell: (x: number, y: number) => ReactNode;
}) {
  const xs = Array.from({ length: nx }, (_, i) => i);
  const ys = Array.from({ length: ny }, (_, i) => i);
  return (
    <div
      role="group"
      aria-label={label}
      style={{ display: 'grid', gridTemplateColumns: `max-content repeat(${nx}, max-content)`, gap: 2, justifyContent: 'start' }}
    >
      <span style={{ ...HEAD, textAlign: 'left', paddingRight: 4 }} title={`${axis}.y runs down, ${axis}.x runs across`}>
        y ↓ x →
      </span>
      {xs.map((x) => (
        <span key={x} style={HEAD}>
          {x}
        </span>
      ))}
      {ys.map((y) => [
        <span key={'h' + y} style={{ ...HEAD, paddingRight: 4, textAlign: 'right' }}>
          {y}
        </span>,
        ...xs.map((x) => cell(x, y)),
      ])}
    </div>
  );
}

interface Preset {
  key: string;
  label: string;
  grid?: Dim3;
  block: Dim3;
}
const LAUNCH_PRESETS: Preset[] = [
  { key: 'fig31', label: 'Fig. 3.1', grid: FIG31.grid, block: FIG31.block },
  { key: 'p46', label: '32 blocks of 128 threads (p. 46)', grid: dim3(32), block: dim3(128) },
];
const BOOK_SHAPES: Dim3[] = [dim3(512, 1, 1), dim3(8, 16, 4), dim3(32, 16, 2), dim3(32, 32, 2)];
const OUR_SHAPE = dim3(1, 1, 128);
const SHAPE_PRESETS: Preset[] = [...BOOK_SHAPES, OUR_SHAPE].map((b) => ({
  key: `b${b.x}-${b.y}-${b.z}`,
  label: formatDim3(b),
  block: b,
}));

type Triple = Record<'x' | 'y' | 'z', string>;
const toText = (d: Dim3): Triple => ({ x: String(d.x), y: String(d.y), z: String(d.z) });
const toDim = (t: Triple): Dim3 => {
  const one = (s: string) => (s.trim() === '' ? NaN : Number(s));
  return { x: one(t.x), y: one(t.y), z: one(t.z) };
};

function GridExplorer() {
  const { reports, loading } = useReports();
  const choices = useMemo(() => [BOOK_CHOICE, CUDA_CHOICE, ...reported(reports)], [reports]);
  const [key, setKey] = useState('book');
  const [g, setG] = useState<Triple>(toText(FIG31.grid));
  const [b, setB] = useState<Triple>(toText(FIG31.block));
  const [selBlock, setSelBlock] = useState<Dim3>({ x: 0, y: 1, z: 0 });
  const [selThread, setSelThread] = useState<Dim3 | null>(null);

  const choice = choices.find((c) => c.key === key) ?? BOOK_CHOICE;
  const grid = toDim(g);
  const block = toDim(b);
  const valid = isDim3(grid) && isDim3(block);
  const drawable = valid && AXES.every((a) => grid[a] >= 1 && block[a] >= 1);
  const verdict = valid ? checkLaunch(grid, block, choice.limits) : null;
  const sb = drawable && inside(selBlock, grid) ? selBlock : ORIGIN;
  const st = drawable && selThread && inside(selThread, block) ? selThread : null;

  const apply = (p: Preset) => {
    if (p.grid) setG(toText(p.grid));
    setB(toText(p.block));
    setSelThread(null);
  };
  const active = (p: Preset) => valid && sameIndex(block, p.block) && (!p.grid || sameIndex(grid, p.grid));

  const fields = (name: string, t: Triple, set: (t: Triple) => void) => (
    <span style={FIELD}>
      <span>
        <code>{name}</code> (x, y, z)
      </span>
      <span className="row" style={{ gap: '0.3rem', flexWrap: 'nowrap' }}>
        {AXES.map((a) => (
          <input
            key={a}
            type="number"
            min="0"
            step="1"
            value={t[a]}
            aria-label={`${name}.${a}`}
            style={{ width: '5.4rem' }}
            onChange={(e) => {
              set({ ...t, [a]: e.target.value });
              setSelThread(null);
            }}
          />
        ))}
      </span>
    </span>
  );

  const shown = {
    bx: drawable ? Math.min(grid.x, DRAW.blocks.x) : 0,
    by: drawable ? Math.min(grid.y, DRAW.blocks.y) : 0,
    bz: drawable ? Math.min(grid.z, DRAW.blocks.z) : 0,
    tx: drawable ? Math.min(block.x, DRAW.threads.x) : 0,
    ty: drawable ? Math.min(block.y, DRAW.threads.y) : 0,
    tz: drawable ? Math.min(block.z, DRAW.threads.z) : 0,
  };
  const blocksCut = drawable && (shown.bx < grid.x || shown.by < grid.y || shown.bz < grid.z);
  const threadsCut = drawable && (shown.tx < block.x || shown.ty < block.y || shown.tz < block.z);
  const layers = (n: number) => Array.from({ length: n }, (_, i) => i);
  const perBlock = valid ? threadsPerBlock(block) : 0n;

  return (
    <Bench
      title="Grid explorer"
      task="Pick a block, then one of its threads. Compare the three ways the page names it."
      foot={<>{choice.note}</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        {fields('gridDim', g, setG)}
        {fields('blockDim', b, setB)}
        <label className="field">
          Judge the launch by
          <select value={choice.key} onChange={(e) => setKey(e.target.value)} aria-label="Judge the launch by">
            {choices.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.5rem' }} role="group" aria-label="Launch presets">
        <span className="muted">Launches</span>
        {LAUNCH_PRESETS.map((p) => (
          <button key={p.key} className="quiet" data-preset={p.key} aria-pressed={active(p)} onClick={() => apply(p)}>
            {p.label}
          </button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '1rem' }} role="group" aria-label="Block shape presets">
        <span className="muted">Block shapes</span>
        {SHAPE_PRESETS.map((p) => (
          <button key={p.key} className="quiet" data-preset={p.key} aria-pressed={active(p)} onClick={() => apply(p)}>
            {p.label}
          </button>
        ))}
        <span className="muted">the first four are the book's (p. 47), the last is ours</span>
      </div>
      {!loading && choices.length === 2 ? (
        <p className="muted" style={{ margin: '0 0 0.8rem' }}>
          No device report on this machine yet, so the list offers the book's rule and CUDA's documented limits. Run{' '}
          <code>./verify.sh</code> to add your own devices.
        </p>
      ) : null}

      <div className="cols" style={{ marginBottom: '1.2rem' }}>
        <div className="panel">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>The launch you are describing</b>
          </p>
          <pre className="mono" data-out="statement" style={{ margin: '0 0 0.7rem', fontSize: '0.78rem', lineHeight: 1.5, overflowX: 'auto' }}>
            {`dim3 dimGrid(${g.x || '?'}, ${g.y || '?'}, ${g.z || '?'});\ndim3 dimBlock(${b.x || '?'}, ${b.y || '?'}, ${b.z || '?'});\nKernelFunction<<<dimGrid, dimBlock>>>(...);`}
          </pre>
          {valid ? (
            <p style={{ margin: 0 }} aria-live="polite">
              <span data-out="blocks" data-value={String(blockCount(grid))}>
                {plural(blockCount(grid), 'block')}
              </span>{' '}
              ×{' '}
              <span data-out="per-block" data-value={String(perBlock)}>
                {plural(perBlock, 'thread')}
              </span>{' '}
              ={' '}
              <b data-out="threads" data-value={String(totalThreads(grid, block))}>
                {plural(totalThreads(grid, block), 'thread')}
              </b>
              <br />
              <span className="muted">One kernel call creates one grid. Every block in it has this shape.</span>
            </p>
          ) : (
            <p style={{ margin: 0 }} data-out="invalid">
              Each field takes a whole number, 0 or larger, that fits in an unsigned 32-bit integer.
            </p>
          )}
        </div>
        <div className="panel" aria-live="polite" data-out="verdict" data-legal={verdict ? String(verdict.legal) : 'none'}>
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>Verdict</b> <span className="muted">by {choice.label.replace(/^The /, 'the ').replace(/^Yours: /, '')}</span>
          </p>
          {!verdict ? (
            <p style={{ margin: 0 }} className="muted">
              No verdict until all six fields hold numbers.
            </p>
          ) : verdict.legal ? (
            <>
              <p style={{ margin: '0 0 0.4rem' }}>
                <span className="chip ok">accepted</span> No clause of this limit set is broken.
              </p>
              {verdict.unchecked.map((u) => (
                <p key={u} className="muted" style={{ margin: '0 0 0.3rem' }} data-out="unchecked">
                  {uncheckedNote(u, choice.kind)}
                </p>
              ))}
            </>
          ) : (
            <>
              <p style={{ margin: '0 0 0.4rem' }}>
                <span className="chip fail">rejected</span>{' '}
                {verdict.violations.length === 1 ? 'One clause fails.' : `${verdict.violations.length} clauses fail.`}
              </p>
              <ul style={{ margin: '0 0 0.5rem', paddingLeft: '1.2em' }}>
                {verdict.violations.map((v) => (
                  <li key={v.field} data-out="violation" data-clause={v.clause}>
                    {describeViolation(v)}
                  </li>
                ))}
              </ul>
              <p className="muted" style={{ margin: 0 }} data-out="error">
                {errorNote(choice.kind)}
              </p>
            </>
          )}
        </div>
      </div>

      {drawable ? (
        <>
          <div className="panel" style={{ marginBottom: '1.2rem', overflowX: 'auto' }}>
            <p style={{ margin: '0 0 0.6rem' }}>
              <b>The grid.</b> Each button is a block, named as the book names it:{' '}
              {grid.z > 1 ? '(blockIdx.z, blockIdx.y, blockIdx.x)' : '(blockIdx.y, blockIdx.x)'}.
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.4rem' }}>
              {layers(shown.bz).map((z) => (
                <div key={z}>
                  {grid.z > 1 ? <div className="mx-label">blockIdx.z = {z}</div> : null}
                  <Lattice
                    nx={shown.bx}
                    ny={shown.by}
                    axis="blockIdx"
                    label={`blocks, layer z ${z}`}
                    cell={(x, y) => (
                      <button
                        key={`${x}-${y}`}
                        data-block={`${x}-${y}-${z}`}
                        aria-pressed={sameIndex(sb, { x, y, z })}
                        aria-label={formatBlockLabel({ x, y, z }, grid)}
                        style={cellStyle(grid.z > 1 ? 56 : 44, 28)}
                        onClick={() => {
                          setSelBlock({ x, y, z });
                          setSelThread(null);
                        }}
                      >
                        {formatBlockLabel({ x, y, z }, grid).replace('Block ', '')}
                      </button>
                    )}
                  />
                </div>
              ))}
            </div>
            {blocksCut ? (
              <p className="muted" style={{ margin: '0.6rem 0 0' }} data-out="blocks-cut">
                Drawn: the first {shown.bx} × {shown.by} × {shown.bz} blocks (x × y × z) of {num(grid.x)} × {num(grid.y)} ×{' '}
                {num(grid.z)}. The rest look the same.
              </p>
            ) : null}
          </div>

          <div className="cols plot">
            <div className="panel" style={{ overflowX: 'auto' }}>
              <p style={{ margin: '0 0 0.6rem' }} data-out="block-title">
                <b>{formatBlockLabel(sb, grid)}.</b> In constructor order its <code>blockIdx</code> is {formatDim3(sb)}. Each
                button is one of its threads and shows the thread's number within the block.
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.2rem' }}>
                {layers(shown.tz).map((z) => (
                  <div key={z}>
                    <div className="mx-label">threadIdx.z = {z}</div>
                    <Lattice
                      nx={shown.tx}
                      ny={shown.ty}
                      axis="threadIdx"
                      label={`threads, layer z ${z}`}
                      cell={(x, y) => (
                        <button
                          key={`${x}-${y}`}
                          data-thread={`${x}-${y}-${z}`}
                          aria-pressed={!!st && sameIndex(st, { x, y, z })}
                          aria-label={formatThreadLabel({ x, y, z })}
                          style={cellStyle(30, 24)}
                          onClick={() => setSelThread({ x, y, z })}
                        >
                          {linearThread({ x, y, z }, block)}
                        </button>
                      )}
                    />
                  </div>
                ))}
              </div>
              {threadsCut ? (
                <p className="muted" style={{ margin: '0.6rem 0 0' }} data-out="threads-cut">
                  Drawn: the first {shown.tx} × {shown.ty} × {shown.tz} threads (x × y × z) of {num(block.x)} ×{' '}
                  {num(block.y)} × {num(block.z)}.
                </p>
              ) : null}
            </div>
            <div className="panel" aria-live="polite" data-out="thread">
              {st ? (
                <>
                  <p style={{ margin: '0 0 0.5rem' }}>
                    <b data-out="label">
                      {formatThreadLabel(st)} of {formatBlockLabel(sb, grid)}
                    </b>
                    <br />
                    <span className="muted">the book's label: highest dimension first</span>
                  </p>
                  <table className="data" style={{ margin: '0 0 0.6rem' }}>
                    <thead>
                      <tr>
                        <th>Built-in</th>
                        <th>(x, y, z)</th>
                        <th>Same value in</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>
                          <code>threadIdx</code>
                        </td>
                        <td className="nowrap" data-out="threadIdx">
                          {formatDim3(st)}
                        </td>
                        <td>one thread of each block</td>
                      </tr>
                      <tr>
                        <td>
                          <code>blockIdx</code>
                        </td>
                        <td className="nowrap" data-out="blockIdx">
                          {formatDim3(sb)}
                        </td>
                        <td>the {plural(perBlock, 'thread')} of this block</td>
                      </tr>
                      <tr>
                        <td>
                          <code>blockDim</code>
                        </td>
                        <td className="nowrap">{formatDim3(block)}</td>
                        <td>every thread of the grid</td>
                      </tr>
                      <tr>
                        <td>
                          <code>gridDim</code>
                        </td>
                        <td className="nowrap">{formatDim3(grid)}</td>
                        <td>every thread of the grid</td>
                      </tr>
                    </tbody>
                  </table>
                  <p style={{ margin: 0 }}>
                    Number within the block, x fastest:
                    <br />
                    <code>x + y·blockDim.x + z·blockDim.x·blockDim.y</code>
                    <br />= <span data-out="linear">{linearThreadWorking(st, block)}</span>
                  </p>
                </>
              ) : (
                <p className="muted" style={{ margin: 0 }}>
                  Click a thread to see its four built-in variables, the book's label for it, and its number within the
                  block.
                </p>
              )}
            </div>
          </div>
        </>
      ) : valid ? (
        <p className="muted" style={{ margin: 0 }} data-out="nothing">
          A dimension of 0 describes no blocks or no threads. There is nothing to draw.
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- find the thread
const DRILL = FIG31;
const DRILL_COUNT = 3;

function FindThread() {
  const [seed, setSeed] = useState(1);
  const targets = useMemo(() => drillTargets(DRILL.grid, DRILL.block, DRILL_COUNT, seed), [seed]);
  const [at, setAt] = useState(0);
  const [click, setClick] = useState<Target | null>(null);
  const [misses, setMisses] = useState(0);
  const [found, setFound] = useState(0);

  const target = targets[Math.min(at, targets.length - 1)];
  const hit = !!click && sameIndex(click.block, target.block) && sameIndex(click.thread, target.thread);
  const finished = hit && at === targets.length - 1;
  const read = click && !hit ? misread(target, click) : null;

  const press = (t: Target) => {
    if (hit) return;
    setClick(t);
    if (sameIndex(t.block, target.block) && sameIndex(t.thread, target.thread)) setFound(found + 1);
    else setMisses(misses + 1);
  };
  const next = () => {
    setAt(at + 1);
    setClick(null);
  };
  const restart = () => {
    setSeed(seed + 1);
    setAt(0);
    setClick(null);
    setMisses(0);
    setFound(0);
  };

  const cell = (blk: Dim3, thr: Dim3) => {
    const me = { block: blk, thread: thr };
    const isClick = !!click && sameIndex(click.block, blk) && sameIndex(click.thread, thr);
    const style: CSSProperties = { ...cellStyle(30, 26) };
    if (isClick) {
      style.background = hit ? 'var(--ok-wash)' : 'var(--fail-wash)';
      style.borderColor = hit ? 'var(--ok)' : 'var(--fail)';
      style.color = hit ? 'var(--ok)' : 'var(--fail)';
    }
    return (
      <button
        key={`${thr.x}-${thr.y}`}
        data-cell={`${blk.x}-${blk.y}-${blk.z}-${thr.x}-${thr.y}-${thr.z}`}
        data-state={isClick ? (hit ? 'found' : 'missed') : undefined}
        aria-label={`a thread of the block at blockIdx.x ${blk.x}, blockIdx.y ${blk.y}`}
        style={style}
        onClick={() => press(me)}
      >
        {isClick ? (hit ? '✓' : '×') : ''}
      </button>
    );
  };

  const blockXs = Array.from({ length: DRILL.grid.x }, (_, i) => i);
  const blockYs = Array.from({ length: DRILL.grid.y }, (_, i) => i);
  const zs = Array.from({ length: DRILL.block.z }, (_, i) => i);

  return (
    <Bench
      title="Find the thread"
      task="The cells carry no names. Use the rulers: x runs across, y runs down, z picks the layer."
      foot={
        <>
          The launch of the book's Fig. 3.1: <code>dimGrid(2, 2, 1)</code>, <code>dimBlock(4, 2, 2)</code>. The first
          target is the book's own example (p. 48); the others are drawn so that reading the label backwards lands on
          a different cell or on none.
        </>
      }
    >
      <p style={{ margin: '0 0 0.8rem', fontSize: '1rem' }} data-out="target" aria-live="polite">
        Target {Math.min(at, targets.length - 1) + 1} of {targets.length}: click{' '}
        <b>
          {formatThreadLabel(target.thread)} of {formatBlockLabel(target.block, DRILL.grid)}
        </b>
        .
      </p>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `max-content repeat(${DRILL.grid.x}, max-content)`,
          gap: '0.7rem',
          alignItems: 'center',
          marginBottom: '1rem',
        }}
      >
        <span />
        {blockXs.map((x) => (
          <span key={x} className="mx-label" style={{ margin: 0, textAlign: 'center' }}>
            blockIdx.x = {x}
          </span>
        ))}
        {blockYs.map((y) => [
          <span key={'y' + y} className="mx-label" style={{ margin: 0 }}>
            blockIdx.y = {y}
          </span>,
          ...blockXs.map((x) => {
            const blk = { x, y, z: 0 };
            return (
              <div key={`${x}-${y}`} className="panel" data-drill-block={`${x}-${y}`} style={{ display: 'flex', gap: '0.9rem' }}>
                {zs.map((z) => (
                  <div key={z}>
                    <div className="mx-label">threadIdx.z = {z}</div>
                    <Lattice
                      nx={DRILL.block.x}
                      ny={DRILL.block.y}
                      axis="threadIdx"
                      label={`block x ${x} y ${y}, layer z ${z}`}
                      cell={(tx, ty) => cell(blk, { x: tx, y: ty, z })}
                    />
                  </div>
                ))}
              </div>
            );
          }),
        ])}
      </div>
      <div className={'narrate' + (read ? ' bad' : '')} data-out="feedback" aria-live="polite">
        {!click ? (
          <>Nothing clicked yet.</>
        ) : hit ? (
          <>
            <b>Found.</b> {formatThreadLabel(target.thread)} is z = {target.thread.z}, y = {target.thread.y}, x ={' '}
            {target.thread.x}: the last number of the label is x. In constructor order its <code>threadIdx</code> is{' '}
            {formatDim3(target.thread)} and its <code>blockIdx</code> is {formatDim3(target.block)}.
          </>
        ) : (
          <>
            <b>
              That is {formatThreadLabel(click.thread)} of {formatBlockLabel(click.block, DRILL.grid)}.
            </b>{' '}
            {read!.block === 'reversed'
              ? `You went to blockIdx.x = ${click.block.x}, blockIdx.y = ${click.block.y}. A block label is (y, x): its first number is y.`
              : read!.block === 'other'
                ? 'The block is not the one asked for. A block label is (y, x).'
                : 'The block is right.'}{' '}
            {read!.thread === 'reversed'
              ? 'For the thread you read the label as (x, y, z). It is (z, y, x): the last number is x.'
              : read!.thread === 'other'
                ? 'The thread is not the one asked for. A thread label is (z, y, x): layer first, then row, then column.'
                : 'The thread is right.'}
          </>
        )}
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        {hit && !finished ? (
          <button className="primary" onClick={next}>
            Next target
          </button>
        ) : null}
        {finished ? (
          <button className="primary" onClick={restart}>
            Three new targets
          </button>
        ) : null}
        <span className="muted" data-out="score" data-found={found} data-misses={misses}>
          Found {found} of {targets.length}, with {plural(misses, 'miss', 'misses')}.
        </span>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- four ways to size a grid
const SIZE_PRESETS = [
  { n: 1000, b: 256 },
  { n: 4000, b: 256 },
];

function SizingBench() {
  const [n, setN] = useState(1000);
  const [b, setB] = useState(256);
  const rows = FORMULAS.map((f) => ({ f, c: coverage(n, b, sizeGrid(n, b, f.key)) }));
  const scale = Math.max(n, ...rows.map((r) => r.c.threads));
  const rem = n % b;
  const pct = (v: number) => (100 * v) / scale + '%';

  return (
    <Bench
      title="Four ways to size a grid"
      task="Find an n for which the second formula launches a block that does nothing. Then say when the first one is right."
      foot={
        <>
          One element per thread, in one dimension. <code>n</code> and <code>b</code> are <code>int</code>, so{' '}
          <code>/</code> between them truncates. The presets are the book's two examples (p. 46).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '1rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            Elements, <code>n</code> = <b data-out="n">{num(n)}</b>
          </span>
          <input type="range" min={1} max={5000} step={1} value={n} aria-label="n" style={{ width: '18rem' }} onChange={(e) => setN(Number(e.target.value))} />
        </label>
        <span className="row" style={{ gap: '0.3rem' }}>
          <button className="quiet" aria-label="n minus 1" onClick={() => setN(Math.max(1, n - 1))}>
            −1
          </button>
          <button className="quiet" aria-label="n plus 1" onClick={() => setN(Math.min(5000, n + 1))}>
            +1
          </button>
        </span>
        <label className="field">
          <span>
            Threads per block, <code>b</code> = <b data-out="b">{num(b)}</b>
          </span>
          <input type="range" min={32} max={1024} step={32} value={b} aria-label="b" style={{ width: '12rem' }} onChange={(e) => setB(Number(e.target.value))} />
        </label>
        <span className="row" role="group" aria-label="Presets" style={{ gap: '0.4rem' }}>
          <span className="muted">Presets</span>
          {SIZE_PRESETS.map((p) => (
            <button
              key={p.n}
              className="quiet"
              aria-pressed={n === p.n && b === p.b}
              onClick={() => {
                setN(p.n);
                setB(p.b);
              }}
            >
              n = {num(p.n)}, b = {p.b}
            </button>
          ))}
        </span>
      </div>
      <div className="panel" style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.3rem' }}>Number of blocks</th>
              <th className="num">Blocks</th>
              <th className="num">Threads</th>
              <th className="num">Surplus threads</th>
              <th className="num">Elements with no thread</th>
              <th className="num">Blocks with no work</th>
              <th style={{ minWidth: '15rem', paddingLeft: '0.8rem' }}>
                Threads against the {num(n)} elements
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ f, c }) => (
              <tr
                key={f.key}
                data-formula={f.key}
                data-blocks={c.blocks}
                data-threads={c.threads}
                data-surplus={c.surplus}
                data-uncovered={c.uncovered}
                data-idle={c.idleBlocks}
              >
                <td className="nowrap" style={{ paddingLeft: '0.3rem' }}>
                  <code style={{ whiteSpace: 'nowrap' }}>{f.code}</code>
                </td>
                <td className="num">{num(c.blocks)}</td>
                <td className="num">{num(c.threads)}</td>
                <td className="num">{num(c.surplus)}</td>
                <td className="num">
                  {c.uncovered > 0 ? <span className="chip fail">{num(c.uncovered)}</span> : <span className="chip ok">0</span>}
                </td>
                <td className="num">{c.idleBlocks > 0 ? <b>{c.idleBlocks}</b> : 0}</td>
                <td style={{ paddingLeft: '0.8rem' }}>
                  <div aria-hidden="true" style={{ display: 'flex', height: 14, position: 'relative' }}>
                    {Array.from({ length: c.blocks }, (_, k) => {
                      const w = workingInBlock(n, b, k);
                      const fill = (100 * w) / b;
                      return (
                        <div
                          key={k}
                          style={{
                            flex: `0 0 ${pct(b)}`,
                            boxSizing: 'border-box',
                            borderRight: '1px solid var(--page)',
                            outline: w === 0 ? '1px dashed var(--ink-2)' : undefined,
                            outlineOffset: -1,
                            background:
                              w === 0
                                ? 'transparent'
                                : `linear-gradient(to right, var(--ink-2) ${fill}%, var(--rule) ${fill}%)`,
                          }}
                        />
                      );
                    })}
                    {c.uncovered > 0 ? (
                      <div
                        style={{
                          flex: `0 0 ${pct(c.uncovered)}`,
                          boxSizing: 'border-box',
                          background: 'var(--fail-wash)',
                          border: '1px solid var(--fail)',
                        }}
                      />
                    ) : null}
                    <div style={{ position: 'absolute', left: pct(n), top: -3, bottom: -3, width: 0, borderLeft: '2px solid var(--ink)' }} />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted" style={{ margin: '0.6rem 0 0' }}>
          In the bars each segment is a block: filled where its threads have an element, faint where they are surplus,
          dashed where the whole block has nothing to do. The line marks element <code>n</code>. Red is data no thread
          reaches.
        </p>
      </div>
      <p style={{ margin: '0.9rem 0 0' }} data-out="sizing-summary" aria-live="polite">
        {rem === 0 ? (
          <>
            <b>
              {num(b)} divides {num(n)} exactly.
            </b>{' '}
            Truncation is right this time, and <code>n / b + 1</code> launches {plural(rows[1].c.blocks, 'block')}, the
            last of which has {num(b)} threads that all fail the test.
          </>
        ) : (
          <>
            <b>
              {num(b)} does not divide {num(n)}: {plural(rem, 'element')} are left over.
            </b>{' '}
            Truncation launches no thread for them. The other three launch one more block, in which {num(b - rem)}{' '}
            threads are surplus.
          </>
        )}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- tables computed from the model
const mark = (legal: boolean) => (legal ? <span className="chip ok">allowed</span> : <span className="chip fail">rejected</span>);

function ShapesTable() {
  const one = dim3();
  return (
    <table className="data" data-table="shapes">
      <thead>
        <tr>
          <th>Block shape</th>
          <th className="num">Threads</th>
          <th>The book's rule</th>
          <th>CUDA's documented limits</th>
        </tr>
      </thead>
      <tbody>
        {[...BOOK_SHAPES, OUR_SHAPE].map((s) => {
          const book = checkLaunch(one, s, BOOK_RULE);
          const cuda = checkLaunch(one, s, CUDA_DOCUMENTED);
          return (
            <tr key={formatDim3(s)} data-shape={`${s.x}-${s.y}-${s.z}`} data-book={String(book.legal)} data-cuda={String(cuda.legal)}>
              <td className="nowrap">
                <code>dim3{formatDim3(s)}</code>
                {s === OUR_SHAPE ? <span className="muted"> ours</span> : null}
              </td>
              <td className="num nowrap">
                {s.x} × {s.y} × {s.z} = {num(threadsPerBlock(s))}
              </td>
              <td>{mark(book.legal)}</td>
              <td>
                {mark(cuda.legal)}
                {cuda.violations.length ? <span className="muted"> {cuda.violations.map(describeViolation).join(' ')}</span> : null}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function LimitsTable() {
  const { reports, loading } = useReports();
  const mine = reported(reports);
  const triple = (d: Dim3 | null, none: string) =>
    d ? (
      <>
        <Cap v={d.x} />, <Cap v={d.y} />, <Cap v={d.z} />
      </>
    ) : (
      <span className="muted">{none}</span>
    );
  const rows = [
    { c: BOOK_CHOICE, name: 'The book (p. 47)', none: 'not stated' },
    { c: CUDA_CHOICE, name: "NVIDIA's documentation", none: 'not stated' },
    ...mine.map((c) => ({ c, name: c.label, none: c.kind === 'metal' ? 'not in a Metal device record' : 'not reported' })),
  ];
  return (
    <>
      <table className="data" data-table="limits">
        <thead>
          <tr>
            <th>Source</th>
            <th className="num">Threads in a block</th>
            <th>Block, per dimension (x, y, z)</th>
            <th>Grid, per dimension (x, y, z)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ c, name, none }) => (
            <tr key={c.key} data-limits={c.key}>
              <td>{name}</td>
              <td className="num">
                {c.limits.maxThreadsPerBlock != null ? num(c.limits.maxThreadsPerBlock) : <span className="muted">{none}</span>}
              </td>
              <td>{triple(c.limits.maxBlockDim, none)}</td>
              <td>{triple(c.limits.maxGridDim, none)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {!loading && mine.length === 0 ? (
        <p className="muted">
          No device report on this machine yet. Run <code>./verify.sh</code> and your own devices appear as rows here.
        </p>
      ) : null}
    </>
  );
}

// ---------------------------------------------------------------- quoted source
// Quoted from llama.cpp, ggml/src/ggml-cuda/pad.cu, as recorded in notes/today/ch03-facts.md (D1).
// S31.test.ts fails if these lines drift from the facts file.
const PAD_KERNEL = `    // blockIdx.z: i3*ne2+i2
    // blockIdx.y: i1
    // blockIDx.x: i0 / CUDA_PAD_BLOCK_SIZE
    // gridDim.y and gridDim.z are capped at 65535, blocks stride over larger ne1 and ne2*ne3`;
const PAD_LAUNCH = `    int  num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    dim3 gridDim(num_blocks, std::min(ne1, 65535), std::min(ne2 * ne3, 65535));
    pad_f32<<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(src, ...);`;

// ---------------------------------------------------------------- questions
const EX3 = {
  bd: dim3(16, 32),
  gd: dim3(sizeGridMinusOne(300, 16), sizeGridMinusOne(150, 32)),
};
const EX3_THREADS = Number(totalThreads(EX3.gd, EX3.bd));
const LABEL_Q = { thread: threadFromLabel([1, 0, 2]), block: dim3(4, 2, 2) };

export const questionsS31: Record<string, Question> = {
  'ch03/3.1/q-label': {
    kind: 'numeric',
    id: 'ch03/3.1/q-label',
    prompt: (
      <>
        A block is launched as <code>dim3(4, 2, 2)</code>. Its threads are numbered from 0 with x moving fastest, then
        y, then z. What number does the book's <b>Thread (1,0,2)</b> get?
      </>
    ),
    answer: linearThread(LABEL_Q.thread, LABEL_Q.block),
    explain: (
      <p>
        The label is (z, y, x), so x = 2, y = 0, z = 1, and the number is{' '}
        {linearThreadWorking(LABEL_Q.thread, LABEL_Q.block)}. Reading the label in constructor order gives 1 + 0×4 +
        2×8 = 17, and a block of 16 threads has no thread 17. That is a useful alarm: when a label read your way
        points outside the block, you have the order backwards.
      </p>
    ),
    source: 'ch. 3 §3.1, p. 48, for the label; NVIDIA CUDA Programming Guide (checked 2026-11-20) for the numbering',
  },
  'ch03/3.1/q-zcap': {
    kind: 'choice',
    id: 'ch03/3.1/q-zcap',
    prompt: (
      <>
        Going by NVIDIA's documented limits, which of these block shapes is rejected although it has fewer than 1024
        threads?
      </>
    ),
    options: [
      { label: <code>dim3(512, 1, 1)</code>, why: '512 threads, and x may go up to 1024.' },
      { label: <code>dim3(8, 16, 4)</code>, why: '512 threads, and every dimension is inside its cap.' },
      {
        label: <code>dim3(32, 16, 2)</code>,
        why: 'Exactly 1024 threads, which is allowed, and z = 2 is far below its cap.',
      },
      {
        label: <code>dim3(1, 1, 128)</code>,
        correct: true,
        why: 'Only 128 threads, but the z dimension of a block is capped at 64.',
      },
    ],
    explain: (
      <p>
        A block shape has to pass two tests: the product of its dimensions is at most 1024, and each dimension is
        inside its own cap of 1024, 1024 and 64. The book states only the first. Its own examples pass or fail both
        tests alike, which is why the gap does not show there.
      </p>
    ),
    source: 'ch. 3 §3.1, p. 47, for the three allowed examples; NVIDIA CUDA Programming Guide, compute capabilities table (checked 2026-11-20), for the per-dimension caps',
  },
  'ch03/3.1/q-silent': {
    kind: 'choice',
    id: 'ch03/3.1/q-silent',
    prompt: (
      <>
        A host function launches a kernel with <code>dimBlock(32, 32, 2)</code>, copies the output buffer back and
        prints it. It never calls <code>cudaGetLastError()</code>. Going by NVIDIA's documentation, what happens?
      </>
    ),
    options: [
      {
        label: 'The file does not compile: nvcc rejects a block of 2048 threads on the launch line.',
        why: 'The compiler sees three integers. The limit belongs to the device and is checked when the launch runs.',
      },
      {
        label: 'The launch statement throws an exception that names the bad configuration.',
        why: 'A launch statement has no value and throws nothing. The failure goes into an error state that you have to query.',
      },
      {
        label: 'The runtime shrinks the block to the 1024 threads the device allows and runs it.',
        why: 'Nothing is adjusted for you. The launch fails as a whole.',
      },
      {
        label: 'No thread runs, nothing is reported, and the program prints whatever the buffer held.',
        correct: true,
      },
    ],
    explain: (
      <p>
        The launch fails and records <code>cudaErrorInvalidConfiguration</code> in the runtime's error state. The
        program was never told to look there, so it carries on and prints stale or uninitialised memory as if it were
        a result. This is why real launch code asks for the error after every launch. The statement comes from
        NVIDIA's documentation; we have not run it on a CUDA device (unverified on device).
      </p>
    ),
    source: 'NVIDIA CUDA Programming Guide and Runtime API reference (checked 2026-11-20); not in the book',
  },
  'ch03/3.1/q-metal-size': {
    kind: 'choice',
    id: 'ch03/3.1/q-metal-size',
    prompt: (
      <>
        In CUDA, <code>dim3 dimGrid(42);</code> describes a 1D grid of 42. What is the Metal host code for the same
        size?
      </>
    ),
    options: [
      { label: <code>MTLSizeMake(42)</code>, why: 'MTLSizeMake takes three arguments and has no defaults.' },
      {
        label: <code>MTLSizeMake(42, 1, 1)</code>,
        correct: true,
        why: 'All three are given, with 1 for the dimensions you do not use.',
      },
      {
        label: <code>MTLSizeMake(42, 0, 0)</code>,
        why: 'Unused dimensions are 1, as in CUDA. With zeros the size would be 42 × 0 × 0.',
      },
      {
        label: (
          <>
            A <code>uint3</code> argument of the kernel function
          </>
        ),
        why: 'The size of a launch is host code, given at the dispatch call. Kernel arguments are where a Metal thread receives its position.',
      },
    ],
    explain: (
      <p>
        <code>MTLSize</code> is Metal's counterpart of <code>dim3</code>, with fields <code>width</code>,{' '}
        <code>height</code> and <code>depth</code>. The rule for unused dimensions is the same as CUDA's, but nothing
        fills them in for you.
      </p>
    ),
    source: 'Apple, MTLSize documentation (checked 2026-11-20); not in the book',
  },
  'ch03/3.1/q-ex3-threads': {
    kind: 'numeric',
    id: 'ch03/3.1/q-ex3-threads',
    prompt: (
      <>
        Host code sets <code>M = 150</code> and <code>N = 300</code> and launches a kernel with{' '}
        <code>dim3 bd(16, 32);</code> and <code>dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1);</code>. All the arithmetic is
        in integers. How many threads are in the grid?
      </>
    ),
    answer: EX3_THREADS,
    unit: 'threads',
    explain: (
      <>
        <p>
          Three steps, which are parts (a), (c) and (b) of the book's exercise. <code>bd</code> has two arguments, so
          its z is 1 and a block holds 16 × 32 = {num(threadsPerBlock(EX3.bd))} threads. The grid is (299/16 + 1,
          149/32 + 1) = (18 + 1, 4 + 1) = ({EX3.gd.x}, {EX3.gd.y}), which is {num(blockCount(EX3.gd))} blocks. So the
          grid has {num(blockCount(EX3.gd))} × {num(threadsPerBlock(EX3.bd))} = {num(EX3_THREADS)} threads.
        </p>
        <p>
          Notice which size went where: x is sized from <code>N</code>, the 300 columns, and y from <code>M</code>, the
          150 rows. The threads span {EX3.gd.x * EX3.bd.x} by {EX3.gd.y * EX3.bd.y} for data of 300 by 150. How many of
          them have an element to work on is the next section's subject.
        </p>
      </>
    ),
    source: 'ch. 3, exercise 3, p. 65',
  },
};

// ---------------------------------------------------------------- the section
const FIG31_TOTAL = Number(totalThreads(FIG31.grid, FIG31.block));
const FIG31_BLOCKS = Number(blockCount(FIG31.grid));
const FIG31_PER = Number(threadsPerBlock(FIG31.block));

export function S31() {
  const launch1d = useLabFile('ch03/snippets/launch_1d.cu');
  const fig31 = useLabFile('ch03/snippets/launch_fig31.cu');
  const checked = useLabFile('ch03/snippets/launch_checked.cu');
  return (
    <>
      <p>
        In chapter 2 one number, <code>i</code>, told a thread what to do. A matrix element needs two numbers, and an
        element of a <code>[batch, sequence, hidden]</code> tensor needs three. CUDA lets the launch itself have up to
        three dimensions, so that a thread is handed coordinates in the shape of its data. This section is about the
        launch: how you state its shape, what each thread can then read about itself, and which shapes a device
        accepts. What a thread does with its coordinates is{' '}
        <a href={href('ch03', '3.2')}>section 3.2</a>.
      </p>
      <Source section="3.1" pages="45–48" figs="Fig. 3.1" />

      <h2>Same code, different coordinates</h2>
      <p>
        Every thread of a grid runs the same kernel function. The only thing that differs from one thread to the next
        is the value of two built-in variables, and a thread uses them to pick its part of the data. Two more
        built-ins carry the shape of the launch into the kernel and read the same in every thread.
      </p>
      <Source section="3.1" pages="45–47" />
      <table className="data">
        <thead>
          <tr>
            <th>Built-in</th>
            <th>Holds</th>
            <th>Range of each component</th>
            <th>Same value in</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <code>gridDim</code>
            </td>
            <td>the size of the grid, in blocks</td>
            <td>set by the launch</td>
            <td>every thread of the grid</td>
          </tr>
          <tr>
            <td>
              <code>blockDim</code>
            </td>
            <td>the size of a block, in threads</td>
            <td>set by the launch</td>
            <td>every thread of the grid</td>
          </tr>
          <tr>
            <td>
              <code>blockIdx</code>
            </td>
            <td>where this block is in the grid</td>
            <td>
              0 to <code>gridDim</code> − 1
            </td>
            <td>all threads of one block</td>
          </tr>
          <tr>
            <td>
              <code>threadIdx</code>
            </td>
            <td>where this thread is in its block</td>
            <td>
              0 to <code>blockDim</code> − 1
            </td>
            <td>one thread of each block</td>
          </tr>
        </tbody>
      </table>
      <p>
        Each of the four has an <code>x</code>, a <code>y</code> and a <code>z</code>. The hierarchy has two levels and
        no more: a grid is a three-dimensional array of blocks, and a block is a three-dimensional array of threads.
        All blocks of a grid have the same shape. The four names belong to CUDA C++ and cannot be changed.
      </p>
      <Today checked="2026-11-20">
        <p>
          NVIDIA documents <code>gridDim</code> and <code>blockDim</code> as <code>dim3</code>, and{' '}
          <code>blockIdx</code> and <code>threadIdx</code> as <code>uint3</code>. Both are structures of three unsigned
          values. Unsigned matters in section 3.3, where an index has to be able to go below zero.
        </p>
      </Today>

      <h2>The execution configuration</h2>
      <p>
        The shape is set at the call. Between <code>&lt;&lt;&lt;</code> and <code>&gt;&gt;&gt;</code> go two values of
        type <code>dim3</code>: first the size of the grid, counted in blocks, then the size of one block, counted in
        threads. A <code>dim3</code> is a vector of three integers named <code>x</code>, <code>y</code> and{' '}
        <code>z</code>. A dimension you do not use is 1, never 0.
      </p>
      <Source section="3.1" pages="46–47" />
      <Code
        tabs={[{ label: 'CUDA', code: launch1d ?? '// loading…', status: 'front-end-checked', tiers: CODE_TIERS }]}
        highlight={[5, 6, 12]}
        caption="The book's 1D example (p. 46): 32 blocks of 128 threads, which is 4096 threads. dimGrid and dimBlock are ordinary host variables with ordinary names; the book launches the same grid from two variables called dog and cat to make the point."
      />
      <p>
        The second function works because the constructor of <code>dim3</code> gives 1 to every component you leave
        out. A single integer therefore fills <code>x</code> and leaves <code>y</code> and <code>z</code> at 1. Inside
        the kernel the two values arrive as <code>gridDim</code> and <code>blockDim</code>: with the launch above,
        every thread reads <code>gridDim.x</code> = 32 and <code>blockDim.x</code> = 128. Once a grid is launched its
        shape stays fixed until the whole grid has finished.
      </p>
      <p>
        Grid and block do not need the same number of dimensions. The book's figure for this section uses a
        two-dimensional grid of three-dimensional blocks.
      </p>
      <Source section="3.1" pages="47–48" figs="Fig. 3.1" />
      <Code
        tabs={[{ label: 'CUDA', code: fig31 ?? '// loading…', status: 'front-end-checked', tiers: CODE_TIERS }]}
        highlight={[15, 16]}
        caption="The launch of the book's Fig. 3.1 (lines 15 and 16), with a kernel of ours behind it so that there is something to launch. Each thread writes one integer: its own number within its block. Line 9 comes back later in this section."
      />
      <Predict
        id="ch03/3.1/p-fig31"
        prompt={
          <>
            How many threads does the launch on lines 15 and 16 create? Commit to a number before you count them on the
            page.
          </>
        }
        unit="threads"
      >
        {(v) => (
          <p>
            <b>{FIG31_TOTAL}.</b> The grid is 2 × 2 × 1 = {FIG31_BLOCKS} blocks and each block is 4 × 2 × 2 ={' '}
            {FIG31_PER} threads.{' '}
            {v === FIG31_TOTAL
              ? 'That is what you had.'
              : v === FIG31_PER
                ? 'Sixteen is one block. The grid has four of them.'
                : v === FIG31_BLOCKS
                  ? 'Four is the number of blocks. Each holds sixteen threads.'
                  : v === 20 || v === 13
                    ? 'The sizes multiply. Every block of the grid holds a full set of threads.'
                    : 'Multiply the three numbers of each dim3, then multiply the two results.'}{' '}
            The explorer below opens on this launch. Real grids hold thousands to millions of threads; this one is
            small enough to draw.
          </p>
        )}
      </Predict>

      <h2>Sizing a grid for n elements</h2>
      <p>
        Shapes are rarely constants. For <code>n</code> elements and blocks of <code>b</code> threads you need enough
        blocks to cover all of <code>n</code>, which means dividing and rounding up. The book writes{' '}
        <code>ceil(n/256.0)</code>: with blocks of 256, <code>n</code> = 1000 gives 4 blocks and <code>n</code> = 4000
        gives 16. The <code>.0</code> is doing work. It makes the division happen in floating point, so that there is
        a fraction left for <code>ceil</code> to round up.
      </p>
      <Source section="3.1" pages="46" />
      <p>
        The other way stays in integers: <code>(n + b - 1) / b</code>. The book's own exercises use integer forms,{' '}
        <code>(N + 128 - 1)/128</code> in chapter 2 (exercise 9, p. 42) and <code>(N - 1)/16 + 1</code> in this chapter
        (exercise 3, p. 65). The explorer puts both ways next to the two shortcuts people reach for first.
      </p>
      <Today checked="2026-11-20">
        <p>
          NVIDIA's guide sizes its grids in integers, <code>(vectorLength + threads-1)/threads</code>, or with{' '}
          <code>cuda::ceil_div</code> from <code>&lt;cuda/cmath&gt;</code>. The <code>ceil</code> form is on no vendor
          page we read. It hands a <code>double</code> to <code>dim3</code> and relies on an implicit conversion, and
          we have not compiled it with nvcc (unverified). The labs use the integer form.
        </p>
      </Today>
      <SizingBench />
      <p>
        Two things go wrong. Truncating leaves the last <code>n mod b</code> elements without a thread, and nothing
        complains: the kernel is correct and the tail of the output is simply never written. Adding one
        unconditionally is right until <code>b</code> divides <code>n</code>, and then it launches a block in which
        every thread fails the test.
      </p>
      <Today checked="2026-11-20">
        <p>
          NVIDIA's guide on the cost of each kind of waste: "Launching extra threads in a block that do no work does
          not incur a large overhead cost, however launching thread blocks in which no threads do work should be
          avoided."
        </p>
      </Today>
      <p>
        The last two formulas always agree, and they never do either. Surplus threads inside the last block are the
        normal price of a grid made of whole blocks, which is why the kernel keeps its <code>if</code>.
      </p>
      <Simplification>
        <p>
          One element per thread, one dimension, and sizes far below the range of <code>int</code>, so the overflow
          that <code>n + b - 1</code> can run into near the top of that range does not appear. The <code>ceil</code>{' '}
          row is computed in double precision and taken as an integer; the conversion that C++ performs when the
          result initialises a <code>dim3</code> is not modelled.
        </p>
      </Simplification>

      <h2>The label order trap</h2>
      <p>
        You write a launch shape x first: <code>dim3(4, 2, 2)</code> is 4 wide in x. The book names blocks and threads
        the other way round, highest dimension first: Block (<code>blockIdx.y</code>, <code>blockIdx.x</code>) and
        Thread (<code>threadIdx.z</code>, <code>threadIdx.y</code>, <code>threadIdx.x</code>). So Block (1,0) is the
        block with <code>blockIdx.y</code> = 1 and <code>blockIdx.x</code> = 0, and Thread (1,0,2) has z = 1, y = 0
        and x = 2. Every figure and worked example from here to the end of the book uses these labels.
      </p>
      <Source section="3.1" pages="48, 49" figs="Fig. 3.1; footnote 1" />
      <p>
        The reason is the order of data indices. You write a matrix element as <code>P[row][col]</code> and a tensor
        shape as <code>(batch, sequence, hidden)</code>: the index that varies slowest first, the one that varies
        fastest last. The book labels threads the way you index data, so that x, the last number of a label, lines up
        with the last index. Read it as "x is the last axis" and the reversal stops being a trap. The book itself
        calls the two opposite orders a source of confusion.
      </p>
      <Today checked="2026-11-20">
        <p>
          The label order is the book's convention. NVIDIA's documentation writes x first everywhere. It also fixes
          how the threads of a block are numbered: "the first index x moves the fastest, followed by y and then z", so
          that <code>threadIdx.y</code> has a stride of <code>blockDim.x</code> and <code>threadIdx.z</code> a stride
          of <code>blockDim.x * blockDim.y</code>. That is line 9 of the kernel above.
        </p>
      </Today>
      <GridExplorer />
      <p>
        Try the block shapes one after another and watch the thread numbers. They count along x, then start the next
        row of y, then move to the next layer of z. A three-dimensional block is one line of threads, folded twice.
      </p>
      <Simplification>
        <p>
          Nothing is launched. A verdict says only that six numbers pass or fail a table of limits. Limits that depend
          on more than the two shapes are not tested: shared memory per block, the registers a kernel uses (a CUDA
          launch under 1024 threads can fail for that reason), and on Metal the thread limit of each compiled kernel.
          Large shapes are drawn in part.
        </p>
      </Simplification>
      <p>
        The same folding rule is what <a href={href('ch03', '3.2')}>section 3.2</a> applies to data, and chapter 4
        uses it to cut a block into warps. Beyond that, the number of dimensions you give a launch is bookkeeping:
        choose the shape that makes your index arithmetic easy to get right.
      </p>
      <Today checked="2026-11-20">
        <p>
          NVIDIA's guide: "The use of multi-dimensional thread blocks and grids is for convenience only and does not
          affect performance."
        </p>
      </Today>
      <p>
        Now without the names on the cells. The drill below uses the launch of Fig. 3.1 and asks for threads by the
        book's labels.
      </p>
      <FindThread />
      <Ask q={questionsS31['ch03/3.1/q-label']} />

      <h2>How large a launch can be</h2>
      <p>
        The book gives three limits. A block holds at most 1024 threads in total on current systems. A grid has at
        most 2<sup>31</sup> − 1 blocks along x and at most 65,535 along y and along z. It adds that the 1024 threads
        can be spread over the three dimensions in any way, with four examples: (512, 1, 1), (8, 16, 4) and (32, 16, 2)
        are allowed, and (32, 32, 2) is not, because its product is 2048.
      </p>
      <Source section="3.1" pages="47" />
      <p>
        "In any way" is too loose. Each dimension of a block has a cap of its own as well, and the cap on z is small.
      </p>
      <Today checked="2026-11-20">
        <p>
          NVIDIA's table of limits, the same for every compute capability it lists: at most 1024 threads in a block; a
          block at most 1024 in x, 1024 in y and 64 in z; a grid at most 2<sup>31</sup> − 1 in x and 65,535 in y and
          z. So (1, 1, 128) has only 128 threads and is not a legal block. The book's four examples come out the same
          under both rules.
        </p>
      </Today>
      <ShapesTable />
      <p>
        Set the explorer above to <code>(1, 1, 128)</code> and switch what it judges by. Then treat all of these
        numbers as documentation, not as facts about your device. The portable way to get them is to ask: a CUDA or
        HIP device reports <code>maxThreadsPerBlock</code>, <code>maxThreadsDim</code> and <code>maxGridSize</code> in
        its device properties.
      </p>
      <LimitsTable />
      <Ask q={questionsS31['ch03/3.1/q-zcap']} />

      <h2>What happens when you break a limit</h2>
      <p>
        The book does not say. It is worth knowing, because the failure does not look like one: a program with an
        illegal launch compiles, runs to the end and prints numbers.
      </p>
      <Beyond>
        <p>
          From NVIDIA's and AMD's current documentation. Nothing in this part was run on a CUDA or HIP device
          (unverified on device).
        </p>
      </Beyond>
      <p>
        A launch with <code>dimBlock(32, 32, 2)</code> compiles, because a shape is ordinary run-time data. The launch
        statement has no value, so there is nothing to test on that line. At run time the call fails. No thread runs,
        and the output buffer keeps whatever it held. The runtime records the failure as{' '}
        <code>cudaErrorInvalidConfiguration</code>, value 9, in an error state it keeps for each host thread, and the
        program continues. You see the error only if you ask for it.
      </p>
      <Today checked="2026-11-20">
        <p>
          NVIDIA: "The function call fails if grid_dim or block_dim exceeds the maximum sizes allowed for the device".{' '}
          <code>cudaGetLastError()</code> returns the error state and resets it to <code>cudaSuccess</code>;{' '}
          <code>cudaPeekAtLastError()</code> returns it without resetting. HIP mirrors both, with{' '}
          <code>hipErrorInvalidConfiguration</code>, also 9, and <code>hipGetLastError()</code>.
        </p>
      </Today>
      <Code
        tabs={[{ label: 'CUDA', code: checked ?? '// loading…', status: 'front-end-checked', tiers: CODE_TIERS }]}
        highlight={[11, 12]}
        caption="Asking after the launch. On the course's CPU emulator, blockShapeIsRejected returns true and the buffer is left untouched (emulator-checked: python3 labs/ch03/snippets/check.py). The emulator also prints a line of its own; the documentation describes no message from a device."
      />
      <p>
        PyTorch does this after every launch. Its CUDA kernels are followed by the macro{' '}
        <code>C10_CUDA_KERNEL_LAUNCH_CHECK()</code>, which is defined as <code>C10_CUDA_CHECK(cudaGetLastError())</code>
        . Do the same in your own launch functions. The course's emulator applies NVIDIA's documented limits, per
        dimension as well as in total: a block of (1, 1, 128) and a grid of 65,536 blocks along y fail there with the
        same error (emulator-checked by the same script). Those are the documented limits, not your device's; a
        real device can also refuse a smaller launch, as the comparison below explains.
      </p>
      <Ask q={questionsS31['ch03/3.1/q-silent']} />

      <h2>The same launch on HIP and Metal</h2>
      <p>
        On HIP the host lines of this section are the same source. On Metal every idea has a counterpart and every
        name is different.
      </p>
      <Beyond>
        <p>
          The book covers CUDA only. This comparison is ours, from AMD's and Apple's current documentation and
          headers. "Not documented" means we looked and found no statement.
        </p>
      </Beyond>
      <table className="data" data-table="backends">
        <thead>
          <tr>
            <th />
            <th>CUDA</th>
            <th>HIP</th>
            <th>Metal</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <b>Size type</b>
            </td>
            <td>
              <code>dim3</code>; components default to 1
            </td>
            <td>
              <code>dim3</code>, same fields, same default
            </td>
            <td>
              <code>MTLSize</code> with <code>width</code>, <code>height</code>, <code>depth</code>; no defaults, unused
              ones must be 1
            </td>
          </tr>
          <tr>
            <td>
              <b>Launch</b>
            </td>
            <td>
              <code>kernel&lt;&lt;&lt;grid, block&gt;&gt;&gt;(args)</code>
            </td>
            <td>the same statement</td>
            <td>
              on a compute encoder: <code>dispatchThreadgroups</code> takes a count of groups, as CUDA does;{' '}
              <code>dispatchThreads</code> takes the total thread grid
            </td>
          </tr>
          <tr>
            <td>
              <b>Coordinates in the kernel</b>
            </td>
            <td>
              built-in variables; <code>gridDim</code> and <code>blockDim</code> are <code>dim3</code>,{' '}
              <code>blockIdx</code> and <code>threadIdx</code> are <code>uint3</code>
            </td>
            <td>
              the same four names, each with a type of its own: use <code>.x</code>, <code>.y</code>, <code>.z</code>{' '}
              and never copy a whole built-in
            </td>
            <td>
              attributed kernel parameters: <code>threadgroups_per_grid</code>, <code>threads_per_threadgroup</code>,{' '}
              <code>threadgroup_position_in_grid</code>, <code>thread_position_in_threadgroup</code>
            </td>
          </tr>
          <tr>
            <td>
              <b>Threads in a block or group</b>
            </td>
            <td>
              1024 for the device; per dimension 1024, 1024, 64. A kernel that uses many registers can fail below 1024
              with <code>cudaErrorLaunchOutOfResources</code>
            </td>
            <td>
              1024; per dimension 1024, 1024, 1024 in AMD's table (read <code>maxThreadsDim</code> on a newer device).
              A kernel compiled with <code>__launch_bounds__</code> refuses a larger block
            </td>
            <td>
              set for each compiled kernel: <code>maxTotalThreadsPerThreadgroup</code> of the pipeline state
            </td>
          </tr>
          <tr>
            <td>
              <b>Grid size</b>
            </td>
            <td>
              x up to 2<sup>31</sup> − 1; y and z up to 65,535
            </td>
            <td>
              x up to 2<sup>32</sup> − 1, y and z up to 65,535 in AMD's table; read <code>maxGridSize</code>
            </td>
            <td>not documented</td>
          </tr>
          <tr>
            <td>
              <b>A launch over a limit</b>
            </td>
            <td>
              <code>cudaErrorInvalidConfiguration</code>, seen through <code>cudaGetLastError()</code>
            </td>
            <td>
              <code>hipErrorInvalidConfiguration</code>, seen through <code>hipGetLastError()</code>
            </td>
            <td>not documented</td>
          </tr>
        </tbody>
      </table>
      <p>
        Two rows deserve a second look. The thread limit is a ceiling for the device, and on all three backends one
        kernel can sit below it. A CUDA kernel that needs many registers per thread can fail to launch with fewer than
        1024 threads. A HIP kernel compiled with a launch bound refuses a larger block. A Metal pipeline state reports
        the limit for its own compiled kernel. So "is this shape legal" is a question about a kernel as well as a
        device; section 4.7 comes back to registers. And the four Metal parameters in the third row are, in the order
        listed, <code>gridDim</code>, <code>blockDim</code>, <code>blockIdx</code> and <code>threadIdx</code>. Metal
        writes positions x first and has no (z, y, x) labels.
      </p>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md, A4, B1, B4; ch03-facts.md, B3, C4)">
        <p>
          NVIDIA: <code>cudaErrorLaunchOutOfResources</code> covers "too many threads for the kernel's register
          count". AMD: a launch is checked against the kernel's <code>__launch_bounds__</code>, and{' '}
          <code>gridDim</code> times <code>blockDim</code> must fit in 32 bits per dimension. Apple: the pipeline's
          limit "varies depending on the complexity of the function".
        </p>
      </Today>
      <Ask q={questionsS31['ch03/3.1/q-metal-size']} />

      <h2>Where you will meet this: a 3D grid over a 4D tensor</h2>
      <p>
        llama.cpp pads a tensor with a kernel whose launch uses everything in this section. A ggml tensor has four
        sizes, <code>ne0</code> to <code>ne3</code>, with <code>ne0</code> the innermost one, the one that varies
        fastest in memory. A grid has three dimensions. This is how four are fitted onto three.
      </p>
      <Code
        tabs={[{ label: 'llama.cpp', code: PAD_KERNEL + '\n    ...\n' + PAD_LAUNCH }]}
        highlight={[7]}
        caption="Quoted from llama.cpp, ggml/src/ggml-cuda/pad.cu (master at b94959a, read 2026-11-20; blockIDx is spelled as in the source). Not our code and not compiled here. The comment is from the kernel, the three lines after the ellipsis from its host function."
      />
      <p>
        The innermost tensor dimension goes on x, split into blocks with the integer round-up from the sizing explorer.
        The next one, <code>ne1</code>, gets one block per index along y. The two outer dimensions are folded into a
        single number, <code>i3*ne2 + i2</code>, along z. The two <code>std::min</code> calls on line 7 are the 65,535
        limit of this section in production code: when <code>ne1</code> or <code>ne2 * ne3</code> is larger, the grid
        stops at the cap and each block loops over the indices it has to cover. The same file is compiled for AMD
        devices.
      </p>
      <Today checked="2026-11-20">
        <p>
          The model of this chapter, one thread per element with the index arithmetic written by hand, is what
          NVIDIA's guide now calls SIMT kernels. The guide also documents a second model, tile kernels, in which the
          compiler picks the block size and these four built-ins do not exist. This course does not use it.
        </p>
      </Today>

      <h2>Check: one launch, read from host code</h2>
      <Ask q={questionsS31['ch03/3.1/q-ex3-threads']} />
      <p>
        Next, <a href={href('ch03', '3.2')}>section 3.2</a> turns a thread's coordinates into a row and a column, and
        those into one offset into a flat array.
      </p>
    </>
  );
}
