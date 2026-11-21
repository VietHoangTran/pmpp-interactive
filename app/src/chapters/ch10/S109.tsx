import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, num, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  DESIGN_LABEL,
  VOCABS,
  blockSums,
  blockView,
  f32,
  f32hex,
  f32short,
  f64short,
  history,
  logits,
  reference,
  runOf,
  segments,
  threadLoads,
  ulpDistance,
  type Design,
} from '../../sim/multiblock';
import { functionSource, type FunctionSource } from '../../sim/toolchain';
import { CommitOrderScene } from './CommitOrderScene';
import { useLabStatus } from './status';

const LAB = 'ch10/lab1_sum';
const fmtInt = (n: number) => n.toLocaleString('en-US');

// Identifiers coloured by the memory they live in, as everywhere in the course.
const SUM_TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', output_ref: 'global', in: 'global', out: 'global',
  partialSums_s: 'shared', input_s: 'shared', warpSums: 'shared',
  partialSum: 'reg', segment: 'reg', i: 'reg', t: 'reg', c: 'reg', idx: 'reg', sum: 'reg',
};

/** The Metal kernel in a source file whose body contains `needle`. */
function metalKernelWith(src: string, needle: string): FunctionSource | null {
  for (const m of src.matchAll(/\bkernel\s+void\s+(\w+)\s*\(/g)) {
    const f = functionSource(src, m[1]);
    if (f && f.text.includes(needle)) return f;
  }
  return null;
}
/** 1-based lines of `text` that contain any of the needles. */
const linesWith = (text: string, needles: string[]) =>
  text.split('\n').flatMap((l, i) => (needles.some((n) => l.includes(n)) ? [i + 1] : []));

function CommitListing() {
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  const status = useLabStatus(LAB);
  const k = cu ? functionSource(cu, 'CoarsenedSumReductionKernel') : null;
  const host = cu ? functionSource(cu, 'launchCoarsenedSum') : null;
  const m = metal ? metalKernelWith(metal, 'atomic_fetch_add_explicit') : null;
  const zeroing = host ? linesWith(host.text, ['emset']) : [];
  if (!k)
    return (
      <p className="muted">
        The kernel is loaded from <code>labs/{LAB}/solution/kernel.cu</code> once lab 10.1 is in place.
      </p>
    );
  const tabs = [
    { label: 'CUDA kernel', code: k.text, tiers: SUM_TIERS, status: status.cu, highlight: linesWith(k.text, ['atomic_ref', 'fetch_add', 'atomicAdd']) },
    ...(host ? [{ label: 'CUDA host', code: host.text, tiers: SUM_TIERS, status: status.cu, highlight: zeroing }] : []),
    ...(m ? [{ label: 'Metal kernel', code: m.text, tiers: SUM_TIERS, status: status.metal, highlight: linesWith(m.text, ['atomic_fetch_add_explicit']) }] : []),
  ];
  return (
    <Code
      tabs={tabs}
      caption={
        <>
          Lab 10.1's reference solution: <code>CoarsenedSumReductionKernel</code> (lines {k.firstLine} to {k.lastLine} of{' '}
          <code>kernel.cu</code>){host ? <>, its host function <code>launchCoarsenedSum</code></> : null}
          {m ? <> and the Metal kernel that commits with <code>atomic_fetch_add_explicit</code></> : null}, from{' '}
          <code>labs/{LAB}/solution/</code>. The marked lines are the commit{zeroing.length ? ' and the zeroing of the total' : ''}. With a coarsening
          factor of 1 the kernel is Fig. 10.18 made safe for any length; section 10.10 explains the factor. Try the lab before
          you study this.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- interactive A: segment calculator
const BLOCK_SIZES = [8, 32, 64, 128, 256, 512, 1024];
const COARSE = [1, 2, 4, 8];

function SegmentCalculator() {
  const [N, setN] = useState(1_000_000);
  const [text, setText] = useState('1000000');
  const [block, setBlock] = useState(1024);
  const [coarse, setCoarse] = useState(1);
  const seg = segments(N, block, coarse);
  const [b, setB] = useState(seg.blocks - 1);
  const [t, setT] = useState(0);
  const [guard, setGuard] = useState(true);
  const [k, setK] = useState(0); // loads shown so far

  const bb = Math.min(b, Math.max(0, seg.blocks - 1));
  const tt = Math.min(t, block - 1);
  const view = blockView(N, block, coarse, bb);
  const loads = threadLoads(N, block, coarse, bb, tt);
  const last = blockView(N, block, coarse, seg.blocks - 1);
  useEffect(() => setK(0), [N, block, coarse, bb, tt, guard]);

  const commitN = (s: string) => {
    setText(s);
    const v = Math.floor(Number(s));
    if (isFinite(v) && v >= 1 && v <= 1e9) {
      setN(v);
      setB(segments(v, block, coarse).blocks - 1);
    }
  };
  const setShape = (nb: number, nc: number) => {
    setBlock(nb);
    setCoarse(nc);
    setB(segments(N, nb, nc).blocks - 1);
    setT(0);
  };

  const cur = k > 0 ? loads[k - 1] : null;
  const say = (l: (typeof loads)[number]) =>
    `Load ${l.k}: index = segment + t + ${l.k}·blockDim = ${fmtInt(view.start)} + ${tt} + ${l.k}·${block} = ${fmtInt(l.index)}. ` +
    (l.inRange
      ? `That is below N = ${fmtInt(N)}, so the thread adds input[${fmtInt(l.index)}].`
      : guard
        ? `That is past the end (N = ${fmtInt(N)}): the guard makes the thread add 0 instead.`
        : `That is past the end (N = ${fmtInt(N)}) and nothing stops the read: out of bounds.`);

  // Drawing: the whole input on top, the selected segment's 2C rows of blockDim slots below.
  const W = 680;
  const X0 = 20;
  const total = seg.blocks * seg.perBlock; // slots up to the end of the last segment
  const sx = (i: number) => X0 + (i / total) * W;
  const ROW = 22;
  const zx = (j: number) => X0 + (j / block) * W;
  const oob = !guard && view.pastEnd > 0;

  return (
    <Bench
      title="Segment calculator"
      task="Pick a length that is not a multiple of the segment. Find the last block, then a thread in it with nothing to load."
      foot={
        <>
          Top bar: the whole input, one segment per block, drawn to scale up to the end of the last segment. Bottom: the selected
          block's segment as 2·C rows of <code>blockDim</code> slots, one row per load; the tick marks the selected thread.
          Hatched slots lie past N.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          N
          <input type="number" min={1} value={text} onChange={(e) => commitN(e.target.value)} style={{ width: '8.5rem' }} aria-label="N" />
        </label>
        <label className="field">
          Threads per block
          <select value={block} onChange={(e) => setShape(+e.target.value, coarse)} aria-label="threads per block">
            {BLOCK_SIZES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="field">
          Coarsening factor C
          <select value={coarse} onChange={(e) => setShape(block, +e.target.value)} aria-label="coarsening factor">
            {COARSE.map((s) => <option key={s} value={s}>{s}{s === 1 ? ' (this section)' : ''}</option>)}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={guard} onChange={(e) => setGuard(e.target.checked)} /> guard loads (exercise 5)
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ minWidth: '16rem' }}>
          <span>Block <b data-testid="block">{fmtInt(bb)}</b> of {fmtInt(seg.blocks)} (0 to {fmtInt(seg.blocks - 1)})</span>
          <input type="range" min={0} max={Math.max(0, seg.blocks - 1)} value={bb} onChange={(e) => setB(+e.target.value)} aria-label="block index" />
        </label>
        <label className="field" style={{ minWidth: '16rem' }}>
          <span>Thread <b data-testid="thread">{tt}</b> of {block} (0 to {block - 1})</span>
          <input type="range" min={0} max={block - 1} value={tt} onChange={(e) => setT(+e.target.value)} aria-label="thread index" />
        </label>
        <button onClick={() => setB(seg.blocks - 1)}>Last block</button>
        <button onClick={() => setT(Math.min(block - 1, last.busyThreads))} disabled={bb !== seg.blocks - 1 || last.idleThreads === 0}>
          First idle thread
        </button>
      </div>

      <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${W + 2 * X0} ${100 + 2 * coarse * ROW}`} role="img" aria-label="The input split into segments" style={{ width: '100%', minWidth: 540, display: 'block' }}>
        <defs>
          <pattern id="seg-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink-3)" strokeWidth="1.6" />
          </pattern>
          <pattern id="seg-hatch-bad" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--fail)" strokeWidth="1.8" />
          </pattern>
        </defs>
        <text x={X0} y={14} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>input (global memory), {fmtInt(N)} elements, {fmtInt(seg.blocks)} segments</text>
        <rect x={sx(0)} y={22} width={sx(N) - sx(0)} height={20} fill="var(--global-wash)" stroke="var(--global)" />
        {total > N ? <rect x={sx(N)} y={22} width={sx(total) - sx(N)} height={20} fill={oob ? 'url(#seg-hatch-bad)' : 'url(#seg-hatch)'} stroke="var(--rule)" /> : null}
        {seg.blocks <= 128
          ? Array.from({ length: seg.blocks - 1 }, (_, i) => (
              <line key={i} x1={sx((i + 1) * seg.perBlock)} x2={sx((i + 1) * seg.perBlock)} y1={22} y2={42} stroke="var(--global)" strokeWidth="0.8" />
            ))
          : null}
        <rect x={sx(view.start)} y={18} width={Math.max(2, sx(view.end) - sx(view.start))} height={28} fill="none" stroke="var(--ink)" strokeWidth="2" />
        <text x={Math.min(W + X0, sx(view.start))} y={60} fontSize="11.5" textAnchor={sx(view.start) > W * 0.7 ? 'end' : 'start'}>
          block {fmtInt(bb)}: elements {fmtInt(view.start)} to {fmtInt(view.end - 1)}
        </text>

        <text x={X0} y={88} fontSize="12" fontWeight="600">block {fmtInt(bb)}'s segment, one row per load (k = 0 .. {2 * coarse - 1})</text>
        {Array.from({ length: 2 * coarse }, (_, r) => {
          const rowStart = view.start + r * block;
          const realHere = Math.max(0, Math.min(block, N - rowStart));
          const y = 96 + r * ROW;
          return (
            <g key={r} data-row={r}>
              <rect x={zx(0)} y={y} width={zx(realHere) - zx(0)} height={ROW - 6} fill="var(--global-wash)" stroke="var(--global)" />
              {realHere < block ? (
                <rect x={zx(realHere)} y={y} width={zx(block) - zx(realHere)} height={ROW - 6} fill={oob ? 'url(#seg-hatch-bad)' : 'url(#seg-hatch)'} stroke="var(--rule)" />
              ) : null}
              <line x1={zx(tt + 0.5)} x2={zx(tt + 0.5)} y1={y - 2} y2={y + ROW - 4}
                stroke={cur && cur.k === r ? (cur.inRange ? 'var(--ok)' : guard ? 'var(--ink)' : 'var(--fail)') : 'var(--ink)'} strokeWidth={cur && cur.k === r ? 3 : 1.6} />
            </g>
          );
        })}
      </svg>
      </div>

      <div className="row" style={{ margin: '0.6rem 0 0.8rem' }}>
        <button onClick={() => setK(0)} disabled={k === 0}>Restart</button>
        <button className="primary" onClick={() => setK(Math.min(loads.length, k + 1))} disabled={k >= loads.length}>Step</button>
        <button onClick={() => setK(loads.length)} disabled={k >= loads.length}>All loads</button>
        <span className="muted">load {k} of {loads.length}</span>
      </div>
      <div className={'narrate' + (cur && !cur.inRange && !guard ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? say(cur) : <>Thread {tt} of block {fmtInt(bb)} has loaded nothing yet. Its segment starts at 2·C·blockDim·b = 2·{coarse}·{block}·{fmtInt(bb)} = {fmtInt(view.start)}. Press Step.</>}
      </div>

      <div className="cols">
        <div className="panel">
          <table className="data" style={{ margin: 0 }} aria-label="This thread's loads">
            <thead><tr><th>load k</th><th>index</th><th>below N?</th><th>adds</th></tr></thead>
            <tbody>
              {loads.map((l) => (
                <tr key={l.k} data-load={l.k} data-index={l.index} data-in={l.inRange ? 1 : 0} style={{ opacity: l.k < k ? 1 : 0.45 }}>
                  <td>{l.k}</td>
                  <td className="mono">{fmtInt(l.index)}</td>
                  <td>{l.inRange ? 'yes' : 'no'}</td>
                  <td>{l.inRange ? <>input[{fmtInt(l.index)}]</> : guard ? '0 (guarded)' : <span className="t-fail">out of bounds</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel" data-testid="facts">
          <p style={{ margin: 0 }}>
            Blocks: ceil({fmtInt(N)} / (2·{coarse}·{block})) = <b data-testid="blocks">{fmtInt(seg.blocks)}</b>.
          </p>
          <p style={{ margin: '0.4rem 0 0' }}>
            Block {fmtInt(bb)} holds <b data-testid="real">{fmtInt(view.real)}</b> real elements of {fmtInt(seg.perBlock)};{' '}
            {view.idleThreads > 0 ? <>{fmtInt(view.idleThreads)} of its threads have nothing to load.</> : <>every thread has work.</>}
          </p>
          <p style={{ margin: '0.4rem 0 0' }}>
            The last block holds <b data-testid="last-real">{fmtInt(seg.lastReal)}</b> real elements
            {seg.pastEnd > 0 ? <>; {fmtInt(seg.pastEnd)} of its loads lie past N.</> : <>: N is a multiple of the segment.</>}
          </p>
          {seg.pastEnd > 0 ? (
            <p style={{ margin: '0.5rem 0 0' }} data-testid="verdict">
              {guard ? (
                <span className="chip ok">guarded: idle threads still take part and add 0</span>
              ) : (
                <span className="chip fail">no guard: the last block reads {fmtInt(seg.pastEnd)} elements past the end</span>
              )}
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive B: commit order
const LENGTHS = [4096, 65536, 1 << 20];
const COMMIT_BLOCKS = [128, 256, 512, 1024];
const DATA_SEED = 2026;

function CommitExplorer() {
  const [n, setNRaw] = useState(65536);
  const [block, setBlockRaw] = useState(256);
  const [design, setDesignRaw] = useState<Design>('atomic');
  const [runs, setRuns] = useState(1);
  const values = useMemo(() => logits(n, DATA_SEED), [n]);
  const ref = useMemo(() => reference(values), [values]);
  const sums = useMemo(() => blockSums(values, block), [values, block]);
  const run = runOf(sums, design, runs - 1);
  const hist = useMemo(() => history(sums, design, runs), [sums, design, runs]);
  // A new data set, block size or design starts a new history.
  const setN = (v: number) => { setNRaw(v); setRuns(1); };
  const setBlock = (v: number) => { setBlockRaw(v); setRuns(1); };
  const setDesign = (v: Design) => { setDesignRaw(v); setRuns(1); };

  const B = sums.length;
  const shown = design === 'atomic' ? (B <= 8 ? run.order.map((_, i) => i) : [0, 1, 2, 3, 4, -1, B - 2, B - 1]) : [];
  const nearest = f32(ref);
  const counts = hist.distinct.map((h) => ({ h, n: hist.totals.filter((x) => f32hex(x) === h).length, v: hist.totals.find((x) => f32hex(x) === h)! }));

  return (
    <Bench
      title="Blocks commit in any order"
      task="Press Run again several times under each design. Then try 4,096 values with 1,024-thread blocks."
      foot={
        <>
          Logit-like float32 values (normal, mean −1, deviation 2.5; seed {DATA_SEED}). Every block sum is computed once, in a
          fixed tree; only the order of the commits changes from run to run. The reference is the float64 sum of the same
          float32 values.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Values
          <select value={n} onChange={(e) => setN(+e.target.value)} aria-label="values">
            {LENGTHS.map((s) => <option key={s} value={s}>{fmtInt(s)}</option>)}
          </select>
        </label>
        <label className="field">
          Threads per block
          <select value={block} onChange={(e) => setBlock(+e.target.value)} aria-label="threads per block">
            {COMMIT_BLOCKS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label="Design">
          {(['atomic', 'two-pass'] as Design[]).map((d) => (
            <button key={d} aria-pressed={design === d} onClick={() => setDesign(d)}>{DESIGN_LABEL[d]}</button>
          ))}
        </span>
        <button className="primary" onClick={() => setRuns(runs + 1)}>Run again</button>
        <button onClick={() => setRuns(1)} disabled={runs === 1}>Clear history</button>
      </div>

      <p style={{ margin: '0 0 0.7rem' }}>
        {fmtInt(B)} blocks of {block} threads, {fmtInt(2 * block)} values each.{' '}
        {design === 'atomic'
          ? <>Run {runs}: the leaders reach the total in this order.</>
          : <>Run {runs}: each leader writes its sum to <code>partial[blockIdx.x]</code>; one block then reduces the {fmtInt(B)} partials in a fixed tree.</>}
      </p>

      {design === 'atomic' ? (
        <table className="data" style={{ margin: '0 0 0.8rem', background: 'var(--page)' }} aria-label="Commits of this run">
          <thead><tr><th style={{ paddingLeft: '0.6rem' }}>commit</th><th>block</th><th>block sum</th><th>total after the add</th></tr></thead>
          <tbody>
            {shown.map((i, j) =>
              i < 0 ? (
                <tr key={'gap' + j}><td colSpan={4} className="muted" style={{ paddingLeft: '0.6rem' }}>… {fmtInt(B - 7)} more commits …</td></tr>
              ) : (
                <tr key={i} data-commit={i}>
                  <td style={{ paddingLeft: '0.6rem' }}>{i + 1}</td>
                  <td>{run.order[i]}</td>
                  <td className="mono t-reg">{f32short(sums[run.order[i]])}</td>
                  <td className="mono t-global">{f32short(run.running[i])}</td>
                </tr>
              ),
            )}
          </tbody>
        </table>
      ) : null}

      <div className="cols">
        <div className="panel" data-testid="final">
          <p style={{ margin: 0 }}>
            Total of run {runs}: <b className="mono" data-testid="total">{f32short(run.total)}</b>{' '}
            <span className="mono" data-testid="total-hex">{f32hex(run.total)}</span>
          </p>
          <p style={{ margin: '0.4rem 0 0' }}>
            float64 reference: <span className="mono">{f64short(ref)}</span>; the nearest float32 is{' '}
            <span className="mono">{f32hex(nearest)}</span>, {ulpDistance(run.total, nearest)} float32 step
            {ulpDistance(run.total, nearest) === 1 ? '' : 's'} from this total.
          </p>
        </div>
        <div className="panel" data-testid="history" data-runs={runs} data-distinct={hist.distinct.length}>
          <p style={{ margin: 0 }}>
            <b>{runs}</b> run{runs === 1 ? '' : 's'} so far, <b>{hist.distinct.length}</b> distinct total{hist.distinct.length === 1 ? '' : 's'}:
          </p>
          <ul style={{ margin: '0.4rem 0 0', paddingLeft: '1.1rem' }}>
            {counts.map((c) => (
              <li key={c.h} className="mono">
                {c.h} ({f32short(c.v)}) × {c.n}
              </li>
            ))}
          </ul>
        </div>
      </div>
      {runs >= 4 ? (
        <p style={{ margin: '0.8rem 0 0' }} data-testid="verdict">
          {hist.distinct.length > 1 ? (
            <span className="chip fail">the same data gave {hist.distinct.length} different totals</span>
          ) : design === 'two-pass' ? (
            <span className="chip ok">the same bits every run</span>
          ) : (
            <span className="chip ok">so far the same bits every run</span>
          )}{' '}
          {design === 'atomic' && B <= 2
            ? 'With two blocks the order cannot matter: a + b and b + a round the same way. Float addition is commutative; it is only not associative.'
            : null}
        </p>
      ) : null}
    </Bench>
  );
}

function MeasuredDistinct({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = Object.keys(reports).flatMap((b) => {
    const row = benchRows(reports[b], LAB).find((r) => r.name === 'determinism');
    const distinct = num(row?.distinct);
    return row && distinct != null ? [{ b, device: String(reports[b].device.name ?? b), distinct, runs: num(row.runs), n: num(row.n) }] : [];
  });
  if (!found.length)
    return (
      <p>
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 10.1 --solution --bench</code> on a machine
        with a GPU, then reload the page: this box will show how many different totals your device produced.
      </p>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b} data-backend={f.b} data-distinct={f.distinct}>
          On <b>{f.device}</b> ({f.b}): <b>{f.distinct}</b> distinct result{f.distinct === 1 ? '' : 's'}
          {f.runs ? <> in {f.runs} runs</> : null}
          {f.n ? <> on {fmtInt(f.n)} floats</> : null}.{' '}
          {f.b === 'emu'
            ? 'The emulator runs threads one at a time in a fixed order, so its blocks always commit in the same order: it can only ever show 1.'
            : f.b === 'race'
              ? 'The race backend runs blocks as operating-system threads, so its order changes with the scheduler, not with a GPU.'
              : f.distinct > 1
                ? predicted <= 1
                  ? 'You expected the atomic kernel to be repeatable. It is not: the blocks finished in different orders.'
                  : 'The atomic commit is not repeatable on this device.'
                : 'Every run gave the same bits here. That is not a promise: nothing in the programming model fixes the order.'}
        </p>
      ))}
      <p className="muted">These are measurements from your machine. Another device, size or load can give a different count.</p>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS109: Record<string, Question> = {
  'ch10/10.9/q-lastblock': {
    kind: 'numeric',
    id: 'ch10/10.9/q-lastblock',
    prompt: (
      <>
        N = 1,000,000 with 1024-thread blocks, each block covering 2048 elements. How many real elements does the last block
        hold?
      </>
    ),
    answer: 576,
    explain: (
      <p>
        ceil(1,000,000 / 2048) = 489 blocks. The first 488 cover 488 × 2048 = 999,424 elements, so the last block starts at
        999,424 and holds 1,000,000 − 999,424 = 576. Threads 0 to 575 load one real element each (their first load); threads
        576 to 1023 load nothing real and, with the guard, add 0. All 1024 still take part in the shuffles and the barrier.
      </p>
    ),
    source: 'ch. 10 §10.9, pp. 244–245; exercise 5, p. 250',
  },
  'ch10/10.9/q-twice': {
    kind: 'choice',
    id: 'ch10/10.9/q-twice',
    prompt: (
      <>
        The host launches the kernel of Fig. 10.18 twice in a row on the same input and the same <code>output</code> buffer,
        without touching <code>*output</code> in between. What does <code>*output</code> hold after the second launch?
      </>
    ),
    options: [
      { label: 'The sum: the second launch overwrites the first.', why: 'Nothing in the kernel writes the total; every block only adds to it.' },
      { label: 'Twice the sum: each launch adds onto what is there.', correct: true },
      { label: 'Undefined: two launches race on the same buffer.', why: 'Launches on one stream run one after the other; the second starts from what the first left.' },
      { label: 'The sum, because fetch_add with relaxed order starts from 0.', why: 'An atomic add starts from whatever is in memory.' },
    ],
    explain: (
      <p>
        Each leader does <code>total += blockSum</code>. The first launch leaves the sum in <code>*output</code>; the second
        adds every block sum onto it again. The host must set <code>*output</code> to the identity, 0, before every launch.
        The book does not show host code for this kernel, so it does not say so; lab 10.1's host function does it.
      </p>
    ),
    source: 'ch. 10 §10.9, p. 245 (Fig. 10.18); the zeroing is our inference, recorded in notes/ch10.md',
  },
  'ch10/10.9/q-max-commit': {
    kind: 'choice',
    id: 'ch10/10.9/q-max-commit',
    prompt: (
      <>
        Exercise 4 turns the sum into a max. You need a float max across blocks that builds with any recent CUDA toolkit
        and also works on Metal. Which design?
      </>
    ),
    options: [
      { label: <><code>atomicMax(output, partialMax)</code> on a <code>float*</code>.</>, why: 'That overload exists on HIP only. CUDA\'s legacy atomicMax takes integer types.' },
      { label: <><code>cuda::atomic_ref&lt;float&gt;::fetch_max</code> on CUDA, and <code>atomic_fetch_max_explicit</code> on an <code>atomic_float</code> on Metal.</>, why: 'Current CCCL accepts float for fetch_max, but the first toolkit that does is not documented; Metal lists atomic max for integer types only.' },
      { label: 'Each block writes its maximum to an array; a second, single-block pass reduces it.', correct: true },
      { label: <>Use <code>fetch_add</code> on the negated values, then negate the total at the end.</>, why: 'A sum of negated values is minus the sum, not a maximum.' },
    ],
    explain: (
      <p>
        The book's second design needs no atomic at all, so it works wherever a kernel can write an array. The atomic
        routes each fail somewhere: no legacy float <code>atomicMax</code> on CUDA, an unknown first toolkit for{' '}
        <code>fetch_max</code> on float, no float atomic max on Metal. A compare-and-swap loop also works (on Metal over{' '}
        <code>atomic_float</code>), at the price of retries under contention.
      </p>
    ),
    source: 'ch. 10 §10.9, pp. 245–246, exercise 4 (p. 250); notes/today/ch10-facts.md A7, B3, C2',
  },
  'ch10/10.9/q-reproducible': {
    kind: 'choice',
    id: 'ch10/10.9/q-reproducible',
    prompt: (
      <>
        A regression test compares the sum of a logits tensor bit for bit across runs on the same GPU and build. Which
        multi-block design passes it?
      </>
    ),
    options: [
      { label: 'The atomic commit with memory_order_relaxed.', why: 'The commit order follows the order blocks finish, which can change between runs.' },
      { label: <>The atomic commit with <code>memory_order_seq_cst</code>, which puts all the adds in one total order.</>, why: 'A memory order constrains how other memory operations are seen around the atomic. It does not choose which block adds first, so the total order can differ from run to run.' },
      { label: 'Per-block partials, then a second pass in a fixed order.', correct: true },
      { label: 'Either, as long as every block reduces its segment with the same tree on every run.', why: 'The block trees are fixed already. What varies is the order in which the block sums meet, and float rounding makes that order visible.' },
    ],
    explain: (
      <p>
        Every block sum is the same on every run: inside a block the order is fixed by the code. What changes is only the
        order in which block sums meet. Write them to an array indexed by block and reduce that array in a fixed tree,
        and nothing depends on timing. That is how <code>cub::DeviceReduce</code>'s default mode is documented to work.
      </p>
    ),
    source: 'ch. 10 §10.9, pp. 245–246; notes/today/ch10-facts.md A13',
  },
  'ch10/10.9/q-argmax-pack': {
    kind: 'choice',
    id: 'ch10/10.9/q-argmax-pack',
    prompt: (
      <>
        To commit an argmax across blocks with one 64-bit <code>atomicMax(unsigned long long*, …)</code>, you pack each
        candidate into a key: the float in the high 32 bits, the index in the low 32. What must the packing preserve so that
        the largest key is the largest value, with ties going to the lowest index?
      </>
    ),
    options: [
      {
        label: 'The raw float bits in the high half and the index in the low half.',
        why: 'Read as unsigned, a negative float has its sign bit set, so it compares above every positive one, and more negative values compare larger.',
      },
      {
        label: 'Float bits made order-preserving as unsigned in the high half; 0xFFFFFFFF − index in the low half.',
        correct: true,
      },
      { label: 'The index in the high half, so that ties are settled first, and the float bits in the low half.', why: 'Then the largest index wins, whatever the values.' },
      {
        label: 'An order-preserving float transform in the high half and the plain index in the low half.',
        why: 'Values right; ties wrong: on equal values the larger key has the larger index.',
      },
    ],
    explain: (
      <p>
        The atomic compares keys as unsigned integers, so the high half must order like the floats, and the low half must
        break ties the way you want. The transform flips the sign bit of a non-negative float and every bit of a
        negative one; then larger floats give larger unsigned values. Storing{' '}
        <code>0xFFFFFFFF − index</code> makes the lower index win. The packing is our technique, not a vendor recipe. On
        Metal a 64-bit atomic max needs the Apple9 family (Apple8 on macOS) and returns nothing; elsewhere the portable route
        is again two passes.
      </p>
    ),
    source: 'ours (not in the book); availability from notes/today/ch10-facts.md A7, C2 and infra-facts C1',
  },
};

export function S109() {
  return (
    <>
      <p>
        Every kernel so far has been one block. The reason is the barrier: <code>__syncthreads()</code> coordinates the
        threads of one block and no others, and a block holds at most 1024 threads, so one block reduces at most 2048
        elements.
      </p>
      <Source section="10.9" pages="244" />
      <p>
        Nor is there a good barrier across blocks. So blocks have to work independently: each reduces its own part of the
        input, no block ever waits for another, and the parts are combined at the end. That raises a question the
        one-block kernels never had to answer: in what order do the parts meet?
      </p>
      <Today checked="2026-11-20 (ch10-facts A8, C2)">
        <p>
          CUDA and HIP have a grid-wide <code>sync()</code>, but only under a cooperative launch whose whole grid fits on
          the device. Metal has none.
        </p>
      </Today>
      <p>
        That limit matters for the reductions you care about. The softmax over a model's output row sums one term per
        vocabulary entry, and vocabularies run from 50,257 entries (GPT-2) to 201,088 (gpt-oss-20b). A row that long needs
        either many blocks, which is this section, or one block in which every thread does a lot of serial work, which is
        section 10.10.
      </p>
      <Beyond>
        <p>
          The softmax framing is ours. Vocabulary sizes from public model configs, read on 2026-11-20
          (notes/today/ch09-facts.md, D7): {VOCABS.map((v) => `${v.name} ${fmtInt(v.size)}`).join(', ')}.
        </p>
      </Beyond>

      <h2>One segment per block</h2>
      <p>
        Cut the input into segments of <code>2·blockDim.x</code> elements. Block <code>b</code> takes the segment that starts
        at <code>2·blockDim.x·b</code>: with 1024 threads, block 0 starts at 0, block 1 at 2048, block 2 at 4096. Inside its
        segment, a block runs any of the single-block reductions you already have; the book reuses the two-stage warp
        reduction of section 10.8. That leaves one number per block. Its leader, thread 0, adds it into the single total
        with an atomic:
      </p>
      <Source section="10.9" pages="244–245" figs="Figs. 10.17, 10.18" />
      <p>
        <code>cuda::atomic_ref&lt;float, cuda::thread_scope_device&gt; output_ref(*output);</code> then{' '}
        <code>output_ref.fetch_add(partialSum, cuda::memory_order_relaxed);</code>. This is the atomic reference from chapter
        9, at device scope because every block adds to the same location, with relaxed order because nothing else is
        published through this add. Under HIP, lab 10.1 writes the legacy <code>atomicAdd(output, partialSum)</code>{' '}
        instead, which is also relaxed and device-wide: there <code>&lt;cuda/atomic&gt;</code> comes from libhipcxx, which
        the course's HIP build links only where ROCm provides it, so the lab does not rely on it.
      </p>
      <Today checked="2026-11-20 (ch10-facts A6; infra-facts A1, B1, B2)">
        <p>
          The book's <code>output_ref(*output)</code> is right: the constructor takes a reference. The current CUDA guide's own
          example passes the pointer. Float atomic adds to global memory flush denormals to zero.
        </p>
      </Today>
      <p>
        Two things the listing leaves to you. First, <code>*output</code> must hold 0 before the launch: the kernel only ever
        adds to it, and the book shows no host code. Second, the printed body never uses its parameter <code>N</code>; it
        assumes N is a multiple of <code>2·blockDim.x</code>. Exercise 5 removes that assumption: launch{' '}
        <code>ceil(N / (2·blockDim.x))</code> blocks and guard every load, so that an index at or past N contributes 0. Lab
        10.1's kernel is written that way.
      </p>
      <Source section="10.9" pages="245, 250" figs="exercise 5" />
      <CommitListing />

      <p>
        In the calculator, the default is the case from the question below. Move to the last block and step through a
        thread's loads, then untick the guard.
      </p>
      <Simplification>
        <p>Index arithmetic only: no values, no timing. The drawing is to scale, so with many segments their borders merge.</p>
      </Simplification>
      <SegmentCalculator />
      <Ask q={questionsS109['ch10/10.9/q-lastblock']} label="Work it out" />
      <p>
        Notice what the out-of-range threads do. They do not return early. They load nothing, hold 0, and still join the
        warp shuffles and the barrier, because those require every lane and every thread of the block to arrive.
      </p>
      <Ask q={questionsS109['ch10/10.9/q-twice']} />

      <h2>Blocks finish in any order</h2>
      <p>
        An atomic add guarantees that two leaders never lose each other's update. It does not decide which leader goes
        first. Blocks finish in whatever order the hardware schedules them, so the block sums meet the total in an order that
        nobody controls. That is only correct if the operator is associative and commutative.
      </p>
      <Source section="10.9" pages="245" />
      <p>
        Integer addition qualifies. Float addition is commutative but, after rounding, not associative (section 10.2): adding
        the same block sums in another order can round differently. The result is still a good sum; it is just not the same
        bits every time. Watch it happen, then try it yourself.
      </p>
      <CommitOrderScene />
      <p>
        Now on more data. Each press of Run again commits the same block sums in a new order. Count the distinct totals under
        each design, then find a size at which the atomic design cannot disagree with itself, and say why.
      </p>
      <Simplification>
        <p>
          A commit order here is a seeded shuffle, not a model of timing: it says nothing about which orders a GPU produces or
          how often. Each block sum is computed once, in a fixed tree, as on a GPU.
        </p>
      </Simplification>
      <CommitExplorer />
      <Predict
        id="ch10/10.9/p-distinct"
        prompt={
          <>
            Lab 10.1 runs its atomic multi-block kernel 20 times on the same 2<sup>24</sup> floats (fewer where memory is
            short). How many distinct results, as bit patterns, do you expect on <em>your</em> GPU?
          </>
        }
        unit="distinct results"
      >
        {(v) => <MeasuredDistinct predicted={v} />}
      </Predict>

      <h2>Without atomics: a second pass</h2>
      <p>
        The other design commits nothing. Each leader writes its block sum to an array, at index <code>blockIdx.x</code>.
        Then a second kernel with a single block reduces that array, or the host copies it back and finishes on the CPU. The
        book offers this for systems where atomics are missing or too expensive, and notes the side benefit: you choose the
        order of the final additions.
      </p>
      <Source section="10.9" pages="245–246" />
      <p>
        That side benefit is the main one when results must not change between runs. Every block sum is already fixed by the
        code; only the meeting order varied, and now it is fixed too. NVIDIA's own library reduction makes this its
        default, and offers the atomic design only on request: the book's two designs are CUB's two modes.
      </p>
      <Today checked="2026-11-20 (ch10-facts A13)">
        <p>
          <code>cub::DeviceReduce</code>: run-to-run deterministic by default (a fixed tree, no atomics); atomic only as{' '}
          <code>not_guaranteed</code>.
        </p>
      </Today>
      <Ask q={questionsS109['ch10/10.9/q-reproducible']} />

      <h2>Max and argmax across blocks</h2>
      <p>
        Exercise 4 asks for a max instead of a sum. Inside a block nothing changes but the operator. The commit is where the
        backends part ways, and argmax, the step behind greedy decoding, is harder still because the value and its index have
        to move together.
      </p>
      <Source section="10.12" pages="250" figs="exercise 4" />
      <table className="data wide" style={{ maxWidth: '58rem' }} aria-label="Committing across blocks">
        <thead>
          <tr><th>Commit</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Float sum</td>
            <td><code>atomic_ref&lt;float&gt;::fetch_add</code> or <code>atomicAdd(float*)</code></td>
            <td>same calls (<code>atomic_ref</code> through libhipcxx)</td>
            <td><code>atomic_fetch_add_explicit</code> on <code>device atomic_float*</code>, Metal 3, Apple7: every Apple silicon Mac</td>
          </tr>
          <tr>
            <td>Float max</td>
            <td>no legacy <code>atomicMax(float*)</code>; <code>fetch_max</code> on float in current CCCL, first toolkit unknown; or a compare-and-swap loop</td>
            <td><code>atomicMax(float*)</code> exists (HIP only); libhipcxx's <code>fetch_max</code> is a compare-and-swap loop</td>
            <td>no float atomic max; a compare-and-swap loop on <code>atomic_float</code>, or a second pass</td>
          </tr>
          <tr>
            <td>(value, index) for argmax</td>
            <td>a packed 64-bit key with <code>atomicMax(unsigned long long*)</code>, or a second pass</td>
            <td>same</td>
            <td>64-bit atomic max needs Apple9 (Apple8 on macOS) and returns nothing; otherwise a second pass</td>
          </tr>
        </tbody>
      </table>
      <p>
        The route that works everywhere is two passes, and it is the one llama.cpp's Metal backend takes for argmax: one
        dispatch writes a partial (value, index) pair per threadgroup, a second reduces them. The other way out is to avoid
        committing across blocks at all, by giving each row one block. That is what the chapter's milestone does (lab 10.3),
        and section 10.10 shows what it costs.
      </p>
      <Beyond>
        <p>
          The table and the llama.cpp route are ours, from vendor documentation and project source read on 2026-11-20
          (notes/today/ch10-facts.md A7, B3, C2, D2; infra-facts C1). Nothing here has run on CUDA or AMD hardware.
        </p>
      </Beyond>
      <Ask q={questionsS109['ch10/10.9/q-max-commit']} />
      <Ask q={questionsS109['ch10/10.9/q-argmax-pack']} />
      <p>
        Lab 10.1 has the any-length kernel, the determinism count and the coarsening sweep; lab 10.3 builds the softmax and
        argmax rows. Both are on the <a href={href('ch10', 'labs')}>labs page</a>.
      </p>
    </>
  );
}
