import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BLOCK_SIZES,
  KERNELS,
  KERNEL_LABEL,
  blockCosts,
  isPow2,
  ones,
  twoStage,
  type Costs,
  type Kernel,
} from '../../sim/shuffle';
import { functionSource } from '../../sim/toolchain';
import { widthChoices } from '../../sim/warps';
import { MIN_W, VIEW_W } from './ShuffleScene';
import { STATUS } from './status';

import { NoteGroup } from './NoteGroup';
const LAB = 'ch10/lab1_sum';
const missing = (s: string | null) => !s || s.startsWith('// could not load');
const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', row: 'global', partialSums_s: 'shared',
  partialSum: 'reg', s: 'reg', t: 'reg', warps: 'reg', lane: 'reg', sg: 'reg', groups: 'reg', width: 'reg',
};

function useWidths() {
  const { reports } = useReports();
  return useMemo(
    () => widthChoices(reports as Reports).filter((c) => isPow2(c.width) && c.width >= 32 && c.width <= 64),
    [reports],
  );
}

// ---------------------------------------------------------------- the listing
function TwoStageListing() {
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  const k = !missing(cu) ? functionSource(cu!, 'TwoStageWarpLevelSumReductionKernel') : null;
  const m = !missing(metal) ? functionSource(metal!, 'two_stage_warp_sum') : null;
  if (!k)
    return (
      <p className="muted">
        The kernel is loaded from <code>labs/{LAB}/solution/kernel.cu</code> once lab 10.1 is on this machine.
      </p>
    );
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: k.text, tiers: TIERS, status: STATUS.cu },
        ...(m ? [{ label: 'Metal', code: m.text, tiers: TIERS, status: STATUS.metal }] : []),
      ]}
      caption={
        <>
          <code>TwoStageWarpLevelSumReductionKernel</code> from lab 10.1's reference solution (
          <code>labs/{LAB}/solution/kernel.cu</code>, lines {k.firstLine} to {k.lastLine}), with its Metal counterpart. It
          calls the portable <code>warp_reduce</code> of section 10.7, so the same file is written to compile for HIP;
          that is unverified, since nothing here has been built with ROCm.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- explorer: count the costs
const KIND_FILL: Record<string, string> = { load: 'var(--global)', shared: 'var(--shared)', shuffle: 'var(--reg)' };
const KIND_WASH: Record<string, string> = { load: 'var(--global-wash)', shared: 'var(--shared-wash)', shuffle: 'var(--reg-wash)' };
const SHORT: Record<Kernel, string> = { shared: 'Fig. 10.10', hybrid: 'Fig. 10.14', twostage: 'Fig. 10.16' };

function Timeline({ costs }: { costs: Costs[] }) {
  const LABEL = 92;
  const most = Math.max(...costs.map((c) => c.steps.length));
  const cw = Math.min(46, (VIEW_W - LABEL - 12) / most);
  const BAR = 50;
  const ROW = BAR + 30;
  const H = costs.length * ROW + 24;
  const warps = costs[0].blockDim / costs[0].width;
  return (
    <svg viewBox={`0 0 ${VIEW_W} ${H}`} role="img" aria-label="Steps of each kernel, one after another, with warps busy and barriers"
      style={{ width: '100%', maxWidth: VIEW_W, minWidth: MIN_W, display: 'block' }}>
      <text x={LABEL} y={12} fontSize="10" style={{ fill: 'var(--ink-2)' }}>one column per step, in order; height: warps busy (of {warps}); pale: lanes busy for nothing</text>
      {costs.map((c, r) => {
        const y0 = 22 + r * ROW;
        return (
          <g key={c.kernel} data-kernel={c.kernel} data-barriers={c.barriers}>
            <text x={LABEL - 8} y={y0 + BAR / 2 + 4} fontSize="11" textAnchor="end">{SHORT[c.kernel]}</text>
            <line x1={LABEL} y1={y0 + BAR} x2={LABEL + most * cw} y2={y0 + BAR} stroke="var(--rule)" />
            {c.steps.map((s, i) => {
              const h = Math.max(2, (s.warps / warps) * BAR);
              const fullH = h;
              const usefulH = (s.useful / s.slots) * fullH;
              const x = LABEL + i * cw;
              return (
                <g key={i} data-step={i} data-warps={s.warps} data-slots={s.slots} data-useful={s.useful} data-barrier={s.barrierBefore ? 1 : 0}>
                  <title>{`${s.label}: ${s.warps} warp${s.warps === 1 ? '' : 's'} busy, ${s.useful} of ${s.slots} lane slots useful`}</title>
                  <rect x={x + 2} y={y0 + BAR - fullH} width={cw - 4} height={fullH} fill={KIND_WASH[s.what]} stroke={KIND_FILL[s.what]} strokeWidth="0.8" />
                  <rect x={x + 2} y={y0 + BAR - usefulH} width={cw - 4} height={usefulH} fill={KIND_FILL[s.what]} />
                  {s.barrierBefore ? <line x1={x + 0.5} y1={y0 - 4} x2={x + 0.5} y2={y0 + BAR + 4} stroke="var(--ink)" strokeWidth="2" /> : null}
                </g>
              );
            })}
            <text x={LABEL} y={y0 + BAR + 16} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
              {c.barriers} barrier{c.barriers === 1 ? '' : 's'} (the vertical lines), {c.steps.length} steps
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function CostExplorer() {
  const widths = useWidths();
  const [B, setB] = useState(1024);
  const [key, setKey] = useState('book');
  const chosen = widths.find((c) => c.key === key) ?? widths[0];
  const W = chosen.width;
  const costs = useMemo(() => KERNELS.map((k) => blockCosts(k, B, W)), [B, W]);
  const [s, h, t] = costs;
  const rows: { label: string; k: string; get: (c: Costs) => string | number }[] = [
    { label: '__syncthreads() executed', k: 'barriers', get: (c) => c.barriers },
    { label: 'shared-memory writes', k: 'writes', get: (c) => c.sharedWrites },
    { label: 'shared-memory reads', k: 'reads', get: (c) => c.sharedReads },
    { label: 'shuffle steps, one after another', k: 'rounds', get: (c) => c.shuffleRounds },
    { label: 'warp-wide shuffles issued in the block', k: 'shuffles', get: (c) => c.warpShuffles },
    { label: 'steps, one after another', k: 'steps', get: (c) => c.steps.length },
    { label: 'lane slots spent (busy warps × width, per step)', k: 'slots', get: (c) => c.slots },
    { label: 'useful additions', k: 'useful', get: (c) => c.useful },
    { label: 'lane slots spent on nothing', k: 'wasted', get: (c) => c.wasted },
  ];
  return (
    <Bench
      title="Hybrid or two-stage: count the costs"
      task="At 1,024 threads and width 32: how many barriers does each kernel execute? Then try 256 threads, and width 64."
      foot={
        <>
          One block reducing 2 × block size values, all three kernels side by side. A lane slot is one lane of one busy warp
          in one step, the book's measure of execution resources (p. 231). The model counts; it does not time.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Block size
          <select value={B} aria-label="block size" onChange={(e) => setB(+e.target.value)}>
            {BLOCK_SIZES.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <label className="field">
          Warp width
          <select value={chosen.key} aria-label="warp width" onChange={(e) => setKey(e.target.value)}>
            {widths.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
      </div>
      <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.8rem' }} data-costs>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>per block</th>
            {costs.map((c) => <th key={c.kernel} className="num">{KERNEL_LABEL[c.kernel]}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.k} data-row={r.k}>
              <td style={{ paddingLeft: '0.6rem' }}>{r.label}</td>
              {costs.map((c) => <td key={c.kernel} className="num" data-kernel={c.kernel}>{r.get(c)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      <Timeline costs={costs} />
      <p className="narrate" role="status" style={{ margin: '0.7rem 0 0' }} data-costs-says>
        Every kernel performs the same {s.useful} useful additions; they differ in what surrounds them. Against the hybrid
        kernel, the two-stage kernel saves {h.barriers - t.barriers} of {h.barriers} barriers and{' '}
        {h.sharedWrites + h.sharedReads - t.sharedWrites - t.sharedReads} shared-memory accesses, and spends{' '}
        {t.wasted - h.wasted} more lane slots on results nobody reads. The book's argument for the trade: once the data is
        loaded, a reduction is latency-bound, so the long waits it removes matter more than the idle lanes it adds. Which
        effect wins on your device is a measurement, at the bottom of this page.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: read past the array
function PastArrayExplorer() {
  const widths = useWidths();
  const [B, setB] = useState(256);
  const [key, setKey] = useState('book');
  const [guarded, setGuarded] = useState(false);
  const chosen = widths.find((c) => c.key === key) ?? widths[0];
  const W = chosen.width;
  const r = useMemo(() => twoStage(ones(2 * B), B, W, guarded), [B, W, guarded]);
  const perRow = 32;
  const n = Math.max(r.slots.length, W);
  const rowsN = Math.ceil(n / perRow);
  const cw = (VIEW_W - 16) / perRow;
  const CH = 22;
  const GAPY = 34;
  const BLOCK = 12 + CH + GAPY + CH + 18;
  const H = rowsN * BLOCK + 6;
  const x = (k: number) => 8 + (k % perRow) * cw;
  const y = (k: number) => Math.floor(k / perRow) * BLOCK + 12;
  const past = r.slots.filter((s) => s.state === 'past').length;
  const garbage = r.slots.filter((s) => s.state === 'garbage').length;
  const reads = r.lanes.filter((l) => l.read != null);
  const badReads = reads.filter((l) => r.slots[l.read!].state !== 'written').length;
  return (
    <Bench
      title="Read past the array"
      task="As printed, at 256 threads: which slots does warp 0 read? Then 1,024 threads at width 64. Then guard the read."
      foot={
        <>
          Top: <code>partialSums_s</code>, one slot per warp, written by that warp's lane 0 in stage 1 (all inputs are 1, so a
          warp's sum is 2 × width). Bottom: what each lane of warp 0 starts stage 2 with. A dashed slot lies past the end of
          the array.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Block size
          <select value={B} aria-label="block size" onChange={(e) => setB(+e.target.value)}>
            {BLOCK_SIZES.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <label className="field">
          Warp width
          <select value={chosen.key} aria-label="warp width" onChange={(e) => setKey(e.target.value)}>
            {widths.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label="Stage-2 read">
          <button aria-pressed={!guarded} onClick={() => setGuarded(false)}>as printed</button>
          <button aria-pressed={guarded} onClick={() => setGuarded(true)}>guarded, padded with 0</button>
        </span>
      </div>
      <pre className="mono" style={{ margin: '0 0 0.6rem', whiteSpace: 'pre-wrap' }} data-read-code>
        {guarded
          ? `__shared__ float partialSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];   // ${r.arraySize} entries at this block size\nfloat partialSum = laneIdx() < warps ? partialSums_s[laneIdx()] : 0.0f;`
          : `__shared__ float partialSums_s[BLOCK_DIM/WARP_SIZE];   // ${B}/${W} = ${r.arraySize} entries\nfloat partialSum = partialSums_s[t];`}
      </pre>
      <svg viewBox={`0 0 ${VIEW_W} ${H}`} role="img" aria-label="Per-warp sums in shared memory and what warp 0 reads" style={{ width: '100%', maxWidth: VIEW_W, minWidth: MIN_W, display: 'block' }}>
        {r.lanes.map((l) => {
          if (l.read == null) return null;
          const k = l.read;
          const ok = r.slots[k].state === 'written';
          return (
            <line key={'a' + l.lane} x1={x(k) + cw / 2} y1={y(k) + CH} x2={x(l.lane) + cw / 2} y2={y(l.lane) + CH + GAPY}
              stroke={ok ? 'var(--shared)' : 'var(--fail)'} strokeWidth="1" strokeDasharray={ok ? undefined : '3 2'} opacity="0.8" />
          );
        })}
        {range(n).map((k) => {
          const s = r.slots[k];
          if (!s) return null;
          const fill = s.state === 'written' ? 'var(--shared)' : s.state === 'garbage' ? 'var(--fail-wash)' : 'none';
          const stroke = s.state === 'written' ? 'var(--shared)' : 'var(--fail)';
          return (
            <g key={'s' + k} data-slot={k} data-state={s.state}>
              {k % 8 === 0 ? <text x={x(k) + cw / 2} y={y(k) - 3} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{k}</text> : null}
              <rect x={x(k) + 0.6} y={y(k)} width={cw - 1.2} height={CH} rx="2" fill={fill} stroke={stroke} strokeDasharray={s.state === 'past' ? '3 2' : undefined} />
              <text x={x(k) + cw / 2} y={y(k) + 14.5} fontSize="9" textAnchor="middle"
                style={{ fill: s.state === 'written' ? 'var(--page)' : 'var(--fail)' }} fontWeight={s.state === 'written' ? 400 : 700}>
                {s.value ?? '?'}
              </text>
            </g>
          );
        })}
        {r.arraySize < n && r.arraySize % perRow !== 0 ? (
          <line x1={x(r.arraySize) - 0.5} y1={y(r.arraySize) - 8} x2={x(r.arraySize) - 0.5} y2={y(r.arraySize) + CH + 6} stroke="var(--ink)" strokeWidth="2" />
        ) : null}
        {r.lanes.map((l) => {
          const yy = y(l.lane) + CH + GAPY;
          const bad = l.value == null;
          return (
            <g key={'l' + l.lane} data-lane={l.lane} data-value={l.value == null ? '?' : l.value}>
              <rect x={x(l.lane) + 0.6} y={yy} width={cw - 1.2} height={CH} rx="2"
                fill={bad ? 'var(--fail-wash)' : l.read == null ? 'var(--cell)' : 'var(--reg-wash)'} stroke={bad ? 'var(--fail)' : l.read == null ? 'var(--rule)' : 'var(--reg)'} />
              <text x={x(l.lane) + cw / 2} y={yy + 14.5} fontSize="9" textAnchor="middle" style={{ fill: bad ? 'var(--fail)' : l.read == null ? 'var(--ink-3)' : 'var(--ink)' }}>
                {l.value == null ? '?' : l.value}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="narrate" role="status" style={{ margin: '0.6rem 0 0' }} data-past-says>
        {r.warps} warp{r.warps === 1 ? '' : 's'} of {W} write {r.warps} partial sum{r.warps === 1 ? '' : 's'} into an array of {r.arraySize}.{' '}
        {guarded
          ? `Lanes 0 to ${r.warps - 1} read them; the other ${W - r.warps} lanes of warp 0 start with 0, the identity of +. ${garbage ? `The ${garbage} unused slots are never read. ` : ''}`
          : badReads
            ? `All ${W} lanes of warp 0 read a slot: ${past} of those reads land past the end of the array, where nothing was written, and the sum is undefined. `
            : `All ${W} lanes of warp 0 read a slot, and every slot was written: this is the one shape the printed listing fits. `}
        <span className={'chip ' + (r.correct ? 'ok' : 'fail')} data-result={r.correct ? 'ok' : 'fail'}>
          {r.result == null ? 'undefined' : r.result}
        </span>{' '}
        <span className="muted">(right answer {r.expected})</span>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the measurement
function measured(reports: Reports) {
  return realBackends(reports).flatMap((b) => {
    const rows = benchRows(reports[b], LAB).filter((r) => r.name === 'rows');
    const blocks = [...new Set(rows.map((r) => num(r.block)).filter((v): v is number => v != null))];
    const block = blocks.includes(256) ? 256 : blocks[0];
    const pick = (k: string) => num(rows.find((r) => r.kernel === k && num(r.block) === block)?.seconds);
    const shared = pick('shared');
    const warp = pick('warp');
    return shared && warp && block
      ? [{ b, device: String(reports[b].device.name ?? b), block, shared, warp, speedup: shared / warp }]
      : [];
  });
}

function MeasuredSpeedup({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = measured(reports);
  if (!found.length)
    return (
      <p data-measured="none">
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 10.1 --solution --bench</code> on a machine
        with a GPU, then reload the page: this box will compare your prediction with the result.
      </p>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b} data-measured={f.b}>
          On <b>{f.device}</b> ({f.b}), rows of {2 * f.block} values reduced by blocks of {f.block}: the shared-memory kernel
          took {fmt(f.shared * 1e3)} ms and the two-stage kernel {fmt(f.warp * 1e3)} ms, so the two-stage kernel ran{' '}
          <b>{fmt(f.speedup)}×</b> as fast.{' '}
          {predicted > f.speedup * 1.5
            ? 'You expected considerably more. Removing barriers helps only as much as they cost on this device, and both kernels read exactly the same data from global memory.'
            : predicted < f.speedup / 1.5
              ? 'You expected considerably less.'
              : 'That is close to what you predicted.'}
        </p>
      ))}
      <p className="muted">These are measurements from your machine, not expected values. Another device can differ by a wide margin in either direction.</p>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS108: Record<string, Question> = {
  'ch10/10.8/q-barriers': {
    kind: 'numeric',
    id: 'ch10/10.8/q-barriers',
    prompt: (
      <>
        A block of 1,024 threads in 32-wide warps. How many times does the hybrid kernel of Fig. 10.14 execute{' '}
        <code>__syncthreads()</code>?
      </>
    ),
    answer: 6,
    explain: (
      <p>
        Five in the loop, one per stride (512, 256, 128, 64, 32), and one after it (line 11). Fig. 10.10 executes 10, one
        per stride from 512 down to 1; the two-stage kernel executes 1. In general the hybrid kernel executes
        log2(block) − log2(width) + 1.
      </p>
    ),
    source: 'book Figs. 10.10, 10.14, 10.16 (pp. 237, 240, 243); model: src/sim/shuffle.ts',
  },
  'ch10/10.8/q-latency': {
    kind: 'choice',
    id: 'ch10/10.8/q-latency',
    prompt: <>Why can a kernel in which more lanes compute results nobody uses be faster than one with fewer such lanes?</>,
    options: [
      {
        label: 'After the load it is latency-bound: barriers and shared-memory trips are long waits, and idle lanes cost little.',
        correct: true,
      },
      { label: 'Wasted lanes are free: the hardware detects additions whose results are never read and skips them.', why: 'Nothing skips them; every lane of the warp executes every shuffle and add.' },
      { label: 'More lanes doing work raise occupancy, so more blocks fit on each SM and hide each other\'s memory latency.', why: 'Occupancy depends on resources a block reserves, not on how many lanes do useful work.' },
    ],
    explain: (
      <p>
        The book's argument (pp. 243–244): the cycles saved by idling lanes would mostly become stall cycles. The
        two-stage kernel trades a few long-latency operations (barriers, shared-memory
        accesses) for many cheap ones (shuffles and adds in lanes whose results are never read). In a kernel that is mostly
        waiting, removing waits is what counts.
      </p>
    ),
    source: 'book §10.8, pp. 243–244',
  },
  'ch10/10.8/q-guard-256': {
    kind: 'choice',
    id: 'ch10/10.8/q-guard-256',
    prompt: (
      <>
        The two-stage kernel exactly as printed (<code>partialSums_s[BLOCK_DIM/WARP_SIZE]</code>, every lane of warp 0 reads{' '}
        <code>partialSums_s[t]</code>), launched with 256 threads per block of 32-wide warps. What does warp 0 read in stage 2?
      </>
    ),
    options: [
      { label: '8 per-warp sums, then 24 reads past the end of the array: the sum is undefined.', correct: true },
      { label: '8 per-warp sums and 24 zeros, because a kernel\'s shared memory starts out zeroed.', why: 'Shared memory is not initialised, and these reads are past the end of the array anyway. Zeros there would hide the bug, not fix it.' },
      { label: '32 per-warp sums: the array has one entry per lane of warp 0, so 32 in every block.', why: 'It has BLOCK_DIM / WARP_SIZE = 256 / 32 = 8.' },
    ],
    explain: (
      <p>
        8 warps write 8 sums into an array of 8; lanes 8 to 31 read past it. The guard is one comparison: read only
        where <code>lane &lt; blockDim.x / warpSize</code>, and add 0 otherwise. The listing is right only when there are exactly
        as many warps as lanes, a block of 1,024 threads of 32-wide warps (footnote 5 has that case in mind). Lab 10.1 sizes
        the array for the largest block and the narrowest warp, and reads only <code>laneIdx() &lt; warps</code>.
      </p>
    ),
    source: 'book Fig. 10.16, p. 243 (notes/ch10.md, "Unclear in the PDF"); ch10-facts B2, D1, D4',
  },
  'ch10/10.8/q-w64-warps': {
    kind: 'numeric',
    id: 'ch10/10.8/q-w64-warps',
    prompt: <>A block of 1,024 threads on hardware with 64-wide warps. How many per-warp partial sums does stage 1 produce?</>,
    answer: 16,
    explain: (
      <p>
        1,024 / 64 = 16 warps, so 16 sums, while warp 0 has 64 lanes. Read unguarded, 48 lanes read past them. On 64-wide
        hardware the printed listing is wrong at every block size.
      </p>
    ),
    source: 'notes/today/ch10-facts.md B2',
  },
  'ch10/10.8/q-metal-lanes': {
    kind: 'choice',
    id: 'ch10/10.8/q-metal-lanes',
    prompt: (
      <>
        A Metal softmax gives each row one threadgroup, large enough to span several SIMD-groups. Each thread adds its{' '}
        <code>exp(x − max)</code> terms into <code>e</code>, then calls <code>float s = simd_sum(e);</code>. What has to
        happen before thread 100, which is not in the first SIMD-group, may divide its terms by the row's sum?
      </>
    ),
    options: [
      {
        label: 'Nothing: simd_sum hands its result to every thread, so thread 100\'s s is already the row\'s sum.',
        why: 'To every active thread of its SIMD-group. Thread 100\'s s is the sum of its own SIMD-group\'s terms only.',
      },
      {
        label: 'Each SIMD-group\'s lane 0 stores its s; a barrier; one SIMD-group sums those and stores the total; a barrier; all read it.',
        correct: true,
      },
      {
        label: 'Thread 100 fetches lane 0\'s s with simd_shuffle, since simd_sum leaves the row\'s sum in lane 0 of the threadgroup.',
        why: 'simd_sum already gives its result to every active lane of the group. The missing part is the other groups.',
      },
      {
        label: 'One threadgroup_barrier(mem_flags::mem_threadgroup) after the simd_sum, so that every SIMD-group has finished before any thread divides.',
        why: 'A barrier orders memory accesses; it does not combine the groups\' sums, which are still in separate registers.',
      },
    ],
    explain: (
      <p>
        Apple's specification says <code>simd_sum</code> broadcasts its result "to all active threads in the SIMD-group",
        not the threadgroup. Combining the groups is Fig. 10.16's second stage, and because Metal's second{' '}
        <code>simd_sum</code> leaves the row sum only in the SIMD-group that ran it, the others read it back from
        threadgroup memory after one more barrier. MLX's <code>softmax_single_row</code> does exactly this, through{' '}
        <code>local_max[simd_group_id]</code> and then <code>local_max[0]</code> for the max, and the same for the sum.
      </p>
    ),
    source: 'MSL spec 6.10.2 (notes/today/ch10-facts.md C1); MLX softmax.h (ch10-facts D3)',
  },
};

// ---------------------------------------------------------------- the section
export function S108() {
  return (
    <>
      <p>
        The hybrid kernel of section 10.7 still pays for shared memory and a barrier in every iteration until one warp is
        left: six barriers in a block of 1,024 threads before <code>warp_reduce</code> takes over. The next step is to use
        warp reduction from the very start.
      </p>
      <Source section="10.8" pages="241" />

      <h2>Two stages, one barrier</h2>
      <p>
        In stage 1 every thread adds its two input elements into a register, and every warp reduces its own 32 values with{' '}
        <code>warp_reduce</code>. All warps work at once, and none of them touches shared memory. Lane 0 of each warp then
        writes its warp's sum to <code>partialSums_s[warpIdx()]</code>, and the block meets at its only{' '}
        <code>__syncthreads()</code>. In stage 2, warp 0 loads the per-warp sums, one per lane, and runs{' '}
        <code>warp_reduce</code> once more; thread 0 writes the result.
      </p>
      <Source pages="241–243" figs="Figs. 10.15, 10.16" />
      <p>
        Why one warp is enough for stage 2: a block has at most 1,024 threads and a warp 32 lanes, so there are at most 32
        partial sums, one per lane of warp 0. Like every kernel so far, this one still reduces a single block's 2,048 values;
        section 10.9 lifts that limit.
      </p>
      <Source pages="242–243" />
      <TwoStageListing />
      <p>
        Compared with Fig. 10.16, the lab's version takes the row each block reduces from <code>blockIdx.x</code> (one block
        per row, so it can be benchmarked on many rows), uses <code>warpSize</code> and the portable{' '}
        <code>warp_reduce</code> of section 10.7, sizes <code>partialSums_s</code> for the largest block and the narrowest
        warp, and lets only as many lanes read it as there are warps. That last change fixes a bug in the listing, which
        the second explorer below takes apart. On Metal the lane, SIMD-group index and group count arrive as attributes,
        because Apple leaves the split of a threadgroup into SIMD-groups to the implementation.
      </p>

      <h2>What it saves, and what it spends</h2>
      <p>
        The book is candid about the trade. The two-stage kernel needs shared memory and a barrier only once, at the boundary
        between the stages. But where the hybrid kernel retired whole warps as the tree shrank, here every warp stays busy
        through stage 1, and in each shuffle step most of its lanes compute something that never reaches lane 0. The book
        calls this "more control divergence". Strictly, <code>warp_reduce</code> has no branch at all; what grows is the
        number of lanes doing work nobody reads, and the explorer counts exactly that.
      </p>
      <NoteGroup>
        <Source pages="243" />
        <Simplification>
          <p>
            The model counts operations per step for one block: barriers, shared-memory reads and writes, shuffles, and lane
            slots. It does not model time, latency, caches or scheduling. Latency is the book's argument, not something the
            model computes.
          </p>
        </Simplification>
      </NoteGroup>
      <CostExplorer />
      <Ask q={questionsS108['ch10/10.8/q-barriers']} label="Work it out" />
      <p>
        Why spend more lanes to save barriers? Once the data has been loaded, the rest of a reduction is short and mostly
        waiting: it is <b>latency-bound</b>. A barrier and a shared-memory round trip are long waits. An idle lane in a warp
        that issues the instruction anyway costs almost nothing, and if it were removed, the book argues, the cycles saved
        "will likely be replaced by stall cycles". Remove the long waits, then, and accept the idle lanes.
      </p>
      <Source pages="243–244" />
      <Ask q={questionsS108['ch10/10.8/q-latency']} />

      <h2>A listing that fits one block size</h2>
      <p>
        Fig. 10.16 declares <code>partialSums_s[BLOCK_DIM/WARP_SIZE]</code> and then lets every lane of warp 0 read{' '}
        <code>partialSums_s[t]</code>. Both are right for 1,024 threads in 32-wide warps: 32 warps, 32 entries, 32 lanes.
        For any smaller block the array has fewer entries than warp 0 has lanes, and the extra lanes read past its end. On
        64-wide hardware a block of 1,024 threads has only 16 warps, so the read overruns at every block size.
      </p>
      <NoteGroup>
        <Source pages="243" figs="Fig. 10.16" />
        <Today checked="2026-11-20 (ch10-facts D1, D3, D4)">
          <p>
            llama.cpp's <code>block_reduce</code>, MLX's Metal reductions and PyTorch's <code>BlockReduceSum</code> all
            guard that read and pad with the identity.
          </p>
        </Today>
      </NoteGroup>
      <p>
        The fix costs one comparison: size the array for the largest block and the narrowest warp any backend documents
        (32 lanes), and let lane <i>t</i> read slot <i>t</i> only if warp <i>t</i> exists, adding 0 otherwise. Compare the
        two versions below at 256 threads, and at 1,024 threads in 64-wide warps.
      </p>
      <Simplification>
        <p>
          A slot that was never written, or that lies past the array, is drawn as undefined. The course's emulator fills
          shared memory with the byte 0xCD, so the bug shows; on a GPU the memory may happen to hold zeros and hide it.
        </p>
      </Simplification>
      <PastArrayExplorer />
      <Ask q={questionsS108['ch10/10.8/q-guard-256']} />
      <Ask q={questionsS108['ch10/10.8/q-w64-warps']} label="Work it out" />

      <h2>The same function in real libraries</h2>
      <p>
        You will find this kernel inside the softmax kernels of the libraries you use. llama.cpp's{' '}
        <code>soft_max_f32</code> calls a <code>block_reduce</code> with this structure twice, once for the row's maximum and
        once for the sum of exponentials, and puts a barrier between the two calls because they share the buffer. PyTorch's{' '}
        <code>BlockReduceSum</code> starts with a barrier for the same reason, so it can be called twice in a row. On Metal,
        MLX's <code>softmax_single_row</code> and llama.cpp's <code>kernel_op_sum_f32</code> do the same with{' '}
        <code>simd_max</code> and <code>simd_sum</code>, which return the result to every lane of the SIMD-group.
      </p>
      <Beyond>
        <p>
          Source read on 2026-11-20 (notes/today/ch10-facts.md C1, D1 to D4). The book does not mention these libraries.
          Lab 10.3 builds a softmax and an argmax over logits; see the <a href={href('ch10', 'labs')}>labs page</a>.
        </p>
      </Beyond>
      <Ask q={questionsS108['ch10/10.8/q-metal-lanes']} />

      <h2>Measure it</h2>
      <p>
        Lab 10.1 times the shared-memory kernel of Fig. 10.10 and the two-stage kernel on the same data: many rows of 512
        values, one block of 256 threads per row, the way a softmax over short rows would run. Commit to a number first.
      </p>
      <Predict
        id="ch10/10.8/p-twostage"
        prompt={
          <>
            On <em>your</em> device, how many times as fast as the shared-memory kernel (Fig. 10.10) will the two-stage kernel
            run on rows of 512 values?
          </>
        }
        unit="× as fast"
      >
        {(v) => <MeasuredSpeedup predicted={v} />}
      </Predict>
    </>
  );
}
