import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, labStatus, num, realBackends, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK, BOOK_BLOCK_SIZES, EX6, EX7, EX7_ASSIGNMENTS, EX8, EX8_KERNELS, MAX_REGS_PER_THREAD, assignment,
  limitsFromReport, notReported, occupancy, pct, regValues, regsAtFullOccupancy, scan, smallestFillingBlock, steps,
  threadValues, type Kernel, type LimitKey, type Limits, type Occupancy, type ScanPoint, type Step,
} from '../../sim/occupancy';

const count = (v: number) => v.toLocaleString('en-US');
const plural = (v: number, one: string, many = one + 's') => `${count(v)} ${v === 1 ? one : many}`;
const toInt = (s: string) => {
  const v = Number(s);
  return Number.isFinite(v) && v > 0 ? Math.floor(v) : 0;
};
const KB = 1024;

// ---------------------------------------------------------------- devices
interface Device {
  key: string;
  label: string;
  limits: Limits;
  /** Where the numbers come from. */
  note: string;
}

const BOOK_DEVICE: Device = {
  key: 'book',
  label: "The book's example",
  limits: BOOK,
  note: "The book's example device: 32 blocks, 2048 threads (64 warps) and 65,536 registers per SM, 1024 threads per block (§4.7, pp. 85–86); 228 KB of shared memory per SM (§5.6, p. 116).",
};
/** One row of NVIDIA's current table of limits (ch04-facts.md, A12): compute capabilities 8.6 and 8.7. */
export const TABLE_1536: Limits = {
  threadSlots: 1536, blockSlots: 16, blockSlotsDerived: false, regs: 65536, shared: null, width: 32, maxThreadsPerBlock: 1024,
};
const TABLE_DEVICE: Device = {
  key: 't1536',
  label: "A row of NVIDIA's table: 1536 thread slots, 16 block slots",
  limits: TABLE_1536,
  note: "The row for compute capabilities 8.6 and 8.7 in NVIDIA's table of limits, read on 2026-11-20 (ch04-facts.md, A12): 16 blocks, 48 warps (1536 threads) and 64 K registers per SM, 1024 threads per block. No shared-memory figure is used for it here.",
};
const EX_DEVICES: Device[] = [
  { key: 'ex6', label: 'The device of exercise 6', limits: EX6, note: 'Exercise 6 (book p. 91): an SM that holds up to 1536 threads and 4 blocks. The exercise gives no register or shared-memory figure. The warp width of 32 is the book\'s.' },
  { key: 'ex7', label: 'The device of exercise 7', limits: EX7, note: 'Exercise 7 (book p. 92): 64 blocks and 2048 threads per SM. The exercise gives no register or shared-memory figure. The warp width of 32 is the book\'s.' },
];

/** CUDA and HIP devices in this machine's reports that carry per-SM fields. */
function reportedDevices(reports: Reports): Device[] {
  return realBackends(reports).flatMap((b) => {
    const d = reports[b].device as Record<string, unknown>;
    const limits = limitsFromReport(d, b);
    if (!limits) return [];
    const missing = notReported(d);
    return [
      {
        key: b,
        label: `Yours: ${String(d.name ?? b)} (${b})`,
        limits,
        note:
          `What this device reported in its device properties, in the run of ${reports[b].generated.slice(0, 10)}.` +
          (b === 'hip' ? ' On HIP the block-slot figure is computed from the thread slots, so it is shown and not counted as a limit.' : '') +
          (missing.length ? ` Missing or 0, and so left out: ${missing.join(', ')}.` : ''),
      },
    ];
  });
}

// ---------------------------------------------------------------- the budget table
const LIMIT_NAME: Record<LimitKey, string> = { threads: 'Thread slots', blocks: 'Block slots', regs: 'Registers', shared: 'Shared memory' };
const LIMIT_LOWER: Record<LimitKey, string> = { threads: 'thread slots', blocks: 'block slots', regs: 'registers', shared: 'shared memory' };
const BAR: Record<LimitKey, string> = { threads: 'var(--ink)', blocks: 'var(--ink)', regs: 'var(--reg)', shared: 'var(--shared)' };
const stacked = (bytes: number) => (bytes >= KB && bytes % KB === 0 ? `${count(bytes / KB)} KB` : `${count(bytes)} B`);

function whyNot(key: LimitKey, why: string | undefined): string {
  if (why === 'derived') return 'on HIP this figure is computed from the thread slots, so it is not a limit of its own';
  if (why === 'unused') return key === 'regs' ? 'registers per thread is set to 0' : 'this kernel uses none';
  return 'no figure for this device';
}

/** One row per budget: what an SM has, what a block takes, how many blocks that allows, and which one binds. */
function BudgetTable({ o }: { o: Occupancy }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="data" style={{ margin: 0 }}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.5rem' }}>Budget</th>
            <th className="num">One SM has</th>
            <th className="num">One block needs</th>
            <th className="num">Blocks it allows</th>
            <th>In use with {plural(o.blocks, 'block')}</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {o.rows.map((r) => {
            const binds = o.binding.includes(r.key);
            const used = o.blocks * r.need;
            const share = r.cap ? Math.min(100, (100 * used) / r.cap) : 0;
            const show = r.key === 'shared' ? stacked : count;
            return (
              <tr key={r.key} data-budget={r.key} data-fit={r.fit ?? 'none'} data-binds={binds ? '' : undefined} style={binds ? { background: 'var(--rule-soft)' } : undefined}>
                <td style={{ paddingLeft: '0.5rem' }}>
                  {r.key === 'shared' ? <span className="chip shared">{LIMIT_NAME[r.key]}</span> : r.key === 'regs' ? <span className="chip reg">{LIMIT_NAME[r.key]}</span> : LIMIT_NAME[r.key]}
                </td>
                <td className="num">{r.cap != null ? show(r.cap) : 'n/a'}</td>
                {r.fit == null ? (
                  <td colSpan={3} className="muted">
                    Not counted: {whyNot(r.key, r.why)}.
                  </td>
                ) : (
                  <>
                    <td className="num">{show(r.need)}</td>
                    <td className="num">{count(r.fit)}</td>
                    <td style={{ minWidth: '9rem' }}>
                      <div aria-hidden="true" style={{ height: 8, background: 'var(--rule-soft)', borderRadius: 2, margin: '0.25rem 0' }}>
                        <div style={{ width: share + '%', height: '100%', background: BAR[r.key], borderRadius: 2 }} />
                      </div>
                      <span className="muted">
                        {show(used)} of {show(r.cap as number)} ({fmt(share)}%)
                      </span>
                    </td>
                  </>
                )}
                <td style={{ paddingRight: '0.5rem' }}>{binds ? <b>binds</b> : null}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const listOf = (keys: LimitKey[]) => {
  const names = keys.map((k) => LIMIT_LOWER[k]);
  return names.length > 1 ? names.slice(0, -1).join(', ') + ' and ' + names[names.length - 1] : (names[0] ?? '');
};
const cap1 = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** One sentence on why occupancy is what it is. */
function verdictOf(o: Occupancy, kernel: Kernel, threadSlots: number | null): ReactNode {
  if (!o.usable)
    return <>Occupancy needs two numbers before anything else: the thread slots of one SM and the threads in one block.</>;
  if (o.blocks === 0)
    return (
      <>
        <span className="chip fail">does not fit</span> Not even one block fits on an SM: a block needs more {listOf(o.binding)} than
        the SM has. NVIDIA's guide says such a kernel fails to launch.
      </>
    );
  if (o.idleSlots === 0)
    return (
      <>
        <b>Every thread slot is taken.</b> {cap1(plural(o.blocks, 'block'))} of {plural(o.warpsPerBlock, 'warp')} each.{' '}
        {o.binding.length > 1 ? `${cap1(listOf(o.binding))} run out together.` : 'No other budget runs out first.'}
      </>
    );
  if (o.binding.length === 1 && o.binding[0] === 'threads')
    return (
      <>
        <b>Thread slots bind, but not cleanly.</b> {cap1(plural(o.blocks, 'block'))} take {count(o.blocks * o.slotsPerBlock)} of{' '}
        {count(threadSlots ?? 0)} slots, and one more block of {count(kernel.threads)} threads does not fit in the{' '}
        {count(o.idleSlots)} that remain.
      </>
    );
  return (
    <>
      <b>
        {cap1(listOf(o.binding))} {o.binding.length === 1 && o.binding[0] === 'shared' ? 'binds' : 'bind'}.
      </b>{' '}
      An SM has room for only {plural(o.blocks, 'block')} of this kernel, which leaves {count(o.idleSlots)} thread slots empty.
    </>
  );
}

// ---------------------------------------------------------------- the calculator
interface CalcPreset {
  key: string;
  label: string;
  kernel: Kernel;
}
const CALC_PRESETS: CalcPreset[] = [
  ...BOOK_BLOCK_SIZES.map((t) => ({ key: 't' + t, label: String(t), kernel: { threads: t, regs: 0, shared: 0 } })),
  { key: 't48', label: '48 (not whole warps)', kernel: { threads: 48, regs: 0, shared: 0 } },
];

function OccupancyCalculator() {
  const { reports, loading } = useReports();
  const mine = useMemo(() => reportedDevices(reports), [reports]);
  const devices = useMemo(() => [BOOK_DEVICE, ...mine, TABLE_DEVICE, ...EX_DEVICES], [mine]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState({
    threadSlots: String(BOOK.threadSlots), blockSlots: String(BOOK.blockSlots), regs: String(BOOK.regs), shared: String(BOOK.shared), width: String(BOOK.width),
  });
  const [work, setWork] = useState({ threads: '256', regs: '0', shared: '0' });

  const customLimits: Limits = {
    threadSlots: toInt(custom.threadSlots) || null,
    blockSlots: toInt(custom.blockSlots) || null,
    blockSlotsDerived: false,
    regs: toInt(custom.regs) || null,
    shared: toInt(custom.shared) || null,
    width: toInt(custom.width) || null,
    maxThreadsPerBlock: null,
  };
  const dev: Device =
    key === 'custom'
      ? { key: 'custom', label: 'Custom numbers', limits: customLimits, note: "Limits you typed. They start from the book's example. A field left at 0 is not counted." }
      : (devices.find((d) => d.key === key) ?? BOOK_DEVICE);
  const kernel: Kernel = { threads: toInt(work.threads), regs: toInt(work.regs), shared: toInt(work.shared) };
  const o = occupancy(dev.limits, kernel);
  const active = (p: CalcPreset) => p.kernel.threads === kernel.threads && p.kernel.regs === kernel.regs && p.kernel.shared === kernel.shared;

  const field = (label: string, value: string, set: (v: string) => void, hint?: string) => (
    <label className="field">
      {label}
      <input type="number" min="0" step="1" value={value} aria-label={label} onChange={(e) => set(e.target.value)} />
      {hint ? <span style={{ fontSize: '0.74rem' }}>{hint}</span> : null}
    </label>
  );
  const fillAt = regsAtFullOccupancy(dev.limits);
  const smallest = smallestFillingBlock(dev.limits);

  return (
    <Bench
      title="Occupancy calculator"
      task="Find the budget that runs out first. Then change one input until a different one does."
      foot={<>{dev.note} Registers are counted as threads × registers per thread, the book's arithmetic.</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          Limits of one SM
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label="Limits of one SM">
            {devices.map((d) => (
              <option key={d.key} value={d.key}>
                {d.label}
              </option>
            ))}
            <option value="custom">Custom numbers</option>
          </select>
        </label>
        {key === 'custom' ? (
          <>
            {field('Thread slots per SM', custom.threadSlots, (v) => setCustom({ ...custom, threadSlots: v }))}
            {field('Block slots per SM', custom.blockSlots, (v) => setCustom({ ...custom, blockSlots: v }))}
            {field('Registers per SM', custom.regs, (v) => setCustom({ ...custom, regs: v }))}
            {field('Shared memory per SM (bytes)', custom.shared, (v) => setCustom({ ...custom, shared: v }))}
            {field('Threads in a warp', custom.width, (v) => setCustom({ ...custom, width: v }))}
          </>
        ) : null}
      </div>
      {!loading && mine.length === 0 ? (
        <p className="muted" data-out="no-device" style={{ margin: '0 0 0.8rem' }}>
          No CUDA or HIP device in this machine's reports, so the list has no entry of yours. Run <code>./verify.sh</code>{' '}
          on a machine with the CUDA or ROCm toolkit and open this page there. A Metal device reports none of these
          limits; the reason is further down.
        </p>
      ) : null}
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        {field('Threads per block', work.threads, (v) => setWork({ ...work, threads: v }))}
        {field('Registers per thread', work.regs, (v) => setWork({ ...work, regs: v }), '0 leaves registers out')}
        {field('Shared memory per block (bytes)', work.shared, (v) => setWork({ ...work, shared: v }), '0 leaves it out')}
      </div>
      <div className="row" style={{ marginBottom: '1rem' }} role="group" aria-label="Block sizes the book works through">
        <span className="muted">The book's block sizes</span>
        {CALC_PRESETS.map((p) => (
          <button key={p.key} className="quiet" data-preset={p.key} aria-pressed={active(p)}
            onClick={() => setWork({ threads: String(p.kernel.threads), regs: String(p.kernel.regs), shared: String(p.kernel.shared) })}>
            {p.label}
          </button>
        ))}
      </div>

      <div className="cols plot">
        <div className="panel">
          <BudgetTable o={o} />
          {o.widthAssumed ? (
            <p className="muted" style={{ margin: '0.6rem 0 0' }}>
              No warp width for this device, so a block is counted thread by thread and no padding is added.
            </p>
          ) : null}
        </div>
        <div className="panel" aria-live="polite">
          <p className="muted" style={{ margin: 0 }}>
            Occupancy (warps)
          </p>
          <p style={{ fontSize: '2rem', fontWeight: 600, margin: '0 0 0.2rem', lineHeight: 1.1 }}>
            <output data-out="occupancy" data-value={o.usable ? pct(o.byWarps) : 'n/a'} data-binding={o.binding.join('+')}>
              {o.usable ? pct(o.byWarps) + '%' : 'n/a'}
            </output>
          </p>
          {o.usable ? (
            <p className="muted" style={{ margin: '0 0 0.6rem' }}>
              <span data-out="blocks">{count(o.blocks)}</span> {o.blocks === 1 ? 'block' : 'blocks'} × {plural(o.warpsPerBlock, 'warp')} ={' '}
              <span data-out="warps">{count(o.residentWarps)}</span> of {count(o.warpSlots ?? 0)} warp slots.{' '}
              <span data-out="idle">{count(o.idleSlots)}</span> thread slots idle.
            </p>
          ) : null}
          {o.usable && o.paddedLanes > 0 ? (
            <p data-out="by-threads" data-value={pct(o.byThreads)} style={{ margin: '0 0 0.6rem' }}>
              By threads it is <b>{pct(o.byThreads)}%</b>: {count(o.residentThreads)} threads hold {count(o.blocks * o.slotsPerBlock)}{' '}
              slots, because each block's last warp carries {o.slotsPerBlock - kernel.threads} padded lanes.
            </p>
          ) : null}
          <p style={{ margin: 0 }} data-out="verdict">
            {verdictOf(o, kernel, dev.limits.threadSlots)}
          </p>
          {o.tooLarge ? (
            <p style={{ margin: '0.6rem 0 0' }}>
              <span className="chip fail">invalid launch</span> This device accepts at most {count(dev.limits.maxThreadsPerBlock ?? 0)}{' '}
              threads in a block.
            </p>
          ) : null}
          {fillAt != null || smallest != null ? (
            <p className="muted" style={{ margin: '0.6rem 0 0' }}>
              {smallest != null ? <>Smallest block that can fill the thread slots: <span data-out="smallest">{count(smallest)}</span> threads. </> : null}
              {fillAt != null ? <>Most registers per thread with every slot filled: <span data-out="fill-regs">{fillAt}</span>.</> : null}
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the staircase
const PW = 430;
const PH = 190;
const PL = 40;
const PR = 12;
const PT = 12;
const PB = 28;

function Staircase({ points, x, axis, label }: { points: ScanPoint[]; x: number; axis: 'regs' | 'threads'; label: string }) {
  const n = points.length;
  const w = PW - PL - PR;
  const h = PH - PT - PB;
  const slot = w / n;
  const px = (i: number) => PL + (i + 0.5) * slot;
  const py = (occ: number) => PT + (1 - occ) * h;
  const at = Math.max(0, points.findIndex((p) => p.x === x));
  const cur = points[at];
  const colour = axis === 'regs' ? 'var(--reg)' : 'var(--ink)';
  let path = '';
  points.forEach((p, i) => {
    const x0 = PL + i * slot;
    path += `${i === 0 ? 'M' : 'L'}${x0.toFixed(2)},${py(p.occupancy).toFixed(2)} L${(x0 + slot).toFixed(2)},${py(p.occupancy).toFixed(2)} `;
  });
  const ticks = axis === 'regs' ? [1, 32, 64, 128, 192, 255] : [points[0].x, points[Math.floor((n - 1) / 2)].x, points[n - 1].x];
  return (
    <svg viewBox={`0 0 ${PW} ${PH}`} role="img" aria-label={label} data-plot={axis} data-x={cur?.x} data-occ={cur ? pct(cur.occupancy) : ''} style={{ width: '100%', display: 'block' }}>
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={PL} x2={PW - PR} y1={py(g)} y2={py(g)} stroke="var(--rule-soft)" />
          <text x={PL - 6} y={py(g) + 3.5} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{g * 100}%</text>
        </g>
      ))}
      {axis === 'threads'
        ? points.map((p, i) => (
            <rect key={p.x} x={PL + i * slot + slot * 0.15} y={py(p.occupancy)} width={slot * 0.7} height={Math.max(0, PT + h - py(p.occupancy))}
              fill={i === at ? 'var(--ink)' : 'var(--ink-3)'} opacity={i === at ? 1 : 0.55} />
          ))
        : <path d={path} fill="none" stroke={colour} strokeWidth="1.8" />}
      {ticks.map((t) => {
        const i = points.findIndex((p) => p.x === t);
        return i < 0 ? null : (
          <text key={t} x={px(i)} y={PH - 10} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t}</text>
        );
      })}
      {cur ? (
        <g>
          <line x1={px(at)} x2={px(at)} y1={PT} y2={PT + h} stroke="var(--ink)" strokeDasharray="3 3" strokeWidth="0.8" />
          <circle cx={px(at)} cy={py(cur.occupancy)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
          <text x={Math.min(PW - PR - 4, Math.max(PL + 4, px(at) + (at > n * 0.7 ? -8 : 8)))} y={Math.max(PT + 10, py(cur.occupancy) - 8)}
            fontSize="11" fontWeight="600" textAnchor={at > n * 0.7 ? 'end' : 'start'}
            style={{ paintOrder: 'stroke', stroke: 'var(--page)', strokeWidth: 3.5, strokeLinejoin: 'round' }}>
            {cur.x}: {pct(cur.occupancy)}%
          </text>
        </g>
      ) : null}
    </svg>
  );
}

/** A slider that spans the plot area above it, so that dragging moves the marker under the pointer. */
const SLIDER = { width: `${(100 * (PW - PL - PR)) / PW}%`, marginLeft: `${(100 * PL) / PW}%` } as const;

const stepText = (s: Step) => `${pct(s.from)}% → ${pct(s.to)}%`;

function CliffExplorer() {
  const { reports } = useReports();
  const devices = useMemo(() => [BOOK_DEVICE, ...reportedDevices(reports), TABLE_DEVICE], [reports]);
  const [key, setKey] = useState('book');
  const dev = devices.find((d) => d.key === key) ?? BOOK_DEVICE;
  // Shared memory is left out here so that registers and slots are the only budgets in play.
  const limits: Limits = useMemo(() => ({ ...dev.limits, shared: null }), [dev]);
  const tValues = useMemo(() => threadValues(limits), [limits]);
  const [threadsRaw, setThreads] = useState(512);
  const [regs, setRegs] = useState(31);
  const threads = tValues.includes(threadsRaw) ? threadsRaw : tValues.reduce((a, b) => (Math.abs(b - threadsRaw) < Math.abs(a - threadsRaw) ? b : a), tValues[0]);
  const kernel: Kernel = { threads, regs, shared: 0 };
  const o = occupancy(limits, kernel);
  const byRegs = useMemo(() => scan(limits, { threads, regs: 0, shared: 0 }, 'regs', regValues()), [limits, threads]);
  const byThreads = useMemo(() => scan(limits, { threads: 0, regs, shared: 0 }, 'threads', tValues), [limits, regs, tValues]);
  const regSteps = steps(byRegs);
  const next = regSteps.find((s) => s.at > regs);
  const width = limits.width ?? 1;
  const hasRegs = limits.regs != null;
  const lost = next ? next.from - next.to : 0;

  return (
    <Bench
      title="Find the cliff"
      task="Drag registers per thread up from 31, one at a time. Then change the block size and see where the steps go."
      foot={<>{dev.note} Shared memory is left out of this explorer. Registers are counted as threads × registers per thread, the book's arithmetic; a real kernel's count comes from the vendor's occupancy function.</>}
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-start' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          Limits of one SM
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label="Limits of one SM for the staircase">
            {devices.map((d) => (
              <option key={d.key} value={d.key}>
                {d.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="cols">
        <div className="panel">
          <p className="ui" style={{ margin: '0 0 0.4rem', fontWeight: 600 }}>
            Occupancy against registers per thread, at {count(threads)} threads per block
          </p>
          {hasRegs ? (
            <>
              <Staircase points={byRegs} x={regs} axis="regs" label={`Occupancy against registers per thread at ${threads} threads per block`} />
              <label className="field" style={{ display: 'flex', width: '100%', marginTop: '0.3rem' }}>
                Registers per thread = {regs}
                <input type="range" min={1} max={MAX_REGS_PER_THREAD} step={1} value={regs} aria-label="Registers per thread"
                  style={SLIDER} onChange={(e) => setRegs(+e.target.value)} />
              </label>
              <div className="row" style={{ marginTop: '0.5rem' }} role="group" aria-label="Steps along registers per thread">
                <span className="muted">Steps at</span>
                {regSteps.length ? regSteps.map((s) => (
                  <button key={s.at} className="quiet" data-step={s.at} aria-pressed={regs === s.at} onClick={() => setRegs(s.at)} title={stepText(s)}>
                    {s.at}: {stepText(s)}
                  </button>
                )) : <span className="muted">none: registers never bind at this block size</span>}
              </div>
            </>
          ) : (
            <p className="muted" data-out="no-regs" style={{ margin: 0 }}>
              This device did not report its registers per SM, so there is no register staircase to draw for it.
            </p>
          )}
        </div>
        <div className="panel">
          <p className="ui" style={{ margin: '0 0 0.4rem', fontWeight: 600 }}>
            Occupancy against threads per block, at {regs} registers per thread
          </p>
          <Staircase points={byThreads} x={threads} axis="threads" label={`Occupancy against threads per block at ${regs} registers per thread`} />
          <label className="field" style={{ display: 'flex', width: '100%', marginTop: '0.3rem' }}>
            Threads per block = {count(threads)} ({plural(threads / width, 'warp')})
            <input type="range" min={tValues[0]} max={tValues[tValues.length - 1]} step={width} value={threads} aria-label="Threads per block"
              style={SLIDER} onChange={(e) => setThreads(+e.target.value)} />
          </label>
          <p className="muted" style={{ margin: '0.5rem 0 0' }}>
            One bar for each whole number of warps. This one is not a staircase: it dips wherever a whole number of
            blocks cannot use up the budget that binds, as 768 cannot use up 2048 thread slots.
          </p>
        </div>
      </div>
      <div className="panel" aria-live="polite" style={{ marginTop: '1.2rem' }}>
        <p style={{ margin: '0 0 0.4rem' }}>
          {count(threads)} threads per block at {regs} registers per thread:{' '}
          <b>
            <output data-out="cliff-occupancy" data-value={pct(o.byWarps)} data-blocks={o.blocks} data-binding={o.binding.join('+')}>
              {plural(o.blocks, 'block')}, {pct(o.byWarps)}%
            </output>
          </b>
          {o.blocks > 0 ? <>. {cap1(listOf(o.binding))} {o.binding.length === 1 && o.binding[0] === 'shared' ? 'binds' : 'bind'}.</> : <>. Not even one block fits.</>}
        </p>
        <p style={{ margin: 0 }} data-out="next-step">
          {!hasRegs ? (
            <>Registers are not counted for this device.</>
          ) : next ? (
            <>
              {next.at - regs === 1 ? <b>One more register per thread</b> : <b>{next.at - regs} more registers per thread</b>} and a block
              stops fitting: occupancy goes from {pct(next.from)}% to {pct(next.to)}%, which is{' '}
              {Math.abs(lost - 0.5) < 1e-9 ? 'half of the SM' : Math.abs(lost - 0.25) < 1e-9 ? 'a quarter of the SM' : `${pct(lost)}% of the SM`}.
            </>
          ) : (
            <>No further step: nothing of this kernel fits at this register count or above.</>
          )}
        </p>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- exercises 7 and 8
const EX7_CHOICES: [string, string][] = [
  ['', 'your answer'],
  ['no', 'not possible'],
  ['25', 'possible, 25%'],
  ['50', 'possible, 50%'],
  ['75', 'possible, 75%'],
  ['100', 'possible, 100%'],
];
const EX8_CHOICES: [string, string][] = [
  ['', 'your answer'],
  ['full', 'full occupancy'],
  ['threads', 'no: thread slots'],
  ['blocks', 'no: block slots'],
  ['regs', 'no: registers'],
];
const LETTER = ['a', 'b', 'c', 'd', 'e'];

/** What the model says for each part, in the vocabulary of the answer lists. */
export const EX7_EXPECTED = EX7_ASSIGNMENTS.map((a) => {
  const r = assignment(EX7, a.blocks, a.threads);
  return r.possible ? String(pct(r.occupancy)) : 'no';
});
export const EX8_EXPECTED = EX8_KERNELS.map((k) => {
  const o = occupancy(EX8, k);
  return o.byWarps === 1 ? 'full' : o.binding[0];
});

function Exercises() {
  const [a7, setA7] = useState<string[]>(EX7_ASSIGNMENTS.map(() => ''));
  const [a8, setA8] = useState<string[]>(EX8_KERNELS.map(() => ''));
  const [shown7, setShown7] = useState(false);
  const [shown8, setShown8] = useState(false);
  const mark = (ok: boolean) => <span className={'chip ' + (ok ? 'ok' : 'fail')}>{ok ? 'right' : 'not this'}</span>;
  const pick = (value: string, set: (v: string) => void, choices: [string, string][], label: string) => (
    <select value={value} aria-label={label} onChange={(e) => set(e.target.value)}>
      {choices.map(([v, text]) => (
        <option key={v} value={v} disabled={v === ''}>
          {text}
        </option>
      ))}
    </select>
  );
  const right7 = a7.filter((v, i) => v === EX7_EXPECTED[i]).length;
  const right8 = a8.filter((v, i) => v === EX8_EXPECTED[i]).length;

  return (
    <Bench
      title="Exercises 7 and 8"
      task="Answer every part from the limits alone, then check. The bars appear after your answers, not before."
      foot="The devices and kernels are the book's (pp. 91–92). The book prints no solutions: the answers shown are this page's model."
    >
      <div className="cols">
        <div className="panel" data-ex="7">
          <p style={{ margin: '0 0 0.6rem' }}>
            <b>Exercise 7.</b> A device allows {count(EX7.blockSlots as number)} blocks and {count(EX7.threadSlots as number)} threads on an
            SM. For each assignment of blocks to one SM: is it possible, and at what occupancy?
          </p>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr><th>On one SM</th><th>Your answer</th><th>{shown7 ? 'The model' : ''}</th></tr>
            </thead>
            <tbody>
              {EX7_ASSIGNMENTS.map((a, i) => {
                const r = assignment(EX7, a.blocks, a.threads);
                return (
                  <tr key={i} data-part={LETTER[i]}>
                    <td>({LETTER[i]}) {a.blocks} blocks of {a.threads} threads</td>
                    <td>{pick(a7[i], (v) => { setA7(a7.map((x, k) => (k === i ? v : x))); setShown7(false); }, EX7_CHOICES, `Exercise 7 (${LETTER[i]})`)}</td>
                    <td data-model={shown7 ? EX7_EXPECTED[i] : undefined}>
                      {shown7 ? (
                        <>
                          {mark(a7[i] === EX7_EXPECTED[i])} {a.blocks} of {EX7.blockSlots} block slots, {count(a.blocks * a.threads)} of{' '}
                          {count(EX7.threadSlots as number)} thread slots: {r.possible ? `possible, ${pct(r.occupancy)}%` : 'not possible'}
                        </>
                      ) : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="row" style={{ marginTop: '0.7rem' }}>
            <button className="primary" disabled={a7.some((v) => v === '')} onClick={() => setShown7(true)}>
              Check exercise 7
            </button>
            {shown7 ? <span data-out="ex7-score">{right7} of {a7.length} right.</span> : <span className="muted">Answer all five first.</span>}
          </div>
          {shown7 ? (
            <p className="muted" style={{ margin: '0.6rem 0 0' }}>
              All five fit within both limits. What differs is how full the SM is. Three of them leave half the thread
              slots empty: with those block sizes the SM could hold twice as many blocks as the assignment puts on it.
            </p>
          ) : null}
        </div>
        <div className="panel" data-ex="8">
          <p style={{ margin: '0 0 0.6rem' }}>
            <b>Exercise 8.</b> A device allows {count(EX8.threadSlots as number)} threads, {count(EX8.blockSlots as number)} blocks and{' '}
            {count(EX8.regs as number)} registers on an SM. Can each kernel reach full occupancy? If not, what stops it?
          </p>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr><th>Kernel</th><th>Your answer</th></tr>
            </thead>
            <tbody>
              {EX8_KERNELS.map((k, i) => (
                <tr key={i} data-part={LETTER[i]}>
                  <td>({LETTER[i]}) {k.threads} threads per block, {k.regs} registers per thread</td>
                  <td data-model={shown8 ? EX8_EXPECTED[i] : undefined}>
                    {pick(a8[i], (v) => { setA8(a8.map((x, j) => (j === i ? v : x))); setShown8(false); }, EX8_CHOICES, `Exercise 8 (${LETTER[i]})`)}{' '}
                    {shown8 ? mark(a8[i] === EX8_EXPECTED[i]) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="row" style={{ marginTop: '0.7rem' }}>
            <button className="primary" disabled={a8.some((v) => v === '')} onClick={() => setShown8(true)}>
              Check exercise 8
            </button>
            {shown8 ? <span data-out="ex8-score">{right8} of {a8.length} right.</span> : <span className="muted">Answer all three first.</span>}
          </div>
        </div>
      </div>
      {shown8
        ? EX8_KERNELS.map((k, i) => {
            const o = occupancy(EX8, k);
            return (
              <div className="panel" key={i} data-ex8-bars={LETTER[i]} style={{ marginTop: '1rem' }}>
                <p style={{ margin: '0 0 0.5rem' }}>
                  <b>8({LETTER[i]})</b> {k.threads} threads per block, {k.regs} registers per thread:{' '}
                  <b>{plural(o.blocks, 'block')}, {pct(o.byWarps)}%</b>. {verdictOf(o, k, EX8.threadSlots)}
                </p>
                <BudgetTable o={o} />
              </div>
            );
          })
        : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- lab 4.4, once it has run
const LAB4 = 'ch04/lab4_blocksize';
const SIZE_KEYS = ['threads_per_block', 'block', 'block_size', 'threadsPerBlock', 'threads_per_threadgroup', 'threadgroup'];

interface SizeTime {
  size: number;
  seconds: number;
}
/** The rows of a lab 4.4 benchmark that give a block size and a time, one per size. */
export function sizeTimes(rows: BenchRow[]): SizeTime[] {
  const out: SizeTime[] = [];
  let name: string | null = null;
  for (const r of rows) {
    const seconds = num(r.seconds);
    const k = SIZE_KEYS.find((key) => num(r[key]) != null);
    if (!k || seconds == null || !(seconds > 0)) continue;
    if (name == null) name = r.name;
    if (r.name !== name) continue; // another series of the same benchmark
    const size = num(r[k]) as number;
    if (!out.some((x) => x.size === size)) out.push({ size, seconds });
  }
  return out.sort((a, b) => a.size - b.size);
}

function BlockSizeReveal({ committed }: { committed: number }) {
  const { reports, loading, reload } = useReports();
  const sets = realBackends(reports).flatMap((b) => {
    const run = reports[b];
    const mine = labStatus(run, LAB4, 'work') === 'pass' ? sizeTimes(benchRows(run, LAB4, 'work')) : [];
    const rows = mine.length >= 2 ? mine : sizeTimes(benchRows(run, LAB4, 'solution'));
    if (rows.length < 2) return [];
    return [{ backend: b, device: String(run.device.name ?? b), date: run.generated.slice(0, 10), own: mine.length >= 2, rows, limits: limitsFromReport(run.device as Record<string, unknown>, b) }];
  });
  if (!sets.length)
    return loading ? (
      <p data-measured="loading">Reading reports.</p>
    ) : (
      <p data-measured="none">
        No measurement on this machine yet. Lab 4.4 on the <a href={href('ch04', 'labs')}>labs page</a> times one kernel
        at several block sizes; run its benchmark with <code>python3 labs/pmpp.py run 4.4 --bench</code>, or{' '}
        <code>./verify.sh</code> without <code>--quick</code>. Your prediction is stored and will be set beside the
        result.{' '}
        <button className="quiet" onClick={reload}>
          Reload results
        </button>
      </p>
    );
  return (
    <>
      {sets.map((s) => {
        const fastest = s.rows.reduce((a, b) => (b.seconds < a.seconds ? b : a));
        const slowest = s.rows.reduce((a, b) => (b.seconds > a.seconds ? b : a));
        const ratio = slowest.seconds / fastest.seconds;
        const close = committed > 0 && Math.abs(committed / ratio - 1) <= 0.1;
        return (
          <div key={s.backend} data-measured={fmt(ratio)} data-backend={s.backend}>
            <p>
              Measured on {s.device} ({s.backend}) with {s.own ? 'your kernel' : 'the reference solution'}, in the run of {s.date}:
              the slowest block size ({count(slowest.size)} threads) took <b>{fmt(ratio)}</b> times as long as the fastest (
              {count(fastest.size)} threads).{' '}
              {!(committed > 0)
                ? 'Your prediction was not a positive ratio, so there is nothing to compare.'
                : close
                  ? 'Your prediction was within 10% of that.'
                  : committed < ratio
                    ? `That is ${fmt(ratio / committed)} times your prediction.`
                    : `Your prediction is ${fmt(committed / ratio)} times that.`}
            </p>
            <div style={{ overflowX: 'auto' }}>
              <table className="data" style={{ margin: '0 0 0.8rem' }}>
                <thead>
                  <tr>
                    <th className="num">Threads per block</th>
                    <th className="num">Time (ms)</th>
                    <th className="num">Against the fastest</th>
                    <th className="num">Occupancy by slots alone (model)</th>
                  </tr>
                </thead>
                <tbody>
                  {s.rows.map((r) => {
                    const o = s.limits ? occupancy({ ...s.limits, regs: null, shared: null }, { threads: r.size, regs: 0, shared: 0 }) : null;
                    return (
                      <tr key={r.size} data-size={r.size}>
                        <td className="num">{count(r.size)}</td>
                        <td className="num">{fmt(r.seconds * 1e3)}</td>
                        <td className="num">{fmt(r.seconds / fastest.seconds)}×</td>
                        <td className="num" data-model={o && o.usable ? pct(o.byWarps) : 'none'}>{o && o.usable ? pct(o.byWarps) + '%' : 'n/a'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="muted">
              {s.limits
                ? 'The last column is the arithmetic of this section on the limits your device reported, with registers left out: a device report does not carry a kernel\'s register count. Lab 4.4 sets the vendor\'s own occupancy function beside it.'
                : 'There is no occupancy column for this device: its report has no per-SM limits. On Metal none are documented, so the times are all there is to go on.'}
            </p>
          </div>
        );
      })}
      <p className="muted">
        A measurement from your machine, not an expected value. Where the times follow the model's occupancy, latency
        tolerance is the likely reason. Where they do not, something the model leaves out is at work, memory bandwidth
        first among them.
      </p>
    </>
  );
}

// ---------------------------------------------------------------- the occupancy call, from lab 4.4
function OccupancyCall() {
  const code = useLabFile(LAB4 + '/provided/occupancy_api.cu');
  if (!code) return null;
  return (
    <Code
      tabs={[{ label: 'CUDA and HIP', code, status: 'front-end-checked; unverified on CUDA and HIP hardware' }]}
      caption={
        <>
          The call lab 4.4's benchmark makes, loaded from <code>labs/{LAB4}/provided/occupancy_api.cu</code>. It has
          been parsed by clang against stub headers and never compiled by nvcc or a HIP compiler.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
const SLOTS_ONLY: Limits = { ...BOOK, regs: null, shared: null };
const O48 = occupancy(SLOTS_ONLY, { threads: 48, regs: 0, shared: 0 });
const CLIFF_A = occupancy(BOOK, { threads: 512, regs: 31, shared: 0 });
const CLIFF_B = occupancy(BOOK, { threads: 512, regs: 33, shared: 0 });
const SMALL = occupancy(SLOTS_ONLY, { threads: 32, regs: 0, shared: 0 });
const EX8C = occupancy(EX8, EX8_KERNELS[2]);

export const questionsS47: Record<string, Question> = {
  'ch04/4.7/q-ex6': {
    kind: 'choice',
    id: 'ch04/4.7/q-ex6',
    prompt: (
      <>
        Book exercise 6. An SM can hold up to 1536 threads and up to 4 blocks. Which block size puts the most threads on
        it?
      </>
    ),
    options: [
      { label: '128 threads per block', why: 'Four block slots hold 4 × 128 = 512 threads. The block slots run out with two thirds of the thread slots empty.' },
      { label: '256 threads per block', why: 'Four blocks, 1024 threads. Block slots run out first again.' },
      { label: '512 threads per block', correct: true },
      { label: '1024 threads per block', why: 'One block fits; a second would need 2048 slots. 1024 threads.' },
    ],
    explain: (
      <p>
        Count each limit separately and take the smaller. At 512 threads the thread slots allow 1536 / 512 = 3 blocks and
        the block slots allow 4, so 3 blocks and 1536 threads: full. The two smaller sizes are stopped by the 4 block
        slots, the largest by a size that does not divide 1536.
      </p>
    ),
    source: 'ch. 4 exercise 6, p. 91; the answer is ours (the book prints none)',
  },
  'ch04/4.7/q-ex8c': {
    kind: 'numeric',
    id: 'ch04/4.7/q-ex8c',
    prompt: (
      <>
        Book exercise 8(c). An SM allows 2048 threads, 32 blocks and 65,536 registers. A kernel runs with 256 threads per
        block and uses 34 registers per thread. What is its occupancy, in percent?
      </>
    ),
    answer: pct(EX8C.byWarps),
    tol: 0.001,
    unit: '%',
    explain: (
      <p>
        One block needs 256 × 34 = 8,704 registers, and 65,536 / 8,704 is 7 and a bit, so {EX8C.blocks} blocks fit by
        registers. Thread slots would allow 8 and block slots 32. Seven blocks are {count(EX8C.residentThreads)} threads
        of 2048: {pct(EX8C.byWarps)}%. Registers are the limiting factor, and the two registers above 32 cost an eighth
        of the SM.
      </p>
    ),
    source: "ch. 4 exercise 8, p. 92, worked with the book's arithmetic of p. 86; the answer is ours",
  },
  'ch04/4.7/q-768': {
    kind: 'numeric',
    id: 'ch04/4.7/q-768',
    prompt: (
      <>
        A device reports 1536 thread slots and 16 block slots per SM. A kernel is launched with 768 threads per block and
        few enough registers that they do not matter. What is its occupancy, in percent?
      </>
    ),
    answer: pct(occupancy({ ...TABLE_1536, regs: null }, { threads: 768, regs: 0, shared: 0 }).byWarps),
    tol: 0.001,
    unit: '%',
    explain: (
      <p>
        1536 / 768 = 2 blocks exactly, well within 16 block slots: every thread slot is taken. On the book's example the
        same launch gives 75%, because 768 does not divide 2048. A block size is good or bad for a particular device,
        which is the reason to read the limits and not remember them.
      </p>
    ),
    source: "NVIDIA's table of per-SM limits, read 2026-11-20 (ch04-facts.md, A12); the arithmetic is the book's, p. 86",
  },
  'ch04/4.7/q-twodefs': {
    kind: 'choice',
    id: 'ch04/4.7/q-twodefs',
    prompt: (
      <>
        Blocks of 48 threads on the book's example device (32 block slots, 2048 thread slots, warps of 32). The book
        defines occupancy with warps on p. 85 and with threads in its summary on p. 90. What does each definition give?
      </>
    ),
    options: [
      { label: '75% by warps and 75% by threads.', why: 'By threads, yes. But each block is two warps, and 32 blocks of two warps take all 64 warp slots.' },
      {
        label: `${pct(O48.byWarps)}% by warps and ${pct(O48.byThreads)}% by threads.`,
        correct: true,
      },
      { label: '100% by both: the SM is full either way.', why: 'Every slot is held, but 16 lanes of every second warp run no thread. Counting threads shows that.' },
      { label: '50% by warps and 75% by threads.', why: 'A 48-thread block is two warps, not one: the second is padded with 16 inactive lanes (section 4.4).' },
    ],
    explain: (
      <>
        <p>
          A block of 48 threads is cut into two warps, the second padded with 16 lanes that run nothing. All{' '}
          {O48.blocks} block slots are used, which makes {O48.residentWarps} warps: every warp slot. So {pct(O48.byWarps)}
          % by warps. Counted in threads it is {O48.blocks} × 48 = {count(O48.residentThreads)} of 2048,{' '}
          {pct(O48.byThreads)}%.
        </p>
        <p>
          Both are right about something. The warp figure says how many warps the scheduler has to choose from, which
          is what latency tolerance needs. The thread figure says how much work those slots carry. The{' '}
          {count(O48.paddedLanes)} padded lanes hold slots and do nothing. With whole-warp block sizes the two
          definitions agree, which is one more reason to use them.
        </p>
      </>
    ),
    source: "ch. 4 §4.7, p. 85 and §4.9, p. 90; padding from §4.4, p. 75; NVIDIA defines occupancy with warps (ch04-facts.md, A13)",
  },
  'ch04/4.7/q-regs64': {
    kind: 'numeric',
    id: 'ch04/4.7/q-regs64',
    prompt: (
      <>
        A kernel needs 64 registers per thread. On the book's example device (65,536 registers and 2048 thread slots per
        SM), what is the highest occupancy any block size can give it, in percent?
      </>
    ),
    answer: 50,
    tol: 0,
    unit: '%',
    explain: (
      <p>
        65,536 / 64 = 1024 threads' worth of registers, however the threads are grouped into blocks. That is half of the
        2048 thread slots. Changing the block size cannot help; only using fewer registers per thread can, and the
        compiler's way of doing that (spilling values to memory) costs time on every access.
      </p>
    ),
    source: 'ch. 4 §4.7, p. 86',
  },
  'ch04/4.7/q-hipslots': {
    kind: 'choice',
    id: 'ch04/4.7/q-hipslots',
    prompt: (
      <>
        A HIP program on an AMD device reads <code>maxBlocksPerMultiProcessor</code> and gets some number N. Is N a limit
        on blocks of the same kind as the book's 32?
      </>
    ),
    options: [
      { label: 'Yes. HIP mirrors the CUDA field, so it is the number of block slots of a compute unit.', why: 'The name is mirrored. The value is computed, and from something else.' },
      {
        label: 'No. It is thread slots divided by the wavefront width: the number of wave slots, not a separate limit.',
        correct: true,
      },
      { label: 'No. It counts the blocks the whole device can hold at once, not the blocks of one compute unit.', why: 'It is per compute unit. What differs is that it is not an independent limit.' },
      { label: 'Yes, and it is smaller than NVIDIA\'s 32, because wider wavefronts leave room for fewer blocks.', why: 'Nothing of the kind is documented, and width varies by AMD product family.' },
    ],
    explain: (
      <p>
        On the book's device, blocks of one warp hit the 32 block slots with half the thread slots empty. AMD's
        description of a compute unit has no block-slot limit, and the runtime fills this field with thread slots /
        width. So the 50% case of this section has no AMD counterpart, and a calculator that treats N as a second limit
        would invent one. What does bind on AMD is wave slots, two register pools and shared memory.
      </p>
    ),
    source: 'HIP runtime source and docs, read 2026-11-20 (ch04-facts.md, B9, B10); not in the book',
  },
};

// ---------------------------------------------------------------- the section
export function S47() {
  return (
    <>
      <p>
        <a href={href('ch04', '4.6')}>Section 4.6</a> ended on a bill. Every warp that is resident on an SM holds a
        place there, and its registers, for as long as it stays. This section does the accounting: given what one SM
        has and what one block of your kernel asks for, how many blocks fit, and how full is the SM then?
      </p>

      <h2>Occupancy</h2>
      <p>
        The measure is <b>occupancy</b>: the number of warps assigned to an SM, divided by the most warps that SM can
        hold. An SM holding half the warps it could is at 50%. By section 4.6 that means half the cover for waits on
        memory that it could have had.
      </p>
      <Source section="4.7" pages="85" />
      <p>
        The book's summary states the same ratio with threads in place of warps. The two agree whenever blocks are
        whole warps, and part company when they are not. This page uses warps and shows the other figure where it
        differs.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 guide (ch04-facts.md, A13)">
        <p>
          NVIDIA defines it with warps too: active warps over the most active warps an SM supports. The threads
          version is on the book's p. 90.
        </p>
      </Today>

      <h2>What is shared out</h2>
      <p>The streaming processors are not what runs out. Four other things are, and each is a fixed amount per SM:</p>
      <ul>
        <li><b>Block slots</b>: how many blocks can be resident at once.</li>
        <li><b>Thread slots</b>: how many threads, which is the same as how many warps.</li>
        <li><b>Registers</b>: one pool, drawn on by every resident thread.</li>
        <li>
          <b>Shared memory</b>: one pool, drawn on by every resident block.{' '}
          <a href={href('ch05', '5.6')}>Section 5.6</a> adds it to this arithmetic.
        </li>
      </ul>
      <p>
        None of them is cut into fixed shares. A block takes what it needs from each pool when it is assigned and
        gives it back when it finishes. That is what lets the same SM hold many small blocks or a few large ones. With
        fixed shares, a small block would waste most of its share and a block larger than a share could not run at all.
      </p>
      <Source section="4.7" pages="85" />

      <h2>Thread slots and block slots</h2>
      <p>
        The book's example device allows 32 blocks and 2048 threads (64 warps) on an SM, and 1024 threads in a block.
        Start with large blocks. Two blocks of 1024 threads fill the 2048 slots. So do four of 512, eight of 256,
        sixteen of 128 and thirty-two of 64. Each of those is 100%.
      </p>
      <Source section="4.7" pages="85" />
      <Predict
        id="ch04/4.7/p-32"
        prompt={
          <>
            Go one step smaller on the same device: blocks of 32 threads. What is the occupancy, in percent?
          </>
        }
        unit="%"
      >
        {(v) => (
          <>
            <p>
              <b>{pct(SMALL.byWarps)}%.</b> Filling 2048 thread slots with 32-thread blocks would take 64 blocks, and
              the SM has 32 block slots. So {SMALL.blocks} blocks, {count(SMALL.residentThreads)} threads, half the
              slots empty.{' '}
              {v === pct(SMALL.byWarps) ? 'That is what you had.' : v === 100 ? 'The thread slots alone would give 100%; the other limit got there first.' : ''}
            </p>
            <p>
              On this device a block needs at least 2048 / 32 = {smallestFillingBlock(BOOK)} threads before full
              occupancy is possible at all.
            </p>
          </>
        )}
      </Predict>
      <p>
        That is the first way the limits interact: a block can be so small that the block slots run out before the
        thread slots do. The second is plainer. A block of 768 threads does not divide 2048. Two fit, a third does
        not, 512 slots stay empty, and occupancy is 75% with neither limit reached.
      </p>
      <Source section="4.7" pages="85–86" />
      <p>
        The calculator holds this arithmetic for any limits. Each row is one budget: what an SM has, what one block
        takes, and how many blocks that budget alone would allow. The SM gets the smallest of those counts. The
        buttons are the seven block sizes the book works through, and one it does not.
      </p>
      <Simplification>
        <p>
          Registers are counted per thread: threads × registers per thread. That is the book's model. NVIDIA says
          registers are shared out among warps and does not publish in what amounts, so for a real kernel the
          vendor's occupancy function is the authority.
        </p>
        <p>Padded lanes are charged a thread slot and no registers. Shared memory the driver reserves is ignored.</p>
      </Simplification>
      <OccupancyCalculator />
      <p>
        Every one of the book's worked cases belongs to its example device. Change the limits and the same block size
        gives a different answer. Work the next question out by hand before you touch the calculator.
      </p>
      <Today checked="2026-11-20 against NVIDIA's compute-capability tables (ch04-facts.md, A12, A15)">
        <p>
          Current NVIDIA GPUs have 16, 24 or 32 block slots and 1024, 1536 or 2048 thread slots per SM, depending on
          the compute capability, and 64 K registers on all of them. A higher capability does not mean more per SM.
        </p>
      </Today>
      <Ask q={questionsS47['ch04/4.7/q-768']} />
      <p>
        Then check it. The list of limits in the calculator has a second set from NVIDIA's own table, with 1536 thread
        slots and 16 block slots. Choose it, press 768, then 1024, and set both beside what the book's example gave.
      </p>
      <Ask q={questionsS47['ch04/4.7/q-ex6']} />
      <Ask q={questionsS47['ch04/4.7/q-twodefs']} />

      <h2>Registers</h2>
      <p>
        Automatic variables in a kernel live in registers (<a href={href('ch05', '5.2')}>section 5.2</a>), so every
        thread of a kernel needs some number of them, and that number is fixed when the kernel is compiled. Registers
        come from a pool per SM as well. The book's example has 65,536. Divide by the 2048 thread slots: a kernel that
        needs up to {regsAtFullOccupancy(BOOK)} registers per thread can fill every slot. A kernel that needs 64 can
        have at most 1024 threads resident, whatever its block size.
      </p>
      <Source section="4.7" pages="86" />
      <p>
        The compiler can bring the count down by <b>spilling</b>: keeping some values in memory and fetching them when
        they are needed. More blocks then fit. Each access to a spilled value takes longer, and the book warns that
        the grid as a whole may come out slower.
      </p>
      <Ask q={questionsS47['ch04/4.7/q-regs64']} />

      <h2>The cliff</h2>
      <p>
        Occupancy does not fall smoothly as a kernel grows. Blocks fit whole or not at all, so it falls in steps, and a
        step can be large.
      </p>
      <Source section="4.7" pages="86" />
      <Predict
        id="ch04/4.7/p-cliff"
        prompt={
          <>
            On the book's example device a kernel runs with 512 threads per block and uses 31 registers per thread. Four
            blocks fit: 100%. You declare two more local variables and the compiler now needs 33 registers per thread.
            What is the occupancy, in percent?
          </>
        }
        unit="%"
      >
        {(v) => (
          <>
            <p>
              <b>{pct(CLIFF_B.byWarps)}%.</b> At 31 registers the four blocks need 2048 × 31 = {count(CLIFF_A.blocks * 512 * 31)}{' '}
              registers, under 65,536. At 33 they would need 2048 × 33 = {count(2048 * 33)}, which is over. The fourth
              block no longer fits, {CLIFF_B.blocks} blocks stay ({count(CLIFF_B.residentThreads)} threads,{' '}
              {count(CLIFF_B.blocks * 512 * 33)} registers), and a quarter of the SM is empty.{' '}
              {v === pct(CLIFF_B.byWarps) ? 'That is what you had.' : v > 90 ? 'A drop of a few percent would be the smooth answer. Blocks are not divisible.' : ''}
            </p>
            <p>
              Two registers, a quarter of the parallelism. The book calls a small increase in resource use that causes
              a large drop like this a <b>performance cliff</b>.
            </p>
          </>
        )}
      </Predict>
      <p>
        The explorer draws the whole staircase. On the left, occupancy as registers per thread go from 1 to 255 at the
        block size you choose. On the right, occupancy for every block size at the register count you choose. Move
        either slider and the other plot is redrawn. Look for three things: where the first step is, how much a single
        step can cost at 1024 threads per block, and what happens to all of it on other limits.
      </p>
      <CliffExplorer />
      <Ask q={questionsS47['ch04/4.7/q-ex8c']} />
      <p>
        Larger blocks make taller steps, because each block that stops fitting takes more warps with it. And the
        positions of the steps belong to the device: at 512 threads per block the first one is at 33 registers on the
        book's example and at 43 on the 1536-slot limits.
        A kernel tuned to sit just under a step on one GPU can be just over it on another.
      </p>

      <h2>Finding the numbers for a real kernel</h2>
      <p>
        The budgets interact, the compiler decides the register count, and the hardware shares registers out in ways
        this arithmetic only approximates. So nobody works out a real kernel's occupancy by hand. The book names two
        tools: the Occupancy Calculator that is part of the Nsight Compute profiler, and occupancy functions you call
        from host code, such as <code>cudaOccupancyMaxActiveBlocksPerMultiprocessor</code>.
      </p>
      <Source section="4.7" pages="87" />
      <p>
        These are the ones you will meet. None of them was run on this machine: on CUDA all of this is unverified
        here, and the names and meanings are NVIDIA's documentation.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 guide, the nvcc and runtime references and the Nsight Compute 2026.4 docs (ch04-facts.md, A13, A14)">
        <p>All of the names in the table exist with these meanings. The Occupancy Calculator replaced the older spreadsheet.</p>
      </Today>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-tools>
          <thead>
            <tr><th style={{ width: '38%' }}>What</th><th>What it tells you, or does</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><code>nvcc --resource-usage</code></td>
              <td>At compile time: the registers and shared memory each kernel needs.</td>
            </tr>
            <tr>
              <td><code>cudaFuncGetAttributes</code>, field <code>numRegs</code></td>
              <td>At run time: the registers each thread of a kernel uses.</td>
            </tr>
            <tr>
              <td><code>cudaOccupancyMaxActiveBlocksPerMultiprocessor</code></td>
              <td>For one kernel, block size and amount of dynamic shared memory: how many blocks fit on an SM. It returns a block count, not a percentage.</td>
            </tr>
            <tr>
              <td><code>cudaOccupancyMaxPotentialBlockSize</code></td>
              <td>Suggests a block size and a smallest grid that give the kernel its best possible occupancy. C++ only.</td>
            </tr>
            <tr>
              <td>Occupancy Calculator in Nsight Compute (<code>ncu-ui</code>)</td>
              <td>The same for every block size at once, as tables and a graph.</td>
            </tr>
            <tr>
              <td><code>__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, …)</code></td>
              <td>Bounds written on the kernel definition, one of the ways NVIDIA lists to steer register use. A launch with more threads per block than the first bound fails.</td>
            </tr>
            <tr>
              <td><code>nvcc --maxrregcount N</code>, or <code>__maxnreg__(N)</code> on one kernel</td>
              <td>Caps registers per thread. A kernel that needs more is likely to spill to memory; NVIDIA says the extra blocks that then fit may still give a net gain.</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        To turn the occupancy function's answer into this section's percentage, round the block size up to whole warps
        and divide by the thread slots: numBlocks × ceil(blockSize / warpSize) × warpSize /
        maxThreadsPerMultiProcessor. Lab 4.4 makes that call for a kernel of yours and sets the result beside your own
        model:
      </p>
      <OccupancyCall />

      <h2>On AMD, through HIP</h2>
      <p>
        AMD defines occupancy the same way, active warps over the most a compute unit can hold, and HIP has the
        occupancy functions under the same names with a <code>hip</code> prefix. What is shared out is different: wave
        slots, a pool of vector registers, a pool of scalar registers, and shared memory, which AMD calls LDS.
        Registers are given out per wave, in fixed-size chunks, not per thread.
      </p>
      <Beyond>
        <p>
          The book covers CUDA only. This is from AMD's HIP documentation and the runtime's source, read on 2026-11-20
          (ch04-facts.md, B9, B10). Nothing here was run on AMD hardware.
        </p>
      </Beyond>
      <p>Three consequences for code that has to run on both vendors:</p>
      <ul>
        <li>
          There is no separate block-slot limit. <code>maxBlocksPerMultiProcessor</code> exists in{' '}
          <code>hipDeviceProp_t</code>, but the runtime computes it as thread slots divided by the wavefront width, so
          it is the number of wave slots. The 32-thread, 50% case above has no AMD counterpart.
        </li>
        <li>
          A block is rounded up to whole waves before anything is divided. On a 64-wide device a 65-thread block costs
          128 thread slots.
        </li>
        <li>
          <code>__launch_bounds__</code> is spelt the same and means something else in its second argument: the minimum
          number of warps per execution unit (a compute unit has four), where CUDA's is the minimum number of blocks
          per SM.
        </li>
      </ul>
      <Ask q={questionsS47['ch04/4.7/q-hipslots']} />

      <h2>On Metal there is nothing to compute</h2>
      <p>
        Apple documents no occupancy figure, no limit on resident threads, no block slots, no register pool, no
        per-kernel register count and no function that returns any of them. That is why the calculator has no entry
        for a Metal device: there are no numbers to put in it, and this page will not invent them.
      </p>
      <Beyond>
        <p>
          Checked 2026-11-20 against the Metal Shading Language specification 4.1, Apple's pages and the macOS SDK
          headers (ch04-facts.md, C5). Nothing was found, which is the finding.
        </p>
      </Beyond>
      <p>
        The nearest documented behaviour is that the thread limit of a threadgroup belongs to the compiled kernel, not
        to the device. <code>maxTotalThreadsPerThreadgroup</code> is a property of a pipeline state, and Apple says it
        can differ between pipelines and depends on the kernel's memory use. That resembles "a kernel that needs more
        gets a smaller block", but Apple does not explain it in terms of registers or occupancy, and the resemblance
        is ours. On Metal the block-size question is settled by measuring.
      </p>

      <h2>More is not always better</h2>
      <p>
        Occupancy is a means. Section 4.6 showed what it buys: enough ready warps to cover the waits. Once a kernel has
        that many, more warps add nothing at the execution units, and the registers or shared memory they take have to
        come from somewhere. A kernel that waits on memory almost every step wants the SM full. A kernel that computes
        for a long time on values in registers may do best with few warps and many registers each, which is how
        chapter 15's matrix multiplication ends up at 12.5% on purpose.
      </p>
      <Beyond>
        <p>
          The 12.5% is the book's (ch. 15, p. 363). AMD's guide says it outright: past the point of hiding latency,
          more occupancy can hurt, and memory-bound kernels are the ones that gain from it (ch04-facts.md, B10).
        </p>
      </Beyond>

      <h2>The book's exercises</h2>
      <p>
        Exercise 6 was a question above. Exercises 7 and 8 are here, with one difference from the calculator: you
        answer first.
      </p>
      <Source section="4.10" pages="91–92" />
      <Exercises />

      <h2>Measure it</h2>
      <p>
        All of this predicts that block size alone can change how fast a memory-bound kernel runs, on a device where
        small blocks leave slots empty. Lab 4.4 on the <a href={href('ch04', 'labs')}>labs page</a> tests that on your
        GPU: one kernel that sums table entries chosen by an index array, so that nearly every step is a load from
        global memory, timed at a range of block sizes. Commit to a number before you run it.
      </p>
      <Predict
        id="ch04/4.7/p-blocksize"
        prompt={
          <>
            On your device, for a kernel that mostly waits on memory: how many times as long will the slowest block size
            take as the fastest? (1 means block size makes no difference.)
          </>
        }
        unit="times"
      >
        {(v) => <BlockSizeReveal committed={v} />}
      </Predict>
      <p>
        One input to every calculation on this page has been "the limits of one SM". Where they come from, for each of
        the three backends, is <a href={href('ch04', '4.8')}>section 4.8</a>.
      </p>
    </>
  );
}
