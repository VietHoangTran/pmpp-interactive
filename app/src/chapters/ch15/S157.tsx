import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_LIMITS,
  BOOK_SHAPE,
  MAX_REGS_PER_THREAD,
  budget,
  regsForBlocks,
  type LimitKey,
  type Limits,
  type Shape,
} from '../../sim/gemmbudget';

const KB = 1024;
const count = (v: number) => v.toLocaleString('en-US');
const bytes = (v: number) => (v >= KB ? `${count(v)} B (${fmt(v / KB)} KB)` : `${count(v)} B`);

/** A long identifier that may break anywhere, for the narrow margin notes. */
function Id({ children }: { children: ReactNode }) {
  return <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>;
}

// ---------------------------------------------------------------- devices
interface Device {
  key: string;
  label: string;
  kind: 'book' | 'gpu' | 'metal' | 'custom';
  backend?: string;
  limits: Limits;
  note: ReactNode;
}

const BOOK: Device = {
  key: 'book',
  label: "The book's example SM",
  kind: 'book',
  limits: BOOK_LIMITS,
  note: (
    <>
      The book's figures: 65,536 registers, at most 255 per thread and 2048 resident threads per SM (§15.7, p. 363),
      and 228 KB of shared memory per SM on its example device (§5.6, p. 116).
    </>
  ),
};

const positive = (v: unknown): number => {
  const n = Number(v);
  return isFinite(n) && n > 0 ? Math.floor(n) : 0;
};

/** The reader's CUDA and HIP devices that report registers and thread slots per SM, and Metal devices. */
function reported(reports: Reports): Device[] {
  return realBackends(reports).flatMap((b): Device[] => {
    const d = reports[b].device;
    const name = String(d.name ?? b);
    const when = reports[b].generated.slice(0, 10);
    if ('maxThreadgroupMemoryLength' in d)
      return [
        {
          key: b,
          label: `Yours: ${name} (${b})`,
          kind: 'metal',
          backend: b,
          limits: { regsPerSm: 0, threadsPerSm: 0, maxRegsPerThread: 0, sharedPerBlock: positive(d.maxThreadgroupMemoryLength), sharedPerSm: 0 },
          note: (
            <>
              What this device reported when the labs last recorded it ({when}): <code>maxThreadgroupMemoryLength</code>.
              Apple documents no register count, no register file and no per-core thread limit, so only shared memory
              is checked (ch15-facts C3).
            </>
          ),
        },
      ];
    if (!('regsPerMultiprocessor' in d) || !('maxThreadsPerMultiProcessor' in d)) return [];
    return [
      {
        key: b,
        label: `Yours: ${name} (${b})`,
        kind: 'gpu',
        backend: b,
        limits: {
          regsPerSm: positive(d.regsPerMultiprocessor),
          threadsPerSm: positive(d.maxThreadsPerMultiProcessor),
          maxRegsPerThread: MAX_REGS_PER_THREAD[b] ?? 0,
          sharedPerBlock: positive(d.sharedMemPerBlock),
          sharedPerSm: positive(d.sharedMemPerMultiprocessor),
        },
        note: (
          <>
            What this device reported when the labs last recorded it ({when}). The per-thread maximum is the
            vendor's documented figure for {b === 'hip' ? 'AMD (256 vector registers, ch15-facts B3)' : 'NVIDIA (255, ch04-facts A12)'}.
            {b === 'hip'
              ? ' AMD allocates registers per wavefront in granules, so the per-thread product below is only the right shape there (ch04-facts B10).'
              : ''}
          </>
        ),
      },
    ];
  });
}

/**
 * What the chapter's labs printed about their own kernels on this backend (labs/ch15/shared/
 * stage_driver.hpp): on CUDA and HIP the row "occupancy" with blocks_per_sm from the occupancy
 * API (-1 when the runtime would not say); on Metal the row "pipeline" with the compiled
 * pipeline's thread cap and static threadgroup memory. Reference solutions only.
 */
function measured(reports: Reports, backend: string | undefined) {
  if (!backend || !reports[backend]) return [];
  return reports[backend].results
    .filter((r) => r.lab.startsWith('ch15/') && r.variant === 'solution')
    .flatMap((r) =>
      r.bench
        // Labs 15.1 to 15.5 write snake_case fields; lab 15.6 spells them as Metal does.
        .filter((row) => (num(row.blocks_per_sm) ?? -1) >= 0 || (num(row.max_threads_per_threadgroup) ?? num(row.maxTotalThreadsPerThreadgroup)) != null)
        .map((row) => ({
          lab: r.lab,
          blocks: num(row.blocks_per_sm),
          threads: num(row.threads_per_block),
          cap: num(row.max_threads_per_threadgroup) ?? num(row.maxTotalThreadsPerThreadgroup),
          staticBytes: num(row.static_threadgroup_bytes) ?? num(row.staticThreadgroupMemoryLength),
        })),
    );
}

// ---------------------------------------------------------------- the calculator
interface Preset {
  key: string;
  label: string;
  s: Partial<Shape>;
}
const PRESETS: Preset[] = [
  { key: 'book', label: 'The book: 8 × 8 at 255 registers', s: { ...BOOK_SHAPE, extra: 175 } },
  { key: 'tiles', label: 'Tile registers only (80)', s: { ...BOOK_SHAPE, extra: 0 } },
  { key: 'r128', label: 'If it fitted in 128', s: { ...BOOK_SHAPE, extra: 48 } },
  { key: 'pipe', label: 'Padded, double-buffered (15.6, 15.8)', s: { ...BOOK_SHAPE, extra: 175, pad: 1, buffers: 2 } },
];
type Field = 'tM' | 'tN' | 'extra' | 'threads' | 'bM' | 'bN' | 'bK' | 'pad';
const FIELDS: { key: Field; label: string; hint?: string }[] = [
  { key: 'tM', label: 'tM' },
  { key: 'tN', label: 'tN' },
  { key: 'extra', label: 'Extra registers', hint: 'addresses, counters, temporaries' },
  { key: 'threads', label: 'Threads per block' },
  { key: 'bM', label: 'bM' },
  { key: 'bN', label: 'bN' },
  { key: 'bK', label: 'bK' },
  { key: 'pad', label: 'Padding columns on A_s' },
];
const LIMIT_NAME: Record<LimitKey, string> = { registers: 'registers', threads: 'thread slots', shared: 'shared memory' };
const toInt = (s: string, min = 0) => {
  const n = Math.floor(Number(s));
  return isFinite(n) && n >= min ? n : min;
};
const CUSTOM_FIELDS: { key: keyof Limits; label: string }[] = [
  { key: 'regsPerSm', label: 'Registers per SM' },
  { key: 'threadsPerSm', label: 'Thread slots per SM' },
  { key: 'maxRegsPerThread', label: 'Most registers per thread' },
  { key: 'sharedPerSm', label: 'Shared memory per SM (bytes)' },
];

function BudgetBench() {
  const { reports, loading } = useReports();
  const devices = useMemo(() => [BOOK, ...reported(reports)], [reports]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState<Record<keyof Limits, string>>({
    regsPerSm: String(BOOK_LIMITS.regsPerSm),
    threadsPerSm: String(BOOK_LIMITS.threadsPerSm),
    maxRegsPerThread: String(BOOK_LIMITS.maxRegsPerThread),
    sharedPerBlock: '0',
    sharedPerSm: String(BOOK_LIMITS.sharedPerSm),
  });
  const [form, setForm] = useState<Record<Field, string>>(() => {
    const s = PRESETS[0].s as Shape;
    return { tM: String(s.tM), tN: String(s.tN), extra: String(s.extra), threads: String(s.threads), bM: String(s.bM), bN: String(s.bN), bK: String(s.bK), pad: String(s.pad) };
  });
  const [buffers, setBuffers] = useState<1 | 2>(1);

  const dev: Device =
    key === 'custom'
      ? {
          key: 'custom', label: 'Custom numbers', kind: 'custom',
          limits: {
            regsPerSm: toInt(custom.regsPerSm), threadsPerSm: toInt(custom.threadsPerSm), maxRegsPerThread: toInt(custom.maxRegsPerThread),
            sharedPerBlock: 0, sharedPerSm: toInt(custom.sharedPerSm),
          },
          note: <>Limits you typed. They start from the book's example SM.</>,
        }
      : (devices.find((d) => d.key === key) ?? BOOK);
  const shape: Shape = {
    tM: toInt(form.tM, 1), tN: toInt(form.tN, 1), extra: toInt(form.extra), threads: toInt(form.threads, 1),
    bM: toInt(form.bM, 1), bN: toInt(form.bN, 1), bK: toInt(form.bK, 1), pad: toInt(form.pad), buffers,
  };
  const b = budget(shape, dev.limits);
  const metal = dev.kind === 'metal';
  const lim = dev.limits;
  const runs = measured(reports, dev.backend);

  const apply = (p: Preset) => {
    const s = p.s as Shape;
    setForm({ tM: String(s.tM), tN: String(s.tN), extra: String(s.extra), threads: String(s.threads), bM: String(s.bM), bN: String(s.bN), bK: String(s.bK), pad: String(s.pad) });
    setBuffers(s.buffers);
  };
  const active = (p: Preset) => (Object.keys(p.s) as (keyof Shape)[]).every((k) => shape[k] === p.s[k]);
  const field = (label: string, value: string, set: (v: string) => void, hint?: string) => (
    <label className="field" key={label}>
      {label}
      <input type="number" min="0" step="1" value={value} aria-label={label} onChange={(e) => set(e.target.value)} />
      {hint ? <span style={{ fontSize: '0.74rem' }}>{hint}</span> : null}
    </label>
  );

  const names = b.binding.map((k) => LIMIT_NAME[k]);
  const list = names.length > 1 ? names.slice(0, -1).join(', ') + ' and ' + names[names.length - 1] : (names[0] ?? '');
  let verdict: ReactNode;
  if (metal)
    verdict = (
      <p style={{ margin: 0 }}>
        Apple publishes no register count, no register file size and no rule for how many threadgroups share a GPU
        core, so no block count or occupancy can be computed. Only shared memory can be checked, on the left.
      </p>
    );
  else if (b.blocks == null)
    verdict = <p style={{ margin: 0 }}>Enter registers per SM and thread slots per SM to get a block count.</p>;
  else if (b.blocks === 0)
    verdict = (
      <p style={{ margin: 0 }}>
        <span className="chip fail">does not fit</span> Not one block of {count(shape.threads)} threads fits: it needs
        more {list} than the SM has. On CUDA the launch fails.
      </p>
    );
  else
    verdict = (
      <p style={{ margin: 0 }}>
        <b>{list.charAt(0).toUpperCase() + list.slice(1)} {b.binding.length > 1 || names[0] === 'thread slots' || names[0] === 'registers' ? 'bind' : 'binds'}.</b>{' '}
        {b.binding.includes('registers') && b.blocks != null && lim.threadsPerSm > 0 && b.resident! < lim.threadsPerSm ? (
          <>
            To fit one more block, a thread would have to make do with{' '}
            <span data-out="next-regs">{count(regsForBlocks(lim.regsPerSm, shape.threads, b.blocks + 1))}</span> registers.
          </>
        ) : b.resident === lim.threadsPerSm ? (
          <>Every thread slot is filled.</>
        ) : null}
      </p>
    );

  return (
    <Bench
      title="Register and shared-memory budget"
      task="Start from the book's kernel, then find how few registers a thread would need for two blocks to fit."
      foot={<>{dev.note} 1 KB is 1024 bytes, as in the book's arithmetic.</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          Limits of one SM
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label="Limits of one SM" style={{ maxWidth: '100%' }}>
            {devices.map((d) => (
              <option key={d.key} value={d.key}>{d.label}</option>
            ))}
            <option value="custom">Custom numbers</option>
          </select>
        </label>
        {key === 'custom' ? CUSTOM_FIELDS.map((f) => field(f.label, custom[f.key], (v) => setCustom({ ...custom, [f.key]: v }))) : null}
      </div>
      {!loading && devices.length === 1 ? (
        <p className="muted" style={{ margin: '0 0 0.8rem' }}>
          No GPU in this machine's reports yet, so the list offers the book's example and custom numbers. Run a lab
          on a machine with a GPU (for example <code>python3 labs/pmpp.py run 15.6 --solution</code>) and reload to see
          your own devices here.
        </p>
      ) : null}
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        {FIELDS.map((f) => field(f.label, form[f.key], (v) => setForm({ ...form, [f.key]: v }), f.hint))}
        <label className="field">
          Buffers
          <select value={buffers} onChange={(e) => setBuffers(Number(e.target.value) as 1 | 2)} aria-label="Buffers">
            <option value={1}>1</option>
            <option value={2}>2 (section 15.8)</option>
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '1rem' }} role="group" aria-label="Presets">
        <span className="muted">Presets</span>
        {PRESETS.map((p) => (
          <button key={p.key} className="quiet" data-preset={p.key} aria-pressed={active(p)} onClick={() => apply(p)}>
            {p.label}
          </button>
        ))}
      </div>

      <div className="cols plot">
        <div className="panel" style={{ overflowX: 'auto' }}>
          <p style={{ margin: '0 0 0.6rem' }}>
            <span className="chip reg">registers per thread</span>{' '}
            <span data-out="regs" data-regs={b.regs} data-wanted={b.wanted}>
              {shape.tM} × {shape.tN} + {shape.tM} + {shape.tN} = {count(b.tileRegs)} for the tiles, + {count(shape.extra)} ={' '}
              <b>{count(b.wanted)}</b>
            </span>
            {b.spills ? (
              <>
                {' '}
                <span className="chip fail">over the limit</span> A thread may have at most {count(lim.maxRegsPerThread)}; the
                compiler keeps the rest in local memory, which is device memory.
              </>
            ) : null}
          </p>
          {metal ? null : (
            <table className="data" style={{ margin: 0 }}>
              <thead>
                <tr>
                  <th style={{ paddingLeft: '0.5rem' }}>Limit</th>
                  <th className="num">One SM has</th>
                  <th className="num">One block needs</th>
                  <th className="num">Blocks that fit</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ['registers', <span className="chip reg">registers</span>, lim.regsPerSm, b.regs * shape.threads],
                    ['threads', 'thread slots', lim.threadsPerSm, shape.threads],
                    ['shared', <span className="chip shared">shared memory</span>, lim.sharedPerSm, b.shared],
                  ] as [LimitKey, ReactNode, number, number][]
                ).map(([k, name, cap, need]) => {
                  const fit = b.byLimit[k];
                  const binds = b.binding.includes(k);
                  return (
                    <tr key={k} data-limit={k} data-fit={fit ?? 'none'} style={binds ? { background: 'var(--rule-soft)' } : undefined}>
                      <td style={{ paddingLeft: '0.5rem' }}>{name}</td>
                      <td className="num">{cap > 0 ? (k === 'shared' ? bytes(cap) : count(cap)) : 'not known'}</td>
                      <td className="num">{k === 'shared' ? bytes(need) : count(need)}</td>
                      <td className="num">{fit == null ? <span className="muted">not counted</span> : count(fit)}</td>
                      <td style={{ paddingRight: '0.5rem' }}>{binds ? <b>binds</b> : null}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          <p className="muted" style={{ margin: '0.7rem 0 0' }} data-out="shared" data-bytes={b.shared}>
            Shared memory per block: A_s {count(shape.bM)} × {count(shape.bK + shape.pad)} floats = {bytes(b.sharedA)}, B_s{' '}
            {count(shape.bK)} × {count(shape.bN)} floats = {bytes(b.sharedB)}
            {buffers === 2 ? ', twice over' : ''}: <b>{bytes(b.shared)}</b>.
            {lim.sharedPerBlock > 0 ? (
              <>
                {' '}
                Against this device's per-block limit of {bytes(lim.sharedPerBlock)}:{' '}
                {b.fitsBlock ? <span className="chip ok" data-out="fits">fits</span> : <span className="chip fail" data-out="fits">does not fit</span>}
              </>
            ) : null}
          </p>
          {!b.covers ? (
            <p style={{ margin: '0.7rem 0 0' }} data-out="covers">
              <span className="chip fail">check the shape</span> {count(shape.threads)} threads × {shape.tM} × {shape.tN} ={' '}
              {count(shape.threads * shape.tM * shape.tN)} outputs, but the block tile has {count(shape.bM * shape.bN)}.
            </p>
          ) : null}
        </div>
        <div className="panel" aria-live="polite">
          <p className="muted" style={{ margin: 0 }}>Occupancy</p>
          <p style={{ fontSize: '2rem', fontWeight: 600, margin: '0 0 0.2rem', lineHeight: 1.1 }}>
            <output data-out="occupancy" data-occupancy={b.occupancy == null ? 'none' : fmt(b.occupancy)} data-blocks={b.blocks ?? 'none'} data-binding={b.binding.join('+')}>
              {b.occupancy == null ? 'not documented' : fmt(b.occupancy) + '%'}
            </output>
          </p>
          {b.occupancy != null ? (
            <p className="muted" style={{ margin: '0 0 0.7rem' }}>
              {count(b.blocks!)} {b.blocks === 1 ? 'block' : 'blocks'} × {count(shape.threads)} threads ={' '}
              {count(b.resident!)} of {count(lim.threadsPerSm)} thread slots
            </p>
          ) : null}
          {b.threadBound != null ? (
            <p style={{ margin: '0 0 0.7rem' }} data-out="bound" data-bound={b.threadBound}>
              Registers alone leave room for {count(lim.regsPerSm)} ÷ {count(b.regs)} = <b>{count(b.threadBound)}</b>{' '}
              threads.
            </p>
          ) : null}
          <div data-out="verdict">{verdict}</div>
          {dev.kind === 'gpu' || dev.kind === 'metal' ? (
            <div style={{ margin: '0.8rem 0 0' }} data-out="measured" data-runs={runs.length}>
              {runs.length ? (
                runs.map((r, i) =>
                  r.blocks != null ? (
                    <p key={i} style={{ margin: '0 0 0.4rem' }}>
                      {r.lab}: the occupancy API reports <b>{r.blocks}</b> {r.blocks === 1 ? 'block' : 'blocks'} of{' '}
                      {r.threads ?? '?'} threads per SM for its reference kernel.
                    </p>
                  ) : (
                    <p key={i} style={{ margin: '0 0 0.4rem' }}>
                      {r.lab}: the compiled pipeline accepts at most <b>{r.cap}</b> threads per threadgroup
                      {r.staticBytes != null ? <> and declares {bytes(r.staticBytes)} of threadgroup memory</> : null}.
                    </p>
                  ),
                )
              ) : null}
              {runs.length > 0 && dev.kind === 'metal' ? (
                <p className="muted" style={{ margin: 0 }}>
                  The labs' kernels declare a cap of 256 threads, so this shows that Metal did not lower it, not how
                  far above 256 it would go.
                </p>
              ) : null}
              {runs.length > 0 ? null : dev.kind === 'gpu' ? (
                <p className="muted" style={{ margin: 0 }}>
                  No chapter 15 lab has reported <code>blocks_per_sm</code> on this device yet. The labs ask the
                  occupancy API, which knows the allocation chunks this estimate leaves out.
                </p>
              ) : (
                <p className="muted" style={{ margin: 0 }}>
                  No chapter 15 lab has run its benchmark on this device yet. When one has, its compiled pipeline's
                  thread cap appears here. Metal sets that cap from the kernel's memory usage and may lower it.
                </p>
              )}
            </div>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

/** The calculator opens once the prediction above it is committed, or on request. */
function GatedBudget() {
  const { progress } = useProgress();
  const [open, setOpen] = useState(false);
  if (open || progress.answers['ch15/15.7/p-blocks'] != null) return <BudgetBench />;
  return (
    <p className="ui" data-out="gate">
      The calculator opens with the book's numbers once you have committed a prediction.{' '}
      <button className="quiet" onClick={() => setOpen(true)}>Open it without predicting</button>
    </p>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS157: Record<string, Question> = {
  'ch15/15.7/q-80': {
    kind: 'numeric',
    id: 'ch15/15.7/q-80',
    prompt: (
      <>
        A thread of the register-tiled kernel (section 15.4) owns an 8 × 8 output tile and, at each step of its inner
        loop, holds one strip of A and one strip of B. What is the fewest registers it needs for these tiles alone,
        counting one 32-bit register per float?
      </>
    ),
    answer: 80,
    unit: 'registers',
    explain: (
      <p>
        The output tile is 8 × 8 = 64 values, live from the first tile pair to the final store. The A strip is a column of
        the thread's 8 rows of A, 8 values; the B strip is a row of its 8 columns of B, 8 more. 64 + 8 + 8 = 80. It is a
        floor: addresses, loop counters and the temporaries of the unrolled code come on top.
      </p>
    ),
    source: 'ch. 15 §15.7, p. 363',
  },
  'ch15/15.7/q-1536': {
    kind: 'numeric',
    id: 'ch15/15.7/q-1536',
    prompt: (
      <>
        The same kernel, still 255 registers per thread and one 256-thread block per SM, runs on an SM whose thread limit
        is 1536 rather than 2048. What is its occupancy?
      </>
    ),
    answer: 16.7,
    tol: 0.005,
    unit: '%',
    explain: (
      <p>
        Registers still allow only one block, so 256 threads are resident: 256 / 1536 = 16.7%. The kernel did not change
        and neither did its speed in principle; only the denominator did. NVIDIA documents 1536 resident threads per SM
        for compute capabilities 8.6, 8.9 and 12.x (RTX 30, 40 and 50), and 1024 for 7.5 and 10.7, where the same kernel
        reads 25%. An occupancy percentage means little without knowing which SM it was computed for.
      </p>
    ),
    source: 'our arithmetic; limits from notes/today/ch15-facts.md A3 and ch04-facts.md A12',
  },
  'ch15/15.7/q-128regs': {
    kind: 'numeric',
    id: 'ch15/15.7/q-128regs',
    prompt: (
      <>
        Suppose the compiler fitted the kernel in 128 registers per thread. On the book's example SM (65,536 registers,
        2048 thread slots), how many 256-thread blocks could be resident?
      </>
    ),
    answer: 2,
    unit: 'blocks',
    explain: (
      <p>
        One block needs 256 × 128 = 32,768 registers, and 65,536 / 32,768 = 2: 512 threads, 25%. That is the arithmetic
        of the book's model. Whether two blocks really fit depends on how the hardware rounds each warp's allocation,
        which NVIDIA does not publish, so for a real kernel ask{' '}
        <code>cudaOccupancyMaxActiveBlocksPerMultiprocessor</code>; the labs print its answer as{' '}
        <code>blocks_per_sm</code>.
      </p>
    ),
    source: 'ch. 15 §15.7, p. 363 (the model); notes/today/ch15-facts.md A3 (the allocation chunks)',
  },
  'ch15/15.7/q-two-phases': {
    kind: 'choice',
    id: 'ch15/15.7/q-two-phases',
    prompt: <>With one block per SM, what does the hardware no longer do for you that it did for chapter 5's kernel?</>,
    options: [
      {
        label: 'Overlap one block\'s loading with another block\'s computing.',
        correct: true,
      },
      {
        label: 'Coalesce the loads of the input tiles, which needs enough warps in flight to merge their addresses.',
        why: 'Coalescing is about how one warp\'s addresses fall, not about how many blocks are resident. The tile loads coalesce as well as before.',
      },
      {
        label: 'Keep the output tile in registers: a lone block must spill its accumulators to make room for the loads.',
        why: 'The output tile is in registers precisely because each thread was given so many of them. That is the cause of the low occupancy, not something it takes away.',
      },
      {
        label: 'Hide the latency of shared-memory reads, which needs other blocks\' warps to switch to.',
        why: 'Register tiling (section 15.4) already cut the shared-memory reads eightfold, and unrolling exposes independent FMAs to cover the rest. The gap that remains is the whole loading phase.',
      },
    ],
    explain: (
      <p>
        With several blocks on an SM, one waits for its tiles while another computes, and the scheduler interleaves
        their warps. With one block, every warp of the SM is at the same barrier: all loading, then all computing. The
        memory hardware idles during the compute and the arithmetic units during the loads. Section 15.8 rearranges the
        loop so that one block does both at once.
      </p>
    ),
    source: 'ch. 15 §15.7, p. 364',
  },
  'ch15/15.7/q-launch-bounds': {
    kind: 'choice',
    id: 'ch15/15.7/q-launch-bounds',
    prompt: (
      <>
        Why does <code>__launch_bounds__(256)</code> on this kernel matter more for the HIP build of the same{' '}
        <code>.cu</code> file?
      </>
    ),
    options: [
      {
        label: 'Without it HIP compiles for the largest block allowed and caps registers per thread to match.',
        correct: true,
      },
      {
        label: 'Without it HIP rejects any launch whose block is not the default 1024 threads.',
        why: 'The bound is an upper limit for code generation and launch validation. A block of 256 is within the default.',
      },
      {
        label: 'HIP ignores __launch_bounds__, so on HIP it changes nothing in the compiled code.',
        why: 'HIP honours it; its second argument even means something different from CUDA\'s (waves per execution unit, not blocks per SM).',
      },
      {
        label: 'AMD wavefronts can be 64 wide, so a 256-thread block of 4 wavefronts must be declared to the compiler.',
        why: 'AMD wavefronts are 64 or 32 wide, and any multiple of the width works without a declaration. The issue is the register budget the compiler assumes.',
      },
    ],
    explain: (
      <p>
        AMD: if no launch bounds are given, the compiler assumes the device's maximum block size, and "reducing
        MAX_THREADS_PER_BLOCK allows the compiler to use more resources per thread than an unconstrained compilation".
        A kernel that wants 80 or more registers per thread for its tiles is the case this was written for. Applying it
        to this kernel is our reading of AMD's statement; it has not been measured here.
      </p>
    ),
    source: 'notes/today/ch15-facts.md B3 and ch08-facts.md B4 (not in the book)',
  },
  'ch15/15.7/q-mlx': {
    kind: 'choice',
    id: 'ch15/15.7/q-mlx',
    prompt: (
      <>
        MLX's Metal GEMM uses 64 to 128 threads per threadgroup and tiles such as 64 × 64 × 16, chosen per device,
        against the book's 256 threads and 128 × 128 × 8. What may you conclude about the register file of Apple GPUs?
      </>
    ),
    options: [
      {
        label: 'Nothing: Apple documents no register count or occupancy rule, so MLX\'s shapes are a design, not evidence.',
        correct: true,
      },
      {
        label: 'Its register file is about a quarter the size of an NVIDIA SM\'s, since MLX\'s threadgroups are a quarter as large.',
        why: 'That reads a hardware figure off a library\'s choice. Tile shapes depend on many things: the matrix instructions used, the threadgroup memory, the data type, measured speed.',
      },
      {
        label: 'Apple GPUs allow at most 128 threads per threadgroup, or MLX would use the book\'s 256.',
        why: 'Metal documents 1024 as the theoretical maximum on Apple4 and later; the real cap is per pipeline. MLX chose fewer.',
      },
    ],
    explain: (
      <p>
        MLX's choice is a fact about MLX: which shape its authors found fast, on which chips. Apple documents no
        register count, register file or occupancy API for Metal kernels; Xcode shows a shader's occupancy, but no
        function returns it. On Metal, tile shapes are compared by measuring them.
      </p>
    ),
    source: 'notes/today/ch15-facts.md C3 and D3; ch06-facts.md C5 (not in the book)',
  },
};

function PredictReveal({ v }: { v: number }) {
  return (
    <p>
      65,536 / 255 = 257, so the registers of one SM hold 257 threads of this kernel: <b>one block</b> of 256, with one
      thread's worth left over.{' '}
      {v === 1
        ? 'That is what you had.'
        : v > 1
          ? 'Thread slots would allow 8 blocks, and shared memory many more; registers allow one.'
          : 'One block does fit: 256 × 255 = 65,280 registers, just under 65,536.'}{' '}
      One block is 256 of 2048 thread slots, 12.5% occupancy. It is also why the block has 256 threads: a block of 512
      would need 130,560 registers and could not be resident at all.
    </p>
  );
}

// ---------------------------------------------------------------- the section
export function S157() {
  return (
    <>
      <p>
        The last four sections bought arithmetic intensity with on-chip storage: a large output tile per block held in
        registers, input tiles staged in shared memory, strips copied into registers. Both stores belong to the SM, and
        every block resident on it draws from them. This section adds up the bill, and arrives at a conclusion that
        looks wrong at first: the kernel is meant to run at low occupancy.
      </p>
      <Source section="15.7" pages="363" />

      <h2>Shared memory is cheap here</h2>
      <p>
        Recall the result of <a href={href('ch15', '15.2')}>section 15.2</a>. For each pair of input tiles a block does
        2·m·n·k operations and loads 4·k·(m + n) bytes, so the ratio is 0.5·m·n/(m + n) and the depth k cancels. A
        deeper input tile buys no intensity, so you keep it shallow. With m = n = 128 and k = 8, the A tile is 128 × 8
        floats and the B tile 8 × 128: 4 KB each. The book says that does not limit occupancy on a modern GPU. On its
        example device, with 228 KB of shared memory per SM, 28 blocks' worth of these tiles would fit.
      </p>
      <Source section="15.7" pages="363" />

      <h2>Registers are not</h2>
      <p>
        Now count what one thread must hold at the same moment. Work it out before reading on.
      </p>
      <Ask q={questionsS157['ch15/15.7/q-80']} label="Work it out" />
      <p>
        Eighty is a floor. Addresses, loop counters and the temporaries of the unrolled code need registers too, and
        unrolling, which is what keeps the tiles in registers at all (section 15.3), raises the number of values live at
        once. The book says kernels like this end up at the most a thread may have: 255 registers. Every thread resident
        on the SM takes that many from one register file.
      </p>
      <Source section="15.7" pages="363" />
      <Predict
        id="ch15/15.7/p-blocks"
        prompt={
          <>
            The book's example SM has 65,536 registers and 2048 thread slots. At 255 registers per thread, how many
            blocks of 256 threads can be resident on it at once?
          </>
        }
        unit="blocks"
      >
        {(v) => <PredictReveal v={v} />}
      </Predict>
      <p>
        Those are the book's figures. On your GPU the answer rests on two numbers that differ between NVIDIA parts, and
        on one that NVIDIA does not publish. The calculator below does the arithmetic for any tile shape and any SM,
        starting from the book's kernel; its second preset shows that even the 80-register floor leaves the kernel
        well short of full occupancy.
      </p>
      <Today checked="2026-11-20 (ch15-facts A3; ch04-facts A12, A14)">
        <p>
          255 per thread and 64 K per SM hold on every compute capability NVIDIA lists. 2048 resident threads only on
          8.0, 9.0, 10.0 and 10.3; 1536 on RTX 30, 40 and 50; 1024 on 7.5 and 10.7.
        </p>
        <p>
          Registers are allocated in chunks of undocumented size, so the exact fit comes from{' '}
          <Id>cudaOccupancyMaxActiveBlocksPerMultiprocessor</Id> (the labs print it as <Id>blocks_per_sm</Id>) or Nsight
          Compute's <Id>launch__occupancy_limit_registers</Id>.
        </p>
      </Today>
      <p>
        Your own CUDA or HIP device appears in its list once a lab run has recorded it, with what the
        occupancy API said about the labs' real kernels beside the estimate. A Metal device is shown with what Apple
        documents, which is shared memory only.
      </p>
      <Simplification>
        <p>
          Registers per thread times threads per block, against the register file: the book's model. Real allocation is
          per warp, in chunks NVIDIA does not publish and granules AMD sizes per target. Block slots and the shared
          memory the driver reserves are not counted.
        </p>
      </Simplification>
      <GatedBudget />
      <Ask q={questionsS157['ch15/15.7/q-1536']} label="Work it out" />
      <Ask q={questionsS157['ch15/15.7/q-128regs']} label="Work it out" />

      <h2>Low occupancy, on purpose</h2>
      <p>
        Chapter 4 taught you to want many warps per SM, so that one waiting on memory costs nothing while another runs.
        This kernel gives most of that up. The book judges the trade worth it: the large register tiles are what raise
        the arithmetic intensity, and that gain outweighs what the missing warps would have hidden. NVIDIA's own
        profiling guidance takes the same side today.
      </p>
      <Today checked="2026-11-20 (ch15-facts A3)">
        <p>
          The Compute Triage Guide: with low occupancy and a busy pipeline, "Launching extra warps is
          counter-productive because the busy pipe cannot absorb more work."
        </p>
      </Today>
      <p>
        The book names two costs of running so few threads, and a remedy for each.
      </p>
      <ol>
        <li>
          <b>Fewer memory instructions in flight.</b> Fewer threads issue fewer loads and stores, which can leave memory
          bandwidth unused. The remedy is to make each instruction move more: vector loads and stores of 16 bytes
          (chapter 6, section 6.3). The book says these were seen "in Section 15.3", which only suggests them as
          exercises (notes, item 9); labs 15.3 and 15.4 supply the code.
        </li>
        <li>
          <b>Fewer warps to cover latency.</b> When the only warps on the SM all wait, nothing runs. The remedy is
          aggressive unrolling (chapter 6, section 6.6): it removes branches and lays out independent instructions the
          compiler can place between a slow instruction and its first use. Done well, most of the compiled kernel is a
          long run of independent fused multiply-adds, and the arithmetic units rarely stall.
        </li>
      </ol>
      <Source section="15.7" pages="364" />

      <h2>What is left: two phases</h2>
      <p>
        One problem survives both remedies. Normally a compute-bound kernel can ignore memory latency, because while one
        block waits for its tiles, another block on the same SM computes. Here there is no other block. The lone block
        alternates between a memory-bound phase, in which tiles load and the arithmetic units idle, and a compute-bound
        phase, in which it computes and the memory hardware idles. If the two are to overlap, the program has to
        arrange it. <a href={href('ch15', '15.8')}>Section 15.8</a> does, with software pipelining.
      </p>
      <Source section="15.7" pages="364" />
      <Ask q={questionsS157['ch15/15.7/q-two-phases']} />
      <p>
        Keep this trade in mind when you read GEMM code in a library or a framework backend. A large array of
        accumulators per thread and a low occupancy are the design, not a defect to fix, and NVIDIA's own template
        library says so in as many words.
      </p>
      <Today checked="2026-11-20 (ch15-facts A3, A12)">
        <p>
          CUTLASS: the accumulators "typically occupy at least half a thread's total register budget", so occupancy
          "is relatively low compared to other classes of GPU workloads."
        </p>
      </Today>

      <h2>On AMD and Apple GPUs</h2>
      <p>
        The trade carries over to AMD, with different numbers. A thread may have up to 256 vector registers, plus 256
        matrix registers on CDNA. AMD documents the register file of a compute unit only as a range, so the calculator
        uses what your device reports, and it allocates registers per wavefront in granules, so the per-thread product
        is only the right shape. AMD's own guide says that GEMMs built on its matrix instructions on CDNA often run at
        low occupancy, because a few warps are enough to keep those units and the memory pipeline busy.
      </p>
      <Beyond>
        <p>
          HIP and Metal are not in the book. Checked 2026-11-20: ch15-facts B3, C3; ch04-facts B10, C5; ch06-facts C5;
          ch08-facts B4.
        </p>
      </Beyond>
      <p>
        One detail matters more on HIP than on CUDA. Unless told otherwise, the HIP compiler generates code for the
        largest block the device allows and budgets registers per thread to match.
      </p>
      <Ask q={questionsS157['ch15/15.7/q-launch-bounds']} />
      <p>
        Apple documents none of this for Metal: no register count, no register file size, no occupancy API. Xcode
        shows a shader's occupancy, but no function returns it. Two documented facts still bear on a register-heavy
        kernel with 256 threads. A pipeline may be given a lower thread cap than the device's maximum ("Metal may lower
        the thread limit when creating the compute pipeline state"), so the chapter's Metal kernels declare{' '}
        <code>[[max_total_threads_per_threadgroup(256)]]</code>, the counterpart of the <code>__launch_bounds__(256)</code>{' '}
        on their CUDA side, and the labs' tests fail if the pipeline's <code>maxTotalThreadsPerThreadgroup</code> comes
        back below 256. And the tiles must fit the threadgroup memory limit, which is the one thing the calculator can
        check on a Metal device.
      </p>
      <Ask q={questionsS157['ch15/15.7/q-mlx']} />
    </>
  );
}
