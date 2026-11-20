import { useMemo, useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  DEVICE,
  OPS,
  PRESETS,
  RESOURCES,
  RESOURCE_NAME,
  STOP_LABS,
  canApply,
  evaluate,
  headroom,
  labNum,
  presetById,
  roofFraction,
  run,
  stopPoints,
  stopVerdict,
  type LabPoint,
  type OpId,
  type Resource,
  type StepVerdict,
} from '../../sim/bottleneck';

export const questionsS69: Record<string, Question> = {
  'ch06/6.9/q-nonbottleneck': {
    kind: 'choice',
    id: 'ch06/6.9/q-nonbottleneck',
    prompt: (
      <>
        A memory-bound kernel already moves its bytes at 90% of the bandwidth you measured in lab 5.1. Someone unrolls
        its loops to cut instruction overhead. What do you expect?
      </>
    ),
    options: [
      {
        label: 'At most about 10% faster, and probably no change: the resource relieved is not the bottleneck.',
        correct: true,
      },
      {
        label: 'A large gain: unrolling always helps.',
        why: 'It removes branches and exposes independent instructions. Neither is what this kernel waits on.',
      },
      {
        label: 'A slowdown: unrolling always costs registers.',
        why: 'It can cost registers, but "always slower" is no more true than "always faster". Here the likely effect is none.',
      },
    ],
    explain: (
      <p>
        At 90% of the roof the kernel can get at most 1/0.9 = 1.11 times faster, whatever you do, unless you change
        how many bytes it moves. Unrolling changes the instruction stream, and the instruction stream is not what
        limits it. This is the first half of p. 153: relieving a resource that is not the bottleneck gains nothing.
      </p>
    ),
    source: 'ch. 6 §6.9, p. 153; roofline from ch. 5 §5.1',
  },
  'ch06/6.9/q-stop': {
    kind: 'numeric',
    id: 'ch06/6.9/q-stop',
    prompt: (
      <>
        A memory-bound kernel moves 12 GB in 5 ms. Lab 5.1 measured 3000 GB/s on that device. What percentage of the
        roof does it reach? (Toy numbers.)
      </>
    ),
    answer: 80,
    tol: 0.01,
    unit: '%',
    explain: (
      <p>
        12 GB / 5 ms = 2400 GB/s, and 2400 / 3000 = 80%. Without moving fewer bytes, the best this kernel can do is 1 /
        0.8 = 1.25 times faster, however clever the next change. Whether 25% is worth chasing is now a decision you
        can make with a number. Chapter 5 worked the same arithmetic on the book&rsquo;s example device (p. 96).
      </p>
    ),
    source: 'ch. 6 §6.9, p. 154; ch. 5 §5.1, p. 96',
  },
  'ch06/6.9/q-sol': {
    kind: 'choice',
    id: 'ch06/6.9/q-sol',
    prompt: (
      <>
        After a change, the profiler&rsquo;s &ldquo;speed of light&rdquo; percentage for memory drops, and the
        kernel&rsquo;s duration also drops. Which do you trust?
      </>
    ),
    options: [
      {
        label: 'The duration. The change may have removed work, which lowers utilisation while the kernel gets faster.',
        correct: true,
      },
      {
        label: 'The percentage: it is the hardware’s own view.',
        why: 'It measures how busy a unit was, not how much work was needed. Fewer bytes moved at the same rate shows up as a lower percentage.',
      },
      { label: 'Neither: the two contradict each other, so the measurement is broken.', why: 'They measure different things and need not move together.' },
    ],
    explain: (
      <p>
        Tiling is the standard example: it cuts the bytes a kernel moves, so memory looks less busy, and the kernel
        finishes sooner. NVIDIA&rsquo;s current triage guide says it in so many words: duration is the ground truth,
        not the utilisation percentage. The course&rsquo;s labs report measured time for the same reason.
      </p>
    ),
    source: 'notes/today/ch06-facts.md A17 (Nsight Compute Compute Triage Guide)',
  },
  'ch06/6.9/q-hw': {
    kind: 'choice',
    id: 'ch06/6.9/q-hw',
    prompt: <>The same kernel is compute-bound on one GPU and memory-bound on another. Is that possible?</>,
    options: [
      {
        label:
          'Yes. The regime belongs to the pair (kernel, device): the kernel fixes the intensity, the device fixes where the threshold is. The bottleneck, and the right optimisation, have to be found again per device.',
        correct: true,
      },
      { label: 'No: intensity is a property of the kernel alone.', why: 'Intensity is. Whether it is above or below the threshold depends on the device’s ratio of compute to bandwidth.' },
      { label: 'Only if the profiler is wrong on one of them.', why: 'Nothing is wrong: the two devices have different thresholds.' },
    ],
    explain: (
      <p>
        A kernel at 5 FLOP/B is memory-bound on a device whose threshold is 20 FLOP/B and compute-bound on one whose
        threshold is 3 (made-up thresholds). Beyond the roofline, occupancy limits, bank counts and cache sizes all
        differ between devices. The book: bottlenecks can be hardware-specific (p. 154). NVIDIA&rsquo;s guide: find the
        dominant limiter again on each new GPU generation.
      </p>
    ),
    source: 'ch. 6 §6.9, p. 154; ch. 5 §5.1, p. 95; notes/today/ch06-facts.md A17',
  },
};

// ---------------------------------------------------------------- the bottleneck loop
const BAR: Record<Resource, string> = {
  global: 'var(--global)',
  shared: 'var(--shared)',
  compute: 'var(--ink-2)',
  barrier: 'var(--ink-3)',
};
const VERDICT_CHIP: Record<StepVerdict, ReactNode> = {
  faster: <span className="chip ok">faster</span>,
  same: <span className="chip">unchanged</span>,
  slower: <span className="chip fail">slower</span>,
};
const pct = (x: number) => `${Math.round(x * 100)}%`;
const LIMITER_WORDS = { blocks: 'block slots', shared: 'shared memory', regs: 'registers' } as const;

function BottleneckLoop() {
  const [presetId, setPresetId] = useState('shared-limited');
  const [ops, setOps] = useState<OpId[]>([]);
  const preset = presetById(presetId);
  const { kernel, steps } = useMemo(() => run(preset.kernel, ops), [preset, ops]);
  const now = evaluate(kernel);
  const start = evaluate(preset.kernel);
  const prev = steps.length ? steps[steps.length - 1].before : null;
  const scale = Math.max(start.total, now.total, ...steps.map((s) => s.after.total), ...(prev ? [prev.total] : [])) * 1.05;
  const choose = (id: string) => {
    setPresetId(id);
    setOps([]);
  };
  return (
    <Bench
      title="Bottleneck loop"
      task="Pick a starting kernel, apply optimisations from the checklist, and watch which bar sets the time."
      foot={
        <>
          Run time is the longest bar. The thin mark on each bar is its length before your last step. Every number
          comes from a toy model (<code>src/sim/bottleneck.ts</code>) whose factors are invented; only their
          directions are the book&rsquo;s.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }} role="radiogroup" aria-label="Starting kernel">
        {PRESETS.map((p) => (
          <button key={p.id} className="quiet" aria-pressed={p.id === presetId} data-preset={p.id} onClick={() => choose(p.id)}>
            {p.label}
          </button>
        ))}
      </div>
      <p className="status-line" style={{ margin: '0 0 0.9rem' }}>
        {preset.about}
      </p>
      <div className="cols plot">
        <div>
          <div style={{ display: 'grid', gap: '0.45rem' }} data-bars>
            {RESOURCES.map((r) => {
              const t = now.times[r];
              const isB = r === now.bottleneck;
              return (
                <div key={r} data-resource={r} data-time={t.toFixed(3)} data-bottleneck={isB ? 'yes' : 'no'}>
                  <div className="row" style={{ justifyContent: 'space-between', gap: '0.2rem 0.6rem' }}>
                    <span>
                      {RESOURCE_NAME[r]} {isB ? <b>· bottleneck</b> : null}
                    </span>
                    <span className="mono">{fmt(t, 3)}</span>
                  </div>
                  <div
                    style={{
                      position: 'relative',
                      height: '0.95rem',
                      background: 'var(--cell)',
                      border: '1px solid var(--rule)',
                      borderRadius: 2,
                    }}
                  >
                    <div
                      style={{
                        width: `${Math.min(100, (100 * t) / scale)}%`,
                        height: '100%',
                        background: BAR[r],
                        opacity: isB ? 1 : 0.55,
                        transition: 'width 220ms ease',
                      }}
                    />
                    {prev ? (
                      <div
                        title="before your last step"
                        style={{
                          position: 'absolute',
                          top: -3,
                          bottom: -3,
                          width: 2,
                          left: `calc(${Math.min(100, (100 * prev.times[r]) / scale)}% - 1px)`,
                          background: 'var(--ink)',
                        }}
                      />
                    ) : null}
                  </div>
                </div>
              );
            })}
          </div>
          <p style={{ margin: '0.9rem 0 0' }} data-total={now.total.toFixed(3)}>
            <b>Run time {fmt(now.total, 3)}</b> (started at {fmt(start.total, 3)}). Bottleneck:{' '}
            <b data-bottleneck-label>{RESOURCE_NAME[now.bottleneck].toLowerCase()}</b>.
          </p>
          <p className="status-line" style={{ margin: '0.3rem 0 0' }} data-occupancy>
            {now.blocksPerSM} of {DEVICE.maxBlocksPerSM} blocks per SM, limited by {LIMITER_WORDS[now.limiter]}: occupancy{' '}
            {pct(now.occupancy)}. Bytes in flight: {now.memRate >= 1 ? 'enough to keep the bus busy' : `${pct(now.memRate)} of what the bus needs`}.
            Grid: {fmt(now.waves, 3)} waves{now.tail > 1.0001 ? `, the last one partial (every time × ${fmt(now.tail, 3)})` : ''}.
          </p>
        </div>
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(8.5rem, auto) 1fr', gap: '0.45rem 0.7rem', alignItems: 'start' }}>
            {OPS.map((o) => {
              const ok = canApply(kernel, o.id);
              return (
                <div key={o.id} style={{ display: 'contents' }}>
                  <button data-op={o.id} disabled={!ok.ok} title={o.effect} onClick={() => setOps((x) => [...x, o.id])}>
                    {o.label}
                  </button>
                  <span className="muted" style={{ fontSize: '0.78rem' }}>
                    {ok.ok ? `${o.effect} (${o.page})` : ok.why}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="row" style={{ marginTop: '0.7rem' }}>
            <button className="quiet" disabled={!ops.length} onClick={() => setOps((x) => x.slice(0, -1))}>
              Undo
            </button>
            <button className="quiet" disabled={!ops.length} onClick={() => setOps([])}>
              Reset
            </button>
          </div>
        </div>
      </div>
      <h4 style={{ margin: '1rem 0 0.4rem', fontFamily: 'var(--sans)', fontSize: '0.9rem' }}>History</h4>
      {steps.length ? (
        <ol style={{ margin: 0, paddingLeft: '1.3rem' }} data-history>
          {steps.map((s, i) => (
            <li key={i} data-step={s.op} data-verdict={s.verdict} style={{ marginBottom: '0.25rem' }}>
              {OPS.find((o) => o.id === s.op)!.label}: {fmt(s.before.total, 3)} → {fmt(s.after.total, 3)}{' '}
              {VERDICT_CHIP[s.verdict]}. Bottleneck {RESOURCE_NAME[s.before.bottleneck].toLowerCase()}
              {s.after.bottleneck !== s.before.bottleneck ? (
                <> → <b>{RESOURCE_NAME[s.after.bottleneck].toLowerCase()}</b></>
              ) : null}
              ; occupancy {pct(s.before.occupancy)}
              {s.after.occupancy !== s.before.occupancy ? ` → ${pct(s.after.occupancy)}` : ''}.
              {s.relievedNonBottleneck ? <span className="muted"> It relieved a resource that was not the bottleneck.</span> : null}
            </li>
          ))}
        </ol>
      ) : (
        <p className="status-line" style={{ margin: 0 }}>
          No steps yet.
        </p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- the stopping rule on your measurements
interface Roof {
  key: string;
  label: string;
  peak: number;
  bw: number;
  when: string;
}
/** GPU backends only: realBackends() leaves out 'emu' but not the race-detection emulator. */
const gpuBackends = (r: Reports) => realBackends(r).filter((b) => b !== 'race');
function measuredRoofs(reports: Reports): Roof[] {
  return gpuBackends(reports).flatMap((b) => {
    const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const peak = num(r?.compute_gflops);
    const bw = num(r?.bandwidth_gb_per_s);
    if (!peak || !bw) return [];
    return [{ key: b, label: `${String(reports[b].device.name ?? b)} (${b})`, peak, bw, when: reports[b].generated.slice(0, 10) }];
  });
}
/** The benchmark rows of the stopping-rule labs on one backend: the reader's work when it has rows, else the reference. */
function labRows(reports: Reports, backend: string) {
  const runOn = reports[backend];
  if (!runOn) return [];
  const labs = [...new Set(runOn.results.map((r) => r.lab))].filter((l) => STOP_LABS.some((s) => l.startsWith(s.prefix)));
  return labs.flatMap((lab) => {
    const work = benchRows(runOn, lab, 'work');
    const variant = work.length ? 'work' : 'solution';
    const rows = work.length ? work : benchRows(runOn, lab, 'solution');
    return rows.map((row) => ({ lab, row: row as Record<string, unknown>, variant }));
  });
}
const BOOK_ROOF = { peak: 66900, bw: 3350 };

function StopRule() {
  const { reports, loading, reload } = useReports();
  const roofs = useMemo(() => measuredRoofs(reports), [reports]);
  const [key, setKey] = useState<string | null>(null);
  const [thr, setThr] = useState(80);
  const roof = roofs.find((r) => r.key === key) ?? roofs[0];
  const rows = roof ? labRows(reports, roof.key) : [];
  const points: (LabPoint & { variant: string })[] = rows.flatMap((r) =>
    stopPoints([{ lab: r.lab, row: r.row }]).map((p) => ({ ...p, variant: r.variant })),
  );
  const have = new Set(points.map((p) => p.lab));
  const missing = STOP_LABS.filter((s) => !have.has(labNum(s.prefix.replace(/_$/, ''))));
  // A kernel of your own, against your roof or the book's example.
  const [mine, setMine] = useState({ gflop: 2, gb: 8, ms: 4, against: 'book' as 'book' | 'yours' });
  const mineRoof = mine.against === 'yours' && roof ? roof : { ...BOOK_ROOF };
  const minePt = roofFraction(mine.gflop * 1e9, mine.gb * 1e9, mine.ms / 1000, mineRoof.peak, mineRoof.bw);
  const verdictText = (f: number) => {
    const v = stopVerdict(f, thr / 100);
    return v === 'stop' ? (
      <span className="chip ok">close to the roof: stop</span>
    ) : v === 'room' ? (
      <span>room left: at most ×{fmt(headroom(f), 3)}</span>
    ) : (
      <span>
        <b>above the model&rsquo;s roof</b>: the byte count is too high
      </span>
    );
  };
  return (
    <Bench
      title="Stopping rule on your measurements"
      task="Move the threshold. Which of your kernels are worth another optimisation?"
      foot={
        <>
          Intensity is the book&rsquo;s count: every element a block uses is loaded once per block, caches ignored
          (chapter 5). Vector addition is 1 FLOP per 12 bytes. The roof is min(peak, bandwidth × intensity) from lab
          5.1&rsquo;s two probes, not the vendor&rsquo;s figures. Every number here is a measurement from this machine
          or arithmetic on one.
        </>
      }
    >
      {roof ? (
        <>
          <div className="row" style={{ marginBottom: '0.8rem' }}>
            {roofs.length > 1 ? (
              <label className="field">
                Device
                <select value={roof.key} onChange={(e) => setKey(e.target.value)}>
                  {roofs.map((r) => (
                    <option key={r.key} value={r.key}>
                      {r.label}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            <label className="field">
              Close enough to stop at {thr}% of the roof
              <input type="range" min="50" max="100" step="5" value={thr} onChange={(e) => setThr(+e.target.value)} aria-label="threshold" />
            </label>
          </div>
          <p className="status-line" style={{ margin: '0 0 0.6rem' }} data-roof={roof.key}>
            Roof on {roof.label}, measured by lab 5.1 on {roof.when}: {fmt(roof.bw, 4)} GB/s, {fmt(roof.peak, 4)} GFLOP/s,
            threshold {fmt(roof.peak / roof.bw, 3)} FLOP/B. Each lab shows your own kernel once you have run it with{' '}
            <code>--bench</code> (marked &ldquo;yours&rdquo;), the reference solution until then.
          </p>
          {points.length ? (
            <div style={{ overflowX: 'auto' }}>
              <table className="data" data-stop-table>
                <thead>
                  <tr>
                    <th>Lab</th>
                    <th>Kernel</th>
                    <th className="num">% of roof</th>
                    <th>Verdict</th>
                    <th className="num">FLOP/B</th>
                    <th className="num">GFLOP/s</th>
                    <th className="num">Roof there</th>
                  </tr>
                </thead>
                <tbody>
                  {points.map((p, i) => {
                    const r = roofFraction(p.flop, p.bytes, p.seconds, roof.peak, roof.bw);
                    return (
                      <tr key={i} data-point={p.lab} data-fraction={r.fraction.toFixed(4)} data-verdict={stopVerdict(r.fraction, thr / 100)}>
                        <td className="nowrap">
                          <a href={href('ch' + p.lab.split('.')[0].padStart(2, '0'), 'labs')}>{p.lab}</a>
                        </td>
                        <td>
                          {p.label}
                          {p.variant === 'work' ? <b> (yours)</b> : null}
                        </td>
                        <td className="num">{fmt(100 * r.fraction, 3)}%</td>
                        <td>{verdictText(r.fraction)}</td>
                        <td className="num">{fmt(r.intensity, 3)}</td>
                        <td className="num">{fmt(r.achieved, 4)}</td>
                        <td className="num">{fmt(r.roof, 4)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <p data-stop="none">No measurement on this machine yet for the kernels below.</p>
          )}
          {points.some((p) => stopVerdict(roofFraction(p.flop, p.bytes, p.seconds, roof.peak, roof.bw).fraction, 1) === 'above') ? (
            <p className="status-line" data-above-note>
              A point above the roof breaks the book&rsquo;s rule that measured points lie under both lines (ch. 5,
              pp. 95–96). The kernel did not beat the hardware: the model counted bytes that the caches served, so
              the true intensity is higher than the count. A profiler&rsquo;s measured DRAM bytes would place it
              properly; until then, read its verdict as unknown.
            </p>
          ) : null}
          {missing.length ? (
            <ul className="status-line" data-missing style={{ paddingLeft: '1.2rem' }}>
              {missing.map((s) => {
                const n = labNum(s.prefix.replace(/_$/, ''));
                return (
                  <li key={s.prefix}>
                    Lab {n} ({s.why}): no measurement on this machine yet. Run <code>python3 labs/pmpp.py run {n} --bench</code>{' '}
                    once your kernel passes (add <code>--solution</code> for the reference).
                  </li>
                );
              })}
            </ul>
          ) : null}
          <button className="quiet" onClick={reload}>
            Reload results
          </button>
        </>
      ) : loading ? (
        <p data-stop="loading">Reading reports.</p>
      ) : (
        <p data-stop="none">
          No measurement on this machine yet: the roof comes from lab 5.1. Run{' '}
          <code>python3 labs/pmpp.py run 5.1 --solution --bench</code>, then the chapter 6 labs with <code>--bench</code>.{' '}
          <button className="quiet" onClick={reload}>
            Reload results
          </button>
        </p>
      )}

      <h4 style={{ margin: '1.2rem 0 0.4rem', fontFamily: 'var(--sans)', fontSize: '0.9rem' }}>A kernel of your own</h4>
      <div className="row" data-mine>
        <label className="field">
          GFLOP done
          <input type="number" min="0" step="any" value={mine.gflop} onChange={(e) => setMine({ ...mine, gflop: Math.max(1e-6, +e.target.value || 0) })} />
        </label>
        <label className="field">
          GB moved
          <input type="number" min="0" step="any" value={mine.gb} onChange={(e) => setMine({ ...mine, gb: Math.max(1e-6, +e.target.value || 0) })} />
        </label>
        <label className="field">
          Time (ms)
          <input type="number" min="0" step="any" value={mine.ms} onChange={(e) => setMine({ ...mine, ms: Math.max(1e-6, +e.target.value || 0) })} />
        </label>
        <label className="field">
          Against
          <select value={mine.against} onChange={(e) => setMine({ ...mine, against: e.target.value as 'book' | 'yours' })}>
            <option value="book">the book&rsquo;s example (H100, p. 94)</option>
            {roof ? <option value="yours">your device (lab 5.1)</option> : null}
          </select>
        </label>
      </div>
      <p style={{ margin: '0.6rem 0 0' }} data-mine-result={minePt.fraction.toFixed(4)}>
        {fmt(minePt.intensity, 3)} FLOP/B, {minePt.regime}-bound there. Achieved {fmt(minePt.achieved, 4)} GFLOP/s against
        a roof of {fmt(minePt.roof, 4)}: <b>{fmt(100 * minePt.fraction, 3)}%</b>. Verdict at {thr}%: {verdictText(minePt.fraction)}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
export function S69() {
  const q = questionsS69;
  return (
    <>
      <p>
        Every optimisation in the checklist spends more of one resource to relieve another. Tiling spends shared
        memory to relieve global bandwidth; double buffering spends shared memory to remove a barrier; coarsening
        spends parallelism to remove an overhead. Two consequences follow, and they are the whole of the strategy.
        If the resource relieved is not the one limiting the kernel, nothing gets faster. If the resource spent is
        the one limiting it, the kernel gets slower.
      </p>
      <Source section="6.9" pages="153" />
      <p>
        The book&rsquo;s example is tiling. When global bandwidth is the limit, it is the right move. When the kernel
        is limited by occupancy, and occupancy is already limited by the shared memory each block takes, more tiling
        removes resident blocks and makes things worse. The model below lets you try both, and the other five rows
        you have met in this chapter.
      </p>
      <Simplification>
        <p>
          Ours, and illustrative: run time is the time of the slowest of four resources; occupancy counts whole blocks
          on a made-up SM; global memory is latency-bound below a made-up number of bytes in flight. Each
          optimisation multiplies a few numbers by an invented factor, in the direction the chapter gives. It shows
          the shape of the reasoning. It predicts nothing about your GPU.
        </p>
      </Simplification>
      <ul>
        <li>
          From <b>Tiled, limited by shared memory</b>, apply tiling. Undo, then try vector loads and unrolling. Which
          resource did each spend, and which did it relieve?
        </li>
        <li>
          From <b>Vector addition</b>, apply vector loads. Then press Reset and apply coarsening three times.
          Explain the unchanged steps and the last one.
        </li>
        <li>
          From <b>Naive matmul</b>, keep going until the bottleneck has moved twice.
        </li>
      </ul>
      <BottleneckLoop />
      <Ask q={q['ch06/6.9/q-nonbottleneck']} />

      <h2>The loop, and the tool that drives it</h2>
      <p>
        So the process is a loop: find the bottleneck, apply one optimisation aimed at it, measure, and find the new
        bottleneck, which is often somewhere else.
      </p>
      <Source section="6.9" pages="153–154" />
      <p>
        You do not find it by reading the code. You find it with a profiler; the book points to NVIDIA&rsquo;s Nsight
        Compute. And the answer depends on the device: the same kernel can hit different limits on different GPUs, so
        the loop has to be run again on each one.
      </p>
      <Today checked="2026-11-20">
        <p>
          The book&rsquo;s reference [1] still resolves, to Nsight Compute v2026.4.0. Its Compute Triage Guide, newer
          than the book&rsquo;s citation, states the loop almost step for step, and adds three rules worth keeping:
          duration is the ground truth, not utilisation percentages; find the dominant limiter again on each new GPU
          generation; stop when the expected gain is small next to the complexity and risk. (ch06-facts A17.)
        </p>
      </Today>
      <Ask q={q['ch06/6.9/q-sol']} />
      <Ask q={q['ch06/6.9/q-hw']} />

      <h2>When to stop</h2>
      <p>
        The loop needs an exit, and the book gives it with chapter 5&rsquo;s{' '}
        <a href={href('ch05', '5.1')}>roofline</a>. Count the operations a kernel does and the bytes it moves; their
        ratio says whether its ceiling is set by bandwidth or by arithmetic. Then compare what it achieves with that
        ceiling. Close to it, there is little left to gain. Well below it, more optimisation is likely to pay.
      </p>
      <Source section="6.9" pages="154" />
      <p>
        Below, the kernels you have measured so far are placed against the roof lab 5.1 measured on your device. One
        of them is the bridge to your own work: the linear layer at batch 1 is a decode step, which section 5.1 showed
        must stay memory-bound. Its percentage tells you how far your decode kernel is from the memory roof, and so
        whether the techniques of 6.1 to 6.3 have anything left to give.
      </p>
      <Beyond>
        <p>
          Reading batch 1 as a decode step is ours (ch. 5, question q-decode). Lab 6.4 times the same layer in the
          layout PyTorch and llama.cpp use.
        </p>
      </Beyond>
      <StopRule />
      <Ask q={q['ch06/6.9/q-stop']} />

      <h2>Which profiler, on which backend</h2>
      <p>
        The book names one tool. On the three backends of this course the tools differ, and so does how directly
        they answer the two questions above: what is the bottleneck, and how far from the roof am I.
      </p>
      <div style={{ overflowX: 'auto' }} className="wide">
        <table className="data" data-profilers>
          <thead>
            <tr>
              <th>Backend</th>
              <th>Profiler</th>
              <th>Roofline</th>
              <th>Uncoalesced loads</th>
              <th>Bank conflicts</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>CUDA</td>
              <td>Nsight Compute</td>
              <td>Yes: speed-of-light rooflines, with separate L1, L2 and DRAM ceilings</td>
              <td>Sectors per request above the ideal (4 for 4-byte loads by a full warp)</td>
              <td>
                <code>l1tex__data_bank_conflicts_pipe_lsu.sum</code>, and an average N-way conflict metric
              </td>
            </tr>
            <tr>
              <td>HIP</td>
              <td>
                <code>rocprofv3</code>; ROCm Compute Profiler (<code>rocprof-compute</code>)
              </td>
              <td>
                <code>rocprof-compute</code> draws one by default, but lists only Instinct GPUs and Ryzen AI Max APUs as
                tested; on a Radeon card expect <code>rocprofv3</code>
              </td>
              <td>Not checked by name</td>
              <td>LDS &ldquo;Bank Conflict Rate&rdquo;</td>
            </tr>
            <tr>
              <td>Metal</td>
              <td>Xcode&rsquo;s Metal debugger, Instruments</td>
              <td>
                None found (unverified absence); memory read and write limiters as a percentage of peak are the
                nearest thing
              </td>
              <td>Last-level-cache and MMU limiters (indirect)</td>
              <td>Tile-memory limiters (indirect)</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        On Metal the question &ldquo;how far from the roof&rdquo; is therefore yours to compute, exactly as the panel
        above does: from a measured time, your byte and operation counts, and a roof you measured.
      </p>
      <Today checked="2026-11-20">
        <p>
          From <code>notes/today/ch06-facts.md</code> A17 (Nsight Compute), B9 and B4 (ROCm Compute Profiler, docs
          3.10.0), C5 (Xcode documentation), and the table &ldquo;Where the backends really differ&rdquo;. Nothing
          here was run on a device by us.
        </p>
      </Today>
    </>
  );
}
