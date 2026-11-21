import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BOOK_OVERSUBSCRIPTION, PRESETS, estimate, idleCycles, period, runs, simulate, utilisation, warpsNeeded, type Cell,
  type LatencyConfig, type Timeline,
} from '../../sim/latency';
import { LatencyScene } from './LatencyScene';

const pct = (fraction: number) => Math.round(fraction * 1000) / 10;
const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;

// ---------------------------------------------------------------- the timeline drawing
const FILL: Record<Cell, string> = { exec: 'var(--ink)', wait: 'var(--global-wash)', ready: 'var(--rule)' };
const STROKE: Record<Cell, string> = { exec: 'var(--ink)', wait: 'var(--global)', ready: 'var(--ink-3)' };
const CELL_NAME: Record<Cell, string> = { exec: 'executing', wait: 'waiting on memory', ready: 'ready, not chosen' };

function CellLegend() {
  return (
    <span className="legend">
      {(['exec', 'wait', 'ready'] as Cell[]).map((c) => (
        <span key={c}>
          <i style={{ background: FILL[c], outline: `1px solid ${STROKE[c]}` }} />
          {CELL_NAME[c]}
        </span>
      ))}
    </span>
  );
}

/** One row per warp, the execution unit underneath, from a cold start. */
function WarpTimeline({ tl, label }: { tl: Timeline; label: string }) {
  const R = tl.rows.length;
  const W = 720;
  const X = 44;
  const plotW = W - X - 8;
  const cw = plotW / tl.cycles;
  // Few warps get tall rows, so that the drawing stays legible when a narrow screen scales it down.
  const rh = Math.max(3, Math.min(R <= 8 ? 26 : 16, Math.floor(232 / R)));
  const ready = tl.rows.reduce((s, row) => s + row.filter((c) => c === 'ready').length, 0);
  const gap = rh >= 8 ? 2 : rh >= 5 ? 1 : 0.5;
  const top = 6;
  const uy = top + R * rh + 10;
  const uh = 16;
  const H = uy + uh + 22;
  const slots = tl.config.slots;
  // The unit strip as runs of cycles with the same number of slots in use.
  const used: { start: number; length: number; n: number }[] = [];
  tl.chosen.forEach((pick, t) => {
    const last = used[used.length - 1];
    if (last && last.n === pick.length) last.length += 1;
    else used.push({ start: t, length: 1, n: pick.length });
  });
  const labelled = R <= 16 ? tl.rows.map((_, w) => w) : [0, R - 1];
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={label}
      data-rows={R}
      data-cycles={tl.cycles}
      data-idle={idleCycles(tl)}
      data-ready={ready}
      style={{ width: '100%', display: 'block' }}
    >
      {tl.rows.map((row, w) => (
        <g key={w} data-warp={w}>
          {runs(row).map((r) => (
            <rect
              key={r.start}
              x={X + r.start * cw}
              y={top + w * rh}
              width={Math.max(0.6, r.length * cw - (cw > 3 ? 0.6 : 0))}
              height={rh - gap}
              fill={FILL[r.state]}
              stroke={rh >= 8 ? STROKE[r.state] : 'none'}
              strokeWidth="0.6"
            />
          ))}
        </g>
      ))}
      {labelled.map((w) => (
        <text key={w} x={X - 6} y={top + w * rh + Math.max(rh / 2 + 3, 8)} fontSize={rh >= 20 ? 12 : 9.5} textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
          w{w}
        </text>
      ))}
      <text x={X - 6} y={uy + 11} fontSize="9.5" textAnchor="end" fontWeight="600">unit</text>
      <rect x={X} y={uy} width={plotW} height={uh} fill="none" stroke="var(--ink-3)" strokeDasharray="2 2" strokeWidth="0.8" />
      {used.filter((u) => u.n > 0).map((u) => (
        <rect key={u.start} x={X + u.start * cw} y={uy + uh * (1 - u.n / slots)} width={u.length * cw} height={(uh * u.n) / slots} fill="var(--ink)" />
      ))}
      <text x={X} y={H - 4} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>cycle 0</text>
      <text x={X + plotW} y={H - 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>cycle {tl.cycles}</text>
    </svg>
  );
}

/** How many cycles of a configuration to draw: three trips round one warp's loop, within limits. */
const drawnCycles = (c: LatencyConfig) => Math.max(48, Math.min(300, 3 * period(c)));

// ---------------------------------------------------------------- explorer
interface PresetButton {
  key: string;
  label: string;
  set: Partial<LatencyConfig>;
}
const PRESET_BUTTONS: PresetButton[] = [
  { key: 'load', label: 'A load every other instruction', set: { ...PRESETS.loadHeavy } },
  { key: 'compute', label: 'Long compute between loads', set: { ...PRESETS.computeHeavy } },
  // 64 warp slots (p. 85) and four processing blocks (p. 76) on the book's example SM.
  { key: 'book', label: "The book's example SM: 64 warps, 4 at a time", set: { warps: 64, slots: 4 } },
];

function WarpsExplorer() {
  const [cfg, setCfg] = useState<LatencyConfig>({ warps: 4, ...PRESETS.loadHeavy, slots: 1 });
  const tl = useMemo(() => simulate(cfg, drawnCycles(cfg)), [cfg]);
  const use = useMemo(() => utilisation(cfg), [cfg]);
  const est = estimate(cfg);
  const need = warpsNeeded(cfg.work, cfg.latency, cfg.slots);
  const idle = idleCycles(tl);
  const active = (b: PresetButton) => (Object.keys(b.set) as (keyof LatencyConfig)[]).every((k) => cfg[k] === b.set[k]);

  const slider = (name: string, key: keyof LatencyConfig, min: number, max: number) => (
    <label className="field">
      {name} = {cfg[key]}
      <input
        type="range"
        min={min}
        max={max}
        value={cfg[key]}
        aria-label={name}
        style={{ width: '11rem' }}
        onChange={(e) => setCfg({ ...cfg, [key]: +e.target.value })}
      />
    </label>
  );

  const short = need - cfg.warps;
  return (
    <Bench
      title="How many warps hide a wait?"
      task="Raise R until the unit stops idling. Then keep going, and see what the extra warps do."
      foot="Cycle counts here are illustrative inputs, not measurements of any device. The drawing starts cold, with every warp ready at cycle 0; the percentage is taken after that start-up has passed."
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        {slider('Resident warps R', 'warps', 1, 64)}
        {slider('Cycles of work between memory accesses c', 'work', 1, 32)}
        {slider('Memory latency L, in cycles', 'latency', 1, 120)}
        {slider('Issue slots', 'slots', 1, 4)}
      </div>
      <div className="row" style={{ marginBottom: '1rem' }} role="group" aria-label="Presets">
        <span className="muted">Presets</span>
        {PRESET_BUTTONS.map((b) => (
          <button key={b.key} className="quiet" data-preset={b.key} aria-pressed={active(b)} onClick={() => setCfg({ ...cfg, ...b.set })}>
            {b.label}
          </button>
        ))}
      </div>
      <div className="cols plot">
        <div className="panel">
          <div style={{ marginBottom: '0.5rem' }}>
            <CellLegend />
          </div>
          <WarpTimeline tl={tl} label={`Timeline of ${cfg.warps} warps over ${tl.cycles} cycles`} />
          <p className="muted" style={{ margin: '0.5rem 0 0' }}>
            One row per warp. The strip marked "unit" is filled when the execution unit issued an instruction in that
            cycle{cfg.slots > 1 ? `, in proportion to how many of its ${cfg.slots} slots were used` : ''}, and empty when
            it had nothing to issue. In the {tl.cycles} cycles drawn it issued nothing in{' '}
            <span data-out="idle">{idle}</span>.
          </p>
        </div>
        <div className="panel" aria-live="polite">
          <p className="muted" style={{ margin: 0 }}>
            Execution unit busy
          </p>
          <p style={{ fontSize: '2rem', fontWeight: 600, margin: '0 0 0.2rem', lineHeight: 1.1 }}>
            <output data-out="util" data-value={pct(use)}>
              {pct(use)}%
            </output>
          </p>
          <p style={{ margin: '0 0 0.6rem' }}>
            Estimate by counting:{' '}
            {cfg.slots > 1
              ? `min(1, R × c / (slots × (c + L))) = min(1, ${cfg.warps} × ${cfg.work} / (${cfg.slots} × ${period(cfg)}))`
              : `min(1, R × c / (c + L)) = min(1, ${cfg.warps} × ${cfg.work} / ${period(cfg)})`}{' '}
            ={' '}
            <b>
              <span data-out="estimate" data-value={pct(est)}>
                {pct(est)}%
              </span>
            </b>
          </p>
          <p style={{ margin: '0 0 0.6rem' }}>
            Warps needed to keep {cfg.slots > 1 ? `all ${cfg.slots} slots` : 'the unit'} busy:{' '}
            {cfg.slots > 1 ? `${cfg.slots} × ` : ''}({cfg.work} + {cfg.latency}) / {cfg.work}, rounded up ={' '}
            <b>
              <span data-out="needed">{need}</span>
            </b>
          </p>
          <p style={{ margin: 0 }} data-out="verdict">
            {short > 0 ? (
              <>
                <b>{plural(short, 'warp')} short.</b> The unit has nothing to issue for {pct(1 - use)}% of its
                {cfg.slots > 1 ? ' slots' : ' cycles'}. Once the cold start has passed, every warp is either executing
                or waiting on memory; none is kept waiting for the unit.
              </>
            ) : short === 0 ? (
              <>
                <b>Exactly enough.</b> As one warp starts to wait, another's data is back.
              </>
            ) : (
              <>
                <b>{plural(-short, 'warp')} more than needed.</b> They add nothing: the unit was already full. Their
                data comes back and they queue for a turn, which is the grey in the drawing. Each one still holds its
                registers.
              </>
            )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the prediction's reveal
const PREDICTED: LatencyConfig = { warps: 16, ...PRESETS.loadHeavy, slots: 1 };
const PREDICTED_TL = simulate(PREDICTED, 96);
const NEEDED = warpsNeeded(PREDICTED.work, PREDICTED.latency);

// ---------------------------------------------------------------- questions
export const questionsS46: Record<string, Question> = {
  'ch04/4.6/q-util': {
    kind: 'numeric',
    id: 'ch04/4.6/q-util',
    prompt: (
      <>
        In the model of this section, 8 warps are resident on a unit with one issue slot. Each computes for 4 cycles
        between memory accesses and then waits 60 cycles. What fraction of its cycles does the execution unit spend
        issuing instructions, in percent?
      </>
    ),
    answer: 50,
    tol: 0.005,
    unit: '%',
    explain: (
      <>
        <p>
          One warp uses the unit for 4 of every 4 + 60 = 64 cycles. Eight of them use 32 of every 64, and none of them
          ever has to queue, so the unit is busy half the time.
        </p>
        <p>
          It would take 64 / 4 = 16 warps to fill it. The kernel did not change between 8 and 16, and neither did the
          hardware. Only the number of warps resident did, and that is what the next section is about.
        </p>
      </>
    ),
    source: 'our model of ch. 4 §4.6, pp. 83–85; a simplification, tested in app/src/sim/latency.ts',
  },
  'ch04/4.6/q-zero': {
    kind: 'choice',
    id: 'ch04/4.6/q-zero',
    prompt: (
      <>
        A CPU pays for every context switch. An SM changes the warp it issues from as often as every instruction and
        pays nothing. Why?
      </>
    ),
    options: [
      {
        label: 'The scheduler predicts which warp will be ready next and loads its state ahead of time.',
        why: 'Nothing is loaded, early or late. NVIDIA also says SMs do no speculative execution at all.',
      },
      {
        label: "Every resident warp's program counter and registers stay in on-chip registers for as long as it lives.",
        correct: true,
      },
      {
        label: 'All warps of a kernel run the same code, so they can share one set of registers and there is nothing to swap.',
        why: 'They share the code, not the data. Each thread of each warp has its own register values (section 4.4).',
      },
      {
        label: "The outgoing warp's state is saved to shared memory, which is fast enough for the cost to round to zero.",
        why: 'Nothing is saved anywhere. The state of a resident warp never leaves the registers it lives in.',
      },
    ],
    explain: (
      <p>
        A context switch costs time because state moves: out to memory for one thread, in from memory for the other.
        An SM keeps the state of every warp assigned to it in hardware for as long as the warp is there, so choosing
        another warp moves nothing. The cost has not vanished. It has become a cost in space: every resident warp
        holds its registers whether or not it is executing, and a kernel that uses many registers per thread leaves
        room for fewer warps.
      </p>
    ),
    source: 'ch. 4 §4.6, p. 84 and its sidebar; NVIDIA wording checked against the CUDA 13.5 guide, 2026-11-20',
  },
  'ch04/4.6/q-area': {
    kind: 'choice',
    id: 'ch04/4.6/q-area',
    prompt: <>What does latency tolerance let a GPU designer leave out, and what does it not do for a single thread?</>,
    options: [
      {
        label:
          'Large caches and branch prediction. A thread that waits still waits just as long; other warps use the unit.',
        correct: true,
      },
      {
        label: 'Large caches and branch prediction. Each thread also finishes sooner, because switching between warps is free.',
        why: 'Free switching keeps the unit busy. It does not shorten any one thread\'s wait.',
      },
      {
        label: 'Floating-point units. Arithmetic can be made to wait, so fewer units are needed, and a single thread is unaffected.',
        why: 'The other way round: arithmetic units are what the saved area is spent on.',
      },
      {
        label: 'Memory channels. With enough warps resident, a slow memory no longer matters to any thread or to the SM.',
        why: 'Memory channels are the other thing the area is spent on. Hiding a wait is not the same as removing the traffic; chapter 5 is about that limit.',
      },
    ],
    explain: (
      <p>
        A CPU core has one thread to run, so it spends area on making that thread's next instruction arrive quickly:
        caches so that data is near, prediction so that a branch does not stall. An SM always has another warp to
        issue from, so it can leave a slow instruction slow and spend the area on arithmetic units and memory
        channels. That raises the work done per second by the whole SM. It does nothing for the time one thread takes
        to get through its own instructions, which is the trade chapter 1 described.
      </p>
    ),
    source: 'ch. 4 §4.6, p. 84',
  },
  'ch04/4.6/q-lowocc': {
    kind: 'choice',
    id: 'ch04/4.6/q-lowocc',
    prompt: (
      <>
        Chapter 15's matrix multiplication keeps large tiles in registers and runs with one block of 256 threads per
        SM, an eighth of the thread slots on the book's example. A decode-time matrix-vector kernel on the same
        device does best with the SM full. Why can the first get away with so few warps?
      </>
    ),
    options: [
      {
        label: 'Between memory accesses its warps compute for many cycles on registers, so a few keep the units busy.',
        correct: true,
      },
      {
        label: 'Matrix multiplication has no control divergence, so each of its warps needs fewer cycles per result.',
        why: 'Neither kernel diverges much. Divergence is about passes within a warp, not about how many warps are needed to cover a wait.',
      },
      {
        label: 'With fewer warps resident each one gets more memory bandwidth, and that is what limits both kernels.',
        why: 'That would favour low occupancy for the memory-bound kernel too, and it is the one that wants the SM full.',
      },
      {
        label: 'Large blocks are scheduled first, so a 256-thread block is never kept waiting for a free SM.',
        why: 'No vendor documents an order of that kind, and it would not explain utilisation inside the SM.',
      },
    ],
    explain: (
      <>
        <p>
          In this section's terms the first kernel has a large c: long stretches of arithmetic on registers for each
          trip to memory. The warps needed are about (c + L) / c for each issue slot, which is close to 1 when c is
          large. The second has a small c, so the same formula asks for many warps.
        </p>
        <p>
          The register-heavy kernel could not have more warps anyway: its registers are what limit it to one block.
          The point is that it loses little by that. AMD's guide says the same of its matrix-core GEMMs: they often
          run at low occupancy because a few warps saturate the pipelines.
        </p>
      </>
    ),
    source: 'ch. 15 §15.7, p. 363; AMD HIP performance guide, checked 2026-11-20; the model is ours',
  },
};

// ---------------------------------------------------------------- the section
export function S46() {
  return (
    <>
      <p>
        Section 4.4 put a warp on a processing block: one fetch, every lane. It left a mismatch unexplained. The
        book's example SM has {BOOK_OVERSUBSCRIPTION.coresPerSm} streaming processors and room for{' '}
        {BOOK_OVERSUBSCRIPTION.threadsPerSm} threads. Sixteen threads for every processor cannot all execute at once.
        What are the other fifteen for?
      </p>
      <p>
        They are there to wait, and that is the design. At any instant an SM executes only some of the warps assigned
        to it. Early designs issued one instruction, for one warp, at a time. Later ones have several processing
        blocks and issue for a few warps at once, like the two-block SM of the book's Fig. 4.8. Either way the warps
        that are resident far outnumber the ones executing.
      </p>
      <Source section="4.6" pages="82–83" figs="refers back to Fig. 4.8" />

      <h2>What the extra warps are for</h2>
      <p>
        Some instructions take a long time to deliver their result. The book's main example is a read from global
        memory: the request leaves the SM for the off-chip DRAM of section 4.1, and the data comes back many cycles
        later. A warp whose next instruction needs that data can do nothing useful until it arrives.
      </p>
      <p>
        The SM does not wait with it. A warp that is waiting on an earlier operation is not selected. Every other warp
        is <b>ready</b>, and when several are ready a priority logic picks one to issue from. Choosing among the ready
        warps at every issue is called <b>fine-grained multithreading</b>. Its effect is that the execution units stay
        busy although individual instructions are slow, which the book calls <b>latency tolerance</b> or latency
        hiding.
      </p>
      <Source section="4.6" pages="83" />
      <p>
        The effect is about time, so watch it in time before reading on. The animation uses a small case: 2 cycles of
        work, then a 6-cycle wait.
      </p>
      <LatencyScene />

      <h2>Other waits it hides</h2>
      <p>
        Memory is the long wait, not the only one. Floating-point arithmetic is pipelined, so a result is not
        available the cycle after it is asked for, and a conditional branch takes time to resolve. The same
        scheduling covers both. While one warp's result is on its way, the unit issues from another.
      </p>
      <Source section="4.6" pages="83–84" />

      <h2>Why the switch is free</h2>
      <p>
        On a CPU, moving a core from one thread to another is a context switch. The program counter and registers of
        the outgoing thread are saved to memory and those of the incoming thread are loaded, and that takes time. A
        design that paid that price at every switch could not afford to switch at every instruction.
      </p>
      <Source section="4.6" pages="84" figs="and its sidebar on context switching" />
      <p>
        An SM never does it. The execution state of every warp assigned to it stays in hardware registers from the
        moment the warp arrives until it finishes. Selecting a different warp changes where the next instruction comes
        from and nothing else, so no idle time appears between one warp's instruction and another's. The book's name
        for this is <b>zero-overhead scheduling</b>.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 guide (ch04-facts.md, A11)">
        <p>
          NVIDIA: each warp's context is "maintained on-chip throughout the warp's lifetime", so switching costs
          nothing; SMs issue in order, with no branch prediction or speculation. How a ready warp is chosen is not
          documented.
        </p>
      </Today>
      <p>
        Free in time is not free in space. Every resident warp holds its registers for as long as it is resident,
        whether it is executing, waiting on memory, or ready and not chosen. A kernel that uses more registers per
        thread leaves room for fewer warps, and section 4.7 works out how many.
      </p>
      <Beyond title="On HIP and Metal">
        <p>
          AMD documents the same for a compute unit (single-cycle switching, all contexts resident) and publishes its
          policy: oldest warp fetches first, issue goes round the pools in turn. No Apple statement found: for Metal
          this is unverified.
        </p>
        <p className="muted">Checked 2026-11-20 (ch04-facts.md, B8).</p>
      </Beyond>
      <Ask q={questionsS46['ch04/4.6/q-zero']} />

      <h2>What the design spends the savings on</h2>
      <p>
        Latency tolerance is the main reason a GPU can do without two things a CPU spends a great deal of chip area
        on: large caches and branch prediction. A CPU core has one thread to run and must make that thread's next
        instruction arrive quickly. An SM has other warps to run, so it can leave a slow instruction slow. The area
        goes to floating-point execution units and memory access channels.
      </p>
      <Source section="4.6" pages="84" />
      <Ask q={questionsS46['ch04/4.6/q-area']} />

      <h2>How many warps is enough</h2>
      <p>
        For this to work an SM has to hold many more threads than it can execute at once. The book's example holds up
        to {BOOK_OVERSUBSCRIPTION.threadsPerSm} threads for {BOOK_OVERSUBSCRIPTION.coresPerSm} streaming processors,{' '}
        {BOOK_OVERSUBSCRIPTION.ratio} times more than can execute in any one cycle, and the book calls that
        oversubscription essential.
      </p>
      <Source section="4.6" pages="84–85" />
      <p>
        How many a particular kernel needs depends on the kernel, and a small model is enough to see how. A warp
        issues c instructions, one per cycle, and the last of them starts a memory access. Then it waits L cycles.
        Then it repeats. Alone, it keeps the unit busy for c of every c + L cycles. R warps keep it busy R times as
        much, until the unit is full:
      </p>
      <p className="ui" style={{ fontSize: '1rem' }}>
        utilisation = min(1, R × c / (c + L)), so the unit is full from R = (c + L) / c warps, rounded up.
      </p>
      <Simplification>
        <p>
          A fixed wait, with no limit on how many warps wait at once: memory has unlimited bandwidth here. One
          instruction per cycle. Every warp runs the same loop.
        </p>
        <p>Ready warps are chosen oldest first. That is our choice; NVIDIA does not document its policy.</p>
      </Simplification>
      <Predict
        id="ch04/4.6/p-warps-needed"
        prompt={
          <>
            A kernel's warps each do {PREDICTED.work} cycles of work and then wait {PREDICTED.latency} cycles on global
            memory. How many warps have to be resident to keep one execution unit busy all the time?
          </>
        }
        unit="warps"
      >
        {(v) => (
          <>
            <p>
              <b>{NEEDED}.</b> One warp uses the unit for {PREDICTED.work} of every {period(PREDICTED)} cycles, so it
              takes {period(PREDICTED)} / {PREDICTED.work} = {NEEDED} of them to fill it.{' '}
              {v === NEEDED
                ? 'That is what you had.'
                : v < NEEDED
                  ? `With ${Math.max(1, Math.floor(v))} the unit would be busy ${pct(utilisation({ ...PREDICTED, warps: Math.max(1, Math.floor(v)) }))}% of the time.`
                  : `With ${Math.floor(v)} it would be no busier than with ${NEEDED}: the extra warps queue.`}{' '}
              Below, {NEEDED} warps from a cold start. The unit strip has no gap.
            </p>
            <div className="panel" data-out="predicted">
              <WarpTimeline tl={PREDICTED_TL} label={`Timeline of ${NEEDED} warps, each working ${PREDICTED.work} cycles and waiting ${PREDICTED.latency}`} />
            </div>
          </>
        )}
      </Predict>
      <p>
        The explorer runs the model and draws what each warp is doing in each cycle. Three things are worth finding.
        Start below the threshold and raise R one warp at a time. Then go past the threshold and look at what the
        extra warps do with their time. Then switch between the two kernel presets without touching R. The fourth
        slider lets the unit issue for more than one warp per cycle, as an SM with several processing blocks does;
        the last preset is the book's example SM, 64 warp slots and four processing blocks.
      </p>
      <WarpsExplorer />
      <Ask q={questionsS46['ch04/4.6/q-util']} />
      <p>
        Below the threshold the unit idles, and every warp you add is put to full use. At the threshold the idling
        stops. Past it, extra warps change nothing you can measure at the unit: their data comes back and they wait
        for a turn. And the threshold itself is set by the ratio of waiting to working, which belongs to the kernel.
        The last preset puts the book's example SM into the model: 64 warp slots, four processing blocks. There it
        stays busy as long as a warp waits no more than 15 cycles for each cycle it works, which is the book's 16 to
        1 seen from the kernel's side.
      </p>
      <p>
        One limit the model leaves out matters for the kernels you care about. More waiting warps means more requests
        in flight, and memory can serve only so many bytes per second. Once that bandwidth is used up, adding warps
        stops helping however idle the execution units are. Chapter 5 calls that ceiling the roofline.
      </p>

      <h2>Which kernels need a full SM</h2>
      <p>
        Put two kernels from an inference stack into the model. When a model decodes one token at a time, most of the
        work is matrix-vector products and embedding lookups: load a weight, multiply and add, load the next. That is
        a small c. A thread spends almost all of its time waiting on global memory, so throughput depends on having
        many warps resident, and such kernels want the SM as full as it will go.
      </p>
      <p>
        A matrix multiplication written to keep its working tiles in registers is the opposite case. Each thread
        loads a strip of inputs and then computes on registers for a long time before it needs memory again. That is
        a large c, and a few warps keep the units busy. Chapter 15 builds such a kernel and runs it on purpose with
        one block of 256 threads per SM, which is 12.5% of the 2048 thread slots it assumes.
      </p>
      <Beyond>
        <p>The pairing of these two kernels with the model is ours. The 12.5% is the book's (ch. 15, p. 363).</p>
        <p>
          AMD's guide makes the same point about its matrix-core GEMMs: low occupancy, because a few warps saturate
          the pipelines.
        </p>
        <p className="muted">Checked 2026-11-20 (ch04-facts.md, B10).</p>
      </Beyond>
      <Ask q={questionsS46['ch04/4.6/q-lowocc']} />

      <h2>What it costs</h2>
      <p>
        Keeping warps resident is what makes the wait disappear, and it is paid for in two currencies the SM has a
        fixed amount of: slots to hold warps and blocks, and registers to hold their state. How those are shared
        out, and how a small change in a kernel can halve the number of warps that fit, is{' '}
        <a href={href('ch04', '4.7')}>section 4.7</a>.
      </p>
    </>
  );
}
