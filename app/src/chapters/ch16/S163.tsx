import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import {
  NEIGHBOURS,
  PRESETS,
  inputsOf,
  presetFronts,
  stencil,
  stepAt,
  wavefronts,
  type NeighbourId,
  type PresetId,
} from '../../sim/wavefront';
import { FIG_16_5, INF, VERTICES, runSteps, stateAfter, stepTrace, type BlockEvent } from '../../sim/floydwarshall';
import { flagsAfter, frontMakespan, numbering, schedule, slotAt, type Rule, type StartOrder } from '../../sim/wavesched';
import { NeighboursScene, SCENE } from './NeighboursScene';

// Section 16.3 of the course: book §16.2 to §16.4 and §16.8 (pp. 374–381, 397–399).
// Four parts: top-down or bottom-up; numbering cells by step (pattern explorer, on
// src/sim/wavefront.ts); Floyd-Warshall in place (stepper, on src/sim/floydwarshall.ts);
// who waits for whom (scene and scheduler explorer, on src/sim/wavesched.ts); and one
// bridge, PyTorch's CTC loss. Every number stated comes from those models.

// The kernel and host loop as shown: both appear verbatim in labs/ch16/snippets/fw_bottomup.cu
// (snippets-s163.test.ts checks it, and pins the line numbers the stepper highlights).
const FW_KERNEL = `#define FW_INF (INT_MAX / 2)  // "no path"; FW_INF + FW_INF still fits in an int

__global__ void FW_bottomup(int k, int* dist, int n_vertices) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    bool inRange = col < n_vertices;   // no return before the barrier
    int row = blockIdx.y;
    int distIndex = n_vertices * row + col;

    __shared__ int dist_k_col;         // dist[row][k], the same for the whole block
    if (threadIdx.x == 0)
        dist_k_col = dist[n_vertices * row + k];
    __syncthreads();

    if (dist_k_col == FW_INF) return;  // no path from row to k: the block retires

    if (inRange) {
        int dist_k_row = dist[k * n_vertices + col];   // dist[k][col]
        if (dist_k_row != FW_INF) {
            int new_distance = dist_k_col + dist_k_row;
            if (new_distance < dist[distIndex])
                dist[distIndex] = new_distance;
        }
    }
}`;
const FW_HOST = `void floydWarshall(int* dist_d, int n_vertices, int threads) {
    dim3 dimBlock(threads);
    dim3 dimGrid((n_vertices + threads - 1) / threads, n_vertices);
    for (int k = 0; k < n_vertices; k++)
        FW_bottomup<<<dimGrid, dimBlock>>>(k, dist_d, n_vertices);  // in order: one stream
}`;
/** Lines of FW_KERNEL the stepper highlights. */
export const FW_LINES = { read: 11, barrier: 12, retire: 14, rowK: 17, compare: 20, write: 21 };
const FW_TIERS: Record<string, Tier> = {
  dist: 'global', dist_d: 'global', dist_k_col: 'shared',
  dist_k_row: 'reg', new_distance: 'reg', col: 'reg', row: 'reg', inRange: 'reg', distIndex: 'reg', k: 'reg',
};

// Numbers the page states, from the models.
const SW8 = presetFronts('smith-waterman', 8, 8);
const H264_8 = presetFronts('h264', 8, 8);
const V = (name: string) => VERTICES.indexOf(name as (typeof VERTICES)[number]);
const FW_A = stepTrace(FIG_16_5, V('a'));
const FW_C = stepTrace(runSteps(FIG_16_5, 2), V('c'));
const FW_C_COL = VERTICES.flatMap((w, i) => (FW_C.before[i][V('c')] === INF || i === V('c') ? [] : [`${w} ${FW_C.before[i][V('c')]}`]));
const cellText = (c: { i: number; j: number; to: number }) => `${VERTICES[c.i]} to ${VERTICES[c.j]} = ${c.to}`;
const listAnd = (xs: string[]) => (xs.length < 2 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1]);
const F2 = SCENE.front2.makespan!;
const C2 = SCENE.counter2.makespan!;
// q-enough-sms asks about a grid the explorer cannot show (it stops at T = 8 and S = 8).
const BIG_T = 10;
const BIG_S = 16;
const F_BIG = frontMakespan(BIG_T, BIG_S);
const C_BIG = schedule(BIG_T, BIG_S, 'counter').makespan!;

export const questionsS163: Record<string, Question> = {
  'ch16/16.3/q-topdown': {
    kind: 'choice',
    id: 'ch16/16.3/q-topdown',
    prompt: (
      <>
        A colleague ports a memoised recursive solver to the GPU: one thread per sub-problem, results cached in a hash
        table in global memory. What is the strongest reason the chapter goes bottom up instead?
      </>
    ),
    options: [
      {
        label:
          "Neighbouring threads' lookups land at unrelated places in the hash table, so a warp's loads do not coalesce, and which sub-problem runs when is decided at run time.",
        correct: true,
      },
      {
        label: 'GPU code cannot call functions, so the recursion cannot be compiled.',
        why: 'Device functions can be called, and can recurse on CUDA and HIP; Metal allows recursion in kernels from Metal 2.4.',
      },
      {
        label: 'GPUs have no hash tables.',
        why: 'A hash table is an array and some arithmetic; any kernel can build one. The trouble is where its lookups land.',
      },
      {
        label: 'Recursion is forbidden on every GPU backend.',
        why: 'Only the CUDA kernel itself (the __global__ function) may not recurse. Device functions may, with a stack whose size you set.',
      },
    ],
    explain: (
      <p>
        Bottom up replaces both problems. A thread computes the address of its cell from its index, so neighbouring
        threads read neighbouring cells, and the order of the steps is fixed before the kernel starts. The book's first
        reason, that GPUs handle deep chains of calls poorly, is about cost: NVIDIA documents a stack per thread and AMD
        one per warp; AMD and Apple warn about its size; none forbids recursion in device functions.
      </p>
    ),
    source: 'ch. 16 §16.2, pp. 374–376; ch16-facts A1, B1, C1',
  },
  'ch16/16.3/q-h264': {
    kind: 'numeric',
    id: 'ch16/16.3/q-h264',
    prompt: (
      <>
        An 8 × 8 table in which each cell needs its left neighbour and the cell above and to the right of it, the H.264
        pattern of Fig. 16.2(d). How many barrier-separated steps does it take? Work it out before you set the explorer
        to H.264.
      </>
    ),
    answer: H264_8.steps,
    unit: 'steps',
    explain: (
      <p>
        Row 0 is a chain: cell (0, c) runs at step c. Cell (1, 0) needs (0, 1), done at step 1, so it runs at step 2, and
        every row starts two steps after the row above: the step of (r, c) is c + 2r. The last cell, (7, 7), runs at step
        7 + 14 = {stepAt(H264_8, 7, 7)}, so the steps are 0 to {H264_8.steps - 1}: {H264_8.steps} of them, against{' '}
        {SW8.steps} for the three-neighbour stencil on the same table, and never more than {H264_8.widest} cells wide.
      </p>
    ),
    source: 'ch. 16 §16.3, pp. 376–377, Fig. 16.2(d)',
  },
  'ch16/16.3/q-live': {
    kind: 'numeric',
    id: 'ch16/16.3/q-live',
    prompt: (
      <>
        You need only the final score of an L × L table with the north, west and north-west stencil, not the alignment
        itself. Besides the front being written, how many earlier fronts must stay in memory?
      </>
    ),
    answer: SW8.reach,
    unit: 'fronts',
    explain: (
      <p>
        North and west lie on the previous anti-diagonal, north-west on the one before it, and nothing reads further back
        (the explorer's "farthest input" reads {SW8.reach}). Three anti-diagonals of at most L cells: memory that grows
        like L instead of L². The book keeps Smith-Waterman's whole table only because the traceback, which recovers the
        alignment, walks back through it from the best cell.
      </p>
    ),
    source: 'ch. 16 §16.3 and §16.5, pp. 377, 384',
  },
  'ch16/16.3/q-fw-changes': {
    kind: 'numeric',
    id: 'ch16/16.3/q-fw-changes',
    prompt: (
      <>
        Start from Fig. 16.5's table as printed. Run step k = a, then step k = b. How many cells does step k = c change?
      </>
    ),
    answer: FW_C.changed.length,
    unit: 'cells',
    explain: (
      <p>
        After steps a and b, column c holds {listAnd(FW_C_COL)}, and row c holds only c to e = 10. Each finite pair gives
        a candidate path through c, and {FW_C.changed.length} of them beat the current entry:{' '}
        {listAnd(FW_C.changed.map(cellText))}. The pair e, e would give 8 + 10 = 18, but e to e is already 0. Rows{' '}
        {listAnd(FW_C.retired.map((i) => VERTICES[i]))} have no path to c, so their blocks retire without reading row c.
      </p>
    ),
    source: 'ch. 16 §16.4, pp. 380–381, Fig. 16.5 (the table as printed); checked by floydwarshall.test.ts',
  },
  'ch16/16.3/q-fw-inplace': {
    kind: 'choice',
    id: 'ch16/16.3/q-fw-inplace',
    prompt: (
      <>
        Why can every thread of step k update <code>dist</code> in place while other threads are reading row k and
        column k of the same table?
      </>
    ),
    options: [
      {
        label:
          'The thread that owns dist[i][k] computes min(dist[i][k], dist[i][k] + dist[k][k]), which with dist[k][k] = 0 is dist[i][k] again, and likewise for row k. So no value that another thread reads changes during the step.',
        correct: true,
      },
      {
        label: 'The threads of a block run in lockstep, so all reads happen before any write.',
        why: 'Different blocks read row k too, and nothing orders blocks within a launch.',
      },
      {
        label: 'Launches are ordered, so step k + 1 sees all of step k.',
        why: 'True, but that orders the steps. The question is about threads within one step.',
      },
      {
        label: 'min is commutative, so the order of the updates does not matter.',
        why: 'The order of updates matters as soon as a value someone reads can change. The argument has to show that it cannot.',
      },
    ],
    explain: (
      <p>
        The argument needs both conditions the book states: dist[k][k] = 0 and no negative weights. Break the first (the
        stepper's −1 switch) and row k and column k change during step k, so what a block reads depends on whether block k
        has already run: the result depends on the order. Then you need two buffers, reading the old table and writing a
        new one (double buffering, chapter 6).
      </p>
    ),
    source: 'ch. 16 §16.4, pp. 379–380',
  },
  'ch16/16.3/q-infinity': {
    kind: 'choice',
    id: 'ch16/16.3/q-infinity',
    prompt: (
      <>
        Fig. 16.4 as printed tests <code>dist_k_col == INFINITY</code>, where <code>dist_k_col</code> is an{' '}
        <code>int</code> and <code>INFINITY</code> comes from <code>&lt;cmath&gt;</code>. What happens?
      </>
    ),
    options: [
      {
        label:
          'The int is converted to float and compared with infinity, which no int equals, so the early exits never fire. If "no edge" is stored as INT_MAX, the sum then overflows.',
        correct: true,
      },
      {
        label: 'It works: INFINITY converts to the largest int.',
        why: 'The conversion goes the other way. When an int meets a float in a comparison, the int becomes a float.',
      },
      {
        label: 'The early exits fire too often, retiring blocks that do have a path.',
        why: 'No int compares equal to positive infinity, so they never fire at all.',
      },
      {
        label: 'It is fine as long as the weights are small.',
        why: 'The test can never be true, whatever the weights.',
      },
    ],
    explain: (
      <p>
        In a comparison of an <code>int</code> with a <code>float</code>, the <code>int</code> is converted to{' '}
        <code>float</code>. <code>INFINITY</code> is a <code>float</code> constant, positive infinity, and no{' '}
        <code>int</code> converts to it. So blocks that should retire do the work anyway, and{' '}
        <code>dist_k_col + dist_k_row</code> adds whatever stands for "no edge". With <code>INT_MAX</code> that sum
        overflows, which is undefined behaviour. The corrected kernel uses the integer sentinel{' '}
        <code>FW_INF = INT_MAX / 2</code>, so <code>FW_INF + FW_INF</code> still fits, and compares with it. A side
        effect worth noticing: with this sentinel a sum that involves <code>FW_INF</code> is never smaller than any entry,
        so the exits only save work.
      </p>
    ),
    source: 'ch. 16 Fig. 16.4, p. 380 (notes/ch16.md, "Unclear in the PDF": INFINITY is undefined in the listing); ch16-facts A3',
  },
  'ch16/16.3/q-ex1': {
    kind: 'numeric',
    id: 'ch16/16.3/q-ex1',
    prompt: (
      <>
        The book's exercise 1. Run Floyd-Warshall with 16 × 16 blocks: each block covers 16 rows and 16 columns of the
        table, loads its 16 values of column k and its 16 values of row k into shared memory once, and then every thread
        updates its cell from there. How many global reads of row k or column k does the block make per cell it updates?
        (Do not count each thread's read of its own cell.)
      </>
    ),
    answer: 32 / 256,
    tol: 0.01,
    unit: 'reads per cell',
    explain: (
      <p>
        32 reads for 256 cells: 0.125 per cell, each value used 16 times. The book's kernel with 256-thread blocks along
        one row makes 1 read of dist[row][k] and 256 reads of row k for 256 cells, about 1.004 per cell: the column value
        is shared by the block, the row values are not. Two things the count leaves out. The 16 column values lie in 16
        different rows, <code>n_vertices</code> apart, so those reads do not coalesce. And the early retirement becomes a
        per-thread test, since a block no longer covers a single row. Which kernel is faster on your GPU is a
        measurement, not something this count settles.
      </p>
    ),
    source: 'ch. 16 §16.10, exercise 1, p. 399; the count is ours (notes/ch16.md, Exercises)',
  },
  'ch16/16.3/q-enough-sms': {
    kind: 'numeric',
    id: 'ch16/16.3/q-enough-sms',
    prompt: (
      <>
        A {BIG_T} × {BIG_T} grid of tiles on {BIG_S} SMs, each tile taking one time step. How many time steps does
        starting each tile on its two flags save over a relaunch per front?
      </>
    ),
    answer: F_BIG - C_BIG,
    unit: 'steps',
    explain: (
      <p>
        None. Every front has at most {BIG_T} tiles, so with {BIG_S} SMs each front runs in one step and a relaunch per
        front takes {F_BIG} steps, 2T − 1. Flags take {C_BIG} as well, because no schedule can beat the chain of tiles
        from the corner to the corner, also 2T − 1 long. Flags pay off only between the extremes: when the machine is narrower than the
        widest front, but not so narrow that every step is full anyway (with one SM both take T² steps).
      </p>
    ),
    source: 'ch. 16 §16.8, pp. 397–399, Fig. 16.21; checked by wavesched.test.ts',
  },
  'ch16/16.3/q-row-owner': {
    kind: 'choice',
    id: 'ch16/16.3/q-row-owner',
    prompt: (
      <>
        The book gives each persistent block one row of tiles by <code>blockIdx.x</code>, and has it spin on the flag of
        the tile above before each tile. Under an ordinary CUDA launch, what is wrong?
      </>
    ),
    options: [
      {
        label:
          'Nothing promises that the block owning the row above has started. A spinning block keeps its SM, so if later rows start first, or the grid does not fit at once, nothing finishes.',
        correct: true,
      },
      {
        label: 'The flag must be declared volatile.',
        why: 'It must be written and read as a device-scope atomic with release and acquire, which volatile is not. And a correct flag does not help if the block that sets it never runs.',
      },
      {
        label: 'Nothing: blocks start in blockIdx order.',
        why: 'No scheduler ordering is promised: blocks may run in any order (chapter 4).',
      },
      {
        label: 'Nothing, as long as the grid has no more blocks than the GPU has SMs.',
        why: 'A grid that fits is the condition for a cooperative launch, and only that launch promises that every block makes progress. Under an ordinary launch nothing backs it.',
      },
    ],
    explain: (
      <p>
        Two documented-safe fixes on CUDA: hand tiles out from an atomic counter in front order, so every tile a block
        waits on was claimed earlier by a block that is already running, or launch the kernel cooperatively. On HIP the
        same source compiles, but AMD documents no progress guarantee between blocks, so the design is unverified there.
        Metal documents no progress guarantee between threadgroups at all; on Metal the relaunch stays.
      </p>
    ),
    source: 'ch. 16 §16.8, p. 398; ch16-facts A4, B3, C4; ch11-facts A6, C5; ch04-facts A5',
  },
  'ch16/16.3/q-ctc': {
    kind: 'choice',
    id: 'ch16/16.3/q-ctc',
    prompt: (
      <>
        PyTorch's CUDA CTC loss gives each block whole sequences and loops over time with a <code>__syncthreads()</code>{' '}
        at every step. Why does it never relaunch, or use flags between blocks?
      </>
    ),
    options: [
      {
        label:
          'Step t needs only step t − 1 of the same sequence, and different sequences are independent, so the whole dependency stays inside one block, where a block barrier covers it.',
        correct: true,
      },
      { label: 'CTC has no dependency between time steps.', why: 'alpha at step t needs three entries of step t − 1.' },
      {
        label: 'It uses atomics instead of a barrier.',
        why: 'The loop has a barrier at every time step; that is the synchronisation.',
      },
      {
        label: 'PyTorch runs the batch one sequence at a time.',
        why: 'Blocks handle different sequences in parallel. The loop over time is inside each block.',
      },
    ],
    explain: (
      <p>
        When the target is wider than the block, the same block walks over chunks of positions in dependency order, left
        to right for alpha, so it still never waits on another block. The cheapest synchronisation that covers the
        dependency is a block barrier, and that is what it uses. The MPS kernel does the same with one threadgroup per
        batch item.
      </p>
    ),
    source: 'ch16-facts D2 (PyTorch LossCTC.cu and LossOps.metal, read 2026-11-20); the mapping to the chapter is ours',
  },
};

// ================================================================ the pattern explorer
/** Lets a row of segment buttons wrap on a narrow screen instead of running off it. */
const WRAP = { flexWrap: 'wrap', rowGap: 4 } as const;
const SEL0: [number, number] = [4, 4];

function PatternExplorer() {
  const [preset, setPreset] = useState<PresetId | null>('smith-waterman');
  const [on, setOn] = useState<NeighbourId[]>(['N', 'W', 'NW']);
  const [rows, setRows] = useState(8);
  const [cols, setCols] = useState(8);
  const [sel, setSel] = useState<[number, number]>(SEL0);
  const f = useMemo(() => wavefronts(rows, cols, stencil(on)), [rows, cols, on]);
  const r0 = Math.min(sel[0], rows - 1);
  const c0 = Math.min(sel[1], cols - 1);
  const s0 = stepAt(f, r0, c0);
  const ins = inputsOf(f, r0, c0);

  const choose = (id: PresetId) => {
    const p = PRESETS.find((x) => x.id === id)!;
    setPreset(id);
    setOn([...p.neighbours]);
    setRows(p.rows);
    setCols(p.cols);
    setSel(p.rows === 1 ? [0, Math.floor(p.cols / 2)] : SEL0);
  };
  const toggle = (id: NeighbourId) => {
    const next = on.includes(id) ? on.filter((x) => x !== id) : [...on, id];
    setOn(next);
    const match = PRESETS.find((p) => p.neighbours.length === next.length && p.neighbours.every((n) => next.includes(n)));
    setPreset(match ? match.id : null);
  };

  const C = 30;
  const P = 8;
  const gw = cols * C + 2 * P;
  const gh = rows * C + 2 * P;
  const cx = (c: number) => P + c * C + C / 2;
  const cy = (r: number) => P + r * C + C / 2;
  const barW = Math.max(6, Math.min(26, Math.floor(440 / f.steps)));
  const barH = 90;
  const sizes = (
    <svg
      viewBox={`0 0 ${f.steps * barW + 40} ${barH + 44}`}
      role="img"
      aria-label={`Cells per step: ${f.sizes.join(', ')}`}
      style={{ width: '100%', maxWidth: f.steps * barW + 40, display: 'block' }}
    >
      {f.sizes.map((n, s) => {
        const h = (n / f.widest) * barH;
        return (
          <g key={s} data-bar={s} data-size={n}>
            <rect x={30 + s * barW + 1} y={18 + barH - h} width={barW - 2} height={h} fill="var(--ink)" fillOpacity={s === s0 ? 0.85 : 0.25} />
            {barW >= 16 ? (
              <text x={30 + s * barW + barW / 2} y={18 + barH - h - 3} fontSize="9.5" textAnchor="middle">{n}</text>
            ) : null}
          </g>
        );
      })}
      <text x={0} y={26} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{f.widest}</text>
      <text x={0} y={18 + barH} fontSize="10" style={{ fill: 'var(--ink-2)' }}>0</text>
      <text x={30} y={barH + 38} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>step 0 to {f.steps - 1}: cells in each step (dark: the selected cell's step)</text>
    </svg>
  );

  return (
    <Bench
      title="Which cells can run together"
      task="Pick a pattern or build one, then click a cell to see what it waits for."
      foot={<>Each number is the step at which the cell can run: one more than the latest step among the cells it reads. Darker means later. Thick outline: the selected cell; dashed outline: the cells it reads; thin outline: the cells that share its step.</>}
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Pattern" style={WRAP}>
          {PRESETS.map((p) => (
            <button key={p.id} aria-pressed={preset === p.id} onClick={() => choose(p.id)} title={p.figure}>
              {p.label}
            </button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted">each cell reads:</span>
        {NEIGHBOURS.map((n) => (
          <label key={n.id} className="check">
            <input type="checkbox" checked={on.includes(n.id)} onChange={() => toggle(n.id)} data-neighbour={n.id} /> {n.label}
          </label>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Rows
          <select value={rows} onChange={(e) => { setRows(+e.target.value); setPreset(null); }} aria-label="Rows">
            {Array.from({ length: 16 }, (_, i) => i + 1).map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          Columns
          <select value={cols} onChange={(e) => { setCols(+e.target.value); setPreset(null); }} aria-label="Columns">
            {Array.from({ length: 16 }, (_, i) => i + 1).map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <span className="muted">{preset ? PRESETS.find((p) => p.id === preset)!.figure : 'your own stencil'}</span>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(16rem, 1fr))' }}>
        <div>
          <svg
            viewBox={`0 0 ${gw} ${gh}`}
            role="img"
            aria-label={`A ${rows} by ${cols} table numbered by step`}
            data-grid={`${rows}x${cols}`}
            style={{ width: '100%', maxWidth: Math.max(160, gw * 1.15), display: 'block' }}
          >
            {f.step.map((s, i) => {
              const r = Math.floor(i / cols);
              const c = i % cols;
              const isSel = r === r0 && c === c0;
              const same = s === s0;
              return (
                <g key={i} data-cell={`${r},${c}`} data-step={s} onClick={() => setSel([r, c])} style={{ cursor: 'pointer' }}>
                  <rect
                    x={P + c * C + 1}
                    y={P + r * C + 1}
                    width={C - 2}
                    height={C - 2}
                    rx="2"
                    fill="var(--ink)"
                    fillOpacity={0.04 + (0.32 * s) / Math.max(1, f.steps - 1)}
                    stroke={isSel || same ? 'var(--ink)' : 'var(--rule)'}
                    strokeWidth={isSel ? 2.8 : same ? 1.6 : 1}
                  />
                  <text x={cx(c)} y={cy(r) + 4} fontSize="11" textAnchor="middle">{s}</text>
                </g>
              );
            })}
            {ins.map(([r, c]) => (
              <rect
                key={`${r},${c}`}
                data-input={`${r},${c}`}
                x={P + c * C + 3.5}
                y={P + r * C + 3.5}
                width={C - 7}
                height={C - 7}
                rx="2"
                fill="var(--ink)"
                fillOpacity={0.14}
                stroke="var(--ink)"
                strokeWidth="2"
                strokeDasharray="4 2.5"
                pointerEvents="none"
              />
            ))}
          </svg>
          <p className="muted" style={{ margin: '0.4rem 0 0' }} data-selected>
            Cell ({r0}, {c0}) runs at step {s0}.{' '}
            {ins.length
              ? `It reads ${ins.map(([r, c]) => `(${r}, ${c}) at step ${stepAt(f, r, c)}`).join(', ')}.`
              : 'It reads only boundary values.'}{' '}
            {f.sizes[s0]} cells share its step.
          </p>
        </div>
        <div>
          {sizes}
          <table className="data" style={{ marginTop: '0.8rem' }}>
            <tbody>
              <tr><td>steps (barrier-separated)</td><td data-k="steps">{f.steps}</td></tr>
              <tr><td>widest step</td><td data-k="widest">{f.widest} cells</td></tr>
              <tr><td>cells per step on average</td><td data-k="mean">{f.mean.toFixed(2)}</td></tr>
              <tr><td>farthest input, fronts back</td><td data-k="reach">{f.reach}</td></tr>
              <tr>
                <td>row r + 1 starts after row r by</td>
                <td data-k="lag">{f.rowLag === null ? 'one row only' : `${f.rowLag} step${f.rowLag === 1 ? '' : 's'}`}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ================================================================ the Floyd-Warshall stepper
const N = VERTICES.length;
const HEAD = { background: 'transparent', border: 0, fontWeight: 650, color: 'var(--ink-2)' } as const;

function describeBlock(b: BlockEvent, k: number, before: number[][]): { text: string; stale: boolean } {
  const v = VERTICES[b.row];
  const kv = VERTICES[k];
  const head = `Block for row ${v} (blockIdx.y = ${b.row}): thread 0 reads dist[${v}][${kv}] into the shared dist_k_col`;
  if (b.retired)
    return {
      text: `${head}. It is FW_INF: there is no path from ${v} to ${kv}, so nothing in this row can improve. After the barrier the whole block returns.`,
      stale: false,
    };
  const stale = b.distKCol !== before[b.row][k] || b.rowK.some((x, col) => x !== before[k][col]);
  const reads = b.rowK.flatMap((x, col) => (x === INF ? [] : [`${VERTICES[col]} = ${x}`]));
  const writes = b.writes.map((w) => `${v} to ${VERTICES[w.col]}: ${w.from === INF ? 'no path' : w.from} becomes ${w.to}`);
  return {
    text:
      `${head}: ${b.distKCol}. Its threads read row ${kv} (${reads.join(', ') || 'no paths'}) and add. ` +
      (writes.length ? `Smaller sums are written: ${writes.join('; ')}.` : 'No sum is smaller than the current entry, so nothing changes.'),
    stale,
  };
}

function FloydStepper() {
  const [k, setK] = useState(0);
  const [m, setM] = useState(0);
  const [neg, setNeg] = useState(false);
  const diag = neg ? -1 : undefined;
  const base = useMemo(() => runSteps(FIG_16_5, k), [k]);
  // The book's kernel: one table, updated in place, blocks run top row first. The two-buffer
  // run is computed only to compare against.
  const trace = useMemo(() => stepTrace(base, k, { mode: 'in-place', diag }), [base, k, diag]);
  const two = useMemo(() => stepTrace(base, k, { mode: 'two-buffers', diag }), [base, k, diag]);
  useEffect(() => setM(0), [k, neg]);

  const table = stateAfter(trace, m);
  const cur = m > 0 ? trace.blocks[m - 1] : undefined;
  const done = trace.blocks.slice(0, m);
  const retiredSoFar = new Set(done.filter((b) => b.retired).map((b) => b.row));
  const changedSoFar = table.flatMap((row, i) => row.flatMap((x, j) => (x !== trace.before[i][j] ? [[i, j]] : [])));
  const complete = m >= N;
  const desc = cur ? describeBlock(cur, k, trace.before) : null;
  const hl = cur ? [cur.retired ? FW_LINES.retire : cur.writes.length ? FW_LINES.write : FW_LINES.compare] : [];
  const kv = VERTICES[k];
  const kStill = !trace.rowKChanged && !trace.colKChanged;
  const same = JSON.stringify(trace.after) === JSON.stringify(two.after);

  return (
    <Bench
      title="One Floyd-Warshall step"
      task="Step through the blocks of one k, then set dist[k][k] to −1 and watch the blocks that run after block k."
      foot={
        <>
          Fig. 16.5's table as printed; a blank cell is FW_INF, no path. One table, updated in place, blocks run top row
          first. Row k and column k are tinted: every block reads them. A filled cell changed during this step. <TierLegend tiers={['reg', 'shared', 'global']} />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted">step k =</span>
        <span className="seg" role="group" aria-label="Step k">
          {VERTICES.map((w, i) => (
            <button key={w} aria-pressed={k === i} onClick={() => setK(i)}>{w}</button>
          ))}
        </span>
        <label className="check">
          <input type="checkbox" checked={neg} onChange={(e) => setNeg(e.target.checked)} data-neg /> set dist[k][k] to −1 for this step
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setM(0)} disabled={m === 0}>Restart the step</button>
        <button onClick={() => setM(m - 1)} disabled={m === 0}>Back</button>
        <button className="primary" onClick={() => setM(m + 1)} disabled={complete}>Next block</button>
        <button onClick={() => setM(N)} disabled={complete}>Run the step</button>
        <button onClick={() => setK(Math.min(N - 1, k + 1))} disabled={k === N - 1}>Next k</button>
        <span className="muted" aria-live="polite" data-progress>
          blocks run: {m} of {N}; retired: {retiredSoFar.size}; cells changed: {changedSoFar.length}
        </span>
      </div>

      <div className={'narrate' + (desc?.stale ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }} data-narrate>
        {desc ? (
          <>
            {desc.text}
            {desc.stale ? (
              <>
                <br />
                <b className="t-fail">
                  Block {kv} has already changed row {kv} or column {kv}, and this block read the new values. With two
                  buffers it would read the old ones: the result depends on who ran first.
                </b>
              </>
            ) : null}
          </>
        ) : (
          <>
            Step k = {kv}: every block will read dist[row][{kv}] (column {kv}) once, and its threads will read row {kv}.
            {neg ? ` dist[${kv}][${kv}] is now −1, which breaks the book's assumption.` : ''} Press Next block.
          </>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 4fr) minmax(0, 6fr)' }}>
        <div>
          <div className="mx-label t-global">dist (global memory), during step k = {kv}</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${N + 1}, auto)` }} data-table>
            <span style={HEAD} />
            {VERTICES.map((w) => <span key={w} style={HEAD}>{w}</span>)}
            {table.map((row, i) => {
              const ran = done.some((b) => b.row === i);
              const retired = retiredSoFar.has(i);
              return [
                <span
                  key={`h${i}`}
                  data-row={VERTICES[i]}
                  data-retired={retired ? 1 : 0}
                  style={{ ...HEAD, fontWeight: cur?.row === i ? 800 : 650, textDecoration: cur?.row === i ? 'underline' : undefined }}
                  title={retired ? 'block retired' : ran ? 'block ran' : ''}
                >
                  {VERTICES[i]}
                </span>,
                ...row.map((x, j) => {
                  const changed = x !== trace.before[i][j];
                  const kLine = i === k || j === k;
                  const cls = changed ? (kLine ? 'bad' : 'hot') : kLine ? 'on' : '';
                  return (
                    <span
                      key={`${i},${j}`}
                      className={cls}
                      data-cell={`${i},${j}`}
                      data-v={x === INF ? '' : x}
                      data-changed={changed ? 1 : 0}
                      style={retired ? { opacity: 0.35 } : undefined}
                    >
                      {x === INF ? '' : x}
                    </span>
                  );
                }),
              ];
            })}
          </div>
          <p className="muted" style={{ margin: '0.5rem 0 0' }}>
            {retiredSoFar.size
              ? `Faded rows: the block retired, because dist[row][${kv}] is FW_INF (${[...retiredSoFar].sort((a, b) => a - b).map((i) => VERTICES[i]).join(', ')}).`
              : 'No block has retired yet.'}
            {cur && !cur.retired ? (
              <> Shared <span className="t-shared">dist_k_col</span> of the last block: {cur.distKCol}.</>
            ) : null}
          </p>
          {complete ? (
            <p style={{ margin: '0.7rem 0 0' }} data-checks>
              <span className={'chip ' + (kStill ? 'ok' : 'fail')} data-check="row-col">
                {kStill ? `row ${kv} and column ${kv} unchanged during this step` : `row ${kv} or column ${kv} changed during this step`}
              </span>{' '}
              <span className={'chip ' + (same ? 'ok' : 'fail')} data-check="same">
                {same ? 'in place equals two buffers' : 'in place differs from two buffers'}
              </span>
            </p>
          ) : (
            <p className="muted" style={{ margin: '0.7rem 0 0' }}>Run the whole step to see the two checks.</p>
          )}
        </div>
        <div>
          <Code tabs={[{ label: 'FW_bottomup, corrected', code: FW_KERNEL, tiers: FW_TIERS }]} highlight={hl} />
        </div>
      </div>
    </Bench>
  );
}

// ================================================================ who waits for whom: the explorer
const RULES: { id: Rule; label: string }[] = [
  { id: 'front', label: 'relaunch per front' },
  { id: 'counter', label: 'flags, tiles from a counter' },
  { id: 'rows', label: 'one persistent block per tile row' },
];

function WaitExplorer() {
  const [T, setT] = useState(4);
  const [S, setS] = useState(2);
  const [rule, setRule] = useState<Rule>('front');
  const [start, setStart] = useState<StartOrder>('index');
  const [t, setTime] = useState(0);
  const sch = useMemo(() => schedule(T, S, rule, start), [T, S, rule, start]);
  const counter = useMemo(() => schedule(T, S, 'counter'), [T, S]);
  const front = frontMakespan(T, S);
  const len = sch.stuck ? sch.timeline.length + 3 : sch.timeline.length;
  useEffect(() => setTime(0), [T, S, rule, start]);
  const num = numbering(T);
  const flags = flagsAfter(sch, t);
  const last = t > 0 ? Array.from({ length: S }, (_, s) => slotAt(sch, t - 1, s)) : [];
  const running = new Set(last.filter((c) => c?.state === 'run').map((c) => c!.tile));
  const waiting = new Set(last.filter((c) => c?.state === 'spin').map((c) => c!.tile));
  let idleSoFar = 0;
  for (let u = 0; u < t; u++) for (let s = 0; s < S; s++) if (slotAt(sch, u, s)?.state !== 'run') idleSoFar++;
  const stuckShown = sch.stuck && t > sch.stuckAt!;

  const G = 36;
  const gsz = T * G + 4;
  const cw = Math.max(12, Math.min(34, Math.floor(600 / len)));
  const rh = 26;
  const chartW = 52 + len * cw + 4;
  const chartH = 22 + S * rh + 4;

  return (
    <Bench
      title="Who waits for whom"
      task="Choose a rule, then step through time. Where do the SMs sit idle?"
      foot={
        <>
          Each tile needs the tile above and the tile to its left, and takes one step. A tile with a filled square has set
          its flag. In the chart, a dashed red slot is an SM doing no work: empty, or holding a tile that waits.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Rule" style={WRAP}>
          {RULES.map((r) => (
            <button key={r.id} aria-pressed={rule === r.id} onClick={() => setRule(r.id)}>{r.label}</button>
          ))}
        </span>
        {rule === 'rows' ? (
          <label className="field">
            Blocks start
            <select value={start} onChange={(e) => setStart(e.target.value as StartOrder)} aria-label="Blocks start">
              <option value="index">in index order</option>
              <option value="reversed">in reverse order</option>
            </select>
          </label>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Tiles per side, T
          <select value={T} onChange={(e) => setT(+e.target.value)} aria-label="Tiles per side">
            {[2, 3, 4, 5, 6, 7, 8].map((v) => <option key={v} value={v}>{v}{v === 4 ? " (the book's example)" : ''}</option>)}
          </select>
        </label>
        <label className="field">
          SMs, S
          <select value={S} onChange={(e) => setS(+e.target.value)} aria-label="SMs">
            {[1, 2, 3, 4, 5, 6, 7, 8].map((v) => <option key={v} value={v}>{v}{v === 2 ? " (the book's example)" : ''}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setTime(0)} disabled={t === 0}>Restart</button>
        <button onClick={() => setTime(t - 1)} disabled={t === 0}>Back</button>
        <button className="primary" onClick={() => setTime(t + 1)} disabled={t >= len}>Step</button>
        <button onClick={() => setTime(len)} disabled={t >= len}>Run to the end</button>
        <span className="muted" aria-live="polite" data-time={t}>after step {t}{sch.stuck ? '' : ` of ${len}`}</span>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 3fr)' }}>
        <div>
          <div className="mx-label">tiles in front order</div>
          <svg viewBox={`0 0 ${gsz} ${gsz}`} role="img" aria-label="Tiles and their flags" style={{ width: '100%', maxWidth: Math.min(220, gsz * 1.3), display: 'block' }}>
            {num.flatMap((row, r) =>
              row.map((n, c) => {
                const isDone = flags.has(n);
                const run = running.has(n);
                const wait = waiting.has(n);
                return (
                  <g key={n} data-tile={n} data-done={isDone ? 1 : 0} data-now={run ? 'run' : wait ? 'wait' : ''}>
                    <rect x={2 + c * G} y={2 + r * G} width={G - 3} height={G - 3} rx="2"
                      fill={isDone ? 'var(--global-wash)' : 'var(--cell)'}
                      stroke={wait ? 'var(--fail)' : run ? 'var(--ink)' : 'var(--rule)'}
                      strokeWidth={run || wait ? 2.4 : 1}
                      strokeDasharray={wait ? '4 3' : undefined} />
                    <text x={2 + c * G + G / 2 - 2} y={2 + r * G + G / 2 + 3} fontSize="12" textAnchor="middle">{n}</text>
                    <rect x={2 + c * G + G - 11} y={2 + r * G + G - 11} width="6" height="6"
                      fill={isDone ? 'var(--global)' : 'none'} stroke={isDone ? 'var(--global)' : 'var(--rule)'} />
                  </g>
                );
              }),
            )}
          </svg>
        </div>
        <div>
          <div className="mx-label">what each SM does, step by step</div>
          <div style={{ overflowX: 'auto' }}>
          <svg viewBox={`0 0 ${chartW} ${chartH}`} role="img" aria-label="SMs by time" style={{ width: '100%', minWidth: Math.min(chartW, 900), maxWidth: Math.max(chartW * 1.6, 320), display: 'block' }} data-chart>
            {Array.from({ length: len }, (_, u) => (
              <text key={u} x={52 + u * cw + cw / 2} y={13} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
                {cw >= 16 || u % 2 === 0 ? u + 1 : ''}
              </text>
            ))}
            {Array.from({ length: S }, (_, s) => (
              <g key={s}>
                <text x={0} y={22 + s * rh + rh / 2 + 4} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>SM {s}</text>
                <rect x={52} y={22 + s * rh} width={len * cw} height={rh} fill="none" stroke="var(--rule-soft)" />
                {Array.from({ length: Math.min(t, len) }, (_, u) => {
                  const cell = slotAt(sch, u, s)!;
                  const x = 52 + u * cw + 1;
                  const y = 22 + s * rh + 2;
                  return (
                    <g key={u} data-slot={`${u},${s}`} data-state={cell.state} data-tile={cell.tile ?? ''}>
                      <rect x={x} y={y} width={cw - 2} height={rh - 4} rx="2"
                        fill={cell.state === 'run' ? 'var(--ink)' : cell.state === 'spin' ? 'var(--fail-wash)' : 'none'}
                        fillOpacity={cell.state === 'run' ? 0.12 : 1}
                        stroke={cell.state === 'run' ? 'var(--ink-2)' : 'var(--fail)'}
                        strokeDasharray={cell.state === 'run' ? undefined : '3 2'} />
                      {cw >= 18 && cell.tile !== null ? (
                        <text x={x + (cw - 2) / 2} y={y + rh / 2 + 2} fontSize="10" textAnchor="middle"
                          style={{ fill: cell.state === 'run' ? 'var(--ink)' : 'var(--fail)' }}>{cell.tile}</text>
                      ) : null}
                    </g>
                  );
                })}
              </g>
            ))}
            {rule === 'front'
              ? sch.barriers.filter((b) => b < t - 1).map((b) => (
                  <line key={b} data-barrier={b} x1={52 + (b + 1) * cw} y1={16} x2={52 + (b + 1) * cw} y2={chartH - 2} stroke="var(--ink-2)" strokeDasharray="2 3" />
                ))
              : null}
          </svg>
          </div>
          <table className="data" style={{ marginTop: '0.8rem' }}>
            <tbody>
              <tr><td>relaunch per front</td><td data-k="front">{front} steps</td></tr>
              <tr><td>flags, tiles from a counter</td><td data-k="counter">{counter.makespan} steps</td></tr>
              <tr>
                <td>this rule</td>
                <td data-k="makespan">{sch.stuck ? 'never finishes' : `${sch.makespan} steps`}</td>
              </tr>
              <tr>
                <td>SM-steps with no work, so far</td>
                <td data-k="idle">{idleSoFar}{sch.stuck ? '' : ` of ${sch.idle}`}</td>
              </tr>
            </tbody>
          </table>
          {stuckShown ? (
            <p className="t-fail" style={{ margin: '0.6rem 0 0' }} data-stuck>
              <b>Stuck.</b> Every SM holds a block that waits, directly or through another waiting block, for a row
              that has no SM, and no block gives its SM up.
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ================================================================ the page
export function S163() {
  return (
    <>
      <p>
        Two decisions shape every wavefront kernel: which cells go together in one step, and how one step waits for the
        one before. This page takes them in turn, with the book's first case study, Floyd-Warshall, in between. Page{' '}
        <a href={href('ch16', '16.6')}>16.6</a> applies both to tiles.
      </p>

      <h2>Top down or bottom up</h2>
      <p>
        Dynamic programming builds an answer from the answers to smaller sub-problems that overlap, and keeps each one so
        that it is computed once. There are two ways to organise that. Top down is the recursion you would write first:
        to get F(n), ask for F(n − 1) and F(n − 2), but look each one up first in a hash table of answers found so far
        (memoisation). Bottom up turns it round: fill a table from the smallest sub-problem upward, so that every
        sub-problem finds its inputs already in the table (tabulation).
      </p>
      <Source section="16.2" pages="374–376" figs="Fig. 16.1" />
      <p>
        The book gives two reasons against top down on a GPU. Kernels do not handle deep chains of function calls
        efficiently: device functions are normally inlined, and a recursive call cannot be inlined without end. And the
        threads of a warp reach into the hash table at scattered places, which do not coalesce.
      </p>
      <Source section="16.2" pages="376" />
      <p>
        The first reason is about cost, not a rule: every backend allows a recursive device function, and what it costs
        is stack memory reserved for every thread or warp. A top-down solver recurses as deep as its longest chain of
        sub-problems, so the stack each thread or warp needs grows with the problem, and AMD's guide points out that the
        GPU's parallelism multiplies it.
      </p>
      <Today checked="2026-11-20 (ch16-facts A1, B1, C1)">
        <p>
          CUDA: <code>__device__</code> functions may recurse, kernels may not (<code>cudaLimitStackSize</code>). HIP: a
          stack per warp (<code>hipLimitStackSize</code>). Metal: from 2.4; Apple advises a small <code>maxCallStackDepth</code>.
        </p>
      </Today>
      <p>
        The second reason is the stronger one, and it is not about recursion. A memo table is a hash table: threads with
        neighbouring sub-problems land at unrelated addresses, so a warp's loads touch many memory segments instead of a
        few (chapter 6). And which sub-problem runs when is decided by the recursion as it goes. Bottom up gives every
        thread a computed address in a plain array, and a schedule fixed before the kernel starts. The rest of the
        chapter is bottom up.
      </p>
      <p>
        It also gives the working definition of a wavefront: the cells filled in one iteration of the bottom-up loop. For
        Fibonacci each iteration fills one cell, so there is nothing to run in parallel. The tables worth a GPU fill many
        cells per iteration, and the next part is about which ones.
      </p>
      <Source section="16.2" pages="376" />
      <Ask q={questionsS163['ch16/16.3/q-topdown']} />

      <h2>Number each cell by when it can run</h2>
      <p>
        Everything in this chapter rests on one rule. A cell can run once every cell it reads is done. So give each cell a
        step number: one more than the latest step among the cells it reads, or 0 if it reads only boundary values. Cells
        with the same number form a wavefront. They read nothing from each other and can all run at once, and between one
        step and the next there must be a barrier of some kind, because step s reads what step s − 1 wrote.
      </p>
      <Source section="16.3" pages="376–377" figs="Fig. 16.2" />
      <p>
        The book draws four patterns. In two of them, checkerboard and financial, every cell reads only the row above, so
        a step is a whole row and every front has the same size. Floyd-Warshall, below, has constant fronts too: one whole
        table per step. In Smith-Waterman and H.264 the fronts grow and then shrink, which leaves threads idle near the
        corners of the table.
      </p>
      <Source section="16.3" pages="377" />
      <Predict
        id="ch16/16.3/p-sw8"
        prompt={
          <>
            An 8 × 8 table in which each cell reads the cell above, the cell to its left and the cell above-left (the
            Smith-Waterman pattern). How many barrier-separated steps does it take?
          </>
        }
        unit="steps"
      >
        {(v) => (
          <p>
            {v === SW8.steps ? 'Right: ' : `It takes ${SW8.steps}: `}the step of cell (r, c) is r + c, from 0 to{' '}
            {SW8.steps - 1}, and in general an R × C table takes R + C − 1 steps. The widest step has {SW8.widest} cells and
            the average is {SW8.mean.toFixed(1)}, so a kernel that gives every step {SW8.widest} threads keeps them a little
            over half busy. Check it in the explorer.
          </p>
        )}
      </Predict>
      <p>
        The explorer numbers any table by this rule. Choose a pattern or tick your own set of inputs, change the size, and
        click a cell to see which cells it reads.
      </p>
      <Simplification>
        <p>
          Every cell costs one unit, there are as many threads as cells in a step, and memory is free: the explorer counts
          steps, not time. The financial pattern is our reading of Fig. 16.2(b), which is small in print.
        </p>
      </Simplification>
      <PatternExplorer />
      <p>
        Three things to take from it. The stencil alone decides how many steps a table needs and how wide they are; the
        values in the cells play no part. A stencil that reads only the row above gives constant fronts. And an input
        above and to the right, as in H.264, makes each row start two steps after the row above instead of one, so a
        square table takes about three times its side in steps instead of two.
      </p>
      <Ask q={questionsS163['ch16/16.3/q-h264']} label="Work it out" />
      <p>
        The rule also tells you how much of a table must be kept. A step reads at most a fixed number of fronts back, and
        anything older can go. The book notes that Fibonacci needs only its last two values, that Floyd-Warshall keeps one
        plane of its three-dimensional space, and that Smith-Waterman keeps its whole table only for the traceback.
      </p>
      <Source section="16.3" pages="377" />
      <Ask q={questionsS163['ch16/16.3/q-live']} label="Work it out" />

      <h2>Floyd-Warshall: one launch per step, one table</h2>
      <p>
        Floyd-Warshall finds the shortest path between every pair of vertices of a directed graph with weighted edges. Let
        d(i, j, k) be the shortest distance from i to j that stops on the way only at vertices 0 to k. Either the best such
        path avoids k, or it goes to k and on from k: d(i, j, k) = min(d(i, j, k − 1), d(i, k, k − 1) + d(k, j, k − 1)).
        Start from the edge weights, with "no path" where there is no edge, and apply the recurrence for k = 0, 1, …, N −
        1. The sub-problems fill an N × N × N space, but each plane needs only the one before it, so only N × N values
        have to be kept at a time.
      </p>
      <Source section="16.4" pages="377–378" figs="Fig. 16.3" />
      <p>
        Look only at the dependencies. Step k reads nothing but row k and column k of step k − 1, so the N² cells of one
        step need nothing from each other: every cell is in every front, N steps of N² cells, a constant front as wide as
        the table. The only
        synchronisation needed is between steps, and the book gets it in the simplest way: one thread per cell, one
        kernel launch per k. Launches in one stream run one after another, so ending the kernel is the barrier.
      </p>
      <Source section="16.4" pages="378–379" />
      <Code
        tabs={[
          { label: 'kernel', code: FW_KERNEL, tiers: FW_TIERS, status: 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware' },
          { label: 'host loop', code: FW_HOST, tiers: FW_TIERS, status: 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware' },
        ]}
        caption={
          <>
            The book's Fig. 16.4, corrected, from <code>labs/ch16/snippets/fw_bottomup.cu</code>; <code>check.py</code>{' '}
            there runs it against a CPU Floyd-Warshall on the emulator. It differs from the page in three places,
            described below.
          </>
        }
      />
      <p>
        Each block covers part of one row of the table, so all its threads need the same value, dist[row][k]. Thread 0
        reads it once into a shared variable, and after the barrier every thread uses it. If it is the sentinel, there is
        no path from this row's vertex to k, nothing in the row can improve, and the whole block returns. Otherwise each
        thread reads its own entry of row k, and these reads are coalesced across the block.
      </p>
      <Source section="16.4" pages="381" />
      <p>Where the listing differs from the page:</p>
      <ul>
        <li>
          <b>The sentinel.</b> The page compares an <code>int</code> with <code>INFINITY</code>, which is a{' '}
          <code>float</code>; no <code>int</code> ever equals it. The listing uses the integer{' '}
          <code>FW_INF = INT_MAX / 2</code>, so that one addition of two sentinels still fits in an <code>int</code>.
        </li>
        <li>
          <b>No return before the barrier.</b> The page returns out-of-range threads at the top, before{' '}
          <code>__syncthreads()</code>. The listing keeps them to the barrier with an <code>inRange</code> flag. The{' '}
          <code>return</code> after the barrier depends only on the shared value, so it is the same for the whole block.
        </li>
        <li>
          <b>No host wait between launches.</b> The page calls <code>cudaDeviceSynchronize()</code> after every launch.
          The order of launches in a stream already gives each step the previous one's writes.
        </li>
      </ul>
      <Today checked="2026-11-20 (ch02-facts A5, B3, C9; ch04-facts A6, B3, C3)">
        <p>
          One stream runs launches in order on CUDA and HIP. On Metal, dispatches in one compute encoder created the
          plain way run in order and see each other's writes.
        </p>
        <p>
          A thread that has returned does not hold <code>__syncthreads()</code> on CUDA. HIP does not document it, and
          Metal documents it for Apple silicon only.
        </p>
      </Today>
      <Ask q={questionsS163['ch16/16.3/q-infinity']} label="The page's listing" />

      <h3>Why one table is enough</h3>
      <p>
        The kernel writes into the table while other threads read row k and column k from it. That is safe only if row k
        and column k do not change during step k, and under two conditions they cannot: dist[k][k] is 0, and no weight is
        negative. The thread that owns dist[i][k] computes min(dist[i][k], dist[i][k] + dist[k][k]), which is dist[i][k]
        again, and the same holds along row k. So no thread reads a value that another is changing, and no second buffer
        is needed. Where a step does change values that others read, you need two buffers: read the old table, write a new
        one, swap (double buffering, chapter 6).
      </p>
      <Source section="16.4" pages="379–380" />
      <p>
        The stepper runs one step of the corrected kernel on the book's eight-vertex example, one block per row. Step
        through k = a first and count the blocks that retire. Then run a and b, and find the cells that step c changes.
        Last, tick dist[k][k] = −1 and step through c again, watching the blocks that run after block c.
      </p>
      <Simplification>
        <p>
          Blocks run one after another, top row first, each to completion; on a GPU they interleave in ways you do not
          choose. That is one of the orders a GPU may produce, enough to show a result that depends on the order. The table is Fig. 16.5's as printed,
          where b to f is 9; the graph beside it says 6.
        </p>
      </Simplification>
      <FloydStepper />
      <p>
        At k = a, {FW_A.retired.length} of the 8 blocks retire at once: only a has a path to a. At k = c, after a and b,{' '}
        {FW_C.retired.length} retire. With dist[k][k] = −1, row k and column k change during their own step, every block
        that runs after block k reads the new values, and the table differs from the one two buffers give. What a block
        reads depends on whether block k ran before it: a race, which two buffers would remove.
      </p>
      <Ask q={questionsS163['ch16/16.3/q-fw-changes']} label="Work it out" />
      <Ask q={questionsS163['ch16/16.3/q-fw-inplace']} />
      <Ask q={questionsS163['ch16/16.3/q-ex1']} label="Book exercise 1" />

      <h2>Who waits for whom</h2>
      <p>
        Floyd-Warshall needs nothing finer than a relaunch: every block of step k + 1 reads row k + 1 and column k + 1,
        which blocks all over step k wrote. A table whose cells depend only on near neighbours deserves better. Ask of every design: what does one unit of work
        wait for, and is that more than it needs? From the coarsest wait to the finest:
      </p>
      <Source section="16.8" pages="397–399" figs="Fig. 16.21" />
      <ol>
        <li>
          <b>The whole grid waits for the whole previous front.</b> A relaunch per front does this; the book's tiled
          Smith-Waterman kernels relaunch once per anti-diagonal of tiles. A grid-wide barrier,{' '}
          <code>grid.sync()</code> under a cooperative launch, saves the launches but not the wait, and needs every block
          resident at once (the API is in chapter 18, §18.7, pp. 445–448).
        </li>
        <li>
          <b>A unit of work waits only for its neighbours.</b> One flag per tile, set once the tile is stored. A tile
          starts when the flags of the tile above and the tile to its left are set. The book likens it to the scan of
          chapter 11.
        </li>
        <li>
          <b>Nothing waits across blocks</b>, because one block, or one warp, owns the whole chain. For short sequences one
          block or even one warp per problem is enough, with barriers or shuffles inside (chapter 10, §10.7).
        </li>
      </ol>
      <Source section="16.6, 16.8" pages="384, 398" />
      <p>
        The animation and the explorer after it run on one model of tiles and SMs, which counts steps and nothing else.
      </p>
      <Simplification>
        <p>
          Every tile takes one step, launches and barriers cost nothing, and a waiting block costs its SM and nothing else.
          S is how many tiles run at once, not a device's SM count. The stuck outcome is one schedule that the missing
          ordering guarantee allows, not a model of any GPU's scheduler.
        </p>
      </Simplification>
      <Predict
        id="ch16/16.3/p-flags-2sm"
        prompt={
          <>
            A 4 × 4 grid of tiles on a GPU with 2 SMs; a relaunch per front takes 10 time steps. If instead each tile
            starts as soon as the tile above it and the tile to its left are done, how many time steps does it take?
          </>
        }
        unit="steps"
      >
        {(v) => (
          <p>
            {v === C2 ? 'Right: ' : `It takes ${C2}: `}one step fewer than the {F2} of a relaunch per front, as in the
            book's Fig. 16.21. The animation shows where the step comes from, and the explorer after it lets you change the
            grid and the number of SMs.
          </p>
        )}
      </Predict>
      <NeighboursScene />
      <Ask q={questionsS163['ch16/16.3/q-enough-sms']} label="Work it out" />
      <p>
        In the explorer, a relaunch per front leaves SMs idle wherever a front has fewer tiles than there are SMs, or a
        number that does not divide evenly. With flags, a free SM takes the next tile in front order and waits only for
        that tile's two neighbours, so a later front can start in the gap. Try one SM, then as many SMs as tiles per side.
      </p>
      <WaitExplorer />
      <p>
        The book's own flag design gives each persistent block a whole row of tiles, by its <code>blockIdx.x</code>, so a
        block checks only the flag of the tile above. Set the explorer to one block per row, start the blocks in reverse,
        and give it fewer SMs than rows.
      </p>
      <Source section="16.8" pages="398" />
      <Ask q={questionsS163['ch16/16.3/q-row-owner']} />
      <p>
        What makes waiting safe is the same as in chapter 11: a block may wait only for work that has already started. A
        tile counter in front order gives exactly that, because every tile a block waits for was claimed earlier, by a
        block that is already running.
      </p>
      <Today checked="2026-11-20 (ch16-facts A4, B3, C4; infra-facts A4, B4)">
        <p>
          CUDA: flags are documented-safe with device-scope release and acquire (<code>cuda::atomic_ref</code>) and tiles
          claimed from an atomic counter in front order, or under a cooperative launch with the whole grid resident.
        </p>
        <p>
          HIP: the same source compiles through libhipcxx, and <code>hipLaunchCooperativeKernel</code> exists, but AMD
          documents no progress guarantee between blocks: unverified.
        </p>
        <p>
          Metal: no grid barrier, no cooperative launch, no progress guarantee between threadgroups. One dispatch per
          front is the only documented-safe design across threadgroups.
        </p>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>What waits</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>the grid, on a relaunch per front</td>
            <td>one launch per front, one stream</td>
            <td>same</td>
            <td>one dispatch per front in one compute encoder</td>
          </tr>
          <tr>
            <td>the grid, on a grid barrier</td>
            <td><code>grid.sync()</code> under <code>cudaLaunchCooperativeKernel</code>, grid resident</td>
            <td><code>hipLaunchCooperativeKernel</code>, all six arguments</td>
            <td>none</td>
          </tr>
          <tr>
            <td>a tile, on its neighbours' flags</td>
            <td>release and acquire at device scope, tiles from an atomic counter</td>
            <td>same source; progress not documented, unverified</td>
            <td>unsafe: no progress guarantee between threadgroups</td>
          </tr>
          <tr>
            <td>a step, inside one block</td>
            <td><code>__syncthreads()</code></td>
            <td>same</td>
            <td><code>threadgroup_barrier</code>, with <code>mem_device</code> if the chain is in device memory</td>
          </tr>
          <tr>
            <td>a step, inside one warp</td>
            <td><code>__shfl_*_sync</code> with a 32-bit lane mask</td>
            <td>same names, 64-bit lane mask</td>
            <td><code>simd_shuffle_*</code>, no mask; read the width from <code>[[threads_per_simdgroup]]</code></td>
          </tr>
        </tbody>
      </table>
      <p className="muted" style={{ fontFamily: 'var(--sans)', fontSize: '0.8rem' }}>
        From ch16-facts, "Where the backends really differ", and infra-facts A3, B3, C2, C5. Never write the warp width as
        a literal 32: it is 32 on NVIDIA, 64 or 32 on AMD, and read from the pipeline on Metal.
      </p>

      <h2>Where this sits in PyTorch</h2>
      <p>
        PyTorch's CTC loss, <code>torch.nn.functional.ctc_loss</code>, fills a table alpha(t, s) in which each entry needs
        alpha(t − 1, s), alpha(t − 1, s − 1) and alpha(t − 1, s − 2). That is a constant front per time step, the
        stencil of the explorer's "Financial; also CTC" pattern. Its CUDA kernel gives each block whole sequences and loops over t with a{' '}
        <code>__syncthreads()</code> at every step, so blocks never wait for each other. When the target is wider than a
        block, the block walks over chunks of positions in dependency order. The MPS kernel does the same with one
        threadgroup per batch item, and its barrier is{' '}
        <code>threadgroup_barrier(mem_flags::mem_device)</code>, because the table lives in device memory and a barrier
        on threadgroup memory alone would not order it.
      </p>
      <Beyond>
        <p>
          The book does not discuss CTC. The kernels were read on 2026-11-20 (ch16-facts D2, C3: <code>LossCTC.cu</code>,{' '}
          <code>LossOps.metal</code>); the mapping to this chapter is ours.
        </p>
      </Beyond>
      <Ask q={questionsS163['ch16/16.3/q-ctc']} label="In PyTorch" />
      <p>
        The <a href={href('ch16', 'labs')}>lab</a> puts the second idea of this chapter, full-width steps, on your own GPU,
        and page <a href={href('ch16', '16.6')}>16.6</a> explains it first.
      </p>
    </>
  );
}
