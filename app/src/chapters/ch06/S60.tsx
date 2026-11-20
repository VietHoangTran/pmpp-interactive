import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Source } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// Prerequisite questions drawn from the notes of chapters 3 to 5. Each explanation
// names where the idea was taught and which section of this chapter builds on it.
export const questionsS60: Record<string, Question> = {
  'ch06/start/q-neighbours': {
    kind: 'numeric',
    id: 'ch06/start/q-neighbours',
    prompt: (
      <>
        In the naive matmul loop <code>Pvalue += M[row*Width + k] * N[k*Width + col];</code> the threads{' '}
        <code>(tx, ty)</code> and <code>(tx + 1, ty)</code> of one block execute the same iteration <code>k</code>{' '}
        together. How many array elements apart are the two addresses they read from <code>N</code>?
      </>
    ),
    answer: 1,
    unit: 'elements',
    explain: (
      <p>
        <code>col</code> differs by one and everything else in <code>k*Width + col</code> is the same, so the two reads
        are neighbours. Their <code>M</code> reads are identical: same <code>row</code>, same <code>k</code>. So across a
        row of threads one operand is a run of consecutive addresses and the other is a single shared value. Taught
        in <a href={href('ch03', '3.4')}>ch. 3 §3.4</a>; section <a href={href('ch06', '6.1')}>6.1</a> turns it into
        a cost you can count.
      </p>
    ),
    source: 'ch. 3 §3.4, pp. 61–62',
  },
  'ch06/start/q-warp2d': {
    kind: 'choice',
    id: 'ch06/start/q-warp2d',
    prompt: <>A kernel runs 16 × 16 blocks on hardware with 32-wide warps. Which threads of a block form warp 0?</>,
    options: [
      { label: 'threadIdx.x 0 to 15 for threadIdx.y 0 and 1: two rows of the block.', correct: true },
      { label: 'threadIdx.x 0 to 31 of row 0.', why: 'The block is only 16 wide; there is no threadIdx.x above 15.' },
      { label: 'Every second thread of the block.', why: 'Warps are consecutive threads in the linearised order, not interleaved.' },
      { label: 'The hardware picks them at run time.', why: 'The partition is fixed: linearise row-major (x fastest), then cut into 32s.' },
    ],
    explain: (
      <p>
        The block is linearised x fastest, then cut into groups of 32, so a warp is two 16-wide rows. Taught in{' '}
        <a href={href('ch04', '4.4')}>ch. 4 §4.4</a>. It matters twice in this chapter: a 16-wide tile row is half a
        warp, so one warp&rsquo;s load touches two separate runs of addresses (
        <a href={href('ch06', '6.1')}>6.1</a>), and its shared-memory accesses spread over banks differently from a
        32-wide row&rsquo;s (<a href={href('ch06', '6.4')}>6.4</a>). The 32 is the book&rsquo;s width; AMD and Apple
        GPUs can differ.
      </p>
    ),
    source: 'ch. 4 §4.4, pp. 75–76',
  },
  'ch06/start/q-hide': {
    kind: 'numeric',
    id: 'ch06/start/q-hide',
    prompt: (
      <>
        Toy numbers. An SM has one resident warp. It issues 4 instructions, then waits 396 cycles for a load, then
        repeats. For what percentage of cycles does the execution unit issue an instruction?
      </>
    ),
    answer: 1,
    tol: 0.01,
    unit: '%',
    explain: (
      <p>
        4 busy cycles in every 400: 1%. With more resident warps the scheduler fills the wait with other warps&rsquo;
        instructions, which is latency hiding (<a href={href('ch04', '4.6')}>ch. 4 §4.6</a>). Section{' '}
        <a href={href('ch06', '6.2')}>6.2</a> adds a second reason to want those warps: each one keeps its own memory
        requests in flight, and the memory system needs many in flight to deliver its bandwidth.
      </p>
    ),
    source: 'ch. 4 §4.6, pp. 82–84 (the numbers are ours)',
  },
  'ch06/start/q-false': {
    kind: 'choice',
    id: 'ch06/start/q-false',
    prompt: (
      <>
        The tiled matmul kernel has two <code>__syncthreads()</code> per phase. Which one guards a <em>false</em>{' '}
        (write-after-read) dependence?
      </>
    ),
    options: [
      { label: 'The one after the dot-product loop (barrier 2).', correct: true },
      {
        label: 'The one after the tile loads (barrier 1).',
        why: 'That one is a true dependence: the dot product really needs the values just written.',
      },
      { label: 'Both.', why: 'Only one of them protects data that is needed; the other only protects a location from being reused too early.' },
      { label: 'Neither: both are needed for correctness, so both are true dependences.', why: 'Both are needed as written, but for different reasons, and only one reason is about the data.' },
    ],
    explain: (
      <p>
        Barrier 2 stops a fast thread from overwriting a tile slot for the next phase while a slow thread is still
        reading it this phase. Nothing is passed from the reader to the writer: the two only share a location. Taught
        in <a href={href('ch05', '5.4')}>ch. 5 §5.4</a>. Section <a href={href('ch06', '6.7')}>6.7</a> removes that
        kind of barrier by writing to a different location.
      </p>
    ),
    source: 'ch. 5 §5.4, pp. 110–111',
  },
  'ch06/start/q-vecadd': {
    kind: 'choice',
    id: 'ch06/start/q-vecadd',
    prompt: (
      <>
        Vector addition does one operation per 12 bytes it moves, and your version runs at 60% of the bandwidth you
        measured. Which kind of change can still make it faster?
      </>
    ),
    options: [
      {
        label: 'Moving the same bytes with fewer, wider, well-formed requests, and enough of them in flight.',
        correct: true,
      },
      { label: 'Tiling it through shared memory.', why: 'Each element is used once by one thread; there is no reuse for a tile to capture.' },
      { label: 'Doing more arithmetic per element loaded.', why: 'The computation needs one add per element. Extra arithmetic is extra work, not speed.' },
      { label: 'A larger tile.', why: 'There is no tile, and no tile size changes a ratio fixed by the computation.' },
    ],
    explain: (
      <p>
        Its ratio, 1 operation per 12 bytes, is fixed by what it computes (
        <a href={href('ch05', '5.1')}>ch. 5 §5.1</a>), so it is memory-bound on every GPU. What is not fixed is how
        efficiently those bytes are fetched, and that is most of this chapter: one warp&rsquo;s requests (6.1), enough
        of them in flight (6.2), wider ones (6.3).
      </p>
    ),
    source: 'ch. 5 §5.1, p. 96',
  },
};

// ---------------------------------------------------------------- your measurements so far
const GRAN = 'ch03/lab4_granularity';
const TILED = 'ch05/lab2_tiled';
const MODES = ['element', 'row', 'col'] as const;
const MODE_WORDS: Record<(typeof MODES)[number], string> = {
  element: 'one thread per element',
  row: 'one thread per row',
  col: 'one thread per column',
};

/** GPU backends only: realBackends() leaves out 'emu' but not the race-detection emulator. */
const gpuBackends = (r: Reports) => realBackends(r).filter((b) => b !== 'race');

/** The reader's own rows when they have run their work with --bench, else the reference solution's. */
function rowsFor(reports: Reports, backend: string, lab: string): { rows: BenchRow[]; whose: string } {
  const mine = benchRows(reports[backend], lab, 'work');
  return mine.length ? { rows: mine, whose: 'your kernels' } : { rows: benchRows(reports[backend], lab, 'solution'), whose: 'the reference solution' };
}

function Measurements() {
  const { reports, loading, reload } = useReports();
  const backends = gpuBackends(reports);
  const per = backends.map((b) => {
    const g = rowsFor(reports, b, GRAN);
    const gran = MODES.map((m) => g.rows.find((r) => r.name === 'granularity' && r.mode === m));
    const t = rowsFor(reports, b, TILED);
    const tiled = t.rows.find((r) => r.name === 'tiled' && num(r.tile) === 16);
    return { b, device: String(reports[b].device.name ?? b), g, gran, t, tiled };
  });
  const any = per.some((p) => p.gran.some(Boolean) || p.tiled);
  return (
    <Bench
      title="Your measurements so far"
      task="Two numbers from earlier labs that this chapter explains or starts from."
      foot={<>Read from this machine&rsquo;s recorded lab runs (<code>reports/runs/</code>). Nothing here is an expected value.</>}
    >
      {loading ? (
        <p data-measured="loading">Reading reports.</p>
      ) : !any ? (
        <p data-measured="none">
          No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 3.4 --solution --bench</code> and{' '}
          <code>python3 labs/pmpp.py run 5.2 --solution --bench</code> (or your own work without{' '}
          <code>--solution</code>), then reload.{' '}
          <button className="quiet" onClick={reload}>
            Reload results
          </button>
        </p>
      ) : (
        per.map((p) => {
          const [el, row, col] = p.gran;
          const sec = (r?: BenchRow) => num(r?.seconds);
          const rowSlower = sec(row) && sec(col) ? sec(row)! > sec(col)! : null;
          return (
            <div key={p.b} data-measured={p.b} style={{ marginBottom: '0.8rem' }}>
              <p style={{ margin: '0 0 0.4rem' }}>
                <b>{p.device}</b> ({p.b})
              </p>
              {el || row || col ? (
                <>
                  <table className="data" style={{ marginBottom: '0.4rem' }}>
                    <thead>
                      <tr>
                        <th>Lab 3.4, {p.g.whose}</th>
                        <th className="num">Time (ms)</th>
                        <th className="num">Time ÷ one per element</th>
                      </tr>
                    </thead>
                    <tbody>
                      {MODES.map((m, i) => {
                        const r = p.gran[i];
                        return (
                          <tr key={m} data-mode={m}>
                            <td>{MODE_WORDS[m]}</td>
                            <td className="num">{sec(r) ? fmt(sec(r)! * 1000, 3) : '–'}</td>
                            <td className="num">{num(r?.time_vs_element) ? fmt(num(r?.time_vs_element)!, 3) : '–'}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <p style={{ margin: '0 0 0.6rem' }} data-row-vs-col={rowSlower == null ? 'unknown' : rowSlower ? 'row-slower' : 'row-not-slower'}>
                    {rowSlower == null ? (
                      <>Lab 3.4 has no row or column timing here yet.</>
                    ) : rowSlower ? (
                      <>
                        One thread per row was slower than one per column, though both do the same arithmetic. Section{' '}
                        <a href={href('ch06', '6.1')}>6.1</a> explains why: look at which addresses neighbouring threads
                        touch in each kernel.
                      </>
                    ) : (
                      <>
                        Here one thread per row was not slower than one per column. Section{' '}
                        <a href={href('ch06', '6.1')}>6.1</a> says which one the coalescing rule predicts to be slower,
                        and lab 6.1 measures the effect on its own.
                      </>
                    )}
                  </p>
                </>
              ) : (
                <p className="status-line" data-gran="none">
                  Lab 3.4: no measurement on this machine yet (<code>python3 labs/pmpp.py run 3.4 --solution --bench</code>).
                </p>
              )}
              {p.tiled ? (
                <p style={{ margin: 0 }} data-tiled={String(num(p.tiled.speedup_vs_naive) ?? '')}>
                  Lab 5.2, {p.t.whose}: the tiled kernel at tile width 16 ran{' '}
                  <b>{num(p.tiled.speedup_vs_naive) ? `${fmt(num(p.tiled.speedup_vs_naive)!, 3)} times` : 'at an unrecorded speed relative to'}</b>{' '}
                  {num(p.tiled.speedup_vs_naive) ? 'as fast as the naive kernel' : 'the naive kernel'}
                  {num(p.tiled.gflops) ? ` (${fmt(num(p.tiled.gflops)!, 4)} GFLOP/s)` : ''}. That is this chapter&rsquo;s
                  starting point; section <a href={href('ch06', '6.9')}>6.9</a> places it against your roof.
                </p>
              ) : (
                <p className="status-line" style={{ margin: 0 }} data-tiled="none">
                  Lab 5.2: no measurement on this machine yet (<code>python3 labs/pmpp.py run 5.2 --solution --bench</code>).
                </p>
              )}
            </div>
          );
        })
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- the chapter at a glance
const MAP: { part: string; rows: [string, string, string][] }[] = [
  {
    part: 'Global memory',
    rows: [
      ['6.1', 'Global memory access coalescing', 'How one warp’s loads become few or many memory transactions'],
      ['6.2', 'Hiding memory latency', 'Why the memory system needs many requests in flight, and where they come from'],
      ['6.3', 'Vector loads and stores', 'Fewer, wider requests for the same bytes'],
    ],
  },
  { part: 'Shared memory', rows: [['6.4', 'Shared memory bank conflicts', 'When a warp’s shared-memory access is split into passes, and padding']] },
  {
    part: 'Work per thread',
    rows: [
      ['6.5', 'Thread coarsening', 'Several units of work per thread, and when that backfires'],
      ['6.6', 'Loop unrolling', 'Fewer branches, more independent instructions, arrays kept in registers'],
    ],
  },
  { part: 'Synchronisation', rows: [['6.7', 'Double buffering', 'Removing the barrier that only protects a location']] },
  {
    part: 'Method',
    rows: [
      ['6.8', 'A checklist of optimisations', 'Twelve rows, from a symptom to what relieves it and what it costs'],
      ['6.9', 'Optimisation strategy', 'Find the bottleneck, fix it, measure, and know when to stop'],
    ],
  },
];
const LABS: [string, string, string][] = [
  ['6.1', 'Coalescing on your device', '6.1'],
  ['6.2', 'Vector loads, with the boundaries done right', '6.3'],
  ['6.3', 'Provoke a bank conflict, then pad it away', '6.4'],
  ['6.4', 'The linear layer in the framework layout, corner-turned (milestone M2, “tuned”)', '6.1, 6.4'],
  ['6.5', 'Double buffering, with a race detector watching', '6.7'],
];

export function S60() {
  const q = questionsS60;
  return (
    <>
      <p>
        Chapter 5 cut how many bytes a kernel asks global memory for. This chapter is about how it asks for them. The
        same bytes can arrive at full speed or at a fraction of it, depending on which addresses the threads of a warp
        request together, how many requests are in flight at once, and how wide each one is. The chapter starts
        from the memory chips themselves (why they deliver data in bursts, and in parallel channels and banks), turns
        to the banks of shared memory, then adds three tools that change how much work each thread does and how often
        it waits: coarsening, unrolling and double buffering. It ends with a checklist of every optimisation in
        Part 1, and a method for choosing among them.
      </p>
      <Source section="6" pages="123–124" />
      <p>
        The method rests on one idea from the chapter&rsquo;s first page. Tuning a kernel almost always trades one
        resource for another, and the trade pays only if it relieves the resource that is actually limiting the
        kernel, its bottleneck. Section <a href={href('ch06', '6.9')}>6.9</a> closes that loop.
      </p>

      <h2>Under a framework call</h2>
      <p>
        <code>torch.nn.Linear</code> stores its weight as <code>(out_features, in_features)</code> and computes{' '}
        <code>y = x W^T</code>: each output is the dot product of a row of <code>x</code> with a row of{' '}
        <code>W</code>. ggml, under llama.cpp, does the same. Your linear layers in labs 3.3 and 5.3 kept the
        book&rsquo;s layout, with the weight stored <code>(in, out)</code> and read down a column. Lab 6.4 moves the
        project&rsquo;s linear layer to the framework layout, and section 6.1 shows why that layout needs a different
        tile load. It matters most where you would expect: a batch-1 decode step is memory-bound however it is written
        (<a href={href('ch05', '5.1')}>section 5.1</a>), so the memory techniques of this chapter are the ones that move
        decode speed.
      </p>
      <Beyond>
        <p>
          The bridge is ours. The PyTorch and ggml conventions are from <code>notes/today/ch03-facts.md</code> D2 and
          D6 (PyTorch 2.15 documentation and source; llama.cpp&rsquo;s <code>ggml_mul_mat</code>).
        </p>
      </Beyond>

      <h2>Your measurements so far</h2>
      <p>
        Two results from earlier labs belong to this chapter. Lab 3.4 timed three ways of splitting a matrix product
        among threads, all doing the same arithmetic; their difference in time is the subject of 6.1. Lab 5.2 timed
        your tiled kernel against the naive one, which is where this chapter starts.
      </p>
      <Measurements />

      <h2>The chapter at a glance</h2>
      <table className="data" data-map>
        <thead>
          <tr>
            <th>About</th>
            <th>Section</th>
            <th>What you will be able to do</th>
          </tr>
        </thead>
        <tbody>
          {MAP.flatMap((g) =>
            g.rows.map(([id, title, what], i) => (
              <tr key={id}>
                <td>{i === 0 ? g.part : ''}</td>
                <td>
                  <a href={href('ch06', id)}>
                    {id} {title}
                  </a>
                </td>
                <td>{what}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
      <p>Five labs, on the <a href={href('ch06', 'labs')}>labs page</a>, each with a prediction to commit before you measure:</p>
      <table className="data" data-labs>
        <thead>
          <tr>
            <th>Lab</th>
            <th>What you build</th>
            <th>Sections</th>
          </tr>
        </thead>
        <tbody>
          {LABS.map(([n, t, s]) => (
            <tr key={n} data-lab={n}>
              <td>{n}</td>
              <td>{t}</td>
              <td>{s}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>Before you start</h2>
      <p>
        Five questions on what this chapter assumes from chapters 3 to 5. Each explanation points to where the idea was
        taught and to the section here that builds on it. They come back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={q['ch06/start/q-neighbours']} label="From chapter 3" />
      <Ask q={q['ch06/start/q-warp2d']} label="From chapter 4" />
      <Ask q={q['ch06/start/q-hide']} label="From chapter 4" />
      <Ask q={q['ch06/start/q-false']} label="From chapter 5" />
      <Ask q={q['ch06/start/q-vecadd']} label="From chapter 5" />
    </>
  );
}
