import { useId, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { DEFAULTS, FACTORS, WHERE_SNIPPETS, bodyText, schedule, whereLives, type Cell, type Place } from '../../sim/unroll';

// labs/ch06/snippets/s66_unroll.cu, which ./verify.sh front-end-checks.
// snippets-s66.test.ts fails if this copy drifts from the file.
const UNROLL = `// Section 6.6, pp. 144-145: the book's 16-iteration loop, made compilable. A and B
// stand for the book's two statements: B(i) uses what A(i) produced, and A(i + 1)
// does not depend on A(i) or B(i). Each thread works on its own 16 elements.
__device__ float A(const float* in, unsigned int i) { return in[i]*in[i]; }
__device__ void B(float* out, unsigned int i, float a) { out[i] = a + 1.0f; }

__global__ void original(const float* in, float* out) {
    unsigned int base = 16*(blockIdx.x*blockDim.x + threadIdx.x);
    for (unsigned int i = 0; i < 16; ++i) {
        float a = A(in, base + i);
        B(out, base + i, a);
    }
}

// Unrolled by 4 and reordered by hand: four A's, then four B's, four times.
__global__ void unrolledReordered(const float* in, float* out) {
    unsigned int base = 16*(blockIdx.x*blockDim.x + threadIdx.x);
    for (unsigned int i = 0; i < 16; i += 4) {
        float a0 = A(in, base + i);
        float a1 = A(in, base + i + 1);
        float a2 = A(in, base + i + 2);
        float a3 = A(in, base + i + 3);
        B(out, base + i, a0);
        B(out, base + i + 1, a1);
        B(out, base + i + 2, a2);
        B(out, base + i + 3, a3);
    }
}

// The same request made to the compiler. #pragma unroll 1 would forbid unrolling.
__global__ void askedToUnroll(const float* in, float* out) {
    unsigned int base = 16*(blockIdx.x*blockDim.x + threadIdx.x);
#pragma unroll 4
    for (unsigned int i = 0; i < 16; ++i) {
        float a = A(in, base + i);
        B(out, base + i, a);
    }
}`;

// ---------------------------------------------------------------- issue timeline
const PER_ROW = 32;
const CW = 20;

function CellBox({ c, x, y, hatch }: { c: Cell; x: number; y: number; hatch: string }) {
  const w = CW - 2;
  const h = 22;
  let body;
  if (c.kind === 'issueA' || c.kind === 'issueB') {
    const a = c.kind === 'issueA';
    body = (
      <>
        <rect x={x} y={y} width={w} height={h} rx="2" fill={a ? 'var(--ink)' : 'var(--ink-2)'} />
        <text x={x + w / 2} y={y + 15} fontSize="9" textAnchor="middle" style={{ fill: 'var(--page)' }}>{(a ? 'A' : 'B') + c.i}</text>
      </>
    );
  } else if (c.kind === 'branch')
    body = (
      <>
        <rect x={x} y={y} width={w} height={h} rx="2" fill="var(--cell)" stroke="var(--ink)" strokeWidth="1.2" />
        <text x={x + w / 2} y={y + 15} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink)' }}>br</text>
      </>
    );
  else if (c.kind === 'stallDep') body = <rect x={x} y={y} width={w} height={h} rx="2" fill={`url(#${hatch})`} stroke="var(--ink-3)" strokeWidth="0.6" />;
  else body = <rect x={x} y={y} width={w} height={h} rx="2" fill="var(--rule)" stroke="var(--ink-3)" strokeWidth="0.6" strokeDasharray="2 2" />;
  return <g data-cell={c.kind}>{body}</g>;
}

function Legend({ hatch }: { hatch: string }) {
  const items: [Cell, string][] = [
    [{ kind: 'issueA', i: 0 }, 'issue A(i)'],
    [{ kind: 'issueB', i: 0 }, 'issue B(i)'],
    [{ kind: 'branch' }, 'loop branch'],
    [{ kind: 'stallDep' }, 'wait for A(i)'],
    [{ kind: 'stallBranch' }, 'wait at the branch'],
  ];
  return (
    <svg viewBox="0 0 640 28" role="img" aria-label="legend" style={{ width: '100%', maxWidth: 640, display: 'block', marginBottom: '0.4rem' }}>
      {items.map(([c, label], k) => (
        <g key={k}>
          <CellBox c={c} x={k * 128} y={2} hatch={hatch} />
          <text x={k * 128 + CW + 4} y={17} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{label}</text>
        </g>
      ))}
    </svg>
  );
}

function IssueTimeline() {
  const [unroll, setUnroll] = useState(1);
  const [reorder, setReorder] = useState(false);
  const [latencyAB, setLatency] = useState(DEFAULTS.latencyAB);
  const [branchStall, setBranch] = useState(DEFAULTS.branchStall);
  const hatch = 'hatch' + useId().replace(/:/g, '');
  const cfg = { ...DEFAULTS, unroll, reorder, latencyAB, branchStall };
  const t = schedule(cfg);
  const base = schedule({ ...cfg, unroll: 1, reorder: false });
  const rows = Math.ceil(t.total / PER_ROW);
  const rowH = 30;

  return (
    <Bench
      title="Issue timeline"
      task="Unroll by 4 and watch the branch stalls; then reorder. What does full unrolling leave?"
      foot="One thread, one instruction per cycle, issued in order. Each square is one cycle; rows wrap every 32 cycles."
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <span className="seg" role="group" aria-label="unroll factor">
          {FACTORS.map((u) => (
            <button key={u} aria-pressed={u === unroll} onClick={() => setUnroll(u)}>{u === 16 ? 'unroll 16 (full)' : `unroll ${u}`}</button>
          ))}
        </span>
        <label className="check">
          <input type="checkbox" checked={reorder} onChange={(e) => setReorder(e.target.checked)} /> reorder: A's first, then B's
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          A-to-B latency: {latencyAB} cycle{latencyAB === 1 ? '' : 's'}
          <input type="range" min={0} max={8} value={latencyAB} aria-label="A-to-B latency" onChange={(e) => setLatency(+e.target.value)} />
        </label>
        <label className="field">
          Stall at each branch: {branchStall} cycle{branchStall === 1 ? '' : 's'}
          <input type="range" min={0} max={8} value={branchStall} aria-label="branch stall" onChange={(e) => setBranch(+e.target.value)} />
        </label>
      </div>
      <p className="ui" style={{ margin: '0 0 0.7rem' }}>
        {unroll === 16 ? 'No loop left. Body: ' : `for (i = 0; i < 16; i += ${unroll}): `}
        <code data-k="body">{bodyText(unroll, reorder)}</code>
      </p>
      <Legend hatch={hatch} />
      <svg viewBox={`0 0 ${PER_ROW * CW + 50} ${rows * rowH + 4}`} role="img" aria-label="cycle-by-cycle issue of one thread"
        style={{ width: '100%', maxWidth: PER_ROW * CW + 50, display: 'block' }}>
        <defs>
          <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="5" height="5" fill="var(--cell)" />
            <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.6" />
          </pattern>
        </defs>
        {Array.from({ length: rows }, (_, r) => (
          <text key={r} x={0} y={r * rowH + 17} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{r * PER_ROW}</text>
        ))}
        {t.cells.map((c, k) => (
          <CellBox key={k} c={c} x={34 + (k % PER_ROW) * CW} y={Math.floor(k / PER_ROW) * rowH + 2} hatch={hatch} />
        ))}
      </svg>
      <table className="data" style={{ background: 'var(--page)', margin: '0.8rem 0 0' }} aria-label="counters">
        <thead><tr><th></th><th>this setting</th><th>no unrolling, same latencies</th></tr></thead>
        <tbody>
          <tr><td>loop branches</td><td data-k="branches">{t.branches}</td><td>{base.branches}</td></tr>
          <tr><td>cycles waiting for A</td><td data-k="stallDep">{t.stallDep}</td><td>{base.stallDep}</td></tr>
          <tr><td>cycles waiting at branches</td><td data-k="stallBranch">{t.stallBranch}</td><td>{base.stallBranch}</td></tr>
          <tr><td>total cycles</td><td data-k="total"><b>{t.total}</b></td><td>{base.total}</td></tr>
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- where does x live?
function WhereXLives() {
  const [pick, setPick] = useState<Record<string, Place>>({});
  const right = WHERE_SNIPPETS.filter((s) => pick[s.id] === whereLives(s).place).length;
  const answered = Object.keys(pick).length;
  return (
    <Bench
      title="Where does x live?"
      task="Sort each kernel: does x end up in registers, or in local memory off the chip?"
      foot={
        <>
          From <code>labs/ch06/snippets/s66_where_x_lives.cu</code>, where <code>COARSEN</code> is 4. The answers are
          what the compiler is likely to do (the book's rule and NVIDIA's list); <code>--ptxas-options=-v</code> shows
          what it did.
        </>
      }
    >
      {WHERE_SNIPPETS.map((s, k) => {
        const mine = pick[s.id];
        const ans = whereLives(s);
        return (
          <div key={s.id} data-snippet={s.id} style={{ marginBottom: '1.2rem', paddingBottom: '0.9rem', borderBottom: '1px solid var(--rule-soft)' }}>
            <p className="ui" style={{ margin: '0 0 0.3rem' }}><b>{k + 1}.</b> {s.label}</p>
            <Code tabs={[{ label: s.id, code: s.code }]} />
            <div className="row" role="group" aria-label={`where x lives in kernel ${k + 1}`}>
              {(['registers', 'local memory'] as Place[]).map((p) => (
                <button key={p} aria-pressed={mine === p} onClick={() => setPick({ ...pick, [s.id]: p })}>
                  {p === 'registers' ? 'registers' : 'local memory (off-chip)'}
                </button>
              ))}
              {mine ? (
                <span data-k="verdict" className={'chip ' + (mine === ans.place ? 'ok' : 'fail')}>
                  {mine === ans.place ? 'right' : 'not this'}: {ans.place === 'registers' ? 'registers' : 'local memory'}
                </span>
              ) : null}
            </div>
            {mine ? <p className="muted" style={{ margin: '0.3rem 0 0' }}>{ans.rule}</p> : null}
          </div>
        );
      })}
      <p className="ui" style={{ margin: 0 }} data-k="score">{answered ? `${right} of ${answered} right so far.` : 'Nothing sorted yet.'}</p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS66: Record<string, Question> = {
  'ch06/6.6/q-branches': {
    kind: 'numeric',
    id: 'ch06/6.6/q-branches',
    prompt: <>A 16-iteration loop is unrolled by a factor of 4. How many loop branches does a thread execute?</>,
    answer: 4,
    explain: (
      <p>
        One per remaining iteration: 16 ÷ 4 = 4, down from 16. Unrolled by 16, the loop disappears and so does its last
        branch.
      </p>
    ),
    source: 'ch. 6 §6.6, p. 144',
  },
  'ch06/6.6/q-reorder': {
    kind: 'choice',
    id: 'ch06/6.6/q-reorder',
    prompt: (
      <>
        When may the unrolled body A(i) B(i) A(i+1) B(i+1) … be reordered to A(i) A(i+1) … B(i) B(i+1) …?
      </>
    ),
    options: [
      { label: 'When A(i+1) does not depend on A(i) or B(i).', correct: true },
      { label: 'Always: the hardware resolves dependences at run time.', why: 'A GPU thread issues in order and does not speculate. If A(i+1) needed B(i)\'s result, moving it in front would compute the wrong thing.' },
      { label: 'Only when B does not depend on A.', why: 'Then there would be no stall to hide. B(i) depending on A(i) is the very reason to move independent work between them.' },
    ],
    explain: (
      <p>
        Moving A(i+1) in front of B(i) changes the order in which they happen, so it is allowed only if nothing A(i+1)
        reads is written by A(i) or B(i). Then the thread has useful work while A(i)'s result is on its way. This is part
        of what a compiler calls instruction scheduling.
      </p>
    ),
    source: 'ch. 6 §6.6, pp. 144–145',
  },
  'ch06/6.6/q-promotion': {
    kind: 'choice',
    id: 'ch06/6.6/q-promotion',
    prompt: <>Why does unrolling matter more after coarsening?</>,
    options: [
      {
        label: 'The coarsened thread keeps its values in a small array x[c], which can live in registers only once every index is a compile-time constant; otherwise it sits in off-chip local memory and that traffic can cancel the gain.',
        correct: true,
      },
      { label: 'Coarsened loops have more branches than there are registers.', why: 'Branches and registers are unrelated resources.' },
      { label: 'Unrolling raises occupancy.', why: 'If anything, unrolling keeps more values live at once and can raise register use.' },
    ],
    explain: (
      <p>
        A register has no address, so an element of an array can be in a register only if the compiler knows at compile
        time which element each instruction means. Fully unrolling the coarsening loop turns x[c] into x[0], x[1], x[2],
        x[3], and each can become its own register.
      </p>
    ),
    source: 'ch. 6 §6.6, pp. 145–146',
  },
  'ch06/6.6/q-pragma1': {
    kind: 'choice',
    id: 'ch06/6.6/q-pragma1',
    prompt: (
      <>
        Someone adds <code>#pragma unroll 1</code> above a coarsening loop that indexes <code>float v[4]</code> with the
        loop counter. What do you expect on CUDA, and how do you check?
      </>
    ),
    options: [
      { label: 'v is likely placed in local memory; nvcc --ptxas-options=-v reports the local-memory (lmem) bytes.', correct: true },
      { label: 'No change: on CUDA the pragma is only a hint.', why: 'NVIDIA documents 1 as "the loop will not be unrolled" (infra A7). It is AMD that calls the numeric form a hint (B7).' },
      { label: 'A compile error: a loop over an array must be unrolled.', why: 'It compiles. The array just has to live somewhere addressable.' },
    ],
    explain: (
      <p>
        With unrolling forbidden, every access is v[c] with a variable c, and NVIDIA lists "arrays for which it cannot
        determine that they are indexed with constant quantities" first among what goes to local memory. In PTX those
        accesses appear as <code>ld.local</code> and <code>st.local</code>.
      </p>
    ),
    source: 'ch. 6 §6.6, p. 145; notes/today/ch06-facts.md A13; infra-facts.md A7',
  },
  'ch06/6.6/q-metal': {
    kind: 'choice',
    id: 'ch06/6.6/q-metal',
    prompt: (
      <>
        You port a coarsened kernel to Metal. Its coarsening loop indexes a thread-local <code>float v[4]</code> with
        the loop counter, and you put MLX&rsquo;s <code>_Pragma("clang loop unroll(full)")</code> in front of it. It
        compiles without a warning, and the AIR the front end emits records the request. Is <code>v</code> in
        registers, and what do you do next?
      </>
    ),
    options: [
      {
        label: "You cannot tell from the source or the AIR. Write a version with constant indices (v[0] to v[3], or four named variables) and measure the two.",
        correct: true,
      },
      {
        label: 'Yes: the AIR records full unrolling, so every index is a constant in the GPU code.',
        why: 'AIR is compiled again by the driver, and Apple documents nothing about what it does with the request. The AIR shows what was asked for, not what runs.',
      },
      {
        label: 'No: Metal ignores pragmas, so delete it and accept local memory.',
        why: 'The specification documents several floating-point pragmas, and the front end records this one. Whether the driver honours it is unknown, not known to be no.',
      },
      {
        label: 'Yes: Apple documents this pragma as the Metal spelling of #pragma unroll.',
        why: 'The Metal Shading Language specification has no loop pragma at all (C4). MLX and llama.cpp use Clang\'s spelling, and benchmark.',
      },
    ],
    explain: (
      <p>
        The local front end accepts both <code>_Pragma("clang loop unroll(full)")</code> and <code>#pragma unroll 4</code>{' '}
        and records them in the AIR it emits; nothing Apple publishes says what the driver does with them. What Apple
        does say is the rule behind the question: avoid &ldquo;implementations that randomly index into thread-scoped
        arrays&rdquo;, because they can push data out to device memory. Constant indices that you write yourself do not
        depend on the pragma, so the hand-written version is the baseline to measure against.
      </p>
    ),
    source: 'notes/today/ch06-facts.md C4, D2, D3 (compile-only observation; not in the book)',
  },
};

export function S66() {
  return (
    <>
      <p>
        A GPU hides latency mostly by switching: when one warp waits for a slow result, the scheduler issues another
        warp's instruction instead (<a href={href('ch04', '4.6')}>chapter 4</a>). Loop unrolling is a second tool, and it
        works inside a single thread. Replicate the loop body some number of times and divide the trip count by the same
        number.
      </p>
      <Source section="6.6" pages="144" />
      <Code
        tabs={[{ label: 's66_unroll.cu', code: UNROLL, status: 'front-end-checked; unverified on CUDA and HIP hardware' }]}
        highlight={[18, 19, 20, 21, 22, 23, 24, 25, 26]}
        caption={
          <>
            The book's 16-iteration loop (p. 144) as a kernel, from <code>labs/ch06/snippets/</code>, then unrolled by 4
            and reordered by hand, then the same request made with a pragma. A and B are ours; the book leaves them
            abstract.
          </>
        }
      />

      <h2>What it gains</h2>
      <p>
        <b>Fewer branches.</b> A CPU guesses which way a branch will go and carries on. A GPU thread does not: it stalls at
        the branch until the outcome is known.
      </p>
      <Source section="6.6" pages="144" />
      <p>
        A loop branches once per iteration, so a loop unrolled by 4 branches a quarter as often.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md, A11; ch06-facts.md, A14)">
        <p>
          NVIDIA: SMs "issue instructions in order and do not perform branch prediction or speculative execution". Its
          guide adds that the compiler may replace a short branch with predication, in which case there is no branch at
          all.
        </p>
      </Today>
      <p>
        <b>Independent instructions to fill the gap.</b> If B(i) needs A(i)'s result, the thread waits between them. In
        the unrolled body there are four A's, and if A(i+1) does not depend on A(i) or B(i), the four can go first: by the
        time B(i) issues, A(i) has had three cycles to finish. Choosing that order is part of instruction scheduling.
      </p>
      <Source section="6.6" pages="144–145" />
      <p>
        The timeline below issues the three versions of the loop, original, unrolled and reordered, so you can count
        the cycles each one loses to branches and to waiting for A.
      </p>
      <Simplification>
        <p>
          One thread alone, with no other warps to switch to; one instruction per cycle, in order; a fixed A-to-B latency
          and a fixed stall after every branch, with the loop's counter update folded into the branch. Real compilers
          schedule more freely.
        </p>
      </Simplification>
      <IssueTimeline />
      <p>
        With the default latencies, unrolling by 4 alone takes the thread from 128 cycles to 92: the branch stalls shrink
        from 32 cycles to 8, while the 48 cycles spent waiting for A stay. Reordering the unrolled body removes those 48 as
        well, for 44 cycles. Full unrolling leaves no loop and no branch.
      </p>
      <Beyond>
        <p>
          In C++ terms the book's condition is about aliasing too: a store through <code>out</code> in B(i) could change
          what a later load through <code>in</code> sees, unless the compiler can tell the two never overlap. That is ours,
          not the book's.
        </p>
      </Beyond>
      <Ask q={questionsS66['ch06/6.6/q-reorder']} />

      <h2>Who unrolls</h2>
      <p>
        Usually the compiler, on its own and aggressively, together with the scheduling above. A loop with a small body and
        a small constant trip count will most likely be unrolled completely, so it has no branches left. You can ask for a
        factor with <code>#pragma unroll N</code> on the line before the loop, and <code>#pragma unroll 1</code> forbids
        unrolling. The book writes the pragma only where it wants you to notice that unrolling is expected.
      </p>
      <Source section="6.6" pages="145" />
      <Ask q={questionsS66['ch06/6.6/q-branches']} label="Work it out" />
      <p>How to ask for unrolling, or forbid it, on each backend:</p>
      <Beyond>
        <p>
          The book covers CUDA only. The other two columns come from vendor documentation and, for Metal, from compiling a
          test kernel with the local front end (notes/today/ch06-facts.md, C4).
        </p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th></th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Ask for unrolling</td>
            <td><code>#pragma unroll</code> (complete, if the trip count is constant) or <code>#pragma unroll N</code></td>
            <td>same spelling; AMD calls the numeric form a hint</td>
            <td>nothing in the language specification</td>
          </tr>
          <tr>
            <td>Forbid it</td>
            <td><code>#pragma unroll 1</code></td>
            <td><code>#pragma unroll 1</code></td>
            <td>nothing documented</td>
          </tr>
          <tr>
            <td>What real code writes</td>
            <td colSpan={2}>the pragma above</td>
            <td><code>_Pragma("clang loop unroll(full)")</code> (MLX, llama.cpp); the front end accepts it and <code>#pragma unroll 4</code>, compile-only</td>
          </tr>
          <tr>
            <td>Source</td>
            <td>infra-facts A7</td>
            <td>infra-facts B7</td>
            <td>ch06-facts C4, D2, D3</td>
          </tr>
        </tbody>
      </table>
      <Ask q={questionsS66['ch06/6.6/q-metal']} />
      <p>
        When you read MLX's Metal kernels you will see <code>STEEL_PRAGMA_UNROLL</code> in front of nearly every inner
        loop, and in llama.cpp's <code>FOR_UNROLL(...)</code>. Both are macros for that Clang pragma: the authors ask for
        full unrolling, and Apple's documentation does not say what the GPU compiler does with the request.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, D2, D3)">
        <p>
          MLX: <code>#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")</code>. llama.cpp:{' '}
          <code>#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)</code>.
        </p>
      </Today>

      <h2>Arrays in registers</h2>
      <p>
        The third gain is the one coarsening needs. A coarsened thread keeps one value per unit of work, so a scalar{' '}
        <code>x</code> becomes an array <code>x[4]</code>, used as <code>foo(x[c])</code> inside the coarsening loop. An
        array indexed with a variable has to live in memory, because registers have no addresses; that memory is off the
        chip, and its traffic can eat the coarsening gain. Unroll the loop fully and every index becomes a constant:{' '}
        <code>foo(x[0]); foo(x[1]); foo(x[2]); foo(x[3]);</code>. Then the compiler can give each element its own register.
      </p>
      <Source section="6.6" pages="145–146" />
      <p>
        One word needs care. Chapter 6 says such arrays go to "global memory"; chapter 5 called the same place local
        memory. NVIDIA calls it local memory, whose physical location is device memory, the same DRAM as global memory, with
        the same latency. AMD's documentation says a thread's private storage for local variables and spilled registers
        resides in global memory. Apple's says that dynamically indexing thread-scoped arrays can force data out to device
        memory. All three describe one thing: off the chip, private to the
        thread.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A13, B6, C4)">
        <p>
          To see it on CUDA, compile with <code>nvcc --ptxas-options=-v</code>: it reports each kernel's local-memory
          bytes (lmem). In PTX the accesses show up as <code>ld.local</code> and <code>st.local</code>.
        </p>
      </Today>
      <WhereXLives />
      <Ask q={questionsS66['ch06/6.6/q-promotion']} />
      <Ask q={questionsS66['ch06/6.6/q-pragma1']} />

      <h2>Where this goes</h2>
      <p>
        Fewer branches, more instructions to schedule, constant indices the compiler can fold, and arrays promoted to
        registers: those are the four benefits the book lists. The last is why unrolling becomes essential once threads are
        coarsened. Chapter 11 (prefix sum) and chapter 15 (matrix multiplication) depend on it for register tiling, where
        each thread keeps several results in registers; chapter 15 also leans on it to make up for low occupancy.
      </p>
      <Source section="6.6" pages="146" figs="§6.8, p. 149 for chapters 11 and 15" />
    </>
  );
}
