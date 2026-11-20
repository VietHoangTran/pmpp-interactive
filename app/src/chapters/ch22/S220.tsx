import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source } from '../../components/notes';
import { href } from '../../lib/router';

// Prerequisite questions drawn from chapters 1, 5, 9, 10 and 15 (their notes, labs/README.md
// and lab 15.6's README). Each explanation names where the idea was taught and where this
// chapter uses it.
export const questionsS220: Record<string, Question> = {
  'ch22/start/q-decode-amdahl': {
    kind: 'numeric',
    id: 'ch22/start/q-decode-amdahl',
    prompt: (
      <>
        A decode loop spends 80% of each token&rsquo;s time in GPU kernels and 20% on the host (sampling, kernel
        launches). You make every kernel 4 times faster and change nothing on the host. What is the overall speed-up
        per token?
      </>
    ),
    answer: 2.5,
    tol: 0.01,
    unit: '×',
    explain: (
      <p>
        1 / (0.2 + 0.8 / 4) = 1 / 0.4 = 2.5. After the change the host&rsquo;s 0.2 is as large as the kernels&rsquo;
        0.2, so half of every token is now work you did not touch, and another 4× on the kernels would take you only to
        1 / 0.25 = 4. Section <a href={href('ch22', '22.3')}>22.3</a> makes this argument for a whole application.
        Taught in <a href={href('ch01', '1.3')}>ch. 1 §1.3</a>.
      </p>
    ),
    source: 'ch. 1 §1.3, p. 9 (the formula: notes/today/ch01-facts.md A16)',
  },
  'ch22/start/q-gemv-ceiling': {
    kind: 'numeric',
    id: 'ch22/start/q-gemv-ceiling',
    prompt: (
      <>
        A float32 matrix-vector product does 2 floating-point operations for each 4-byte weight it reads. On a device
        whose threshold is 20 FLOP/B (the book&rsquo;s example), at most what percentage of the peak arithmetic can it
        use?
      </>
    ),
    answer: 2.5,
    tol: 0.01,
    unit: '%',
    explain: (
      <p>
        2 / 4 = 0.5 FLOP/B, and 0.5 / 20 = 2.5%. No kernel trick moves it, because each weight is used once per token.
        The only levers are more rows per weight (batching, section <a href={href('ch22', '22.5')}>22.5</a>) or fewer
        bytes per weight (quantisation). Taught in <a href={href('ch05', '5.1')}>ch. 5 §5.1</a>.
      </p>
    ),
    source: "ch. 5 §5.1, pp. 94–97 (20 FLOP/B is the book's H100 example)",
  },
  'ch22/start/q-emu-blind': {
    kind: 'choice',
    id: 'ch22/start/q-emu-blind',
    prompt: (
      <>
        A histogram kernel counts with a plain <code>bins[b]++</code> and passes every test on the <code>emu</code>{' '}
        backend. Why, and what would catch the bug?
      </>
    ),
    options: [
      {
        label: (
          <>
            <code>emu</code> runs each thread until it reaches a synchronisation point, so no increment is ever
            interrupted; the <code>race</code> backend reports the colliding lines.
          </>
        ),
        correct: true,
      },
      {
        label: (
          <>
            <code>emu</code> gives every thread a private copy of <code>bins</code>, so increments never meet; any
            real GPU would catch it on the first run.
          </>
        ),
        why: 'There is one array, checked by AddressSanitizer. And a real GPU may pass too: a race only shows when two increments actually overlap.',
      },
      {
        label: 'Increments commute, so the order of the threads cannot matter: the kernel is correct and needs no other test.',
        why: 'Commuting makes the order irrelevant, not the read-modify-write safe. Two threads that load the same old count lose one increment.',
      },
      {
        label: (
          <>
            <code>emu</code> only checks array bounds, so it would pass a kernel that counted into the wrong bins as well.
          </>
        ),
        why: 'Every lab compares the result with a CPU reference, so wrong bins fail. What emu cannot produce is an interleaving inside one increment.',
      },
    ],
    explain: (
      <p>
        <code>bins[b]++</code> is a load, an add and a store. A lost update needs another thread to run between the
        load and the store, and <code>emu</code> never switches threads there. That is why every lab about shared
        updates also builds for <code>race</code>, which runs the threads concurrently under ThreadSanitizer. Section{' '}
        <a href={href('ch22', '22.2')}>22.2</a> calls this pattern scatter: whenever two threads may write one output,
        you need an atomic, and a tool that can see its absence. Taught in <a href={href('ch09', '9.2')}>ch. 9 §9.2</a>.
      </p>
    ),
    source: 'labs/README.md (the emulators); ch. 9 §9.2, pp. 203–209',
  },
  'ch22/start/q-tree-writers': {
    kind: 'choice',
    id: 'ch22/start/q-tree-writers',
    prompt: <>In one step of chapter 10&rsquo;s block reduction tree, how many threads write each partial-sum slot?</>,
    options: [
      { label: 'Exactly one, so the tree needs a barrier between steps but no atomics.', correct: true },
      {
        label: 'Two, one for each operand, so every slot needs an atomic add.',
        why: 'Two values are read, but only the thread that owns the slot writes it.',
      },
      {
        label: 'Every thread of the warp, so the warp must take turns at the slot.',
        why: 'Each active thread has its own slot in a step; the warp does not share one.',
      },
    ],
    explain: (
      <p>
        Thread t reads two partial sums and writes one slot that no other thread writes in that step: the owner
        computes. A slot&rsquo;s reader and writer meet only across steps, and the barrier separates those. Section{' '}
        <a href={href('ch22', '22.2')}>22.2</a> calls this output-centric, judged one iteration at a time: each thread
        gathers the inputs of the one output it owns. Taught in <a href={href('ch10', '10.3')}>ch. 10 §10.3</a>.
      </p>
    ),
    source: 'ch. 10 §10.3–10.4, pp. 227–234; ch. 22 §22.2, p. 534',
  },
  'ch22/start/q-m3-rows': {
    kind: 'numeric',
    id: 'ch22/start/q-m3-rows',
    prompt: (
      <>
        Milestone M3 (lab 15.6) computes <code>Y</code> in block tiles of 128 rows by 128 columns, and the rows are the
        batch. At batch 3, what percentage of the rows a block computes are real?
      </>
    ),
    answer: 2.34,
    tol: 0.05,
    unit: '%',
    explain: (
      <p>
        3 / 128 = 2.34%. The other 125 rows are zero padding and the kernel does their arithmetic anyway, so every
        batch from 1 to 128 launches the same grid and does the same work. A batching measurement on M3 therefore shows
        a flat time per step up to batch 128, whether or not the kernel is limited by memory. Section{' '}
        <a href={href('ch22', '22.5')}>22.5</a> and lab 22.2 come back to this. Taught in{' '}
        <a href={href('ch15', 'labs')}>chapter 15&rsquo;s labs</a>.
      </p>
    ),
    source: 'labs/ch15/lab6_linear/README.md; notes/today/ch22-facts.md E3',
  },
};

// The chapter's four steps as the summary names them (p. 540), then the two sections that
// widen the frame.
const STEPS: { step: string; where: string; link?: string; what: string }[] = [
  { step: 'Problem formulation', where: '22.4', link: '22.4', what: 'Pose the problem so that a fast algorithm exists, even if that means giving up some exactness.' },
  { step: 'Algorithm selection', where: '22.1', link: '22.1', what: 'Choose among algorithms for the same problem by what each trades on your hardware.' },
  { step: 'Problem decomposition', where: '22.2', link: '22.2', what: 'Decide which thread owns which output, or which input, and pay for that choice.' },
  { step: 'Optimisation', where: 'chapters 2 to 21', what: 'Everything else in the book: tiling, coalescing, privatisation, coarsening and the rest.' },
];
const FRAME: { step: string; where: string; what: string }[] = [
  { step: "Amdahl's law for a whole application", where: '22.3', what: 'Decide which modules go on the GPU; whatever stays behind caps the speed-up.' },
  { step: 'Batching', where: '22.5', what: 'Optimise one request’s latency, or the throughput of many requests at once.' },
];

export function S220() {
  const q = questionsS220;
  return (
    <>
      <p>
        Every chapter so far has handed you a technique: tiling, privatisation, coarsening, a scan network, a merge by
        co-ranks. This one steps back to the decisions you make before any of them. What exactly is the problem? Which
        algorithm solves it? How do you cut it into pieces that threads can own? And is the result fast where it
        counts, in the whole application? The book calls this parallel thinking, and it casts the programmer as a
        collaborator who can reshape the problem a domain scientist brings, not only implement it.
      </p>
      <Source pages="529" />
      <p>
        The chapter&rsquo;s summary names the steps in order, and the table below uses them as a map. Two sections widen
        the frame: Amdahl&rsquo;s law applied to a whole program, and the choice between latency and throughput.
      </p>
      <Source section="22.6" pages="540" />
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-map>
          <thead>
            <tr>
              <th>Step</th>
              <th>Where</th>
              <th>The decision</th>
            </tr>
          </thead>
          <tbody>
            {STEPS.map((s) => (
              <tr key={s.step} data-step={s.where}>
                <td>{s.step}</td>
                <td style={{ whiteSpace: 'nowrap' }}>{s.link ? <a href={href('ch22', s.link)}>{s.where}</a> : s.where}</td>
                <td>{s.what}</td>
              </tr>
            ))}
            {FRAME.map((s) => (
              <tr key={s.step} data-step={s.where}>
                <td>{s.step}</td>
                <td>
                  <a href={href('ch22', s.where)}>{s.where}</a>
                </td>
                <td>{s.what}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        Formulation is first in the summary&rsquo;s list but fourth among the sections. This course keeps the
        book&rsquo;s order, so you will have watched selection and decomposition at work before you meet it. For it the
        chapter cites its single reference, a paper on what its author calls computational thinking.
      </p>
      <Aside>
        <p>
          J. M. Wing, &ldquo;Computational thinking&rdquo;, <i>Communications of the ACM</i> 49(3), 2006, pp. 33–35. The
          chapter&rsquo;s one reference (p. 540; checked against its DOI record, ch22-facts A10).
        </p>
      </Aside>

      <h2>A chapter without code</h2>
      <p>
        There are no code listings in this chapter, no exercises and no device figures. It is reasoning about the
        decisions behind the other chapters, illustrated with them. So the course adds two labs of its own, on the{' '}
        <a href={href('ch22', 'labs')}>labs page</a>. Lab 22.1 cuts a decode-time matrix-vector product two ways. Lab
        22.2 is milestone M8, a batching study that measures latency and throughput on your own device.
      </p>

      <h2>Where you will meet these decisions</h2>
      <p>
        Every one of them is made, explicitly or by default, in the GPU path of an inference engine:
      </p>
      <ul>
        <li>
          Which kernel multiplies the weights at which batch size is algorithm selection: llama.cpp keeps a
          matrix-vector kernel and a matrix kernel for the same operation and switches at a fixed batch of about 8 tokens,
          lowered for some weight types on particular GPUs where it was measured to cross sooner (
          <a href={href('ch22', '22.1')}>22.1</a>).
        </li>
        <li>
          Which thread owns which output of a decode-step product, and whether partial sums meet in an atomic, is
          decomposition (<a href={href('ch22', '22.2')}>22.2</a>).
        </li>
        <li>
          What stays on the host in the decode loop, such as sampling and kernel launches, is the serial part that caps
          the speed-up of every kernel you write (<a href={href('ch22', '22.3')}>22.3</a>).
        </li>
        <li>
          How many requests a server batches, and how much longer each user waits for it, is the trade between latency
          and throughput (<a href={href('ch22', '22.5')}>22.5</a>).
        </li>
      </ul>
      <p>
        The book&rsquo;s own examples are scans, sorts, histograms and an electrostatic map, with LLM queries only in
        section 22.5. The inference examples on these pages are the course&rsquo;s, and each is labelled.
      </p>
      <Beyond>
        <p>These four mappings onto inference are ours. The llama.cpp switch is checked in section 22.1 (ch22-facts D6).</p>
      </Beyond>

      <h2>Before you start</h2>
      <p>
        Five questions on what this chapter leans on from chapters 1, 5, 9, 10 and 15. Each explanation points to where
        the idea was taught and to the section here that uses it. They come back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={q['ch22/start/q-decode-amdahl']} label="From chapter 1" />
      <Ask q={q['ch22/start/q-gemv-ceiling']} label="From chapter 5" />
      <Ask q={q['ch22/start/q-emu-blind']} label="From chapter 9 and the labs" />
      <Ask q={q['ch22/start/q-tree-writers']} label="From chapter 10" />
      <Ask q={q['ch22/start/q-m3-rows']} label="From chapter 15" />
    </>
  );
}
