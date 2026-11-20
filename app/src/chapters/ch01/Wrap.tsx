import { Ask, type Question } from '../../components/Ask';
import { Aside, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { BOOK_CASES, ceiling, remaining, speedup, untouchedShare } from '../../sim/amdahl';
import { chaptersBefore, goal, leadsTo, prerequisites } from '../../sim/bookmap';
import { design, finish } from '../../sim/designs';
import { BUDGET } from './TwoDesignsScene';

// Chapter 1 has no code, so this page shows no kernel and needs no snippet file.
// Every number its questions use comes from the models the sections run on;
// Wrap.test.ts pins them.

const n = (v: number) => v.toLocaleString('en-US', { maximumFractionDigits: 2 });
const and = (xs: (string | number)[]) =>
  xs.length <= 1 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1];

// ---- q-reprofile: Amdahl's law applied twice (sim/amdahl.ts).
export const PROFILE = { p: 0.9, s: 9 };
const LEFT = remaining(PROFILE.p, PROFILE.s);
const HOST_SHARE = untouchedShare(PROFILE.p, PROFILE.s);
const REACHED = speedup(PROFILE.p, PROFILE.s);
const TOP = ceiling(PROFILE.p);

// ---- q-both: the two designs of the start page on a job with two phases (sim/designs.ts).
const FAST = design(BUDGET, 2);
const MANY = design(BUDGET, 0);
export const JOB = { chain: 1200, crowd: 1024, length: 100 };
const last = (d: typeof FAST, chains: number, length: number) => finish(d, chains, length).last;
const CHAIN = { fast: last(FAST, 1, JOB.chain), many: last(MANY, 1, JOB.chain) };
const CROWD = { fast: last(FAST, JOB.crowd, JOB.length), many: last(MANY, JOB.crowd, JOB.length) };
const ALONE_FAST = CHAIN.fast + CROWD.fast;
const ALONE_MANY = CHAIN.many + CROWD.many;
const SPLIT = CHAIN.fast + CROWD.many;
const WRONG_WAY = CHAIN.many + CROWD.fast;
const CROWD_WAVES = { fast: finish(FAST, JOB.crowd, JOB.length).waves, many: finish(MANY, JOB.crowd, JOB.length).waves };

// ---- q-skip9: what the book's arrows put behind chapter 9 (sim/bookmap.ts).
const SKIPPED = 9;
const BEHIND_SKIPPED = leadsTo([SKIPPED]);
const SAMPLING = goal('sampling').chapters;
const SAMPLING_BLOCKED = SAMPLING.filter((c) => prerequisites([c]).includes(SKIPPED));
const SAMPLING_FREE = SAMPLING.filter((c) => !SAMPLING_BLOCKED.includes(c));

// ---- the reading-path figures quoted under "What to carry forward".
const BEFORE_MATMUL = chaptersBefore([15]);
const BEFORE_ATTENTION = chaptersBefore([20]);

export const questionsWrap: Record<string, Question> = {
  'ch01/wrap/q-reprofile': {
    kind: 'numeric',
    id: 'ch01/wrap/q-reprofile',
    prompt: (
      <>
        A profile of a decode step shows {n(PROFILE.p * 100)}% of the time in kernels and the rest on the host. You
        make the kernels {PROFILE.s} times faster, change nothing on the host, and profile again. What share of the
        run time does the host part take now?
      </>
    ),
    answer: 50,
    tol: 0.005,
    unit: '%',
    explain: (
      <p>
        The kernels shrink from {n(PROFILE.p * 100)}% of the old time to {fmt(LEFT.accelerated * 100)}%. The host part
        is still {fmt(LEFT.untouched * 100)}% of the old time, so the step now takes {fmt(LEFT.total * 100)}% of what
        it did ({fmt(REACHED)}× overall), and {fmt(LEFT.untouched * 100)} of those {fmt(LEFT.total * 100)} points are
        the host's: {fmt(HOST_SHARE * 100)}%. The ceiling for this step was {fmt(TOP)}×. You are halfway
        to it, and all the kernel work still to come can at most double the speed again. A profile describes the
        program as it was when you took it. After every step that works, take another before you choose the next.
      </p>
    ),
    source: "our numbers; the law is the book's, ch. 1 §1.3, p. 9",
  },
  'ch01/wrap/q-both': {
    kind: 'choice',
    id: 'ch01/wrap/q-both',
    prompt: (
      <>
        A job has two phases. First one chain of {n(JOB.chain)} operations, each needing the result of the one before.
        Then a crowd: {n(JOB.crowd)} independent chains of {JOB.length} operations each. A machine has both designs of the
        start page: A with {FAST.units} units of speed {FAST.speed}, and B with {MANY.units} units of speed{' '}
        {MANY.speed}. In that page's model, which arrangement finishes the job first?
      </>
    ),
    options: [
      {
        label: 'Both phases on A.',
        why: `A finishes the chain at ${n(CHAIN.fast)}. Then it needs ${n(CROWD_WAVES.fast)} waves for the crowd, ${n(CROWD.fast)} more: ${n(ALONE_FAST)} in all.`,
      },
      {
        label: 'Both phases on B.',
        why: `B clears the crowd in ${n(CROWD_WAVES.many)} waves, but the single chain keeps one slow unit busy until ${n(CHAIN.many)}: ${n(ALONE_MANY)} in all.`,
      },
      { label: 'The chain on A, then the crowd on B.', correct: true },
      {
        label: 'The chain on B, then the crowd on A.',
        why: `That gives each phase to the design that suits it least: ${n(CHAIN.many)} + ${n(CROWD.fast)} = ${n(WRONG_WAY)}.`,
      },
    ],
    explain: (
      <p>
        A finishes the chain at {n(CHAIN.fast)}. B then clears the crowd in {n(CROWD_WAVES.many)} waves,{' '}
        {n(CROWD.many)} more: {n(SPLIT)}, against {n(ALONE_MANY)} for B alone and {n(ALONE_FAST)} for A alone. This is
        the book's reason for programming the two processors together: with one or a few threads the CPU wins, with
        many the GPU wins, so the sequential parts stay on the host and the numerically heavy parts go to the device.
        It is also Amdahl's law seen from the hardware side. On B alone the chain is {n(CHAIN.many)} of the{' '}
        {n(ALONE_MANY)}, and no number of units shortens it. The model leaves out what a real split costs: moving the
        data between the two. Chapter 2 makes that cost visible.
      </p>
    ),
    source: "our model of the book's argument, ch. 1 §1.1, pp. 5–6",
  },
  'ch01/wrap/q-morethreads': {
    kind: 'choice',
    id: 'ch01/wrap/q-morethreads',
    prompt: (
      <>
        A kernel launches exactly one thread per arithmetic unit of the device. Each thread spends most of its time
        waiting, because every memory access takes a long time to come back. Memory is far from saturated: it could
        deliver several times more bytes per second than this kernel asks for. What does a throughput-oriented design
        expect you to change?
      </>
    ),
    options: [
      {
        label: 'Nothing. A unit executes one thread at a time, so more threads than units cannot help.',
        why: 'True at any one instant. But a waiting thread is not executing, and while it waits its unit could run another thread if there were one.',
      },
      { label: 'Launch many more threads than there are units, so that a unit always has one that is ready.', correct: true },
      {
        label: 'Launch fewer threads, so that each gets its data sooner.',
        why: 'An access is slow here because the design allows it to be, not because the threads compete. Memory has throughput to spare.',
      },
      {
        label: 'Make the arithmetic of each thread faster.',
        why: 'The units already sit idle most of the time. Shorter arithmetic only makes the waits a larger share.',
      },
    ],
    explain: (
      <p>
        The design buys throughput by letting single operations take long, in arithmetic and in memory. That pays off
        only if software supplies enough threads to fill the waits, and the book says GPU code is expected to have a
        large number of threads for exactly this reason. Notice the condition in the question: memory was not
        saturated. If it were, bytes could not arrive any faster whatever the number of threads, which is the flat
        part of the "What caps the kernel" explorer in section 1.3. A long latency and a saturated throughput are
        different problems, and more threads cure only the first. Chapter 4 shows the hardware that does the filling.
      </p>
    ),
    source: 'ch. 1 §1.1, p. 5; the mechanism is ch. 4 §4.6, pp. 82–85',
  },
  'ch01/wrap/q-skip9': {
    kind: 'choice',
    id: 'ch01/wrap/q-skip9',
    prompt: (
      <>
        To save time you decide to skip chapter {SKIPPED}, histogram. By the book's prerequisite arrows, which of your
        goals now has a gap in front of it?
      </>
    ),
    options: [
      {
        label: 'Fast matrix multiplication, chapter 15.',
        why: `Chapter 15 needs chapters ${and(BEFORE_MATMUL)} and nothing else.`,
      },
      {
        label: 'Attention, chapter 20.',
        why: `The chart's road to chapter 20 is chapters ${and(BEFORE_ATTENTION)}. Chapter ${SKIPPED} is on another branch.`,
      },
      { label: `Sampling, chapters ${and(SAMPLING)}.`, correct: true },
      {
        label: 'None of them.',
        why: `The chart puts ${BEHIND_SKIPPED.length} chapters behind chapter ${SKIPPED}: ${and(BEHIND_SKIPPED)}.`,
      },
    ],
    explain: (
      <p>
        Chapter {SKIPPED} has an arrow into chapter 11 (scan), 11 has one into 12 (filter), and 12 into 14 (sort). So{' '}
        {SAMPLING_BLOCKED.length} of the {SAMPLING.length} chapters that the sampling goal ends at sit behind it:{' '}
        {and(SAMPLING_BLOCKED)}. Chapter {and(SAMPLING_FREE)} is tied to chapter {SKIPPED} only by the dashed arrow.
        The chart gives no reasons. The Preface suggests one: chapter {SKIPPED} is where the book introduces its
        atomics interface, and the scan and filter chapters both use atomic operations. That link is our reading. The
        general point is that a prerequisite chart answers two questions: what a chapter needs, and what skipping a
        chapter puts out of reach.
      </p>
    ),
    source: 'Fig. P.1, p. xxiii; Preface, pp. xviii–xx; §1.7, p. 16',
  },
  'ch01/wrap/q-sharedmemory': {
    kind: 'choice',
    id: 'ch01/wrap/q-sharedmemory',
    prompt: (
      <>
        The book's recipe for moving a computation to a GPU has six steps: (1) pick the part to parallelise, (2)
        allocate device memory, (3) copy the data to the device, (4) write the kernel, (5) launch it, (6) copy the
        results back. Its Fig. 1.1 gives the GPU a memory of its own. On a machine where the CPU and the GPU share one
        memory, which steps lose the reason they were there for?
      </>
    ),
    options: [
      {
        label: 'Steps 2 and 5: with one memory there is nothing to allocate and nothing to launch.',
        why: 'The GPU still needs a buffer it is allowed to use, and its threads still have to be started.',
      },
      { label: 'Steps 3 and 6, the two copies.', correct: true },
      {
        label: 'Step 4: the kernel can then be an ordinary host function.',
        why: 'The kernel is the computation itself, written for many threads. Where the data lives does not change who runs it.',
      },
      {
        label: 'None. The recipe does not depend on how memory is arranged.',
        why: 'Two of the six steps do nothing except carry bytes from one memory to another.',
      },
    ],
    explain: (
      <p>
        Two memories mean the inputs have to be carried across and the results carried back. With one memory that both
        processors can address, a buffer can be visible to both with no copy: Apple documents this for its GPUs, where
        a buffer in shared storage mode is "allocated in system memory" and the CPU and GPU "share access" to it. What
        does not go away is order: the host must not read a result before the GPU work that produces it has finished.
        A program may still copy, when its data starts out in memory that was not allocated as such a buffer. That is
        then a choice of the program, not a demand of the hardware. Allocation, the kernel and the launch all remain.
        Chapter 2 takes this up.
      </p>
    ),
    source:
      'book §1.7, p. 15, and Fig. 1.1, p. 4; Apple documentation checked 2026-11-20 (notes/today/ch05-facts.md, C6; ch01-facts.md, C3)',
  },
};

// Backends that are not a GPU: the emulator and its race-detection flavour.
const EMULATED = new Set(['emu', 'race']);
const NAME: Record<string, string> = { metal: 'Metal', cuda: 'CUDA', hip: 'HIP' };
// The runtime writes device properties as strings ('1' or 'true'); booleans are accepted too.
const yes = (v: unknown) => v === true || v === '1' || v === 'true';

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const devices = Object.keys(reports).filter((b) => !EMULATED.has(b));
  const unified = reports.metal?.device?.hasUnifiedMemory;
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch01/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          A CPU is built for the latency of one thread, a GPU for the total throughput of very many. By the book's cost
          rule, latency is far more expensive to buy than throughput, so one budget cannot buy both. Which design
          finishes a piece of work first depends on how many of its pieces are independent, not on how much arithmetic
          it holds.
        </li>
        <li>
          A throughput-oriented processor lets single operations take a long time, so it expects many more threads
          than it can execute at one instant. Sequential parts stay on the host and the heavy data-parallel parts go
          to the device, which is why the programming model has both.
        </li>
        <li>
          A peak rate describes hardware, not a program. Before you use one, find out which kind of unit it belongs to
          and what it assumes.
        </li>
        <li>
          A speed-up is one time divided by another, and it says as much about the baseline as about the port. If a
          fraction p of the run is made s times faster, the whole gains 1 / ((1 − p) + p / s) and never more than 1 /
          (1 − p). Work out that ceiling before you write the kernel. Profile again afterwards, because p has changed.
        </li>
        <li>
          A straightforward port tends to stop where memory saturates. The book's figure for that point, about 10×, is
          a rule of thumb. Past it the gain comes from fewer bytes per operation, not from faster arithmetic.
        </li>
        <li>
          Four things make the fast version hard: a parallel algorithm that does more work than the sequential one,
          memory-bound execution, sensitivity to the input, and synchronisation overhead. Say which of them a kernel
          suffers from before you change it.
        </li>
        <li>
          The book draws its chapters as a chart of prerequisites, which this course treats as advice. By its arrows,
          fast matmul needs chapters {and(BEFORE_MATMUL)} and nothing else,
          attention needs {BEFORE_ATTENTION.length} chapters, and sampling is the longest road. From chapter 12 upward,
          a chapter number means something else in the 4th edition.
        </li>
        <li>
          Correctness travels with the source and performance does not. HIP is CUDA's programming model on AMD GPUs,
          and Metal is a second code base. On every device, measure, and query what you would otherwise assume.
        </li>
      </ul>
      <Source pages="xvii–xxvi, 1–18" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Ask q={questionsWrap['ch01/wrap/q-reprofile']} />
      <Ask q={questionsWrap['ch01/wrap/q-both']} />
      <Ask q={questionsWrap['ch01/wrap/q-morethreads']} />
      <Ask q={questionsWrap['ch01/wrap/q-skip9']} />
      <Ask q={questionsWrap['ch01/wrap/q-sharedmemory']} />

      <h2>The book's exercises</h2>
      <p>
        There are none. Chapter 1 has no exercise section and ends with its references on p. 18. What it has are
        worked numbers, and each of them is worked somewhere on these pages. Checking yourself against them is our
        suggestion, not the book's.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="worked">
        <thead>
          <tr><th>The book's number</th><th>Page</th><th>Where it is worked here</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Twice the throughput for twice the power; half the latency for four times the power</td>
            <td>5</td>
            <td>the animation and the "Spend a power budget" explorer on the <a href={href('ch01', 'start')}>start page</a></td>
          </tr>
          <tr>
            <td>34, 67 and 1,979 TFLOPS on the book's example GPU</td>
            <td>3</td>
            <td>the question on the 16-bit figure, <a href={href('ch01', 'start')}>start page</a></td>
          </tr>
          <tr>
            <td>200 s against 10 s is a speed-up of 20×</td>
            <td>9</td>
            <td>"Speed-up" in <a href={href('ch01', '1.3')}>1.3</a></td>
          </tr>
          <tr>
            <td>30% of the run made 100 times faster: {BOOK_CASES[0].book}×</td>
            <td>9</td>
            <td>the first prediction in <a href={href('ch01', '1.3')}>1.3</a>, and a preset of its Amdahl explorer</td>
          </tr>
          <tr>
            <td>The same 30% made infinitely fast: {BOOK_CASES[1].book}×</td>
            <td>9</td>
            <td>the reveal of that prediction, and a preset</td>
          </tr>
          <tr>
            <td>99% of the run made 100 times faster: {BOOK_CASES[2].book}×</td>
            <td>9</td>
            <td>the second prediction in <a href={href('ch01', '1.3')}>1.3</a>, and a preset</td>
          </tr>
          <tr>
            <td>Results above 100× need more than 99.9% of the work in the parallel part</td>
            <td>9–10</td>
            <td>the task in the header of the Amdahl explorer and the question after it, in <a href={href('ch01', '1.3')}>1.3</a></td>
          </tr>
          <tr>
            <td>About 10× when a straightforward port saturates memory</td>
            <td>10</td>
            <td>the "What caps the kernel" explorer in <a href={href('ch01', '1.3')}>1.3</a>; measured on your own device from chapter 5 on</td>
          </tr>
        </tbody>
      </table>

      <h2>Where the book has aged</h2>
      <p>
        A first chapter is full of dates and counts, and they age faster than the ideas. The sections' own notes cover
        the H100 figures (<a href={href('ch01', 'start')}>start page</a>), the 10× rules of thumb (
        <a href={href('ch01', '1.3')}>1.3</a>), and GPU generations, CUDA versions, dynamic parallelism and the
        publisher's contents list (<a href={href('ch01', '1.7')}>1.7</a>). The table has the rest of what we checked.
        None of it changes an argument of the chapter.
      </p>
      <Today checked="2026-11-20 (notes/today/ch01-facts.md, A3, A4, A6, A7, A10 to A12, A15; ch05-facts.md, A9)">
        <p>
          Everything here comes from vendors' and standards bodies' own pages. A few of those pages could only be read
          through a tool that returns extracted text. Nothing was verified on hardware, and nothing in this chapter
          rests on a measurement.
        </p>
      </Today>
      <table className="data wide" style={{ maxWidth: '62rem' }} data-table="aged">
        <thead>
          <tr><th style={{ minWidth: '13rem' }}>The book says</th><th>What we found</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>NVIDIA introduced CUDA in 2007, with hardware support starting from the G80 (pp. 5–7).</td>
            <td>
              NVIDIA's current programming guide and its corporate timeline both date CUDA to 2006. The oldest toolkit
              in its archive, 1.0, is dated June 2007. We could not find the G80 named on a current NVIDIA page, so
              that part is the book's statement only.
            </td>
          </tr>
          <tr>
            <td>More than 1 billion CUDA-enabled GPUs are in use, and there are millions of CUDA programmers (pp. xvii, 6).</td>
            <td>
              The billion could not be verified. NVIDIA's keynote of March 2026 speaks of "hundreds of millions of GPUs
              and computing systems around the world that run CUDA". Millions of developers is NVIDIA's own figure
              too.
            </td>
          </tr>
          <tr>
            <td>Intel and AMD server processors give each core two hardware threads (hyper-threading) (p. 3).</td>
            <td>
              True of some Intel and AMD parts, not of the Intel E-core Xeon we checked, whose threads equal its cores.
              Ampere, the book's ARM example, calls its cores single-threaded. The book's core counts have since been
              passed, which leaves its contrast between a few heavyweight threads and very many light ones intact.
            </td>
          </tr>
          <tr>
            <td>A server CPU of 2022 peaks at "a few TFLOPS" (p. 3).</td>
            <td>The right order of magnitude, going by the export-compliance figures Intel publishes for its processors.</td>
          </tr>
          <tr>
            <td>CPUs and GPUs together have brought exascale computing to clusters (p. 10).</td>
            <td>
              Exascale holds: the TOP500 list of November 2026 has five systems above one exaflop. Its text names GPU
              accelerators for some of the five and not for all, so do not read the sentence as "every exascale
              machine pairs CPUs with GPUs".
            </td>
          </tr>
          <tr>
            <td>OpenMP was designed for CPUs and extended to GPUs, and its compilers are still improving (pp. 12–13).</td>
            <td>
              The history holds. A report on directives for attached accelerators appeared in November 2012 and was
              superseded by OpenMP 4.0 in July 2013; the current specification is 6.1. "Still improving" is the
              authors' opinion, and we did not check it.
            </td>
          </tr>
          <tr>
            <td>OpenCL was developed jointly in 2009 by Apple, Intel, AMD/ATI and NVIDIA (p. 13).</td>
            <td>
              Khronos ratified and released OpenCL 1.0 on 9 December 2008, the year the book's own reference [9]
              gives, and the list of companies behind it is much longer. OpenCL is still maintained, at version 3.1.
            </td>
          </tr>
          <tr>
            <td>Reference [4] is the CUDA C++ Programming Guide, Release 13.0, 2025 (p. 18).</td>
            <td>
              NVIDIA has restructured the guide since. The old address is gone, section links from older material no
              longer resolve, and the 13.0 text is kept as an archive that the new guide links to.
            </td>
          </tr>
          <tr>
            <td>The 4th edition is dated 2022 (p. xvii); the copyright line of this one reads 2027 (p. 1).</td>
            <td>
              The publisher dates the 4th edition May 28, 2022. For the 5th its pages give two publication dates,
              February 27 and June 3, 2026. When you cite the book, give the edition, not the year.
            </td>
          </tr>
        </tbody>
      </table>

      <h2>Where the PDF looks wrong</h2>
      <p>
        This edition is new and the copy this course was built from has a few apparent slips in the Preface and
        chapter 1. None of them touches an argument. They are listed so that you are not thrown by them when you read
        the book alongside.
      </p>
      <ul data-list="errata">
        <li>
          The Preface says the book is organised "into four major parts" (p. xxii), then describes three and says the
          old fourth part is gone (p. xxiv). The chart on p. xxiii and §1.7 (p. 15) both show three. The publisher's
          pre-publication contents list had four, which is probably where the phrase comes from.
        </li>
        <li>
          §1.7 says radix sort "relies on the scan pattern covered in Chapter 14" (p. 16). Scan is chapter 11
          everywhere else, and chapter 14 is sort. The Preface puts it differently again: radix sort builds on the
          stable filter of chapter 12 (p. xx). The two agree only indirectly, because the stable filter uses scan.
        </li>
        <li>
          The text refers to "Fig. 1.1(a)" and "Fig. 1.1(b)" (pp. 4–5). The figure has no such labels, only CPU and
          GPU. A thin bar under the GPU's rows is unlabelled, and the "small cache memories" the text mentions could
          be that bar, the small blocks at the start of each row, or both. Our redrawing on the{' '}
          <a href={href('ch01', 'start')}>start page</a> labels its bands.
        </li>
        <li>
          In the chart of Fig. P.1 (p. xxiii), the line from chapter 9 to chapters 17 and 21 passes directly under the
          box of chapter 14. We read it as starting at chapter 9. The drawing is tight enough that an arrow from 14
          cannot be ruled out, and the explorer in <a href={href('ch01', '1.7')}>1.7</a> marks both arrows as
          uncertain.
        </li>
        <li>
          Chapter titles differ from place to place. Chapter 16 is "Wavefront Parallelism" in the Preface and "Dynamic
          Progr. and Wavefront Parallelism" in the chart. Chapter 23 is "Multi-GPU Programming" in the Preface and
          "Programming a Heterogeneous Cluster" in the chart. Chapter 22 is "Problem Decomposition" in the chart,
          while §1.7 describes it as an introduction to computational thinking. Chapter 16's example is one algorithm
          in the Preface and two in §1.7 (p. 17).
        </li>
        <li>
          The Preface's teaching plan says its first phase "consists of 14 chapters", which would be chapters 2 to 15,
          and then schedules chapters 7 through 14 for the last five weeks (p. xxiv). Chapter 15 is left without a
          week.
        </li>
        <li>
          Smaller things on pp. 16–17: "coverts" for converts, "filter patter", "divide-and-concur", and "leverage
          techniques scatter-to-gather transformations". On p. 10 the book says "many-core GPUs" once where the rest
          of the chapter says many-thread; it means the same thing.
        </li>
        <li>
          Not a slip, though it looks like one: 1,979 TFLOPS for 16-bit arithmetic next to 67 for 32-bit (p. 3). The
          chapter gives no explanation. The <a href={href('ch01', 'start')}>start page</a> does.
        </li>
      </ul>
      <Aside>
        <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
      </Aside>

      <h2>What this chapter left out</h2>
      <p>
        This is a condensed chapter. It keeps the design argument, the speed-up arithmetic, the four difficulties and
        the map of the book. Here is the rest, and where to find it.
      </p>
      <ul data-list="omitted">
        <li>
          The end of single-CPU scaling (pp. 1–3): a few sentences on the start page. The "virtuous cycle", the von
          Neumann model and the program counter, and the "concurrency revolution": not taught.
        </li>
        <li>
          Installed base and form factor as reasons GPUs won, with the medical-imaging example; computing through
          graphics interfaces before CUDA; the G80; FPGAs as other accelerators (pp. 6–7). Not taught.
        </li>
        <li>
          §1.2, "Why more speed or parallelism?" (pp. 7–9): the tour of applications, from molecular biology and video
          coding to games, digital twins and the history of deep learning. Only its rules of thumb from p. 7 are kept,
          in section 1.3.
        </li>
        <li>
          The peach of Fig. 1.2 (pp. 10–11): a short paragraph in section 1.3 and no drawing. Its story of how much
          of the peach each interface could reach is dropped.
        </li>
        <li>
          §1.5 (pp. 12–13): how OpenMP's directives and pragmas work, MPI's domain decomposition, and the design of
          OpenCL. One paragraph in section 1.7. Programming at cluster level is also left out of this course's
          chapter 23.
        </li>
        <li>§1.6, the three goals (p. 14): one sentence each in section 1.7.</li>
        <li>
          The chapter-by-chapter previews of §1.7 (pp. 15–18): replaced by the reading-path explorer. The six-step
          recipe on p. 15 is only named; chapter 2 works through it.
        </li>
        <li>
          "How to use the book" in the Preface (pp. xxiv–xxvi): the history of the authors' course, its two teaching
          phases, the final project, the class competition, the teaching kit and the companion site. Also the
          acknowledgments (p. xxvii) and the references (p. 18).
        </li>
      </ul>

      <h2>The project so far</h2>
      <p>
        Nothing yet. Chapter 1 has no code, so it has no lab and no milestone of the through-line project, the small
        transformer you will end up running. The first milestone, M0, is chapter 2's: the elementwise operations.
        There was also nothing here to step through, break or measure. That starts in chapter 2 as well.
      </p>
      <p>
        One thing is worth doing now. From the repository root, run <code>python3 labs/pmpp.py doctor</code>. It lists
        the backends this course can build on your machine, says why any of them is skipped, and prints what each
        device reports about itself.{' '}
        <span data-devices>
          {devices.length
            ? `Lab results recorded on this machine so far come from ${and(devices.map((b) => `${NAME[b] ?? b} (${String(reports[b].device?.name ?? 'unnamed device')})`))}.`
            : 'No lab results from a GPU are recorded on this machine yet. The first will come from chapter 2.'}
        </span>{' '}
        {unified !== undefined ? (
          <span data-unified>
            {yes(unified) ? (
              <>
                Its Metal device reports <code>hasUnifiedMemory</code> as true: one memory for CPU and GPU, the case of
                the last question above.
              </>
            ) : (
              <>
                Its Metal device reports <code>hasUnifiedMemory</code> as false: the GPU has memory of its own, as in
                Fig. 1.1.
              </>
            )}
          </span>
        ) : null}
      </p>
      <p>
        <a href={href('ch02', 'start')}>Chapter 2</a> writes the first kernel. It turns host and device into code,
        works through the six steps that section 1.7 only named, and turns the rounding up of the two-wave question on
        the start page into the formula for the size of a launch. It is also where a machine with one memory for CPU
        and GPU first changes what you write. Its first lab times a one-thread CPU loop beside your kernel, so the
        baseline of every later
        speed-up is one you measured yourself.
      </p>
    </>
  );
}
