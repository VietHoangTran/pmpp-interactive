import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { DEFAULTS, simulate } from '../../sim/chunkpipe';

// The summary page shows one host sequence of its own, for a question. Its checked copy is the
// region [shown] of labs/ch23/snippets/rerecord.cu, which the CUDA front-end check parses;
// Wrap.snippets.test.ts fails if the two drift apart and pins the pipeline numbers below to
// the stepper's model (src/sim/chunkpipe.ts).
export const REREC = `produceA<<<grid, block, 0, s1>>>(x, n);      // x = 1
cudaEventRecord(done, s1);
cudaStreamWaitEvent(s2, done, 0);
produceB<<<grid, block, 0, s1>>>(x, n);      // x = x + 1
cudaEventRecord(done, s1);
consumeC<<<grid, block, 0, s2>>>(y, x, n);   // y = 2 * x`;

// q-pp: two pipeline stages on two GPUs with a link between them, 8 micro-batches. It is the
// stepper's three-lane pipeline with the lanes renamed (GPU 0 = copy-in, link = kernel,
// GPU 1 = copy-out); the model takes whole-job totals, so each per-micro-batch time is times 8.
export const PP = { micro: 8, gpu: 4, link: 1 };
const ppRun = simulate({
  ...DEFAULTS,
  a: PP.gpu * PP.micro,
  b: PP.link * PP.micro,
  c: PP.gpu * PP.micro,
  n: PP.micro,
  streams: PP.micro,
  engines: 2,
  order: 'depth',
  overhead: 0,
  host: 'pinned',
  queues: Infinity,
  view: 'cuda',
});
export const PP_RESULT = { total: ppRun.total, serial: ppRun.serial, bubble: ppRun.bubble, fill: 2 * PP.gpu + PP.link };

export const questionsWrap: Record<string, Question> = {
  'ch23/wrap/q-rerecord': {
    kind: 'choice',
    id: 'ch23/wrap/q-rerecord',
    prompt: (
      <>
        In the sequence above, <code>s1</code> and <code>s2</code> are two streams and <code>consumeC</code> is meant to
        read <code>x</code> after <code>produceB</code> has added to it. Is that guaranteed?
      </>
    ),
    options: [
      {
        label: 'Yes: s2 waits on done, and done was last recorded after produceB.',
        why: (
          <>
            The wait was issued on line 3. It took <code>done</code> as it stood then; the record on line 5 changes what
            later waits see, not this one.
          </>
        ),
      },
      { label: 'No: the wait took done as it stood on line 3, after produceA.', correct: true },
      {
        label: 'Yes: the host issued consumeC last, so it runs last.',
        why: 'Only inside one stream. Two streams have no order between them unless an event or a host wait gives them one.',
      },
      {
        label: 'No: recording one event twice is an error, so line 5 fails.',
        why: 'NVIDIA allows it: a new record overwrites what the event captured. Fig. 23.14 records its three events again in every iteration.',
      },
    ],
    explain: (
      <p>
        <code>cudaStreamWaitEvent</code> makes later work in <code>s2</code> wait for whatever <code>done</code> had
        captured when the wait was issued, which on line 3 is <code>produceA</code>. A later record is seen only by waits
        issued after it, and there are none. So <code>consumeC</code> may run while <code>produceB</code> is still adding to{' '}
        <code>x</code>, and different elements of <code>y</code> may come out 2 or 4. Issue the wait after the record it
        needs. The book&rsquo;s loops do: each iteration records, then waits. A Metal event works differently again: a
        counter that a wait asks for by value.
      </p>
    ),
    source: "our sequence; NVIDIA's runtime reference in notes/today/ch23-facts.md, A5 (Metal: C2); events as in §23.3, p. 560",
  },
  'ch23/wrap/q-decode': {
    kind: 'choice',
    id: 'ch23/wrap/q-decode',
    prompt: (
      <>
        A decode loop makes one token per step. The host launches the step&rsquo;s kernels; <code>cudaMemcpy</code> copies
        the sampled token id into a host <code>int</code>; the host checks it for the end of the sequence and launches the
        next step. A trace shows the GPU idle between steps. Which change keeps it busy?
      </>
    ),
    options: [
      {
        label: 'Launch the sampling kernel in a stream of higher priority.',
        why: 'While the host waits for the copy nothing else is queued, so a priority, a hint about pending kernels, has nothing to prefer.',
      },
      { label: 'Queue step n + 1 before reading token n, and check for the end of the sequence one step late.', correct: true },
      {
        label: 'Switch to cudaMemcpyAsync into the same int, then synchronise right after it, before the next launch.',
        why: 'The host still waits for the copy before it queues anything, and from pageable memory the copy may not be asynchronous at all.',
      },
      {
        label: "Spread each step's kernels over more streams so that they overlap.",
        why: "Each kernel of a step needs the one before it, so little can run side by side, and the gap between steps stays.",
      },
    ],
    explain: (
      <p>
        The idle time is a round trip: the GPU finishes, the copy lands, the host reads the token, and only then is the
        next step queued. If step n + 1&rsquo;s kernels read the token from device memory, the host does not need its
        value to launch them. Queue them first, copy the token into pinned memory with <code>cudaMemcpyAsync</code>, and
        wait only where you read it. If the token ended the sequence, one extra step is thrown away. It is the
        chapter&rsquo;s norm copy, queued early and waited for once, applied to decoding; mlx-lm&rsquo;s generate loop
        does the same on Metal, queueing the next step with <code>async_eval</code> before it reads the current token.
      </p>
    ),
    source: 'the norm copy of §23.3, pp. 560–561; notes/today/ch23-facts.md, A2, A6 and D6 (mlx-lm); the mapping to decoding is ours',
  },
  'ch23/wrap/q-pp': {
    kind: 'numeric',
    id: 'ch23/wrap/q-pp',
    prompt: (
      <>
        Two GPUs each hold half of a model&rsquo;s layers, and a batch is cut into {PP.micro} micro-batches. Each
        micro-batch takes {PP.gpu} time units on GPU 0, {PP.link} unit to cross the link, then {PP.gpu} units on GPU 1.
        Each GPU, and the link, handles one micro-batch at a time, in order. When does the last micro-batch leave GPU 1?
      </>
    ),
    answer: PP_RESULT.total,
    tol: 0,
    unit: 'units',
    explain: (
      <p>
        It is section 23.3&rsquo;s pipeline with the lanes renamed: GPU 0 is the copy in, the link the kernel, GPU 1 the
        copy out. The first micro-batch needs {PP.gpu} + {PP.link} + {PP.gpu} = {PP_RESULT.fill} units to get through;
        after it the slowest stage releases one every {PP.gpu}: {PP_RESULT.fill} + {PP.micro - 1} × {PP.gpu} ={' '}
        {PP_RESULT.total}, against {PP_RESULT.serial} one after another. The link is hidden because it is faster than
        either GPU. Each GPU still idles {PP_RESULT.bubble} units while the pipeline fills or drains, however many
        micro-batches there are: more of them shrink that bubble&rsquo;s share, not its size.
      </p>
    ),
    source: "the stepper's model (ours, notes/today/ch23-facts.md, E1) on the overlap of §23.3, pp. 553–555; \"bubble\" as PyTorch's pipelining page uses it, D5",
  },
  'ch23/wrap/q-ex2': {
    kind: 'numeric',
    id: 'ch23/wrap/q-ex2',
    prompt: (
      <>
        The call <code>MPI_Send(ptr_a, 1000, MPI_FLOAT, 2000, 4, MPI_COMM_WORLD)</code> moved 4,000
        bytes. What is the size of each element?
      </>
    ),
    answer: 4,
    tol: 0,
    unit: 'bytes',
    explain: (
      <p>
        The count is the second argument, 1,000; 2,000 is the destination rank and 4 the tag. 4,000 / 1,000 = 4 bytes,
        the book&rsquo;s option (c). An <code>MPI_FLOAT</code> has the size of a <code>float</code> on the host.
      </p>
    ),
    source: 'ch. 23 §23.7, exercise 2, p. 576; parameters from Fig. 23.8 and the datatype sizes, p. 550',
  },
  'ch23/wrap/q-ex3': {
    kind: 'choice',
    id: 'ch23/wrap/q-ex3',
    prompt: (
      <>
        Which is true? (a) <code>MPI_Send()</code> is blocking by default. (b) <code>MPI_Recv()</code> is
        blocking by default. (c) MPI messages must be at least 128 bytes.
      </>
    ),
    options: [
      {
        label: 'Only (a), about MPI_Send.',
        why: (
          <>
            True, but <code>MPI_Recv</code> is in the same position: the book offers <code>MPI_Irecv</code> as its
            non-blocking counterpart.
          </>
        ),
      },
      {
        label: 'Only (b), about MPI_Recv.',
        why: (
          <>
            True, but <code>MPI_Send</code> is in the same position: the book offers <code>MPI_Isend</code> as its
            non-blocking counterpart.
          </>
        ),
      },
      {
        label: 'Only (c), the 128-byte minimum.',
        why: 'Nothing in the chapter sets a minimum, and its norm reduction sends a single 4-byte float.',
      },
      { label: 'Both (a) and (b); (c) is false.', correct: true },
    ],
    explain: (
      <p>
        The chapter never says it in so many words, but it presents <code>MPI_Isend</code> and <code>MPI_Irecv</code>,
        followed by <code>MPI_Wait</code>, as the non-blocking versions of the two calls (p. 551), and it calls{' '}
        <code>MPI_Sendrecv</code> blocking (p. 567). So (a) and (b) are both true and (c) is false. The exercise asks for
        one statement; we read it as having two right answers and say so, rather than pick one.
      </p>
    ),
    source: 'ch. 23 §23.7, exercise 3, p. 576; §23.2, pp. 551, 553; §23.4, p. 567',
  },
};

const NAME: Record<string, string> = { metal: 'Metal', cuda: 'CUDA', hip: 'HIP', emu: 'the emulator' };

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const q = questionsWrap;
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch23/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const backends = realBackends(reports);
  const tried = [...backends, ...(reports.emu ? ['emu'] : [])];
  const passing = tried.filter((b) => labStatus(reports[b], 'ch23/lab1_overlap', 'work') === 'pass');
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <p>
        The book uses one stencil to teach three ways of spreading work over many GPUs. What this course keeps is the part
        that works on one GPU, and the habit of asking, for every wait, who is waiting and for what.
      </p>
      <Source section="23.6" pages="574–575" />
      <ul data-carry>
        <li>
          A launch, an asynchronous copy and an NCCL call return once the work is queued. That the work is done is
          something only a synchronise or an event can tell you.
        </li>
        <li>
          A stream is an ordered queue; work in different streams may overlap. To hide copies, cut the job into chunks
          over several streams: the total becomes the time to fill the pipeline plus one chunk&rsquo;s slowest stage per
          further chunk. Smaller chunks shrink the fill-and-drain bubble, but every chunk adds fixed costs, so there is a
          best chunk count to measure.
        </li>
        <li>
          An asynchronous copy overlaps only from pinned host memory. From pageable memory it is legal, and it quietly
          stops being asynchronous.
        </li>
        <li>
          Events order work across streams without the host. A wait takes the event as it was recorded at that moment.
        </li>
        <li>
          A stream priority is a hint for kernels still waiting to start. Issue urgent work first, and measure. ROCm
          ignores priorities by default, and Metal has none.
        </li>
        <li>
          The waiting can move. With MPI the host waits for every exchange; NCCL queues the exchange in a stream;
          NVSHMEM sends from inside the kernel and needs a barrier to know the data has arrived.
        </li>
        <li>
          On Apple silicon there is no copy to hide. The overlap there is the CPU filling and draining shared buffers
          while the GPU computes, with several buffers in flight.
        </li>
      </ul>

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the pages did not show, followed by the two of the book&rsquo;s
        exercises that no other page works.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Beyond>
        <p>
          The event sequence, the decode loop and the two-GPU pipeline are ours. The sequence is front-end-checked and has
          never been run.
        </p>
      </Beyond>
      <Code
        tabs={[
          {
            label: 'CUDA (not a lab file: for this question only)',
            code: REREC,
            status: 'front-end-checked; never run',
            tiers: { x: 'global', y: 'global' },
          },
        ]}
      />
      <Ask q={q['ch23/wrap/q-rerecord']} label="Read the code" />
      <Ask q={q['ch23/wrap/q-decode']} />
      <Ask q={q['ch23/wrap/q-pp']} label="Work it out" />
      <Ask q={q['ch23/wrap/q-ex2']} label="Book exercise 2" />
      <Ask q={q['ch23/wrap/q-ex3']} label="Book exercise 3" />

      <h2>The book&rsquo;s exercises</h2>
      <p>
        The chapter ends with three exercises (p. 576). The book prints no answers, so the answers here are ours. Two of
        them have more than one defensible answer, and the pages say so instead of picking one.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr>
            <th>Exercise</th>
            <th>About</th>
            <th>Where it is worked here</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>1a–1d</td>
            <td>Points computed, halo points, boundary and internal points per process, for a 64 × 512 grid on 16 ranks</td>
            <td>
              the explorer on the <a href={href('ch23', 'start')}>start page</a>, preset &ldquo;Exercise 1&rdquo;, which
              shows both readings: with and without the two edge columns the kernel never writes
            </td>
          </tr>
          <tr>
            <td>1e</td>
            <td>Bytes each process sends in stage 2</td>
            <td>
              the question &ldquo;Book exercise 1e&rdquo; on the <a href={href('ch23', 'start')}>start page</a>, and the
              prediction after it for 32 GPUs
            </td>
          </tr>
          <tr>
            <td>2</td>
            <td>Element size from an <code>MPI_Send</code> call and its byte count</td>
            <td>the question &ldquo;Book exercise 2&rdquo; above</td>
          </tr>
          <tr>
            <td>3</td>
            <td>Which MPI statement is true</td>
            <td>the question &ldquo;Book exercise 3&rdquo; above, with both true statements</td>
          </tr>
        </tbody>
      </table>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy this course was built from has a number of apparent slips in chapter 23. Most are small; the line-19
        parenthesis would stop the code compiling.
      </p>
      <ul data-list="errata">
        <li>p. 546 says Fig. 23.5 shows five essential MPI functions; it shows four.</li>
        <li>
          Fig. 23.14 line 19 (p. 557) has an extra closing parenthesis. Page <a href={href('ch23', '23.4')}>23.3–23.5</a> shows
          the corrected line, and its unit test proves the printed one does not compile.
        </li>
        <li>p. 559 places the wait on the bottom stream at line 32; in the figure it is line 31, as p. 564 says.</li>
        <li>
          pp. 561–562 speak of &ldquo;the cudaMemset operation&rdquo; while the code and Fig. 23.15 concern{' '}
          <code>cudaMemsetAsync</code>.
        </li>
        <li>
          Identifiers drift: <code>l2Norm</code> (p. 561) for <code>l2norm</code>, <code>cudaStreamsynchronize</code> (p.
          562), <code>nccl_comm</code> in Fig. 23.20 for <code>ncclComm</code> in Fig. 23.17, and <code>PE</code> and{' '}
          <code>numPEs</code> in Fig. 23.25, never defined.
        </li>
        <li>
          <code>launch_jacobi_kernel</code> and <code>blockReduce</code> are used and never defined, and no launch
          dimensions are given. The course&rsquo;s snippets declare them.
        </li>
        <li>p. 562 calls the host-synchronisation arrows of Fig. 23.15 thick dotted; the legend draws them dashed.</li>
        <li>
          The sentence on periodic boundaries (p. 551) can be read either way. The code wraps around; Figs. 23.2, 23.3 and
          23.12 draw GPU 0 with no top neighbour.
        </li>
        <li>
          Exercise 1 calls the kernel a 5-point stencil; Fig. 23.1 averages the four neighbours and reads the centre only
          for the residue.
        </li>
        <li>
          The kernel accumulates the sum of squared residues in a variable called <code>l2norm</code>, and the text
          alternates between the norm and its square (p. 553).
        </li>
        <li>Every host loop tests <code>l2norm &gt; tol</code> before the first iteration; the starting value is never shown.</li>
        <li>
          Fig. 23.13(b) starts the bottom exchange where the top exchange ends, which is what a blocking MPI call does, but
          the figure comes before streams and does not say which version it draws.
        </li>
        <li>Exercise 1 does not say whether edge columns count; exercise 3 has two true statements and asks for one.</li>
        <li>
          Wording, as printed: &ldquo;when may be interested&rdquo; (p. 558), &ldquo;responsible these rows&rdquo; (p.
          544), &ldquo;intialize&rdquo; (p. 566), &ldquo;calls nvshmem_finalize releases&rdquo; (p. 572), and a missing full
          stop after &ldquo;[1]&rdquo; (p. 551).
        </li>
        <li>
          Not an error: which way priorities run. The book does not say; NVIDIA does, a lower number is a higher priority,
          and the book&rsquo;s <code>low, high</code> order is right.
        </li>
      </ul>
      <Aside>
        <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
      </Aside>
      <p>
        Some of what the chapter says has changed since it was written, or was never quite so. Each item was checked
        against current vendor documentation.
      </p>
      <Today checked="2026-11-20 (notes/today/ch23-facts.md A1 to A16, B2, B3)">
        <p>
          ROCm 10.2 accepts stream priorities and ignores them by default, and its <code>asyncEngineCount</code> is a
          constant, the same on every device. The book says nothing about AMD.
        </p>
      </Today>
      <ul data-list="dated">
        <li>
          &ldquo;<code>cudaMemset</code> is blocking&rdquo; (p. 559) is wrong for device memory: it does not make the host
          wait. The reason to use <code>cudaMemsetAsync</code> is that the plain call is queued in the legacy default
          stream, which waits for every blocking stream and makes them wait for it.
        </li>
        <li>
          &ldquo;<code>cudaMemcpyAsync()</code> requires&rdquo; pinned memory (p. 563): NVIDIA&rsquo;s reference says a
          copy from pageable memory still works but reverts to synchronous behaviour. On systems where the GPU reads host
          page tables, <code>cudaMallocHost</code> may not page-lock at all.
        </li>
        <li>
          Priorities as the fix for a kernel that occupies the GPU (p. 558): NVIDIA calls a priority a hint that does not
          preempt running work, guarantees no order and does not touch copies.
        </li>
        <li>
          &ldquo;Most MPI implementations ... are built with&rdquo; CUDA awareness &ldquo;enabled&rdquo; (p. 552): only
          MPICH turns it on by itself, when it finds CUDA at configure time; Open MPI needs to be built for it.
          MVAPICH2&rsquo;s GPU line is now MVAPICH-Plus. NCCL&rsquo;s documentation warns that using NCCL and CUDA-aware
          MPI on the same GPUs at once can deadlock.
        </li>
        <li>
          NCCL &ldquo;may add support for one-sided communication soon&rdquo; (p. 575): it already has, since version
          2.29, beside symmetric windows (2.27) and a device-side API (2.28).
        </li>
        <li>
          Setup the figures leave out: <code>ncclCommInitRank</code> needs the CUDA device set first (Fig. 23.17), and
          NVSHMEM recommends initialising <code>nvshmemx_init_attr_t</code> with{' '}
          <code>NVSHMEMX_INIT_ATTR_INITIALIZER</code> (Fig. 23.24).
        </li>
        <li>
          References (p. 576): the CUB link no longer resolves (the documentation moved under <code>/cccl/unstable/</code>
          ), and the NVSHMEM link redirects. NVIDIA&rsquo;s &ldquo;Multi GPU programming models&rdquo; repository, the
          book&rsquo;s own code for these loops, is live.
        </li>
      </ul>

      <h2>What this chapter left out</h2>
      <p>
        This chapter is condensed. It keeps what works on one GPU and leaves the cluster plumbing in the book. Nothing
        here runs MPI, NCCL or NVSHMEM.
      </p>
      <ul data-left>
        <li>
          The Jacobi kernel itself is described, not taught (Fig. 23.1, pp. 542–543). Its block reduction and atomic commit
          are chapter 10&rsquo;s.
        </li>
        <li>
          The 1D against 2D decomposition beyond one switch of the <a href={href('ch23', 'start')}>start page</a>
          &rsquo;s explorer (p. 545).
        </li>
        <li>
          MPI in full (section 23.2, pp. 545–553, Figs. 23.4 to 23.11): shared against distributed memory, processes,
          ranks and communicators, <code>MPI_Init</code> and its companions, <code>MPI_Send</code>, <code>MPI_Recv</code>{' '}
          and <code>MPI_Sendrecv</code> with their parameters, <code>MPI_Isend</code> and <code>MPI_Irecv</code>, periodic
          boundaries, CUDA-aware MPI and <code>MPI_Allreduce</code>. One paragraph on page{' '}
          <a href={href('ch23', '23.4')}>23.3–23.5</a> and the first loop in words on the start page stand in for it.
        </li>
        <li>
          NCCL&rsquo;s setup, its send and receive calls, group calls and the full NCCL loop (section 23.4, pp. 563–568,
          Figs. 23.16 to 23.21), kept as one paragraph, the exchange lines, a column of the comparison table and a beat
          of the loop animation on page 23.3–23.5. The explorer there models only the two MPI loops.
        </li>
        <li>
          NVSHMEM&rsquo;s symmetric heap, its in-kernel puts and gets, its initialisation with MPI, the on-stream barrier,
          kernel fusion and fine- against coarse-grained puts (section 23.5, pp. 568–574, Figs. 23.22 to 23.25), kept as
          one paragraph, a column of the same table and the same beat.
        </li>
        <li>The API signature boxes (Figs. 23.5, 23.8 to 23.11, 23.16, 23.18, 23.19 and 23.22) and the references (p. 576).</li>
      </ul>
      <p>
        If you need the cluster versions, NVIDIA&rsquo;s &ldquo;Multi GPU programming models&rdquo; repository, the
        book&rsquo;s reference 4, holds runnable code for each loop.
      </p>

      <h2>The project so far</h2>
      <p data-project={passing.join(',') || 'none'}>
        Chapter 23 adds no milestone. The course project runs on one GPU and never splits a model across devices, and lab
        23.1 is optional.{' '}
        {tried.length === 0
          ? 'No lab results are recorded on this machine yet.'
          : passing.length
            ? `Your own runStaged or runInFlight passes on ${passing.map((b) => NAME[b] ?? b).join(' and ')} on this machine.`
            : 'Your own version of lab 23.1 has not passed on this machine yet.'}{' '}
        See the <a href={href('ch23', 'labs')}>lab</a>.
      </p>
      <p>
        What does carry into the project is the habit from page <a href={href('ch23', '23.3')}>23.3</a>: when your model
        copies a token back every step, queue the copy early, keep the host buffer pinned, and wait only where the value
        is needed.
      </p>

      <h2>Where next</h2>
      <p>
        <a href={href('ch24', 'start')}>Chapter 24</a> closes the book. It has no code: it places every chapter in the
        book&rsquo;s three steps and sets one table of 2007 hardware against 2025, including the host link whose cost
        this chapter has been hiding.
      </p>
    </>
  );
}
