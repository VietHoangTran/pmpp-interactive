import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { siteLine } from '../../sim/bfspriv';
import { gridSummary } from '../../sim/gridbfs';
import { region } from '../../sim/histpriv';
import { functionSource } from '../../sim/toolchain';
import { GridBarrierStepper } from './GridBarrierStepper';
import { OneLaunchScene } from './OneLaunchScene';
import { PRIV_TIERS, PrivateQueueStepper } from './PrivateQueueStepper';

// Section 18.6 of this course: the book's §18.6 to §18.8 (pp. 442-449). What the book says
// comes from notes/ch18.md; vendor facts from notes/today/ch18-facts.md (A1-A5, A7, A8, A10,
// A11, A13, B1-B6, C1-C4, C6, D1-D5, E1). Models: src/sim/bfspriv.ts (the private-queue
// stepper) and src/sim/gridbfs.ts (the grid-barrier stepper, "Does it fit?", the prediction,
// the scene). Listings: labs/ch18/snippets/s186_privatised_variants.cu and
// labs/ch18/snippets/s186_coop_swap.cu, loaded at run time; S186.snippets.test.ts ties the
// code this page quotes in its prose to those files.

const GRID = gridSummary(1000);
const fmt = (n: number) => n.toLocaleString('en-US');

export const questionsS186: Record<string, Question> = {
  'ch18/18.6/q-atomics': {
    kind: 'numeric',
    id: 'ch18/18.6/q-atomics',
    prompt: (
      <>
        One level appends 3,000 vertices from 24 blocks, and no block's private queue overflows. Fig. 18.12 makes 3,000
        atomic operations on the global counter, one per vertex. How many does Fig. 18.15 make?
      </>
    ),
    answer: 24,
    unit: 'atomics',
    explain: (
      <p>
        One per block: thread 0's reservation of the block's whole range. The 3,000 appends still happen, but on 24
        private counters in shared memory, each contended only by its own block. An overflow would add one global atomic
        per entry that does not fit.
      </p>
    ),
    source: 'ch. 18 §18.6, p. 442; src/sim/bfspriv.ts (levelAtomics)',
  },
  'ch18/18.6/q-overflow': {
    kind: 'numeric',
    id: 'ch18/18.6/q-overflow',
    prompt: (
      <>
        Capacity 8, with the corrected counter reset. A block of 4 threads discovers 13 new vertices in one level. How
        many of them go straight to the public frontier?
      </>
    ),
    answer: 5,
    unit: 'entries',
    explain: (
      <p>
        The first 8 increments return 0 to 7 and get private slots. The other 5 return 8 or more, put the counter back
        to 8 and append to the public frontier with a device-scope atomic. The hub preset of the stepper with capacity
        8 and 1 block of 4 threads is this case.
      </p>
    ),
    source: 'ch. 18 §18.6, pp. 443–444',
  },
  'ch18/18.6/q-overflow-counter': {
    kind: 'numeric',
    id: 'ch18/18.6/q-overflow-counter',
    prompt: (
      <>
        The same block: capacity 8, 13 new vertices, the corrected counter reset. What does the private counter{' '}
        <code>numCurrFrontier_s</code> hold after barrier 2, when thread 0 reads it to reserve public slots?
      </>
    ),
    answer: 8,
    unit: 'entries',
    explain: (
      <p>
        Every increment past capacity is followed by that thread's store of the capacity, so whatever the interleaving
        the counter ends at min(13, 8) = 8: exactly the number of private entries to copy out. Thirteen would reserve 5
        public slots that nobody fills. The "whatever the interleaving" is our own check, by enumerating every
        interleaving, not a vendor's.
      </p>
    ),
    source: 'ch. 18 §18.6, pp. 443–444; notes/today/ch18-facts.md E1 (ours)',
  },
  'ch18/18.6/q-line29': {
    kind: 'choice',
    id: 'ch18/18.6/q-line29',
    prompt: (
      <>
        The book's <code>numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;</code> gives the right counts in every test you
        run. Why change it?
      </>
    ),
    options: [
      {
        label: 'It races with other threads\' fetch_add on the same counter: a data race, undefined under the memory model of pp. 433–434.',
        correct: true,
      },
      {
        label: 'It loses increments: a thread that adds between another\'s fetch_add and its store is wiped out, so the count comes out low.',
        why: 'Only a thread at or past capacity stores, and it stores the capacity. Every interleaving ends at min(N, capacity), which is the right number. The defect is the race itself.',
      },
      {
        label: 'It is slower than an atomic store, because the plain store forces the counter out of shared memory.',
        why: 'Speed is not the issue; a plain store is no slower, and the counter stays in shared memory. The issue is that it is undefined behaviour.',
      },
      {
        label: 'Nothing needs to change: the store is benign, because every thread that reaches it writes the same capacity.',
        why: 'The book itself says idempotent races still violate the memory model (pp. 433–434), and here the store also races with read-modify-writes, not only with stores of the same value.',
      },
    ],
    explain: (
      <p>
        A race is undefined whether or not the numbers come out right on the hardware you test. The stepper's race
        panel reports this line against the <code>fetch_add</code>, and the course's race backend does whenever two
        threads of a block overflow at the same time. On Metal the
        counter is a <code>threadgroup atomic_uint</code>, which cannot be assigned plainly, so only the atomic form can
        be written.
      </p>
    ),
    source: 'ch. 18 §18.6, p. 444; facts A7, C4, E1',
  },
  'ch18/18.6/q-two-runs': {
    kind: 'choice',
    id: 'ch18/18.6/q-two-runs',
    prompt: <>Two runs of Fig. 18.15 on the same graph produce different public frontiers. What can differ between them?</>,
    options: [
      {
        label: 'Where each block\'s range sits, the order inside it, and where overflow entries land; not the set of vertices or its size.',
        correct: true,
      },
      {
        label: 'Nothing should differ: the blocks reserve their ranges in block order, so a difference means a bug.',
        why: 'Atomics on one address are served in no fixed order, so blocks reserve in whatever order they arrive. An unstable filter is allowed to come out in any order.',
      },
      {
        label: 'The count: which entries overflow depends on timing, so a vertex can be appended twice, once through its private queue and once straight to the public frontier.',
        why: 'Which entries overflow depends on timing; how many vertices are appended does not. Each new vertex takes one path, private or public, and the compare-and-swap appends it exactly once.',
      },
    ],
    explain: (
      <p>
        Fig. 18.16 draws exactly this: each block's entries in one contiguous range, the ranges in reservation order.
        In the stepper, change the block order and compare the public frontier: same set, different arrangement.
      </p>
    ),
    source: 'ch. 18 §18.6, p. 444, Fig. 18.16; notes/today/ch18-facts.md "Already settled" (ch09 A3, B1)',
  },
  'ch18/18.6/q-commit-if': {
    kind: 'choice',
    id: 'ch18/18.6/q-commit-if',
    prompt: (
      <>
        A colleague commits with{' '}
        <code>if (threadIdx.x &lt; numCurrFrontier_s) currFrontier[start + threadIdx.x] = currFrontier_s[threadIdx.x];</code>.
        It passes every test with capacity 64 and 128 threads per block, and fails with capacity 256. Why?
      </>
    ),
    options: [
      {
        label: 'One entry per thread: entries from index blockDim.x up are never copied, and capacity 64 has none.',
        correct: true,
      },
      {
        label: 'Capacity 256 overflows the shared memory of a block, so the launch fails or the private queue wraps around.',
        why: '256 words is 1 KB; the failure is lost entries, not a launch error or a wrap.',
      },
      {
        label: 'The commit is no longer coalesced: with 256 entries, neighbouring threads write addresses far apart.',
        why: 'They are still consecutive. The problem is correctness: slots reserved for entries 128 to 255 are never written.',
      },
    ],
    explain: (
      <p>
        The strided loop (index <code>threadIdx.x + k·blockDim.x</code>) lets a block copy a queue of any length. Try it in
        the stepper: commit with an <code>if</code>, 2 threads per block and capacity 4.
      </p>
    ),
    source: 'ch. 18 §18.6, p. 444; src/sim/bfspriv.ts',
  },
  'ch18/18.6/q-toolarge': {
    kind: 'choice',
    id: 'ch18/18.6/q-toolarge',
    prompt: (
      <>
        A device has S SMs, and the occupancy query says k blocks per SM. You request S·k + 1 blocks through the
        cooperative launch. What happens on CUDA, and on HIP?
      </>
    ),
    options: [
      {
        label: 'Both refuse it at launch: cudaErrorCooperativeLaunchTooLarge and hipErrorCooperativeLaunchTooLarge.',
        correct: true,
      },
      {
        label: 'It launches, and the first grid.sync() deadlocks: the extra block waits for a slot that never frees.',
        why: 'That is the risk the book describes (p. 445), but the cooperative API checks the grid against the same product before launching anything.',
      },
      {
        label: 'CUDA refuses it with cudaErrorCooperativeLaunchTooLarge; HIP launches it and deadlocks.',
        why: 'HIP\'s runtime compares the grid with its own occupancy-based limit and returns hipErrorCooperativeLaunchTooLarge (our reading of its source).',
      },
    ],
    explain: (
      <p>
        The deadlock of p. 445 is what you would risk only by going around the cooperative API. Check the return code
        of the launch: it is the only report you get.
      </p>
    ),
    source: 'ch. 18 §18.7, pp. 445–446; notes/today/ch18-facts.md A3, B3',
  },
  'ch18/18.6/q-barrier2': {
    kind: 'choice',
    id: 'ch18/18.6/q-barrier2',
    prompt: <>Remove the second <code>grid.sync()</code> of the frontier swap. What can go wrong?</>,
    options: [
      {
        label: 'Thread 0 may reset the counter before another thread reads it: that thread reads 0, leaves the loop, and the rest hang at the next grid.sync().',
        correct: true,
      },
      {
        label: 'Appends of the next level may reach the counter before thread 0 resets it, so the reset wipes them out and those vertices are never expanded at all.',
        why: 'That is what the third barrier prevents.',
      },
      {
        label: 'A thread may read the counter before every block has finished appending, and get a partial count.',
        why: 'That is the first barrier\'s job.',
      },
      {
        label: 'Nothing: every thread already holds the count, so the second barrier only orders work for speed.',
        why: 'Each of the three guards a different ordering. Untick it in the stepper, in order.',
      },
    ],
    explain: (
      <p>
        In the stepper, untick grid.sync() 2 and run in order (thread 0 first): thread 0 reads, resets, and the next
        reader sees 0. A thread that leaves the kernel never reaches the next barrier, and a barrier waits for every
        thread of the grid.
      </p>
    ),
    source: 'ch. 18 §18.7, pp. 447–448; src/sim/gridbfs.ts',
  },
  'ch18/18.6/q-ex3': {
    kind: 'choice',
    id: 'ch18/18.6/q-ex3',
    prompt: (
      <>
        Exercise 3: give each level its own counter, all zeroed before the launch. How many grid barriers does each
        level need, and where?
      </>
    ),
    options: [
      {
        label: 'One, after the appends and before reading counter[L]. Nothing is reset, so no read can race with a reset.',
        correct: true,
      },
      {
        label: 'Two: one after the appends, before reading counter[L], and one after the read, before the next appends.',
        why: 'The barrier after the read protected the reset. With no reset there is nothing for it to protect.',
      },
      {
        label: 'None: each level has its own counter, so no read can meet a write.',
        why: 'A thread must still not read counter[L] before every thread has appended to it.',
      },
    ],
    explain: (
      <p>
        The price is a bound on the number of levels: the counters are allocated before the launch, one per level the
        search may reach. The book calls this a more general form of double buffering (p. 448). The stepper's Exercise 3
        kernel is correct in every order with its one barrier, and wrong without it.
      </p>
    ),
    source: 'ch. 18 §18.7, p. 448; Exercise 3, p. 450; src/sim/gridbfs.ts',
  },
  'ch18/18.6/q-hip-launch': {
    kind: 'choice',
    id: 'ch18/18.6/q-hip-launch',
    prompt: (
      <>
        The book's <code>cudaLaunchCooperativeKernel((void*)bfs_kernel, numBlocks, numThreadsPerBlock, kernelArgs);</code>{' '}
        compiles for CUDA and not for HIP. Which single line works for both through the course's header?
      </>
    ),
    options: [
      {
        label: 'The same call with ", 0, 0" added for the shared-memory size and the stream.',
        correct: true,
      },
      {
        label: 'The same call with hip in place of cuda: the header renames it.',
        why: 'The header does rename it, but hipLaunchCooperativeKernel has no default arguments, so four arguments still do not compile.',
      },
      {
        label: 'hipLaunchCooperativeKernel(bfs_kernel, numBlocks, numThreadsPerBlock, 0, 0, 0), as in AMD\'s example.',
        why: 'That passes 0 as the argument array of a kernel that has six parameters. AMD\'s own example has this slip; do not copy it.',
      },
    ],
    explain: (
      <p>
        On CUDA the four-argument call resolves to a C++ overload whose last two parameters default to 0, so the
        six-argument form is valid there too. llama.cpp passes all six for the same reason.
      </p>
    ),
    source: 'notes/today/ch18-facts.md A1, B1, D2',
  },
  'ch18/18.6/q-metal-plan': {
    kind: 'choice',
    id: 'ch18/18.6/q-metal-plan',
    prompt: <>You port the single-launch BFS to Metal. Which plan works?</>,
    options: [
      {
        label: 'One dispatch per level on a serial compute encoder of an MTLCommandQueue, which orders them and tracks hazards, each sized by an indirect dispatch from a GPU-written count.',
        correct: true,
      },
      {
        label: 'One dispatch, with a threadgroup barrier at the end of each level so every thread sees the new frontier.',
        why: 'A threadgroup barrier synchronises one threadgroup, not the grid. Metal has no grid-wide barrier.',
      },
      {
        label: 'One dispatch sized from the occupancy query, every threadgroup spinning on a device atomic between levels.',
        why: 'Metal has no occupancy query and documents no forward progress between threadgroups, so a spin barrier may never be released.',
      },
      {
        label: 'One dispatch per level on a concurrent encoder, with no barriers between them, so the levels can overlap.',
        why: 'On a concurrent encoder the dispatches may overlap, and you are responsible for ordering them, with memoryBarrier(scope:).',
      },
    ],
    explain: (
      <p>
        <code>dispatchThreadgroups(indirectBuffer:...)</code> reads the threadgroup count just before the grid starts,
        so the CPU never reads the frontier size. A count of 0 is not documented: write at least 1 and let the kernel
        return early on an empty frontier. The ordering holds on <code>MTLCommandQueue</code>, not on{' '}
        <code>MTL4CommandQueue</code>, which tracks no hazards.
      </p>
    ),
    source: 'notes/today/ch18-facts.md C1, C2, C3',
  },
  'ch18/18.6/q-hubs': {
    kind: 'choice',
    id: 'ch18/18.6/q-hubs',
    prompt: (
      <>
        A frontier holds 10,000 vertices of degree 2 and three of degree 50,000, one thread per vertex. What goes wrong,
        and what fixes it?
      </>
    ),
    options: [
      {
        label: 'Three threads work 25,000 times longer than the rest; give each hub a block, or a thread per edge.',
        correct: true,
      },
      {
        label: 'Nothing: the 10,000 small vertices keep the device busy while the three hub threads work through their lists.',
        why: 'The level ends when its slowest thread ends. The small vertices finish almost at once, and then three threads work alone.',
      },
      {
        label: 'The hubs\' 150,000 edges overflow the private queues; make each queue large enough to hold all they find.',
        why: 'The overflow path handles that already. The problem is the time one thread spends walking 50,000 edges.',
      },
    ],
    explain: (
      <p>
        That is degree bucketing (thread, warp or block by degree) or the edge-centric mapping of §18.4 applied to the
        heavy vertices only, which is what cuGraph's high-degree segment does.
      </p>
    ),
    source: 'ch. 18 §18.8, pp. 448–449; notes/today/ch18-facts.md D1',
  },
};

/** The three forms of the overflow branch, from the snippet file, with the reset line marked. */
function ResetForms() {
  const src = useLabFile('ch18/snippets/s186_privatised_variants.cu');
  if (!src) return null;
  // rows: how many source lines the marked statement spans (the atomic store and the clamp take two).
  const tab = (label: string, kernel: string, site: 'reset' | 'clamp', rows: number, status: string) => {
    const f = functionSource(src, kernel);
    const text = f ? f.text : `// ${kernel} not found`;
    const line = f ? siteLine(text, site) : -1;
    const highlight = line >= 0 ? Array.from({ length: rows }, (_, k) => line + 1 + k) : [];
    return { label, code: text, tiers: PRIV_TIERS, highlight, status };
  };
  const checked = 'emulator-checked (emu and race) and front-end-checked; unverified on CUDA and HIP hardware';
  return (
    <Code
      tabs={[
        tab('corrected: atomic store', 'bfsPrivatised', 'reset', 2, checked),
        tab('corrected: clamp', 'bfsPrivatisedClamp', 'clamp', 2, checked),
        tab("the book's line 29", 'bfsPlainStore', 'reset', 1, 'emulator-checked; the race emulator reports this line'),
      ]}
      caption={
        <>
          Fig. 18.15 three ways, from <code>labs/ch18/snippets/s186_privatised_variants.cu</code>. Marked: the line that
          differs. The capacity is a template parameter, so that one build can try 1 to 8.
        </>
      }
    />
  );
}

/** The host code of pp. 445-446 and the vendor-guarded include, from the cooperative snippet. */
function LaunchCode() {
  const src = useLabFile('ch18/snippets/s186_coop_swap.cu');
  if (!src) return null;
  const header = region(src, 'header');
  const launch = region(src, 'launch');
  const ex3 = functionSource(src, 'bfsCoopEx3');
  const status = 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware';
  return (
    <Code
      tabs={[
        { label: 'sizing and launch', code: launch ? launch.text : '// region launch not found', status },
        { label: 'include, CUDA and HIP', code: header ? header.text : '// region header not found', status },
        { label: 'Exercise 3 kernel', code: ex3 ? ex3.text : '// bfsCoopEx3 not found', tiers: PRIV_TIERS, status },
      ]}
      caption={
        <>
          From <code>labs/ch18/snippets/s186_coop_swap.cu</code>. The launch passes six arguments where the book passes
          four, so the same line compiles for HIP; the include is chosen by vendor. The full kernel, private queue and
          all, is the labs page's reference solution.
        </>
      }
    />
  );
}

export function S186() {
  return (
    <>
      <p>
        The frontier kernel of the last page has two costs left: one counter that every appending thread in the grid
        hits, and one launch, with a read-back, per level. This page removes the first with a queue private to each
        block and the second with a barrier across the whole grid, fixes one racy line in the book's code, and says
        what each backend can and cannot do.
      </p>

      <h2>The counter is the hot spot</h2>
      <p>
        In Fig. 18.12 every new vertex costs one <code>fetch_add</code> on <code>numCurrFrontier</code>, one address in
        global memory, and the book expects contention there to be high. Appending to a frontier is chapter 12's
        unstable filter, and chapter 12 had two cures for its counter: combine the atomics of a warp into one, and
        privatise. For the first, the book says you can assume the compiler combines them. Read that as "may".{' '}
        <a href={href('ch12', '12.3')}>Section 12.3</a> showed how to do it by hand: a ballot, a population count, one
        atomic per warp, a broadcast.
      </p>
      <Source section="18.6" pages="442" />
      <Today checked="2026-11-20 (ch18 facts A8, D1)">
        <p>
          NVIDIA states automatic warp aggregation only in a 2017 blog post. AMD's LLVM backend does it for
          wavefront-uniform addresses, such as this counter. Apple documents nothing. cuGraph's BFS aggregates by hand
          with a ballot.
        </p>
      </Today>
      <p>
        The second cure is this section. Each block keeps its own queue and counter in shared memory, as chapter 9's
        histogram kept private bins. Threads contend only with their own block, shared-memory atomics are cheaper, and
        the copy to the public frontier at the end can be coalesced.
      </p>
      <Source section="18.6" pages="442" />
      <Ask q={questionsS186['ch18/18.6/q-atomics']} label="Work it out" />

      <h2>A private queue per block</h2>
      <p>Fig. 18.15 does it in four phases, separated by three block barriers:</p>
      <ol>
        <li>
          Thread 0 zeroes the shared counter <code>numCurrFrontier_s</code>. Barrier 1.
        </li>
        <li>
          Traverse as before. A thread that wins the compare-and-swap does <code>fetch_add(1)</code> on the private
          counter, with block scope. Below <code>PRIVATE_FRONTIER_CAPACITY</code> it stores the vertex in{' '}
          <code>currFrontier_s</code>. Otherwise the queue is full: it puts the counter back to the capacity and appends
          the vertex straight to the public frontier, with a device-scope atomic. Barrier 2.
        </li>
        <li>
          Thread 0 reserves the block's range with one <code>fetch_add</code> of the private count on the public
          counter, and keeps the old value in <code>currFrontierStartIdx</code>, in shared memory. Barrier 3.
        </li>
        <li>
          Commit: thread t copies private entries t, t + <code>blockDim.x</code>, and so on, to{' '}
          <code>currFrontierStartIdx</code> plus the entry's index. In each pass consecutive threads write consecutive
          addresses, so the stores coalesce, as in <a href={href('ch06', '6.1')}>chapter 6</a>.
        </li>
      </ol>
      <Source section="18.6" pages="443–444" figs="Fig. 18.15" />
      <p>
        The two counters need different scopes: block scope for the one in shared memory, which only the block sees,
        and device scope for the public one, which every block updates. A scope narrower than the set of threads that
        touch an object is a data race.
      </p>
      <Today checked="2026-11-20 (ch18 facts, “Already settled”: ch09 A2)">
        <p>
          <code>cuda::thread_scope_block</code> and <code>thread_scope_device</code> are libcu++'s names; HIP gets the
          same through libhipcxx. MSL's atomic read-modify-writes take no scope.
        </p>
      </Today>
      <p>
        The result, drawn in Fig. 18.16: each block's entries sit in one contiguous range of the public frontier, but
        the ranges are in the order the blocks reserved, not block order, because atomics on one address are served in
        no fixed order. Overflow entries land wherever their own atomic put them.
      </p>
      <Source section="18.6" pages="444" figs="Fig. 18.16" />
      <Ask q={questionsS186['ch18/18.6/q-two-runs']} />
      <p>
        The stepper runs one level of the corrected kernel on 1 to 4 blocks. Six things to find. Start at capacity 8 on
        the fan-out level and read the atomics panel; then drop the capacity until blocks overflow, and work out when
        overflow starts. Change the block order and compare the public frontier. Choose the book's plain store: every
        order is right, and the race panel still reports it. Untick each barrier and find an order that fails: barrier 2
        in order, barrier 3 reversed, and barrier 1 only with the small leftover (the emulator's 0xCD fill is far above
        any capacity, so early appends overflow and nothing is lost). Commit with an <code>if</code> and give a block
        more entries than threads. Turn off the overflow check.
      </p>
      <Simplification>
        <p>
          One step is one statement of one thread, and blocks run one after another, as on the course's emulator. No
          time, warps or caches. The race panel's check is the kind ThreadSanitizer makes; it ignores scopes, as the
          emulator does.
        </p>
      </Simplification>
      <PrivateQueueStepper />
      <Ask q={questionsS186['ch18/18.6/q-overflow']} label="Work it out" />
      <Ask q={questionsS186['ch18/18.6/q-overflow-counter']} label="Work it out" />
      <Ask q={questionsS186['ch18/18.6/q-commit-if']} label="Bug hunt" />

      <h2>The one racy line</h2>
      <p>
        Line 29 of Fig. 18.15, <code>numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;</code>, is a plain store to a
        counter that other threads of the block may be updating with <code>fetch_add</code> at that moment. The book
        calls it restoring the counter. Under the memory model the book itself invokes for the benign races of pp.
        433–434, a plain access that conflicts with a concurrent atomic one is a data race, and a data race is
        undefined behaviour. Fig. 18.17 repeats the line (its line 35).
      </p>
      <Source section="18.6" pages="444" />
      <Today checked="2026-11-20 (ch18 facts A7)">
        <p>
          NVIDIA's definition: two potentially concurrent conflicting actions, at least one not atomic, are a data
          race, and any data race is undefined. The book does not discuss this line.
        </p>
      </Today>
      <p>
        There are two fixes. Store through the same reference, <code>numCurrFrontier_s_ref</code>, with a relaxed{' '}
        <code>store</code>, which is atomic; or drop the store and let thread 0 clamp the count after barrier 2, when
        nobody else touches it. Either way the counter that
        the reservation reads is min(N, capacity) for N discoveries, in every interleaving.
      </p>
      <Beyond title="Our check">
        <p>
          The min(N, capacity) claim comes from enumerating every interleaving for up to 5 threads and capacity 4, not
          from a vendor. The stepper's model repeats the enumeration in its unit tests.
        </p>
      </Beyond>
      <ResetForms />
      <p>
        Why bother, when the numbers come out right? Because a data race is undefined behaviour: tests that pass with
        today's compiler and GPU promise nothing about the next ones. And the tools flag it. The course's{' '}
        <code>race</code> backend, which runs a block's threads at the same time, reports this line against the{' '}
        <code>fetch_add</code> whenever two threads overflow together; the stepper's race panel, which checks every
        order it runs, always does. On Metal the question does not arise: the
        counter is a <code>threadgroup atomic_uint</code>, which cannot be assigned plainly, so only the atomic form can
        be written.
      </p>
      <Today checked="2026-11-20 (ch18 facts C4)">
        <p>
          Compiled, not run, with the local Metal toolchain: the private counter reset with{' '}
          <code>atomic_store_explicit</code> before a <code>threadgroup_barrier</code>.
        </p>
      </Today>
      <Ask q={questionsS186['ch18/18.6/q-line29']} />
      <p>
        One more thing the book leaves out: no kernel checks the public frontier's capacity, and the host allocation is
        never shown. With the compare-and-swap each vertex is appended at most once per search, so a buffer of{' '}
        <code>numVertices</code> entries is enough. That bound is ours, derived from the code; the lab puts guard words
        after each frontier buffer and checks them.
      </p>
      <Source section="18.6" pages="443–444" />

      <h2>The same queue in llama.cpp</h2>
      <p>
        A mixture-of-experts layer has to gather, for each expert, the tokens routed to it: a work list per expert,
        built on the GPU, with no idea in advance how long each list is. llama.cpp's CUDA and HIP backend builds those
        lists in <code>mm_ids_helper</code> with this page's pattern: a private store in shared memory per block, a flush
        and resume when it fills (the overflow path, done differently), and a strided commit. Two differences are worth
        noticing. Each entry's slot comes from a ballot rank rather than an atomic, so the list stays in token order,
        which Fig. 18.16's queue does not. And each block computes where its range starts from the counts of
        lower-numbered experts instead of claiming it with a <code>fetch_add</code>.
      </p>
      <Beyond title="Bridge (ours)">
        <p>
          From llama.cpp's <code>ggml/src/ggml-cuda/mmid.cu</code>, read on 2026-11-20 (facts D4). Not in the book.
        </p>
      </Beyond>

      <h2>One launch for every level</h2>
      <p>
        Every kernel so far runs once per level. Between levels the host waits, copies the frontier size back, and
        launches again. The book's judgement: with large frontiers, as in social networks, that overhead is
        negligible; with small ones, as in road networks, it may dominate.
      </p>
      <Source section="18.7" pages="445" />
      <Predict
        id="ch18/18.6/p-grid-levels"
        prompt={<>A road-like graph: a 1000 × 1000 grid, each vertex linked to its 4 neighbours, searched from a corner. How many levels does the search have after the root?</>}
        unit="levels"
      >
        {() => (
          <p data-reveal>
            {fmt(GRID.levels)} levels: level k is the k-th diagonal, and the far corner is 999 + 999 hops away. The
            per-level host loop makes {fmt(GRID.launches)} launches, the last one finding the frontier empty, and no
            frontier ever holds more than {fmt(GRID.maxFrontier)} of the grid's {fmt(GRID.vertices)} vertices. Each
            launch pays the same fixed cost. The <a href={href('ch18', 'labs')}>lab's</a> "floor" row measures that cost
            on your device.
          </p>
        )}
      </Predict>
      <p>
        One launch for the whole search needs a barrier across the grid between levels. That barrier has a catch: a
        grid may have more blocks than can be resident at once, and the hardware brings blocks in as others finish (
        <a href={href('ch04', '4.2')}>chapter 4</a>). A resident block waiting at a grid barrier for a block that has not
        started never finishes, and the block that has not started waits for a slot that is never freed: a deadlock.
        So every block must be resident, which means launching no more than the device can hold. Chapter 11's
        single-pass scan avoided the problem instead: a block waits only for earlier blocks, and takes its index when
        it starts, so it never waits for one that has not been scheduled (<a href={href('ch11', '11.9')}>section
        11.9</a>).
      </p>
      <Source section="18.7" pages="445" />
      <Aside>
        <p>Typo on p. 445: "thread blocks that are not schedule" means scheduled.</p>
      </Aside>
      <p>
        The book sizes the grid as SMs times blocks per SM, from{' '}
        <code>cudaOccupancyMaxActiveBlocksPerMultiprocessor</code> and the device's <code>multiProcessorCount</code>,
        and launches through <code>cudaLaunchCooperativeKernel</code> with the arguments in a <code>void*</code> array,
        because the <code>&lt;&lt;&lt; &gt;&gt;&gt;</code> syntax cannot ask for a cooperative launch.
      </p>
      <Source section="18.7" pages="445–446" />
      <Today checked="2026-11-20 (ch18 facts A1, A2)">
        <p>
          NVIDIA prints the same recipe, and the runtime checks exactly that product. The four-argument call is valid
          CUDA: a C++ overload defaults the last two.
        </p>
      </Today>
      <LaunchCode />
      <p>
        Inside the kernel, <code>this_grid()</code> returns the grid as a group. The thread count is now fixed, so a
        frontier may have more elements than threads: each thread starts at <code>grid.thread_rank()</code> and steps by{' '}
        <code>grid.num_threads()</code>. The rest of a level is Fig. 18.15, private queue and all. On CUDA both calls
        return <code>unsigned long long</code>; the book's <code>unsigned int</code> narrows them, which is harmless for
        any grid small enough to be resident.
      </p>
      <Source section="18.7" pages="446–447" figs="Fig. 18.17" />
      <Today checked="2026-11-20 (ch18 facts A4)">
        <p>The return types are from NVIDIA's guide.</p>
      </Today>
      <p>
        What the book does not say: under the cooperative API a grid that is too large is never launched. The call
        returns <code>cudaErrorCooperativeLaunchTooLarge</code>. The deadlock is what you risk only by going around that
        API, and what <code>grid.sync()</code> does in a kernel launched with <code>&lt;&lt;&lt; &gt;&gt;&gt;</code> is
        not documented: <code>grid.is_valid()</code> tells you whether the grid can synchronise. The course's emulator
        stops such a kernel with a message naming the cooperative launch; that is the emulator's behaviour, not a GPU's.
      </p>
      <Today checked="2026-11-20 (ch18 facts A3; the emulator's source)">
        <p>
          Cooperative launch needs compute capability 6.0 or later. The emulator refuses a cooperative grid of more
          than 1,024 threads, a limit of its own.
        </p>
      </Today>
      <Simplification>
        <p>
          The animation draws time in launches and levels, never in seconds. A round trip is drawn as the work of 10
          vertices in every strip: an assumption for the picture, not a measurement.
        </p>
      </Simplification>
      <OneLaunchScene />
      <Ask q={questionsS186['ch18/18.6/q-toolarge']} />

      <h2>Three barriers, three jobs</h2>
      <p>
        At the end of each level the threads swap frontiers (Fig. 18.17, lines 65 to 74). The first{' '}
        <code>grid.sync()</code> waits until every append is in. Every thread then reads <code>*numCurrFrontier</code>{' '}
        into its own <code>numPrevFrontier</code>. The second waits until everyone has read it. Grid thread 0 resets the
        counter to 0. The third makes the reset visible before anyone appends for the next level. Then each thread swaps
        its own two pointers. Each thread keeps private copies of the size and the pointers, and every thread reads the
        same value, so they all leave the loop together.
      </p>
      <Source section="18.7" pages="447–448" figs="Fig. 18.17" />
      <Today checked="2026-11-20 (ch18 facts A5)">
        <p>
          <code>sync()</code> makes earlier writes visible to the whole group. Keep each <code>atomic_ref</code> to the
          counter inside the <code>if</code> that appends, so none is alive during the plain read and reset.
        </p>
      </Today>
      <p>
        The stepper runs the whole search in one launch of 2 or 3 blocks of 2 threads. Untick one barrier at a time and
        find the order that breaks it: one gives a thread a partial count, one makes a thread read the reset 0 and leave
        the loop, and one throws away appends made before a late reset. Then switch to Exercise 3. The second tab, "Does
        it fit?", puts a grid of G blocks on R resident slots.
      </p>
      <Simplification>
        <p>
          The private queue is left out: each new vertex is appended straight to the public counter, so a thread runs
          from one grid barrier to the next in one step. The kernel is pinned to{' '}
          <code>labs/ch18/snippets/s186_coop_swap.cu</code> and run on the emulator. No time; the residency tab counts
          blocks, it does not compute occupancy.
        </p>
      </Simplification>
      <GridBarrierStepper />
      <Ask q={questionsS186['ch18/18.6/q-barrier2']} label="Bug hunt" />
      <p>
        Two of the three barriers exist because one counter is reused. Exercise 3 removes the reuse: allocate a counter
        per level, zero them all before the launch, append to counter L and read counter L after one barrier. The book
        calls it a more general form of double buffering; the code is in the third tab of the listing above.
      </p>
      <Source section="18.7" pages="448" />
      <Ask q={questionsS186['ch18/18.6/q-ex3']} label="Exercise 3, p. 450" />
      <p>
        Two other benefits, says the book. The CPU is free while the search runs. And state can stay in shared memory
        and registers from one level to the next instead of going through global memory, which this kernel does not
        use but a fused kernel can. Chapter 11's single-pass scan fused its phases with a one-way wait on the previous
        block; this is a full barrier in both directions.
      </p>
      <Source section="18.7" pages="448" />

      <h2>On HIP and Metal</h2>
      <p>
        HIP has the same cooperative groups with the same names, behind a different header,{' '}
        <code>&lt;hip/hip_cooperative_groups.h&gt;</code>, so one <code>.cu</code> file needs an <code>#if</code> around
        the include, as llama.cpp has. <code>hipLaunchCooperativeKernel</code> takes all six arguments and has no
        defaults, so the book's four-argument call does not compile for HIP; pass <code>0, 0</code> for shared memory
        and stream on both. AMD's own example passes <code>0</code> as the argument array of a kernel with parameters:
        do not copy it. HIP's ranks are 32-bit. The runtime refuses a grid larger than its occupancy limit with{' '}
        <code>hipErrorCooperativeLaunchTooLarge</code>.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. HIP and Metal facts are from the facts file (B1 to B3, C1 to C3) and were not run on
          AMD or Apple hardware. That HIP checks the same product as CUDA is our reading of its runtime source. AMD
          states no memory guarantee for <code>sync()</code>: unverified.
        </p>
      </Beyond>
      <p>
        Metal has no grid barrier, no cooperative launch and no occupancy query, so the single launch has no Metal form.
        What carries over is one dispatch per level. On an <code>MTLCommandQueue</code> with tracked resources and a
        plain compute encoder, the dispatches run in order and each sees the previous one's writes; on{' '}
        <code>MTL4CommandQueue</code>, which tracks no hazards, that does not hold. And Metal can do something CUDA and
        HIP cannot do as a launch: <code>dispatchThreadgroups(indirectBuffer:...)</code> takes the next level's size from
        a buffer the GPU wrote, so the CPU never reads it. A threadgroup count of 0 is not documented; write at least 1
        and return early on an empty frontier.
      </p>
      <Today checked="2026-11-20 (ch18 facts C1 to C3, and the course's Metal runtime)">
        <p>
          The course's Metal runtime uses an <code>MTLCommandQueue</code> and a plain compute encoder, so the labs get
          the ordered, hazard-tracked behaviour.
        </p>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>Feature</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Grid-wide barrier</td>
            <td><code>this_grid().sync()</code> under a cooperative launch</td>
            <td>same names; header <code>&lt;hip/hip_cooperative_groups.h&gt;</code></td>
            <td>none; one dispatch per level</td>
          </tr>
          <tr>
            <td>Cooperative launch</td>
            <td><code>cudaLaunchCooperativeKernel</code>; last two arguments default to 0</td>
            <td><code>hipLaunchCooperativeKernel</code>, all six required</td>
            <td>none</td>
          </tr>
          <tr>
            <td>Grid too large</td>
            <td>refused: <code>cudaErrorCooperativeLaunchTooLarge</code></td>
            <td>refused: <code>hipErrorCooperativeLaunchTooLarge</code></td>
            <td>not applicable</td>
          </tr>
          <tr>
            <td>Sizing a resident grid</td>
            <td><code>multiProcessorCount</code> × occupancy query</td>
            <td>same product (CUs, or WGPs, adjusted in the runtime)</td>
            <td>no core count, no occupancy query</td>
          </tr>
          <tr>
            <td>Rank and size</td>
            <td>64-bit <code>thread_rank()</code>, <code>num_threads()</code></td>
            <td>32-bit</td>
            <td>none</td>
          </tr>
          <tr>
            <td>Visibility across the barrier</td>
            <td>documented for <code>sync()</code></td>
            <td>not stated</td>
            <td>at the dispatch boundary: ordered, tracked hazards on <code>MTLCommandQueue</code></td>
          </tr>
          <tr>
            <td>Next level's size without a CPU read</td>
            <td>graph conditional WHILE node (CUDA 12.3 and later)</td>
            <td>none</td>
            <td><code>dispatchThreadgroups(indirectBuffer:...)</code></td>
          </tr>
          <tr>
            <td>Private counter reset (line 29)</td>
            <td>atomic store or clamp; the plain store is a race</td>
            <td>same</td>
            <td>only an atomic store can be written</td>
          </tr>
        </tbody>
      </table>
      <Ask q={questionsS186['ch18/18.6/q-hip-launch']} />
      <Ask q={questionsS186['ch18/18.6/q-metal-plan']} />

      <h2>Other ways to cut launches</h2>
      <p>
        A cooperative kernel is not the only answer to a long loop of small launches, and in llama.cpp it is the
        exception. Token generation has the small-frontier shape (our reading): many short kernels per token, each
        launched by the host.
        CUDA graphs record many launches and replay them for the cost of one. A conditional WHILE node (CUDA 12.3 and
        later) even loops on the device: a kernel ends the loop with <code>cudaGraphSetConditional</code>, which would
        remove the BFS host's read-back, and cooperative launches are allowed in the body without MPS. Programmatic
        dependent launch (compute capability 9.0 and later) lets the next kernel start before the last one ends. HIP has
        graphs, but neither of the other two.
      </p>
      <Today checked="2026-11-20 (ch18 facts A10, B5)">
        <p>None of this is in the book.</p>
      </Today>
      <p>
        What the engines do: llama.cpp uses CUDA and HIP graphs, and programmatic dependent launch on CUDA only. On
        Metal, which has no graph API, it uses concurrent encoders with memory barriers and encodes command buffers on
        several threads. The one cooperative kernel found in its CUDA and HIP backend (searched 2026-11-20) is a softmax
        for very long rows, one block per SM, launched with six arguments. PyTorch's CUDA graphs forbid the <code>.item()</code>-style read a BFS host loop makes, except
        through <code>torch.cond</code>.
      </p>
      <Today checked="2026-11-20 (ch18 facts D2, D3, D5)">
        <p>Read in llama.cpp's source and PyTorch 2.15's docs.</p>
      </Today>

      <h2>Hubs, and the rest of §18.8</h2>
      <p>
        In a vertex-centric kernel a thread's work is its vertex's degree. In a social graph a few vertices have degrees
        orders of magnitude above the rest, and their threads hold up the whole grid. One remedy is the edge-centric
        mapping of §18.4. The other is degree bucketing: sort the frontier into buckets by degree and run each bucket
        with a group of the right size, a thread for small degrees, a warp for medium, a block for large. The book
        attributes the three tiers to Merrill, Garland and Grimshaw (2012).
      </p>
      <Source section="18.8" pages="448–449" />
      <p>
        That paper does something more dynamic: inside one kernel it gives the longest lists a block, the middle ones a
        warp, and spreads the rest over the block with a prefix sum, and it builds its queues with prefix sums rather
        than atomics. cuGraph today splits vertices at warp size and at 1,024, and runs its high-degree segment
        edge-centric.
      </p>
      <Today checked="2026-11-20 (ch18 facts A11, D1)">
        <p>Read in the 2012 paper and in cuGraph's source.</p>
      </Today>
      <Ask q={questionsS186['ch18/18.6/q-hubs']} />
      <p>
        The book names, and does not develop, graphs larger than GPU memory and preprocessing that reorders vertices for
        parallelism, locality or load balance. For production graph analytics it points to RAPIDS cuGraph, whose
        documentation now lives at docs.nvidia.com/cugraph, with <code>cugraph.bfs</code>. AMD's hipGRAPH and rocGRAPH
        are in early access, and no Apple library was found. Mind the name: "graph" in CUDA graphs, HIP graphs and
        MPSGraph means recorded GPU work, not a graph to traverse.
      </p>
      <Source section="18.8–18.9" pages="449" />
      <Today checked="2026-11-20 (ch18 facts A13, B6, C6)">
        <p>Library status as of the check date.</p>
      </Today>
      <p>
        The <a href={href('ch18', 'labs')}>lab</a> puts the privatised queue and the cooperative kernel to work on a
        road-like grid and a power-law graph, against a floor that measures the fixed cost of a level on your device.
      </p>
    </>
  );
}
