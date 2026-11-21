import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { functionSource } from '../../sim/toolchain';

// The kernel of the first question is a snippet of our own, loaded at run time, so there is
// no copy to drift: labs/ch11/snippets/wrap_mass_before.cu. Wrap.snippets.test.ts ties its
// helpers to lab 11.1, runs it on the course's emulator, and pins every number below.
export const MASS_FILE = 'ch11/snippets/wrap_mass_before.cu';
/** The shape the first question asks about: one block of 256 threads per row of 256 tokens, 32-lane warps. */
export const MASS = { block: 256, cols: 256, width: 32 } as const;
/** The tokens the first lane of every warp but the first writes: width, 2 width, ... below min(block, cols). */
export const warpStarts = (block: number, cols: number, width: number): number[] =>
  Array.from({ length: Math.ceil(Math.min(block, cols) / width) }, (_, k) => k * width).filter((c) => c > 0);

/**
 * Fig. 11.3 on one block of B slots with any operator: slots at or past values.length are
 * padded with `pad`, then for stride = 1, 2, 4, ... every slot i >= stride combines slot
 * i - stride into itself, all reading the values of the step before. Returns all B slots.
 */
export function koggeStoneBlock(values: number[], B: number, op: (a: number, b: number) => number, pad: number): number[] {
  let cur = Array.from({ length: B }, (_, i) => (i < values.length ? values[i] : pad));
  for (let stride = 1; stride < B; stride *= 2) {
    const prev = cur;
    cur = prev.map((v, i) => (i >= stride ? op(prev[i - stride], v) : v));
  }
  return cur;
}
/** The second question's row: six logits, all negative, in a block of 8 threads. */
export const LOGITS = [-3.5, -1.25, -2, -0.5, -4, -1];
export const LOGITS_BLOCK = 8;

/** The third question's row: one token of 0.9999, then 100,000 tokens of 10^-9. */
export const TAIL = { head: 0.9999, count: 100000, each: 1e-9 } as const;
/** What the two orders give for the last token in float32 (pinned by the unit test against src/sim/scan.ts). */
export const TAIL_RESULT = { loop: '0.9999', koggeStone: '1' } as const;
/** A top-p threshold between the two results, for the explanation. */
export const TAIL_P = 0.99995;

export const questionsWrap: Record<string, Question> = {
  'ch11/wrap/q-mass-before': {
    kind: 'choice',
    id: 'ch11/wrap/q-mass-before',
    prompt: (
      <>
        The kernel above writes the mass before each token of a sorted row, the exclusive scan that a top-p mask compares
        with p, for rows of up to one block of tokens. It runs lab 11.1's inclusive block scan, then takes the inclusive
        value one lane to the left. It compiles and runs. With blocks of {MASS.block} threads on rows of {MASS.cols} tokens,
        on a GPU with {MASS.width}-lane warps, which outputs are wrong?
      </>
    ),
    options: [
      {
        label: 'None: a shuffle up by one hands every lane its left neighbour\'s inclusive value.',
        why: 'Every lane but one. A shuffle stays inside the warp, and lane 0 has no lane to its left there.',
      },
      {
        label: `Tokens ${warpStarts(MASS.block, MASS.cols, MASS.width).join(', ')} of every row: each holds its own inclusive mass.`,
        correct: true,
      },
      {
        label: 'Token 0 of every row.',
        why: 'The kernel sets it to 0, which is right: nothing is ranked before the first token.',
      },
      {
        label: 'Every token of warps 1 to 7.',
        why: 'blockScan has already added the earlier warps\' total to every lane of those warps. Only one lane per warp reads across a boundary.',
      },
    ],
    explain: (
      <>
        <p>
          A shuffle never leaves its warp, and a lane below the shift gets its own value back (section{' '}
          <a href={href('ch11', '11.4')}>11.4</a>). Lane 0 of every warp therefore receives its own inclusive mass, which
          includes its own probability. The test <code>threadIdx.x == 0</code> repairs that for the first warp only. In a
          row of {MASS.cols} that leaves {warpStarts(MASS.block, MASS.cols, MASS.width).length} tokens per row too large by
          exactly their own probability, and those are the tokens a top-p mask can get wrong: when the crossing token sits
          at a warp boundary, <code>massBefore &lt; p</code> fails and the token is dropped. That is the inclusive rule's
          off-by-one, at every 32nd position.
        </p>
        <p>
          The fix is a lane test, not a thread test. Lane 0 of warp w takes the scanned total of the warps before it,{' '}
          <code>warpSums_s[w − 1]</code>, still in shared memory after <code>blockScan</code> (0 for warp 0). Subtracting
          the token's own probability from its inclusive value also works, at the price of one more rounding. On a
          64-lane wavefront the wrong tokens are 64, 128 and 192. Metal's <code>simd_prefix_exclusive_sum</code> returns 0
          in the first lane of each SIMD-group (ch11-facts C1), so it needs the same warp offset added.
        </p>
      </>
    ),
    source: 'ours, from ch. 11 §11.1 p. 253 and §11.4 p. 264; shuffle-up rule: notes/today/ch11-facts.md A3, B1, C1; run on the course emulator by Wrap.snippets.test.ts',
  },
  'ch11/wrap/q-maxpad': {
    kind: 'choice',
    id: 'ch11/wrap/q-maxpad',
    prompt: (
      <>
        You turn Fig. 11.3 into a running maximum (<code>fmaxf</code> in place of <code>+</code>), but the threads past
        the end still load <code>0.0f</code>, the identity of addition, not of max. You run it on {LOGITS.length} logits,
        all negative ({LOGITS.join(', ')}), in a block of {LOGITS_BLOCK} threads. Which of the stored outputs are wrong?
      </>
    ),
    options: [
      {
        label: 'All of them: 0 is larger than every logit, so it wins every maximum.',
        why: 'It would, if any stored output ever read a padded slot. Which direction does each step read in?',
      },
      { label: 'None of them.', correct: true },
      {
        label: 'The last stored output only: it is the one next to the padding.',
        why: 'Being next to the padding does not matter. Each step reads the slot stride to the left, never to the right.',
      },
      {
        label: 'Only those in the same warp as the padding.',
        why: 'There is one warp here, and the answer does not depend on warps: Fig. 11.3 works through shared memory.',
      },
    ],
    explain: (
      <p>
        Every step of an inclusive scan network reads to the left: slot i combines slot i − stride into itself. The padded
        slots sit to the right of every real one, so their 0s flow only into other padded slots, which are never stored. The
        outputs are{' '}
        {koggeStoneBlock(LOGITS, LOGITS_BLOCK, Math.max, 0)
          .slice(0, LOGITS.length)
          .join(', ')}
        ; the two padded slots end as 0 and nobody reads them. The pad would matter elsewhere: an exclusive scan puts the
        identity in slot 0 (p. 253), and a 0 there claims a maximum no logit had. So would a scan that runs right to left.
        Write the operator's real identity anyway (<code>-FLT_MAX</code> for max, chapter 10); it costs nothing and
        survives the next change.
      </p>
    ),
    source: 'ours, from ch. 11 §11.1 p. 253 (identity) and Fig. 11.3, p. 256; ch. 10 §10.1 for the identity of max',
  },
  'ch11/wrap/q-tail': {
    kind: 'choice',
    id: 'ch11/wrap/q-tail',
    prompt: (
      <>
        A sorted row of 100,001 probabilities: one token of {TAIL.head}, then {TAIL.count.toLocaleString('en-US')} tokens
        of 10<sup>−9</sup>, another 10<sup>−4</sup> in all. You compute its inclusive scan in float32 two ways: with Fig.
        11.1's loop and in Kogge-Stone's order. What does each give for the last token?
      </>
    ),
    options: [
      {
        label: 'Both about 1, within a few units in the last place.',
        why: 'Follow the loop: each step adds 10⁻⁹ to a running sum near 0.9999. Is that enough to change a float32 there?',
      },
      { label: `The loop ${TAIL_RESULT.loop}, Kogge-Stone ${TAIL_RESULT.koggeStone}.`, correct: true },
      {
        label: `The loop ${TAIL_RESULT.koggeStone}, Kogge-Stone ${TAIL_RESULT.loop}.`,
        why: 'The other way round. Which order adds the tiny values to each other before adding them to the large one?',
      },
      {
        label: `Both ${TAIL_RESULT.loop}: 10⁻⁹ is too small to register next to 0.9999 in float32.`,
        why: 'One of them is. Two of them added together, then four, then eight, are not, once they are big enough.',
      },
    ],
    explain: (
      <>
        <p>
          Float32 numbers just below 1 are 2<sup>−24</sup>, about 6 × 10<sup>−8</sup>, apart, so adding anything under
          half of that to 0.9999 rounds back to 0.9999. The loop adds the tail one token at a time to the large running sum
          and loses all 10<sup>−4</sup> of it. Kogge-Stone adds neighbours first: pairs, then groups of 4, 8, and so on, so
          the tail's values are summed among themselves into chunks large enough to count before they meet the head. Its
          last token comes out as 1, as exact arithmetic says.
        </p>
        <p>
          Two lessons. The plain loop is not the most accurate order, only the most reproducible one, which is why the labs
          run it in double as their reference. And in a top-p sampler with p = {TAIL_P}, the float loop's mass never
          reaches p, so it keeps all 100,001 tokens, where the tree's keeps about half of them.
        </p>
      </>
    ),
    source: 'ours, from ch. 11 §11.2, pp. 255–256 (the order of Kogge-Stone\'s additions); float addition: notes/today/ch10-facts.md A9; computed with src/sim/scan.ts by Wrap.snippets.test.ts',
  },
  'ch11/wrap/q-batch1': {
    kind: 'choice',
    id: 'ch11/wrap/q-batch1',
    prompt: (
      <>
        Lab 11.6's design gives each row of probabilities one block of 256 threads. Decoding one sequence, the batch is 1:
        a single row of 128,256 tokens. What change lets the scan use the whole GPU?
      </>
    ),
    options: [
      {
        label: 'Launch 1,024 threads per block instead of 256.',
        why: 'Still one block, on one SM; the rest of the device stays idle.',
      },
      {
        label: 'Split the row across many blocks and join their scans, as lab 11.4 does (or lab 11.5 on CUDA).',
        correct: true,
      },
      {
        label: 'Raise the coarsening factor, so that each thread scans more of the row.',
        why: 'That gives the one block\'s threads more work each. It adds no blocks.',
      },
      {
        label: 'Use Brent-Kung inside the block, to save additions.',
        why: 'Additions are not what limits a memory-bound scan, and the row is still read by one block.',
      },
    ],
    explain: (
      <p>
        One block per row is right when there are many rows: they are independent, so 64 rows fill 64 blocks and need no
        communication between blocks. At batch 1 the same design runs one block on the whole device, and lab 11.6's
        benchmark measures exactly that shape. The answer is section <a href={href('ch11', '11.9')}>11.9</a>'s: many blocks
        on the one row, joined by a scan of their totals, in several kernels anywhere or in one kernel on CUDA. The
        projects do the same: on CUDA llama.cpp hands a row that is long for its batch, a single row above all, to CUB's
        device scan, and on Apple GPUs PyTorch splits a long axis across threadgroups in three passes.
      </p>
    ),
    source: 'ours, from ch. 11 §11.9, pp. 273–277; notes/today/ch11-facts.md D1, D4',
  },
  'ch11/wrap/q-subtract': {
    kind: 'choice',
    id: 'ch11/wrap/q-subtract',
    prompt: (
      <>
        A batch of sequences is packed end to end in one buffer, and you already have G, the inclusive scan of the whole
        buffer. Sequence k occupies positions s to e. For a position i inside it, which gives sequence k's own inclusive
        scan, the one that restarts at s?
      </>
    ),
    options: [
      { label: 'G[i] − G[s − 1], with G[−1] taken as 0.', correct: true },
      { label: 'G[i] − G[s], the total up to the sequence start.', why: 'G[s] already includes the sequence\'s first element, so this drops it.' },
      { label: 'G[i] − x[s].', why: 'That removes the first element of the sequence and keeps everything before it.' },
      { label: 'G[e] − G[i], the part of the sequence up to i.', why: 'That is the mass after position i within the sequence, a suffix, not a prefix.' },
    ],
    explain: (
      <>
        <p>
          G[s − 1] is the total of everything before the sequence (an exclusive scan at s, p. 253), and subtracting it
          leaves the sum from s to i. This works because addition can be undone. A max scan cannot be restarted that way,
          since nothing undoes a max: there you need a scan that resets wherever a flag marks a sequence start, a segmented
          scan.
        </p>
        <p>
          In float the subtraction has a price. G grows over the whole batch, and the difference of two large, nearly equal
          totals keeps only the bits the large numbers had left: a short sequence late in a big batch gets a scan far less
          accurate than one computed on its own.
        </p>
      </>
    ),
    source: 'ours, from the definitions of ch. 11 §11.1, pp. 252–253',
  },
};

/** The kernel of the first question, from the snippet file. */
function MassBeforeKernel() {
  const file = useLabFile(MASS_FILE);
  const fn = file && !file.startsWith('// could not load') ? functionSource(file, 'massBeforeRows') : null;
  return (
    <Code
      tabs={[
        {
          label: 'CUDA (not a lab file: for this question only)',
          code: fn ? fn.text : file ?? `// loading labs/${MASS_FILE}`,
          status: 'front-end-checked and emulator-checked; wrong on purpose',
          tiers: {
            probs: 'global',
            massBefore: 'global',
            row: 'global',
            out: 'global',
            warpSums_s: 'shared',
            p: 'reg',
            inclusive: 'reg',
            before: 'reg',
          },
        },
      ]}
      caption={
        <>
          From <code>labs/{MASS_FILE}</code>. Its mask type, <code>laneIdx</code>, <code>warpIdx</code>,{' '}
          <code>warpScan</code> and <code>blockScan</code> are lab 11.1's, character for character. The unit tests run it
          on the course's emulator.
        </>
      }
    />
  );
}

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const m5 = backends.filter((b) => labStatus(reports[b], 'ch11/lab6_topp', 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch11/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          A scan returns every partial total, not just the last. The inclusive and exclusive forms differ by one position:
          the exclusive scan gives start offsets and the mass before a token, and top-p keeps a token while that mass is
          below p.
        </li>
        <li>
          Every parallel scan regroups the additions, so it needs an associative operator. Float addition is only nearly
          associative: two orders give different last bits, and the plain loop is not even the most accurate one. Test
          with exact data and a band, against a reference run in double.
        </li>
        <li>
          Kogge-Stone takes log₂N steps and does N·log₂N − (N − 1) additions. Its second barrier guards only a false
          dependence, which two buffers remove; warp shuffles and scan-scan-add leave two barriers per block scan.
        </li>
        <li>
          Extra work costs when execution units are few. Coarsening gives each thread a sequential run, about 2N additions
          in all, and the block stays coalesced by loading through shared memory chunk by chunk, then working in registers.
        </li>
        <li>
          A scan does about one addition per 8 bytes, so it is memory-bound: judge it against your copy bandwidth.
          Across blocks the designs differ in bytes per element, 16, 12 or 8.
        </li>
        <li>
          Separate kernels join blocks safely on every backend. One kernel needs blocks that wait for each other: CUDA
          documents the progress that makes a counter-claimed index and release/acquire flags safe; AMD and Apple document
          none, so on Metal the global scan is several dispatches.
        </li>
        <li>
          Brent-Kung does less work in more steps. Inside a warp the lanes are there anyway, so Kogge-Stone wins where the
          parallel scan is left after coarsening.
        </li>
      </ul>
      <Source section="11.11" pages="285–286" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Beyond>
        <p>The kernel and the situations below are ours.</p>
      </Beyond>
      <MassBeforeKernel />
      <Ask q={q['ch11/wrap/q-mass-before']} label="Find the defect" />
      <Ask q={q['ch11/wrap/q-maxpad']} />
      <Ask q={q['ch11/wrap/q-tail']} />
      <Ask q={q['ch11/wrap/q-batch1']} />
      <Ask q={q['ch11/wrap/q-subtract']} />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with seven exercises (p. 286). Each is worked somewhere in this course; the book prints no
        answers, so the answers here are ours.
      </p>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-table="exercises">
          <thead>
            <tr>
              <th>Exercise</th>
              <th>About</th>
              <th>Where it is worked here</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td>Kogge-Stone on 4 6 7 1 2 8 5 2, step by step</td>
              <td>
                the stepper and a question in <a href={href('ch11', '11.2')}>11.2</a>; the steppers of{' '}
                <a href={href('ch11', '11.3')}>11.3</a> run on the same values
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>Divergence only in the first warp, for strides up to 16</td>
              <td>
                the "Which warps diverge?" explorer and a question at width 64 in <a href={href('ch11', '11.2')}>11.2</a>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>Additions of Fig. 11.3 on 2,048 elements</td>
              <td>
                a question in <a href={href('ch11', '11.2')}>11.2</a>, with both readings of the exercise
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td>Vector loads and no bank conflicts in Fig. 11.13</td>
              <td>
                the bank map and its questions in <a href={href('ch11', '11.7')}>11.7</a>; an untested "go further" in{' '}
                <a href={href('ch11', 'labs')}>lab 11.2</a>
              </td>
            </tr>
            <tr>
              <td>5</td>
              <td>
                <code>interBlockScan</code> with decoupled lookback
              </td>
              <td>
                the "Who waits for whom" simulator and a question in <a href={href('ch11', '11.9')}>11.9</a>; an untested
                "go further" in <a href={href('ch11', 'labs')}>lab 11.5</a>
              </td>
            </tr>
            <tr>
              <td>6</td>
              <td>Brent-Kung on 4 6 7 1 2 8 5 2, step by step</td>
              <td>
                the up-sweep and down-sweep stepper and a question in <a href={href('ch11', '11.10')}>11.10</a>
              </td>
            </tr>
            <tr>
              <td>7</td>
              <td>A block-level Brent-Kung kernel</td>
              <td>
                "Exercise 7: a block-level kernel" in <a href={href('ch11', '11.10')}>11.10</a>, emulator-checked, with a
                question on its barriers
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy of the book this course was built from has a number of apparent slips in chapter 11, and several listings
        that are narrower than they look. They are listed so that you are not thrown by them when you read the book
        alongside. Where one changes code, the labs and snippets have a corrected, tested version.
      </p>
      <ul data-list="errata">
        <li>
          Figs. 11.12, 11.13 and 11.18 take <code>N</code> and never use it, with no bounds checks: they are right only
          when N is a whole number of block segments, which the text never says. Labs 11.2, 11.4 and 11.5 check every load
          and store.
        </li>
        <li>
          Fig. 11.18 (p. 281) has no comma between <code>unsigned int* flags</code> and <code>unsigned int N</code>, and
          would not compile as printed. Its line 12 uses <code>blockDim.x</code> where line 14 uses{' '}
          <code>BLOCK_DIM</code>.
        </li>
        <li>
          The flags of Fig. 11.17 and the counter of Fig. 11.18 must be zero before every launch. The book says so for the
          counter (p. 280) and only implies it for the flags (p. 278). Lab 11.5's launcher zeroes both.
        </li>
        <li>
          Fig. 11.17 reads the flag with <code>fetch_add(0, acquire)</code>. That works, but it writes where only a read is
          needed; NVIDIA's own example of the protocol uses <code>load(acquire)</code> (ch11-facts A5), and so does lab
          11.5.
        </li>
        <li>
          Fig. 11.8 passes the literal <code>0xffffffff</code> and assumes 32 lanes, which HIP rejects at compile time and
          a 64-lane wavefront scans in halves (<a href={href('ch11', '11.4')}>11.4</a>; ch11-facts B1). Fig. 11.9 declares a{' '}
          <code>__shared__</code> array inside a <code>__device__</code> function, which Metal refuses and HIP does not
          document. Both are CUDA-only, not wrong for CUDA; the labs use a typed mask, <code>warpSize</code>, and an array
          declared in the kernel.
        </li>
        <li>
          Fig. 11.3 (p. 256) prints line 14 as <code>__ syncthreads();</code> with a space, and p. 258 writes{' '}
          <code>__syncthread()</code> once.
        </li>
        <li>
          p. 258 gives the corrupted value of the race as 2x₁ + 2x₂ + x₃ + x₄ + x₅ + x₆. Worked through, it is x₀ + 2x₁ +
          2x₂ + x₃ + x₄ + x₅ + x₆ (<a href={href('ch11', '11.2')}>11.2</a>). The point, x₁ and x₂ counted twice, stands.
        </li>
        <li>
          p. 259 has thread 1 writing Σx₁..x₂ and reading x₁ in the double-buffering example. In iteration 0 thread 1
          writes Σx₀..x₁, and it is thread 2 that reads x₁ and writes Σx₁..x₂ (<a href={href('ch11', '11.3')}>11.3</a>).
        </li>
        <li>
          p. 265 contrasts "the code in Fig. 11.5" with "the code in Fig. 11.5"; the second is Fig. 11.9. p. 269 calls the
          block scan <code>blockSum</code>; the function is <code>blockScan</code>.
        </li>
        <li>
          p. 283 places the reduction phase in "the bottom portion of Fig. 11.19"; p. 282 puts it in the top half. In Fig.
          11.20 (p. 284), columns 9 and 11 are shaded one level darker than the legend gives in the first row, and column 9
          in the second (<a href={href('ch11', '11.10')}>11.10</a>).
        </li>
        <li>
          Exercise 3 asks for the additions on 2,048 elements without saying whether that is one segment (20,481) or two
          blocks of 1,024 (18,434), since a block holds at most 1,024 threads.
        </li>
        <li>
          The chapter never says what <code>__shfl_up_sync</code> returns to the lanes below the stride; the code simply
          does not use it. All three vendors document that those lanes get their own value back, which is why the test{' '}
          <code>laneIdx() &gt;= stride</code> is required (<a href={href('ch11', '11.4')}>11.4</a>; ch11-facts A3).
        </li>
        <li>
          The summary (p. 285) says the dynamic block index prevents deadlocks; the body (p. 277) only says that earlier
          blocks must be scheduled before later ones, and never names the deadlock that{' '}
          <a href={href('ch11', '11.9')}>11.9</a>'s simulator shows.
        </li>
        <li>
          Iterations are counted from 1 in the prose of §11.2 and from 0 in Figs. 11.2 and 11.4. The course says which it
          means each time.
        </li>
        <li>
          Typesetting on pp. 283–284 subscripts only the first digit of two-digit indices ("x₁1" for x₁₁). Typos: "wap-level
          scan" (p. 264), "They key difference" (p. 276), "to eliminated" (exercise 4).
        </li>
        <li>
          p. 272 gives the example device 66.9 TFLOPS; NVIDIA prints 67 (ch11-facts A1). Every number derived from it
          still holds.
        </li>
        <li>
          p. 273 says even a copy kernel does not reach 100% of the bandwidth. No vendor statement of that was found
          (ch11-facts A2); it is unverified, and lab 11.1 measures your copy instead.
        </li>
      </ul>
      <Aside>
        <p>
          These are our readings of one PDF, checked against current vendor documentation on 2026-11-20
          (notes/today/ch11-facts.md). If your copy differs, trust your copy.
        </p>
      </Aside>
      <p>
        Some of the book is left out on purpose, so that you can tell an omission from a gap. Exercise 4 and exercise 5
        are explained and modelled, and offered as untested "go further" items in labs 11.2 and 11.5, not as labs.
        Multiple lookback is described in the prose of section 11.9 but not simulated. The 11.9 simulator models the
        grid-wide barrier of method (a), single lookback and decoupled lookback, whose worst case approaches the
        every-block-adds-everything extreme of Fig. 11.16(c); none of the labs builds the barrier, and the cooperative
        launch it needs is chapter 18's. Exercise 7 is a checked snippet on the 11.10 page, not a lab. Of the applications on p. 251, only
        filter (chapter 12) and radix sort (chapter 14) are followed; the adder-circuit history and the energy remark get a
        sentence each. Thrust and CUB are named, not used, and of the references only the decoupled look-back report is
        cited, because CUB and rocPRIM cite it.
      </p>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md A8; ch10-facts A12)">
        <p>
          The book's Thrust and CUB links now live under <code>nvidia.github.io/cccl/unstable/</code>. CUB's{' '}
          <code>DeviceScan</code> has used decoupled look-back since CUB 1.0.1 (2013).
        </p>
      </Today>

      <h2>The project so far</h2>
      <p>
        Lab 11.6's top-p keep mask is the top-p piece of milestone M5 of the through-line project: the cumulative mass
        before each token of a sorted row, the keep decision with its stated tie rule, and the count of kept tokens.{' '}
        <span data-m5={m5.join(',') || 'none'}>
          {backends.length === 0
            ? 'No lab results are recorded on this machine yet.'
            : m5.length
              ? `Your own lab 11.6 passes on ${m5.map(where).join(' and ')} on this machine.`
              : 'Your own lab 11.6 does not pass on any backend of this machine yet.'}
        </span>{' '}
        See the <a href={href('ch11', 'labs')}>labs</a>.
      </p>
      <p>
        Chapter 12 uses the scan as a tool. A keep mask says which tokens survive; an exclusive scan of that mask gives
        each surviving token its slot in a short, dense list, which is how the kept tokens of M5 are compacted before a
        sample is drawn. Start at <a href={href('ch12')}>chapter 12</a>.
      </p>
    </>
  );
}
