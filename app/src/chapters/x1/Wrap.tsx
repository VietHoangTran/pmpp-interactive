import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, NoteGroup, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { LAB } from '../../sim/chain';

// Module X1's summary: what to carry forward, a mixed check on situations the sections did
// not show, where each of the module's tasks is worked (the book has no exercises for it),
// where the sources disagree (the module has no PDF errata), and milestone M5.

export const questionsWrap: Record<string, Question> = {
  'x1/wrap/q-fuse-select': {
    kind: 'numeric',
    id: 'x1/wrap/q-fuse-select',
    prompt: (
      <>
        You try to save the penalty pass. Instead of one kernel that reads each logit and its count and writes the
        penalised logit back, lab X1.1's select applies the penalties on the fly: each of its five reads of a logit (four
        histogram passes and the gather) also reads that token's count. Per logit, ignoring the k output pairs, how many
        bytes does the fused design move?
      </>
    ),
    answer: 40,
    unit: 'bytes',
    explain: (
      <p>
        Five reads of a 4-byte logit and a 4-byte count: 5 × 8 = 40 bytes. The two kernels apart move 12 for the penalty
        pass (read, read, write) and 20 for the select, 32 in all. Fusing pays when it removes a round trip through memory;
        here it removes one write and one read but multiplies the reads of the counts by five. Count before you fuse.
      </p>
    ),
    source: 'ours: the byte counts of X1.3 (12 per penalty pass) and X1.4 (20 per key for the select)',
  },
  'x1/wrap/q-near-boundary': {
    kind: 'choice',
    id: 'x1/wrap/q-near-boundary',
    prompt: (
      <>
        Lab X1.3 passes on a CUDA machine and on your Mac. You log every token of both runs and find 3 draws, out of many
        thousands, where the two backends chose different tokens. The CPU reference had flagged all three as near a
        boundary. What does that tell you?
      </>
    ),
    options: [
      {
        label: 'Nothing is wrong: exp, addition order and rounding differ by backend, and those draws sat within that difference of a boundary.',
        correct: true,
      },
      {
        label: 'The two backends drew different random numbers for those rows, because the vendors ship different Philox libraries.',
        why: 'M5 uses no library generator. Its own Philox passes the known-answer vectors on each backend, and u is compared bit for bit.',
      },
      {
        label: 'One of the two kernels has a real defect, one that the tests did not happen to catch on these rows.',
        why: 'A defect shows on decisive draws, which must match exactly. These three sat inside the band the reference recorded.',
      },
      {
        label: 'The CPU reference is the one that is wrong, because it computes in double.',
        why: 'Double is the closer of the three. The band exists because both devices compute in float, each in its own way.',
      },
    ],
    explain: (
      <p>
        u is the same bits everywhere; the cumulative weights are not. CUDA's <code>expf</code> and Metal's fast{' '}
        <code>exp</code> have different error bounds, the two block reductions add in different orders, and Metal may
        round toward zero. A u that lands within that difference of a boundary can pick either neighbour, and both are
        correct samples. That is why M5 promises the same token only for decisive draws, and why the digest covers those
        alone.
      </p>
    ),
    source: 'X1.8 and lab X1.3 README; notes/today/x1-facts.md C5; ch10-facts.md A11, C4',
  },
  'x1/wrap/q-pad-vocab': {
    kind: 'choice',
    id: 'x1/wrap/q-pad-vocab',
    prompt: (
      <>
        A model's output projection is padded, so every logits row has 128 more columns than the vocabulary has tokens,
        and the matmul leaves 0.0 in them. You call <code>launchSampleRows</code> with <code>vocab</code> set to the padded
        width. What should go into the padding columns first, so that no backend can ever draw one?
      </>
    ),
    options: [
      {
        label: 'A large finite negative logit such as −1e4: its weight underflows to 0, and a zero weight never wins the draw.',
        correct: true,
      },
      {
        label: 'Nothing: leave them at 0.0, since top-k and top-p will remove them.',
        why: 'A logit of 0 is an ordinary value. With logits near 0, or with top-k 0, a padding id can survive and be drawn.',
      },
      {
        label: 'Minus infinity, the mask value the graph samplers use.',
        why: "Under Metal's default fast math the behaviour of infinity is undefined (ch10-facts C3), so the mask is not safe on every backend.",
      },
      {
        label: '−FLT_MAX, the most negative finite float.',
        why: 'The draw computes (l − l₀) / T. At T below 1, −FLT_MAX − l₀ divided by T overflows to minus infinity, which is the previous problem again.',
      },
      {
        label: 'Leave the logits alone and pass the real vocabulary size as vocab.',
        why: 'vocab is both the row length and the row stride, so from row 1 on every row would be read from the wrong place.',
      },
    ],
    explain: (
      <p>
        −1e4 divided by any temperature the sampler accepts stays finite, and exp of it is far below the smallest float, so
        the weight is 0 on every backend that rounds sensibly. Lab X1.3's tests plant exactly such logits and check that
        they never win, and that held on Metal and on the emulators; vendor accuracy bounds alone do not promise an exact
        0. The counts of the padding ids are 0, so the penalties leave them alone.
      </p>
    ),
    source: 'ours; lab X1.3 README (test 3); notes/today/ch10-facts.md C3',
  },
  'x1/wrap/q-n-samples': {
    kind: 'choice',
    id: 'x1/wrap/q-n-samples',
    prompt: (
      <>
        A request asks for four completions of one prompt. Your server gives them four rows of one batch, the request's
        seed, and step 0 for each. What has to differ between the four rows for M5 to produce four different
        completions?
      </>
    ),
    options: [
      {
        label: 'The stream id: the same seed, stream, step and logits give the same u and the same token at every step.',
        correct: true,
      },
      {
        label: 'Nothing: the four rows run in four different blocks, so each draws its own random number.',
        why: "The block index is not an input to M5's u; that was FlashInfer's choice. u depends on (seed, stream, step) alone.",
      },
      {
        label: 'The temperature: give each of the four rows a slightly different T, so their distributions differ.',
        why: 'That changes the distributions a little. With the same u and nearly the same weights, the rows still land on the same token almost every time.',
      },
      {
        label: 'Nothing, provided top-k is 0, because the whole-row path sorts each row on its own.',
        why: 'The sort is deterministic. Identical rows sort identically and give identical candidates.',
      },
    ],
    explain: (
      <p>
        M5's draw is a pure function of the data and of (seed, stream, step). Four identical rows with the same three
        numbers are the same computation four times. Give each completion its own stream: lab X1.3's tests do exactly that,
        one stream per row. The same property is what makes one completion reproducible when a request moves between batch
        slots.
      </p>
    ),
    source: 'ours; X1.7 (M5’s mapping); notes/today/x1-facts.md A6',
  },
  'x1/wrap/q-hip-popc': {
    kind: 'choice',
    id: 'x1/wrap/q-hip-popc',
    prompt: (
      <>
        You port lab X1.1's gather to HIP and rank the kept keys with <code>__popc(vote &amp; lanesBelow)</code>, where{' '}
        <code>vote</code> is the 64-bit value <code>__ballot_sync</code> returns there. On a 64-wide wavefront, what goes
        wrong?
      </>
    ),
    options: [
      {
        label: 'Lanes 32 to 63 count only the keepers in lanes 0 to 31, so two lanes can share a slot and one key is lost.',
        correct: true,
      },
      { label: 'It does not compile: HIP rejects a 64-bit argument to __popc.', why: 'The cut is silent: the 64-bit value is converted to 32 bits (ch12-facts B1).' },
      { label: 'Nothing goes wrong: HIP returns a 32-bit vote even on 64-wide targets, so __popc sees every lane.', why: 'The vote is 64 bits on every AMD target (infra-facts B3).' },
      {
        label: 'Only the order of the output changes; the set of kept keys stays right.',
        why: 'Colliding slots mean a lost key and a slot left holding the sentinel, and the warp totals are cut short as well.',
      },
    ],
    explain: (
      <p>
        Lane 40's rank should count the keepers in lanes 0 to 39. <code>__popc</code> sees only bits 0 to 31, so if lane 32
        also keeps, both lanes compute the same slot. The warp's total, <code>__popc(vote)</code>, is cut the same way, so
        every later warp's offset is wrong too. The fix is the lab's: a typed lane mask and a popcount that matches it,{' '}
        <code>__popcll</code> on HIP.
      </p>
    ),
    source: 'notes/today/ch12-facts.md B1; infra-facts.md B3; lab X1.1 README',
  },
};

// ---------------------------------------------------------------- milestone M5
/** Every lab that supplies a kernel of M5, from chain.ts's LAB list, with its directory. */
const M5: { num: string; chapter: string; title: string; dir: string }[] = [
  { ...LAB.counts, dir: 'ch09/lab3_tokencounts' },
  { ...LAB.softmax, dir: 'ch10/lab3_softmax_argmax' },
  { ...LAB.topp, dir: 'ch11/lab6_topp' },
  { ...LAB.compact, dir: 'ch12/lab4_candidates' },
  { ...LAB.sort, dir: 'ch14/lab3_sortrows' },
  { ...LAB.radix, dir: 'x1/lab1_topk' },
  { ...LAB.filters, dir: 'x1/lab2_filters' },
  { ...LAB.m5, dir: 'x1/lab3_sampler' },
];
const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
/** Backends (devices first, then the emulator) on which the reader's own work of a lab passes. */
const passingOn = (reports: Reports, lab: string) =>
  [...realBackends(reports), 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports, loading } = useReports();
  const ran = Object.keys(reports).length > 0;
  const own = passingOn(reports, 'x1/lab3_sampler');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('x1/') && !k.includes('/p-'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          A sampler is a chain of row-wise stages: penalties change logits, top-k, top-p and min-p remove candidates,
          temperature reshapes what is left, and a draw picks one. Top-k depends only on the order of the logits; top-p and
          min-p depend on where temperature runs. Projects disagree on the order, so name it.
        </li>
        <li>
          Every stage is memory-bound. The questions that matter are how many times the row is read and written, and how
          much of the device is busy. With one block per row, batch 1 occupies one multiprocessor.
        </li>
        <li>
          Only the penalties and top-k read the whole vocabulary. Top-k is a selection: a radix select narrows in on the
          k-th key a byte at a time and moves about a quarter of a full sort's bytes. Everything after it works on k values.
        </li>
        <li>
          Top-p and min-p each keep a prefix of the sorted candidates, so together they keep the shorter prefix. Top-p keeps
          the token that crosses p (an exclusive scan below p); min-p is one comparison against the largest logit plus
          ln(min-p), computed once.
        </li>
        <li>
          Same seed, same bits: a counter-based generator such as Philox keyed by (seed, stream, step), checked against
          published vectors, and a bit-to-float conversion that is exact everywhere. Same bits do not mean the same token:
          float weights differ between backends, so promise agreement only for draws that are not within rounding of a
          boundary, and test exactly that.
        </li>
        <li>One tie rule, applied at every stage, is the only way two backends agree on a row with equal logits.</li>
      </ul>
      <NoteGroup>
        <Beyond title="Ours, not the book's">
          <p>
            The book's transformer ends at the un-embedding layer's distribution over tokens. Everything above is this
            module's own.
          </p>
        </Beyond>
        <Source section="20.1" pages="481" />
      </NoteGroup>

      <h2>A mixed check</h2>
      <p>
        These draw on the whole module, on situations the sections did not show.
        {asked ? ` So far in this module you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Ask q={q['x1/wrap/q-fuse-select']} label="Count the bytes" />
      <Ask q={q['x1/wrap/q-near-boundary']} />
      <Ask q={q['x1/wrap/q-pad-vocab']} />
      <Ask q={q['x1/wrap/q-n-samples']} />
      <Ask q={q['x1/wrap/q-hip-popc']} label="Find the defect" />

      <h2>The module's exercises</h2>
      <p>
        The book has no exercises for this module, because it has no sampling. The plan's goal for X1 is to build greedy,
        top-k and top-p sampling on the GPU and test them against a CPU reference. Here is where each part of that is
        worked.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr><th>Task</th><th>Worked in a section</th><th>Built in a lab</th></tr>
        </thead>
        <tbody>
          <tr><td>Name the stages and fix their order</td><td>the chain explorer and scene in <a href={href('x1', 'X1.1')}>X1.1</a></td><td>lab X1.3, every setting</td></tr>
          <tr><td>Greedy, and its tie rule</td><td><a href={href('x1', 'X1.2')}>X1.2</a></td><td>lab X1.3 at temperature 0 (lab 10.3's argmax)</td></tr>
          <tr><td>Temperature</td><td><a href={href('x1', 'X1.2')}>X1.2</a> and its placement in <a href={href('x1', 'X1.6')}>X1.6</a></td><td>lab X1.3's draw</td></tr>
          <tr><td>Repetition, frequency and presence penalties</td><td>the penalties explorer in <a href={href('x1', 'X1.3')}>X1.3</a></td><td>lab X1.3, tests 4 and 5</td></tr>
          <tr><td>Top-k without a sort</td><td>stepper, scene and cost explorer in <a href={href('x1', 'X1.4')}>X1.4</a></td><td>lab X1.1</td></tr>
          <tr><td>Top-p</td><td><a href={href('x1', 'X1.5')}>X1.5</a></td><td>lab X1.2 part A, then fused in part B</td></tr>
          <tr><td>Min-p</td><td><a href={href('x1', 'X1.6')}>X1.6</a></td><td>lab X1.2</td></tr>
          <tr><td>Random numbers that agree across backends</td><td>the rounds scene and explorers in <a href={href('x1', 'X1.7')}>X1.7</a></td><td>lab X1.3, tests 1 and 2</td></tr>
          <tr><td>The draw</td><td>stepper and scene in <a href={href('x1', 'X1.8')}>X1.8</a></td><td>lab X1.3, test 3</td></tr>
          <tr><td>Test against a CPU reference, across backends</td><td>"Exact agreement across backends" in <a href={href('x1', 'X1.8')}>X1.8</a></td><td>lab X1.3, tests 5 to 7, and the digest on the <a href={href('x1', 'labs')}>labs page</a></td></tr>
          <tr><td>Debug a fused filter</td><td>the boundary rules of <a href={href('x1', 'X1.5')}>X1.5</a></td><td>lab X1.4</td></tr>
        </tbody>
      </table>
      <p>
        The book's own exercises on the parts are mapped on their chapters' summary pages: histograms in{' '}
        <a href={href('ch09', 'wrap')}>chapter 9</a>, reduction in <a href={href('ch10', 'wrap')}>chapter 10</a>, scan in{' '}
        <a href={href('ch11', 'wrap')}>chapter 11</a>, filtering in <a href={href('ch12', 'wrap')}>chapter 12</a> and
        sorting in <a href={href('ch14', 'wrap')}>chapter 14</a>.
      </p>

      <h2>Where the sources disagree</h2>
      <p>
        This module has no PDF to be wrong: it is not in the book. What it has instead are projects, papers and vendor
        libraries that disagree with each other, and with what a reader would assume. These are the disagreements the
        module was built around, all checked in the facts pass.
      </p>
      <ul>
        <li>
          <b>Order.</b> llama.cpp's default chain and mlx-lm truncate before temperature; transformers applies temperature
          first; mlx-lm runs top-p, then min-p, then top-k; the min-p paper applies min-p after temperature.
        </li>
        <li>
          <b>Ties.</b> llama.cpp's top-k keeps the lowest columns; transformers keeps every token tied with the k-th and can
          return more than k; mlx-lm's top-p keeps every token tied at the boundary; llama.cpp's Metal argmax returns the
          highest tied index. M5 sends every tie to the lowest id.
        </li>
        <li>
          <b>Repetition penalty.</b> The CTRL paper divides regardless of sign. llama.cpp multiplies a logit at or below 0
          and divides a positive one; transformers and mlx-lm test below 0, which gives the same result at 0.
        </li>
        <li>
          <b>The draw.</b> llama.cpp's host draw takes the first running sum at or above u × total, its backend path the
          first strictly above. <code>torch.multinomial(probs, 1)</code> is an exponential race, not an inverse CDF. MLX's{' '}
          <code>categorical</code> switches algorithm with the batch shape, so the same key can give different tokens.
        </li>
        <li>
          <b>Random numbers.</b> <code>hiprand_uniform</code> on AMD and <code>curand_uniform</code> turn the same 32 bits
          into different floats for 1 input in 256 (our count). AMD's table says rocRAND's Philox is not cuRAND's sequence,
          while its device header initialises the same way by our reading: unresolved. PyTorch's MPS Philox runs 9 rounds by
          our reading of its source. MSL has no random number function, and may round float arithmetic toward zero.
        </li>
      </ul>
      <Today checked="2026-11-20 in the facts pass (notes/today/x1-facts.md B1, B2, C1, C5, D1 to D9, E2; ch10-facts.md D2; ch11-facts.md D3; ch14-facts.md D2)">
        <p>The "Summary of corrections" in notes/today/x1-facts.md lists every item with its source.</p>
      </Today>
      <p>
        Two places where our own facts file and plan were wrong, found while building the labs. The facts file's note on
        the two library conversions said the samples it printed agree from 2²⁴ upward; that sentence was incomplete, and E2
        now carries a dated correction. By our binade count exactly 2²⁴ of the 2³² inputs differ, a quarter of the inputs
        in [2²⁴, 2²⁵) among them, and <code>app/src/sim/philox.test.ts</code> checks it. And the plan expected the emulator to catch every missing
        barrier of X1.8 and lab X1.4. It catches X1.8's barrier A. A write after a read across warps, with every read
        before the first warp-level rendezvous (X1.8's barrier B, lab X1.4's defect 3), passes in all three of its orders,
        and only the race backend reports it.
      </p>
      <Aside>
        <p>These are our readings of the sources on 2026-11-20. Projects move; check the file before you rely on a line.</p>
      </Aside>

      <h2>The project so far</h2>
      <p>
        Lab X1.3's <code>launchSampleRows</code> completes milestone M5: <code>[rows × vocab]</code> logits and token counts
        in, one token id per row out, the same token for the same (seed, stream, step) on every backend for every decisive
        draw.{' '}
        <span data-m5-own={loading ? 'loading' : own.join(',') || 'none'}>
          {loading
            ? 'Reading reports.'
            : !ran
              ? 'No lab results are recorded on this machine yet.'
              : own.length
                ? `Your own sampler passes on ${own.map(where).join(' and ')} on this machine.`
                : 'Your own sampler does not pass on any backend of this machine yet.'}
        </span>{' '}
        See the <a href={href('x1', 'labs')}>labs</a>.
      </p>
      <table className="data" data-table="m5">
        <thead>
          <tr><th>Piece of M5</th><th>Lab</th><th>Your own work passes on</th></tr>
        </thead>
        <tbody>
          {M5.map((m) => {
            const on = passingOn(reports, m.dir);
            return (
              <tr key={m.dir} data-m5={m.num} data-pass={loading ? 'loading' : on.join(',') || 'none'}>
                <td>{m.title}</td>
                <td className="nowrap">
                  <a href={href(m.chapter, 'labs')}>lab {m.num}</a>
                </td>
                <td>{loading ? 'reading reports' : on.length ? on.map(where).join(', ') : <span className="muted">nothing yet</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p>
        Next in this course's order is chapter 20 of the book, on large language models (pp. 477–512): the attention
        kernels of milestone M6 and the KV cache of M7. Module X3's milestone M10 then runs a small Llama-style model on
        M7's decode step and, on a GPU when lab X1.3's solution is present, can draw its tokens with this sampler in place
        of the argmax.
      </p>
    </>
  );
}
