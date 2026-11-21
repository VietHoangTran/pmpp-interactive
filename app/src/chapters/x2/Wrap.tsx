import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// Module X2's summary: what to carry forward, a mixed check on situations the sections did
// not show, where each of the module's tasks is worked (the book has no exercises for it),
// where the sources disagree (the module has no PDF errata), and milestone M9.

const FACTS = 'notes/today/x2-facts.md';

// The KV-cache question's numbers: llama.cpp's q8_0 cache type is 34 bytes per 32 values
// (ch20-facts D3), FP16 is 64; weights and cache 4 GB each in the question.
const KV_SPEEDUP = 8 / (4 + (4 * 34) / 64);

export const questionsWrap: Record<string, Question> = {
  'x2/wrap/q-kv-cache': {
    kind: 'numeric',
    id: 'x2/wrap/q-kv-cache',
    prompt: (
      <>
        A Q4_0 model decodes one sequence at a long context. Each step reads its weights, 4 GB, and its KV cache, also 4
        GB, stored in FP16. You switch the cache to llama.cpp's <code>q8_0</code> cache type, 34 bytes per 32 values. By
        the bytes model, how many times faster is a decode step?
      </>
    ),
    answer: Number(KV_SPEEDUP.toFixed(2)),
    tol: 0.02,
    unit: '×',
    explain: (
      <p>
        FP16 stores 32 values in 64 bytes, so the cache shrinks to 34/64 of 4 GB, 2.125 GB. The step reads 4 + 2.125 =
        6.125 GB instead of 8: 8 / 6.125 ≈ {KV_SPEEDUP.toFixed(2)}. Quantising the weights did nothing for the cache, and
        at a long context the cache is half the bytes, so it caps what weight formats alone can buy. The same count, with
        the cache's bytes in place of the weights', is the argument of section X2.1.
      </p>
    ),
    source: 'ours: the bytes model of X2.1; notes/today/ch20-facts.md D3 (34 bytes per 32 values); x2-facts E1',
  },
  'x2/wrap/q-own-layout': {
    kind: 'choice',
    id: 'x2/wrap/q-own-layout',
    prompt: (
      <>
        You design your own 4-bit format: blocks of 32 weights, each a 4-byte float scale followed by 16 bytes of codes,
        20 bytes per block, all in one array whose start is 16-byte aligned. Which loads of a block's codes are aligned?
      </>
    ),
    options: [
      { label: 'A 4-byte load of the codes is aligned in every block; a 16-byte load only in every fourth block.', correct: true },
      {
        label: 'Every load is aligned in every block, because 20 is a multiple of 4.',
        why: 'A multiple of 4 is enough for 4-byte loads, not for 16-byte ones: block i\'s codes start at 20i + 4, and 16 divides that only when i is 3, 7, 11, ...',
      },
      {
        label: 'A 4-byte load is misaligned in every other block, as it is in Q4_0.',
        why: 'Q4_0\'s scale is 2 bytes, which leaves its codes at 2 mod 4 in even blocks. A 4-byte scale keeps every block\'s codes at a multiple of 4.',
      },
      {
        label: 'Only byte loads are safe: a packed format guarantees no alignment.',
        why: 'Alignment is arithmetic on addresses, not a property of packing. Work out 20i + 4 for each width.',
      },
    ],
    explain: (
      <p>
        Block i starts at byte 20i, and its codes at 20i + 4. That is always a multiple of 4. It is a multiple of 16 only
        when 4i + 4 is, which is i = 3, 7, 11 and so on. So 4-byte words are safe everywhere and 16-byte vector loads are
        not, and nothing in the emulators would tell you: they do not check alignment. Splitting the scales into their
        own array, as lab X2.1 and MLX do, makes every row of codes as aligned as the array.
      </p>
    ),
    source: 'ours: X2.3\'s alignment arithmetic; x2-facts A5; notes/today/infra-facts.md A6 (float4, int4 and uint4: size 16, alignment 16)',
  },
  'x2/wrap/q-row-int': {
    kind: 'choice',
    id: 'x2/wrap/q-row-int',
    prompt: (
      <>
        You port lab X2.3's integer path to Q8_0 weights against Q8_1 activations. To save the float work, you add the
        integer sums of all 128 blocks of a 4096-wide row in one <code>int32</code>, then multiply once by the first
        block's d_w × d_x. What is wrong?
      </>
    ),
    options: [
      {
        label: 'Each block has its own d_w and d_x, so the integer sums are in different units.',
        correct: true,
      },
      {
        label: 'The int32 overflows: 128 blocks of int8 products exceed 2³¹.',
        why: 'One block is at most 516,128; 128 of them are about 66 million, far below 2³¹. An int32 holds about 4,160 worst-case blocks.',
      },
      {
        label: 'Nothing: integer addition is exact, so the grouping cannot change the sum.',
        why: 'Exact integers in different units still add to nonsense. 3 metres plus 2 feet is not 5 of anything.',
      },
      {
        label: 'The first block\'s d_w × d_x is in FP16, so the one product loses precision.',
        why: 'Both scales are widened to float before they are multiplied, and one rounding is tiny. The real error is far larger: the other 127 blocks have other scales.',
      },
    ],
    explain: (
      <p>
        A block's dot product is d_w × d_x × Σ q_w q_x, and the scales change from block to block. Integer inside a block,
        float across blocks: that is the shape of llama.cpp's <code>mmvq</code>, and the reason the per-block rescale cannot
        be hoisted. The honest version of the idea is a different format: one scale per row lets you sum the whole row in
        integers, and it costs accuracy instead of float work, the per-row error X2.2's sweep shows rising on heavy tails.
      </p>
    ),
    source: 'x2-facts D3, E3; ours',
  },
  'x2/wrap/q-torch-zero': {
    kind: 'choice',
    id: 'x2/wrap/q-torch-zero',
    prompt: (
      <>
        You want your Q4_0 GEMV to serve weights from PyTorch's int4 kernel, whose rule is: value = q × scale + zero, with
        q in −8..7, a BF16 scale and a BF16 zero per group of 128. Apart from re-packing the codes as q + 8, what must
        your kernel add?
      </>
    ),
    options: [
      {
        label: 'Groups of 128, BF16 scale and zero decoding, and zero × Σx added once per group.',
        correct: true,
      },
      {
        label: 'BF16 decoding only: q in −8..7 is Q4_0\'s range, so the zero is always 0.',
        why: 'PyTorch\'s zero is a float added after scaling, stored per group. It is not 0 in general.',
      },
      {
        label: 'BF16 decoding, and an integer zero point subtracted from each code in the loop.',
        why: 'That is MPSGraph\'s convention, scale × (q − zeroPoint) with an integer zero. PyTorch\'s is a float added after scaling, and either way it belongs outside the inner loop.',
      },
      {
        label: 'BF16 decoding, and a first pass to float32, because a float zero cannot be fused.',
        why: 'Any per-group offset b folds out: Σ (d q + b) x = d Σ q x + b Σ x. An unfused pass would be slower than float32.',
      },
    ],
    explain: (
      <p>
        Your kernel decodes an FP16 d; these scales and zeros are BF16, a different 16-bit layout, so it needs a second
        decoder. For a group with scale d and offset b, Σ (d q + b) x = d Σ q x + b Σ x. Q4_0's b is −8d, Q4_1's is m, MLX's is its
        bias, PyTorch's is its zero. The inner loop stays codes times activations; the offset costs one multiply-add per
        group, from a sum of x that every output row shares. Four projects, four conventions for one idea, so read the
        dequantiser before you reuse a layout.
      </p>
    ),
    source: 'x2-facts D6 (int4mm.cu), C4 (MPSGraph), D1, D5, E4',
  },
  'x2/wrap/q-prefill': {
    kind: 'choice',
    id: 'x2/wrap/q-prefill',
    prompt: (
      <>
        Your server runs a 512-token prompt through a Q4_0 model before decoding, and its memory has no room for a float
        copy of the weights. Which kernel should run each linear layer of that prefill?
      </>
    ),
    options: [
      {
        label: 'Dequantise one tile of W at a time into shared memory and feed lab 15.6\'s tiled GEMM.',
        correct: true,
      },
      {
        label: 'Lab X2.2\'s GEMV, launched once per group of 8 tokens.',
        why: 'That reads every weight 64 times. At 512 tokens the product is arithmetic-bound, and a GEMV has no reuse of W across rows beyond its batch.',
      },
      {
        label: 'Dequantise the whole of W to float32 once, then run lab 15.6\'s tiled GEMM.',
        why: 'It is correct, but it needs room for a float32 copy of W, 7.1 times the bytes of the Q4_0 weights, and an extra pass over it. At 512 tokens that pass is a modest share of an arithmetic-bound GEMM, so memory, not time, rules it out here; the tile-at-a-time route avoids both, as llama.cpp\'s Metal mul_mm does (D4).',
      },
      {
        label: 'torch._int_mm on the Q4_0 codes, which is fine with 512 rows of activations.',
        why: 'It multiplies int8 by int8 into int32; it knows nothing of 4-bit codes or block scales. Its M > 16 rule is not the obstacle here.',
      },
    ],
    explain: (
      <p>
        At 512 tokens each weight is used 512 times, so the intensity is far above any ridge point and the problem is
        chapter 15's, not this module's. Bytes per weight no longer set the speed; reuse does. llama.cpp's Metal{' '}
        <code>mul_mm</code> does exactly this: it dequantises each tile into threadgroup memory and multiplies with matrix
        instructions. The quantised format saves memory and load time, and costs a little unpacking per tile.
      </p>
    ),
    source: 'x2-facts D4 (mul_mm), A6; ch15 §15.6; ours',
  },
};

// ---------------------------------------------------------------- milestone M9
/** Every lab whose code runs in M9's step, with its directory. */
const M9: { num: string; chapter: string; title: string; dir: string }[] = [
  { num: '2.3', chapter: 'ch02', title: 'M0: bias and residual adds', dir: 'ch02/lab3_elementwise' },
  { num: '10.3', chapter: 'ch10', title: 'M4: softmax and argmax over rows', dir: 'ch10/lab3_softmax_argmax' },
  { num: '22.1', chapter: 'ch22', title: 'the float32 gather GEMV', dir: 'ch22/lab1_gemv' },
  { num: '22.2', chapter: 'ch22', title: 'M8: the batched decode step M9 starts from', dir: 'ch22/lab2_batching' },
  { num: 'X2.1', chapter: 'x2', title: 'the int8-weight GEMV on Q8_0 codes and scales', dir: 'x2/lab1_q8gemv' },
  { num: 'X2.2', chapter: 'x2', title: "the 4-bit GEMV on llama.cpp's Q4_0 blocks", dir: 'x2/lab2_q4gemv' },
  { num: 'X2.4', chapter: 'x2', title: 'M9: the quantised decode step', dir: 'x2/lab4_m9' },
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
  const own = passingOn(reports, 'x2/lab4_m9');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('x2/') && !k.includes('/p-'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          At decode, a linear layer is a matrix-vector product that reads every weight once, so its speed is set by bytes
          per weight, scales included: Q8_0 is 8.5 bits, Q4_0 4.5. Predict the speed-up as the ratio of bytes moved, then
          measure how much of the device's bandwidth the kernel reached.
        </li>
        <li>
          Fuse the dequantise into the product. A separate pass that writes float weights and reads them back moves more
          bytes than not quantising at all.
        </li>
        <li>
          A format is a rule, a layout and a scale type, and every project has its own: llama.cpp's Q4_0 divides the
          signed extreme by −8 and packs weight j with weight j + 16; MLX's affine format keeps a scale and a bias per
          group and packs neighbours from the low bits up. Name whose format you mean. Each extra bit roughly halves the
          error; small blocks help most on heavy tails.
        </li>
        <li>
          An 18-byte block is only 2-byte aligned. Load its codes as bytes or 16-bit words, or keep scales and codes in
          separate arrays. No emulator catches a misaligned load; check by reading.
        </li>
        <li>
          Fold the offset out of the inner loop through the sum of the activations, once per block. Then choose the inner
          product by backend: float multiply-adds on masked nibbles with pre-divided activations where there is no
          integer dot (Metal), or activations quantised to int8 and a byte dot instruction (llama.cpp on CUDA).
        </li>
        <li>
          Keep the sums wide: an integer sum inside a block in int32, a float across blocks, never half. Bound each level
          before you choose its type.
        </li>
        <li>
          Judge a quantised model by its logits and its tokens, not by the error of its weights.
        </li>
      </ul>
      <Beyond title="Ours, not the book's">
        <p>
          Appendix A's §A.5 says smaller formats fetch fewer bytes per operation, and stops. Everything above is this
          module's own, built from chapter 5's intensity, chapter 10's reductions and chapter 22's GEMV.
        </p>
      </Beyond>
      <Source section="A.5" pages="592–593" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole module, on situations the sections did not show.
        {asked ? ` So far in this module you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Ask q={q['x2/wrap/q-kv-cache']} label="Count the bytes" />
      <Ask q={q['x2/wrap/q-own-layout']} />
      <Ask q={q['x2/wrap/q-row-int']} label="Find the defect" />
      <Ask q={q['x2/wrap/q-torch-zero']} />
      <Ask q={q['x2/wrap/q-prefill']} />

      <h2>The module's exercises</h2>
      <p>
        The book has no exercises for this module, because it has no quantised kernel. The plan's goal for X2 is to write
        an int8 and a 4-bit-weight GEMV and measure the bandwidth win. Here is where each part of that is worked.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr><th>Task</th><th>Worked in a section</th><th>Built in a lab</th></tr>
        </thead>
        <tbody>
          <tr><td>Predict a decode GEMV's speed from bytes per weight</td><td>the roofline explorer and scene in <a href={href('x2', 'X2.1')}>X2.1</a></td><td>the predictions of lab X2.1 and lab X2.2</td></tr>
          <tr><td>Turn a measured time into a fraction of bandwidth</td><td>the bandwidth calculator in <a href={href('x2', 'X2.1')}>X2.1</a></td><td>the "% of lab 5.1" columns on the <a href={href('x2', 'labs')}>labs page</a></td></tr>
          <tr><td>Quantise a block by llama.cpp's and MLX's rules</td><td>the block stepper in <a href={href('x2', 'X2.2')}>X2.2</a></td><td>the CPU quantiser checks of lab X2.1 and lab X2.2</td></tr>
          <tr><td>Trade error against bits, block size and outliers</td><td>the error sweep in <a href={href('x2', 'X2.2')}>X2.2</a></td><td>none: a model's error is measured in lab X2.4</td></tr>
          <tr><td>Pack and unpack two codes per byte without a misaligned load</td><td>the layouts, unpack steps and scene in <a href={href('x2', 'X2.3')}>X2.3</a></td><td>lab X2.2</td></tr>
          <tr><td>Map a warp onto rows of quantised blocks</td><td>the stepper in <a href={href('x2', 'X2.4')}>X2.4</a></td><td>lab X2.1 and lab X2.2</td></tr>
          <tr><td>Fold the offset; float path against integer path</td><td>the three-path stepper and scene in <a href={href('x2', 'X2.5')}>X2.5</a></td><td>lab X2.2 (the fold) and lab X2.3</td></tr>
          <tr><td>Choose the accumulator at each level</td><td>the ladder, scene and bug spot in <a href={href('x2', 'X2.6')}>X2.6</a></td><td>lab X2.3, defect 2</td></tr>
          <tr><td>Write an int8-weight GEMV and measure it</td><td><a href={href('x2', 'X2.1')}>X2.1</a> and <a href={href('x2', 'X2.4')}>X2.4</a></td><td>lab X2.1</td></tr>
          <tr><td>Write a 4-bit-weight GEMV and measure it</td><td><a href={href('x2', 'X2.3')}>X2.3</a> and <a href={href('x2', 'X2.4')}>X2.4</a></td><td>lab X2.2</td></tr>
          <tr><td>Debug an integer-path kernel</td><td><a href={href('x2', 'X2.5')}>X2.5</a> and <a href={href('x2', 'X2.6')}>X2.6</a></td><td>lab X2.3</td></tr>
          <tr><td>Quantise a decode step; judge it by logits and tokens</td><td>the step explorer and toy decoder in <a href={href('x2', 'X2.7')}>X2.7</a></td><td>lab X2.4, milestone M9</td></tr>
        </tbody>
      </table>
      <p>
        The book's own exercises on the parts are mapped on their chapters' summary pages: number formats and summation
        in <a href={href('appA', 'wrap')}>appendix A</a>, intensity in <a href={href('ch05', 'wrap')}>chapter 5</a>,
        reduction in <a href={href('ch10', 'wrap')}>chapter 10</a> and decode in <a href={href('ch22', 'wrap')}>chapter 22</a>.
      </p>

      <h2>Where the sources disagree</h2>
      <p>
        This module has no PDF to be wrong: it is not in the book. What it has instead are projects and vendors that
        disagree with each other, and with what a reader would assume. These are the disagreements the module was built
        around, all checked in the facts pass.
      </p>
      <ul>
        <li>
          <b>The offset.</b> Four conventions for one idea: llama.cpp's Q4_1 is d·q + m; MLX's affine is s·q + bias;
          PyTorch's int4 kernel is q·scale + zero, a float added after scaling; MPSGraph's is scale·(q − zeroPoint) with
          an integer zero point.
        </li>
        <li>
          <b>MLX's docs and MLX's code.</b> The documentation takes the scale from max minus min with the minimum as the
          bias. The code takes the larger-magnitude extreme as the bias, allows a negative scale, and adjusts the scale so
          that zero is representable, exactly in real arithmetic and up to one rounding in float32.
        </li>
        <li>
          <b>Nibble order.</b> llama.cpp's Q4_0 packs weight j with weight j + 16; MLX packs neighbours from the low bits of
          a 32-bit word up; Q4_K puts the low nibbles of 32 bytes first in each 64 weights. MSL's new 4-bit types document
          no order at all.
        </li>
        <li>
          <b>The integer dot.</b> CUDA has <code>__dp4a</code> from sm_61. HIP has no <code>__dp4a</code>: an undocumented
          header function and builtins that differ by target, the wrong one a compile error. MSL has no integer dot; its{' '}
          <code>dot</code> of two <code>half4</code> returns <code>half</code>. "One .cu compiles for HIP unchanged" is not
          true here.
        </li>
        <li>
          <b>PyTorch's own page.</b> <code>torch.ao.quantization</code> is superseded by torchao, while the 2.15 page still
          plans its deletion "in 2.10".
        </li>
      </ul>
      <Today checked={`2026-11-20 in the facts pass (${FACTS} B1, B2, C1, C2, C4, D1, D2, D5, D6)`}>
        <p>The "Summary of corrections" in {FACTS} lists every item with its source.</p>
      </Today>
      <p>
        And where our own facts file and design were wrong, found while building the module. The facts file said Q4_0's
        codes were misaligned for a 4-byte load in odd blocks; with an aligned array they are at 2 mod 4 in the even
        blocks, and A5 now carries a dated correction. It said MLX's code makes zero dequantise to exactly 0; in float32
        that holds up to the rounding of one division, and D5 says so. It gave the int32 row bound as about 133,144
        terms; exactly 133,144 fit and the next overflows. The design expected the decode step's speed-up at batch 8 to
        fall slightly; on the book's example device the bytes model makes the Q4_0 head arithmetic-bound there, and the
        predicted speed-up falls by about a third (section <a href={href('x2', 'X2.7')}>X2.7</a>). And the first draft of
        the labs' fixture expected an all-zero Q4_0 block to store a scale of +0; llama.cpp's rule divides 0 by −8 and
        stores −0, <code>0x8000</code>.
      </p>
      <Aside>
        <p>These are our readings of the sources on 2026-11-20. Projects move; check the file before you rely on a line.</p>
      </Aside>

      <h2>What this module leaves out</h2>
      <p>
        You will meet these in real stacks, and none of them is here: choosing scales by calibration (GPTQ, AWQ, and
        llama.cpp's importance matrix, which changes only how d is chosen); kernels for the K-quants, of which
        X2.2 and X2.3 show only the Q4_K layout; Marlin-style int4 GEMMs; integer, FP8 and MX formats on tensor or matrix cores; handling outlier channels in
        the activations; and quantised KV caches beyond the byte count in this page's first question.
      </p>
      <Beyond title="Not checked">
        <p>
          The facts pass read none of these kernels: vLLM, Marlin, AWQ and GPTQ are on its "Not checked" list ({FACTS}).
          This list names them; any claim about how they work is unverified until you read their code.
        </p>
      </Beyond>

      <h2>The project so far</h2>
      <p>
        Lab X2.4's <code>launchQDecodeStep</code> completes milestone M9: the decode step with int8 and 4-bit weights, its
        speed-up measured on your device and its error reported against float32, by logits and by greedy tokens.{' '}
        <span data-m9-own={loading ? 'loading' : own.join(',') || 'none'}>
          {loading
            ? 'Reading reports.'
            : !ran
              ? 'No lab results are recorded on this machine yet.'
              : own.length
                ? `Your own step passes on ${own.map(where).join(' and ')} on this machine.`
                : 'Your own step does not pass on any backend of this machine yet.'}
        </span>{' '}
        See the <a href={href('x2', 'labs')}>labs</a>.
      </p>
      <table className="data" data-table="m9">
        <thead>
          <tr><th>Piece of M9</th><th>Lab</th><th>Your own work passes on</th></tr>
        </thead>
        <tbody>
          {M9.map((m) => {
            const on = passingOn(reports, m.dir);
            return (
              <tr key={m.dir} data-m9={m.num} data-pass={loading ? 'loading' : on.join(',') || 'none'}>
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
        Next in this course's order is module X3, Reading real backends. It finds the kernels behind an op in llama.cpp,
        MLX and PyTorch, including the quantised matrix-vector kernels this module rebuilt. Its milestone M10 is a
        generated Llama-style model run end to end on the course's kernels and checked token by token. GGUF files made
        for llama.cpp store their tensors in the block formats you can now read (x2-facts D1).
      </p>
    </>
  );
}
