import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Source, Today } from '../../components/notes';
import { labStatus, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// Prerequisite questions drawn from chapters 4, 5, 6, 10 and 15 and their facts files. Each
// explanation names where the idea was taught and which section of this chapter uses it.
export const questionsS200: Record<string, Question> = {
  'ch20/start/q-gemv-ratio': {
    kind: 'numeric',
    id: 'ch20/start/q-gemv-ratio',
    prompt: (
      <>
        A decode step multiplies one fp32 activation row by a 4096 × 4096 weight matrix. Counting only the bytes of the
        weights, how many floating-point operations does it do per byte?
      </>
    ),
    answer: 0.5,
    tol: 0.01,
    unit: 'FLOP/B',
    explain: (
      <p>
        Each weight is read once and used in one multiply-add: 2 FLOP for 4 bytes, so 0.5 FLOP/B. The two vectors add
        16 KiB against 64 MiB of weights and change nothing. With one row there is no second use of any weight, so no
        tile size can raise the ratio; only more rows can (batching). Taught in{' '}
        <a href={href('ch05', '5.1')}>ch. 5 §5.1</a>. Sections 20.4 and 20.6 build on it: decode is this operation,
        repeated over every layer and over the whole KV cache.
      </p>
    ),
    source: 'ch. 5 §5.1, pp. 94–97; ch. 20 pp. 491–492, 504–505',
  },
  'ch20/start/q-overflow': {
    kind: 'choice',
    id: 'ch20/start/q-overflow',
    prompt: (
      <>
        A row of fp32 logits, all equal to 100, goes through softmax <em>without</em> subtracting the row max. What comes
        out?
      </>
    ),
    options: [
      { label: 'NaN in every entry.', correct: true },
      { label: '1/n in every entry.', why: 'That is the max-subtracted result. Without the subtraction nothing finite is left to divide.' },
      { label: '0 in every entry.', why: 'That would need a finite numerator over an infinite sum. Here the numerators overflow too.' },
      { label: '1 in the first entry, 0 elsewhere.', why: 'Every entry is treated alike; nothing singles out the first.' },
    ],
    explain: (
      <p>
        exp(100) is beyond the largest float (exp of anything above about 88.7 is), so every numerator and the sum are
        infinity, and infinity divided by infinity is NaN. Subtracting the max makes every exponent at most 0 and the
        row comes out as 1/n each. Lab 10.3 tests exactly this with rows offset by +90 (
        <a href={href('ch10', 'labs')}>ch. 10 labs</a>). Eq. 20.2 (section <a href={href('ch20', '20.2')}>20.2</a>) is the
        same fix, and section <a href={href('ch20', '20.3')}>20.3</a>&rsquo;s kernel uses it.
      </p>
    ),
    source: 'lab 10.3 (labs/ch10/lab3_softmax_argmax); ch. 20 Eq. 20.2, p. 484',
  },
  'ch20/start/q-lane0': {
    kind: 'choice',
    id: 'ch20/start/q-lane0',
    prompt: (
      <>
        After a shuffle-down warp reduction (or <code>cub::WarpReduce</code>), every lane must divide its own values by
        the warp&rsquo;s sum. What is missing?
      </>
    ),
    options: [
      {
        label: 'Only lane 0 has the sum; broadcast it from lane 0 first.',
        correct: true,
      },
      {
        label: 'Nothing: every lane ends a shuffle-down reduction holding the sum.',
        why: 'Shuffle-down folds the values towards lane 0. The other lanes end with partial sums.',
      },
      {
        label: 'A __syncthreads() after the reduction, so that every lane can read the sum.',
        why: 'The sum is in a register of lane 0. A barrier makes no other lane able to see a register.',
      },
      {
        label: 'Nothing, provided the warp is fully converged when it divides.',
        why: 'Convergence decides who executes together, not which register holds what.',
      },
    ],
    explain: (
      <p>
        A shuffle-down reduction folds towards lane 0, and CUB documents <code>WarpReduce</code>&rsquo;s output as
        valid in lane 0 only (facts A2). Broadcast it, or reduce with xor shuffles so that every lane ends with the full
        sum. Metal&rsquo;s <code>simd_sum</code> already returns it to every lane (facts C3). Taught in{' '}
        <a href={href('ch10', '10.7')}>ch. 10 §10.7</a>. The book&rsquo;s flash kernel (section 20.5, Figs. 20.12 and
        20.15) does this broadcast after each warp reduction.
      </p>
    ),
    source: 'ch. 10 §10.7; notes/today/ch20-facts.md A2, C3; ch. 20 pp. 500–503',
  },
  'ch20/start/q-two-reductions': {
    kind: 'choice',
    id: 'ch20/start/q-two-reductions',
    prompt: (
      <>
        In lab 10.3&rsquo;s softmax, <code>blockReduceMax</code> leaves the row max in <code>scratch[0]</code> and every
        thread reads it. The next call, <code>blockReduceSum</code>, starts with lane 0 of each warp writing its
        warp&rsquo;s partial sum into <code>scratch[warp]</code>. Why must a <code>__syncthreads()</code> sit between the
        two calls?
      </>
    ),
    options: [
      {
        label: 'Warp 0 may overwrite scratch[0] before slower warps have read the max.',
        correct: true,
      },
      {
        label: 'So that the max is already in scratch[0] when the other threads go to read it.',
        why: 'That read-after-write is guarded by the barrier inside blockReduceMax, before its return.',
      },
      {
        label: 'Without it the compiler may reorder the two calls.',
        why: 'The barrier orders threads against each other; reordering one thread\'s own statements is not the risk.',
      },
      {
        label: 'It is not needed: the warps of one block run in lockstep.',
        why: 'Nothing promises that. Warps of a block run in any order between barriers.',
      },
    ],
    explain: (
      <p>
        The write needs nothing from the slow warp; the conflict exists only because the location is reused. That is a
        write-after-read, the second barrier of the tiled matmul (<a href={href('ch05', '5.4')}>ch. 5 §5.4</a>), and
        lab 10.3&rsquo;s comment says so at that barrier. Section <a href={href('ch20', '20.3')}>20.3</a> puts the
        book&rsquo;s kernel, which broadcasts through a separate scalar, next to this one and finds that there the same
        barrier is not what keeps it right.
      </p>
    ),
    source: 'ch. 5 §5.4, pp. 110–111; labs/ch10/lab3_softmax_argmax/solution/kernel.cu',
  },
  'ch20/start/q-qkt-linear': {
    kind: 'choice',
    id: 'ch20/start/q-qkt-linear',
    prompt: (
      <>
        Q and K are row-major [N × d]. Which of your milestone calls computes <code>S = Q K^T</code> with no transpose
        or extra copy?
      </>
    ),
    options: [
      { label: 'Lab 15.6\'s launchLinear(Q, K, S, N, d, N).', correct: true },
      {
        label: 'Lab 5.3\'s launchGemm(Q, K, S, N, d, N).',
        why: 'launchGemm multiplies by its second argument as stored, [k × l]. K is [N × d], so this needs K^T in memory.',
      },
      {
        label: 'Either one, since both compute row-major products.',
        why: 'They differ in how the second operand is stored, which is the whole question.',
      },
      {
        label: 'Neither, without a transpose kernel first.',
        why: 'One of them already reads its second operand transposed.',
      },
    ],
    explain: (
      <p>
        <code>launchLinear(X, W, Y, batch, in, out)</code> computes Y = X W<sup>T</sup> with W stored (out, in), like an{' '}
        <code>nn.Linear</code> weight. K is exactly that with N outputs of width d. It has no scale argument, so 1/√d
        goes elsewhere; lab 10.3&rsquo;s <code>launchSoftmaxRows</code> already takes one. P V is the other way round
        (V is [N × d], the second operand as stored), which is lab 5.3&rsquo;s <code>launchGemm</code>. Taught in{' '}
        <a href={href('ch15', '15.1')}>ch. 15 §15.1</a> and <a href={href('ch15', 'labs')}>lab 15.6</a>; section 20.3
        builds the unfused pipeline from these two calls.
      </p>
    ),
    source: 'ch. 15 §15.1; labs/ch15/lab6_linear/README.md; labs/ch05/lab3_gemm',
  },
  'ch20/start/q-coalesce': {
    kind: 'choice',
    id: 'ch20/start/q-coalesce',
    prompt: (
      <>
        In the book&rsquo;s layout, the lanes of a 32-wide warp each read <code>Q[row*d + lane + 32*k]</code> for the
        same <code>k</code> and the same <code>row</code>. Is the load coalesced?
      </>
    ),
    options: [
      { label: 'Yes: 32 consecutive floats of one row.', correct: true },
      {
        label: 'No: each lane later reads other columns too.',
        why: 'Coalescing is about one load instruction across the warp, not what a lane does across iterations.',
      },
      { label: 'Only when d is 32.', why: 'Any d works: for one k the 32 addresses are consecutive whatever the row length.' },
      { label: 'Only for k = 0.', why: 'Changing k shifts the whole run by 32 floats; it stays one consecutive run.' },
    ],
    explain: (
      <p>
        For a fixed <code>k</code> the addresses are <code>row*d + 32*k</code> plus 0 to 31: one consecutive run of
        128 bytes. This is the book&rsquo;s <code>load_Q</code> (Fig. 20.10), which deals each row of Q out to the lanes
        in interleaved order exactly so that every load is coalesced. Taught in{' '}
        <a href={href('ch06', '6.1')}>ch. 6 §6.1</a>; used in section 20.5.
      </p>
    ),
    source: 'ch. 6 §6.1; ch. 20 Fig. 20.10, p. 498',
  },
  'ch20/start/q-wave64': {
    kind: 'choice',
    id: 'ch20/start/q-wave64',
    prompt: (
      <>
        A kernel written for 32-lane warps hard-codes 32 and calls shuffles with their default width. On an AMD gfx9
        GPU, what is <code>warpSize</code>, and what breaks?
      </>
    ),
    options: [
      {
        label: '64; an upper-half lane asking for lane 5 gets the lower half\'s lane 5.',
        correct: true,
      },
      { label: '32: HIP reports a warp size of 32 on every AMD GPU, so the kernel runs unchanged.', why: 'HIP reports the hardware width: 64 on gfx9, 32 on gfx10 and later.' },
      { label: '64; the compiler splits each wavefront in two, so nothing breaks.', why: 'Nothing splits it. The default width is the whole 64-lane wavefront.' },
      { label: '64; nothing breaks, because shuffle source lanes wrap at 32.', why: 'They wrap at the width argument, which defaults to warpSize, 64.' },
    ],
    explain: (
      <p>
        AMD&rsquo;s gfx9 GPUs (the Instinct line) run 64-lane wavefronts (ch05 facts B5). HIP&rsquo;s shuffle computes
        the source as <code>(src_lane &amp; (width − 1)) + (self &amp; ~(width − 1))</code> with{' '}
        <code>width = warpSize</code>, so hardware lane 40 asking for lane 5 gets hardware lane 5, which belongs to the
        other logical warp (facts B2). Fixes: make the width a compile-time parameter equal to the wavefront, or pass
        width 32 to every shuffle. Taught in <a href={href('ch04', '4.4')}>ch. 4 §4.4</a> and chapter 10; section 20.5
        meets it in the book&rsquo;s flash kernel.
      </p>
    ),
    source: 'notes/today/ch05-facts.md B5; notes/today/ch20-facts.md B2',
  },
};

// ---------------------------------------------------------------- milestone readiness
// What the chapter composes: M2 and M3 are the GEMMs of the unfused pipeline, M4 the row
// softmax lab 20.1's kernel grows from. Pass/fail comes from this machine's reports only.
const MILESTONES: { m: string; lab: string; num: string; call: string; role: string }[] = [
  { m: 'M2', lab: 'ch05/lab3_gemm', num: '5.3', call: 'launchGemm', role: 'O = P V' },
  { m: 'M3', lab: 'ch15/lab6_linear', num: '15.6', call: 'launchLinear', role: 'S = Q K^T' },
  { m: 'M4', lab: 'ch10/lab3_softmax_argmax', num: '10.3', call: 'launchSoftmaxRows', role: 'the row softmax lab 20.1 extends' },
];
const DEVICES: { b: string; name: string }[] = [
  { b: 'metal', name: 'Metal' },
  { b: 'cuda', name: 'CUDA' },
  { b: 'hip', name: 'HIP' },
];

function cell(reports: Reports, b: string, lab: string) {
  const run = reports[b];
  if (!run) return { work: 'no run on this machine', ref: '', tone: 'none' as const };
  const w = labStatus(run, lab, 'work');
  const r = labStatus(run, lab, 'solution');
  const untouched = run.work_state?.[lab] === 'untouched';
  const work = w === 'pass' ? 'yours passes' : w === 'fail' ? 'yours fails' : untouched ? 'not started' : 'no run of yours';
  const ref = r === 'pass' ? `reference verified on ${DEVICES.find((d) => d.b === b)?.name}` : r === 'fail' ? 'reference failed here' : 'reference not run';
  return { work, ref, tone: w === 'pass' ? ('ok' as const) : w === 'fail' ? ('fail' as const) : ('none' as const) };
}

function Readiness() {
  const { reports, loading, reload } = useReports();
  const any = DEVICES.some((d) => reports[d.b]);
  return (
    <Bench
      title="What lab 20.1 builds on"
      task="Find your three milestones. A milestone that fails here will fail inside attention too."
      foot={
        <>
          Read from this machine&rsquo;s recorded runs (<code>reports/runs/</code>). Run{' '}
          <code>python3 labs/pmpp.py run 10.3</code> (or 5.3, 15.6) to record yours.{' '}
          <button className="quiet" onClick={reload}>
            Reload results
          </button>
        </>
      }
    >
      <table className="data" data-table="milestones" style={{ margin: 0, background: 'var(--page)' }}>
        <thead>
          <tr>
            <th>Milestone</th>
            <th>Call</th>
            {DEVICES.map((d) => (
              <th key={d.b}>{d.name}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {MILESTONES.map((m) => (
            <tr key={m.m}>
              <td>
                <b>{m.m}</b>, <a href={href(m.lab.split('/')[0], 'labs')}>lab {m.num}</a>
              </td>
              <td>
                <code>{m.call}</code>
                <div className="muted">{m.role}</div>
              </td>
              {DEVICES.map((d) => {
                const c = cell(reports, d.b, m.lab);
                return (
                  <td key={d.b} data-milestone={m.m} data-backend={d.b} data-work={loading ? 'loading' : c.work}>
                    {loading ? (
                      <span className="muted">reading reports</span>
                    ) : c.tone === 'none' ? (
                      <span className="muted">{c.work}</span>
                    ) : (
                      <span className={'chip ' + c.tone}>{c.work}</span>
                    )}
                    {!loading && c.ref ? <div className="muted">{c.ref}</div> : null}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="status-line" data-readiness={loading ? 'loading' : any ? 'runs' : 'none'} style={{ margin: '0.7rem 0 0' }}>
        {loading
          ? 'Reading reports/runs.'
          : any
            ? 'Lab 20.1\'s benchmark builds the unfused pipeline from the reference solutions of M3 (Q K^T) and M2 (P V); your own M4 reduction is what you reuse in its kernel.'
            : 'No runs on this machine yet. Lab 20.1\'s benchmark builds the unfused pipeline from the reference solutions of M3 (Q K^T) and M2 (P V); your own M4 reduction is what you reuse in its kernel.'}
      </p>
    </Bench>
  );
}

const ROADMAP: { sec: string; what: string; lab: string }[] = [
  { sec: '20.1, 20.2', what: 'Every block of a decoder layer as a kernel; the shape of every matrix in one head.', lab: '' },
  { sec: '20.3', what: 'Mask and softmax with one block per row; what the unfused pipeline moves.', lab: 'Lab 20.1 (M6, part one)' },
  { sec: '20.4', what: 'KV caching: a decode step is a matrix-vector pass over the whole cache.', lab: '' },
  { sec: '20.5', what: 'Flash attention: one fused kernel; S and P never reach memory.', lab: 'Labs 20.2 (M6, part two) and 20.3 (race hunt)' },
  { sec: '20.6, 20.7', what: 'Cache size, decode intensity, and MQA, GQA and MLA, which shrink the cache.', lab: 'Labs 20.4 (M7) and 20.5' },
];

export function S200() {
  return (
    <>
      <p>
        A language model does one thing at inference: given every token so far, the system prompt, your input and what
        it has already written, it predicts the next one. Attention is the part of each layer that reads that context
        and decides which earlier tokens matter for the next one. It is also the part whose cost grows with the
        context, which is why it gets a chapter.
      </p>
      <Source pages="477–478" />
      <p>
        The book tells the history in a paragraph (transformers in 2017, BERT in 2018, alternatives such as state-space
        models being explored). The rest of the chapter is kernels: a softmax, the algebra of a cache, and a fused
        attention kernel you will write. The chapter prints no timings at all, so every number on these pages that is
        not arithmetic comes from your own labs.
      </p>
      <Aside>
        <p>
          &ldquo;Architecture&rdquo; in this chapter means the network&rsquo;s, not the GPU&rsquo;s (footnote 1, p. 477).
        </p>
      </Aside>

      <h2>A number to hold on to</h2>
      <p>
        Before anything else, one prediction. Section 20.6 will size the memory that attention needs to keep around
        between tokens; commit your guess now and see how far off intuition is.
      </p>
      <Predict
        id="ch20/start/p-kv-size"
        prompt={
          <>
            Llama 2 7B, one conversation of 4,096 tokens, 16-bit numbers. How much memory does its KV cache (every
            layer&rsquo;s keys and values for every token) take?
          </>
        }
        unit="GiB"
      >
        {(v) => (
          <div data-reveal="kv-size">
            <p>
              The book&rsquo;s Eq. 20.8: batch × tokens × 2 (K and V) × layers × hidden dimension × bytes = 1 × 4096 × 2 ×
              32 × 4096 × 2 B = 2<sup>31</sup> B = <b>2 GiB</b>, or 512 KiB per token.{' '}
              {v >= 1 && v <= 4
                ? 'You were in the right range.'
                : v < 1
                  ? 'You guessed lower. Each token keeps a key and a value of 4,096 numbers in each of 32 layers: 262,144 numbers, every one of them kept for the rest of the conversation.'
                  : 'You guessed higher than the book\'s figure.'}{' '}
              One conversation&rsquo;s cache is about a seventh of the 14 GB the book gives for the model&rsquo;s weights (2 GiB is 2.15 GB),
              and it grows with every token. Section <a href={href('ch20', '20.6')}>20.6</a> lets you vary each factor.
            </p>
            <p className="muted">
              Book pp. 505–506 (printed &ldquo;2 GB&rdquo;; the unit is binary). Llama 2 7B&rsquo;s 32 layers and hidden
              size 4096 are confirmed in notes/today/ch20-facts.md E2 and F1.
            </p>
          </div>
        )}
      </Predict>

      <h2>The chapter in kernel terms</h2>
      <p>
        Each section either explains a matrix or gives you a kernel. The labs compose milestones you already have into
        the attention path of an inference engine.
      </p>
      <table className="data" data-table="roadmap">
        <thead>
          <tr>
            <th>Section</th>
            <th>What you get</th>
            <th>Lab</th>
          </tr>
        </thead>
        <tbody>
          {ROADMAP.map((r) => (
            <tr key={r.sec}>
              <td className="nowrap">{r.sec}</td>
              <td>{r.what}</td>
              <td>{r.lab ? <a href={href('ch20', 'labs')}>{r.lab}</a> : <span className="muted">none</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        One scope statement covers the whole chapter. The book&rsquo;s attention is causal, one head, one sequence,
        32-bit floats, with the head width and tile sizes fixed at compile time. Production kernels add heads, batches,
        16-bit and 8-bit types, paged caches and split decoding; where sections 20.5 to 20.7 mention those, they are
        marked as not in the book.
      </p>
      <Source section="20.5" pages="495" />

      <h2>Where you are starting from</h2>
      <p>
        Lab 20.1 does not start from nothing. Its benchmark builds the unfused pipeline out of your earlier milestones,
        and its kernel grows out of your chapter 10 softmax.
      </p>
      <Readiness />

      <h2>What this is under PyTorch</h2>
      <p>
        <code>torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)</code>, on tensors whose head
        dimension is third from the end, is Eq. 20.1 of this chapter. Its documented reference code builds a
        lower-triangular mask, fills the masked logits with minus infinity, takes a softmax and multiplies by V: the
        steps of section 20.2, literally. What usually runs is a fused kernel: on CUDA, FlashAttention-2 or a
        memory-efficient kernel, with a C++ implementation as the fallback; on Apple GPUs, PyTorch&rsquo;s own Metal
        decode and prefill kernels, adapted from MLX.
      </p>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md D5, A6; infra-facts.md D1, D2)">
        <p>
          PyTorch 2.15 documents three implementations behind SDPA. llama.cpp&rsquo;s attention lives in{' '}
          <code>ggml-cuda/fattn-*.cuh</code> (also built for HIP) and <code>ggml-metal/kernels/fa_*.metal</code>.
        </p>
      </Today>
      <p>
        One trap to file away for decode. With <code>is_causal=True</code> and a query shorter than the keys, PyTorch
        aligns the mask to the upper left, so a single new token attends to key 0 only. FlashAttention and MLX align it
        to the lower right, where the last query sees every key. Section 20.3&rsquo;s lab kernel takes the second
        convention.
      </p>
      <Beyond>
        <p>Not in the book. PyTorch 2.15&rsquo;s documentation, FlashAttention&rsquo;s changelog (2.1) and MLX&rsquo;s documentation, checked 2026-11-20 (facts D5, A8, D4).</p>
      </Beyond>

      <h2>Before you start</h2>
      <p>
        Seven questions on what this chapter assumes. Each answer says where the idea was taught and where in this
        chapter it returns; all of them come back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={questionsS200['ch20/start/q-gemv-ratio']} label="From chapter 5" />
      <Ask q={questionsS200['ch20/start/q-overflow']} label="From chapter 10" />
      <Ask q={questionsS200['ch20/start/q-lane0']} label="From chapter 10" />
      <Ask q={questionsS200['ch20/start/q-two-reductions']} label="From chapters 5 and 10" />
      <Ask q={questionsS200['ch20/start/q-qkt-linear']} label="From chapter 15" />
      <Ask q={questionsS200['ch20/start/q-coalesce']} label="From chapter 6" />
      <Ask q={questionsS200['ch20/start/q-wave64']} label="From chapters 4 and 10" />
    </>
  );
}
