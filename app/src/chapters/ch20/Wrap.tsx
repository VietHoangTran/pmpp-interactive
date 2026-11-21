import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { visibleCount } from '../../sim/attnref';
import { BOOK, METAL_DOCUMENTED, plan } from '../../sim/flashplan';
import { KiB, eq212 } from '../../sim/kvmath';
import { mergeD } from '../../sim/onlinesoftmax';

// ---------------------------------------------------------------- the numbers the questions take from the models
/** Two blocks' partial (max, denominator) for one row, merged by Eq. 20.4 (src/sim/onlinesoftmax.ts). */
export const SPLIT = { mA: 2, DA: 3, mB: 5, DB: 1.5, ...mergeD(2, 3, 5, 1.5) };
/** Verifying 4 drafted tokens against 1000 cached keys: what the first query sees (src/sim/attnref.ts). */
export const SPEC = { nq: 4, nk: 1000, first: visibleCount(0, 4, 1000), last: visibleCount(3, 4, 1000) };
/** Lab 20.2's tiles with K^T and V double-buffered, per B_c (src/sim/flashplan.ts). */
const doubled = (bc: number) => {
  const p = plan({ ...BOOK, bc });
  return { single: p.bytes.total, double: p.bytes.total + p.bytes.kt + p.bytes.v };
};
export const DOUBLE = { bc16: doubled(16), bc8: doubled(8), limit: METAL_DOCUMENTED };
/** A GQA model's cache per token in llama.cpp's q8_0 (34 bytes per 32 values, facts D3), Eq. 20.12. */
export const Q8 = { l: 32, hq: 32, hkv: 8, d: 128, bytes: eq212(1, 1, 32, 32, 4, 128, 34 / 32) };
const r3 = (v: number) => Math.round(v * 1000) / 1000;

export const questionsWrap: Record<string, Question> = {
  'ch20/wrap/q-split-merge': {
    kind: 'numeric',
    id: 'ch20/wrap/q-split-merge',
    prompt: (
      <>
        Flash-Decoding splits one head&rsquo;s keys between two blocks. Block A ends with running max {SPLIT.mA} and
        denominator {SPLIT.DA}, block B with max {SPLIT.mB} and denominator {SPLIT.DB}, each relative to its own max. What
        is the denominator of the whole row, relative to the row&rsquo;s max?
      </>
    ),
    answer: r3(SPLIT.D),
    tol: 0.004,
    explain: (
      <p>
        The row&rsquo;s max is {SPLIT.m}. Block A&rsquo;s terms were measured against {SPLIT.mA}, so each is e
        <sup>{SPLIT.mA} − {SPLIT.m}</sup> = {r3(SPLIT.aFactor)} of what it is against {SPLIT.m}: {SPLIT.DA} ×{' '}
        {r3(SPLIT.aFactor)} + {SPLIT.DB} × 1 = {r3(SPLIT.D)}. That is Eq. 20.4 with the two blocks as the disjoint sets A
        and B. The outputs merge with the same two factors, and the row&rsquo;s logsumexp is {SPLIT.m} + ln{' '}
        {r3(SPLIT.D)} = {r3(SPLIT.m + Math.log(SPLIT.D))}, the one number per row and split that Flash-Decoding writes.
        Lab 20.4 does this merge across the warps of a block; section <a href={href('ch20', '20.5')}>20.5</a> has the
        algebra.
      </p>
    ),
    source: 'Eq. 20.4, p. 494; Flash-Decoding, notes/today/ch20-facts.md A8; the numbers are ours',
  },
  'ch20/wrap/q-spec-mask': {
    kind: 'numeric',
    id: 'ch20/wrap/q-spec-mask',
    prompt: (
      <>
        Speculative decoding verifies {SPEC.nq} drafted tokens in one pass: attention with nq = {SPEC.nq} queries against
        nk = {SPEC.nk} keys, the drafted tokens included. Under lab 20.1&rsquo;s lower-right causal mask, how many keys does
        the first of the {SPEC.nq} queries see?
      </>
    ),
    answer: SPEC.first,
    tol: 0,
    unit: 'keys',
    explain: (
      <p>
        Row r sees columns c ≤ r + (nk − nq). For r = 0 that is columns 0 to {SPEC.nk - SPEC.nq}: {SPEC.first} keys,
        everything before the drafts and the first draft itself. The last query sees all {SPEC.last}. An upper-left mask,
        PyTorch&rsquo;s <code>is_causal</code> when nq &lt; nk, would let the first query see one key. Fig. 20.18 checks
        the drafts in parallel; this is the mask that makes each check see exactly its own prefix.
      </p>
    ),
    source: 'Fig. 20.18, p. 507; lab 20.1 README (lower-right alignment); facts A8, D5; the setting is ours',
  },
  'ch20/wrap/q-vocab': {
    kind: 'choice',
    id: 'ch20/wrap/q-vocab',
    prompt: (
      <>
        After the last layer you need the softmax of 128,000 vocabulary logits for one sequence. You call lab 20.1&rsquo;s{' '}
        <code>launchCausalSoftmax</code> with batch 1, nq = 1, nk = 128,000. It is correct. Why is it slow?
      </>
    ),
    options: [
      {
        label: 'The grid is one block: 256 threads take the whole row while the GPU idles.',
        correct: true,
      },
      {
        label: 'The lower-right mask hides most of the vocabulary, so the row is read many times.',
        why: 'With nq = 1 the single row sees every column, and each column is read three times, as in any row.',
      },
      {
        label: 'Logits that wide overflow fp32, so the kernel falls back to slower arithmetic.',
        why: 'Subtracting the max keeps every exponent at or below 0, whatever the row length.',
      },
      {
        label: 'Its two block reductions cost far more than its three passes over the row.',
        why: 'Each reduction is a few shuffle steps and one barrier, once per row. The passes are 500 reads per thread.',
      },
    ],
    explain: (
      <p>
        The grid is one block per row, and here there is one row: 256 threads read 500 columns each, three times. The cure is the one this chapter keeps returning to:
        split the row across many blocks, let each produce its own max and denominator, and merge them with Eq. 20.4, as
        Flash-Decoding does for keys. Module X1 builds the sampling kernels that follow this softmax; the book applies
        Fig. 20.4 only to attention scores.
      </p>
    ),
    source: 'Fig. 20.4, pp. 486–488; Eq. 20.4, p. 494; the setting is ours',
  },
  'ch20/wrap/q-double-buffer': {
    kind: 'choice',
    id: 'ch20/wrap/q-double-buffer',
    prompt: (
      <>
        A colleague double-buffers lab 20.2&rsquo;s K<sup>T</sup> and V tiles, two of each chosen by <code>j &amp; 1</code>,
        and deletes the barrier after the warp loop, keeping the one after each load. On a device with 32 KB of
        threadgroup memory that ran your kernel at B_c = 16, what happens?
      </>
    ),
    options: [
      {
        label: 'Wrong results: a fast warp loads into a tile that slow warps are still reading from.',
        why: 'Tile j + 1 goes into the buffer last read in iteration j − 1, and a warp passes iteration j\'s load barrier only after every warp has finished iteration j − 1.',
      },
      {
        label: 'Correct, but the tiles outgrow 32 KB at B_c = 16; the host must pick B_c = 8.',
        correct: true,
      },
      {
        label: 'Correct, and nothing else changes: one barrier per tile is simply a gain.',
        why: 'Two of each tile is twice the shared memory for K^T and V, and the limit does not move.',
      },
      {
        label: 'Wrong results unless a __syncwarp() takes the deleted barrier\'s place.',
        why: 'The hazard the deleted barrier guarded was between warps; a warp barrier orders nothing between them.',
      },
    ],
    explain: (
      <p>
        The load barrier of iteration j already proves that every warp finished iteration j − 1, which is the last reader
        of the buffer iteration j + 1 overwrites: the same argument as lab 15.5&rsquo;s double buffering. Each warp
        touches only its own rows of S_i, so that tile needs no second copy. The cost is space: at B_c = 16 the tiles grow
        from {DOUBLE.bc16.single.toLocaleString('en-US')} to {DOUBLE.bc16.double.toLocaleString('en-US')} bytes, over{' '}
        {DOUBLE.limit.toLocaleString('en-US')}; at B_c = 8 they take {DOUBLE.bc8.double.toLocaleString('en-US')}. Lab
        20.2&rsquo;s host code would pick 8. Whether one barrier fewer pays for tiles half as wide is a measurement.
      </p>
    ),
    source: 'Fig. 20.9 line 39, p. 496; double buffering, ch. 15 §15.8; facts C1; the variant is ours (src/sim/flashplan.ts)',
  },
  'ch20/wrap/q-q8-cache': {
    kind: 'numeric',
    id: 'ch20/wrap/q-q8-cache',
    prompt: (
      <>
        A GQA model has {Q8.l} layers, {Q8.hq} query heads and {Q8.hkv} K/V heads of {Q8.d}. llama.cpp stores its cache
        as q8_0: blocks of 32 values in 34 bytes. How many KiB does one token&rsquo;s cache take?
      </>
    ),
    answer: Q8.bytes / KiB,
    tol: 0.01,
    unit: 'KiB',
    explain: (
      <p>
        Eq. 20.12 per token: 2 × {Q8.l} layers × {Q8.hkv} K/V heads × {Q8.d} = 65,536 values, at 34/32 bytes each:{' '}
        {Q8.bytes.toLocaleString('en-US')} bytes, {Q8.bytes / KiB} KiB. The same model with 32 K/V heads in fp16 would
        take 512 KiB per token: grouping gives 4×, 8-bit storage almost 2× more. Both also cut the bytes a decode step
        reads per token, which is what sets its time (section <a href={href('ch20', '20.6')}>20.6</a>). Module{' '}
        <a href={href('x2', 'X2.7')}>X2</a> (X2.7) quantises the weights of a decode step that has no attention, so it has
        no cache to quantise; no lab in the course builds a quantised cache.
      </p>
    ),
    source: 'Eq. 20.12, p. 509; q8_0 block size, notes/today/ch20-facts.md D3; the model shape is ours',
  },
};

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports, loading } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const name = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const pass = (lab: string) => backends.filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const m6 = pass('ch20/lab1_causal_softmax').filter((b) => pass('ch20/lab2_flash').includes(b));
  const m7 = pass('ch20/lab4_decode');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch20/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <p>
        The book&rsquo;s summary is a paragraph: unfused attention, then the KV cache, then flash attention, then what the
        cache costs in memory and how models shrink it. The parts to keep:
      </p>
      <Source section="20.8" pages="510" />
      <ul>
        <li>
          Attention is two GEMMs with a row softmax between them. Unfused, S and P are N × N matrices written to and read
          back from device memory; the softmax kernel is a block-per-row reduction with the mask as its loop bound.
        </li>
        <li>
          A softmax over disjoint pieces can be merged exactly if each piece carries its max: rescale each piece&rsquo;s
          denominator and output by e<sup>m_piece − m_all</sup> and add. Tiles of a row, warps of a block, blocks of a
          decode step and splits of a vocabulary all merge with that one rule.
        </li>
        <li>
          Flash attention keeps a running max, denominator and unnormalised output per row and divides once at the end.
          S and P exist only one tile at a time in shared memory, so the N × N traffic disappears; what is left is two
          small matrix products per tile.
        </li>
        <li>
          Barriers follow data, not code. The book&rsquo;s listing omits the warp barrier between the P write and{' '}
          <code>compute_O</code>, and a GPU may pass every test without it: shuffles do not order memory, but nothing forces the lanes apart either.
        </li>
        <li>
          With a KV cache, adding a token changes one row of Q, K, V, S, P and O. Prefill is a GEMM and compute-bound;
          decode is a matrix-vector pass over the whole cache, about one operation per byte, and batching does not help it
          because every sequence has its own cache.
        </li>
        <li>
          The cache costs 2 · layers · K/V heads · d · bytes per token. MQA, GQA, MLA and narrower types shrink it and
          raise decode&rsquo;s intensity; paging removes the memory reserved for tokens that never come.
        </li>
        <li>
          Every number here that is not arithmetic came from your labs. The book prints no timing, and flash attention in
          fp32 warp loops is not automatically faster than tuned GEMMs.
        </li>
      </ul>

      <h2>A mixed check</h2>
      <p>
        These use the whole chapter on situations the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Beyond>
        <p>The settings and the colleague&rsquo;s variant in these questions are ours, picked for the questions.</p>
      </Beyond>
      <Ask q={q['ch20/wrap/q-split-merge']} />
      <Ask q={q['ch20/wrap/q-spec-mask']} />
      <Ask q={q['ch20/wrap/q-vocab']} />
      <Ask q={q['ch20/wrap/q-double-buffer']} />
      <Ask q={q['ch20/wrap/q-q8-cache']} />

      <h2>The book&rsquo;s exercises</h2>
      <p>
        The chapter ends with four exercises (p. 511), all about the flash attention kernel, and the book gives no
        solution for any of them. Each is worked here.
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
              <td>Host code that launches the kernel of Fig. 20.9</td>
              <td>
                <code>launchFlashAttention</code> in lab 20.2, and its Metal host code, which also chooses B_c; on the{' '}
                <a href={href('ch20', 'labs')}>labs page</a>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>CUB&rsquo;s <code>WarpReduce</code>: its types, its arguments, its operators</td>
              <td>
                section <a href={href('ch20', '20.3')}>20.3</a> (what changed in CUB, and the question on porting Fig.
                20.4), the start page&rsquo;s question on where the result lands, and section{' '}
                <a href={href('ch20', '20.5')}>20.5</a>&rsquo;s table of corrections
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>
                <code>initialize</code>, and why O<sub>i</sub> and D<sub>i</sub> start at 0
              </td>
              <td>
                the question and the online-softmax stepper in <a href={href('ch20', '20.5')}>20.5</a> (try the
                minus-infinity start); lab 20.2
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td>Every load from K and V in Fig. 20.11 is coalesced</td>
              <td>
                the question in <a href={href('ch20', '20.5')}>20.5</a>; the same layout question on the{' '}
                <a href={href('ch20', 'start')}>start page</a> for Q
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy this course was built from has apparent slips in chapter 20, some in the code and some in the text. The
        code ones change what you write, so the labs use corrected, tested forms.
      </p>
      <Aside>
        <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
      </Aside>
      <ul data-list="errata">
        <li>
          p. 487: &ldquo;gridDim.y and gridDim.z are 0&rdquo; for a one-dimensional grid. Unused dimensions are 1. See{' '}
          <a href={href('ch20', '20.3')}>20.3</a>.
        </li>
        <li>
          Figs. 20.4, 20.12 and 20.15 use <code>NEG_INFINITY</code>, which no CUDA header defines, and{' '}
          <code>cub::Max()</code> and <code>cub::Sum()</code>, which current CUB no longer has. Figs. 20.12 and 20.15
          also give all 16 warps one <code>WarpReduce::TempStorage</code>, which is not CUB&rsquo;s documented use. The
          labs use finite start values and hand-written reductions (<a href={href('ch20', '20.3')}>20.3</a>,{' '}
          <a href={href('ch20', '20.5')}>20.5</a>).
        </li>
        <li>
          Figs. 20.13 and 20.15: each lane writes one element of its warp&rsquo;s P row and then reads the whole row, with
          no barrier between. A <code>__syncwarp()</code> is needed; lab 20.3 is about it.
        </li>
        <li>
          Fig. 20.9 divides N by B_r and B_c, so N must be a multiple of both, which the text does not say. The labs round
          up and mask by index.
        </li>
        <li>
          Fig. 20.16 writes the final denominator D &ldquo;for training&rdquo;. Without the row max, D cannot rebuild P;
          FlashAttention-2 stores the logsumexp m + log D, and so do the labs.
        </li>
        <li>
          p. 495: in the first line of the continued derivation of Eq. 20.6, the second exponent is printed l − m<sub>B</sub>{' '}
          where l − m<sub>A∪B</sub> is meant (<a href={href('ch20', '20.5')}>20.5</a>).
        </li>
        <li>
          p. 502: line 6 of Fig. 20.14 is said to compute the B term of Eq. 20.4; it rescales the old denominator, the A
          term. On the same page the sum of Fig. 20.15 is said to use m<sub>B</sub>; the code uses the updated max (
          <a href={href('ch20', '20.5')}>20.5</a>).
        </li>
        <li>
          p. 499: the text calls the loader <code>load_KTV()</code>, the listing <code>load_KT_and_V</code>; V is tagged
          with marker 5, which in Fig. 20.8 is the P tile; and <code>addr()</code> is said to come from chapter 16, where
          the macro is <code>pad()</code>.
        </li>
        <li>
          p. 510, Eq. 20.15: the middle expression should have 2h<sub>q</sub> + N in the denominator. The limit, about 2h
          <sub>q</sub>, is unaffected (<a href={href('ch20', '20.7')}>20.7</a>).
        </li>
        <li>
          pp. 508–509: g<sub>q</sub> is called the number of groups, but Eq. 20.12 stores h<sub>q</sub>/g<sub>q</sub>{' '}
          K/V heads, which makes g<sub>q</sub> the query heads per K/V head (<a href={href('ch20', '20.7')}>20.7</a>).
        </li>
        <li>
          Fig. 20.1&rsquo;s attention call-out ends with &ldquo;S x V&rdquo;; it is P × V. p. 481 says a model&rsquo;s
          parameter count is the product of depth, head dimension and head count, which it is not (
          <a href={href('ch20', '20.1')}>20.1</a>).
        </li>
        <li>
          p. 489 calls the new column of Q K<sup>T</sup> &ldquo;all zero elements&rdquo;; the masked logits are −∞ and
          become 0 only after the softmax (<a href={href('ch20', '20.4')}>20.4</a>).
        </li>
        <li>
          p. 503 says the new contributions use the max &ldquo;up to the new row&rdquo;; &ldquo;new tile&rdquo; seems
          meant (<a href={href('ch20', '20.5')}>20.5</a>).
        </li>
        <li>
          p. 506, the &ldquo;PaLM 2&rdquo; row: PaLM 2&rsquo;s shape was never published, and the numbers are PaLM
          62B&rsquo;s, which uses multi-query attention with one K/V head of 256. Its 4,096-token 16-bit cache is 256 MiB,
          not 8 GB (facts E2; <a href={href('ch20', '20.6')}>20.6</a>).
        </li>
        <li>
          p. 506, the GPT-3 row: GPT-3&rsquo;s context window is 2,048 tokens, so a 4,096-token cache does not occur; at
          its window the cache is 9 GiB, not 18 (facts E2; <a href={href('ch20', '20.6')}>20.6</a>).
        </li>
        <li>
          pp. 509–510, MLA: the latent is one per token per layer, not shared across layers, and DeepSeek-V2&rsquo;s is
          4.5 head widths, not d, so Eq. 20.14 understates its cache 4.5 times. K and V are absorbed into the other
          projections, not decompressed before use (facts E5; <a href={href('ch20', '20.7')}>20.7</a>).
        </li>
        <li>
          p. 509, PagedAttention: the blocks are not loaded from system memory when needed. They live anywhere in GPU
          memory through a block table; host memory is only where evicted blocks go under memory pressure (facts E6;{' '}
          <a href={href('ch20', '20.7')}>20.7</a>).
        </li>
        <li>
          p. 506, in-flight batching: not &ldquo;slotting a new request into the batch with the lowest workload&rdquo;.
          Finished requests leave the batch and new ones join after every iteration, as Orca and TensorRT-LLM describe
          it (facts E6; <a href={href('ch20', '20.6')}>20.6</a>).
        </li>
        <li>
          Small slips: Fig. 20.3 draws a marker 4 that the text never mentions; the KV-cache sizes on p. 506 are in binary
          gigabytes.
        </li>
      </ul>
      <p>
        The code corrections rest on the vendors&rsquo; current documentation, quoted in the facts file, not on memory.
      </p>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md A1, A2, A3, A5, A7, E2, E3, E5, E6; infra-facts A11)">
        <p>
          Each code correction above is checked against current CUDA, CUB and FlashAttention-2 documentation, and each
          factual one against the cited papers; the facts file gives the quotations.
        </p>
      </Today>

      <h2>What this course shortened</h2>
      <p>
        Every section of the chapter has its own page. Kept to a sentence or a pointer: the chapter&rsquo;s history and
        terminology footnotes (pp. 477–478); tokenizer, embedding and positional encoding (pp. 479–480), which you know
        and no kernel here depends on; why heads specialise (p. 482); retrieval-augmented generation (p. 481); the
        persistent grid with a grid-wide barrier (p. 493), whose mechanism <a href={href('ch18', '18.6')}>section
        18.6</a> covers; FlashAttention-3&rsquo;s
        Hopper techniques (p. 504); and mixture of experts (p. 510), one paragraph and a question. Not covered: training
        and the backward pass (pp. 482, 493), which the book defers to its reference [6]. LayerNorm and the feed-forward
        of Fig. 20.1 are classified in <a href={href('ch20', '20.1')}>20.1</a>, not implemented.
      </p>
      <p>
        Not in the book and not built in this chapter, though named where they matter: half-precision and tensor-core
        attention, split-KV decoding across blocks, paged-cache kernels, sliding windows, rotary embeddings and quantised
        caches. Of these, only rotary embeddings are built elsewhere in the course, in module X3 (
        <a href={href('x3', 'X3.7')}>X3.7, the Llama step: RMSNorm, RoPE and SwiGLU</a>, and its first lab).
      </p>

      <h2>The project so far</h2>
      <p>
        Labs 20.1 and 20.2 are milestone M6: causal attention for a prefill, unfused and fused, both writing the same
        logsumexp. Labs 20.4 and 20.5 are milestone M7: one decode step with a KV cache, per head and grouped.{' '}
        <span data-milestones={loading ? 'loading' : `m6:${m6.join(',') || 'none'};m7:${m7.join(',') || 'none'}`}>
          {loading
            ? 'Reading reports.'
            : backends.length === 0
              ? 'No lab results are recorded on this machine yet.'
              : `${m6.length ? `Your M6 passes on ${m6.map(name).join(' and ')} on this machine` : 'Your M6 does not pass on this machine yet'}; ${m7.length ? `your M7 passes on ${m7.map(name).join(' and ')}` : 'your M7 does not pass yet'}.`}
        </span>{' '}
        See the <a href={href('ch20', 'labs')}>labs</a>. With M2 to M7 the project has the matrix products and the
        attention of a decoder layer, for a prefill and for one decode step at a time.
      </p>
      <p>
        <a href={href('ch21')}>Chapter 21</a> is the book&rsquo;s next case study, on scatter versus gather. For the
        project, the next stops are <a href={href('ch22')}>chapter 22</a>, whose batching study, milestone M8, measures
        what batching a decode step buys in throughput and costs in latency, and{' '}
        <a href={href('x1')}>module X1</a>, which samples the next token from the softmax this chapter&rsquo;s layers
        feed.
      </p>
    </>
  );
}
