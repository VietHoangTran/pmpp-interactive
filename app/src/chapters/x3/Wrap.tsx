import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// Module X3's summary: what to carry forward, a mixed check on situations the sections did
// not show, where each of the module's tasks is worked (the book has no exercises for it),
// where the sources disagree (the module has no PDF errata), and milestone M10.

export const questionsWrap: Record<string, Question> = {
  'x3/wrap/q-permute-row': {
    kind: 'numeric',
    id: 'x3/wrap/q-permute-row',
    prompt: (
      <>
        You convert SmolLM2-135M's <code>k_proj.weight</code>, 192 rows (3 KV heads of 64) by 576, for llama.cpp's NORM
        pairing with its converter's <code>permute(weights, n_head = 9, n_head_kv = 3)</code>. Within a head it sends row
        k × 32 + j to row 2j + k. Row 100 of the Hugging Face tensor lands at which row of the converted one?
      </>
    ),
    answer: 73,
    unit: 'row',
    explain: (
      <p>
        For K the function switches to <code>n_head_kv</code>, so the heads are 192 / 3 = 64 rows. Row 100 is row 36 of
        head 1, and 36 = 1 × 32 + 4, so k = 1, j = 4, and it goes to 2 × 4 + 1 = 9 of that head: 64 + 9 = 73. Its partner
        under Hugging Face's half-split pairing, row 68 (row 4 of head 1), goes to row 72. The two rows that one rotation
        mixes now sit next to each other, which is what the consecutive pairing expects. Permute K with 9 heads instead and
        192 / 9 / 2 is not a whole number, so the reshape fails.
      </p>
    ),
    source: 'notes/today/x3-facts.md B6 (permute), D4 (shapes), D6; the arithmetic is ours',
  },
  'x3/wrap/q-batch-flip': {
    kind: 'choice',
    id: 'x3/wrap/q-batch-flip',
    prompt: (
      <>
        With llama.cpp on an NVIDIA GPU and float32 weights, one prompt decoded alone and the same prompt decoded as one of
        16 sequences in a batch give the same greedy tokens up to step 30, then different ones. Neither run has a bug. What
        is the likely reason?
      </>
    ),
    options: [
      {
        label: 'At batch 16 the products run on another kernel that adds in another order, and one step’s margin was below that difference.',
        correct: true,
      },
      {
        label: 'In a batched decode, cache rows of the other 15 sequences leak into this sequence’s attention once the cache grows long.',
        why: 'That would be a bug, and it would change the logits at every step, not flip one token at step 30.',
      },
      {
        label: 'Greedy decoding still draws one random number per step, and the batch changed which number this sequence received.',
        why: 'Greedy takes the argmax: no random number is drawn (llama2.c at -t 0.0; M4).',
      },
      {
        label: 'GPU float addition rounds differently on every run, so two runs of any model drift apart after a few dozen decoding steps.',
        why: 'The same kernel adding the same values in the same order gives the same bits. What changed here is the kernel, and so the order.',
      },
    ],
    explain: (
      <p>
        llama.cpp picks its matrix-product kernel by the number of tokens: vector kernels up to 8, then a matrix kernel or
        cuBLAS. Each adds the products in its own order, and float addition is not associative, so the logits differ in
        their last bits. That is harmless until a step's top two logits are closer than the difference; then the token
        flips, and every later step sees another input. This is why lab X3.2 checks batch invariance on one kernel and
        checks tokens only where the margin is large.
      </p>
    ),
    source: 'notes/today/x3-facts.md B4, D7; ch22-facts.md D6; ch10-facts.md A9',
  },
  'x3/wrap/q-nmse-zero': {
    kind: 'choice',
    id: 'x3/wrap/q-nmse-zero',
    prompt: (
      <>
        You add an op to llama.cpp whose output is exactly 0 in almost every element, because most inputs are masked. Your
        CUDA kernel is within 1e-6 of the CPU everywhere, yet <code>test-backend-ops</code> reports an NMSE far above
        1e-7. What is going on?
      </>
    ),
    options: [
      {
        label: 'NMSE divides the squared error by the reference’s sum of squares, tiny here, so small errors give a large ratio.',
        correct: true,
      },
      {
        label: 'Your kernel must have written a NaN somewhere, because an NMSE that large cannot come from rounding errors alone.',
        why: 'A NaN would make the NMSE NaN. Ordinary errors can give any ratio when the denominator is small.',
      },
      {
        label: 'The tool compares your kernel with another GPU backend on the machine, and that backend’s rounding differs from yours.',
        why: 'By default it compares each backend with the CPU backend.',
      },
      {
        label: 'The 1e-7 threshold is meant for a whole model; a single op is held to a much stricter threshold of its own.',
        why: 'The other way round: 1e-7 is the per-op default, which an op’s test may override.',
      },
    ],
    explain: (
      <p>
        <code>nmse</code> sums (a − b)² over the elements and divides by the sum of a², the reference's own size. That makes
        it a relative error, which suits outputs with some magnitude. Here the denominator is the squares of the few
        unmasked outputs, so errors of 1e-6 everywhere else can dominate it. An op like this needs its own threshold, which
        the tool lets a test override, or test inputs whose output is not mostly zero.
      </p>
    ),
    source: 'notes/today/x3-facts.md D7 (nmse, max_nmse_err); the diagnosis is ours',
  },
  'x3/wrap/q-rms-dtype': {
    kind: 'choice',
    id: 'x3/wrap/q-rms-dtype',
    prompt: (
      <>
        You profile a PyTorch model on CUDA. You expect each <code>torch.nn.functional.rms_norm</code> to be one fused
        kernel, but you see several small kernels per norm and a warning. The norm's weight is bfloat16 and the activations
        are float32. What do you check first?
      </>
    ),
    options: [
      {
        label: 'The dtypes: when weight and input differ, rms_norm warns “Cannot dispatch to fused implementation.”',
        correct: true,
      },
      {
        label: 'Whether this GPU has bfloat16 arithmetic, since the fused kernel can only run where bfloat16 is native.',
        why: 'The dispatch path read in the facts pass tests the tensors’ dtypes, not the device.',
      },
      {
        label: 'Whether the model is really on CUDA, since the fused RMSNorm is registered only for Apple GPUs.',
        why: 'native_functions.yaml lists CUDA: _fused_rms_norm_cuda beside MPS: _fused_rms_norm_mps.',
      },
      {
        label: 'Whether the rows are longer than 2,048 elements, the limit of the warp-per-row kernel PyTorch uses for short rows.',
        why: 'That limit belongs to softmax’s persistent kernel (ch10-facts D4), not to RMSNorm.',
      },
    ],
    explain: (
      <p>
        <code>rms_norm</code> calls <code>at::_fused_rms_norm</code>, which dispatches to <code>_fused_rms_norm_cuda</code>
        . It falls back to a composite of ordinary tensor ops for complex input, and warns "Cannot dispatch to fused
        implementation." when the weight's dtype differs from the input's: several kernels instead of one. Give the weight
        the activations' dtype, or the other way round, and the fused kernel can run.
      </p>
    ),
    source: 'notes/today/x3-facts.md A3, B2 (layer_norm.cpp lines 345 to 372)',
  },
};

// ---------------------------------------------------------------- milestone M10
/** Every lab M10's step runs, as lab X3.2's provided/pieces.cmake copies them. */
const M10: { num: string; chapter: string; title: string; dir: string; need: 'required' | 'if present' | 'yours' }[] = [
  { num: '2.3', chapter: 'ch02', title: 'M0: the residual add', dir: 'ch02/lab3_elementwise', need: 'required' },
  { num: '6.4', chapter: 'ch06', title: 'M2: the linear layer', dir: 'ch06/lab4_linear', need: 'required' },
  { num: '10.3', chapter: 'ch10', title: 'M4: argmax', dir: 'ch10/lab3_softmax_argmax', need: 'required' },
  { num: '20.4', chapter: 'ch20', title: 'M7: the KV append and decode attention', dir: 'ch20/lab4_decode', need: 'required' },
  { num: '22.2', chapter: 'ch22', title: 'M8: the embedding gather', dir: 'ch22/lab2_batching', need: 'required' },
  { num: 'X3.1', chapter: 'x3', title: 'RMSNorm, RoPE and SwiGLU', dir: 'x3/lab1_llama_pieces', need: 'required' },
  { num: '15.6', chapter: 'ch15', title: 'M3: the register-tiled linear layer', dir: 'ch15/lab6_linear', need: 'if present' },
  { num: '20.5', chapter: 'ch20', title: 'M7: grouped-query decode attention', dir: 'ch20/lab5_gqa_decode', need: 'if present' },
  { num: '22.1', chapter: 'ch22', title: 'The gather GEMV', dir: 'ch22/lab1_gemv', need: 'if present' },
  { num: 'X1.3', chapter: 'x1', title: 'M5: the sampler', dir: 'x1/lab3_sampler', need: 'if present' },
  { num: 'X3.2', chapter: 'x3', title: 'M10: the step that composes them', dir: 'x3/lab2_tinyllama', need: 'yours' },
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
  const own = passingOn(reports, 'x3/lab2_tinyllama');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('x3/') && !k.includes('/p-'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          Every project has one place where an op becomes a backend call. In llama.cpp it is a switch over{' '}
          <code>GGML_OP_*</code> per backend, after a fusion pass; in MLX an <code>eval_gpu</code> per primitive in each
          backend's directory; in PyTorch the dispatch keys of <code>native_functions.yaml</code>, and for some ops a kernel
          generated at build time. Start there and follow one name.
        </li>
        <li>
          On Metal, follow the string, not the function name: the host asks for a pipeline by the name in a{' '}
          <code>[[host_name]]</code>, and variants are function constants where CUDA code uses template parameters. HIP
          compiles llama.cpp's and PyTorch's CUDA sources; MLX has no AMD backend.
        </li>
        <li>
          The kernels are the course's. An elementwise op is chapter 2's kernel, coarsened with vector loads in MLX and
          PyTorch; RMSNorm and softmax are chapter 10's two-stage block reduction; a decode GEMV is lab 22.1's. Read for
          what differs: the exponential, the number of passes over the row, the tail.
        </li>
        <li>
          The shape picks the kernel: a vector kernel for one or a few tokens, a tiled kernel or the vendor's library for
          more. Attention may be in the tree, in a library or in a submodule.
        </li>
        <li>
          A Llama step is the course's kernels plus RMSNorm, RoPE and SwiGLU. RoPE has two pairings; moving a checkpoint
          between them permutes the Q and K rows, with the KV head count for K. A real checkpoint also differs in dtype,
          head size and constants, and its size checks its header.
        </li>
        <li>
          Compare with a reference by an error measure on the logits, and require equal greedy tokens only where the top-two
          margin is larger than that error. A flip at a close step changes every later token without any kernel being
          wrong.
        </li>
      </ul>
      <Beyond title="Ours, not the book's">
        <p>
          The book reads no project's source. Every claim above about llama.cpp, MLX and PyTorch was read in the facts pass
          on 2026-11-20 at the commits the <a href={href('x3', 'start')}>start page</a> lists.
        </p>
      </Beyond>

      <h2>A mixed check</h2>
      <p>
        These draw on the whole module, on situations the sections did not show.
        {asked ? ` So far in this module you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Ask q={q['x3/wrap/q-permute-row']} label="Follow the row" />
      <Ask q={q['x3/wrap/q-batch-flip']} />
      <Ask q={q['x3/wrap/q-nmse-zero']} />
      <Ask q={q['x3/wrap/q-rms-dtype']} label="Read the profile" />

      <h2>The module's exercises</h2>
      <p>
        The book has no exercises for this module, because it reads no real backend. The plan's goal for X3 is to find the
        kernel for a given op in each of three real codebases and explain it, and its milestone M10 is a Llama-style model
        checked token by token against a reference. Here is where each part of that is worked.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr><th>Task</th><th>Worked in a section</th><th>Built in a lab</th></tr>
        </thead>
        <tbody>
          <tr><td>Find the kernel for an op in each project</td><td>the op map, routes and scene in <a href={href('x3', 'X3.1')}>X3.1</a></td><td>none: reading only</td></tr>
          <tr><td>Name the HIP and Metal counterpart of a CUDA kernel</td><td>the op map's HIP column in <a href={href('x3', 'X3.1')}>X3.1</a>; variants in <a href={href('x3', 'X3.2')}>X3.2</a></td><td>lab X3.1: one <code>.cu</code> for CUDA, HIP and the emulators, and its Metal files</td></tr>
          <tr><td>Explain an elementwise or gather kernel</td><td>the launch-shape explorer in <a href={href('x3', 'X3.2')}>X3.2</a></td><td>lab X3.1's SwiGLU</td></tr>
          <tr><td>Explain a row reduction</td><td>stepper and scene in <a href={href('x3', 'X3.3')}>X3.3</a></td><td>lab X3.1's RMSNorm, and the port review of part B</td></tr>
          <tr><td>Predict the matrix-product kernel from the shape</td><td>the route explorer in <a href={href('x3', 'X3.4')}>X3.4</a></td><td>lab X3.2's benchmark, gather against M2 and M3</td></tr>
          <tr><td>Find an attention kernel, in the tree or not</td><td><a href={href('x3', 'X3.5')}>X3.5</a></td><td>lab X3.2 runs lab 20.4's or 20.5's decode kernel</td></tr>
          <tr><td>Predict a sampler's token on a tie</td><td>the tie stepper in <a href={href('x3', 'X3.6')}>X3.6</a></td><td>lab X3.2's greedy step, M4's lowest index</td></tr>
          <tr><td>Lay out a Llama step; keep the RoPE pairings apart</td><td>stepper, explorer and scenes in <a href={href('x3', 'X3.7')}>X3.7</a></td><td>lab X3.1's RoPE, lab X3.2's step and its pairing test</td></tr>
          <tr><td>Read a checkpoint's header; check a download</td><td>decoder and explorer in <a href={href('x3', 'X3.8')}>X3.8</a></td><td>lab X3.2's loader and its size check</td></tr>
          <tr><td>Test against a reference; prepare a contribution</td><td>explorer and scene in <a href={href('x3', 'X3.9')}>X3.9</a></td><td>lab X3.2's teacher-forced and free-running tests</td></tr>
          <tr><td>Run a small real checkpoint and compare tokens</td><td>the conversions in <a href={href('x3', 'X3.8')}>X3.8</a></td><td>lab X3.2 runs a generated model; its README describes the real-checkpoint path, unverified</td></tr>
        </tbody>
      </table>
      <p>
        The book's own exercises on the kernels the projects use are mapped on their chapters' summary pages: elementwise
        kernels in <a href={href('ch02', 'wrap')}>chapter 2</a>, coarsening and vector loads in{' '}
        <a href={href('ch06', 'wrap')}>chapter 6</a>, reduction in <a href={href('ch10', 'wrap')}>chapter 10</a>, matrix
        products in <a href={href('ch15', 'wrap')}>chapter 15</a> and attention in <a href={href('ch20', 'wrap')}>chapter
        20</a>.
      </p>

      <h2>Where the sources disagree</h2>
      <p>
        This module has no PDF to be wrong: it is not in the book. What it has instead are three projects that disagree
        with each other, and with what a reader would assume. These are the disagreements the module was built around, all
        read in the facts pass.
      </p>
      <ul>
        <li>
          <b>Where the kernel is.</b> PyTorch's CUDA <code>add</code> has no source file: torchgen writes it at build time.
          PyTorch's CUDA flash attention is in a submodule, and MLX's CUDA attention calls cuDNN for prefill and long
          decodes.
        </li>
        <li>
          <b>One project, two backends.</b> MLX's softmax takes the maximum and then the sum on Metal, with{' '}
          <code>fast::exp</code>, and is a one-pass online normaliser on CUDA, with <code>__expf</code>. On Metal, PyTorch's
          softmax uses <code>precise::exp</code> where MLX uses <code>fast::exp</code>.
        </li>
        <li>
          <b>One kernel per op.</b> No project sends every matrix product to one kernel: all three pick by shape.
        </li>
        <li>
          <b>RoPE.</b> llama.cpp runs Llama with consecutive pairs and permutes Hugging Face's Q and K rows to fit; Hugging
          Face, mlx-lm's default and SmolLM2 use the half-split pairing; llama2.c consecutive pairs; PyTorch has no RoPE op.
          llama2.c's <code>export.py</code> permutes K with the query head count, which by our reading does not fit a
          grouped-query model.
        </li>
        <li>
          <b>Ties and AMD.</b> llama2.c and <code>torch.argmax</code> keep the first maximum; llama.cpp's GPU argmax kernels
          do not. llama.cpp's and PyTorch's CUDA files are their HIP files; MLX has no AMD backend.
        </li>
        <li>
          <b>Formats.</b> GGUF is what llama.cpp loads, but not the simplest first format; none of the small real
          checkpoints has head size 128.
        </li>
      </ul>
      <Today checked="2026-11-20 in the facts pass (notes/today/x3-facts.md A3, B3 to B6, B9, D5, D6, D9, E2; ch10-facts.md D2)">
        <p>The "Summary of corrections" in notes/today/x3-facts.md lists every item with its source.</p>
      </Today>
      <p>
        Three places where our own plan and design were wrong, found while building the module. The plan's M10 says "run a
        small real checkpoint"; none fits M7's head size of 128 and the build must download nothing, so lab X3.2 runs a
        generated model and describes the real path, unverified. The design asked for a test that any nonzero offset
        sends every thread of MLX's <code>binary_vv</code> to its scalar path; that is false for two-wide loads with an
        offset of two floats, and section <a href={href('x3', 'X3.2')}>X3.2</a> uses the exact rule. And the design said
        MLX's Metal softmax reads the row three times; the facts give only the order of its reductions, so section{' '}
        <a href={href('x3', 'X3.3')}>X3.3</a> does not count its reads.
      </p>
      <Aside>
        <p>These are our readings of the sources on 2026-11-20. Projects move; check the file before you rely on a line.</p>
      </Aside>

      <h2>The project so far</h2>
      <p>
        Lab X3.2's <code>launchLlamaStep</code> completes milestone M10: a Llama-style decoder runs end to end on the
        course's own kernels and is checked logit by logit and token by token against a CPU reference.{' '}
        <span data-m10-own={loading ? 'loading' : own.join(',') || 'none'}>
          {loading
            ? 'Reading reports.'
            : !ran
              ? 'No lab results are recorded on this machine yet.'
              : own.length
                ? `Your own step passes on ${own.map(where).join(' and ')} on this machine.`
                : 'Your own step does not pass on any backend of this machine yet.'}
        </span>{' '}
        See the <a href={href('x3', 'labs')}>labs</a>. The step runs each earlier lab's reference solution; the table shows
        where your own version of each passes.
      </p>
      <table className="data" data-table="m10">
        <thead>
          <tr><th>Piece of M10</th><th>Lab</th><th>In the step</th><th>Your own work passes on</th></tr>
        </thead>
        <tbody>
          {M10.map((m) => {
            const on = passingOn(reports, m.dir);
            return (
              <tr key={m.dir} data-m10={m.num} data-pass={loading ? 'loading' : on.join(',') || 'none'}>
                <td>{m.title}</td>
                <td className="nowrap">
                  <a href={href(m.chapter, 'labs')}>lab {m.num}</a>
                </td>
                <td>{m.need}</td>
                <td>{loading ? 'reading reports' : on.length ? on.map(where).join(', ') : <span className="muted">nothing yet</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p>
        M10 runs in float32. Module X2's M9 exists, but its launch function takes a quantised matrix descriptor, not the
        float weights M10's linear-layer slot expects, so the two do not meet yet: an adapter between them is the next
        piece of the project.
      </p>
      <p>
        This is the last module in this course's order. What remains is the book's closing chapter,{' '}
        <a href={href('ch24', 'start')}>chapter 24</a> (pp. 577–580), whose summary page helps you pick the module that serves your next goal.
      </p>
    </>
  );
}
