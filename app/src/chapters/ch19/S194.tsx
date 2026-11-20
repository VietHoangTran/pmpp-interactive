import { useEffect, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  CARDS, CUDNN_FWD_ALGOS, DISPATCH_PRESETS, LIMITS, LLAMA_GUESSES, LLAMA_PATH_TEXT, METAL_BUFFER_FLOOR, MIOPEN_FWD_ALGOS,
  NAMES, PRESETS, bookGemm, chapterOutputDim, clampField, cudnnValues, duplication, flops, grade, inputPerSample,
  libraryGemm, llamaCuda, mlx, outputDimGgml, outputDimSteps, outputDimTorch, shape, slow2d, unfoldSize,
  type Check, type Layer, type LlamaPath,
} from '../../sim/convlib';

// ---------------------------------------------------------------- questions
export const questionsS194: Record<string, Question> = {
  'ch19/19.4/q-names': {
    kind: 'choice',
    id: 'ch19/19.4/q-names',
    prompt: (
      <>
        A cuDNN call describes its filters with <code>K = 64</code> and <code>R = S = 3</code>, and its input with{' '}
        <code>C = 32</code>. What is that layer in the chapter&rsquo;s names?
      </>
    ),
    options: [
      { label: 'M = 64 output maps, a 3 × 3 filter (K = 3), C = 32 input maps.', correct: true },
      { label: 'K = 64, M = 3, C = 32.', why: 'That keeps cuDNN’s letters. In the chapter, K is the side of the filter and M counts output maps.' },
      { label: 'M = 64, K = 9, C = 32.', why: 'R × S = 9 is the number of taps per input map, K². The chapter’s K is the side, 3.' },
      { label: 'M = 32, K = 3, C = 64.', why: 'C means the same in both conventions: 32 input maps. The 64 output maps are the chapter’s M.' },
    ],
    explain: (
      <p>
        Table 19.1 renames two things. cuDNN&rsquo;s <code>K</code> counts output feature maps, the chapter&rsquo;s{' '}
        <code>M</code>; <code>R</code> and <code>S</code> are the filter&rsquo;s height and width, where the chapter uses
        one <code>K</code> for a square filter. The table&rsquo;s caption warns about exactly this.
      </p>
    ),
    source: 'ch. 19 §19.4, Table 19.1, p. 473',
  },
  'ch19/19.4/q-padding': {
    kind: 'choice',
    id: 'ch19/19.4/q-padding',
    prompt: (
      <>
        The book says padding is there for better memory alignment or vectorised execution. What does cuDNN&rsquo;s{' '}
        <code>pad_h</code> do?
      </>
    ),
    options: [
      {
        label: 'It adds pad_h rows of zeros at the top and at the bottom of every input, chapter 7’s ghost cells, and so changes the output size.',
        correct: true,
      },
      {
        label: 'It rounds each row of the tensor up to an aligned length; the output size does not change.',
        why: 'Padding of that kind exists inside cuDNN, for Tensor Cores, and you do not ask for it. pad_h appears in the output-size formula.',
      },
      {
        label: 'It pads the output: P grows by exactly 2·pad_h.',
        why: 'The zeros go around the input. P grows because more windows fit in the padded input, by about 2·pad_h / u rows.',
      },
      {
        label: 'Nothing in the forward pass; it matters only for training.',
        why: 'It is in the forward formula: the input is read at p·u + r − pad, with zeros outside the image.',
      },
    ],
    explain: (
      <p>
        cuDNN documents <code>pad_h</code> as rows of zeros implicitly added at the top and at the bottom of the input
        images (<code>pad_w</code> likewise at the left and right). They appear in the output-size formula, so they change{' '}
        <code>P</code> and <code>Q</code>. Alignment padding does exist, but inside the library: for packed NCHW data whose
        channel count is not a multiple of 8, cuDNN pads the tensors itself to use Tensor Cores.
      </p>
    ),
    source: 'ch. 19 p. 473; ch19-facts A2',
  },
  'ch19/19.4/q-workspace': {
    kind: 'choice',
    id: 'ch19/19.4/q-workspace',
    prompt: (
      <>
        Which cuDNN forward algorithm (<code>CUDNN_CONVOLUTION_FWD_ALGO_</code> followed by a name) is the explicit
        unfolding the book argues against on p. 467, and what does its workspace hold?
      </>
    ),
    options: [
      { label: 'GEMM; the workspace holds the unfolded input matrix.', correct: true },
      { label: 'IMPLICIT_GEMM; the workspace holds tiles of the unfolded matrix.', why: 'IMPLICIT_GEMM never forms the matrix. It is Fig. 19.11: tiles live in on-chip memory.' },
      { label: 'IMPLICIT_PRECOMP_GEMM; the workspace holds the unfolded matrix.', why: 'Its workspace holds precomputed indices, not the matrix.' },
      { label: 'DIRECT; the workspace holds the filter bank.', why: 'DIRECT does no matrix multiplication at all; it is the kind of kernel section 19.2 writes.' },
    ],
    explain: (
      <p>
        The documentation describes <code>GEMM</code> as an explicit matrix product that needs a significant memory
        workspace for the matrix of input data: the book&rsquo;s <code>B</code>, with the duplication of Eq. 19.4 and an
        extra pass over global memory to build it. PyTorch&rsquo;s slow CUDA path and llama.cpp&rsquo;s generic graph take
        the same explicit route.
      </p>
    ),
    source: 'ch. 19 p. 467; ch19-facts A3, D2, D4',
  },
  'ch19/19.4/q-precomp': {
    kind: 'choice',
    id: 'ch19/19.4/q-precomp',
    prompt: (
      <>
        <code>IMPLICIT_PRECOMP_GEMM</code> needs a workspace that <code>IMPLICIT_GEMM</code> does not. What is in it, and
        why would it help?
      </>
    ),
    options: [
      {
        label: 'Precomputed indices into the input, so the kernel spends less on Eq. 19.5’s divides and remainders for every element it loads.',
        correct: true,
      },
      { label: 'A copy of the unfolded input, so it can call an ordinary GEMM.', why: 'That is GEMM, the explicit algorithm. PRECOMP still never forms the matrix.' },
      { label: 'The filter weights in another layout.', why: 'The documentation says the workspace holds indices that help build the input matrix implicitly.' },
      { label: 'Partial sums from a split reduction.', why: 'Its description names precomputed indices, nothing else.' },
    ],
    explain: (
      <p>
        cuDNN says it needs some workspace to precompute indices for the implicit construction of the input matrix.
        Eq. 19.5 costs an integer divide and a remainder per loaded element, and CUDA&rsquo;s best-practices guide says
        each can compile to up to 20 instructions. CUTLASS documents the same trade: pointer deltas precomputed on the
        host, and a fast divide-and-modulo for what is left. So the book&rsquo;s &ldquo;computes the mapping
        dynamically&rdquo; describes one of cuDNN&rsquo;s three GEMM-based algorithms.
      </p>
    ),
    source: 'ch. 19 p. 474; ch19-facts A3, A4, A7, A8',
  },
  'ch19/19.4/q-tf32': {
    kind: 'choice',
    id: 'ch19/19.4/q-tf32',
    prompt: (
      <>
        Your FP32 convolution kernel passes against a CPU reference, yet on an NVIDIA GPU it differs from{' '}
        <code>torch.nn.functional.conv2d</code> by about 1e-3 relative. What do you check first?
      </>
    ),
    options: [
      { label: 'Whether cuDNN ran the convolution in TF32, which PyTorch allows by default.', correct: true },
      { label: 'Whether cuDNN flipped the filter.', why: 'PyTorch selects cross-correlation, the book’s formula. A flipped filter would give different numbers altogether, not a small relative error.' },
      {
        label: 'Whether cudnn.benchmark chose a non-deterministic algorithm.',
        why: 'cuDNN documents every forward algorithm as deterministic. Choosing another changes speed and support.',
      },
      {
        label: 'Nothing: summation order alone explains it.',
        why: 'Order does change the last bits (appendix A), so you compare with a tolerance. Rule out a lower precision first.',
      },
    ],
    explain: (
      <p>
        <code>torch.backends.cudnn.allow_tf32</code> defaults to True, so cuDNN convolutions on FP32 tensors may run in
        TF32. PyTorch says the flag is going to be deprecated in favour of{' '}
        <code>torch.backends.cudnn.conv.fp32_precision</code> (&ldquo;ieee&rdquo; or &ldquo;tf32&rdquo;), whose default
        its page does not state, so print what you got. On Apple, <code>MPSCNNConvolution</code> uses FP16 weights unless
        its data source opts into FP32.
      </p>
    ),
    source: 'ch19-facts A3, A6, C4',
  },
  'ch19/19.4/q-find': {
    kind: 'choice',
    id: 'ch19/19.4/q-find',
    prompt: (
      <>
        What do MIOpen&rsquo;s Find step and PyTorch&rsquo;s <code>torch.backends.cudnn.benchmark</code> have in common?
      </>
    ),
    options: [
      {
        label: 'Both time several algorithms for a new problem shape and keep the fastest, so the first call with a new shape is slow.',
        correct: true,
      },
      { label: 'Both make results deterministic.', why: 'PyTorch has a separate flag for that, and benchmarking may even pick different algorithms on different runs.' },
      { label: 'Both are optional ways to save memory.', why: 'Find is mandatory before miopenConvolutionForward unless you use immediate mode, and neither is about memory.' },
      { label: 'Both choose the data layout.', why: 'The layout is in the tensor descriptors you pass. These choose the algorithm.' },
    ],
    explain: (
      <p>
        With <code>benchmark</code> set, cuDNN benchmarks several convolution algorithms for each new set of sizes and
        selects the fastest. MIOpen&rsquo;s <code>miopenFindConvolutionForwardAlgorithm</code> compiles and times its
        solvers and returns them sorted; it is mandatory before <code>miopenConvolutionForward</code>, and it caches its
        results on disk so later sessions start faster. Immediate mode skips Find and looks the answer up instead.
      </p>
    ),
    source: 'ch19-facts A6, B3',
  },
  'ch19/19.4/q-dispatch': {
    kind: 'choice',
    id: 'ch19/19.4/q-dispatch',
    prompt: (
      <>
        llama.cpp&rsquo;s CUDA backend receives a 1 × 1 convolution with stride 1 and no padding, from 256 to 512
        channels on a 14 × 14 map. What does it run?
      </>
    ),
    options: [
      { label: 'A cuBLAS GEMM directly on the input.', correct: true },
      { label: 'An explicit unfold, then a GEMM.', why: 'With K = 1 the unfold would copy the input unchanged (Eq. 19.4 gives exactly 1), so there is nothing to build.' },
      { label: 'Its implicit-GEMM kernel.', why: 'That is the general path. This shape has a test of its own: there is nothing to gather.' },
      { label: 'The direct kernel.', why: 'The direct kernel is for short reductions on small maps; here C·K·K = 256 is not below 64.' },
    ],
    explain: (
      <p>
        With K = 1, stride 1 and no padding, Eq. 19.5 reduces to <code>B[u, v] = X[n, u, v / W_out, v % W_out]</code>:{' '}
        <code>B</code> is <code>X</code> viewed as a C × H·W matrix. llama.cpp&rsquo;s <code>pointwise</code> test sends
        this case to <code>cublasGemmEx</code>, and PyTorch&rsquo;s slow path skips its unfold for the same shapes. A 1 × 1
        convolution is a linear layer applied at every pixel, which is what an LLM&rsquo;s projections do to every
        token.
      </p>
    ),
    source: 'ch19-facts D1, D4; notes, "the K = 1 case"',
  },
};

// ---------------------------------------------------------------- helpers
const n = (v: number) => v.toLocaleString('en-US');
const mib = (bytes: number) => `${(bytes / 2 ** 20).toLocaleString('en-US', { maximumFractionDigits: 1 })} MiB`;
const ratio = (v: number) => v.toLocaleString('en-US', { maximumFractionDigits: 2 });
const gflop = (v: number) => (v >= 1e9 ? `${(v / 1e9).toLocaleString('en-US', { maximumFractionDigits: 2 })} GFLOP` : `${n(v)} FLOP`);
const sameLayer = (a: Layer, b: Layer) => (Object.keys(a) as (keyof Layer)[]).every((k) => a[k] === b[k]);
const describeLayer = (l: Layer) =>
  `N ${l.N}, C ${l.C}, ${l.H} × ${l.W} input, M ${l.M}, ${l.K} × ${l.K} filter, pad ${l.pad}, stride ${l.stride}, dilation ${l.dilation}`;

/** A number box that lets you type freely and commits clamped integers to the layer. */
function NumField({ k, label, hint, value, onChange }: { k: keyof Layer; label: string; hint: string; value: number; onChange: (v: number) => void }) {
  const [text, setText] = useState(String(value));
  const [focus, setFocus] = useState(false);
  useEffect(() => {
    if (!focus) setText(String(value));
  }, [value, focus]);
  return (
    <label className="field" style={{ minWidth: 0 }}>
      <span>
        <b className="mono">{label}</b> <span className="muted">{hint}</span>
      </span>
      <input
        type="number"
        min={LIMITS[k][0]}
        max={LIMITS[k][1]}
        step={1}
        value={text}
        data-field={k}
        aria-label={`${label}, ${hint}`}
        style={{ width: '6.5rem' }}
        onFocus={() => setFocus(true)}
        onBlur={() => {
          setFocus(false);
          setText(String(value));
        }}
        onChange={(e) => {
          setText(e.target.value);
          const v = Number(e.target.value);
          if (e.target.value.trim() !== '' && Number.isFinite(v)) onChange(clampField(k, v));
        }}
      />
    </label>
  );
}

const BOOK_FIELDS: { k: keyof Layer; hint: string }[] = [
  { k: 'N', hint: 'samples' },
  { k: 'C', hint: 'input maps' },
  { k: 'H', hint: 'input height' },
  { k: 'W', hint: 'input width' },
  { k: 'M', hint: 'output maps' },
  { k: 'K', hint: 'filter side' },
];
const LIB_FIELDS: { k: keyof Layer; hint: string }[] = [
  { k: 'pad', hint: 'zeros per side' },
  { k: 'stride', hint: 'step' },
  { k: 'dilation', hint: 'tap spacing' },
];

// ---------------------------------------------------------------- explorer 1: describe a layer to a library
function NamesTable({ layer }: { layer: Layer }) {
  const values = cudnnValues(layer);
  return (
    <table className="data" style={{ margin: 0 }} data-names>
      <thead>
        <tr>
          <th>cuDNN</th>
          <th>Means</th>
          <th>Chapter</th>
          <th className="num">Value</th>
        </tr>
      </thead>
      <tbody>
        {NAMES.filter((r) => r.cudnn in values).map((r) => (
          <tr key={r.cudnn} data-name={r.cudnn}>
            <td>
              <code>{r.cudnn}</code>
            </td>
            <td>
              {r.meaning}
              {r.from !== 'Table 19.1' ? <span className="muted"> (not in Table 19.1)</span> : null}
            </td>
            <td data-chapter={r.chapter ?? ''}>
              {r.chapter ? (
                r.cudnn === 'K' ? (
                  <b>
                    <code>{r.chapter}</code>, not the chapter&rsquo;s K
                  </b>
                ) : (
                  <code>{r.chapter}</code>
                )
              ) : (
                <span className="muted">none: {r.none}</span>
              )}
            </td>
            <td className="num" data-value={values[r.cudnn]}>
              {n(values[r.cudnn])}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function OutputSize({ layer }: { layer: Layer }) {
  const sh = shape(layer);
  const axis = (name: 'P' | 'Q', input: number, inName: 'H' | 'W', padName: string, strideName: string) => {
    const st = outputDimSteps(input, layer.K, layer.pad, layer.stride, layer.dilation);
    const filt = name === 'P' ? 'R' : 'S';
    return (
      <div data-axis={name} style={{ marginBottom: '0.6rem' }}>
        <div className="mono" style={{ fontSize: '0.86rem' }}>
          {name} = 1 + ({inName} + 2·{padName} − (({filt} − 1)·dilation + 1)) / {strideName}
        </div>
        <div className="mono" style={{ fontSize: '0.86rem' }} data-steps>
          {st.quotient == null ? (
            <>
              = no output: the taps span {st.span}, the padded input has {input + 2 * layer.pad}
            </>
          ) : (
            <>
              = 1 + ({input} + 2·{layer.pad} − (({layer.K} − 1)·{layer.dilation} + 1)) / {layer.stride} = 1 + {st.room} /{' '}
              {layer.stride} = <b data-result>{st.result}</b>
            </>
          )}
        </div>
      </div>
    );
  };
  const torch = outputDimTorch(layer.H, layer.K, layer.pad, layer.stride, layer.dilation);
  const ggml = outputDimGgml(layer.H, layer.K, layer.pad, layer.stride, layer.dilation);
  const chap = chapterOutputDim(layer.H, layer.K);
  return (
    <div className="panel" data-outsize data-p={sh.P} data-q={sh.Q}>
      <p style={{ margin: '0 0 0.5rem' }}>
        <b>Output size</b> <span className="muted">(cuDNN&rsquo;s formula, integer division)</span>
      </p>
      {axis('P', layer.H, 'H', 'pad_h', 'u')}
      {layer.W !== layer.H ? axis('Q', layer.W, 'W', 'pad_w', 'v') : null}
      {sh.valid ? (
        <>
          <p style={{ margin: '0 0 0.4rem' }} data-pq>
            Output: <b>{sh.P} × {sh.Q}</b> per map.
            {layer.W === layer.H ? <span className="muted"> Q is the same, since W = H.</span> : null}
          </p>
          <p style={{ margin: '0 0 0.4rem' }} data-others>
            PyTorch&rsquo;s formula gives {torch} and llama.cpp&rsquo;s gives {ggml}.
          </p>
          <p style={{ margin: 0 }} data-chapter-formula={sh.chapterApplies ? 'applies' : 'differs'}>
            The chapter&rsquo;s H − K + 1 = {chap}:{' '}
            {sh.chapterApplies ? (
              <span className="chip ok">the same, with no padding, stride 1, dilation 1</span>
            ) : chap === sh.P ? (
              <span className="chip ok">the same here, by coincidence of the settings</span>
            ) : (
              <span className="chip fail">wrong for this layer, which pads, strides or dilates</span>
            )}
          </p>
        </>
      ) : null}
    </div>
  );
}

function GemmPanel({ layer }: { layer: Layer }) {
  const b = bookGemm(layer);
  const g = libraryGemm(layer);
  return (
    <div className="panel" data-gemm>
      <p style={{ margin: '0 0 0.5rem' }}>
        <b>As a matrix multiplication</b>
      </p>
      <p style={{ margin: '0 0 0.5rem' }} data-book-gemm={`${b.rows}x${b.inner}x${b.cols}x${b.count}`}>
        <span className="muted">The book&rsquo;s orientation, per sample (pp. 464–466):</span>
        <br />F ({n(b.rows)} × {n(b.inner)}) times B ({n(b.inner)} × {n(b.cols)}) gives Y ({n(b.rows)} × {n(b.cols)}),{' '}
        {b.count === 1 ? 'once' : `${n(b.count)} times, one per sample`}.
      </p>
      <p style={{ margin: '0 0 0.5rem' }} data-lib-gemm={`${g.rows}x${g.inner}x${g.cols}`}>
        <span className="muted">The library orientation CUTLASS documents, one product for the batch:</span>
        <br />
        N·P·Q × C·R·S times C·R·S × K: {n(g.rows)} × {n(g.inner)} times {n(g.inner)} × {n(g.cols)}, output pixels as
        rows.
      </p>
      <p style={{ margin: 0 }} data-flops={flops(layer)}>
        Either way, {gflop(flops(layer))} for the layer.
      </p>
    </div>
  );
}

function UnfoldPanel({ layer }: { layer: Layer }) {
  const sh = shape(layer);
  const u = unfoldSize(layer);
  const d = duplication(layer);
  const b = bookGemm(layer);
  const reading =
    layer.K === 1 && layer.stride === 1 && layer.pad === 0
      ? 'B is X itself: with a 1 × 1 filter there is nothing to unfold.'
      : d > 1
        ? 'Overlapping windows copy pixels into several columns (zeros from padding count too).'
        : d === 1
          ? 'No pixel is copied twice: the windows tile the input without overlapping.'
          : 'Some pixels are never read: the stride skips past them.';
  return (
    <div className="panel" data-unfold data-ratio={d}>
      <p style={{ margin: '0 0 0.5rem' }}>
        <b>The unfolded input B</b>
      </p>
      <p style={{ margin: '0 0 0.4rem' }}>
        Per sample: C·K·K × P·Q = {n(b.inner)} × {n(sh.P * sh.Q)} = {n(u.perSample)} elements, against {n(inputPerSample(layer))} in X.
      </p>
      <p style={{ margin: '0 0 0.4rem' }} data-dup>
        Duplication <b>{ratio(d)}×</b>. {reading}
      </p>
      <p style={{ margin: '0 0 0.4rem' }}>
        Unfolded for the whole batch at once: {n(u.batch)} elements, {mib(u.batchBytes)} in FP32.
      </p>
      <p style={{ margin: 0 }} data-limits={`${u.overInt32 ? 'int32' : ''}${u.overMetalFloor ? ' metal' : ''}`.trim()}>
        {u.overInt32 ? (
          <span className="chip fail">more elements than a 32-bit int can index</span>
        ) : (
          <span className="muted">Within reach of 32-bit indices.</span>
        )}{' '}
        {u.overMetalFloor ? (
          <span className="chip fail">over the {n(METAL_BUFFER_FLOOR / 2 ** 20)} MiB Apple guarantees for one Metal buffer</span>
        ) : (
          <span className="muted">Under the {n(METAL_BUFFER_FLOOR / 2 ** 20)} MiB Apple guarantees for one Metal buffer.</span>
        )}
      </p>
    </div>
  );
}

function Flashcards() {
  const [i, setI] = useState(0);
  const [pick, setPick] = useState<string | null>(null);
  const [right, setRight] = useState(0);
  const [done, setDone] = useState(0);
  const card = CARDS[i];
  const ok = pick != null && grade(card, pick);
  const choose = (o: string) => {
    if (pick != null) return;
    setPick(o);
    setDone((x) => x + 1);
    if (grade(card, o)) setRight((x) => x + 1);
  };
  const next = () => {
    if (i + 1 < CARDS.length) setI(i + 1);
    else {
      setI(0);
      setRight(0);
      setDone(0);
    }
    setPick(null);
  };
  const other = card.from === 'chapter' ? 'cuDNN’s' : 'the chapter’s';
  return (
    <div className="panel" data-card={card.id}>
      <p className="muted" style={{ margin: '0 0 0.4rem' }}>
        Card {i + 1} of {CARDS.length}. <span data-score>{right} of {done} right so far.</span>
      </p>
      <p style={{ margin: '0 0 0.7rem' }}>
        {card.from === 'chapter' ? 'The chapter’s' : 'cuDNN’s'}{' '}
        <b className="mono" style={{ fontSize: '1.5rem', padding: '0 0.3rem', border: '1px solid var(--ink)', borderRadius: 3 }}>
          {card.symbol}
        </b>{' '}
        is what in {other} names?
      </p>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {card.options.map((o) => (
          <button key={o} aria-pressed={pick === o} disabled={pick != null && pick !== o} onClick={() => choose(o)} data-option={o}>
            {o}
          </button>
        ))}
      </div>
      {pick != null ? (
        <p style={{ margin: '0 0 0.7rem' }} data-verdict={ok ? 'right' : 'wrong'}>
          <span className={'chip ' + (ok ? 'ok' : 'fail')}>{ok ? 'Right' : `No: ${card.answer}`}</span> {card.why}
        </p>
      ) : null}
      <button className="primary" onClick={next} disabled={pick == null}>
        {i + 1 < CARDS.length ? 'Next card' : 'Start again'}
      </button>
    </div>
  );
}

function LayerFields({ layer, setLayer }: { layer: Layer; setLayer: (l: Layer) => void }) {
  const field = (f: { k: keyof Layer; hint: string }) => (
    <NumField key={f.k} k={f.k} label={f.k} hint={f.hint} value={layer[f.k]} onChange={(v) => setLayer({ ...layer, [f.k]: v })} />
  );
  return (
    <>
      <p className="ui muted" style={{ margin: '0 0 0.3rem', fontSize: '0.85rem' }}>
        The chapter&rsquo;s names
      </p>
      <div className="row" style={{ marginBottom: '0.7rem', flexWrap: 'wrap' }} data-book-fields>
        {BOOK_FIELDS.map(field)}
      </div>
      <p className="ui muted" style={{ margin: '0 0 0.3rem', fontSize: '0.85rem' }}>
        Library only: not in the book&rsquo;s kernels
      </p>
      <div className="row" style={{ marginBottom: '0.9rem', flexWrap: 'wrap' }} data-lib-fields>
        {LIB_FIELDS.map(field)}
      </div>
    </>
  );
}

function DescribeBench({ layer, setLayer }: { layer: Layer; setLayer: (l: Layer) => void }) {
  const [mode, setMode] = useState<'explore' | 'cards'>('explore');
  const presetId = PRESETS.find((p) => sameLayer(p.layer, layer))?.id ?? '';
  const sh = shape(layer);
  return (
    <Bench
      title="Describe a layer to a library"
      task="Find cuDNN’s name for M, a setting that changes P without touching H or K, and a stride that ends all duplication."
      foot={
        <>
          Computed by a tested model (src/sim/convlib.ts) from Table 19.1 and cuDNN&rsquo;s documented formula
          (ch19-facts A2, A7, C2). Sizes are counts of elements; nothing here is a measurement.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', flexWrap: 'wrap' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          Preset
          <select
            aria-label="Preset"
            value={presetId}
            onChange={(e) => {
              const p = PRESETS.find((x) => x.id === e.target.value);
              if (p) setLayer(p.layer);
            }}
            style={{ maxWidth: '100%' }}
          >
            <option value="" disabled>
              your own layer
            </option>
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <span className="seg" role="group" aria-label="Mode" style={{ alignSelf: 'flex-end' }}>
          <button aria-pressed={mode === 'explore'} onClick={() => setMode('explore')}>
            Explore
          </button>
          <button aria-pressed={mode === 'cards'} onClick={() => setMode('cards')}>
            Flashcards
          </button>
        </span>
      </div>
      {mode === 'cards' ? (
        <Flashcards />
      ) : (
        <>
          <LayerFields layer={layer} setLayer={setLayer} />
          <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
            <div>
              <NamesTable layer={layer} />
              <p className="muted" style={{ fontSize: '0.85rem', margin: '0.5rem 0 0' }}>
                The book calls cuDNN&rsquo;s input tensor <code>D</code> (the chapter&rsquo;s <code>X</code>), its filter
                tensor <code>F</code>, and its output <code>O</code> (the chapter&rsquo;s <code>Y</code>).
              </p>
            </div>
            <div style={{ display: 'grid', gap: '0.8rem' }}>
              <OutputSize layer={layer} />
              {sh.valid ? (
                <>
                  <GemmPanel layer={layer} />
                  <UnfoldPanel layer={layer} />
                </>
              ) : (
                <p className="panel" style={{ margin: 0 }} data-invalid>
                  <span className="chip fail">no output</span> The filter&rsquo;s taps span more rows or columns than
                  the padded input has, so there is nothing to multiply.
                </p>
              )}
            </div>
          </div>
        </>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer 2: which kernel would they run?
function Checks({ checks }: { checks: Check[] }) {
  return (
    <ul style={{ margin: '0 0 0.5rem', paddingLeft: '1.1rem' }}>
      {checks.map((c) => (
        <li key={c.label} data-check={c.label} data-holds={c.holds ? 1 : 0}>
          {c.label} <span className="muted">(here <span className="mono">{c.shown}</span>)</span>:{' '}
          {c.holds ? <b>holds</b> : <span className="muted">does not hold</span>}
        </li>
      ))}
    </ul>
  );
}

function BackendCard({ id, title, file, children }: { id: string; title: string; file?: string; children: ReactNode }) {
  return (
    <div className="panel" data-backend={id}>
      <p style={{ margin: '0 0 0.4rem' }}>
        <b>{title}</b>
        {file ? (
          <>
            {' '}
            <span className="muted" style={{ fontSize: '0.85rem' }}>
              ({file})
            </span>
          </>
        ) : null}
      </p>
      {children}
    </div>
  );
}

function DispatchBench({ layer, setLayer }: { layer: Layer; setLayer: (l: Layer) => void }) {
  const [presetId, setPresetId] = useState<string | null>(null);
  const [guess, setGuess] = useState<LlamaPath | null>(null);
  // Nothing is revealed until the first prediction: the page's starting layer is one of the presets.
  const [started, setStarted] = useState(false);
  const active = presetId ? DISPATCH_PRESETS.find((p) => p.id === presetId) : undefined;
  const onPreset = active != null && sameLayer(active.layer, layer);
  const pending = onPreset && guess == null;
  const sh = shape(layer);
  const llama = llamaCuda(layer);
  const torch = slow2d(layer);
  const apple = mlx(layer);
  const b = bookGemm(layer);
  return (
    <Bench
      title="Which kernel would they run?"
      task="Pick a preset and predict llama.cpp’s choice before you see it; then change the layer above and watch the rules."
      foot={
        <>
          Shape conditions as read in each project&rsquo;s source on 2026-11-20 (ch19-facts D1, D3, D4), in a tested model
          (src/sim/convlib.ts). Data types, hardware checks and anything the facts file does not record are not
          modelled. The code changes; read it again before relying on a rule.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', flexWrap: 'wrap' }} data-presets>
        {DISPATCH_PRESETS.map((p) => (
          <button
            key={p.id}
            aria-pressed={presetId === p.id && onPreset}
            data-preset={p.id}
            onClick={() => {
              setLayer(p.layer);
              setPresetId(p.id);
              setGuess(null);
            }}
          >
            {p.label}
          </button>
        ))}
      </div>
      <p style={{ margin: '0 0 0.8rem' }} data-layer>
        <span className="muted">Layer:</span> {describeLayer(layer)}
        {sh.valid ? (
          <>
            {' '}
            → output {sh.P} × {sh.Q}.
          </>
        ) : (
          <> → no output.</>
        )}{' '}
        <span className="muted">Change it in the explorer above.</span>
      </p>

      {pending ? (
        <div className="panel" data-predict>
          <p style={{ margin: '0 0 0.6rem' }}>
            <b>Before you look:</b> what does llama.cpp&rsquo;s CUDA backend run for {active!.label}?
          </p>
          <div className="row" style={{ flexWrap: 'wrap' }}>
            {LLAMA_GUESSES.map((g) => (
              <button
                key={g}
                data-guess={g}
                onClick={() => {
                  setGuess(g);
                  setStarted(true);
                }}
                style={{ textAlign: 'left' }}
              >
                {LLAMA_PATH_TEXT[g]}
              </button>
            ))}
          </div>
        </div>
      ) : !started ? (
        <p className="panel" style={{ margin: 0 }} data-not-started>
          Start with one of the three presets. You predict before anything is shown; after that, every change to the
          layer updates the backends at once.
        </p>
      ) : !sh.valid ? (
        <p className="panel" data-invalid>
          <span className="chip fail">no output</span> This layer has no output, so no backend runs anything.
        </p>
      ) : (
        <>
          {onPreset && guess ? (
            <p style={{ margin: '0 0 0.8rem' }} data-guess-verdict={guess === llama.path ? 'right' : 'wrong'}>
              <span className={'chip ' + (guess === llama.path ? 'ok' : 'fail')}>{guess === llama.path ? 'You had it' : 'Not that'}</span>{' '}
              You said {LLAMA_PATH_TEXT[guess]}.
            </p>
          ) : null}
          <div style={{ display: 'grid', gap: '0.7rem' }}>
            <BackendCard id="llama" title="llama.cpp, CUDA and HIP" file="ggml-cuda/conv2d.cu">
              <Checks checks={llama.checks} />
              <p style={{ margin: 0 }} data-choice={llama.path}>
                <b>Runs:</b> {LLAMA_PATH_TEXT[llama.path]}.{' '}
                {llama.path === 'cublas'
                  ? 'With K = 1 the unfolded matrix is the input itself, so there is nothing to gather.'
                  : llama.path === 'direct'
                    ? 'A short reduction on a small map: the code’s comment says such layers do not amortise conversion and launch costs, the p. 472 argument about small matrices.'
                    : llama.path === 'implicit'
                      ? 'For long reductions on small maps it can also split the C·K·K sum across blocks (split-K); that condition is not modelled.'
                      : 'The facts file records both tests but not which the code makes first, so this explorer will not guess.'}
              </p>
            </BackendCard>
            <BackendCard id="pytorch" title="PyTorch’s CUDA fallback, Slow2d" file="ConvolutionMM2d.cu">
              <p style={{ margin: '0 0 0.3rem' }} className="muted">
                requires_columns is true if any of these holds:
              </p>
              <Checks checks={torch.checks} />
              <p style={{ margin: 0 }} data-columns={!torch.modelled ? 'unknown' : torch.requiresColumns ? 'yes' : 'no'}>
                {!torch.modelled ? (
                  <>
                    <b>Not modelled:</b> this test has no dilation, and which backend serves a dilated layer is not
                    traced in the facts file.
                  </>
                ) : torch.requiresColumns ? (
                  <>
                    <b>Unfolds</b> each sample into a {n(b.inner)} × {n(b.cols)} column matrix, then runs one GEMM per
                    sample: section 19.3, explicitly.
                  </>
                ) : (
                  <>
                    <b>Skips the unfold:</b> the GEMM reads the input as it is.
                  </>
                )}{' '}
                <span className="muted">PyTorch&rsquo;s other CUDA backends include Cudnn and Miopen.</span>
              </p>
            </BackendCard>
            <BackendCard id="mlx" title="MLX, Metal" file="backend/metal/conv.cpp">
              <Checks checks={apple.checks} />
              <p style={{ margin: 0 }} data-mlx={apple.path}>
                {apple.path === 'winograd-candidate' ? (
                  <>
                    <b>A Winograd candidate.</b> MLX also asks for large enough inputs, a condition not modelled here.
                    {apple.dilationUnknown ? ' The facts file does not say whether this test looks at dilation.' : ''}
                  </>
                ) : (
                  <>
                    <b>Not Winograd:</b> an implicit GEMM (a specialised or a general kernel; the conditions are not in
                    the facts file), or, as the last resort, an explicit unfold done in row tiles that fit one Metal buffer.
                  </>
                )}
              </p>
            </BackendCard>
            <BackendCard id="libraries" title="cuDNN and MIOpen">
              <p style={{ margin: 0 }}>
                Chosen at run time, not by a rule you can read: the caller names an algorithm (cuDNN&rsquo;s legacy API),
                heuristics pick an engine (its graph API), PyTorch&rsquo;s <code>cudnn.benchmark</code> times several,
                MIOpen&rsquo;s Find times its solvers or immediate mode looks one up.
              </p>
            </BackendCard>
            <BackendCard id="apple" title="Apple: MPSGraph, MPSCNNConvolution, MPP convolution2d">
              <p style={{ margin: 0 }} data-apple>
                No algorithm is exposed or documented.
              </p>
            </BackendCard>
          </div>
        </>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- the p-outsize reveal
const STEM = PRESETS.find((p) => p.id === 'stem')!.layer;
function OutsizeReveal({ committed, onLoad }: { committed: number; onLoad: () => void }) {
  const st = outputDimSteps(STEM.H, STEM.K, STEM.pad, STEM.stride, STEM.dilation);
  const answer = st.result;
  const chap = chapterOutputDim(STEM.H, STEM.K);
  return (
    <>
      <p data-outsize-answer={answer}>
        cuDNN&rsquo;s formula: P = 1 + ({STEM.H} + 2·{STEM.pad} − (({STEM.K} − 1)·{STEM.dilation} + 1)) / {STEM.stride} = 1 +{' '}
        {st.room} / {STEM.stride} = 1 + {st.quotient} = <b>{answer}</b>, with integer division.{' '}
        {committed === answer
          ? 'That is what you predicted.'
          : committed === chap
            ? `${chap} is the chapter’s H − K + 1, which holds only with no padding and stride 1.`
            : committed === answer + 1
              ? `${st.room} / ${STEM.stride} is ${st.room / STEM.stride}; the division truncates, it does not round up.`
              : `Check each term: the padding adds ${2 * STEM.pad} rows, the filter spans ${st.span}, and the stride divides what is left.`}
      </p>
      <p style={{ margin: 0 }}>
        The same formula sizes outputs in PyTorch ({outputDimTorch(STEM.H, STEM.K, STEM.pad, STEM.stride)}) and llama.cpp (
        {outputDimGgml(STEM.H, STEM.K, STEM.pad, STEM.stride)}).{' '}
        <button onClick={onLoad} data-load-stem>
          Load this layer into the explorer
        </button>
      </p>
    </>
  );
}

// ---------------------------------------------------------------- the backend table, cut to the library rows (ch19-facts, last table)
const DIFFER: [string, string, string, string][] = [
  [
    'Vendor library',
    'cuDNN 9.28.1: graph API (C++ and Python frontend, C backend); legacy cudnnConvolutionForward deprecated since 9.0',
    'MIOpen 3.6.2: miopenConvolutionForward after Find, or immediate mode; hipDNN at an early stage',
    'MPSGraph convolution2D; MPSCNNConvolution on MPSImage; MPP convolution2d inside shaders (Metal 4)',
  ],
  [
    'Forward algorithms',
    `${CUDNN_FWD_ALGOS.length}: GEMM, IMPLICIT_GEMM, IMPLICIT_PRECOMP_GEMM, DIRECT, FFT, FFT_TILING, WINOGRAD (not on Hopper and later), WINOGRAD_NONFUSED`,
    `${MIOPEN_FWD_ALGOS.length}: GEMM, Direct, FFT, Winograd, ImplicitGEMM`,
    'None exposed or documented',
  ],
  [
    'Choosing one',
    'The caller passes algo (legacy) or heuristics pick an engine; PyTorch’s cudnn.benchmark times several',
    'Find times solvers and caches on disk; immediate mode uses a database and heuristics',
    'Not exposed',
  ],
  ['Explicit im2col as an API', 'cudnnIm2Col (deprecated in 9.0), one row per output pixel', 'None in miopen.h', 'MPSGraph imToCol (macOS 14 and later)'],
  [
    'Layouts',
    'NCHW, NHWC, NCHW_VECT_C, any strides (some routines refuse); Tensor Cores require NHWC, converted internally',
    'NCHW and NHWC fully supported; CHWN not supported',
    'MPSGraph NCHW or NHWC (weights OIHW or HWIO); MPSCNN weights OHWI; MPP NHWC and HWIO only',
  ],
  ['Filter flip', 'Cross-correlation mode in its formulas; frameworks select it', 'Cross-correlation', 'Not documented; PyTorch passes weights unflipped to MPSGraph'],
  [
    'Precision trap against an FP32 reference',
    'PyTorch lets cuDNN convolutions use TF32 by default',
    'Not checked here',
    'MPSCNNConvolution weights FP16 unless the data source opts into FP32; MPP relaxed_precision off by default; MSL fast math',
  ],
];

// ---------------------------------------------------------------- the page
export function S194() {
  const q = questionsS194;
  const [layer, setLayer] = useState<Layer>(PRESETS.find((p) => p.id === 'lenet-c1')!.layer);
  const loadStem = () => {
    setLayer(STEM);
    document.querySelector('section.bench[aria-label="Describe a layer to a library"]')?.scrollIntoView({ block: 'start' });
  };
  return (
    <>
      <p>
        A framework usually does not run a convolution with a kernel of its own. It describes the layer to a vendor
        library, and the library decides how to compute it. This page is about that hand-off: what cuDNN and its AMD and
        Apple counterparts need to be told, how their names map onto the chapter&rsquo;s, which of the chapter&rsquo;s
        kernels hide behind their options, and what they decide without asking you.
      </p>

      <h2>What cuDNN is</h2>
      <p>
        cuDNN is NVIDIA&rsquo;s library of tuned routines for the building blocks of deep learning, made so that
        frameworks can use GPUs without writing kernels. Its interface is C. Inputs and outputs must already sit in
        device memory, as with cuBLAS. It is thread-safe: you can call it from several host threads. The forward and
        the backward convolution share one descriptor of the layer, and tensors and filters are described by opaque
        descriptors that let you give the layout as an arbitrary stride per dimension. At its heart is a batched
        convolution, and the book covers the forward direction.
      </p>
      <Source section="19.4" pages="472" />
      <p>
        All of that is still documented, but it is now what NVIDIA calls the legacy API. Read it as the model of a
        library call, not as the call you would write today.
      </p>
      <Today checked="2026-11-20 (ch19-facts A1)">
        <p>
          <code>cudnnConvolutionForward</code> and its descriptors have been deprecated since cuDNN 9.0 and live on in
          the legacy libraries. NVIDIA recommends its graph API, through the C++ and Python frontend. Three conditions the
          book leaves out: threads must not use one cuDNN handle at the same time, a tensor is capped at 2
          Giga-elements, and some routines reject unusual strides.
        </p>
      </Today>
      <p>
        One slip on p. 472: the book says the device-memory requirement was discussed earlier in the chapter. Nothing
        earlier in chapter 19 discusses it.
      </p>

      <h2>Who calls it</h2>
      <p>
        The book names PyTorch, Caffe, TensorFlow and MXNet as frameworks built on cuDNN, and says most deep-learning
        frameworks use it, so their users get these implementations without writing CUDA.
      </p>
      <Source section="19.4" pages="472, 474" />
      <p>
        Two of those four have stopped. MXNet was retired in 2023 and Caffe has had no commit since 2020. NVIDIA&rsquo;s
        own list today has PyTorch, JAX and XLA, its Megatron Core and Transformer Engine, and, for LLM serving, vLLM and
        SGLang through FlashInfer. That last route is for attention, not convolution.
      </p>
      <Today checked="2026-11-20 (ch19-facts A6)">
        <p>Apache MXNet: retired September 2023, moved to the Attic February 2024. BVLC/caffe: last commit 2020-02-13.</p>
      </Today>

      <h2>Two sets of names</h2>
      <p>
        Table 19.1 lists cuDNN&rsquo;s parameters, and its caption warns that they are not the chapter&rsquo;s. Most
        letters carry over: <code>N</code> samples, <code>C</code> input maps, <code>H</code> and <code>W</code> for the
        input&rsquo;s height and width. Two do not. cuDNN&rsquo;s <code>K</code> is the number of output feature maps,
        which the chapter calls <code>M</code>. The filter&rsquo;s height and width are <code>R</code> and{' '}
        <code>S</code>, where the chapter uses one <code>K</code> for a square filter. The remaining four,{' '}
        <code>u</code>, <code>v</code>, <code>pad_h</code> and <code>pad_w</code>, are strides and padding, which the
        chapter&rsquo;s kernels do not have. The book calls the tensors <code>D</code> (input), <code>F</code> (filters)
        and <code>O</code> (output), and the output&rsquo;s height and width <code>P</code> and <code>Q</code>.
      </p>
      <Source section="19.4" pages="472–473" figs="Table 19.1" />
      <Ask q={q['ch19/19.4/q-names']} />

      <h2>The output size</h2>
      <p>
        For the output size the book gives only a dependence, <code>P = f(H; R; u; pad_h)</code>, with no formula. cuDNN
        documents one: <code>P = 1 + (H + 2·pad_h − ((R − 1)·dilation + 1)) / u</code>, with integer division. The term{' '}
        <code>(R − 1)·dilation + 1</code> is how many input rows the filter&rsquo;s taps span: R when dilation is 1, more
        when the taps are spread apart. PyTorch documents the same formula with a floor, and llama.cpp computes it in{' '}
        <code>ggml_calc_conv_output_size</code>. The chapter&rsquo;s kernels are the case pad 0, stride 1, dilation 1,
        where it reduces to <code>H − K + 1</code>.
      </p>
      <Today checked="2026-11-20 (ch19-facts A2)">
        <p>
          From <code>cudnnGetConvolution2dForwardOutputDim</code>, legacy since 9.0 like the rest. Dilation and groups are
          cuDNN parameters that Table 19.1 does not list.
        </p>
      </Today>
      <Ask q={q['ch19/19.4/q-padding']} />
      <p>
        Padding needs a correction. The book says it serves memory alignment or vectorised execution. That is not what
        the parameter documents. <code>pad_h</code> is the number of zero rows cuDNN adds at the top and at the bottom of
        each input image, and <code>pad_w</code> the zero columns at the left and right: chapter 7&rsquo;s ghost cells.
        They enter the formula, so they change the output size; padding a 3 × 3 layer by 1 keeps its output as large as
        its input. Alignment padding exists too, but inside cuDNN, which pads NCHW tensors itself when the channel count
        is not a multiple of 8, so that Tensor Cores can be used.
      </p>
      <Source section="19.4" pages="473" />
      <Predict
        id="ch19/19.4/p-outsize"
        prompt={
          <>
            A layer has <code>H = 224</code>, a 7 × 7 filter (<code>R = 7</code>), vertical stride <code>u = 2</code>,{' '}
            <code>pad_h = 3</code> and dilation 1. What is <code>P</code>, the output height?
          </>
        }
        unit="rows"
      >
        {(v) => <OutsizeReveal committed={v} onLoad={loadStem} />}
      </Predict>
      <p>
        Now set layers of your own: the chapter&rsquo;s six names, then the three parameters its kernels lack. Look for
        a setting that changes <code>P</code> without touching <code>H</code> or <code>K</code>, and a stride that makes
        the duplication of section 19.3 disappear. The flashcards drill the two sets of names.
      </p>
      <Simplification>
        <p>
          2D only, groups 1, a square filter, and one value each of padding, stride and dilation for both axes; cuDNN
          takes them per axis. Shapes and counts are exact for the layer. Nothing here models time.
        </p>
      </Simplification>
      <DescribeBench layer={layer} setLayer={setLayer} />

      <h2>The algorithms</h2>
      <p>
        cuDNN offers several algorithms for a convolutional layer: ones based on matrix multiplication (the book files
        both GEMM and Winograd here), ones based on the FFT, and others. Its GEMM-based algorithm is section 19.3. It
        does not gather the unfolded input in global memory and then call a matrix multiply; it generates the parts it
        needs as it goes and loads them only into on-chip memory.
      </p>
      <Source section="19.4" pages="473" />
      <p>
        The documentation lists eight forward algorithms today, and three of them are this chapter. <code>GEMM</code> is
        the explicit unfold, with a significant workspace to hold the unfolded input. <code>IMPLICIT_GEMM</code> is Fig.
        19.11. <code>IMPLICIT_PRECOMP_GEMM</code> is implicit too, but first precomputes some indices into a workspace,
        so the book&rsquo;s &ldquo;computes the mapping dynamically&rdquo; describes only one of the three.{' '}
        <code>DIRECT</code> is the kind of kernel section 19.2 writes. FFT and Winograd are other mathematics, outside
        this chapter. Every forward algorithm is documented as deterministic; choosing one changes speed and which cases
        are supported.
      </p>
      <Today checked="2026-11-20 (ch19-facts A3, A4)">
        <p>
          <code>WINOGRAD</code> is not supported on Hopper and later GPUs. <code>cudnnIm2Col</code>, deprecated in 9.0,
          was a public explicit unfold; it built the transpose of the book&rsquo;s <code>B</code>, one row per output
          pixel of every sample.
        </p>
      </Today>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-algos>
          <thead>
            <tr>
              <th>cuDNN forward algorithm</th>
              <th>In this chapter</th>
              <th>Workspace, as documented</th>
              <th>HIPIFY to MIOpen</th>
            </tr>
          </thead>
          <tbody>
            {CUDNN_FWD_ALGOS.map((a) => (
              <tr key={a.name} data-algo={a.name}>
                <td>
                  <code>{a.name.replace('CUDNN_CONVOLUTION_FWD_ALGO_', '')}</code>
                  {a.note ? <div className="muted">{a.note}</div> : null}
                </td>
                <td>{a.chapter}</td>
                <td>{a.workspace}</td>
                <td>{a.hipify ? <code>{a.hipify.replace('miopenConvolutionFwdAlgo', '')}</code> : <span className="muted">none listed</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted" style={{ fontSize: '0.85rem', marginTop: '-0.8rem' }}>
          Each name is prefixed with <code>CUDNN_CONVOLUTION_FWD_ALGO_</code>, each MIOpen value with{' '}
          <code>miopenConvolutionFwdAlgo</code>. From ch19-facts A3 and B4.
        </p>
      </div>
      <Ask q={q['ch19/19.4/q-workspace']} />
      <Ask q={q['ch19/19.4/q-precomp']} />

      <h2>How the matrix multiplication runs</h2>
      <p>
        Underneath, the book says, is a matrix multiplication that reaches a high fraction of the device&rsquo;s peak
        arithmetic. It reads fixed-size tiles of both inputs into on-chip memory and computes a tile of the output from
        them, and while it computes on the current tiles it is already fetching the next ones. That is the software
        pipelining of section <a href={href('ch15', '15.8')}>15.8</a>: the wait for memory hides behind arithmetic.
      </p>
      <Source section="19.4" pages="473–474" />
      <p>
        The tiling knows nothing about convolutions, so where a tile&rsquo;s edges fall in the input is not simple.
        cuDNN computes that correspondence as the computation runs and uses it to load the right elements: extra index
        arithmetic, in exchange for reusing the whole matrix-multiplication engine. The book adds that NVIDIA&rsquo;s
        routine is similar to a 2011 DGEMM for Fermi GPUs, its reference 10. cuDNN documents none of its kernels, so that
        comparison cannot be checked.
      </p>
      <Aside>
        <p>Unverified: cuDNN documents no kernel internals, and the 2011 paper could not be fetched (ch19-facts A4, A10).</p>
      </Aside>
      <p>
        NVIDIA does document the technique, in CUTLASS, its open-source GEMM library. Its implicit-GEMM convolution forms
        tiles of the convolution matrix on the fly, as data moves from global to shared memory, and hands them to its
        warp-level GEMM components. It splits the reduction index exactly as Eq. 19.5 does. And it says that computing
        those divisions and remainders directly would cost too much, so it precomputes pointer deltas on the host and
        uses a fast divide-and-modulo for the rest. Its GEMM is the transpose of the book&rsquo;s: N·P·Q rows of output
        pixels, the K output maps as columns, a reduction over C·R·S, on NHWC data.
      </p>
      <Today checked="2026-11-20 (ch19-facts A7, A8)">
        <p>
          CUDA&rsquo;s best-practices guide: integer division and modulo can compile to up to 20 instructions. CUTLASS
          recommends 128-bit aligned NHWC tensors, with C and K multiples of 32.
        </p>
      </Today>

      <h2>The last step: layout</h2>
      <p>
        After computing, the book says, cuDNN transposes the result into whatever data layout the user asked for.
      </p>
      <Source section="19.4" pages="474" />
      <p>
        The documentation ties that conversion to Tensor Cores and nothing else. Tensor Cores require NHWC, so cuDNN
        converts NCHW to NHWC and back when Tensor Core math is requested and actually used, and not otherwise. A request
        is only a request: Tensor Cores may not be used.
      </p>
      <Today checked="2026-11-20 (ch19-facts A5)">
        <p>
          cuDNN&rsquo;s formats are NCHW, NHWC and NCHW_VECT_C, with other orders through strides. NHWC has limited
          support in the legacy forward call.
        </p>
      </Today>

      <h2>Choosing an algorithm in practice</h2>
      <p>
        With the legacy API you name the algorithm in the call. In practice a framework chooses, and PyTorch lets you
        steer it. With <code>torch.backends.cudnn.benchmark</code> set, cuDNN benchmarks several algorithms the first time
        it meets a new set of sizes and keeps the fastest; <code>torch.backends.cudnn.deterministic</code> restricts it to
        deterministic ones. That is why the first iteration of a model with benchmarking on is slow, and why two runs on
        the same machine can end up with different kernels.
      </p>
      <Beyond>
        <p>From PyTorch 2.15&rsquo;s documentation of <code>torch.backends</code> and its notes on randomness (ch19-facts A6).</p>
      </Beyond>
      <p>
        Precision is a choice too. PyTorch lets cuDNN run FP32 convolutions in TF32 by default:{' '}
        <code>torch.backends.cudnn.allow_tf32</code> is True. PyTorch says that flag is going to be deprecated in favour of{' '}
        <code>torch.backends.cudnn.conv.fp32_precision</code>, set to &ldquo;ieee&rdquo; or &ldquo;tf32&rdquo;, and the
        page does not state its default. When you compare your kernel with <code>conv2d</code> on an NVIDIA GPU, print
        the setting and compare with a tolerance.
      </p>
      <Today checked="2026-11-20 (ch19-facts A6)">
        <p>
          The finer-grained precision settings arrived after PyTorch 2.9. Slow or naive convolution fallbacks built on
          CUDA GEMMs inherit the matmul precision setting.
        </p>
      </Today>
      <Ask q={q['ch19/19.4/q-tf32']} />

      <h2>On AMD: MIOpen</h2>
      <p>
        AMD&rsquo;s counterpart has the shape of cuDNN&rsquo;s legacy API: a handle, opaque descriptors, and a forward
        call that takes an algorithm and a workspace. MIOpen 3.6.2 has five forward algorithms, GEMM, direct, FFT,
        Winograd and implicit GEMM, and it computes cross-correlation, as the book does. Two differences matter in
        practice. A Find step, which compiles and times a set of solvers for your problem, is mandatory before{' '}
        <code>miopenConvolutionForward</code>; it caches its results on disk so later sessions run faster, and immediate
        mode replaces it with a lookup in a database. And of Exercise 2&rsquo;s three layouts, NCHW and NHWC are fully
        supported and CHWN is not supported at all.
      </p>
      <Today checked="2026-11-20 (ch19-facts B3)">
        <p>
          MIOpen&rsquo;s descriptor takes padding, stride and dilation like cuDNN&rsquo;s. No general thread-safety
          statement for MIOpen was found (unverified); its header says kernel timings on one handle are not thread-safe.
        </p>
      </Today>
      <p>
        Porting a cuDNN call is not mechanical. HIPIFY maps <code>cudnnConvolutionForward</code> to{' '}
        <code>miopenConvolutionForward</code> and five of the eight algorithm values, but lists no MIOpen counterpart for
        the descriptor setter <code>cudnnSetConvolution2dDescriptor</code> (MIOpen&rsquo;s initialiser takes the mode
        first and has no compute type), nor for <code>IMPLICIT_PRECOMP_GEMM</code>, <code>FFT_TILING</code> or{' '}
        <code>WINOGRAD_NONFUSED</code>. AMD&rsquo;s graph-style library, hipDNN, says of itself that it is in the early
        stages of development.
      </p>
      <Today checked="2026-11-20 (ch19-facts B4)">
        <p>From HIPIFY&rsquo;s table of the cuDNN API supported by MIOpen, and hipDNN&rsquo;s README.</p>
      </Today>
      <Ask q={q['ch19/19.4/q-find']} />

      <h2>On Apple: three routes, no menu</h2>
      <p>
        Apple has three ways to run a convolution, and none of them exposes an algorithm choice. MPSGraph&rsquo;s{' '}
        <code>convolution2D</code> takes NCHW or NHWC data and OIHW or HWIO weights; NCHW with OIHW is the book&rsquo;s
        layout. MPSGraph also has <code>imToCol</code> (macOS 14 and later), an explicit unfold, the book&rsquo;s
        materialised <code>B</code>, as a graph operation. <code>MPSCNNConvolution</code> works on{' '}
        <code>MPSImage</code> objects rather than raw buffers and takes OHWI weights only. Metal Performance
        Primitives&rsquo; <code>convolution2d</code> runs inside your own shader on Metal 4, used by a whole threadgroup,
        with NHWC activations, HWIO weights and groups of 1 only.
      </p>
      <Today checked="2026-11-20 (ch19-facts C3, C4, C5)">
        <p>From the macOS 27.0 SDK headers and section 7.2.2 of the Metal Shading Language specification 4.1.</p>
      </Today>
      <p>
        Two traps. <code>MPSCNNConvolution</code> uses FP16 weights unless its data source reports FP32, so against an
        FP32 reference it can be further off than you expect. And two things are not documented: whether MPSGraph&rsquo;s
        convolution flips the filter (PyTorch passes <code>nn.Conv2d</code> weights to it unreversed, so in practice it
        computes cross-correlation), and where MPP&rsquo;s <code>convolution2d</code> places its window relative to the
        offset you give it (PyTorch&rsquo;s kernel records a rule it measured).
      </p>
      <Aside>
        <p>
          Two slips in Apple&rsquo;s headers: the im2col descriptor calls NCHW &ldquo;batch x channels x width x
          height&rdquo;, and the convolution header gives the dilated kernel width as (dilationRateInX − 1)·kernelWidth + 1
          where (kernelWidth − 1)·dilation + 1 is meant (ch19-facts C3).
        </p>
      </Aside>

      <h2>What real backends run</h2>
      <p>
        For your third goal, reading real backends: here is where each one sends a 2D convolution. PyTorch&rsquo;s list
        of convolution backends has <code>Cudnn</code>, <code>Miopen</code>, <code>Mps</code> and <code>Slow2d</code>, its
        own CUDA fallback. <code>Slow2d</code> is section 19.3 with an explicit unfold, one sample at a time, and it skips
        the unfold when the filter is 1 × 1 with stride 1 and no padding.
      </p>
      <Beyond>
        <p>Read from each project&rsquo;s default branch on 2026-11-20 (ch19-facts D1 to D4). Line numbers drift.</p>
      </Beyond>
      <p>
        llama.cpp&rsquo;s CUDA backend, which also builds for HIP, has <code>conv2d_implicit_gemm_f16</code>: Fig. 19.11
        in production. Its grid is the one we derived for Fig. 19.11 in section 19.3 (the book states only that z is the
        sample): x over tiles of output pixels, y over tiles of output channels, z over the batch times the pieces of
        a split reduction. It adds ceiling division and every bounds check Fig. 19.11 leaves out. It gathers the input tile with
        Eq. 19.5, using literal divisors for 3 × 3 and 1 × 1 filters and a precomputed fast division otherwise, and it
        handles padding beforehand by writing a zero-padded copy of the input. Its Metal backend has an explicit unfold, a
        direct kernel and <code>kernel_conv_2d_mm</code>, an implicit GEMM on simdgroup matrices. MLX works channels-last
        and picks among Winograd, implicit GEMM and, as the last resort, an explicit unfold done in row tiles so that each
        fits one Metal buffer.
      </p>
      <p>
        The dispatch rules of llama.cpp and MLX are where the book&rsquo;s argument on p. 472, that the GEMM formulation
        pays when the matrices are large, turns into code. The explorer applies them to the layer you set above.
      </p>
      <Source section="19.3" pages="472" />
      <p>
        Pick one of the three presets, commit to what llama.cpp&rsquo;s CUDA backend runs, and only then compare the
        other backends. After that, change the layer in the explorer above and watch which conditions flip.
      </p>
      <Simplification>
        <p>
          Shape conditions only, as read on 2026-11-20. Data types, hardware checks, the split-K condition and anything
          else the facts file does not record are not modelled, and the explorer says so where it matters.
        </p>
      </Simplification>
      <DispatchBench layer={layer} setLayer={setLayer} />
      <Ask q={q['ch19/19.4/q-dispatch']} />

      <h2>Where the backends differ</h2>
      <p>
        The library rows of the backend table in this chapter&rsquo;s facts file. Every cell is a documented fact or is
        marked as not checked; nothing here was run on a device.
      </p>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-differ>
          <thead>
            <tr>
              <th>Feature</th>
              <th>CUDA</th>
              <th>HIP</th>
              <th>Metal</th>
            </tr>
          </thead>
          <tbody>
            {DIFFER.map(([f, c, h, m]) => (
              <tr key={f} data-row={f}>
                <td>{f}</td>
                <td>{c}</td>
                <td>{h}</td>
                <td>{m}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        Next: the <a href={href('ch19', 'labs')}>lab</a>, where you build the explicit route yourself and time it against
        the direct kernel, and the chapter&rsquo;s <a href={href('ch19', 'wrap')}>summary</a>.
      </p>
    </>
  );
}
