import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { NMSE_LLAMACPP, SLIDER, START_S, STEPS, compare, nmse, type Comparison } from '../../sim/tokencompare';
import { LateDivergenceScene } from './LateDivergenceScene';

// Section X3.9: testing a GPU run against a reference, and contributing. The explorer and the
// scene run on src/sim/tokencompare.ts (pinned in tokencompare.test.ts). Vendor and project
// facts are from notes/today/x3-facts.md D7, D8, F1, F2 (and B3, ch10 D3 for MLX's softmax).

const MONO: CSSProperties = { fontFamily: 'var(--sans)', fontVariationSettings: "'MONO' 1" };
const e1 = (v: number) => (v === 0 ? '0' : v.toExponential(1));
const sLabel = (e: number) => {
  const [m, x] = (10 ** e).toExponential(1).split('e');
  return (
    <>
      {m} × 10<sup>{String(Number(x)).replace('-', '−')}</sup>
    </>
  );
};

// ---------------------------------------------------------------- charts

const CW = 720;
const CH = 150;
const L = 52;
const R = 12;
const T = 24;
const B = 24;
const xAt = (t: number) => L + ((t + 0.5) / STEPS) * (CW - L - R);

function LogChart({ lo, hi, label, children, part }: { lo: number; hi: number; label: string; children: (y: (v: number) => number) => ReactNode; part: string }) {
  const y = (v: number) => {
    const l = Math.log10(Math.max(v, 10 ** lo));
    return T + ((hi - Math.min(l, hi)) / (hi - lo)) * (CH - T - B);
  };
  const ticks: number[] = [];
  for (let e = Math.ceil(lo); e <= hi; e += lo < -12 ? 4 : 2) ticks.push(e);
  return (
    <div style={{ overflowX: 'auto' }}>
    <svg viewBox={`0 0 ${CW} ${CH}`} role="img" aria-label={label} data-part={part}
      style={{ width: '100%', minWidth: 540, display: 'block', background: 'var(--page)', border: '1px solid var(--rule)', borderRadius: 4 }}>
      {ticks.map((e) => (
        <g key={e}>
          <line x1={L} y1={y(10 ** e)} x2={CW - R} y2={y(10 ** e)} stroke="var(--rule-soft)" />
          <text x={L - 5} y={y(10 ** e) + 3.5} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>1e{e}</text>
        </g>
      ))}
      {Array.from({ length: STEPS }, (_, t) => (t % 2 === 0 ? (
        <text key={t} x={xAt(t)} y={CH - 8} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{t}</text>
      ) : null))}
      <text x={L} y={12} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{label}</text>
      {children(y)}
    </svg>
    </div>
  );
}

function NmseChart({ c }: { c: Comparison }) {
  return (
    <LogChart lo={-18} hi={2} label="NMSE per step against the reference (log scale); x: step" part="nmse">
      {(y) => (
        <>
          <line x1={L} y1={y(NMSE_LLAMACPP)} x2={CW - R} y2={y(NMSE_LLAMACPP)} stroke="var(--ink)" strokeDasharray="5 3" />
          <text x={CW - R - 4} y={y(NMSE_LLAMACPP) - 4} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink)' }}>llama.cpp's default per op, 1e-7</text>
          {c.steps.map((r) => (
            <g key={r.t}>
              <circle cx={xAt(r.t) + 4} cy={y(r.frNmse)} r={3.6} fill="none" stroke={r.frTok !== r.refTok || r.frIn !== r.refIn ? 'var(--fail)' : 'var(--ink-3)'} strokeWidth={1.4} />
              <circle cx={xAt(r.t) - 4} cy={y(r.tfNmse)} r={3.4} fill={r.tfTok !== r.refTok ? 'var(--fail)' : 'var(--ink)'} />
            </g>
          ))}
        </>
      )}
    </LogChart>
  );
}

function MarginChart({ c }: { c: Comparison }) {
  const lo = -7;
  return (
    <LogChart lo={lo} hi={1} label="Reference top-two margin per step (log scale); bars at or below the line are flagged" part="margins">
      {(y) => (
        <>
          {c.steps.map((r) => (
            <rect key={r.t} x={xAt(r.t) - 7} y={y(r.margin)} width={14} height={Math.max(1, y(10 ** lo) - y(r.margin))} rx="1.5"
              fill={r.flagged ? 'var(--fail)' : 'var(--ink-3)'} opacity={r.flagged ? 0.9 : 0.55} data-step={r.t} data-flagged={r.flagged ? 'yes' : 'no'} />
          ))}
          <line x1={L} y1={y(2 * c.tol)} x2={CW - R} y2={y(2 * c.tol)} stroke="var(--ink)" strokeDasharray="5 3" />
          <text x={CW - R - 4} y={y(2 * c.tol) - 4} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink)' }}>
            2 × largest logit error = {e1(2 * c.tol)}
          </text>
        </>
      )}
    </LogChart>
  );
}

function Streams({ c }: { c: Comparison }) {
  const rows = [
    { id: 'ref', label: 'reference', tok: c.steps.map((r) => r.refTok) },
    { id: 'fr', label: 'GPU, free running', tok: c.steps.map((r) => r.frTok) },
    { id: 'tf', label: 'GPU, teacher forced', tok: c.steps.map((r) => r.tfTok) },
  ];
  const cell: CSSProperties = { ...MONO, width: '1.55rem', textAlign: 'center', padding: '0.12rem 0', borderRadius: 3, fontSize: 'var(--step--1)' };
  return (
    <div style={{ overflowX: 'auto' }} data-part="streams">
      <table style={{ borderCollapse: 'separate', borderSpacing: 2, margin: 0 }}>
        <thead>
          <tr>
            <th />
            {c.steps.map((r) => (
              <th key={r.t} style={{ ...cell, fontWeight: 400, color: r.flagged ? 'var(--fail)' : 'var(--ink-3)' }} title={r.flagged ? 'flagged: margin too small to judge' : undefined}>
                {r.flagged ? '?' : r.t}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} data-row={row.id}>
              <td style={{ paddingRight: '0.5rem', whiteSpace: 'nowrap', color: 'var(--ink-2)', fontSize: 'var(--step--1)' }}>{row.label}</td>
              {row.tok.map((tok, t) => {
                const bad = row.id !== 'ref' && tok !== c.steps[t].refTok;
                const first = row.id === 'fr' && t === c.firstDivergence;
                return (
                  <td key={t} data-bad={bad ? 'yes' : 'no'}
                    style={{ ...cell, background: bad ? 'var(--fail-wash)' : 'var(--cell)', color: bad ? 'var(--fail)' : 'var(--ink)', border: `1px solid ${first ? 'var(--fail)' : bad ? 'var(--fail-wash)' : 'var(--rule)'}`, outline: first ? '1px solid var(--fail)' : undefined }}>
                    {tok}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CompareRuns() {
  const [e, setE] = useState(Math.log10(START_S));
  const c = useMemo(() => compare(10 ** e), [e]);
  const flips = c.tfDiffers;
  return (
    <Bench
      title="Compare two runs"
      task="Drag the perturbation up. Where does token agreement first fail, and what does NMSE say then?"
      foot={
        <>
          A toy model of 16 tokens and 24 greedy steps; the "GPU" adds a fixed pattern of noise, scaled by the slider, to every logit.
          Filled dots: teacher-forced NMSE. Rings: free-running NMSE (red once the inputs differ). Streams: red cells differ from the
          reference; a ? over a step marks it flagged.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 18rem' }}>
          perturbation per logit, up to s = <span data-part="s" data-s={10 ** e}>{sLabel(e)}</span>
          <input type="range" min={SLIDER.min} max={SLIDER.max} step={SLIDER.step} value={e} aria-label="perturbation size"
            onChange={(ev) => setE(Math.round(Number(ev.target.value) * 10) / 10)} style={{ width: '100%' }} />
        </label>
      </div>
      <NmseChart c={c} />
      <div style={{ height: '0.6rem' }} />
      <MarginChart c={c} />
      <div style={{ height: '0.8rem' }} />
      <Streams c={c} />
      <p style={{ margin: '0.8rem 0 0' }} data-part="readout" data-first={c.firstDivergence ?? ''} data-fr={c.frDiffers.length}
        data-tf={flips.join(',')} data-flagged={c.flagged.join(',')} data-max-nmse={c.maxTfNmse}>
        {c.firstDivergence == null ? (
          <>
            <span className="chip ok">free running: all {STEPS} tokens match</span>{' '}
          </>
        ) : (
          <>
            <span className="chip fail">free running: first divergence at step {c.firstDivergence}</span>, then {c.frDiffers.length} of {STEPS}{' '}
            tokens differ.{' '}
          </>
        )}
        Teacher forced: {flips.length ? <>{flips.length === 1 ? 'step' : 'steps'} <b>{flips.join(', ')}</b> differ</> : 'no step differs'}
        {c.flagged.length ? <>; flagged {c.flagged.length === 1 ? 'step' : 'steps'} {c.flagged.join(', ')}</> : '; nothing flagged'}. Largest
        teacher-forced NMSE <b>{e1(c.maxTfNmse)}</b>, {c.maxTfNmse < NMSE_LLAMACPP ? 'under' : 'over'} llama.cpp's 1e-7.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the four entry points

const ENTRY: { fw: string; call: string; backend: string; note: string }[] = [
  { fw: 'MLX', call: 'mx.fast.metal_kernel(name, input_names, output_names, source, ...)', backend: 'Metal', note: 'source is the function body; MLX writes the signature. math_mode "safe" by default.' },
  { fw: 'MLX', call: 'mx.fast.cuda_kernel(...)', backend: 'CUDA', note: 'grid counts threads, not blocks, "for compatibility with metal_kernel".' },
  { fw: 'PyTorch', call: 'torch.mps.compile_shader(source)', backend: 'MPS (Metal)', note: 'a whole MSL kernel; call it as lib.<kernel name>(tensors, values).' },
  { fw: 'PyTorch', call: 'torch.utils.cpp_extension.load_inline(name, cpp_sources, cuda_sources=None, ...)', backend: 'CUDA; ROCm builds', note: 'a C++ extension compiled at run time. ROCm not checked separately.' },
  { fw: 'llama.cpp', call: 'none: a new GGML_OP_*', backend: 'every backend', note: 'CPU first, then each backend, each with test-backend-ops cases.' },
];

// ---------------------------------------------------------------- questions

export const questionsSX39: Record<string, Question> = {
  'x3/X3.9/q-nmse': {
    kind: 'numeric',
    id: 'x3/X3.9/q-nmse',
    prompt: (
      <>
        The reference computes a = (1, 2, 2) and your backend b = (1, 2, 2.03). What NMSE does test-backend-ops report?
      </>
    ),
    answer: nmse([1, 2, 2], [1, 2, 2.03]),
    tol: 1e-6,
    explain: (
      <p>
        Sum of squared differences: 0.03² = 0.0009. Sum of squares of the reference: 1 + 4 + 4 = 9. NMSE = 0.0009 / 9 = 1e-4, a thousand
        times llama.cpp's default threshold of 1e-7 per op. That one value is off by 1.5%, which is a bug, not rounding; the threshold is
        far tighter than that.
      </p>
    ),
    source: 'x3-facts D7 (tests/test-backend-ops.cpp, nmse and max_nmse_err)',
  },
  'x3/X3.9/q-late-flip': {
    kind: 'choice',
    id: 'x3/X3.9/q-late-flip',
    prompt: (
      <>
        Your GPU run matches the reference token for token for 40 steps; from step 41 every token differs. Teacher-forced logits agree with
        the reference to an NMSE of 1e-10 at every step. What is most likely?
      </>
    ),
    options: [
      { label: 'Step 41’s top two logits were within rounding of each other. Check its margin before suspecting a kernel.', correct: true },
      { label: 'A kernel has a bug that only shows after 40 tokens, such as a KV cache overflow.', why: 'Then teacher-forced logits at step 41 and later would disagree too, by far more than 1e-10.' },
      { label: 'The reference is wrong from step 41.', why: 'Possible but not indicated: the teacher-forced logits agree with it everywhere.' },
      { label: 'NMSE cannot see errors in late steps.', why: 'It is computed per step; at 1e-10 every step agrees.' },
    ],
    explain: (
      <p>
        Agreement to 1e-10 says every kernel computes the same function as the reference, up to rounding. A free-running run that then
        diverges completely from one step on has met a step whose top two logits were closer than the rounding difference: the GPU took the
        runner-up, and every later step started from a different token. Look up step 41's margin. If it is below twice the logit error, the
        step is too close to judge, and the run is not a failure.
      </p>
    ),
    source: 'ours, from x3-facts D7 (two tests for M10)',
  },
  'x3/X3.9/q-cuda-grid': {
    kind: 'choice',
    id: 'x3/X3.9/q-cuda-grid',
    prompt: (
      <>
        You call <code>mx.fast.cuda_kernel</code> with <code>grid=(n // 256, 1, 1)</code> and <code>threadgroup=(256, 1, 1)</code>, as you
        would launch a CUDA kernel over n elements. What runs?
      </>
    ),
    options: [
      { label: 'n / 256 threads in total, so most elements are never touched. MLX’s grid counts threads; pass grid=(n, 1, 1).', correct: true },
      { label: 'n threads, as in CUDA.', why: 'That is CUDA’s meaning of the grid: a count of blocks. MLX’s is a count of threads.' },
      { label: '256 × n threads.', why: 'Nothing multiplies here; the grid is already the total.' },
      { label: 'An error: MLX’s CUDA kernels have no threadgroup argument.', why: 'The grid and threadgroup sizes are both given; only the grid’s unit differs.' },
    ],
    explain: (
      <p>
        MLX's documentation for <code>cuda_kernel</code> says the grid is in threads and not in threadgroups, for compatibility with{' '}
        <code>metal_kernel</code>. A CUDA habit of passing the number of blocks launches 256 times too few threads, and a kernel with a
        bounds check fails silently: most of the output is never written.
      </p>
    ),
    source: 'x3-facts D8 (python/src/fast.cpp)',
  },
  'x3/X3.9/q-math-mode': {
    kind: 'choice',
    id: 'x3/X3.9/q-math-mode',
    prompt: (
      <>
        You port lab 10.3's softmax into <code>mx.fast.metal_kernel</code>. On the same input it differs from <code>mx.softmax</code> in the
        last bits. Which explanation do the facts support?
      </>
    ),
    options: [
      {
        label: 'The two are compiled differently: metal_kernel uses math_mode "safe" by default, while MLX’s own softmax calls fast::exp; the sums also run in different orders.',
        correct: true,
      },
      { label: 'Your port must have a bug: a correct port matches bit for bit.', why: 'Different exp implementations and summation orders change the last bits of a correct result.' },
      { label: 'mx.softmax runs on the CPU.', why: 'It has a Metal kernel, softmax_single_row.' },
      { label: 'metal_kernel converts the input to half precision.', why: 'Nothing in its documentation says so; the dtypes are yours.' },
    ],
    explain: (
      <p>
        MLX's Metal softmax uses <code>fast::exp</code>; a custom kernel is compiled with math mode "safe" unless you say otherwise. Add a
        different reduction order (lab 10.3's two-stage <code>simd_sum</code> against MLX's own), and identical results would be the
        surprise. Compare with a tolerance, as every lab in the course does.
      </p>
    ),
    source: 'x3-facts D8 (math_mode), B3 and ch10-facts D3 (fast::exp in softmax_single_row); ch10-facts A9 (summation order)',
  },
  'x3/X3.9/q-new-op': {
    kind: 'choice',
    id: 'x3/X3.9/q-new-op',
    prompt: <>You want to add a new fused op to llama.cpp, with a CUDA and a Metal kernel. What does its CONTRIBUTING.md ask?</>,
    options: [
      { label: 'CPU support first, then the backends; test-backend-ops cases for the op; and a test-backend-ops run, which needs at least two backends.', correct: true },
      { label: 'Register your kernel through the plug-in API; no CPU version is needed.', why: 'There is no plug-in API: a new op is a GGML_OP_* with a CPU implementation.' },
      { label: 'A benchmark showing a speed-up on every backend.', why: 'The rules quoted ask for consistency tests, not a speed-up on every backend.' },
      { label: 'The CUDA kernel only; maintainers port it to Metal and HIP.', why: 'HIP does compile the CUDA file, but Metal is separate code, and the rules start from the CPU.' },
    ],
    explain: (
      <p>
        "When adding support for a new model or feature, focus on CPU support only in the initial PR." Then each backend, and for any new or
        changed ggml operator, cases in <code>test-backend-ops</code>, which compares a backend with the CPU. A bug fix also needs a
        regression test that fails before the change and passes after.
      </p>
    ),
    source: 'x3-facts D8, F1 (llama.cpp CONTRIBUTING.md)',
  },
  'x3/X3.9/q-ai-policy': {
    kind: 'choice',
    id: 'x3/X3.9/q-ai-policy',
    prompt: <>Under llama.cpp's contribution policy, which of these is allowed?</>,
    options: [
      { label: 'Submitting AI-assisted kernel code that you have checked line by line, with the AI use disclosed.', correct: true },
      { label: 'An AI-written pull request description, disclosed as such.', why: 'Prohibited: AI may not write posts, PR descriptions or replies to people, disclosed or not.' },
      { label: 'AI-generated code that passes test-backend-ops, without mentioning the AI.', why: 'Undisclosed AI use may get the account banned from contributing.' },
      { label: 'Letting an AI answer the reviewers’ questions on your behalf.', why: '"Responding to humans" is on the prohibited list.' },
    ],
    explain: (
      <p>
        AI-generated code is allowed, and you are fully responsible for every line. You must disclose how AI was used. Using AI to write
        your posts (bug reports, feature requests, pull request descriptions, discussions, replies) is strictly prohibited.
      </p>
    ),
    source: 'x3-facts F1 (CONTRIBUTING.md, AI policy)',
  },
};

// ---------------------------------------------------------------- the section

export function SX39() {
  return (
    <>
      <p>
        Your GPU produces tokens. Are they right? "The same tokens as the reference" sounds like the obvious test, and it is the one that
        fails for no good reason. This page shows how the projects compare a backend with a reference, why exact greedy agreement can fail
        late with every kernel correct, how to run your own kernel inside MLX or PyTorch, and what each project asks of a kernel change.
      </p>
      <Beyond>
        <p>
          Module X3 is ours. Project facts from notes/today/x3-facts.md (D7, D8, F1, F2), read at pinned commits on 2026-11-20. The
          two-test design and the toy model are ours too.
        </p>
      </Beyond>

      <h2>How llama.cpp tests a backend</h2>
      <p>
        llama.cpp's <code>test-backend-ops</code> runs the same GGML ops on two backends and compares each result with the CPU backend's.
        Not by equality: by a normalised error, NMSE = Σ (a − b)² / Σ a², with a the reference. The default pass mark is 1e-7, and an op
        can override it. You pick ops with <code>-o</code> (for example <code>-o ADD,MUL_MAT</code>) and a backend with <code>-b</code>:{' '}
        <code>CUDA0</code>, <code>MTL0</code>.
      </p>
      <Today checked="2026-11-20 (notes/today/x3-facts.md D7; table “Where the backends really differ”)">
        <p>
          <code>max_nmse_err()</code> returns 1e-7 by default (test-backend-ops.cpp line 1266). The HIP backend is tested by the same tool,
          from the same CUDA sources; on Metal the backend is <code>MTL0</code>.
        </p>
      </Today>
      <p>
        Two things to notice. Dividing by Σ a² makes the measure independent of the scale of the output, so one threshold serves a softmax
        and a matmul. And 1e-7 is a figure per op, on one op's output. It says nothing about how far the logits of a whole model may drift
        after thirty layers.
      </p>
      <Ask q={questionsSX39['x3/X3.9/q-nmse']} label="Work it out" />

      <h2>Why one test is not enough</h2>
      <p>
        Logits from two backends differ by rounding: summation orders differ (chapter 10), <code>exp</code> is implemented differently, and
        Metal's default math mode is not strict IEEE. Usually that difference is far smaller than the gap between the top two logits, and
        greedy decoding picks the same token. But sometimes the top two are closer than the rounding difference. Then the GPU can take the
        runner-up, and from that step on it is generating a different text.
      </p>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md A9, A11; x1-facts C5; ch05-facts C7)">
        <p>
          Float summation is not associative, so a different reduction order changes the last bits; <code>expf</code>'s accuracy differs by
          backend; Metal does not default to IEEE-strict math, and Apple's sources disagree on which mode is the default (ch05 C7).
        </p>
      </Today>
      <p>
        So M10 runs two tests. <b>Teacher forced</b>: feed both runs the reference's tokens, so every step sees the same input, and require
        the logits to agree within a stated tolerance. <b>Free running</b>: let the GPU feed back its own tokens and require the same greedy
        tokens, but only on steps whose top-two margin is more than twice that tolerance. A step closer than that is flagged as too close to
        judge, not counted as a failure.
      </p>
      <Simplification>
        <p>
          The perturbation is synthetic: the same fixed pattern of independent noise per logit, scaled by the slider. Real backend
          differences are structured and correlated across logits and steps. The toy model's logits do react to the previous token, as a
          real model's do, but its close steps are built in.
        </p>
      </Simplification>
      <CompareRuns />
      <p>
        Three things to find. Exact token agreement fails first at step 19, while every NMSE is still below 1e-11, at least ten thousand
        times under llama.cpp's threshold. Once it fails it keeps failing: free running, every later token differs, and the rings in the
        first chart jump to about 1, because the two runs are no longer computing on the same input. Teacher forced, only the close steps
        can differ, and each one that does was flagged first. Drag to 1e-3 and the first divergence moves to step 13, with every NMSE
        still under 1e-7.
      </p>
      <Ask q={questionsSX39['x3/X3.9/q-late-flip']} />
      <LateDivergenceScene />

      <h2>Your kernel inside the framework you use</h2>
      <p>
        You do not need to fork a framework to try a kernel in it. MLX and PyTorch each take kernel source from Python and compile it at run
        time, on Metal and on CUDA. llama.cpp has no such door: a kernel there is a new GGML op.
      </p>
      <Today checked="2026-11-20 (notes/today/x3-facts.md D8)">
        <p>
          MLX <code>python/src/fast.cpp</code> at the pinned commit; PyTorch docs 2.15 for <code>compile_shader</code> and{' '}
          <code>load_inline</code>. MLX has no AMD backend (B9).
        </p>
      </Today>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ width: '100%' }} data-part="entry">
          <thead>
            <tr><th>Framework</th><th>Backend</th><th>Entry point, and what to know</th></tr>
          </thead>
          <tbody>
            {ENTRY.map((r) => (
              <tr key={r.call}>
                <td>{r.fw}</td>
                <td>{r.backend}</td>
                <td>
                  <code style={{ fontSize: 'var(--step--1)', wordBreak: 'break-word' }}>{r.call}</code>
                  <div className="muted">{r.note}</div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Ask q={questionsSX39['x3/X3.9/q-cuda-grid']} />
      <p>
        The bridge from this course: port lab 10.3's <code>softmax_rows</code> into <code>mx.fast.metal_kernel</code> and compare it with{' '}
        <code>mx.softmax</code>. The source string is the kernel's body only; MLX generates the signature from your input and output names.
        Lab 10.3 receives its scratch buffer from <code>launch.mm</code>; in a body you can declare it instead, as{' '}
        <code>threadgroup float scratch[32];</code>. Its launch was one threadgroup of 256 threads per row, so the grid is rows × 256 threads,
        not rows. Which index attributes the generated signature provides is in MLX's documentation for <code>metal_kernel</code>; this page
        has not run the port.
      </p>
      <Ask q={questionsSX39['x3/X3.9/q-math-mode']} />

      <h2>Contributing a kernel change</h2>
      <p>
        <b>llama.cpp.</b> If you change ggml, run <code>test-backend-ops</code>, which needs two backends. If you change or add an operator,
        add its cases there. A bug fix needs a reproducible issue and a regression test that fails before your change and passes after. A
        new feature starts with CPU support only. Style: 4 spaces, brackets on the same line, <code>void * ptr</code>,{' '}
        <code>int &amp; a</code>.
      </p>
      <Ask q={questionsSX39['x3/X3.9/q-new-op']} />
      <p>
        Its AI policy is explicit. AI-generated code is allowed, and you are responsible for every line. You must disclose how AI was used;
        undisclosed use can get an account permanently banned. Using AI to write posts, bug reports, pull request descriptions or replies to
        people is prohibited.
      </p>
      <Today checked="2026-11-20 (notes/today/x3-facts.md F1, F2)">
        <p>
          llama.cpp CONTRIBUTING.md (last changed 2026-11-16); MLX and PyTorch CONTRIBUTING.md at the pinned commits. PyTorch's
          "AI-Assisted Development" section was not read.
        </p>
      </Today>
      <Ask q={questionsSX39['x3/X3.9/q-ai-policy']} />
      <p>
        <b>MLX.</b> Every contribution needs thorough tests; run <code>python3 python/tests/run.py</code> and format with{' '}
        <code>pre-commit run --all</code>. AI-assisted coding is allowed, and its output must meet the same standard as anything else.{' '}
        <b>PyTorch.</b> Run <code>python test/run_test.py</code>, or one file with, for example, <code>pytest test/test_nn.py -k Loss -v</code>.
        Its guide has a section on AI-assisted development, which this course has not read: read it before you open a pull request.
      </p>
      <p>
        Whichever project you contribute to, the habit is the one this page built: compare with a reference by a normalised error, state the
        tolerance, and when exact agreement fails, look at the margin before you look at the kernel. Lab X3.2 on the{' '}
        <a href={href('x3', 'labs')}>labs page</a> runs both tests on M10.
      </p>
    </>
  );
}
