import { Fragment, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BLOCKS,
  MILESTONES,
  MILESTONE_LABEL,
  PRIMITIVES,
  PRIMITIVE_LABEL,
  crossover,
  decodeBytes,
  judge,
  layerFlops,
  type FfnKind,
  type LayerShape,
  type Milestone,
  type Phase,
  type Primitive,
} from '../../sim/llmflops';

export const questionsS201: Record<string, Question> = {
  'ch20/20.1/q-layernorm': {
    kind: 'choice',
    id: 'ch20/20.1/q-layernorm',
    prompt: (
      <>
        LayerNorm (Fig. 20.1: <code>(x − μ)/σ · γ + β</code> for each token&rsquo;s row) is closest to which kernel you
        have already written?
      </>
    ),
    options: [
      { label: 'M4\'s row softmax: per row, reductions then an elementwise pass.', correct: true },
      { label: 'A tiled GEMM, because γ and β are learned parameters like weights.', why: 'γ and β are vectors applied elementwise. Nothing is multiplied by a matrix.' },
      { label: 'A scan, since the variance needs the mean first.', why: 'Needing one total before another is two reductions in sequence, not a running prefix.' },
      { label: 'A privatised histogram, one bin per feature.', why: 'Nothing is counted into bins; every element is read and rewritten in place.' },
    ],
    explain: (
      <p>
        Per row: reduce to the mean, reduce to the variance (or both at once), then rewrite each element. That is the
        shape of lab 10.3&rsquo;s softmax, one block per row, with different operators. Today&rsquo;s open models use
        RMSNorm, which drops the mean and needs one reduction (facts E8).
      </p>
    ),
    source: 'ch. 20 p. 481, Fig. 20.1; ch. 10 and lab 10.3',
  },
  'ch20/20.1/q-flops-share': {
    kind: 'numeric',
    id: 'ch20/20.1/q-flops-share',
    prompt: (
      <>
        One layer with the book&rsquo;s shapes: four d_model × d_model projections, a two-matrix ReLU feed-forward with
        d_ff = 4 d_model, no skipping of the masked half. At N = 4096 and d_model = 4096, what are the FLOPs of Q K
        <sup>T</sup> and P V together, divided by the FLOPs of the linear layers?
      </>
    ),
    answer: 0.1667,
    tol: 0.03,
    explain: (
      <p>
        Per token the two products cost 2 × 2 N d_model = 4 N d_model FLOPs and the linear layers 2 × (4 + 8)
        d_model² = 24 d_model². The ratio is N / (6 d_model) = 1/6. The products only catch up at N = 6 d_model, here
        24,576 tokens.
      </p>
    ),
    source: "our arithmetic from the book's shapes (pp. 480–483); the explorer above",
  },
  'ch20/20.1/q-new-kernels': {
    kind: 'choice',
    id: 'ch20/20.1/q-new-kernels',
    prompt: <>Which blocks of Fig. 20.1 need a kernel you had not written by the end of chapter 15?</>,
    options: [
      { label: 'Q K^T with the causal mask, its row softmax, and P V.', correct: true },
      { label: 'The feed-forward, because of the activation between its two layers.', why: 'Two GEMMs and an elementwise ReLU: M1 to M3 and M0.' },
      { label: 'LayerNorm, because of its learned γ and β.', why: 'Two row reductions and an elementwise pass, the shape of M4.' },
      { label: 'The embedding lookup, because it indexes by token.', why: 'A gather: one row copied per token ID, no arithmetic.' },
    ],
    explain: (
      <p>
        Everything else is a GEMM or GEMV, a row reduction, elementwise work or a gather. Even the three attention steps
        can run unfused on old kernels (two GEMMs and a softmax); what is new is the causal mask and, in section 20.5,
        fusing them so the N × N matrices never exist. The book says the same of its unfused version: the softmax is
        the only new piece (p. 486).
      </p>
    ),
    source: 'ch. 20 Fig. 20.1, p. 479, and p. 486',
  },
  'ch20/20.1/q-rope': {
    kind: 'choice',
    id: 'ch20/20.1/q-rope',
    prompt: (
      <>
        Llama-family models rotate Q and K by position inside each attention layer, and cache K after the rotation. What
        must a decode kernel do to the cached keys at each step?
      </>
    ),
    options: [
      { label: 'Nothing: only the new q and k get rotated, at their position.', correct: true },
      { label: 'Rotate every cached key again, by the distance to the new position.', why: 'Each key was rotated once, by its own position, before it was stored. That is the point of caching after the rotation.' },
      { label: 'Rotate V the same way, so that P V lines up with the keys.', why: 'Only Q and K are rotated; their inner product carries the position. V is untouched.' },
      { label: 'Add a positional encoding to the cache before reading it.', why: 'That is the original transformer\'s scheme, applied once to the embeddings, not to a cache.' },
    ],
    explain: (
      <p>
        Meta&rsquo;s reference code applies the rotary embedding to Q and K and then writes K into the cache (facts D3),
        so the cache holds rotated keys and the decode step rotates only the new token&rsquo;s query and key at its own
        position. A cache written by one convention and read by another is a silent bug.
      </p>
    ),
    source: 'notes/today/ch20-facts.md E8, D3',
  },
  'ch20/20.1/q-hidden': {
    kind: 'numeric',
    id: 'ch20/20.1/q-hidden',
    prompt: (
      <>
        DeepSeek-V2 has 128 attention heads of width 128 and a hidden dimension of 5120. What does the book&rsquo;s
        &ldquo;hidden dimension = h_q × d&rdquo; (p. 505) give for it?
      </>
    ),
    answer: 16384,
    explain: (
      <p>
        128 × 128 = 16,384, more than three times the real 5120. Heads × head width need not equal the model width, so
        the cache formulas of section 20.6 need h and d, never the hidden size. The book&rsquo;s identity holds for
        GPT-3 (96 × 128 = 12,288), Llama 2 7B and Mistral 7B (32 × 128 = 4096).
      </p>
    ),
    source: 'notes/today/ch20-facts.md E2, E8',
  },
};

// ---------------------------------------------------------------- which kernel is this block?
function Classify() {
  const [prim, setPrim] = useState<Record<string, Primitive | ''>>({});
  const [mile, setMile] = useState<Record<string, Milestone | ''>>({});
  const [checked, setChecked] = useState(false);
  const verdicts = BLOCKS.map((b) => judge(b, prim[b.id] || null, mile[b.id] || null));
  const right = verdicts.filter((v) => v.primitive && v.milestone).length;
  const answered = BLOCKS.filter((b) => prim[b.id] && mile[b.id]).length;
  const reveal = () => {
    setPrim(Object.fromEntries(BLOCKS.map((b) => [b.id, b.primitive[0]])));
    setMile(Object.fromEntries(BLOCKS.map((b) => [b.id, b.milestone[0]])));
    setChecked(true);
  };
  return (
    <Bench
      title="Which kernel is this block?"
      task="For each block of Fig. 20.1, pick what runs it and which milestone builds it. Then check."
      foot={
        <>
          Top to bottom through one decoder layer, plus what happens before and after the stack. Some blocks accept two
          answers; the feedback says why.
        </>
      }
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-table="classify" style={{ margin: 0, background: 'var(--page)' }}>
          <thead>
            <tr>
              <th>Block</th>
              <th>What runs it</th>
              <th>Milestone</th>
              {checked ? <th>Verdict</th> : null}
            </tr>
          </thead>
          <tbody>
            {BLOCKS.map((b, i) => {
              const v = verdicts[i];
              const ok = v.primitive && v.milestone;
              return (
                <Fragment key={b.id}>
                <tr data-block={b.id} data-verdict={checked ? (ok ? 'right' : 'wrong') : ''} style={checked ? { borderBottom: 0 } : undefined}>
                  <td style={{ minWidth: '9rem' }}>
                    {b.name}
                    {b.fig ? <span className="muted"> ({b.fig})</span> : null}
                  </td>
                  <td>
                    <select
                      aria-label={`What runs ${b.name}`}
                      value={prim[b.id] ?? ''}
                      onChange={(e) => {
                        setPrim({ ...prim, [b.id]: e.target.value as Primitive });
                        setChecked(false);
                      }}
                    >
                      <option value="">choose</option>
                      {PRIMITIVES.map((p) => (
                        <option key={p} value={p}>
                          {PRIMITIVE_LABEL[p]}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td>
                    <select
                      aria-label={`Milestone for ${b.name}`}
                      value={mile[b.id] ?? ''}
                      onChange={(e) => {
                        setMile({ ...mile, [b.id]: e.target.value as Milestone });
                        setChecked(false);
                      }}
                    >
                      <option value="">choose</option>
                      {MILESTONES.map((m) => (
                        <option key={m} value={m}>
                          {MILESTONE_LABEL[m]}
                        </option>
                      ))}
                    </select>
                  </td>
                  {checked ? (
                    <td>
                      {ok ? <span className="chip ok">right</span> : <span className="chip fail">{!v.primitive && !v.milestone ? 'both wrong' : !v.primitive ? 'what runs it' : 'milestone'}</span>}
                    </td>
                  ) : null}
                </tr>
                {checked ? (
                  <tr data-why={b.id}>
                    <td colSpan={4} className="muted" style={{ whiteSpace: 'normal', paddingTop: 0 }}>
                      {b.why}
                    </td>
                  </tr>
                ) : null}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" onClick={() => setChecked(true)} disabled={answered === 0}>
          Check
        </button>
        <button className="quiet" onClick={reveal}>
          Show the answers
        </button>
        <span className="muted" role="status" data-score={checked ? right : ''}>
          {checked ? `${right} of ${BLOCKS.length} rows right.` : `${answered} of ${BLOCKS.length} rows answered.`}
        </span>
      </div>
      {checked && right === BLOCKS.length ? (
        <p style={{ margin: '0.8rem 0 0' }} data-discovery>
          Only three rows say M6. Everything else in the layer is a kernel you already have: the chapter&rsquo;s new
          work is the three attention steps, and the reason they need new work is the size of what passes between them.
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- where the arithmetic goes
const D_MODELS = [512, 1024, 2048, 4096, 8192, 16384];
const PARTS: { key: 'proj' | 'qk' | 'pv' | 'ffn'; label: string; alpha: number }[] = [
  { key: 'proj', label: 'Q, K, V, O projections', alpha: 0.82 },
  { key: 'ffn', label: 'feed-forward', alpha: 0.5 },
  { key: 'qk', label: 'Q K^T', alpha: 0.24 },
  { key: 'pv', label: 'P V', alpha: 0.1 },
];
const fmtFlops = (x: number) => {
  if (x >= 1e12) return (x / 1e12).toPrecision(3) + ' TFLOP';
  if (x >= 1e9) return (x / 1e9).toPrecision(3) + ' GFLOP';
  if (x >= 1e6) return (x / 1e6).toPrecision(3) + ' MFLOP';
  return x.toPrecision(3) + ' FLOP';
};
const fmtMiB = (x: number) => {
  const m = x / 2 ** 20;
  return (m >= 10 ? Math.round(m).toLocaleString('en-US') : m.toPrecision(2)) + ' MiB';
};
const pct = (x: number) => (100 * x).toFixed(x < 0.01 ? 2 : 1) + '%';
const MAX_LOG = 17; // 131,072 tokens

function Explorer() {
  const [logN, setLogN] = useState(12);
  const [dModel, setDModel] = useState(4096);
  const [ratio, setRatio] = useState(4);
  const [ffn, setFfn] = useState<FfnKind>('relu');
  const [skip, setSkip] = useState(false);
  const [phase, setPhase] = useState<Phase>('prefill');
  const n = Math.round(2 ** logN);
  const shape: LayerShape = { dModel, ffRatio: ratio, ffn, causalSkip: skip };
  const f = layerFlops(n, shape, phase);
  const cross = crossover(shape, phase);
  const bytes = decodeBytes(n, shape, 1, 2);
  const curve = useMemo(
    () =>
      Array.from({ length: MAX_LOG * 8 + 1 }, (_, i) => {
        const x = i / 8;
        return { x, y: layerFlops(Math.round(2 ** x), shape, phase).attnShare };
      }),
    [dModel, ratio, ffn, skip, phase], // eslint-disable-line react-hooks/exhaustive-deps
  );
  // Plot geometry.
  const W = 640;
  const H = 210;
  const L = 44;
  const R = 12;
  const T = 12;
  const Bm = 34;
  const px = (x: number) => L + (x / MAX_LOG) * (W - L - R);
  const py = (y: number) => T + (1 - y) * (H - T - Bm);
  const crossX = Math.log2(Math.max(1, cross));
  return (
    <Bench
      title="Where a layer's arithmetic goes"
      task="Find the context length at which Q K^T and P V cost as much as the linear layers. Then switch to one decode token."
      foot={<>Our arithmetic from the book&rsquo;s shapes, not the book&rsquo;s. FLOPs count the matrix products only, 2 per multiply-add.</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ minWidth: '14rem' }}>
          Context length N: <b data-n>{n.toLocaleString('en-US')}</b>
          <input type="range" min={0} max={MAX_LOG} step={0.25} value={logN} onChange={(e) => setLogN(+e.target.value)} aria-label="context length (log scale)" />
        </label>
        <label className="field">
          d_model
          <select value={dModel} onChange={(e) => setDModel(+e.target.value)} aria-label="d_model">
            {D_MODELS.map((d) => (
              <option key={d} value={d}>
                {d.toLocaleString('en-US')}
              </option>
            ))}
          </select>
        </label>
        <label className="field" style={{ minWidth: '10rem' }}>
          d_ff / d_model: <b>{ratio}</b>
          <input type="range" min={1} max={8} step={0.25} value={ratio} onChange={(e) => setRatio(+e.target.value)} aria-label="d_ff over d_model" />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="seg" role="group" aria-label="Feed-forward">
          {(['relu', 'swiglu'] as FfnKind[]).map((k) => (
            <button key={k} aria-pressed={ffn === k} onClick={() => setFfn(k)}>
              {k === 'relu' ? 'ReLU, 2 matrices' : 'SwiGLU, 3 matrices'}
            </button>
          ))}
        </span>
        <span className="seg" role="group" aria-label="Phase">
          {(['prefill', 'decode'] as Phase[]).map((p) => (
            <button key={p} aria-pressed={phase === p} onClick={() => setPhase(p)}>
              {p === 'prefill' ? 'prefill of N tokens' : 'one decode token'}
            </button>
          ))}
        </span>
        <label className="check">
          <input type="checkbox" checked={skip} onChange={(e) => setSkip(e.target.checked)} disabled={phase === 'decode'} /> skip the masked half
        </label>
      </div>

      <div aria-label="Share of FLOPs" data-shares style={{ display: 'flex', height: '1.6rem', border: '1px solid var(--rule)', borderRadius: 3, overflow: 'hidden' }}>
        {PARTS.map((p) => (
          <div
            key={p.key}
            title={`${p.label}: ${pct(f[p.key] / f.total)}`}
            data-part={p.key}
            data-share={(f[p.key] / f.total).toFixed(4)}
            style={{ width: `${(100 * f[p.key]) / f.total}%`, background: 'var(--ink)', opacity: p.alpha }}
          />
        ))}
      </div>
      <div className="legend" style={{ marginTop: '0.4rem', display: 'flex', flexWrap: 'wrap', gap: '0.3rem 1rem' }}>
        {PARTS.map((p) => (
          <span key={p.key}>
            <i style={{ background: 'var(--ink)', opacity: p.alpha }} />
            {p.label} {pct(f[p.key] / f.total)}
          </span>
        ))}
      </div>
      <p style={{ margin: '0.8rem 0' }} data-readout>
        {phase === 'prefill' ? 'Prefill of' : 'One decode token at context'} {n.toLocaleString('en-US')}
        {phase === 'prefill' ? ' tokens' : ''}: {fmtFlops(f.total)} per layer. The attention products are{' '}
        <b data-ratio={f.attnOverLinear.toFixed(4)}>{f.attnOverLinear < 10 ? f.attnOverLinear.toPrecision(3) : Math.round(f.attnOverLinear)}</b> times the
        linear layers; they draw level at N = <b data-cross={Math.round(cross)}>{Math.round(cross).toLocaleString('en-US')}</b>.
      </p>
      {phase === 'decode' ? (
        <p style={{ margin: '0 0 0.8rem' }} data-bytes>
          Bytes this token reads per layer, 16-bit, one sequence: weights <b data-wbytes={bytes.weights}>{fmtMiB(bytes.weights)}</b>, its K
          and V cache <b data-cbytes={bytes.cache}>{fmtMiB(bytes.cache)}</b>. A batch of b sequences reads the weights once and
          b caches, so the cache read is the larger once b × N passes{' '}
          <b data-bcross={Math.round(bytes.crossBatchN)}>{Math.round(bytes.crossBatchN).toLocaleString('en-US')}</b>.
        </p>
      ) : null}

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Attention products as a share of the layer's FLOPs, against context length">
        {[0, 0.25, 0.5, 0.75, 1].map((y) => (
          <g key={y}>
            <line x1={L} x2={W - R} y1={py(y)} y2={py(y)} stroke="var(--rule)" strokeWidth={y === 0.5 ? 1 : 0.6} strokeDasharray={y === 0.5 ? '4 3' : undefined} />
            <text x={L - 6} y={py(y) + 4} textAnchor="end" fontSize="11">
              {y * 100}%
            </text>
          </g>
        ))}
        {[0, 4, 8, 12, 16].map((x) => (
          <text key={x} x={px(x)} y={H - Bm + 16} textAnchor="middle" fontSize="11">
            {(2 ** x).toLocaleString('en-US')}
          </text>
        ))}
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" fontSize="11">
          context length N (log scale)
        </text>
        <text x={L + 8} y={T + 10} fontSize="11" fill="var(--ink-2)">
          share of the layer&rsquo;s FLOPs in Q K^T and P V
        </text>
        <polyline fill="none" stroke="var(--ink)" strokeWidth={2} points={curve.map((p) => `${px(p.x)},${py(p.y)}`).join(' ')} />
        {crossX <= MAX_LOG ? (
          <g data-crossmark>
            <line x1={px(crossX)} x2={px(crossX)} y1={T} y2={H - Bm} stroke="var(--ink-2)" strokeDasharray="3 3" />
            <text x={px(crossX) + (crossX > MAX_LOG - 4 ? -4 : 4)} y={T + 10} fontSize="11" textAnchor={crossX > MAX_LOG - 4 ? 'end' : 'start'}>
              products = linear layers
            </text>
          </g>
        ) : null}
        <circle cx={px(logN)} cy={py(f.attnShare)} r={5} fill="var(--page)" stroke="var(--ink)" strokeWidth={2} />
      </svg>
    </Bench>
  );
}

export function S201() {
  return (
    <>
      <p>
        You know this architecture from the PyTorch side, so this section only does one job: name the kernel behind each
        block. An LLM is decoder-only and auto-regressive. It runs the whole stack once per generated token, feeding
        each new token back in, until it emits an end-of-sequence token. (The original transformer also had encoder
        layers that mix the input once; discriminative tasks such as classification need no decoder. Generation is what
        LLMs do, and it is what this chapter optimises.)
      </p>
      <Source section="20.1" pages="478–479" />

      <h2>Fig. 20.1 as a list of kernels</h2>
      <p>
        Read the figure from the top. The tokenizer turns text into integer IDs through a table built in training; the
        embedding layer looks each ID up in a learned table; the original transformer adds a positional encoding to each
        embedding. The rows, one per token so far, form the matrix X that enters the first layer. Each layer is
        multi-head attention, an Add &amp; Norm, a feed-forward network and a second Add &amp; Norm. After the last
        layer, the un-embedding layer turns the final row into a distribution over the vocabulary, and something
        chooses the next token from it; the book gives that choice one sentence.
      </p>
      <Source section="20.1" pages="479–481" figs="Fig. 20.1" />
      <p>
        Call-out 5 zooms into the attention heads: a linear projection to Q, K and V, then Q K<sup>T</sup>, a softmax,
        and a product with V; the heads&rsquo; outputs are concatenated and projected once more. Sort every block
        before reading on.
      </p>
      <Aside>
        <p>
          Fig. 20.1&rsquo;s call-out ends with &ldquo;S x V&rdquo;. It is P × V: the softmax output times V, as Fig. 20.2
          and Eq. 20.1 have it.
        </p>
      </Aside>
      <Classify />
      <Ask q={questionsS201['ch20/20.1/q-layernorm']} />
      <Ask q={questionsS201['ch20/20.1/q-new-kernels']} />

      <h2>The model&rsquo;s dimensions</h2>
      <p>
        The book names three: depth (the number of layers), the width of each embedding vector, and the number of
        attention heads per layer. It then says a model&rsquo;s parameter count is the product of the three.
      </p>
      <Source section="20.1" pages="481" />
      <p>
        Take that as a figure of speech. With the book&rsquo;s shapes a layer holds 4 d_model² attention weights and 8
        d_model² feed-forward weights; 32 layers of width 4096 come to 6.4 billion, while 32 × 4096 × 32 is about four
        million.
      </p>
      <Beyond>
        <p>The parameter count is our arithmetic from the book&rsquo;s shapes.</p>
      </Beyond>
      <p>
        One more piece of notation to fix now. The book writes d both for the embedding width and for a head&rsquo;s
        width, which is harmless while it draws one head whose matrices are all d wide. In this course&rsquo;s labs d is
        the <b>head width</b> (128 in GPT-3, Llama 2 7B and Mistral 7B), and d_model is the model width, the number of heads
        times d in most models.
      </p>
      <Ask q={questionsS201['ch20/20.1/q-hidden']} label="Check the identity" />

      <h2>Where the arithmetic goes</h2>
      <p>
        The book calls the attention heads the most expensive part of the layer and spends the chapter on them. In
        arithmetic, at the context lengths you usually run, they are not: the linear layers dominate. In bytes it
        depends on how the work is run, and the reasons are what the rest of the chapter is about. Find the crossover
        yourself.
      </p>
      <Source section="20.1" pages="481–482" />
      <p>
        The explorer counts the multiply-adds of one layer&rsquo;s matrix products, for a whole prompt (prefill) or for
        one new token (decode), and plots the attention products&rsquo; share as the context grows.
      </p>
      <Simplification>
        <p>
          Matrix products only. Softmax, norms, biases, activations, the embedding and the un-embedding to the
          vocabulary are left out; so are grouped-query attention&rsquo;s smaller K and V projections.
        </p>
      </Simplification>
      <Explorer />
      <p>
        Per token, the four projections and the ReLU feed-forward cost 24 d_model² FLOPs, and the two attention
        products 4 N d_model. Their ratio is N / (6 d_model): at d_model = 4096 the products draw level only past 24,000
        tokens, and skipping the masked half pushes that to about 49,000. Decode has the same ratio, because one token
        meets N keys. So attention&rsquo;s problem at ordinary lengths is not FLOPs. Where it is a problem, it is{' '}
        <b>bytes</b>, in two places. In unfused prefill, the N × N matrices S and P that the pipeline writes and reads
        (section 20.3) dominate the traffic. In decode, one sequence reads 2 N d_model cache elements per layer against
        12 d_model² weights: the same N / (6 d_model), so for one sequence at ordinary lengths the weights still
        dominate. But a batch shares the weights and not the caches, so the cache read wins once b × N passes 6
        d_model, which batching (section 20.6) reaches quickly. Switch the explorer to one decode token to see both.
      </p>
      <Beyond>
        <p>
          The book gives no FLOP or byte counts. These are ours, from its shapes: four d_model × d_model projections,
          d_ff = 4 d_model, 2 FLOPs per multiply-add, each weight and cache element read once.
        </p>
      </Beyond>
      <Ask q={questionsS201['ch20/20.1/q-flops-share']} label="Work it out" />

      <h2>What today&rsquo;s open models changed</h2>
      <p>
        Fig. 20.1 is the original transformer. The models you will load differ in three ways that reach into the
        kernels. Normalisation comes before each sub-layer and is RMSNorm, one reduction instead of two. The
        feed-forward is SwiGLU, three matrices instead of two (switch the explorer to it). And position is a rotation
        applied to Q and K inside every attention layer, not a vector added once at the bottom; the KV cache therefore
        stores keys that are already rotated.
      </p>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md E8, D3, E2)">
        <p>
          Llama 2: &ldquo;pre-normalization using RMSNorm&rdquo;, &ldquo;SwiGLU&rdquo;, &ldquo;rotary positional
          embeddings&rdquo;; PaLM also lists rotary embeddings. Heads × head width equals the model width for GPT-3, Llama 2 7B
          and Mistral 7B, not for DeepSeek-V2.
        </p>
      </Today>
      <Ask q={questionsS201['ch20/20.1/q-rope']} />
      <p>
        Two more remarks from the section. Training is dominated by large matrix products in the attention heads and the
        feed-forward layers, which is chapter 15&rsquo;s territory. And retrieval-augmented generation pastes retrieved
        documents into the prompt, which makes N much larger: the rest of the chapter is about what a larger N costs.
        Next, section <a href={href('ch20', '20.2')}>20.2</a> opens one head.
      </p>
      <Source section="20.1" pages="481–482" />
    </>
  );
}
