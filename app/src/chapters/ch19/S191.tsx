import { useEffect, useMemo, useState, type KeyboardEvent } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BOOK_F,
  BOOK_LAYER,
  BOOK_X,
  LENET,
  batchTraffic,
  changedByFlip,
  fAt,
  forward,
  hOut,
  lessonRegion,
  perInputMap,
  shapes,
  terms,
  wOut,
  type LayerShape,
  type Stage,
} from '../../sim/cnnlayer';

// Section 19.1: the convolutional layer as the book defines it. The worked example of
// Fig. 19.2 as a term-by-term stepper, LeNet-5's three convolution layers with their
// shapes and batch traffic, and the sequential reference from lab 19.1's driver.

const S = BOOK_LAYER;
const LAB_MAIN = 'ch19/lab1_convgemm/main.cpp';
const big = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => (v >= 0.1 ? (v * 100).toFixed(1) : v >= 0.001 ? (v * 100).toFixed(2) : '<0.1') + '%';
const dims = (d: number[]) => d.join(' × ');
const bytes = (b: number) => (b >= 1 << 20 ? `${(b / (1 << 20)).toFixed(1)} MiB` : b >= 1024 ? `${(b / 1024).toFixed(1)} KiB` : `${b} B`);

// ---------------------------------------------------------------- questions

export const questionsS191: Record<string, Question> = {
  'ch19/19.1/q-flip': {
    kind: 'numeric',
    id: 'ch19/19.1/q-flip',
    prompt: (
      <>
        Suppose a library rotated every filter by 180° before sliding it, which is what the word convolution means in
        signal processing. What would <code>Y[0, 0, 0]</code> of the book&rsquo;s example become?
      </>
    ),
    answer: 15,
    explain: (
      <>
        <p>
          Rotated, F[0, 0] = [1 1; 2 2] becomes [2 2; 1 1], and against the patch [1 2; 1 1] it gives 2 + 4 + 1 + 1 = 8
          instead of 7. F[0, 1] (all ones) and F[0, 2] ([0 1; 1 0]) look the same after a half turn, so they still give 5
          and 2. Total 15. Tick the rotation switch in the explorer to see which other outputs move.
        </p>
        <p>
          The book and PyTorch agree with each other: neither flips. A flip changes the result whenever a filter is not
          symmetric under a half turn, so weights trained under one convention give wrong answers under the other.
        </p>
      </>
    ),
    source: 'ch. 19 §19.1, p. 457; notes/today/ch07-facts.md D1 (PyTorch, cuDNN, MIOpen compute cross-correlation); ch19-facts.md C3 (MPSGraph documents neither choice)',
  },
  'ch19/19.1/q-outsize': {
    kind: 'numeric',
    id: 'ch19/19.1/q-outsize',
    prompt: (
      <>
        <code>nn.Conv2d(6, 16, 5)</code> with every other argument left at its default receives a 14 × 14 input. What
        is the height of its output?
      </>
    ),
    answer: 10,
    explain: (
      <p>
        The default <code>padding=0</code> is the book&rsquo;s layer with no ghost cells: H − K + 1 = 14 − 5 + 1 = 10.
        This is LeNet&rsquo;s C3. To keep 14 × 14 you would ask for <code>padding=2</code>, chapter 7&rsquo;s zero ghost
        cells.
      </p>
    ),
    source: 'ch. 19 §19.1, p. 457; notes/today/ch07-facts.md D1 (PyTorch 2.15 Conv2d defaults)',
  },
  'ch19/19.1/q-weights': {
    kind: 'numeric',
    id: 'ch19/19.1/q-weights',
    prompt: (
      <>
        LeNet&rsquo;s C3 maps 6 input maps of 14 × 14 to 16 output maps of 10 × 10 with 5 × 5 filters. How many weights
        does F hold?
      </>
    ),
    answer: 2400,
    unit: 'weights',
    explain: (
      <p>
        One filter per (output map, input map) pair: 16 · 6 = 96 filters of 5 · 5 = 25 weights, so 2400. The map size
        does not appear: the same weights slide over every position. F&rsquo;s shape is 16 × 6 × 5 × 5.
      </p>
    ),
    source: 'ch. 19 §19.1, pp. 456, 458',
  },
  'ch19/19.1/q-batch': {
    kind: 'choice',
    id: 'ch19/19.1/q-batch',
    prompt: (
      <>
        In the batched layer, X and Y gain an N dimension and F does not. As N grows, what can happen to the traffic
        for F <em>per sample</em>?
      </>
    ),
    options: [
      { label: 'It can fall as 1/N, because one fetch of F can serve every sample.', correct: true },
      {
        label: 'Nothing: every sample needs all the weights, so each sample costs the same F traffic.',
        why: 'Every sample does need every weight. It does not need its own copy fetched from memory: a kernel that keeps a weight on chip while it serves several samples pays for that weight once.',
      },
      { label: 'It grows, because a larger grid has more blocks fetching F.', why: 'More blocks can fetch more, but nothing forces them to; the ideal is still one fetch for the whole batch.' },
      { label: 'F must be copied once per sample before the layer runs.', why: 'F is shared, not replicated (p. 459): there is one array for the whole batch.' },
    ],
    explain: (
      <p>
        The weights are shared across the batch (p. 459), so the ideal traffic for F is fixed while the work grows with
        N. Per sample it falls as 1/N, and the layer&rsquo;s ratio of operations to bytes rises. The LeNet explorer above
        shows it: at batch 1, C5&rsquo;s weights are 99% of its traffic; at 256, about a quarter. It is the same reason
        batching raises a linear layer&rsquo;s ratio in section 5.1.
      </p>
    ),
    source: 'ch. 19 §19.1, p. 459; ch. 5 §5.1 (the linear-layer explorer)',
  },
};

// ---------------------------------------------------------------- one output, term by term

const cellKeys = (fn: () => void) => (e: KeyboardEvent) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fn();
  }
};

function OneOutput() {
  const [sel, setSel] = useState({ m: 0, h: 0, w: 0 });
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [flip, setFlip] = useState(false);

  const Ho = hOut(S);
  const Wo = wOut(S);
  const tr = useMemo(() => terms(S, BOOK_X, BOOK_F, sel.m, sel.h, sel.w, flip), [sel, flip]);
  const Y = useMemo(() => forward(S, BOOK_X, BOOK_F, flip), [flip]);
  const Yplain = useMemo(() => forward(S, BOOK_X, BOOK_F), []);
  const changed = useMemo(() => new Set(changedByFlip(S, BOOK_X, BOOK_F)), []);
  const parts = perInputMap(S, BOOK_X, BOOK_F, sel.m, sel.h, sel.w, flip);
  const total = tr.length;
  const at = Math.min(k, total);
  const cur = at > 0 ? tr[at - 1] : undefined;
  const sum = cur ? cur.sum : 0;

  useEffect(() => setK(0), [sel, flip]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setK((x) => x + 1), 520);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  // Patch cells of the selected output, and the filters it uses.
  const inPatch = (i: number) => {
    const h = Math.floor(i / S.W);
    const w = i % S.W;
    return h >= sel.h && h < sel.h + S.K && w >= sel.w && w < sel.w + S.K;
  };
  // A filter as displayed: rotated when the switch is on, so position (p, q) on screen is what term (p, q) uses.
  const shown = (m: number, c: number) =>
    Array.from({ length: S.K * S.K }, (_, i) => {
      const p = Math.floor(i / S.K);
      const q = i % S.K;
      return flip ? BOOK_F[fAt(S, m, c, S.K - 1 - p, S.K - 1 - q)] : BOOK_F[fAt(S, m, c, p, q)];
    });
  const subtotals = parts.map((v, c) => (cur && cur.c > c ? v : cur && cur.c === c ? tr.filter((t) => t.c === c && t.k < at).reduce((a, t) => a + t.product, 0) : null));

  const say = cur
    ? `c = ${cur.c}, p = ${cur.p}, q = ${cur.q}: X[${cur.c}, ${sel.h + cur.p}, ${sel.w + cur.q}] × F[${sel.m}, ${cur.c}, ${flip ? `${S.K - 1 - cur.p}, ${S.K - 1 - cur.q}` : `${cur.p}, ${cur.q}`}] = ${cur.x} × ${cur.f} = ${cur.product}. Running sum ${cur.sum}.` +
      (at === total ? ` That is Y[${sel.m}, ${sel.h}, ${sel.w}] = ${parts.join(' + ')} = ${sum}.` : '')
    : `Y[${sel.m}, ${sel.h}, ${sel.w}] adds ${total} products: ${S.K * S.K} from each of the ${S.C} input maps, each through the filter F[${sel.m}, c]. Press Step.`;

  return (
    <Bench
      title="One output, term by term"
      task="Click any output, then step through the loops over c, p and q."
      foot={
        <>
          Every cell here is in global memory; the running sum is the one value a thread would keep in a register. The
          numbers are the book&rsquo;s (Fig. 19.2), computed by the same loop nest as Fig. 19.3.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setK(0); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setK(at - 1); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setK(at + 1); }} disabled={at >= total}>Step</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => { setPlaying(false); setK(total); }} disabled={at >= total}>Finish</button>
        <span className="muted" aria-live="polite">term {at} of {total}</span>
        <label className="check">
          <input type="checkbox" checked={flip} onChange={(e) => setFlip(e.target.checked)} /> rotate filters 180° (true
          convolution; not in the book)
        </label>
      </div>
      <div className="narrate" role="status" data-out="term" data-k={at} data-sum={sum} style={{ marginBottom: '0.9rem' }}>
        {say}
      </div>
      <div className="row" style={{ alignItems: 'flex-start', gap: '1.6rem' }}>
        <div>
          <div className="mx-label t-global">Input maps X (C = 3, each 3 × 3)</div>
          <div className="row" style={{ gap: '0.7rem', alignItems: 'flex-start' }}>
            {Array.from({ length: S.C }, (_, c) => (
              <div key={c}>
                <div className="muted" style={{ fontSize: '0.74rem' }}>X{c}{subtotals[c] != null ? <> · gives <b data-part={c}>{subtotals[c]}</b></> : null}</div>
                <div className="mx global" data-grid={`X${c}`} style={{ gridTemplateColumns: `repeat(${S.W}, auto)` }}>
                  {Array.from({ length: S.H * S.W }, (_, i) => {
                    const xi = c * S.H * S.W + i;
                    const hot = cur?.xi === xi;
                    return <span key={i} className={hot ? 'hot' : inPatch(i) ? 'on' : ''}>{BOOK_X[xi]}</span>;
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="mx-label t-global">Filters F[m, c] ({flip ? 'rotated 180°' : 'as stored'})</div>
          {Array.from({ length: S.M }, (_, m) => (
            <div key={m} className="row" style={{ gap: '0.7rem', marginBottom: '0.4rem', alignItems: 'flex-end' }}>
              <span className="muted" style={{ fontSize: '0.74rem', width: '3.6rem' }}>for Y{m}</span>
              {Array.from({ length: S.C }, (_, c) => (
                <div key={c} className="mx global" data-grid={`F${m}${c}`} style={{ gridTemplateColumns: `repeat(${S.K}, auto)` }}>
                  {shown(m, c).map((v, i) => {
                    const hot = cur && m === sel.m && c === cur.c && i === cur.p * S.K + cur.q;
                    return <span key={i} className={hot ? 'hot' : m === sel.m ? 'on' : ''}>{v}</span>;
                  })}
                </div>
              ))}
            </div>
          ))}
        </div>
        <div>
          <div className="mx-label t-global">Output maps Y (click one)</div>
          <div className="row" style={{ gap: '0.7rem', alignItems: 'flex-start' }}>
            {Array.from({ length: S.M }, (_, m) => (
              <div key={m}>
                <div className="muted" style={{ fontSize: '0.74rem' }}>Y{m}</div>
                <div className="mx global" data-grid={`Y${m}`} style={{ gridTemplateColumns: `repeat(${Wo}, auto)` }}>
                  {Array.from({ length: Ho * Wo }, (_, i) => {
                    const h = Math.floor(i / Wo);
                    const w = i % Wo;
                    const yi = m * Ho * Wo + i;
                    const isSel = sel.m === m && sel.h === h && sel.w === w;
                    const moved = flip && changed.has(yi);
                    const pick = () => setSel({ m, h, w });
                    return (
                      <span
                        key={i}
                        role="button"
                        tabIndex={0}
                        aria-pressed={isSel}
                        aria-label={`Y[${m}, ${h}, ${w}]`}
                        data-y={`${m},${h},${w}`}
                        data-value={Y[yi]}
                        data-changed={moved ? 'yes' : 'no'}
                        className={isSel ? 'hot' : ''}
                        title={moved ? `the book's formula gives ${Yplain[yi]}` : undefined}
                        style={{ cursor: 'pointer', outline: moved ? '2px solid var(--ink)' : undefined, outlineOffset: -3 }}
                        onClick={pick}
                        onKeyDown={cellKeys(pick)}
                      >
                        {Y[yi]}
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
          <p style={{ margin: '0.8rem 0 0' }}>
            <span className="muted">register sum </span>
            <b className="t-reg" style={{ fontSize: '1.3rem' }}>{sum}</b>
          </p>
          {flip ? (
            <p className="muted" style={{ margin: '0.4rem 0 0', maxWidth: '12rem' }} data-out="changes">
              Outlined: {changed.size} of {Y.length} outputs differ from the book&rsquo;s (
              {[...changed].map((i) => `${Yplain[i]} → ${Y[i]}`).join(', ')}).
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- LeNet-5

const STAGE_TEXT: Record<string, string> = {
  INPUT: 'A 32 × 32 grey image: one input map.',
  S2: 'Subsampling: each output pixel summarises a 2 × 2 neighbourhood of one map, so the maps halve in size. Named in the book, described in Appendix B; not computed here.',
  S4: 'Subsampling again: sixteen 10 × 10 maps become sixteen 5 × 5 maps. Not computed here.',
  F6: 'Fully connected: 84 outputs, each reading all 120 values from C5 (p. 458). Not computed here.',
  OUTPUT: 'Gaussian connections producing 10 values, one per digit class (p. 459). Not computed here.',
};

function Bar({ t }: { t: ReturnType<typeof batchTraffic> }) {
  const seg = (v: number) => `${(100 * v) / t.total}%`;
  return (
    <div style={{ display: 'flex', height: '0.8rem', width: '100%', minWidth: '6rem', border: '1px solid var(--global)', borderRadius: 2, overflow: 'hidden' }} aria-hidden>
      <div style={{ width: seg(t.xBytes), background: 'var(--global-wash)' }} />
      <div style={{ width: seg(t.fBytes), background: 'var(--global)' }} />
      <div style={{ width: seg(t.yBytes), background: 'var(--global-wash)', borderLeft: '1px solid var(--global)' }} />
    </div>
  );
}

function LeNet() {
  const [id, setId] = useState('C1');
  const [lgN, setLgN] = useState(0);
  const N = 2 ** lgN;
  const stage = LENET.find((s) => s.id === id) as Stage;
  const conv = LENET.filter((s) => s.layer) as (Stage & { layer: LayerShape })[];

  return (
    <Bench
      title="LeNet-5's convolution layers"
      task="Compare C1, C3 and C5, then move the batch size and watch F's share of the traffic."
      foot={
        <>
          Bar: X, <b className="t-global">F</b>, Y, in bytes. Ideal traffic: each element of X, F and Y is moved exactly
          once for the whole batch, as 4-byte floats. Real kernels move more; section 19.2 counts what the direct kernel
          asks for.
        </>
      }
    >
      <div className="seg" role="group" aria-label="LeNet-5 stages" style={{ flexWrap: 'wrap', rowGap: 4, marginBottom: '0.9rem' }}>
        {LENET.map((s) => (
          <button key={s.id} aria-pressed={s.id === id} onClick={() => setId(s.id)}>
            {s.id}
            <span style={{ fontWeight: 400, opacity: 0.72 }}> {s.out}</span>
          </button>
        ))}
      </div>
      <div className="cols">
        <div className="panel" data-out="stage" data-id={id}>
          {stage.layer ? (
            (() => {
              const l = stage.layer;
              const sh = shapes(l, N);
              const t = batchTraffic(l, N);
              return (
                <>
                  <p style={{ margin: '0 0 0.5rem' }}>
                    <b>{id}</b>: convolution with {l.K} × {l.K} filters, no ghost cells.
                  </p>
                  <table className="data" style={{ margin: 0 }}>
                    <tbody>
                      <tr><td>X <span className="muted">(N × C × H × W)</span></td><td data-out="X">{dims(sh.X)}</td></tr>
                      <tr><td>F <span className="muted">(M × C × K × K)</span></td><td data-out="F">{dims(sh.F)}</td></tr>
                      <tr><td>Y <span className="muted">(N × M × H_out × W_out)</span></td><td data-out="Y">{dims(sh.Y)}</td></tr>
                      <tr><td>filters</td><td className="num" data-out="filters">{big(sh.filters)}</td></tr>
                      <tr><td>weights</td><td className="num" data-out="weights">{big(sh.weights)}</td></tr>
                      <tr><td>multiply-adds per sample</td><td className="num" data-out="macs">{big(sh.macsPerSample)}</td></tr>
                    </tbody>
                  </table>
                  <label className="field" style={{ marginTop: '0.8rem', width: '100%' }}>
                    Batch size N = {N}
                    <input type="range" min="0" max="8" value={lgN} aria-label="batch size" onChange={(e) => setLgN(+e.target.value)} />
                  </label>
                  <table className="data" style={{ margin: '0.4rem 0 0' }}>
                    <tbody>
                      <tr><td>multiply-adds for the batch</td><td className="num" data-out="batch-macs">{big(t.macs)}</td></tr>
                      <tr><td>ideal bytes for the batch</td><td className="num" data-out="total">{bytes(t.total)}</td></tr>
                      <tr><td>of which F</td><td className="num" data-out="fshare">{pct(t.fShare)}</td></tr>
                      <tr><td>F bytes per sample</td><td className="num" data-out="fpersample">{big(Math.round(t.fBytesPerSample * 100) / 100)}</td></tr>
                    </tbody>
                  </table>
                </>
              );
            })()
          ) : (
            <p style={{ margin: 0 }} data-out="not-computed">
              <b>{id}</b> ({stage.out}). {STAGE_TEXT[id]}
            </p>
          )}
        </div>
        <div className="panel">
          <p style={{ margin: '0 0 0.5rem' }}>The three convolution layers at batch size {N}</p>
          <table className="data" style={{ margin: 0, width: '100%' }} aria-label="LeNet-5 convolution layers compared">
            <thead>
              <tr>
                <th>layer</th>
                <th className="num">C</th>
                <th className="num">map</th>
                <th className="num">M</th>
                <th className="num nowrap">F share</th>
                <th style={{ width: '30%' }}>traffic</th>
              </tr>
            </thead>
            <tbody>
              {conv.map((s) => {
                const t = batchTraffic(s.layer, N);
                return (
                  <tr key={s.id} data-row={s.id} data-fshare={t.fShare} style={s.id === id ? { background: 'var(--rule-soft)' } : undefined}>
                    <td>{s.id}</td>
                    <td className="num">{s.layer.C}</td>
                    <td className="num nowrap">{s.layer.H} × {s.layer.W}</td>
                    <td className="num">{s.layer.M}</td>
                    <td className="num">{pct(t.fShare)}</td>
                    <td><Bar t={t} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the sequential reference

function Reference() {
  const text = useLabFile(LAB_MAIN);
  if (text == null) return null;
  const region = lessonRegion(text, 'cpu-reference');
  return (
    <Code
      tabs={[{ label: 'CPU reference', code: region.code, tiers: {} }]}
      caption={
        <>
          {region.found ? 'The CPU reference' : 'Lab 19.1’s driver (the reference region was not found, so the whole file is shown)'}, loaded
          from <code>labs/{LAB_MAIN}</code>. Every run of lab 19.1 checks your kernels against it with a tolerance.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- the page

export function S191() {
  return (
    <>
      <p>
        Convolutional networks are older than the GPUs that made them practical. In the book&rsquo;s short history they
        were invented in the late 1980s, and feed-forward networks came back around 2006.
      </p>
      <Source section="19.1" pages="453–455" />
      <p>
        In 2012 a University of Toronto entry won the ImageNet challenge with a network of about 60 million parameters,
        trained on two GPUs with a CUDA library called cuda-convnet; its error rate was 15.3% against 26.2% for the
        runner-up. The book dates deep learning&rsquo;s take-off in industry to that year.
      </p>
      <Today checked="2026-11-20 (ch19-facts A9)">
        <p>
          The NIPS 2012 paper says five to six days of training, not a week, and credits the 15.3% to &ldquo;a variant of
          this model&rdquo;.
        </p>
      </Today>
      <p>
        The chapter&rsquo;s running example is LeNet-5, a small digit classifier with three kinds of layer:
        convolution, subsampling and fully connected. The book&rsquo;s claim is that convolution layers take most of a
        CNN&rsquo;s time, so the chapter covers only them, and only their forward path. That path is the one to speed up
        first: inference runs only the forward path, and training runs it too.
      </p>
      <Source section="19.1" pages="455" figs="Fig. 19.1" />

      <h2>A layer is a bank of 2D convolutions</h2>
      <p>
        A layer&rsquo;s inputs and outputs are stacks of 2D images called feature maps. A filter is chapter 7&rsquo;s
        mask: a small K × K array of weights. What is new is the bookkeeping. Every output map draws on every input
        map, each through its own filter, so a layer with C input maps and M output maps has M · C filters. Output map m
        is the sum of C ordinary 2D convolutions, one per input map c, each with the filter for the pair (m, c).
      </p>
      <Source section="19.1" pages="456" figs="Fig. 19.2(a)" />
      <p>
        The book leaves out the activation function. It can be applied afterwards, as a separate kernel or at the end of
        the same one, and it changes nothing about how the sums are computed.
      </p>
      <Source section="19.1" pages="456" />

      <h2>One output by hand</h2>
      <p>
        Fig. 19.2 makes this concrete: three 3 × 3 input maps, two output maps, and so six 2 × 2 filters. Pixel (0, 0)
        of output map 0 takes the top-left 2 × 2 patch of each input map, multiplies it element by element with that
        map&rsquo;s filter for output 0, and adds everything up: 7 from map 0, 5 from map 1, 2 from map 2, so 14. The
        book uses small integers so you can check by hand; a real layer works in floating point.
      </p>
      <Source section="19.1" pages="457" figs="Eq. 19.1" />
      <Predict
        id="ch19/19.1/p-pixel"
        prompt={
          <>
            Before using the explorer, compute pixel (0, 1) of output map 0 by hand. Every patch moves one column to the
            right; the filters stay the same.
          </>
        }
      >
        {(v) => {
          const parts = perInputMap(S, BOOK_X, BOOK_F, 0, 0, 1);
          const y = parts.reduce((a, b) => a + b, 0);
          return (
            <p>
              {parts.join(' + ')} = <b>{y}</b>.{' '}
              {v === y
                ? 'That is what you had.'
                : `Click Y0's top-right cell in the explorer and step through the twelve products to find where yours went another way. The usual slip is to move the filter instead of the patch.`}
            </p>
          );
        }}
      </Predict>
      <p>
        The explorer draws Fig. 19.2. Click any of the eight outputs to light up its three patches and three filters,
        then step through the twelve multiply-adds in the order of Fig. 19.3&rsquo;s loops over c, p and q.
      </p>
      <Simplification>
        <p>
          Integers, as in the book, and no activation. The order of the steps is Fig. 19.3&rsquo;s loop order; nothing
          here says how a GPU schedules them.
        </p>
      </Simplification>
      <OneOutput />
      <p>
        Two things to take from it. Each output map sums over every input map, so one input pixel influences every
        output map. And the formula slides each filter over the input as it is stored, without flipping it.
      </p>

      <h2>No ghost cells</h2>
      <p>
        Chapter 7 gave every input pixel an output by pretending the image is surrounded by{' '}
        <a href={href('ch07', '7.1')}>ghost cells</a> of zeros. LeNet does not. An output exists only where the whole K
        × K patch fits inside the input, so each side loses K − 1 pixels: with 5 × 5 filters, a 32 × 32 image gives 28
        × 28. In code that is <code>H_out = H - K + 1</code> and <code>W_out = W - K + 1</code>.
      </p>
      <Source section="19.1" pages="457" />
      <p>
        This is <code>nn.Conv2d</code> with its default <code>padding=0</code>, exactly. PyTorch also documents{' '}
        <code>Conv2d</code> as cross-correlation, which is the book&rsquo;s formula: no flip. So the book&rsquo;s layer
        and PyTorch&rsquo;s give the same numbers for the same weights.
      </p>
      <Today checked="2026-11-20 (notes/today/ch07-facts.md, D1)">
        <p>
          PyTorch 2.15 calls <code>Conv2d</code> &ldquo;the valid 2D cross-correlation operator&rdquo;, default{' '}
          <code>padding=0</code>. cuDNN offers both modes and PyTorch asks for cross-correlation; MIOpen computes cross-correlation
          (ch07-facts D1, ch19-facts B3); Apple&rsquo;s MPSGraph documents neither (C3).
        </p>
      </Today>
      <Ask q={questionsS191['ch19/19.1/q-outsize']} label="Work it out" />
      <Ask q={questionsS191['ch19/19.1/q-flip']} label="Work it out" />

      <h2>Three tensors and their shapes</h2>
      <p>
        For one sample, the layer reads X, of shape C × H × W, and F, of shape M × C × K × K, and writes Y, of shape M ×
        H_out × W_out.
      </p>
      <Source section="19.1" pages="458" />
      <p>
        All three are flat arrays in row-major order with the highest dimension first. So <code>F[m, c, _, _]</code> is
        the K × K filter that carries input map c into output map m, and the filters for one output map sit next to each
        other in memory.
      </p>
      <Aside>
        <p>
          Two slips on p. 458: the text names the filter array <code>W</code>, the code <code>F</code>; and no exercise
          takes up the fully connected layer it leaves &ldquo;as an exercise&rdquo;.
        </p>
      </Aside>
      <p>
        PyTorch stores <code>nn.Conv2d.weight</code> with shape (out_channels, in_channels, kH, kW): the book&rsquo;s F,
        element for element. Section 19.3 makes use of that.
      </p>
      <Today checked="2026-11-20 (notes/today/ch19-facts.md, D4)">
        <p>
          PyTorch 2.15 documents the weight as &ldquo;(out_channels, in_channels/groups, kernel_size[0],
          kernel_size[1])&rdquo;; with one group that is M × C × K × K.
        </p>
      </Today>
      <p>
        In LeNet, C1 turns one 32 × 32 image into six 28 × 28 maps. C3 turns six 14 × 14 maps into sixteen 10 × 10 maps
        and needs 96 filters. C5 turns sixteen 5 × 5 maps into 120 maps of one pixel each, with 1920 filters: a 5 × 5
        filter on a 5 × 5 map has exactly one place to sit.
      </p>
      <Source section="19.1" pages="458" figs="Fig. 19.1" />
      <Simplification>
        <p>
          Shapes and counts only. Traffic is the ideal: each element of X, F and Y moved once per batch as a 4-byte
          float, with no caches and no particular kernel.
        </p>
      </Simplification>
      <LeNet />
      <p>
        Read the comparison table from top to bottom. Early layers have few channels and large maps; late layers have
        many channels and small maps. Section 19.3 comes back to this when it asks whether the matrices of a convolution
        stay large enough to use a GPU well in every layer.
      </p>
      <Ask q={questionsS191['ch19/19.1/q-weights']} label="Work it out" />

      <h2>A batch of samples</h2>
      <p>
        Running several inputs through the layer together gives a batched convolution layer; the input and output maps
        that belong to one input are a <b>sample</b>. X and Y gain a leading dimension N. F does not: the same weights
        serve every sample. The book&rsquo;s reason to batch is a bigger grid, with more thread blocks to keep the GPU
        busy; it is especially common in training.
      </p>
      <Source section="19.1" pages="459" figs="Fig. 19.4" />
      <p>
        There is a second reason, and you have met it before. Move the batch slider in the LeNet explorer and watch
        F&rsquo;s share of the bytes: a weight fetched once can serve every sample. In section 5.1&rsquo;s{' '}
        <a href={href('ch05', '5.1')}>linear-layer explorer</a> the same effect raised a linear layer&rsquo;s ratio of
        operations to bytes. C5 is close kin: at batch 1 its 48,000 weights are almost all of its traffic, each used
        once, and Fig. 19.1 even labels that stage a full connection.
      </p>
      <Beyond>
        <p>
          The traffic counts in the explorer are ours, from the shapes on p. 458 and the ideal-traffic method of section
          5.1. The book gives no numbers for batching.
        </p>
      </Beyond>
      <Ask q={questionsS191['ch19/19.1/q-batch']} />

      <h2>The sequential reference</h2>
      <p>
        Fig. 19.4 is the whole batched layer as seven nested loops: n, m, h and w pick an output element, and c, p and q
        add up its products. Lab 19.1&rsquo;s driver keeps a C++ version of it as the reference that every kernel you
        write in this chapter is checked against.
      </p>
      <Source section="19.1" pages="457, 459" figs="Figs. 19.3, 19.4" />
      <Reference />
      <p>Where it differs from the page:</p>
      <ul>
        <li>
          The book writes <code>X[n, c, h + p, w + q]</code> as multi-dimensional pseudo-code; the reference computes
          the flat row-major index itself, as chapter 3 taught.
        </li>
        <li>It computes H_out and W_out from H, W and K instead of assuming them.</li>
        <li>
          It accumulates in <code>double</code>, so that the reference is more accurate than the float kernels it
          judges. The test still compares with a tolerance.
        </li>
        <li>
          It also records, for each output, the sum of the products&rsquo; absolute values; the tests scale their
          tolerance by it, so a large output is allowed a proportionally larger rounding error.
        </li>
        <li>The loops stay in the book&rsquo;s order, so you can read it line for line against Fig. 19.4.</li>
      </ul>
    </>
  );
}
