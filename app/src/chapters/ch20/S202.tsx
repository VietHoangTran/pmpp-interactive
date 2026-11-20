import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { EXP_LIMIT_F32, attention, differing, dotTerms, makeAttnProblem, visible } from '../../sim/attnref';

export const questionsS202: Record<string, Question> = {
  'ch20/20.2/q-s-bytes': {
    kind: 'numeric',
    id: 'ch20/20.2/q-s-bytes',
    prompt: <>One head, N = 4096 tokens, head width d = 128, fp32. How many MiB does S take?</>,
    answer: 64,
    unit: 'MiB',
    explain: (
      <p>
        4096 × 4096 × 4 B = 64 MiB. Q is 4096 × 128 × 4 B = 2 MiB, so S is N / d = 32 times the size of the matrix it
        came from, and it grows with the square of the context while Q grows linearly.
      </p>
    ),
    source: 'ch. 20 pp. 483–484 (shapes); our arithmetic',
  },
  'ch20/20.2/q-row-nonzeros': {
    kind: 'numeric',
    id: 'ch20/20.2/q-row-nonzeros',
    prompt: <>N = 8, causal mask. How many non-zero entries does row 5 of P have (rows and columns counted from 0)?</>,
    answer: 6,
    explain: (
      <p>
        Columns 0 to 5: token 5 attends to every earlier token <em>and to itself</em>. The book&rsquo;s words are
        &ldquo;previously generated tokens&rdquo;; the mask keeps the diagonal.
      </p>
    ),
    source: 'ch. 20 p. 484',
  },
  'ch20/20.2/q-no-scale': {
    kind: 'choice',
    id: 'ch20/20.2/q-no-scale',
    prompt: <>You forget the 1/√d with d = 128. What happens to each row of P?</>,
    options: [
      { label: 'Logits grow about 11.3 times; each row gets close to one-hot.', correct: true },
      { label: 'Nothing: softmax does not care about the scale of its inputs.', why: 'Softmax ignores a constant added to every logit of a row, not a factor multiplying them.' },
      { label: 'The rows no longer sum to 1, so O is scaled up by about 11.3.', why: 'Any softmax row sums to 1; the scale changes how the mass is shared, not how much there is.' },
      { label: 'Masked entries become non-zero, because the mask is scaled too.', why: 'Minus infinity times any positive factor is still minus infinity.' },
    ],
    explain: (
      <p>
        √128 ≈ 11.3, so every logit grows by that factor. Differences between logits grow with them and the largest
        dominates: the row sharpens towards one-hot. The book&rsquo;s reason for the factor is training: saturated
        softmax rows give vanishing gradients. Untick &ldquo;scale by 1/√d&rdquo; above to watch it at d = 6.
      </p>
    ),
    source: 'ch. 20 p. 484',
  },
  'ch20/20.2/q-shift': {
    kind: 'choice',
    id: 'ch20/20.2/q-shift',
    prompt: (
      <>
        Why does subtracting each row&rsquo;s max leave P unchanged, while subtracting a different constant from each{' '}
        <em>column</em> would not?
      </>
    ),
    options: [
      { label: 'A row\'s shift scales its numerators and sum alike; a column shift does not.', correct: true },
      { label: 'Because the row max is the only value that keeps every exponent finite in float32.', why: 'Any constant at least the max would also keep them finite; finiteness is not why P is unchanged.' },
      { label: 'Because softmax is applied to columns, so only a column shift reaches it.', why: 'Softmax normalises each row of S + M (Eq. 20.2): a row sums to 1.' },
      { label: 'It does change P slightly, by the float rounding of the subtraction.', why: 'Rounding is a float32 detail; the question is about the algebra, where the change is exactly none.' },
    ],
    explain: (
      <p>
        Subtracting m<sub>r</sub> from every logit of row r multiplies each numerator e<sup>l</sup> and their sum by the
        same factor e<sup>−m<sub>r</sub></sup>, which cancels. A shift that differs between columns multiplies the terms
        of one row by different factors, which does not cancel. In the viewer, offset every logit by +90: P does not
        move (it is one shift per row, the same for all).
      </p>
    ),
    source: 'ch. 20 Eq. 20.2, p. 484',
  },
  'ch20/20.2/q-rows-x': {
    kind: 'numeric',
    id: 'ch20/20.2/q-rows-x',
    prompt: (
      <>
        The book&rsquo;s toy example: whole words as tokens, no system prompt, the prompt &ldquo;The title of this book
        is Programming&rdquo;. In this baseline, how many rows does X have in the generation iteration that emits
        &ldquo;Processors&rdquo;?
      </>
    ),
    answer: 9,
    unit: 'rows',
    explain: (
      <p>
        Seven prompt tokens, plus &ldquo;Massively&rdquo; (emitted by the summarization phase) and &ldquo;Parallel&rdquo;
        (generation iteration 1): 9 rows in iteration 2, which emits &ldquo;Processors&rdquo;. The baseline recomputes
        every row of K, V, Q, S, P and O in every layer, every iteration. Section{' '}
        <a href={href('ch20', '20.4')}>20.4</a> shows that only the last row is new.
      </p>
    ),
    source: 'ch. 20 pp. 485–486, Fig. 20.3',
  },
};

// ---------------------------------------------------------------- one head, every matrix
const HOT: CSSProperties = { background: 'var(--ink)', color: 'var(--page)', borderColor: 'var(--ink)', fontWeight: 650 };
const ON: CSSProperties = { background: 'var(--rule-soft)', borderColor: 'var(--ink-2)' };
const num = (x: number, digits = 2) => {
  if (Number.isNaN(x)) return 'NaN';
  if (x === Infinity) return '∞';
  if (x === -Infinity) return '−∞';
  const s = Number(x.toFixed(digits)).toString();
  return s.replace('-', '−');
};

function Grid({
  label,
  rows,
  cols,
  value,
  style,
  cls,
  onPick,
  name,
}: {
  label: string;
  rows: number;
  cols: number;
  value: (r: number, c: number) => string;
  style?: (r: number, c: number) => CSSProperties | undefined;
  cls?: (r: number, c: number) => string;
  onPick?: (r: number, c: number) => void;
  name: string;
}) {
  return (
    <div>
      <div className="mx-label">{label}</div>
      <div className="mx" data-grid={name} style={{ gridTemplateColumns: `repeat(${cols}, auto)` }}>
        {Array.from({ length: rows * cols }, (_, k) => {
          const r = Math.floor(k / cols);
          const c = k % cols;
          return (
            <span
              key={k}
              data-r={r}
              data-c={c}
              className={cls?.(r, c) ?? ''}
              style={{ ...style?.(r, c), cursor: onPick ? 'pointer' : undefined }}
              onClick={onPick ? () => onPick(r, c) : undefined}
              onMouseEnter={onPick ? () => onPick(r, c) : undefined}
              role={onPick ? 'button' : undefined}
              tabIndex={onPick ? 0 : undefined}
              onKeyDown={onPick ? (e) => (e.key === 'Enter' || e.key === ' ') && onPick(r, c) : undefined}
            >
              {value(r, c)}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function Viewer() {
  const [n, setN] = useState(6);
  const [d, setD] = useState(4);
  const [seed, setSeed] = useState(1);
  const [scale, setScale] = useState(true);
  const [mask, setMask] = useState(true);
  const [subtract, setSubtract] = useState(true);
  const [offset, setOffset] = useState(0);
  const [pick, setPick] = useState<[number, number]>([3, 1]);
  const prob = useMemo(() => makeAttnProblem(n, d, seed), [n, d, seed]);
  const opts = { scale, mask, subtractMax: subtract, offset, precision: 'f32' as const };
  const res = useMemo(() => attention(prob, opts), [prob, scale, mask, subtract, offset]); // eslint-disable-line react-hooks/exhaustive-deps
  const causal = useMemo(() => attention(prob, { ...opts, mask: true }), [prob, scale, subtract, offset]); // eslint-disable-line react-hooks/exhaustive-deps
  const leaks = mask ? [] : differing(res.O, causal.O, 1e-6, true);
  const leakRows = new Set(leaks.map(([r]) => r));
  const [pi, pj] = [Math.min(pick[0], n - 1), Math.min(pick[1], n - 1)];
  const vis = !mask || visible(pi, pj, n, n);
  const dot = dotTerms(prob, pi, pj, 'f32');
  const nanCount = res.P.flat().filter(Number.isNaN).length;
  const live = res.live[pi];
  const rowNaN = res.P[pi].some(Number.isNaN);

  const setSize = (nn: number, dd: number) => {
    setN(nn);
    setD(dd);
    setPick(([r, c]) => [Math.min(r, nn - 1), Math.min(c, nn - 1)]);
  };

  return (
    <Bench
      title="One head, every matrix"
      task="Point at a cell of P. Then switch the mask, the scale and the max subtraction off, one at a time."
      foot={
        <>
          Q and K hold values from {'{−1, −0.5, 0, 0.5, 1}'} and V integers from −3 to 3, so you can check a dot product
          in your head. All arithmetic is float32. Hover or tap a cell of S or P.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field" style={{ minWidth: '8rem' }}>
          tokens N: <b>{n}</b>
          <input type="range" min={2} max={8} value={n} onChange={(e) => setSize(+e.target.value, d)} aria-label="N" />
        </label>
        <label className="field" style={{ minWidth: '8rem' }}>
          head width d: <b>{d}</b>
          <input type="range" min={2} max={6} value={d} onChange={(e) => setSize(n, +e.target.value)} aria-label="d" />
        </label>
        <button className="quiet" onClick={() => setSeed((s) => s + 1)}>
          New numbers
        </button>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="check">
          <input type="checkbox" checked={scale} onChange={(e) => setScale(e.target.checked)} /> scale by 1/√d
        </label>
        <label className="check">
          <input type="checkbox" checked={mask} onChange={(e) => setMask(e.target.checked)} /> causal mask
        </label>
        <label className="check">
          <input type="checkbox" checked={subtract} onChange={(e) => setSubtract(e.target.checked)} /> subtract the row max
        </label>
        <button aria-pressed={offset === 90} onClick={() => setOffset(offset === 90 ? 0 : 90)}>
          {offset === 90 ? 'Logits offset by +90' : 'Offset every logit by +90'}
        </button>
      </div>

      <div className="narrate" role="status" data-narrate style={{ marginBottom: '0.9rem' }}>
        {vis ? (
          <>
            P[{pi}][{pj}]: q<sub>{pi}</sub> · k<sub>{pj}</sub> ={' '}
            {dot.terms.map((t, k) => (
              <span key={k}>
                {k ? ' + ' : ''}({num(t.q, 1)})({num(t.k, 1)})
              </span>
            ))}{' '}
            = {num(dot.value, 2)}
            {scale ? <>; × 1/√{d} = {num(res.S[pi][pj] * res.scale, 3)}</> : null}
            {offset ? <>; + 90 = {num(res.logits[pi][pj], 3)}</> : null}. Row {pi}&rsquo;s max is {num(res.m[pi], 3)}
            {subtract ? (
              <>
                , so the numerator is exp({num(res.logits[pi][pj], 3)} − {num(res.m[pi], 3)}) = {num(res.E[pi][pj], 4)}
              </>
            ) : (
              <>
                , not subtracted, so the numerator is exp({num(res.logits[pi][pj], 3)}) = {num(res.E[pi][pj], 4)}
                {res.E[pi][pj] === Infinity ? <> (beyond float32 above {num(EXP_LIMIT_F32, 2)})</> : null}
              </>
            )}
            ; the row&rsquo;s sum is {num(res.sum[pi], 4)}, and p = <b data-p>{num(res.P[pi][pj], 4)}</b>.
          </>
        ) : (
          <>
            P[{pi}][{pj}] is masked: column {pj} is a token after token {pi}, M adds −∞ to its logit, and exp(−∞) is exactly{' '}
            <b data-p>0</b>. Row {pi} has {live} non-zero entries.
          </>
        )}
        {nanCount ? (
          <>
            <br />
            <b className="t-fail" data-nan={nanCount}>
              {nanCount} entries of P are NaN: their numerators overflowed to infinity, and so did the sums.
            </b>
          </>
        ) : null}
        {leakRows.size ? (
          <>
            <br />
            <b className="t-fail" data-leak={[...leakRows].join(',')}>
              Without the mask, rows {[...leakRows].join(', ')} of O mix in values of later tokens: the future leaks into the
              past.
            </b>
          </>
        ) : null}
      </div>

      <div className="row" style={{ alignItems: 'flex-start', gap: '1.2rem', marginBottom: '1rem' }}>
        <Grid
          name="Q"
          label={`Q (${n} × ${d})`}
          rows={n}
          cols={d}
          value={(r, c) => num(prob.Q[r][c], 1)}
          style={(r) => (r === pi ? ON : undefined)}
        />
        <Grid
          name="KT"
          label={`K^T (${d} × ${n})`}
          rows={d}
          cols={n}
          value={(r, c) => num(prob.K[c][r], 1)}
          style={(_, c) => (c === pj ? ON : undefined)}
        />
      </div>
      <div className="row" style={{ alignItems: 'flex-start', gap: '1.2rem', marginBottom: '1rem' }}>
        <Grid
          name="S"
          label={scale ? 'S / √d' : 'S (unscaled)'}
          rows={n}
          cols={n}
          value={(r, c) => num(res.S[r][c] * res.scale, 2)}
          cls={(r, c) => (mask && !visible(r, c, n, n) ? 'void' : '')}
          style={(r, c) => (r === pi && c === pj ? HOT : undefined)}
          onPick={(r, c) => setPick([r, c])}
        />
        <Grid
          name="M"
          label="M"
          rows={n}
          cols={n}
          value={(r, c) => (mask && !visible(r, c, n, n) ? '−∞' : '0')}
          cls={(r, c) => (mask && !visible(r, c, n, n) ? 'void' : '')}
        />
      </div>
      <div className="row" style={{ alignItems: 'flex-start', gap: '1.2rem' }}>
        <Grid
          name="P"
          label="P = softmax(S/√d + M)"
          rows={n}
          cols={n}
          value={(r, c) => num(res.P[r][c], 2)}
          cls={(r, c) => (Number.isNaN(res.P[r][c]) ? 'bad' : res.P[r][c] === 0 ? 'void' : '')}
          style={(r, c) => (r === pi && c === pj ? HOT : r === pi ? ON : undefined)}
          onPick={(r, c) => setPick([r, c])}
        />
        <Grid
          name="V"
          label={`V (${n} × ${d})`}
          rows={n}
          cols={d}
          value={(r, c) => num(prob.V[r][c], 0)}
          style={(r) => (res.P[pi][r] !== 0 ? ON : undefined)}
        />
        <Grid
          name="O"
          label={`O = P V (${n} × ${d})`}
          rows={n}
          cols={d}
          value={(r, c) => num(res.O[r][c], 2)}
          cls={(r, c) => (Number.isNaN(res.O[r][c]) || leaks.some(([a, b]) => a === r && b === c) ? 'bad' : '')}
          style={(r) => (r === pi ? ON : undefined)}
        />
      </div>
      <p className="muted" style={{ margin: '0.8rem 0 0' }} data-rowinfo>
        {rowNaN
          ? `Row ${pi} of P has no usable entries: it overflowed. Row ${pi} of O is NaN with it.`
          : `Row ${pi} of P has ${live} non-zero entries; row ${pi} of O is the weighted mean of the highlighted rows of V.`}
      </p>
    </Bench>
  );
}

export function S202() {
  return (
    <>
      <p>
        A layer&rsquo;s attention is a set of heads that do not talk to each other. Each has its own weights and
        computes its own output from the same input, and the outputs are only concatenated at the end. For a GPU that
        means the heads are one more grid dimension, so the book designs a single head and the rest of the chapter does
        too. In training the heads drift apart and specialise (some track distance, some syntax, some topic), which is
        why there are many of them.
      </p>
      <Source section="20.2" pages="482" />
      <p>
        The labs index heads the way PyTorch&rsquo;s attention call does: batch first, then heads, then tokens, then the
        head width, <code>[batch, heads, N, d]</code>. One head of one sequence is then a contiguous N × d matrix.
      </p>

      <h2>Every matrix of one head</h2>
      <p>
        X is N × d: one row per token so far, whether it came from the system prompt, your input, retrieved documents or
        the model&rsquo;s own output. N is the sequence length, also called the context length, and it grows as the
        conversation does. Three learned d × d matrices project X: Q = X W<sub>Q</sub>, K = X W<sub>K</sub>, V = X W
        <sub>V</sub>, each N × d. (The book draws them square for simplicity; in general they are d<sub>x</sub> ×
        d<sub>q</sub> and so on, and Q K<sup>T</sup> forces d<sub>q</sub> = d<sub>k</sub>.)
      </p>
      <Source section="20.2" pages="482–483" figs="Fig. 20.2" />
      <p>
        Then Eq. 20.1: O = softmax(Q K<sup>T</sup> / √d + M) V. Entry (i, j) of S = Q K<sup>T</sup> is the inner product
        of token i&rsquo;s query with token j&rsquo;s key, which the book reads as their similarity. S is N × N. M adds −∞
        to every entry above the diagonal, so no token can look at a later one. P, the row-wise softmax, is N × N, and
        O = P V is N × d again, ready for the Add &amp; Norm. As before, d here is the head width.
      </p>
      <Source section="20.2" pages="483–484" />
      <Ask q={questionsS202['ch20/20.2/q-s-bytes']} label="Work it out" />

      <h2>The mask, the scale and the max</h2>
      <p>
        Three details of Eq. 20.1 and 20.2 matter for a kernel, and the viewer lets you switch each one off. The
        factor 1/√d keeps dot products from growing with the head width; Fig. 20.2 leaves it out, the code has it.
        The mask turns P lower-triangular: row i of P is a probability distribution over tokens 0 to i, and row i of O
        is a weighted mean of rows 0 to i of V. And the softmax subtracts each row&rsquo;s maximum m<sub>r</sub> before
        exponentiating (Eq. 20.2).
      </p>
      <Source section="20.2" pages="484–485" />
      <p>
        That subtraction multiplies the numerators and the sum of a row by the same factor e<sup>−m<sub>r</sub></sup>,
        so P cannot change. What changes is the range: every exponent is at most 0, so nothing overflows, and masked
        logits give e<sup>−∞</sup> = 0 exactly. Predict before you try it: with the subtraction off and every logit
        raised by 90, what does row 0 of P show?
      </p>
      <Simplification>
        <p>
          A few tokens and one head. Float32 is emulated by rounding every operation once (Math.fround); a GPU&rsquo;s
          own exp rounds slightly differently, so last digits can differ.
        </p>
      </Simplification>
      <Viewer />
      <Ask q={questionsS202['ch20/20.2/q-row-nonzeros']} />
      <Ask q={questionsS202['ch20/20.2/q-no-scale']} />
      <Ask q={questionsS202['ch20/20.2/q-shift']} />

      <h2>Two phases</h2>
      <p>
        Fig. 20.3 runs the toy prompt &ldquo;The title of this book is Programming&rdquo;, seven word-tokens. The{' '}
        <b>summarization</b> phase (prefill, as section 20.4 renames it) pushes all seven rows through every layer as
        full matrices and emits &ldquo;Massively&rdquo;. Then the <b>generation</b> phase begins: the new token is
        embedded, appended to X as one more row, and every layer runs again, emitting &ldquo;Parallel&rdquo;, then
        &ldquo;Processors&rdquo;, then the end-of-sequence token.
      </p>
      <Source section="20.2" pages="485–486" figs="Fig. 20.3" />
      <p>
        In this baseline each iteration recomputes K, V, Q, S, P and O for every row, although all rows but the last are
        the same as last time. Section <a href={href('ch20', '20.4')}>20.4</a> removes that waste; section{' '}
        <a href={href('ch20', '20.3')}>20.3</a> first writes the softmax as a kernel.
      </p>
      <Aside>
        <p>Generation is also called the autoregression or decoding phase (p. 485).</p>
      </Aside>
      <Ask q={questionsS202['ch20/20.2/q-rows-x']} label="Count" />

      <h2>What this is in PyTorch</h2>
      <p>
        PyTorch documents the reference semantics of its attention call as exactly these steps: a lower-triangular
        boolean mask, <code>masked_fill</code> with minus infinity, the scaled product of the query with the transposed
        key, a softmax, and a product with V. Grouped-query attention is expressed by repeating each K and V head for its
        group of query heads (<code>repeat_interleave</code>), which section 20.7 returns to. It is also what a CPU
        reference for this chapter&rsquo;s labs has to compute.
      </p>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md D5)">
        <p>
          <code>torch.nn.functional.scaled_dot_product_attention</code>, PyTorch 2.15 documentation: the reference code
          uses <code>tril</code>, <code>masked_fill_(…, float(&quot;-inf&quot;))</code> and{' '}
          <code>repeat_interleave</code> on the head dimension.
        </p>
      </Today>
    </>
  );
}
