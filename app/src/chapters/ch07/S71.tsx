import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import {
  ASYMMETRIC,
  BOOK_1D,
  BOOK_2D,
  EX2,
  EX3_FILTERS,
  POLICIES,
  POLICY_LABEL,
  conv1d,
  conv2d,
  ghostCounts2d,
  ghostFormula2d,
  patch2d,
  radiusOf,
  skippedMap,
  window1d,
  type Policy,
} from '../../sim/convolve';

// Section 7.1: the definition, the boundary policies, the flip that the book (and
// PyTorch) does not do, the 2D form, and counting ghost cells (Exercises 1 to 6).
// Every number shown comes from sim/convolve.ts.

const fmtNum = (v: number) => (Number.isInteger(v) ? String(v) : String(Math.round(v * 100) / 100));

export const questionsS71: Record<string, Question> = {
  'ch07/7.1/q-ex2': {
    kind: 'choice',
    id: 'ch07/7.1/q-ex2',
    prompt: (
      <>
        Exercise 2. Convolve <code>N = {'{4, 1, 3, 2, 3}'}</code> with <code>F = {'{2, 1, 4}'}</code>, using the
        book&rsquo;s formula and zero ghost cells. What is the output?
      </>
    ),
    options: [
      { label: '{8, 21, 13, 20, 7}', correct: true },
      {
        label: '{6, 23, 11, 20, 11}',
        why: 'That is the filter reversed: textbook convolution. The book\'s formula, like PyTorch\'s, multiplies f[0] with the leftmost element of the window.',
      },
      {
        label: '{21, 13, 20}',
        why: 'That is what you get with no ghost cells at all, as PyTorch does with padding=0. The book gives every input an output.',
      },
      {
        label: '{16, 21, 13, 20, 19}',
        why: 'That replicates the edge value into the ghost cells. The exercise uses the book\'s default, zeros.',
      },
    ],
    explain: (
      <p>
        With r = 1, y[i] = 2·x[i−1] + 1·x[i] + 4·x[i+1]. So y[0] = 2·0 + 4 + 4·1 = 8, y[1] = 2·4 + 1 + 4·3 = 21, y[2] =
        2·1 + 3 + 4·2 = 13, y[3] = 2·3 + 2 + 4·3 = 20, and y[4] = 2·2 + 3 + 4·0 = 7. Load Exercise 2 into section 7.1&rsquo;s
        sliding-window explorer and switch the flip on to see where the second option comes from.
      </p>
    ),
    source: 'ch. 7 Exercise 2, p. 180; notes/today/ch07-facts.md D1',
  },
  'ch07/7.1/q-ex3': {
    kind: 'choice',
    id: 'ch07/7.1/q-ex3',
    prompt: (
      <>
        Exercise 3, one part of it. Which of these filters moves every value one place to the right, so that{' '}
        <code>y[i] = x[i-1]</code>?
      </>
    ),
    options: [
      { label: '[0, 1, 0]', why: 'The identity: y[i] = x[i].' },
      { label: '[0, 0, 1]', why: 'f[2] multiplies x[i+1], so y[i] = x[i+1]: every value moves one place to the left.' },
      { label: '[1, 0, 0]', correct: true },
      { label: '[−1/2, 0, 1/2]', why: 'This one computes (x[i+1] − x[i−1]) / 2.' },
      { label: '[1/3, 1/3, 1/3]', why: 'An average of three neighbours.' },
    ],
    explain: (
      <>
        <p>
          In the book&rsquo;s formula f[0] multiplies the leftmost element of the window, x[i−1], so [1, 0, 0] copies
          x[i−1] into y[i]. Under the flipped, textbook definition, [1, 0, 0] and [0, 0, 1] swap meanings, which is why
          the answer to this exercise depends on which definition you use.
        </p>
        <p>
          The other two are worth naming. [−1/2, 0, 1/2] is a central difference, the discrete first derivative, which
          is where chapter 8&rsquo;s stencils start. [1/3, 1/3, 1/3] is a three-point moving average, a smoothing
          filter.
        </p>
      </>
    ),
    source: 'ch. 7 Exercise 3, p. 180',
  },
  'ch07/7.1/q-policy': {
    kind: 'choice',
    id: 'ch07/7.1/q-policy',
    prompt: <>An array of 7 inputs and a 5-tap filter. Switching from zero ghost cells to replicating the edge can change which outputs?</>,
    options: [
      { label: 'Only y[0] and y[6].', why: 'With r = 2 the window of y[1] also reaches one place past the start.' },
      { label: 'y[0], y[1], y[5] and y[6].', correct: true },
      { label: 'All seven.', why: 'The windows of y[2], y[3] and y[4] lie entirely inside the array; they never read a ghost cell.' },
      { label: 'None: the policy only matters for the filter, not for x.', why: 'Ghost cells stand in for missing elements of x.' },
    ],
    explain: (
      <p>
        A window reaches r = 2 places to each side. Only outputs within r of an end have a window that sticks out, so
        only the first r and the last r outputs depend on the policy. In a large image that is a thin border, which is
        why the policy is a correctness question far more than a performance one.
      </p>
    ),
    source: 'ch. 7 §7.1, pp. 161–162',
  },
  'ch07/7.1/q-ex5': {
    kind: 'numeric',
    id: 'ch07/7.1/q-ex5',
    prompt: (
      <>
        Exercise 5 for an 8×8 image and a 3×3 filter. If ghost taps are skipped instead of multiplied by zero, how many
        multiplications are left?
      </>
    ),
    answer: 484,
    unit: 'multiplications',
    explain: (
      <>
        <p>
          Count one dimension first. Along a row of 8 outputs with r = 1, the first and the last output each lose one
          tap, so there are 8·3 − 2 = 22 real taps per row direction. A tap is real when it is real in both directions,
          so the total is 22² = 484.
        </p>
        <p>
          With ghost taps there are 64·9 = 576, and the ghost cells themselves form a frame one wide around the image:
          10² − 8² = 36. The ghost counter in section 7.1 checks all three.
        </p>
      </>
    ),
    source: 'ch. 7 Exercise 5, p. 180',
  },
  'ch07/7.1/q-pytorch': {
    kind: 'choice',
    id: 'ch07/7.1/q-pytorch',
    prompt: (
      <>
        <code>torch.nn.functional.conv2d</code> on a 1×1×7×7 input with a 5×5 weight and <code>padding=0</code>. In the
        book&rsquo;s terms, what does it compute?
      </>
    ),
    options: [
      {
        label: 'Only the 3×3 outputs whose patch contains no ghost cell.',
        correct: true,
      },
      { label: 'All 7×7 outputs, with zero ghost cells.', why: 'That needs padding=2 (or padding="same").' },
      { label: 'All 7×7 outputs, with the edge values replicated.', why: 'Replication is a padding_mode of nn.Conv2d, and it only matters when there is padding.' },
      { label: 'The same as the book, but with the filter flipped.', why: 'PyTorch documents Conv2d as cross-correlation: no flip, like the book.' },
    ],
    explain: (
      <p>
        No padding means no ghost cells, so only outputs whose 5×5 patch lies entirely inside the image exist: rows and
        columns 2 to 4, a 3×3 result. <code>padding=2</code> gives the book&rsquo;s 7×7 result with zero ghost cells;
        an <code>nn.Conv2d</code> with <code>padding_mode=&apos;replicate&apos;</code> or{' '}
        <code>&apos;circular&apos;</code> gives the book&rsquo;s other two policies.
      </p>
    ),
    source: 'notes/today/ch07-facts.md D1',
  },
};

// ---------------------------------------------------------------- explorer: the sliding window
interface Preset {
  key: string;
  label: string;
  x: readonly number[];
  f: readonly number[];
}
const PRESETS: Preset[] = [
  { key: 'book', label: "the book's x and f (Figs. 7.1 to 7.3)", x: BOOK_1D.x, f: BOOK_1D.f },
  { key: 'ex2', label: 'Exercise 2', x: EX2.x, f: EX2.f },
  ...EX3_FILTERS.map((e) => ({ key: 'ex3' + e.key, label: `Exercise 3 (${e.key}): ${e.printed}`, x: BOOK_1D.x, f: e.f })),
  { key: 'asym', label: 'an asymmetric filter, [3, 2, 1] (ours)', x: BOOK_1D.x, f: ASYMMETRIC },
];

/** Parse "8, 2, 5" or "1/3 1/3 1/3"; null when anything is not a number. */
function parseList(s: string): number[] | null {
  const parts = s.split(/[\s,]+/).filter(Boolean);
  if (!parts.length) return null;
  const out: number[] = [];
  for (const p of parts) {
    const m = /^(-?\d+(?:\.\d+)?)(?:\/(\d+(?:\.\d+)?))?$/.exec(p.replace('−', '-'));
    if (!m) return null;
    const v = m[2] ? Number(m[1]) / Number(m[2]) : Number(m[1]);
    if (!isFinite(v)) return null;
    out.push(v);
  }
  return out;
}

function SlidingWindow() {
  const { progress } = useProgress();
  const [preset, setPreset] = useState('book');
  const [xText, setXText] = useState(BOOK_1D.x.join(', '));
  const [fText, setFText] = useState(BOOK_1D.f.join(', '));
  const [i, setI] = useState(2);
  const [policy, setPolicy] = useState<Policy>('zero');
  const [flip, setFlip] = useState(false);

  const px = parseList(xText);
  const pf = parseList(fText);
  const xErr = !px ? 'x must be a list of numbers.' : px.length > 16 ? 'x can have at most 16 elements here.' : null;
  const fErr = !pf ? 'f must be a list of numbers.' : pf.length % 2 === 0 ? 'f needs an odd number of weights (2r + 1).' : pf.length > 9 ? 'f can have at most 9 weights here.' : null;
  const p = PRESETS.find((q) => q.key === preset) ?? PRESETS[0];
  const x = !xErr && px ? px : [...p.x];
  const f = !fErr && pf ? pf : [...p.f];
  const n = x.length;
  const r = radiusOf(f.length);
  const ii = Math.min(i, n - 1);

  const w = window1d(x, f, ii, { policy, flip });
  const y = conv1d(x, f, { policy, flip });
  const yZero = conv1d(x, f, { policy: 'zero', flip });
  const yUnflipped = conv1d(x, f, { policy, flip: false });
  const touches = x.map((_, k) => window1d(x, f, k).touchesGhost);
  const byPolicy = y.map((v, k) => (Math.abs(v - yZero[k]) > 1e-9 ? k : -1)).filter((k) => k >= 0);
  const byFlip = y.map((v, k) => (Math.abs(v - yUnflipped[k]) > 1e-9 ? k : -1)).filter((k) => k >= 0);

  // y[0] of the book's data is Exercise 1: hidden until a prediction is committed.
  const edited = xText.trim() !== p.x.join(', ') || fText.trim() !== p.f.join(', ');
  const locked = preset === 'book' && !edited && policy === 'zero' && !flip && !progress.answers['ch07/7.1/p-y0'];
  // Exercise 2's data, however it got here (preset or typed): every output hidden until the question is answered.
  const same = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((v, k) => Math.abs(v - b[k]) < 1e-9);
  const ex2Locked = same(x, EX2.x) && same(f, EX2.f) && !progress.answers['ch07/7.1/q-ex2'];
  const hide = (k: number) => ex2Locked || (locked && k === 0);

  const load = (key: string) => {
    const q = PRESETS.find((z) => z.key === key)!;
    setPreset(key);
    setXText(q.x.map(fmtNum).join(', '));
    setFText(q.f.map(fmtNum).join(', '));
    setI(Math.min(i, q.x.length - 1));
  };

  // Drawing: one column per logical index -r .. n-1+r.
  const C = 46;
  const L = 92;
  const col = (k: number) => L + (k + r) * C;
  const W = L + (n + 2 * r) * C + 8;
  const rows = { x: 26, f: 104, prod: 170, y: 262 };

  return (
    <Bench
      title="The sliding window"
      task="Drag the output index across the array, then change the boundary policy and switch the flip on."
      foot={
        <>
          Dashed cells are ghost cells: positions outside x. What they hold depends on the policy. Outputs drawn dashed
          have a window that reaches a ghost cell. Fractions such as 1/3 can be typed into x and f.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Data
          <select value={preset} onChange={(e) => load(e.target.value)} aria-label="Preset">
            {PRESETS.map((q) => (
              <option key={q.key} value={q.key}>
                {q.label}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          x
          <input type="text" value={xText} onChange={(e) => setXText(e.target.value)} aria-label="x" style={{ width: '14rem' }} />
        </label>
        <label className="field">
          f
          <input type="text" value={fText} onChange={(e) => setFText(e.target.value)} aria-label="f" style={{ width: '9rem' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Output index i = {ii}
          <input type="range" min={0} max={n - 1} value={ii} onChange={(e) => setI(+e.target.value)} aria-label="Output index" style={{ width: '12rem' }} />
        </label>
        <span className="seg" role="group" aria-label="Boundary policy">
          {POLICIES.map((q) => (
            <button key={q} aria-pressed={policy === q} onClick={() => setPolicy(q)}>
              {POLICY_LABEL[q]}
            </button>
          ))}
        </span>
        <label className="check">
          <input type="checkbox" checked={flip} onChange={(e) => setFlip(e.target.checked)} /> flip the filter (textbook
          convolution)
        </label>
      </div>
      {xErr || fErr ? (
        <p className="t-fail" role="alert" style={{ margin: '0 0 0.6rem' }}>
          {xErr ?? fErr} Showing the last valid data.
        </p>
      ) : null}
      <svg
        viewBox={`0 0 ${W} 300`}
        role="img"
        aria-label={`Window of y[${ii}]`}
        data-i={ii}
        data-sum={hide(ii) ? '' : w.sum}
        style={{ width: '100%', maxWidth: W, minWidth: 460, display: 'block' }}
      >
        <text x={8} y={rows.x + 24} fontSize="13" fontWeight="600">x</text>
        {Array.from({ length: n + 2 * r }, (_, c) => {
          const k = c - r;
          const ghost = k < 0 || k >= n;
          const inWin = k >= ii - r && k <= ii + r;
          const tap = inWin ? w.taps[k - (ii - r)] : null;
          const shown = ghost ? (inWin ? tap!.value : null) : x[k];
          return (
            <g key={c}>
              <rect x={col(k)} y={rows.x} width={C - 4} height={34} rx="2"
                fill={inWin ? 'var(--rule-soft)' : ghost ? 'transparent' : 'var(--cell)'}
                stroke={inWin ? 'var(--ink)' : 'var(--rule)'} strokeWidth={inWin ? 1.8 : 1}
                strokeDasharray={ghost ? '4 3' : undefined} />
              {shown != null ? (
                <text x={col(k) + (C - 4) / 2} y={rows.x + 22} fontSize="13" textAnchor="middle"
                  fontWeight={k === ii ? 700 : 400} style={{ fill: ghost ? 'var(--ink-2)' : 'var(--ink)' }}>
                  {fmtNum(shown)}
                </text>
              ) : null}
              <text x={col(k) + (C - 4) / 2} y={rows.x - 6} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
                {`x[${k}]`}
              </text>
            </g>
          );
        })}
        <text x={8} y={rows.f + 22} fontSize="13" fontWeight="600">{flip ? 'f, reversed' : 'f'}</text>
        {w.taps.map((t) => (
          <g key={'f' + t.j}>
            <rect x={col(ii + t.j)} y={rows.f} width={C - 4} height={30} rx="2" fill="var(--cell)" stroke="var(--ink-2)" />
            <text x={col(ii + t.j) + (C - 4) / 2} y={rows.f + 20} fontSize="13" textAnchor="middle">{fmtNum(t.weight)}</text>
            <text x={col(ii + t.j) + (C - 4) / 2} y={rows.f + 44} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
              f[{t.fIndex}]
            </text>
          </g>
        ))}
        <text x={8} y={rows.prod + 22} fontSize="13" fontWeight="600">products</text>
        {w.taps.map((t) => (
          <g key={'p' + t.j}>
            <rect x={col(ii + t.j)} y={rows.prod} width={C - 4} height={30} rx="2" fill="var(--rule-soft)" stroke="var(--rule)"
              strokeDasharray={t.ghost ? '4 3' : undefined} />
            <text x={col(ii + t.j) + (C - 4) / 2} y={rows.prod + 20} fontSize="13" textAnchor="middle">
              {hide(ii) ? '?' : fmtNum(t.product)}
            </text>
          </g>
        ))}
        <text x={col(ii - r)} y={rows.prod + 56} fontSize="13.5" fontWeight="600" data-out="sum">
          {ex2Locked
            ? 'These outputs are Exercise 2: answer it below to see them'
            : hide(ii)
              ? 'y[0] is Exercise 1: commit a prediction above to see it'
              : `y[${ii}] = ${w.taps.map((t) => fmtNum(t.product)).join(' + ').replace(/\+ -/g, '− ')} = ${fmtNum(w.sum)}`}
        </text>
        <text x={8} y={rows.y + 22} fontSize="13" fontWeight="600">y</text>
        {y.map((v, k) => (
          <g key={'y' + k} style={{ cursor: 'pointer' }} onClick={() => setI(k)}>
            <rect x={col(k)} y={rows.y} width={C - 4} height={30} rx="2" fill={k === ii ? 'var(--rule-soft)' : 'var(--cell)'}
              stroke={k === ii ? 'var(--ink)' : 'var(--rule)'} strokeWidth={k === ii ? 1.8 : 1}
              strokeDasharray={touches[k] ? '4 3' : undefined} />
            <text x={col(k) + (C - 4) / 2} y={rows.y + 20} fontSize="13" textAnchor="middle" fontWeight={k === ii ? 700 : 400}>
              {hide(k) ? '?' : fmtNum(v)}
            </text>
          </g>
        ))}
      </svg>
      <p className="ui" style={{ margin: '0.6rem 0 0' }} data-out="y" data-y={JSON.stringify(y.map((v, k) => (hide(k) ? null : Math.round(v * 1000) / 1000)))}>
        y = [{y.map((v, k) => (hide(k) ? '?' : fmtNum(v))).join(', ')}].{' '}
        <span data-out="policy-changed">
          {policy === 'zero'
            ? `Outputs whose window reaches a ghost cell: ${touches.map((t, k) => (t ? `y[${k}]` : null)).filter(Boolean).join(', ') || 'none'}.`
            : `Compared with zero ghost cells, this policy changed ${byPolicy.length ? byPolicy.map((k) => `y[${k}]`).join(', ') : 'nothing'}.`}
        </span>{' '}
        {flip ? (
          <span data-out="flip-changed">
            Compared with the book&rsquo;s un-flipped form, the flip changed {byFlip.length ? byFlip.map((k) => `y[${k}]`).join(', ') : 'nothing'}.
          </span>
        ) : null}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: one output's patch
function PatchExplorer() {
  const { N, F } = BOOK_2D;
  const [sel, setSel] = useState({ row: 2, col: 2 });
  const P = useMemo(() => conv2d(N, F), [N, F]);
  const p = patch2d(N, F, sel.row, sel.col);
  const r = p.ry;
  const rowSums = [0, 1, 2, 3, 4].map((k) => p.taps.slice(k * 5, k * 5 + 5).reduce((s, t) => s + t.product, 0));
  const inPatch = (row: number, col: number) => Math.abs(row - sel.row) <= r && Math.abs(col - sel.col) <= r;
  const cell = { minWidth: '1.9rem' };
  return (
    <Bench
      title="One output's patch"
      task="Click any output of P. Try the centre, an edge and a corner."
      foot="The input and filter are the book's Fig. 7.4. Ghost cells (dashed) hold 0, the book's default."
    >
      <div className="row" style={{ alignItems: 'flex-start', gap: '1.4rem' }}>
        <div>
          <div className="mx-label">N, with the patch of P[{sel.row}][{sel.col}]</div>
          <div className="mx" style={{ gridTemplateColumns: `repeat(${N.width + 2 * r}, auto)` }} data-grid="N">
            {Array.from({ length: (N.height + 2 * r) * (N.width + 2 * r) }, (_, k) => {
              const row = Math.floor(k / (N.width + 2 * r)) - r;
              const col = (k % (N.width + 2 * r)) - r;
              const ghost = row < 0 || row >= N.height || col < 0 || col >= N.width;
              const on = inPatch(row, col);
              if (ghost)
                return on ? (
                  <span key={k} className="void" style={cell}>0</span>
                ) : (
                  <span key={k} style={{ ...cell, visibility: 'hidden' }} />
                );
              const centre = row === sel.row && col === sel.col;
              return (
                <span key={k} style={{ ...cell, ...(on ? { background: 'var(--rule-soft)', borderColor: 'var(--ink)' } : {}), fontWeight: centre ? 700 : 400 }}>
                  {N.data[row * N.width + col]}
                </span>
              );
            })}
          </div>
        </div>
        <div>
          <div className="mx-label">filter</div>
          <div className="mx" style={{ gridTemplateColumns: 'repeat(5, auto)' }}>
            {F.data.map((v, k) => (
              <span key={k} style={cell}>{v}</span>
            ))}
          </div>
        </div>
        <div>
          <div className="mx-label">products</div>
          <div className="mx" style={{ gridTemplateColumns: 'repeat(5, auto)' }} data-grid="products">
            {p.taps.map((t, k) => (
              <span key={k} className={t.ghost ? 'void' : ''} style={cell}>{t.product}</span>
            ))}
          </div>
        </div>
        <div>
          <div className="mx-label">P (click an output)</div>
          <div className="mx" style={{ gridTemplateColumns: `repeat(${N.width}, auto)` }} data-grid="P">
            {P.data.map((v, k) => {
              const row = Math.floor(k / N.width);
              const col = k % N.width;
              const on = row === sel.row && col === sel.col;
              return (
                <span key={k} role="button" tabIndex={0} aria-label={`P[${row}][${col}]`} data-rc={`${row},${col}`}
                  onClick={() => setSel({ row, col })} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setSel({ row, col })}
                  style={{ ...cell, cursor: 'pointer', ...(on ? { background: 'var(--ink)', color: 'var(--page)', fontWeight: 700 } : {}) }}>
                  {v}
                </span>
              );
            })}
          </div>
        </div>
      </div>
      <p className="ui" style={{ margin: '0.9rem 0 0' }} data-out="patch" data-sum={p.sum} data-ghost={p.ghostTaps}>
        P[{sel.row}][{sel.col}] = {rowSums.join(' + ')} = <b>{p.sum}</b> (one term per row of products).{' '}
        {p.ghostTaps ? `${p.ghostTaps} of its 25 taps fall on ghost cells and add nothing.` : 'Its whole patch lies inside the image.'}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: count the ghost cells
const M_CHOICES = [1, 3, 5, 7, 9, 11, 13, 15];

function GhostCounter() {
  const { progress } = useProgress();
  const [n, setN] = useState(7);
  const [m, setM] = useState(5);
  const [showF, setShowF] = useState(false);
  const r = radiusOf(m);
  const b = ghostCounts2d(n, n, m, m);
  const fm = ghostFormula2d(n, n, m, m);
  const answered = !!progress.answers['ch07/7.1/q-ex5'];
  // 8×8 with a 3×3 filter is the Exercise 5 question: its counts of what is skipped stay hidden until it is answered.
  const ex5Locked = !answered && n === 8 && m === 3;
  const skipped = useMemo(() => skippedMap(n, n, m, m), [n, m]);
  const maxSkip = Math.max(1, ...skipped);

  // Picture: the image with each output shaded by how many of its taps are skipped, inside
  // its frame of ghost cells.
  const S = 300;
  const cellPx = S / (n + 2 * r);
  const o = r * cellPx;
  return (
    <Bench
      title="Count the ghost cells"
      task="Set the image and filter sizes. Find how each count grows, then compare with the formulas."
      foot="Every count is made by visiting every tap of every output. Dashed: the frame of ghost cells. Shading: how many of an output's taps are skipped."
    >
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div>
          <div className="row" style={{ marginBottom: '0.8rem' }}>
            <label className="field">
              Image N×N, N = {n}
              <input type="range" min={1} max={64} value={n} onChange={(e) => setN(+e.target.value)} aria-label="Image width N" style={{ width: '13rem' }} />
            </label>
            <label className="field">
              Filter M×M
              <select value={m} onChange={(e) => setM(+e.target.value)} aria-label="Filter width M">
                {M_CHOICES.map((v) => (
                  <option key={v} value={v}>
                    {v} × {v} (r = {radiusOf(v)})
                  </option>
                ))}
              </select>
            </label>
          </div>
          <table className="data" style={{ margin: 0 }} data-out="counts">
            <thead>
              <tr>
                <th>Count</th>
                <th>By visiting every tap</th>
                {showF ? <th>Formula</th> : null}
              </tr>
            </thead>
            <tbody>
              <tr data-row="ghost">
                <td>Ghost cells</td>
                <td>{b.ghostCells.toLocaleString('en-US')}</td>
                {showF ? <td>{fm.ghostCells.toLocaleString('en-US')}</td> : null}
              </tr>
              <tr data-row="all">
                <td>Multiplications, ghost taps included</td>
                <td>{b.allTaps.toLocaleString('en-US')}</td>
                {showF ? <td>{fm.allTaps.toLocaleString('en-US')}</td> : null}
              </tr>
              <tr data-row="real">
                <td>Multiplications, ghost taps skipped</td>
                <td>{ex5Locked ? '?' : b.realTaps.toLocaleString('en-US')}</td>
                {showF ? <td>{fm.realTaps.toLocaleString('en-US')}</td> : null}
              </tr>
            </tbody>
          </table>
          <p className="ui" style={{ margin: '0.7rem 0 0' }} data-out="share">
            {ex5Locked ? (
              <>This is the Exercise 5 question: answer it above to see what skipping saves here.</>
            ) : (
              <>
                Skipping ghost taps saves {fmtNum(Math.round(((b.allTaps - b.realTaps) / b.allTaps) * 1000) / 10)} % of
                the multiplications here.
              </>
            )}
          </p>
          <div style={{ marginTop: '0.9rem' }}>
            {showF ? (
              <div className="panel" data-out="formulas">
                <p style={{ margin: '0 0 0.4rem' }}>
                  With r = (M − 1)/2, for an N×N image and an M×M filter (Exercise 5):
                </p>
                <ul style={{ margin: '0 0 0.4rem', paddingLeft: '1.2rem' }}>
                  <li>ghost cells: (N + M − 1)² − N² = 4r(N + r)</li>
                  <li>multiplications with ghost taps: N²·M²</li>
                  <li>without them: (N·M − r(r + 1))²</li>
                </ul>
                <p style={{ margin: '0 0 0.4rem' }}>
                  In 1D (Exercise 4) the same counts are M − 1, N·M and N·M − r(r + 1): each end loses 1 + 2 + … + r
                  taps. For an N1×N2 image and an M1×M2 filter (Exercise 6) multiply the two directions: (N1 + M1 − 1)(N2
                  + M2 − 1) − N1·N2, N1·N2·M1·M2, and (N1·M1 − r1(r1 + 1))·(N2·M2 − r2(r2 + 1)).
                </p>
                <p className="muted" style={{ margin: 0 }}>
                  The last formula assumes N ≥ r: every end has r outputs to lose taps from.{' '}
                  {n < r ? <b>Here N &lt; r, and it disagrees with the count.</b> : null}
                </p>
              </div>
            ) : (
              <p className="ui" style={{ margin: 0 }}>
                <button onClick={() => setShowF(true)} disabled={!answered}>
                  Show the formulas
                </button>{' '}
                {answered ? null : <span className="muted">Answer the Exercise 5 question above first.</span>}
              </p>
            )}
          </div>
        </div>
        <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Image with its frame of ghost cells" style={{ width: '100%', maxWidth: 320 }}>
          {r > 0 ? (
            <path d={`M0 0H${S}V${S}H0Z M${o} ${o}V${o + n * cellPx}H${o + n * cellPx}V${o}Z`} fillRule="evenodd"
              fill="var(--rule-soft)" stroke="var(--ink-2)" strokeWidth="1" strokeDasharray="4 3" />
          ) : null}
          {r > 0 && cellPx >= 7
            ? Array.from({ length: (n + 2 * r) * (n + 2 * r) }, (_, k) => {
                const row = Math.floor(k / (n + 2 * r)) - r;
                const col = (k % (n + 2 * r)) - r;
                if (row >= 0 && row < n && col >= 0 && col < n) return null;
                return (
                  <rect key={'g' + k} x={o + col * cellPx + 1} y={o + row * cellPx + 1} width={cellPx - 2} height={cellPx - 2}
                    fill="none" stroke="var(--ink-3)" strokeWidth="0.8" strokeDasharray="2 2" />
                );
              })
            : null}
          <rect x={o} y={o} width={n * cellPx} height={n * cellPx} fill="var(--cell)" stroke="var(--ink)" strokeWidth="1" />
          {skipped.map((s, k) =>
            s ? (
              <rect key={k} x={o + (k % n) * cellPx} y={o + Math.floor(k / n) * cellPx} width={cellPx} height={cellPx}
                fill="var(--ink)" fillOpacity={0.12 + 0.6 * (s / maxSkip)} />
            ) : null,
          )}
          <text x={S / 2} y={S / 2 + 4} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
            {n} × {n}
          </text>
        </svg>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the section
export function S71() {
  return (
    <>
      <p>
        A convolution replaces every element of an array with a weighted sum of itself and its neighbours. The weights
        are the same for every element and come from a small array called the <b>filter</b>. Most of the literature
        calls it the convolution kernel; the book says filter, to keep &ldquo;kernel&rdquo; for the CUDA functions that
        compute it, and so does this course.
      </p>
      <Source section="7.1" pages="159–160" />
      <p>
        A filter has an odd number of weights, 2r + 1, so that it is centred on the element being computed with r
        neighbours on each side; r is its <b>radius</b>. For an input x and a filter f, output i is
      </p>
      <p style={{ textAlign: 'center' }}>
        <code>y[i] = f[0]·x[i−r] + f[1]·x[i−r+1] + … + f[2r]·x[i+r]</code>,
      </p>
      <p>
        a sum over j from −r to r of <code>f[j+r]·x[i+j]</code>. Read it as an inner product: the filter laid over a
        window of x centred on i. With the book&rsquo;s x = [8, 2, 5, 4, 1, 7, 3] and f = [1, 3, 5, 3, 1], r = 2, the
        window for y[2] is x[0] to x[4] and y[2] = 8 + 6 + 25 + 12 + 1 = 52. Slide one place and y[3] = 2 + 15 + 20 + 3
        + 7 = 47.
      </p>
      <Source section="7.1" pages="160–161" figs="Figs. 7.1, 7.2" />
      <p>
        Since every output is an inner product with the same filter, a whole convolution can be written as one matrix
        multiplication. The book leaves that to chapter 19 (<a href={href('ch19', '19.3')}>§19.3</a>), and it is how libraries usually run it.
      </p>
      <Source pages="161" figs="footnote 1" />

      <h2>The book&rsquo;s formula does not flip the filter</h2>
      <p>
        Look at which weight meets which input. In the formula above f[0] multiplies the leftmost element of the
        window. The convolution of signal-processing textbooks reverses the filter first, so that f[0] meets the
        rightmost one. The book never mentions the difference, and every filter it uses is symmetric, where the two
        agree.
      </p>
      <p>
        The book&rsquo;s choice is the one you already use. PyTorch documents <code>Conv2d</code> as the &ldquo;valid
        2D cross-correlation operator&rdquo;, which is this un-flipped form; it asks cuDNN for its cross-correlation
        mode, its ROCm path calls MIOpen&rsquo;s <code>miopenConvolution</code>, documented as
        &ldquo;Cross-Correlation convolution&rdquo;, and MLX&rsquo;s <code>conv_general</code> only flips when you pass{' '}
        <code>flip=True</code>. So the weights of a trained convolution layer are applied exactly as stored.
      </p>
      <Today checked="2026-11-20 (notes/today/ch07-facts.md, D1)">
        <p>
          PyTorch 2.15 <code>Conv2d</code>: &ldquo;valid 2D cross-correlation&rdquo;. cuDNN mode{' '}
          <code>CUDNN_CROSS_CORRELATION</code> set by PyTorch; MIOpen <code>miopenConvolution</code>; MLX 0.32.4{' '}
          <code>conv_general(..., flip=False)</code>.
        </p>
      </Today>

      <h2>Ghost cells</h2>
      <p>
        Near the ends the window runs off the array. The book&rsquo;s fix is to pretend the missing elements exist and
        hold a default value, usually 0 (an audio signal is silent before the recording starts). It calls these
        imaginary elements <b>ghost cells</b>. With one ghost cell on the left, y[1] = 1·0 + 3·8 + 5·2 + 3·5 + 1·4 = 53.
        Tiling will create a second kind of extra cell later in the chapter, the halo.
      </p>
      <Source section="7.1" pages="161–162" figs="Fig. 7.3" />
      <p>
        Zero is a policy, not a law. The book names two others: copy the nearest real element into the ghost cells, or
        treat the array as circular so that it wraps around.
      </p>
      <Predict
        id="ch07/7.1/p-y0"
        prompt={
          <>
            Exercise 1. With zero ghost cells, what is y[0] for the book&rsquo;s x = [8, 2, 5, 4, 1, 7, 3] and f = [1, 3,
            5, 3, 1]?
          </>
        }
      >
        {(v) => (
          <>
            <p>
              {v === 51 ? 'Right: ' : ''}y[0] = 1·0 + 3·0 + 5·8 + 3·2 + 1·5 = <b data-out="y0">51</b>. Two ghost cells on
              the left this time. The explorer below now shows it too.
            </p>
            <p className="muted">Source: ch. 7 Exercise 1, p. 180 (the exercise calls the output P[0]; the figure calls it y).</p>
          </>
        )}
      </Predict>
      <SlidingWindow />
      <p>
        Three things to find in the explorer. Only the first r and last r outputs change with the policy. The flip
        changes nothing for the book&rsquo;s symmetric filter and nearly everything for the asymmetric one. And the
        Exercise 3 filters do what their shapes suggest only once you know which definition is in force.
      </p>
      <Ask q={questionsS71['ch07/7.1/q-ex2']} label="Exercise 2" />
      <Ask q={questionsS71['ch07/7.1/q-ex3']} label="Exercise 3" />
      <Ask q={questionsS71['ch07/7.1/q-policy']} />
      <p>
        PyTorch exposes the same choice. An <code>nn.Conv2d</code> takes <code>padding_mode</code>{' '}
        <code>&apos;zeros&apos;</code>, <code>&apos;replicate&apos;</code> or <code>&apos;circular&apos;</code>, which are
        the book&rsquo;s three policies, and <code>&apos;reflect&apos;</code>, a fourth the book does not mention. Its
        default <code>padding=0</code> means no ghost cells at all: only outputs whose window fits inside the input
        exist, so the output is smaller than the input.
      </p>
      <Beyond>
        <p>
          GPUs can apply these policies in hardware when an image is read as a texture: CUDA and HIP address modes
          Border, Clamp, Wrap and Mirror; Metal samplers <code>clamp_to_zero</code>, <code>clamp_to_edge</code> and{' '}
          <code>repeat</code>. NVIDIA&rsquo;s Programming Guide says textures give no performance advantage for
          non-graphics code on current GPUs, so the course, like the book, reads plain buffers (facts A6, B4, C5).
        </p>
      </Beyond>

      <h2>Two dimensions</h2>
      <p>
        In 2D the filter is a small square (it need not be, but usually is) of 2r + 1 rows and columns, and an output is
        the sum of the element-by-element products of the filter with the patch of the input centred on it:{' '}
        <code>P[y][x]</code> is the sum over j and k from −r to r of <code>f[j+r][k+r]·N[y+j][x+k]</code>. The book
        writes the indices as subscripts and warns that real code linearises them, because N and P are allocated
        dynamically: <code>N[(y+j)*width + (x+k)]</code>, as in chapter 3.
      </p>
      <Source section="7.1" pages="162–163" figs="Fig. 7.4" />
      <p>
        The book&rsquo;s example is a 7×7 input and a 5×5 filter. P[2][2] has its whole patch inside the input; its 25
        products add up to 321 (by rows, 27 + 56 + 95 + 84 + 59; the row sums are ours). At P[1][0] the patch hangs over
        the top edge by one row and over the left edge by two columns, and with zeros there the sum is 112.
      </p>
      <Source section="7.1" pages="163–164" figs="Fig. 7.5" />
      <PatchExplorer />
      <Ask q={questionsS71['ch07/7.1/q-pytorch']} label="In PyTorch" />

      <h2>How much of the work is at the edges?</h2>
      <p>
        Exercises 4 to 6 ask how many ghost cells there are and how many multiplications a kernel does with and without
        the ghost taps. The answers matter for section 7.2: a thread skips exactly its ghost taps, so these counts say
        how much of the image sits in the awkward border. Work the 8×8 case first, then use the counter to check it and
        to find the general formulas.
      </p>
      <Source pages="180" figs="Exercises 4–6" />
      <Ask q={questionsS71['ch07/7.1/q-ex5']} label="Exercise 5" />
      <p>
        The counter visits every tap of every output and counts. Change N and M one at a time and look for the pattern
        before you open the formulas.
      </p>
      <Simplification>
        <p>
          The counter counts taps, not time. Whether a skipped tap is cheaper than multiplying by a zero depends on the
          hardware and the compiler; section 7.2 looks at what the branch does to a warp.
        </p>
      </Simplification>
      <GhostCounter />
      <p>
        For a large image the skipped taps are a thin border: the share they save falls roughly as 1/N. So the boundary
        policy is a question of getting the right answer, and the edge threads are a small part of the cost. Section{' '}
        <a href={href('ch07', '7.2')}>7.2</a> turns the formula into a kernel with one thread per output.
      </p>
    </>
  );
}
