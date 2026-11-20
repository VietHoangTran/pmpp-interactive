import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  ACT_DIST,
  OUTLIER_J,
  PATH_LABEL,
  ROW_LENGTHS,
  bitwiseAgreement,
  exactDequantDot,
  exactFloatDot,
  hex,
  histogram,
  makeCase,
  rowErrors,
  sci,
  show,
  tracePath,
  type ActDist,
  type Counts,
  type Path,
  type PathRun,
  type QStep,
} from '../../sim/qdot';
import { IntegerPathScene } from './IntegerPathScene';

// Section X2.5: one block's dot product along the float path the Metal kernels use and the
// integer path llama.cpp's CUDA and HIP decode uses. Everything shown is computed by
// src/sim/qdot.ts and pinned in qdot.test.ts. Module X2 is ours: no book pages.
const FACTS = 'notes/today/x2-facts.md';
const CHECKED = `2026-11-20 (${FACTS}`;
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);
/** The dot4_i8 helper, cut from lab X2.3's tested solution: from its comment to its closing brace. */
function helperOf(src: string | null): string | null {
  if (!src) return null;
  const i = src.indexOf('// Four signed byte products');
  const end = i < 0 ? -1 : src.indexOf('\n}\n', i);
  return end < 0 ? null : src.slice(i, end + 2);
}

// ---------------------------------------------------------------- the stepper
const COUNT_ROWS: { k: keyof Counts; label: string }[] = [
  { k: 'mul', label: 'float multiplies' },
  { k: 'add', label: 'float adds' },
  { k: 'cmp', label: 'float compares' },
  { k: 'int', label: 'integer operations' },
  { k: 'conv', label: 'conversions' },
];
const list = (v: number[], d = 4) => v.map((x) => show(x, d)).join(', ');

function describe(s: QStep, run: PathRun, d: number): string {
  const lanes = s.lanes ?? [];
  const js = lanes.map((l) => l.j).join(', ');
  switch (s.kind) {
    case 'group':
      return `Weights ${js}: codes ${lanes.map((l) => l.code).join(', ')}, so (q − 8)·d = ${list(lanes.map((l) => l.wv))} (exact in float32). Four multiply-adds with x = ${list(lanes.map((l) => l.xv))}: acc = ${show(s.acc, 7)}.`;
    case 'prepare':
      return `Once per token for this block, shared by every row: Σx = ${show(s.vals!.sumx, 7)}, and each activation divided by 1, 256, 16 or 4096 to match the mask that will meet it (exact: powers of two).`;
    case 'word':
      return `16-bit word ${hex(s.word!, 4)} masked four ways (0x000F, 0x0F00, 0x00F0, 0xF000) gives weights ${js} as ${lanes.map((l) => l.wv).join(', ')}: codes ${lanes.map((l) => l.code).join(', ')} times 1, 256, 16, 4096, never shifted. Times the pre-divided activations, each product is exactly code × x: ${list(lanes.map((l) => l.prod))}. Four accumulators: ${list(s.accs!, 5)}.`;
    case 'fold':
      return `The offset, once per block: −8 · Σx = ${show(s.vals!.fold, 7)}, plus the four accumulators = ${show(s.acc, 7)}. No weight ever had 8 subtracted from it.`;
    case 'rescale':
      return `Times the block's scale d = ${show(d, 6)}: the block's result is ${show(s.acc, 8)}.`;
    case 'quantise-max':
      return `Once per token for this block: a 32-wide max finds amax = ${show(s.vals!.amax, 6)}; d8 = amax / 127 = ${show(s.vals!.dFloat, 6)}, stored as FP16 ${show(s.vals!.d, 6)}.`;
    case 'quantise-round': {
      const q = run.q8!;
      return `Each x / d8 rounded to a signed byte: ${q.qs.slice(0, 8).join(', ')}, … Their sum is ${s.vals!.sum}; s = d8 · ${s.vals!.sum} = ${show(s.vals!.s, 6)} in FP16.`;
    }
    case 'dp4a':
      return `Word ${hex(s.word!)}${s.shift ? ' shifted right by 4' : ''}, masked with 0x0F0F0F0F: ${hex(s.masked!)}, the unsigned codes ${lanes.map((l) => l.wv).join(', ')} of weights ${js}. dp4a with the bytes ${lanes.map((l) => l.xv).join(', ')} adds ${lanes.reduce((t, l) => t + l.prod, 0)}: sumi = ${s.acc}.`;
    case 'correct':
      return `The correction, once per block: sumi · d8 − 8 · s = ${s.vals!.sumi} · ${show(s.vals!.d8, 6)} − 8 · ${show(s.vals!.s, 6)} = ${show(s.vals!.scaled, 7)} − ${show(s.vals!.off, 7)} = ${show(s.acc, 7)}.`;
  }
}

function ThreePaths() {
  const [path, setPath] = useState<Path>('A');
  const [outlier, setOutlier] = useState(false);
  const [n, setN] = useState(0);
  const c = useMemo(() => makeCase(outlier), [outlier]);
  const runs = useMemo(() => ({ A: tracePath('A', c), B: tracePath('B', c), C: tracePath('C', c) }), [c]);
  const exact = useMemo(() => exactDequantDot(c.block, c.x), [c]);
  const exactW = useMemo(() => exactFloatDot(c.w, c.x), [c]);
  const run = runs[path];
  const total = run.steps.length;
  const at = Math.min(n, total);
  const cur = at > 0 ? run.steps[at - 1] : undefined;
  const sofar = run.steps.slice(0, at);
  const sum = (perRow: boolean, k: keyof Counts) => sofar.filter((s) => s.perRow === perRow).reduce((t, s) => t + (s.counts[k] ?? 0), 0);
  const choose = (p: Path) => {
    setPath(p);
    setN(0);
  };
  const done = at >= total;
  return (
    <Bench
      title="One block, three paths"
      task="Step through path A, then B, then C. Then tick the large activation and run C again."
      foot={
        <>
          One Q4_0 block of 32 weights (quantised by llama.cpp's reference rule) and 32 float activations, seeded.
          Float arithmetic is float32, rounded at every step the C source rounds. Counts are source-level operations for
          this block: per row, and per token (shared by every output row that reads the same activations).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Path">
          {(['A', 'B', 'C'] as Path[]).map((p) => (
            <button key={p} aria-pressed={path === p} onClick={() => choose(p)}>{PATH_LABEL[p]}</button>
          ))}
        </span>
        <label className="check">
          <input type="checkbox" checked={outlier} onChange={(e) => { setOutlier(e.target.checked); setN(0); }} /> large activation in channel {OUTLIER_J}
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setN(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={done}>Step</button>
        <button onClick={() => setN(total)} disabled={done}>Run to the end</button>
        <span className="muted" aria-live="polite" data-at={at}>step {at} of {total}</span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }} data-narrate="">
        {cur ? describe(cur, run, c.block.d) : (
          <>
            Nothing has run yet. The block's scale is d = {show(c.block.d, 6)} (FP16) and its codes run from 0 to 15.
            {path === 'A' ? ' Path A turns every code into a float weight first.' : path === 'B' ? ' Path B never subtracts 8 from a code.' : ' Path C first turns the activations into bytes.'} Press Step.
          </>
        )}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div>
          {cur?.lanes ? (
            <div style={{ overflowX: 'auto' }}>
            <table className="data" style={{ margin: 0 }} data-lanes="">
              <thead>
                <tr>
                  <th>weight</th>
                  <th className="num">code</th>
                  <th className="num">{path === 'A' ? '(q − 8)·d' : path === 'B' ? 'masked' : 'nibble byte'}</th>
                  <th className="num">{path === 'A' ? 'x' : path === 'B' ? 'x pre-divided' : 'x as int8'}</th>
                  <th className="num">product</th>
                </tr>
              </thead>
              <tbody>
                {cur.lanes.map((l) => (
                  <tr key={l.j}>
                    <td>{l.j}{outlier && l.j === OUTLIER_J ? ' (large)' : ''}</td>
                    <td className="num t-global">{l.code}</td>
                    <td className="num">{show(l.wv, 6)}</td>
                    <td className="num">{show(l.xv, 6)}</td>
                    <td className="num">{show(l.prod, 6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          ) : (
            <p className="muted" style={{ margin: 0 }}>{cur ? 'This step works on the whole block, not on four weights.' : ''}</p>
          )}
          <p style={{ margin: '0.8rem 0 0' }}>
            running value: <b className="t-reg" data-acc="">{cur ? show(cur.acc, 8) : '0'}</b>{' '}
            <span className="muted">{path === 'C' && cur?.kind === 'dp4a' ? '(int32, exact)' : '(float32)'}</span>
          </p>
        </div>
        <table className="data" style={{ margin: 0 }} data-counts="">
          <thead>
            <tr><th>so far</th><th className="num">per row</th><th className="num">per token</th></tr>
          </thead>
          <tbody>
            {COUNT_ROWS.map((r) => (
              <tr key={r.k}>
                <td>{r.label}</td>
                <td className="num" data-count={r.k}>{sum(true, r.k)}</td>
                <td className="num">{sum(false, r.k)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {done ? (
        <table className="data" style={{ margin: '1rem 0 0' }} data-results="">
          <thead>
            <tr><th>path</th><th className="num">block result</th><th className="num">minus exact</th><th className="num">float multiplies per row</th></tr>
          </thead>
          <tbody>
            {(['A', 'B', 'C'] as Path[]).map((p) => (
              <tr key={p} style={p === path ? { fontWeight: 620 } : undefined} data-path={p} data-result={String(runs[p].result)}>
                <td>{PATH_LABEL[p]}</td>
                <td className="num">{show(runs[p].result, 9)}</td>
                <td className="num">{runs[p].result === exact ? '0' : sci(runs[p].result - exact)}</td>
                <td className="num">{runs[p].perRow.mul}</td>
              </tr>
            ))}
            <tr>
              <td>exact, dequantised weights</td>
              <td className="num" data-exact={String(exact)}>{show(exact, 9)}</td>
              <td className="num">0</td>
              <td />
            </tr>
            <tr>
              <td className="muted">exact, the float32 weights before quantising</td>
              <td className="num muted">{show(exactW, 9)}</td>
              <td className="num muted">{sci(exactW - exact)}</td>
              <td />
            </tr>
          </tbody>
        </table>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the explorer
function ActivationCost() {
  const [n, setN] = useState(4096);
  const [dist, setDist] = useState<ActDist>('normal');
  const sel = useMemo(() => rowErrors(n, dist), [n, dist]);
  const other = useMemo(() => rowErrors(n, dist === 'normal' ? 'outliers' : 'normal'), [n, dist]);
  const LO = -7;
  const HI = -1;
  const bins = histogram(sel.actErr, LO, HI, 4);
  const obins = histogram(other.actErr, LO, HI, 4);
  const max = Math.max(...bins.map((b) => b.count), ...obins.map((b) => b.count), 1);
  const gw = 620;
  const gh = 150;
  const bw = gw / bins.length;
  const xl = (e: number) => 40 + ((Math.log10(e) - LO) / (HI - LO)) * gw;
  const T = 34; // top of the bars
  return (
    <Bench
      title="What activation quantisation costs"
      task="Compare the two distributions at each row length. Where does the error of a large channel go?"
      foot={
        <>
          200 seeded rows: standard-normal weights quantised to Q4_0, activations as chosen. Relative error is |C − B|
          divided by the sum of |w·x| over the row (the dequantised weights). The large channels are ours: one in every 128,
          of magnitude about 20 against a standard normal. Errors below 1e−7 are counted in the first bar.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Row length
          <select aria-label="Row length" value={n} onChange={(e) => setN(+e.target.value)}>
            {ROW_LENGTHS.map((v) => <option key={v} value={v}>{v.toLocaleString('en-US')}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label="Activations">
          {ACT_DIST.map((d) => (
            <button key={d.id} aria-pressed={dist === d.id} onClick={() => setDist(d.id)}>{d.label}</button>
          ))}
        </span>
      </div>
      <svg viewBox="0 0 700 226" role="img" aria-label="Histogram of the relative error of path C against path B" style={{ width: '100%', maxWidth: 700, display: 'block', minWidth: 540 }}>
        {obins.map((b, i) => (
          <rect key={'o' + i} x={40 + i * bw + 1} y={T + gh - (b.count / max) * gh} width={bw - 2} height={(b.count / max) * gh}
            fill="none" stroke="var(--ink-3)" strokeDasharray="3 2" />
        ))}
        {bins.map((b, i) => (
          <rect key={i} x={40 + i * bw + 3} y={T + gh - (b.count / max) * gh} width={bw - 6} height={(b.count / max) * gh} fill="var(--ink-2)" data-bin={i} data-count={b.count} />
        ))}
        <line x1={40} x2={40 + gw} y1={T + gh} y2={T + gh} stroke="var(--rule)" />
        {Array.from({ length: HI - LO + 1 }, (_, k) => LO + k).map((e) => (
          <text key={e} x={xl(10 ** e)} y={T + gh + 16} fontSize="10.5" textAnchor="middle">1e{e}</text>
        ))}
        <line x1={xl(sel.median)} x2={xl(sel.median)} y1={T - 4} y2={T + gh} stroke="var(--ink)" strokeWidth="2" />
        <text x={xl(sel.median)} y={12} fontSize="10.5" textAnchor="middle">median</text>
        <line x1={xl(sel.weightMedian)} x2={xl(sel.weightMedian)} y1={T - 4} y2={T + gh} stroke="var(--global)" strokeWidth="2" strokeDasharray="5 3" />
        <text x={xl(sel.weightMedian)} y={26} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--global)' }}>weights to 4 bits (median)</text>
        <text x={40} y={222} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>solid: {ACT_DIST.find((d) => d.id === dist)!.label}; dashed outline: {ACT_DIST.find((d) => d.id !== dist)!.label}</text>
      </svg>
      <p style={{ margin: '0.6rem 0 0' }} data-summary="" data-median={String(sel.median)} data-p95={String(sel.p95)}>
        Path C against path B: median <b>{sci(sel.median)}</b>, 95th percentile <b>{sci(sel.p95)}</b>{' '}
        (the other distribution: median {sci(other.median)}). What quantising the weights to Q4_0 already cost, against
        the float32 weights: median <b className="t-global">{sci(sel.weightMedian)}</b>.
      </p>
    </Bench>
  );
}

function BitwiseReveal({ predicted }: { predicted: number }) {
  const r = bitwiseAgreement(40);
  return (
    <>
      <p data-bitwise={r.same}>
        In the model, <b>{r.same} of 40</b> seeded blocks give the same float32 bits on both paths
        {predicted === r.same ? ', as you said' : ''}. The largest gap, relative to the block's sum of |w·x|, is{' '}
        {sci(r.worstGap)}: float rounding, nothing more.
      </p>
      <p>
        In exact arithmetic the two are the same number for every block (the unit test checks 80 of them). In float32
        they round different intermediate values: path A rounds each scaled weight's product into one sum, path B
        rounds code times pre-divided activation into four sums and folds the offset at the end. So a kernel that
        switches from one to the other changes the low bits of its output, and its tests need a tolerance, not equality.
      </p>
    </>
  );
}

function ActErrorReveal({ predicted }: { predicted: number }) {
  const r = rowErrors(4096, 'normal');
  const pct = r.median * 100;
  return (
    <p data-act-reveal="">
      The model's median over 200 rows of 4,096 is <b>{pct.toPrecision(2)}%</b> of the sum of |w·x|
      {predicted > pct * 10 ? ', well under your guess' : predicted < pct / 10 ? ', well above your guess' : ', near your guess'}. Use
      the explorer below to see how it moves with the row length and with large channels.
    </p>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSX25: Record<string, Question> = {
  'x2/X2.5/q-fold': {
    kind: 'choice',
    id: 'x2/X2.5/q-fold',
    prompt: (
      <>
        A Q4_0 block stores codes q in 0..15 and means (q − 8)·d. Which rewriting takes the −8 out of the loop over
        the 32 weights, and what does it cost per block?
      </>
    ),
    options: [
      {
        label: 'Σ d·(q − 8)·x = d·Σ q·x − 8·d·Σ x: one multiply-add per block, given Σ x, which is computed once per token per block and shared by every output row.',
        correct: true,
      },
      { label: 'd·Σ q·x − 8·Σ x: it costs nothing extra, because d only scales the first term.', why: 'The 8 is in units of d as well: the correction is −8·d·Σx, which is why the kernels multiply by d after folding.' },
      { label: 'None: the subtraction is per weight and has to stay in the loop.', why: 'The sum is linear in the codes, so the constant part of every term can be summed separately.' },
      { label: 'It can leave the loop only once the activations are int8.', why: 'It is plain algebra on any numbers: the Metal kernels fold it with float activations.' },
    ],
    explain: (
      <p>
        Σ (d·q + b)·x = d·Σ q·x + b·Σ x with b = −8·d. The inner loop then touches only codes and activations. Σ x
        depends on the activations alone, so one sum per block of 32 serves every one of the thousands of rows that
        read the same activations. llama.cpp's Metal kernel keeps it as <code>sumy</code>, its CUDA kernel inside
        Q8_1's <code>s</code>, MLX as <code>sum</code> (with its bias in place of −8·d).
      </p>
    ),
    source: `ours (algebra); the kernels that do it: ${FACTS} D3 to D6, E4`,
  },
  'x2/X2.5/q-why-not-int-row': {
    kind: 'choice',
    id: 'x2/X2.5/q-why-not-int-row',
    prompt: <>Path C keeps an exact integer sum inside each block and rescales it. Why not keep one integer sum over the whole row and rescale once at the end?</>,
    options: [
      { label: 'Every block has its own weight scale and activation scale, so the integer sums of different blocks are in different units.', correct: true },
      { label: 'An int32 would overflow over a whole row.', why: 'Even the worst case of 16,129 per product takes 133,144 products to overflow, far more than a 4,096-wide row. Overflow is not the reason.' },
      { label: 'dp4a cannot be called more than eight times on the same accumulator.', why: 'It can be called any number of times; it is just an add into an integer.' },
      { label: 'It would be slower.', why: 'It would be faster, and wrong.' },
    ],
    explain: (
      <p>
        A block's integer sum means d<sub>w</sub>·d<sub>x</sub> times that integer. Adding two blocks' integers before
        scaling assumes their scales are equal, which they are not: each set of 32 weights and each set of 32
        activations got its own. The row is Σ<sub>blocks</sub> d<sub>w</sub>·d<sub>x</sub>·(integer), so it is
        integer inside a block and float across blocks. Section X2.6 shows why that is also the right width for each.
      </p>
    ),
    source: `ours; the kernel's structure: ${FACTS} D3, E3`,
  },
  'x2/X2.5/q-unsigned': {
    kind: 'choice',
    id: 'x2/X2.5/q-unsigned',
    prompt: <>llama.cpp's CUDA kernel feeds the unsigned codes 0..15, not −8..7, into a signed byte dot product. Why is that allowed, and where does the 8 go?</>,
    options: [
      { label: '0..15 is a valid signed byte, so the signed dp4a reads it correctly; the 8 comes off once per block afterwards, as 8·s with s = d·Σq of the activation block.', correct: true },
      { label: 'dp4a sign-extends 4-bit values automatically.', why: 'dp4a works on bytes. After the mask each byte holds a value from 0 to 15, which as a signed byte is just that positive number.' },
      { label: 'The CUDA intrinsic has a mixed unsigned × signed overload for this.', why: 'It has no mixed overload (PTX has one). None is needed: 0..15 is non-negative in a signed byte.' },
      { label: 'The 8 is folded into d when the model is loaded.', why: 'A constant offset cannot be folded into a scale; it has to be subtracted somewhere, and that is the s term.' },
    ],
    explain: (
      <p>
        With unsigned codes the integer sum is Σ q·x<sub>q</sub>; the true block value needs Σ (q − 8)·x<sub>q</sub>,
        which is that minus 8·Σ x<sub>q</sub>. The activation quantiser already computed Σ x<sub>q</sub> and stored
        it, times d<sub>x</sub>, as s. So the kernel returns d<sub>w</sub>·(sumi·d<sub>x</sub> − 8·s): the subtraction
        happens once per block, in float.
      </p>
    ),
    source: `${FACTS} A2 (dp4a's overloads), D3 (vec_dot_q4_0_q8_1, quantize_q8_1)`,
  },
  'x2/X2.5/q-prescale-exact': {
    kind: 'choice',
    id: 'x2/X2.5/q-prescale-exact',
    prompt: <>Path B divides activations by 16, 256 and 4096 so that masked, unshifted nibbles give the right products. Why does that division lose nothing in float32?</>,
    options: [
      { label: 'Powers of two change only the exponent, so the quotient is exact unless it falls below the bottom of float’s normal range.', correct: true },
      { label: 'Because the mask multiplies the value back by the same factor.', why: 'The mask does supply the factor, but that would not undo a rounding made at the division. There is none to undo.' },
      { label: 'Because activations need fewer than 24 bits.', why: 'They do not; a float32 activation can use all 24.' },
      { label: 'It loses a little, inside the tests’ tolerance.', why: 'It loses nothing for any activation of ordinary size: the unit test checks every one it generates.' },
    ],
    explain: (
      <p>
        x / 4096 has the same 24-bit significand as x and an exponent 12 smaller. Only a value already near
        float's smallest normal (about 1.2e−38) would run out of exponent. So code × (x / 16<sup>k</sup>) × 16<sup>k</sup>,
        which is what a masked nibble times a pre-divided activation computes, equals code × x to the bit, before the
        accumulator rounds it. In half precision the same trick is not safe: section X2.6 shows why.
      </p>
    ),
    source: `${FACTS} E4, D4`,
  },
  'x2/X2.5/q-hip-sdot4': {
    kind: 'choice',
    id: 'x2/X2.5/q-hip-sdot4',
    prompt: <>You build a HIP kernel for an RDNA3 GPU with a helper that calls <code>__builtin_amdgcn_sdot4</code>. What happens?</>,
    options: [
      { label: 'A compile error: sdot4 needs the dot1-insts feature, which RDNA3 and RDNA4 do not have. They have the mixed-sign sudot4, which llama.cpp calls with both operands marked signed.', correct: true },
      { label: 'It compiles to four scalar multiplies: correct, just slower.', why: 'Calling a builtin on a target without its feature is a compile error, not a slow path; clang’s own test says "needs target feature dot1-insts".' },
      { label: 'It works, because HIP maps it to __dp4a.', why: 'HIP has no __dp4a at all.' },
      { label: 'It compiles, but gives wrong answers for unsigned nibbles.', why: 'The problem comes earlier: it does not compile for that target.' },
    ],
    explain: (
      <p>
        The compiler's target description gives the signed 8-bit four-way dot (dot1) to gfx906, CDNA and RDNA2, and the
        mixed-sign form (dot8) to RDNA3 and RDNA4. llama.cpp's <code>ggml_cuda_dp4a</code> chooses with an{' '}
        <code>#if</code> on the architecture; gfx900 and RDNA1 get inline assembly.
      </p>
    ),
    source: `${FACTS} B1, B2`,
  },
  'x2/X2.5/q-metal': {
    kind: 'choice',
    id: 'x2/X2.5/q-metal',
    prompt: <>Metal has no per-thread integer dot product. What do llama.cpp, MLX and PyTorch do in their Metal 4-bit matrix-vector kernels?</>,
    options: [
      { label: 'Stay in float: masked nibbles times activations pre-divided by 16, 256 and 4096, with the offset folded out through Σx.', correct: true },
      { label: 'Quantise the activations to int8 and multiply bytes in a loop.', why: 'None of them quantises the activations on Metal; that is llama.cpp’s CUDA and HIP path.' },
      { label: 'Call the matrix API with int4b_format operands.', why: 'Apple added 4-bit integer operands to its tensor API, but a code search found neither llama.cpp nor MLX using int4b_format.' },
      { label: 'Dequantise each weight to half and call dot.', why: 'That would accumulate in half (dot of half4 returns half). llama.cpp converts to float4 before dot.' },
    ],
    explain: (
      <p>
        All three read 16-bit words of codes, mask them without shifting, and divide the matching activations by the
        same powers of two first, so every product is code × activation exactly, and fold the offset once per block.
        It is path B in the stepper, and the idiom to copy on Metal.
      </p>
    ),
    source: `${FACTS} D4, D5, D6, C1, C2`,
  },
};

// ---------------------------------------------------------------- the page
export function SX25() {
  const dot4 = helperOf(loaded(useLabFile('x2/lab3_qhunt/solution/kernel.cu')));
  return (
    <>
      <p>
        A quantised GEMV (section X2.4) spends its inner loop on one block's dot product: 32 four-bit codes, one
        scale, 32 activations. There are two families of ways to compute it. Dequantise the codes to floats and
        multiply in float, or turn the activations into small integers too and multiply in integers. Which one a
        kernel uses depends on the backend, and both have a trick that makes them cheaper than they look.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          Module X2 is ours: the book has no integer formats. Everything said here about llama.cpp, MLX, PyTorch and the
          vendors was checked in the facts pass on 2026-11-20 ({FACTS}).
        </p>
      </Beyond>
      <p>
        One thing to know before the details. When llama.cpp decodes a Q4_0 model on a CUDA or ROCm GPU, your
        activation vector is itself quantised to int8, 32 values at a time, for every token, so that the weights can
        be multiplied as bytes. On Apple GPUs the same model stays in float.
      </p>
      <Today checked={`${CHECKED}, D3, D4)`}>
        <p>
          llama.cpp's CUDA matrix-vector path calls <code>quantize_row_q8_1_cuda</code> on the activations before its
          kernel; the HIP build compiles the same source. Its Metal decode kernels never quantise the activations.
        </p>
      </Today>

      <h2>Three ways to one number</h2>
      <p>
        <b>Path A</b> is the definition: turn every code into its weight, (q − 8)·d, then multiply-add with the
        activation. It is correct and it is wasteful: a subtraction, a conversion and a multiplication by d for every
        weight, before the multiply that does the work.
      </p>
      <p>
        <b>Path B</b> moves everything constant out of the loop. Because Σ d·(q − 8)·x = d·(Σ q·x − 8·Σ x), the loop
        only needs codes times activations; the 8 is subtracted once, through Σ x, and d is applied once. Σ x depends
        only on the activations, so it is computed once per block per token and shared by every row of the weight
        matrix. A second trick removes the shifts: reading two bytes as one 16-bit word and masking it four ways leaves
        each nibble multiplied by 1, 16, 256 or 4096. Divide the matching activation by the same power of two in
        advance and the product is code × activation, exactly.
      </p>
      <p>
        <b>Path C</b> is llama.cpp's CUDA and HIP decode. First the 32 activations become a Q8_1 block: a 32-wide max,
        d = max / 127, every activation rounded to a signed byte, and s = d × (sum of the bytes). Then the weight codes,
        still unsigned, meet the activation bytes in a four-way byte dot product (dp4a), eight times per block, into an
        exact integer. One correction and one rescale per block turn it back into a float. The max and the sum are
        chapter 10's warp reduction, applied to groups of 32 whatever the hardware's warp width.
      </p>
      <Source section="10.7" pages="237–241" />
      <Today checked={`${CHECKED}, D3 to D6)`}>
        <p>
          Path B is what llama.cpp's <code>mul_mv</code> kernels, MLX's <code>qmv</code> and PyTorch's MPS{' '}
          <code>int4pack_mm</code> do: activations pre-divided by 16, 256 and 4096, offset folded through the sum of the
          activations. Path C is llama.cpp's <code>vec_dot_q4_0_q8_1</code>.
        </p>
      </Today>
      <p>
        The stepper runs all three on the same block. Watch two things as you go: the running value, and the counts
        of work per row against work per token.
      </p>
      <Simplification>
        <p>
          Operation counts are source-level, not instructions: an FMA counts as one multiply and one add, a dp4a as one
          integer operation. There is no timing. Lab X2.3 measures the real thing on your device.
        </p>
      </Simplification>
      <ThreePaths />
      <Predict
        id="x2/X2.5/p-bitwise"
        prompt={
          <>
            Paths A and B are equal in exact arithmetic. Over 40 seeded blocks computed in float32, in how many do they
            give the same bits?
          </>
        }
        unit="of 40"
      >
        {(v) => <BitwiseReveal predicted={v} />}
      </Predict>
      <Ask q={questionsSX25['x2/X2.5/q-fold']} />
      <Ask q={questionsSX25['x2/X2.5/q-prescale-exact']} />

      <h2>What the integer path costs</h2>
      <p>
        Path C buys exact integer arithmetic for 32 products at the price of three things, and you should be able to
        name them before reading on. An extra pass over the activations for every token. The rounding of every
        activation to one of 255 levels, set by the largest one in its block. And the FP16 rounding of s, the stored
        sum that the correction uses.
      </p>
      <Predict
        id="x2/X2.5/p-act-error"
        prompt={
          <>
            For a 4,096-wide row with standard-normal weights and activations, how large do you expect path C's error
            against path B to be, as a percentage of the sum of |w·x|? (Path B rounds only in float32.)
          </>
        }
        unit="%"
      >
        {(v) => <ActErrorReveal predicted={v} />}
      </Predict>
      <ActivationCost />
      <p>
        A large activation sets its block's scale. Every other activation in the block is then rounded to a coarser
        grid, so the block's whole contribution gets worse, not just the large term. The explorer's second
        distribution puts one large channel in every 128, and the error rises accordingly. Next to it is the error that
        quantising the weights to 4 bits already introduced: the activation rounding is the smaller of the two here.
      </p>
      <Ask q={questionsSX25['x2/X2.5/q-why-not-int-row']} />
      <Ask q={questionsSX25['x2/X2.5/q-unsigned']} />
      <IntegerPathScene />

      <h2>Prefill dequantises tiles</h2>
      <p>
        Everything above is decode: one token, a matrix-vector product, bandwidth-bound. For prefill, many tokens at
        once, llama.cpp's Metal <code>mul_mm</code> dequantises each tile of weights into threadgroup memory and then
        multiplies tiles in float with SIMD-group matrices. That is path A again, but at tile granularity: each
        dequantised weight is reused by every token in the tile, so its cost is spread thin. It is the shape lab
        15.6's tiled GEMM would take if its weight tiles were loaded from Q4_0 blocks.
      </p>
      <Aside>
        <p>
          The tiled GEMM is chapter 15 (book pp. 349–370) and <a href={href('ch15', 'labs')}>lab 15.6</a>. llama.cpp's
          tile path: {FACTS} D4.
        </p>
      </Aside>
      <Beyond title="Not used here: integer matrix instructions">
        <p>
          CUDA's <code>mma</code> has s8 and s4 forms from sm_75 and sm_80, Hopper's <code>wgmma</code> s8 only, and
          Blackwell's <code>tcgen05</code> <code>kind::i8</code> only on specific targets. AMD has i8 MFMA on CDNA and
          iu8/iu4 WMMA on RDNA3 and RDNA4. Metal's tensor API multiplies char and int4b_format operands but accepts only
          power-of-two (UE8M0) block scales. None of them is in the labs. ({FACTS} A4, B3, C3.)
        </p>
      </Beyond>

      <h2>What each backend offers for a byte dot product</h2>
      <p>
        Here the portability story of the course gets a new entry. The list of things one <code>.cu</code> file does
        not do unchanged for HIP already had warp masks, cooperative groups, half-precision headers and matrix
        intrinsics. Add the per-thread integer dot product: CUDA's <code>__dp4a</code> does not exist under HIP, and
        the AMD builtins that replace it depend on the GPU generation.
      </p>
      <Today checked={`${CHECKED}, A2, A3, B1, B2, B4, C1)`}>
        <p>
          AMD's builtins are documented only in the compiler's target description; no prose page was found for them or
          for <code>amd_mixed_dot</code>. Nothing in the table was run on CUDA or AMD hardware.
        </p>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-backends="">
        <thead>
          <tr><th>Per thread</th><th>CUDA</th><th>HIP (ROCm)</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Four-way int8 dot, 32-bit accumulate</td>
            <td><code>__dp4a</code>, from sm_61; same-sign overloads only (PTX also has mixed sign)</td>
            <td>No <code>__dp4a</code>. <code>__builtin_amdgcn_sdot4</code> on gfx906, CDNA, RDNA2; <code>__builtin_amdgcn_sudot4</code> on RDNA3 and RDNA4; the wrong one does not compile. The header's <code>amd_mixed_dot</code> is undocumented</td>
            <td>None: MSL's <code>dot</code> is for float and half vectors</td>
          </tr>
          <tr>
            <td>Half dot with float accumulate</td>
            <td>None (<code>__hfma2</code> accumulates in half)</td>
            <td><code>amd_mixed_dot(__half2, __half2, float, bool)</code></td>
            <td>None: <code>dot(half4, half4)</code> returns half</td>
          </tr>
          <tr>
            <td>FP8 dot</td>
            <td>None</td>
            <td><code>__builtin_amdgcn_dot4_f32_fp8_fp8</code> and variants, gfx11.7 and gfx12</td>
            <td>None</td>
          </tr>
        </tbody>
      </table>
      <Ask q={questionsSX25['x2/X2.5/q-hip-sdot4']} />
      <Ask q={questionsSX25['x2/X2.5/q-metal']} />
      <p>
        So a lab that wants a byte dot product in one source file has two choices. A plain helper with four byte
        products, which compiles everywhere and is what the emulators run. Or an <code>#if</code> ladder like
        llama.cpp's <code>ggml_cuda_dp4a</code>: under HIP, <code>__builtin_amdgcn_sdot4(a, b, c, false)</code> on
        CDNA, RDNA2 and gfx906, <code>__builtin_amdgcn_sudot4(true, a, true, b, c, false)</code> on RDNA3 and RDNA4,
        inline assembly on RDNA1 and gfx900, four byte products otherwise; on CUDA, <code>__dp4a</code> from sm_61 and
        four byte products below it. Lab X2.3's helper is the CUDA half of that ladder.
      </p>
      <Today checked={`${CHECKED}, B1)`}>
        <p>
          The ladder is read from llama.cpp's <code>ggml-cuda/common.cuh</code>. Its AMD branches are compiler builtins
          and inline assembly; none of them is in the course's labs.
        </p>
      </Today>
      {dot4 ? (
        <Code
          tabs={[{ label: 'dot4_i8', code: dot4, status: 'four-product branch: emulator-checked; __dp4a branch: unverified' }]}
          caption={
            <>
              From the reference solution of lab X2.3, <code>labs/x2/lab3_qhunt/solution/kernel.cu</code>. The HIP build
              and the emulators take the four-product branch. The <code>__dp4a</code> branch is unverified: the
              course's CUDA front-end check compiles the host side only (<code>--cuda-host-only</code>), so code under{' '}
              <code>__CUDA_ARCH__</code> is never parsed by it, and no CUDA device has run it yet.
            </>
          }
        />
      ) : (
        <p className="muted" data-dot4-missing="">
          The helper is in lab X2.3's reference solution, <code>labs/x2/lab3_qhunt/solution/kernel.cu</code>, which
          could not be read from this checkout.
        </p>
      )}
      <p>
        On your own GPU, lab X2.3 (<a href={href('x2', 'labs')}>labs page</a>) puts path C into a whole GEMV and hides
        defects in it for you to find.
      </p>
    </>
  );
}
