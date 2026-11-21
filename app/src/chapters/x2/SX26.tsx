import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BOUND_Q4_SIGNED,
  BOUND_Q4_UNSIGNED,
  BOUND_Q8Q8,
  FRAGMENTS,
  INNER,
  INNER_LIMIT,
  LADDER_LENGTHS,
  OUTER,
  PAIRING,
  accumulate,
  int32Terms,
  parseFragments,
  runFragment,
  sci,
  show,
  type DataSet,
  type FragmentId,
  type FragmentRun,
  type Inner,
  type Outer,
  type Pairing,
} from '../../sim/qdot';
import { NarrowAccumulatorScene } from './NarrowAccumulatorScene';

// Section X2.6: how wide each level of a quantised dot product's sum must be. Every number is
// from src/sim/qdot.ts (bounds, innerSum, accumulate, runFragment), pinned in qdot.test.ts.
// Module X2 is ours; the anchor is appendix A, §A.6 (and §A.2 for FP16's range).
const FACTS = 'notes/today/x2-facts.md';
const CHECKED = `2026-11-20 (${FACTS}`;

// ---------------------------------------------------------------- the ladder
function Ladder() {
  const [inner, setInner] = useState<Inner>('int16');
  const [outer, setOuter] = useState<Outer>('fp32');
  const [data, setData] = useState<DataSet>('worst');
  const [k, setK] = useState(7); // 4,096
  const [pairing, setPairing] = useState<Pairing>('q8q8');
  const [step, setStep] = useState<1 | 4>(1);
  const n = LADDER_LENGTHS[k];
  const r = useMemo(() => accumulate({ inner, outer, data, n, pairing, step }), [inner, outer, data, n, pairing, step]);
  const tr = r.block0;
  const lim = INNER_LIMIT[inner];
  const isInt = inner === 'int16' || inner === 'int32';
  // block 0's running sum, |value| on a log axis from 1 to 1e10
  const gx = 56, gy = 16, gw = 600, gh = 170;
  const LO = 0, HI = 10;
  const yl = (v: number) => gy + gh - ((Math.log10(Math.max(1, Math.abs(v))) - LO) / (HI - LO)) * gh;
  const xs = (i: number) => gx + ((i + 0.5) / tr.values.length) * gw;
  const bad = (i: number) =>
    (tr.first.wrap !== undefined && i + 1 >= tr.first.wrap) || (tr.first.inf !== undefined && i + 1 >= tr.first.inf);
  const status = r.overflowed ? 'inf' : r.wrapped ? 'wrapped' : r.rounded ? 'rounded' : 'exact';
  return (
    <Bench
      title="The accumulator ladder"
      task="Start with int16 on the worst case, then climb: int32, FP16, FP32. Then put FP16 on the outer sum and drag the row length."
      foot={
        <>
          One row of n weights and n activations, coded in blocks of 32 as llama.cpp does: per block an integer sum of
          32 products (the inner accumulator), turned into a float with the block's scales, and the blocks added up (the
          outer accumulator). Realistic: seeded standard-normal weights and activations. Worst case: every code at its
          extreme (127, or 15 for the unsigned nibbles), with the realistic scales. The reference is the same formula
          in binary64 on exact integer sums.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field">
          Inner accumulator
          <select aria-label="Inner accumulator" value={inner} onChange={(e) => setInner(e.target.value as Inner)}>
            {INNER.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
          </select>
        </label>
        <label className="field">
          Outer accumulator
          <select aria-label="Outer accumulator" value={outer} onChange={(e) => setOuter(e.target.value as Outer)}>
            {OUTER.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
          </select>
        </label>
        <label className="field">
          Pairing
          <select aria-label="Pairing" value={pairing} onChange={(e) => setPairing(e.target.value as Pairing)}>
            {PAIRING.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
          </select>
        </label>
        <label className="field">
          Integer step
          <select aria-label="Integer step" value={step} onChange={(e) => setStep(+e.target.value as 1 | 4)} disabled={!isInt}>
            <option value={1}>one product at a time</option>
            <option value={4}>four at a time (dp4a)</option>
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Data">
          {(['realistic', 'worst'] as DataSet[]).map((d) => (
            <button key={d} aria-pressed={data === d} onClick={() => setData(d)}>{d === 'worst' ? 'worst case' : 'realistic'}</button>
          ))}
        </span>
        <label className="field" style={{ minWidth: '14rem' }}>
          Row length: {n.toLocaleString('en-US')} ({n / 32} blocks)
          <input type="range" aria-label="Row length" min={0} max={LADDER_LENGTHS.length - 1} value={k} onChange={(e) => setK(+e.target.value)} />
        </label>
      </div>
      <svg viewBox="0 0 700 240" role="img" aria-label="The first block's running inner sum against the accumulator's limits" style={{ width: '100%', maxWidth: 700, display: 'block', minWidth: 540 }}>
        <rect x={gx} y={gy} width={gw} height={gh} fill="var(--cell)" stroke="var(--rule)" />
        {[0, 2, 4, 6, 8, 10].map((e) => (
          <g key={e}>
            <line x1={gx} x2={gx + gw} y1={yl(10 ** e)} y2={yl(10 ** e)} stroke="var(--rule-soft)" />
            <text x={gx - 6} y={yl(10 ** e) + 3} fontSize="10" textAnchor="end">{e === 0 ? '1' : `1e${e}`}</text>
          </g>
        ))}
        {lim.max < 1e10 ? (
          <g>
            <line x1={gx} x2={gx + gw} y1={yl(lim.max)} y2={yl(lim.max)} stroke="var(--fail)" strokeWidth="1.5" />
            <text x={gx + gw - 4} y={yl(lim.max) - 4} fontSize="10" textAnchor="end" style={{ fill: 'var(--fail)' }}>largest {INNER.find((t) => t.id === inner)!.label}: {show(lim.max)}</text>
          </g>
        ) : null}
        {!isInt && lim.exactTo < 1e10 ? (
          <g>
            <line x1={gx} x2={gx + gw} y1={yl(lim.exactTo)} y2={yl(lim.exactTo)} stroke="var(--ink-3)" strokeDasharray="4 3" />
            <text x={gx + 4} y={yl(lim.exactTo) - 4} fontSize="10" style={{ fill: 'var(--ink-2)' }}>every integer exact up to {show(lim.exactTo)}</text>
          </g>
        ) : null}
        <path d={tr.exact.map((v, i) => `${i ? 'L' : 'M'}${xs(i)},${yl(v)}`).join(' ')} fill="none" stroke="var(--ink-3)" strokeWidth="1.5" />
        {tr.values.map((v, i) => (
          <g key={i}>
            {Number.isFinite(v) ? (
              <circle cx={xs(i)} cy={yl(v)} r={3.2} fill={bad(i) ? 'var(--fail)' : v === tr.exact[i] ? 'var(--reg)' : 'var(--page)'}
                stroke={bad(i) ? 'var(--fail)' : 'var(--reg)'} />
            ) : (
              <text x={xs(i)} y={gy + 10} fontSize="10" textAnchor="middle" style={{ fill: 'var(--fail)' }}>∞</text>
            )}
          </g>
        ))}
        <text x={gx} y={gy + gh + 16} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
          block 1's running sum, step by step ({tr.values.length} steps of {tr.step} product{tr.step > 1 ? 's' : ''}).
        </text>
        <text x={gx} y={gy + gh + 30} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
          Grey line: exact sum. Filled dot: exact; open: rounded; red: wrapped or overflowed.
        </text>
        <text x={gx} y={gy + gh + 48} fontSize="11" data-block0={String(tr.values[tr.values.length - 1])}>
          block 1 ends at {show(tr.values[tr.values.length - 1])} (exact {show(tr.exact[tr.exact.length - 1])})
          {tr.first.wrap ? `; wrapped at step ${tr.first.wrap}` : ''}
          {tr.first.inf ? `; infinity at step ${tr.first.inf}` : ''}
          {!tr.first.inf && tr.first.inexact ? `; first rounding at step ${tr.first.inexact}` : ''}
        </text>
      </svg>
      <div className="row" style={{ marginTop: '0.6rem' }} data-ladder="" data-status={status} data-relerr={String(r.relErr)}>
        <span>
          Inner sums over {n / 32} blocks: {r.wrapped} wrapped, {r.overflowed} overflowed, {r.rounded} rounded.
        </span>
        <span>
          Row result {show(r.result, 8)} against {show(r.reference, 8)}: relative error{' '}
          <b className={Number.isFinite(r.relErr) && r.relErr < 1e-6 ? 't-ok' : 't-fail'}>{Number.isFinite(r.relErr) ? sci(r.relErr) : 'overflow'}</b>
          {Number.isFinite(r.absErr) ? <> (absolute {sci(r.absErr)})</> : null}
        </span>
        <span className={'chip ' + (status === 'exact' ? 'ok' : status === 'rounded' ? '' : 'fail')} data-chip="">{status === 'exact' ? 'inner sums exact' : status === 'rounded' ? 'inner sums rounded' : status === 'wrapped' ? 'inner sums wrapped' : 'inner sums overflowed'}</span>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the bug spot
/** The four fragments are a lesson snippet, labs/x2/snippets/accumulators.md, tied to FRAGMENT_MODEL by qdot.snippets.test.ts. */
const FRAG_FILE = 'x2/snippets/accumulators.md';

function reason(r: FragmentRun, base: FragmentRun): string {
  const ratio = r.realistic / base.realistic;
  switch (r.id) {
    case 'metal-half-dot':
      return `dot(half4, half4) returns half, so every four products are summed and rounded in half. On the realistic row its error is ${sci(r.realistic)}, about ${Math.round(ratio)} times the integer-then-float fragment's ${sci(base.realistic)}. It does not overflow on this worst case (four products of 889), but four products past 65,504 would. Converting both operands to float4 before dot costs nothing.`;
    case 'cuda-hfma2':
      return `Each half lane accumulates half the row. On the realistic row the error is ${sci(r.realistic)}, about ${Math.round(ratio).toLocaleString('en-US')} times the safe fragment's; on the worst-case row a lane passes 65,504 at product ${r.worst.at} and the result is infinity.`;
    case 'short-sum':
      return `On the realistic row it gives exactly what fragment 4 gives (${sci(r.realistic)}): the tests you would write first pass. On the worst-case block (codes 15 against 127, at most 60,960) the short wraps on dp4a step ${r.worst.at} of 8, and the row comes out as ${show(r.worst.result)} instead of ${show(r.worst.reference)}.`;
    case 'int-then-float':
      return `A block sum is at most 60,960, which an int32 holds with room to spare, and float holds each block's result and the row's: the worst-case row is exactly ${show(r.worst.result)}, and the realistic error is ${sci(r.realistic)}, float rounding.`;
  }
}

function BugSpot() {
  const runs = useMemo(() => Object.fromEntries(FRAGMENTS.map((id) => [id, runFragment(id)])) as Record<FragmentId, FragmentRun>, []);
  const md = useLabFile(FRAG_FILE);
  const texts = useMemo(() => (md && !md.startsWith('// could not load') ? parseFragments(md) : []), [md]);
  const [marks, setMarks] = useState<Partial<Record<FragmentId, boolean>>>({});
  const [checked, setChecked] = useState(false);
  const all = FRAGMENTS.every((id) => marks[id] !== undefined);
  const right = FRAGMENTS.filter((id) => marks[id] === runs[id].safe).length;
  return (
    <Bench
      title="Safe or unsafe?"
      task="Mark each fragment, then check. Every verdict comes with what the model measured."
      foot={
        <>
          Fragments from <code>labs/{FRAG_FILE}</code>: they show the accumulator, nothing else. Fragments 2 to 4 are
          emulator-checked and front-end-checked (a unit test runs them, as printed, through{' '}
          <code>labs/x2/snippets/check/accumulators_check.cu</code> and matches the model's results), unverified on
          CUDA; fragment 1, in MSL, is not compiled (unverified). q, x4, w2, x2, v and u are the
          block's codes and activations; dot4 is dp4a's arithmetic. Their arithmetic is modelled in qdot.ts on a
          4,096-wide row: realistic seeded data, and a worst case of codes 15 against 127 with unit scales (so the float
          fragments see weights of 7 and activations of 127). The dot of half4 is modelled as a chain of half
          multiply-adds; Metal does not document its order.
        </>
      }
    >
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(19rem, 1fr))' }}>
        {FRAGMENTS.map((id) => {
          const f = texts.find((t) => t.id === id);
          const r = runs[id];
          const ok = marks[id] === r.safe;
          return (
            <div key={id} className="panel" data-frag={id}>
              <div style={{ fontWeight: 600, marginBottom: '0.3rem' }}>
                {f ? f.title : `Fragment ${FRAGMENTS.indexOf(id) + 1}`} {f ? <span className="muted">({f.lang})</span> : null}
              </div>
              {f ? (
                <pre className="mono" style={{ fontSize: '0.74rem', margin: '0 0 0.5rem', overflowX: 'auto' }} data-frag-code="">{f.code}</pre>
              ) : (
                <p className="muted" style={{ margin: '0 0 0.5rem' }}>{md ? `Could not read labs/${FRAG_FILE}.` : 'Loading the fragment.'}</p>
              )}
              <div className="row">
                {[true, false].map((safe) => (
                  <label key={String(safe)} className="check">
                    <input type="radio" name={'frag-' + id} checked={marks[id] === safe}
                      onChange={() => { setMarks({ ...marks, [id]: safe }); setChecked(false); }} />
                    {safe ? 'safe' : 'unsafe'}
                  </label>
                ))}
              </div>
              {checked ? (
                <p style={{ margin: '0.5rem 0 0' }} data-verdict={r.safe ? 'safe' : 'unsafe'}>
                  <span className={'chip ' + (ok ? 'ok' : 'fail')}>{ok ? 'right' : 'not quite'}: {r.safe ? 'safe' : 'unsafe'}</span>{' '}
                  {reason(r, runs['int-then-float'])}
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" onClick={() => setChecked(true)} disabled={!all}>Check</button>
        {checked ? <span data-score={right}>{right} of 4 right.</span> : <span className="muted">Mark all four first.</span>}
      </div>
    </Bench>
  );
}

function BoundReveal({ predicted }: { predicted: number }) {
  return (
    <>
      <p data-bound-reveal="">
        It is <b>{BOUND_Q8Q8.toLocaleString('en-US')}</b>: 32 products, each at most 127 × 127 = 16,129
        {predicted === BOUND_Q8Q8 ? ', exactly your number' : ''}. Does it fit in FP16? No, twice over. FP16's largest
        finite value is 65,504, so the sum overflows; and FP16 holds every integer only up to 2,048, so it was
        already rounding long before that. An int32 holds it with four thousand times room to spare, and a float holds
        it exactly, because it is below 2<sup>24</sup>.
      </p>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSX26: Record<string, Question> = {
  'x2/X2.6/q-q8-bound': {
    kind: 'numeric',
    id: 'x2/X2.6/q-q8-bound',
    prompt: <>What is the largest magnitude the integer sum of one Q8_0 × Q8_1 block (32 int8 codes times 32 int8 codes, each in −127..127) can reach?</>,
    answer: 516128,
    explain: <p>32 × 127 × 127 = 516,128. Past FP16's 65,504; far inside int32; below 2<sup>24</sup>, so exact in float.</p>,
    source: `${FACTS} E3 (our arithmetic)`,
  },
  'x2/X2.6/q-int32-terms': {
    kind: 'numeric',
    id: 'x2/X2.6/q-int32-terms',
    prompt: (
      <>
        An int32 accumulator adds worst-case int8 products, each 127 × 127 = 16,129. How many can it add before the
        next one overflows?
      </>
    ),
    answer: int32Terms(16129),
    explain: (
      <p>
        (2<sup>31</sup> − 1) / 16,129 = 133,144.6, so 133,144 products fit and the 133,145th overflows. No real kernel
        gets near it, and not because rows are short: every block of 32 has its own scales, so the integer sum is
        rescaled into float after 32 products at most.
      </p>
    ),
    source: `${FACTS} E3 ("about 133,144 worst-case terms"; the exact count is ours)`,
  },
  'x2/X2.6/q-nibble-int16': {
    kind: 'choice',
    id: 'x2/X2.6/q-nibble-int16',
    prompt: <>A kernel sums one Q4_0 block's unsigned nibbles (0..15) times int8 activations, 32 products, in a <code>short</code>, four at a time with dp4a. Is that safe?</>,
    options: [
      { label: `No: the sum can reach ${BOUND_Q4_UNSIGNED.toLocaleString('en-US')}, past 32,767.`, correct: true },
      { label: 'Yes: 15 × 127 = 1,905 per product, far below 32,767.', why: 'One product is; 32 of them are not.' },
      { label: 'Yes, as long as the codes go in signed, as −8..7.', why: `Signed codes reach ${BOUND_Q4_SIGNED.toLocaleString('en-US')}, which fits with 255 to spare. That is a coincidence of 32 × 8 × 127, not a margin to build on, and llama.cpp feeds the unsigned codes.` },
      { label: 'No, but only because dp4a returns 32 bits.', why: 'dp4a’s 32-bit result is fine; storing it into 16 bits is what loses the top.' },
    ],
    explain: (
      <p>
        Each dp4a adds up to 4 × 15 × 127 = 7,620: 30,480 after four steps, 38,100 after five. The ladder above shows it with the Q4_0
        pairing, the worst case and four products per step. On realistic data it passes, which is what makes it a
        good planted defect: lab X2.3 has it.
      </p>
    ),
    source: `${FACTS} E3; lab X2.3`,
  },
  'x2/X2.6/q-half-prescale': {
    kind: 'choice',
    id: 'x2/X2.6/q-half-prescale',
    prompt: <>To save registers, a Metal kernel stores its pre-divided activations (x / 16, x / 256, x / 4096) in half. An activation of 1e−4 meets the 4096 divisor. What happens?</>,
    options: [
      { label: 'It rounds to 0, below half the smallest FP16 subnormal.', correct: true },
      { label: 'Nothing: dividing by a power of two is exact.', why: 'Exact only while the result is representable. FP16 runs out of exponent at 2^−24.' },
      { label: 'It becomes an FP16 subnormal with only a few significand bits.', why: 'It is below half the smallest subnormal, so there is nothing left to round to but 0.' },
      { label: 'It overflows once the masked nibble scales it back up.', why: 'Small values underflow; it is the large ones that overflow.' },
    ],
    explain: (
      <p>
        1e−4 / 4096 is about 2.4e−8. FP16's smallest positive value is 2<sup>−24</sup> ≈ 5.96e−8, and anything below
        half of it rounds to 0, so that weight's contribution vanishes. In float the same quotient is an ordinary normal
        number: keep pre-divided activations in float. The pre-division trick is exact in float32 (section X2.5)
        precisely because float has exponent range to spare; half does not.
      </p>
    ),
    source: `notes/today/appA-facts.md A2 (FP16 smallest denormal 2^-24); ${FACTS} E4`,
  },
  'x2/X2.6/q-metal-dot': {
    kind: 'choice',
    id: 'x2/X2.6/q-metal-dot',
    prompt: <>A Metal kernel dequantises four weights to <code>half4 w</code>, loads <code>half4 x</code>, and adds <code>dot(w, x)</code> to a float. What does the dot accumulate in, and what is the fix?</>,
    options: [
      { label: 'In half; convert both to float4 before dot.', correct: true },
      { label: 'In float, because the result is added to a float.', why: 'The conversion to float happens after the dot has already been computed in half.' },
      { label: 'In float: Metal always widens dot products.', why: 'The spec says the dot of a halfn has the scalar type half.' },
      { label: 'In half; make the row accumulator half too.', why: 'That makes it worse: fragment 2 of the bug spot.' },
    ],
    explain: (
      <p>
        The MSL specification defines <code>dot</code> for floatn and halfn, returning the scalar type of its arguments,
        so the dot of a halfn is half.
        llama.cpp's <code>mul_mv</code> writes <code>dot(float4(xb4[i]), float4(yl4[i]))</code>.
      </p>
    ),
    source: `${FACTS} C1`,
  },
  'x2/X2.6/q-order': {
    kind: 'choice',
    id: 'x2/X2.6/q-order',
    prompt: <>Lab X2.2's kernel gives outputs that differ in the last digits between two backends. Is that a bug?</>,
    options: [
      { label: 'Not necessarily: the float sums may run in different orders.', correct: true },
      { label: 'Yes: integer arithmetic should make results identical.', why: 'Only the sums inside a block are integers; everything across blocks is float, added in whatever order each kernel chose.' },
      { label: 'Probably: one backend is likely accumulating in half somewhere.', why: 'Possible, but different orders of float additions are enough to explain a last-digit difference.' },
      { label: 'No: GPU float arithmetic varies from run to run, so ignore it.', why: 'A given kernel on a given device is usually repeatable; the difference comes from the order of the additions, which is a property of the code.' },
    ],
    explain: (
      <p>
        Appendix A's point, from §A.6: float addition is not associative, so order decides the low bits of a long sum. A 32-lane warp and a different lane
        count, a different split of the row, or a tree instead of a chain all reorder it. A test should allow an error
        scaled by the sum of the magnitudes of the terms, and should fail on anything larger.
      </p>
    ),
    source: 'book appendix A §A.6, pp. 593–594',
  },
};

// ---------------------------------------------------------------- the page
export function SX26() {
  return (
    <>
      <p>
        Appendix A made one point about long sums: their order and their width decide how much of the answer
        survives. A half accumulator holds integers exactly only to 2,048 and overflows at 65,504; lab A.1 measured
        what that does to a long sum on your own device. A quantised dot product is a sum on two levels, inside each
        block and across the blocks, and each level needs its own answer to "how wide?".
      </p>
      <Source section="A.6" pages="593–594" />
      <Beyond title="Ours, not the book's">
        <p>
          The appendix has no integer formats. The bounds here are our arithmetic; what the kernels accumulate in was
          checked in the facts pass on 2026-11-20 ({FACTS} E3, C1, D3 to D5). FP16's limits: appendix A, §A.2, and
          its facts file, A2. Lab A.1 is on the <a href={href('appA', 'labs')}>appendix's labs page</a>.
        </p>
      </Beyond>

      <h2>Bound every partial sum</h2>
      <p>
        Start inside a block. With Q8_0 weights and Q8_1 activations, every product is at most 127 × 127 in magnitude,
        and a block has 32 of them. Before you read on, work out the largest sum.
      </p>
      <Predict id="x2/X2.6/p-q8-bound" prompt={<>The largest magnitude of one Q8_0 × Q8_1 block's integer sum?</>}>
        {(v) => <BoundReveal predicted={v} />}
      </Predict>
      <p>
        The 4-bit case depends on how the codes go in. llama.cpp's CUDA kernel feeds the unsigned nibbles, 0 to 15,
        and corrects for the 8 later: up to 15 × 127 × 32 = 60,960 per block, more than a 16-bit integer holds. Signed
        codes, −8 to 7, reach 8 × 127 × 32 = 32,512, which fits a 16-bit integer by 255. And an int32 would take
        133,144 worst-case int8 products before overflowing. No kernel adds that many without rescaling, because every
        block has its own scale.
      </p>
      <Ask q={questionsSX26['x2/X2.6/q-q8-bound']} label="Check the bound" />
      <Ask q={questionsSX26['x2/X2.6/q-int32-terms']} label="Work it out" />
      <Ask q={questionsSX26['x2/X2.6/q-nibble-int16']} />

      <h2>Watch the narrow choices fail</h2>
      <p>
        The ladder sums a whole row on both levels with the accumulators you choose. The plot shows the first block's
        inner sum step by step, on a log scale, against the largest value the inner type holds. Below it is what
        happened to every block and to the row.
      </p>
      <Simplification>
        <p>
          The worst case is constructed: realistic data rarely gets near the bounds. But a kernel has to be right for
          every input it can be given, and the worst case is one of them. The model has no lanes and no timing.
        </p>
      </Simplification>
      <Ladder />
      <p>
        Four things to find. int16 wraps inside one worst-case block, and with Q8 codes it wraps on a few realistic
        blocks too. FP16 rounds from the very first product (16,129 is not a half) and reaches infinity on the fifth.
        With int32 inside, an FP16 outer sum drifts as you drag the row longer: its absolute error grows with the row.
        int32 then FP32 stays at float rounding at every length.
      </p>
      <p>
        The animation adds one worst-case block into all four types side by side, then sums a whole row across its
        blocks in half and in float.
      </p>
      <Simplification>
        <p>
          Every value comes from <code>qdot.ts</code>: one product at a time, in order, each add rounded as its type
          rounds. No lanes, no other orders, nothing timed.
        </p>
      </Simplification>
      <NarrowAccumulatorScene />

      <h2>The rule the real kernels follow</h2>
      <p>
        Integer inside a block, float across blocks. llama.cpp's CUDA decode keeps the block's dp4a sum in an{' '}
        <code>int</code>, turns each block into a float, and adds the blocks into float partial sums before the warp
        reduction. On the float path the accumulators are float too: llama.cpp's Metal kernels sum in float, and MLX's
        quantised matrix-vector kernel declares its accumulator type as float whatever the input type.
      </p>
      <Today checked={`${CHECKED}, D3, D4, D5)`}>
        <p>
          llama.cpp: <code>int sumi</code> per block and <code>float tmp[..]</code> across blocks in{' '}
          <code>mul_mat_vec_q</code>. MLX: <code>typedef float U;</code> in <code>qmv_fast_impl</code>.
        </p>
      </Today>
      <p>
        Where the trap is depends on the backend. On Metal, <code>dot</code> of two half4 vectors returns a half, so a
        kernel that dequantises to half and calls it has a half accumulator without saying so; converting to float4
        first, as llama.cpp does, removes it. CUDA has no half dot product at all, and a chain of{' '}
        <code>__hfma2</code> accumulates in half. HIP has one that accumulates in float,{' '}
        <code>amd_mixed_dot(__half2, __half2, float, bool)</code>, from an undocumented header.
      </p>
      <Today checked={`${CHECKED}, C1, A3, B4)`}>
        <p>
          MSL 4.1, section 6.9: for halfn arguments, dot returns half. The CUDA Math API's half2 page has no dot
          function. HIP's <code>amd_hip_fp16.h</code> defines the float-accumulating one.
        </p>
      </Today>
      <Ask q={questionsSX26['x2/X2.6/q-metal-dot']} />
      <p>
        ggml, llama.cpp's tensor library, goes one step further and makes the accumulator a property of the graph: an
        operation can be marked as needing F32 accumulation, or allowing BF16 or F16. It is the same choice you just
        made with the ladder, written down where every backend can read it.
      </p>
      <Today checked="2026-11-20 (notes/today/appA-facts.md, D3)">
        <p>
          <code>ggml_prec_set_acc</code> with <code>GGML_PREC_F32</code>, <code>_BF16</code> or <code>_F16</code>, in{' '}
          <code>ggml.h</code>; llama.cpp asks for F32 on flash attention.
        </p>
      </Today>

      <h2>Spot the narrow accumulator</h2>
      <p>
        Four fragments of the inner loop, the way you might find them in a kernel. Each one accumulates somewhere;
        decide which are safe for every input.
      </p>
      <BugSpot />

      <h2>Pre-divided activations stay float</h2>
      <p>
        Section X2.5's float path divides activations by 16, 256 and 4096 so that masked nibbles need no shift. That is
        exact in float32 because float has exponent range to spare. Half does not: its smallest positive value is
        about 6e−8, and a small activation divided by 4096 can fall below it.
      </p>
      <Ask q={questionsSX26['x2/X2.6/q-half-prescale']} />

      <h2>Where it shows up next</h2>
      <p>
        Lab X2.3 (<a href={href('x2', 'labs')}>labs page</a>) hides four defects in an integer-path GEMV. One of them is
        exactly this section: a block sum one type too narrow, which passes on realistic data and fails on the worst
        case. And when its outputs differ from another backend's in the last digits, the question below is the one to
        ask first.
      </p>
      <Ask q={questionsSX26['x2/X2.6/q-order']} />
    </>
  );
}
