import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { CROSSING, MLX_TIES, VOCAB_PRESETS, rowView, sortPlan, sortShare } from '../../sim/nucleus';
import { topP } from '../../sim/scan';

// Section X1.5: top-p over a sorted row. The explorer's numbers come from src/sim/nucleus.ts
// (rowView, sortPlan), the worked rows from src/sim/scan.ts's topP; both pinned in their tests.

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const n0 = (v: number) => v.toLocaleString('en-US');
/** Lab X1.2's directory, by the prefix pmpp.py matches. */
const LAB_X12 = 'x1/lab2';

// ---------------------------------------------------------------- the crossing row
function CrossingBars() {
  const r = topP(CROSSING.probs, CROSSING.p, 'exclusive');
  const W = 560;
  const X0 = 14;
  const sx = (v: number) => X0 + v * (W - 2 * X0);
  const crossing = r.before.findIndex((b, i) => b < CROSSING.p && r.upto[i] >= CROSSING.p);
  return (
    <svg viewBox={`0 0 ${W} 92`} role="img" aria-label="Each token's piece of the unit interval, and p" data-part="crossing"
      style={{ width: '100%', maxWidth: W, display: 'block', margin: '0.4rem 0 0.6rem' }}>
      {CROSSING.probs.map((v, i) => (
        <g key={i} data-token={i} data-keep={r.keep[i] ? '1' : '0'} data-before={r.before[i]}>
          <rect x={sx(r.before[i])} y={28} width={sx(r.before[i] + v) - sx(r.before[i])} height={24}
            fill={r.keep[i] ? 'var(--ink-3)' : 'var(--cell)'} fillOpacity={r.keep[i] ? 0.35 : 1}
            stroke={i === crossing ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={i === crossing ? 2.2 : 1} />
          <text x={(sx(r.before[i]) + sx(r.before[i] + v)) / 2} y={44} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{i}</text>
          <text x={sx(r.before[i])} y={68} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{r.before[i]}</text>
        </g>
      ))}
      <line x1={sx(CROSSING.p)} y1={16} x2={sx(CROSSING.p)} y2={58} stroke="var(--ink)" strokeWidth={2} />
      <text x={sx(CROSSING.p)} y={11} fontSize="11.5" textAnchor="middle" fontWeight="650" style={{ fill: 'var(--ink)' }}>p = {CROSSING.p}</text>
      <text x={X0} y={86} fontSize="10" style={{ fill: 'var(--ink-2)' }}>numbers under the bars: the mass ranked before each token</text>
    </svg>
  );
}

// ---------------------------------------------------------------- sort what you need
const KS = [0, 10, 40, 100, 1000];
const fmtShare = (x: number) => (x >= 0.1 ? `${fmt(x * 100, 3)}%` : x >= 0.001 ? `${fmt(x * 100, 2)}%` : `${fmt(x * 100, 1)}%`);

function SortExplorer() {
  const [V, setV] = useState(151936);
  const [s, setS] = useState(1.5);
  const [seed, setSeed] = useState(1);
  const [p, setP] = useState(0.95);
  const [k, setK] = useState(0);
  const row = useMemo(() => rowView(V, s, seed), [V, s, seed]);
  const plan = sortPlan(row, p, k, 0.05);
  const strategies = [
    { id: 'full', label: 'sort the whole row', n: plan.sizes.full, note: 'what lab 14.3 does with the row as it is' },
    { id: 'topk', label: 'top-k first, then sort k', n: plan.sizes.topK, note: plan.sizes.topK == null ? 'top-k is off' : 'lab X1.1, then lab 14.3 on k values' },
    {
      id: 'adaptive', label: "llama.cpp's host route", n: plan.sizes.adaptive,
      note: V <= 1024 ? '1,024 candidates or fewer: a full sort' : plan.widened ? 'the top 256 hold less than p: widened to the row' : 'the top 256 reach p: a partial sort of 256',
    },
    { id: 'minp', label: 'min-p 0.05 first, then sort the survivors', n: plan.sizes.minPFirst, note: "when the chain has min-p, as llama.cpp's default does: section X1.6" },
  ];
  // The cumulative mass against rank on a log axis, sampled at about 160 ranks.
  const PW = 560;
  const PH = 170;
  const L = 38;
  const B = 24;
  const lx = (rank: number) => L + (Math.log10(rank) / Math.log10(V)) * (PW - L - 10);
  const ly = (m: number) => PH - B - m * (PH - B - 10);
  const pts: [number, number][] = [];
  for (let j = 0; j <= 160; j++) {
    const rank = Math.max(1, Math.round(Math.pow(V, j / 160)));
    pts.push([lx(rank), ly(row.before[rank - 1] + row.probs[rank - 1])]);
  }
  const kept = plan.keptFull;
  const { reports } = useReports();
  const measured = realBackends(reports).flatMap((b) =>
    sortShare(benchRows(reports[b], labDir(reports[b]))).map((m) => ({ ...m, backend: b, device: String(reports[b].device.name ?? b) })),
  );
  return (
    <Bench
      title="Sort what you need"
      task="Start peaked, then flatten the row until llama.cpp's route widens; then turn top-k on."
      foot={
        <>
          The curve is the mass of the most likely tokens, up to and including each rank (log scale). Top-p keeps every token
          whose mass before it is below p: the curve's crossing of p, plus the token that crosses.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Vocabulary
          <span className="seg" role="group" aria-label="Vocabulary">
            {VOCAB_PRESETS.map((v) => (
              <button key={v} aria-pressed={V === v} onClick={() => setV(v)}>{n0(v)}</button>
            ))}
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          shape s = {s.toFixed(2)} ({s < 0.8 ? 'flat' : s < 1.4 ? 'in between' : 'peaked'})
          <input type="range" min={0} max={3} step={0.05} value={s} aria-label="shape" onChange={(e) => setS(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <button onClick={() => setSeed(seed + 1)}>another row</button>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          p = {p.toFixed(2)}
          <input type="range" min={1} max={100} step={1} value={Math.round(p * 100)} aria-label="p" onChange={(e) => setP(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
        <span style={FIELD}>
          Top-k first
          <span className="seg" role="group" aria-label="Top-k">
            {KS.map((x) => (
              <button key={x} aria-pressed={k === x} onClick={() => setK(x)}>{x === 0 ? 'off' : x}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="cols plot">
        <svg viewBox={`0 0 ${PW} ${PH}`} role="img" aria-label="Cumulative mass against rank" data-part="curve"
          style={{ width: '100%', maxWidth: PW, display: 'block', background: 'var(--page)', border: '1px solid var(--rule)', borderRadius: 4 }}>
          <line x1={L} y1={ly(0)} x2={PW - 10} y2={ly(0)} stroke="var(--rule)" />
          <line x1={L} y1={ly(1)} x2={PW - 10} y2={ly(1)} stroke="var(--rule-soft)" />
          <text x={L - 4} y={ly(1) + 4} fontSize="10" textAnchor="end">1</text>
          <text x={L - 4} y={ly(0) + 4} fontSize="10" textAnchor="end">0</text>
          <line x1={L} y1={ly(p)} x2={PW - 10} y2={ly(p)} stroke="var(--ink)" strokeDasharray="4 3" />
          <text x={(L + PW) / 2} y={ly(p) + 13} fontSize="10" textAnchor="middle">p = {p.toFixed(2)}</text>
          <polyline points={pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ')} fill="none" stroke="var(--global)" strokeWidth={2} />
          <line x1={lx(Math.min(kept, V))} y1={ly(0)} x2={lx(Math.min(kept, V))} y2={ly(1)} stroke="var(--ink)" />
          <text x={lx(Math.min(kept, V)) + (lx(Math.min(kept, V)) > PW - 140 ? -4 : 4)} y={ly(0) - 6} fontSize="10.5" fontWeight="650"
            textAnchor={lx(Math.min(kept, V)) > PW - 140 ? 'end' : 'start'}>top-p keeps {n0(kept)}</text>
          {V > 256 ? <line x1={lx(256)} y1={ly(0)} x2={lx(256)} y2={ly(0) + 6} stroke="var(--ink-2)" /> : null}
          {V > 256 ? <text x={lx(256)} y={PH - 6} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>256</text> : null}
          <text x={L} y={PH - 6} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>rank 1</text>
          <text x={PW - 10} y={PH - 6} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{n0(V)}</text>
        </svg>
        <div>
          <p style={{ marginTop: 0 }} data-readout="kept" data-kept={kept} data-kept-k={plan.keptAfterTopK ?? ''}>
            Top-p at p = {p.toFixed(2)} keeps <b>{n0(kept)}</b> of {n0(V)} tokens ({fmtShare(kept / V)} of the row).
            {plan.keptAfterTopK != null ? (
              <> After top-k = {k}, renormalised over those {k}, it keeps <b>{n0(plan.keptAfterTopK)}</b>.</>
            ) : null}
          </p>
          <p className="muted" style={{ marginBottom: 0 }}>
            The most likely token holds {fmtShare(row.probs[0])} of the mass.
          </p>
        </div>
      </div>
      <div style={{ overflowX: 'auto', marginTop: '0.8rem' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '34rem' }} data-part="strategies">
          <thead>
            <tr><th style={{ paddingLeft: '0.6rem' }}>Strategy</th><th className="num">Values sorted</th><th style={{ width: '30%' }}>log scale, up to the row</th><th /></tr>
          </thead>
          <tbody>
            {strategies.map((x) => (
              <tr key={x.id} data-strategy={x.id} data-n={x.n ?? ''}>
                <td style={{ paddingLeft: '0.6rem' }}>{x.label}</td>
                <td className="num">{x.n == null ? 'off' : n0(x.n)}</td>
                <td>
                  {x.n != null ? (
                    <span style={{ display: 'inline-block', height: 10, background: 'var(--global)', width: `${Math.max(1, (Math.log10(Math.max(1, x.n)) / Math.log10(V)) * 100)}%` }} />
                  ) : null}
                </td>
                <td className="muted">{x.id === 'minp' ? <a href={href('x1', 'X1.6')}>{x.note}</a> : x.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {measured.length ? (
        <div className="panel" style={{ marginTop: '0.8rem' }} data-part="measured">
          {measured.map((m, i) => (
            <p key={i} style={{ margin: 0 }}>
              Lab X1.2 on {m.device} ({m.backend}), {m.rows} × {n0(m.cols)}, {m.k ? `top-k ${m.k}` : 'top-k off'}: the sort was{' '}
              <b>{fmtShare(m.share)}</b> of the composed chain ({fmt(m.sort * 1e3)} of {fmt(m.chain * 1e3)} ms).
            </p>
          ))}
        </div>
      ) : (
        <p className="muted" style={{ margin: '0.6rem 0 0' }} data-part="measured-none">
          When lab X1.2 has run here (<code>python3 labs/pmpp.py run x1/lab2 --solution --bench</code>), its measured sort
          share appears beside the model.
        </p>
      )}
    </Bench>
  );
}

/** The lab directory under x1/ whose name starts lab2_, as this machine's reports name it. */
function labDir(run: { results: { lab: string }[] } | undefined): string {
  return run?.results.find((r) => r.lab.startsWith(LAB_X12 + '_'))?.lab ?? LAB_X12;
}

function MeasuredShare({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports).flatMap((b) =>
    sortShare(benchRows(reports[b], labDir(reports[b]))).map((m) => ({ ...m, backend: b, device: String(reports[b].device.name ?? b) })),
  );
  if (!found.length)
    return (
      <p>
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run x1/lab2 --solution --bench</code> and come
        back: this box compares your prediction with your own device's numbers, with top-k off and with top-k = 40.
      </p>
    );
  return (
    <>
      {found.map((m, i) => (
        <p key={i} data-reveal={m.backend}>
          {m.device} ({m.backend}), {m.k ? `top-k = ${m.k}` : 'top-k off'}: the sort took <b>{fmtShare(m.share)}</b> of the
          chain{m.k ? '.' : `; you predicted ${predicted}%.`}
        </p>
      ))}
      <p className="muted">Measurements from your machine, not expected values.</p>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSX15: Record<string, Question> = {
  'x1/X1.5/q-mlx-ties': {
    kind: 'numeric',
    id: 'x1/X1.5/q-mlx-ties',
    prompt: (
      <>
        Sorted probabilities [0.5, 0.125, 0.125, 0.125, 0.125] and p = 0.6. M5 keeps 2 tokens. How many does mlx-lm's{' '}
        <code>apply_top_p</code> keep?
      </>
    ),
    answer: topP(MLX_TIES.probs, MLX_TIES.p, 'mlx').kept,
    unit: 'tokens',
    explain: (
      <p>
        The mass before token 1 is 0.5, below 0.6, so the boundary token is token 1, with probability 0.125. mlx-lm then
        keeps every token at least as likely as the boundary token: all four 0.125s, and the 0.5. transformers keeps 2,
        and which of the four ties it keeps depends on its unstable sort.
      </p>
    ),
    source: 'notes/today/ch11-facts.md D3 (our reading of mlx-lm), D4',
  },
  'x1/X1.5/q-inclusive-bug': {
    kind: 'choice',
    id: 'x1/X1.5/q-inclusive-bug',
    prompt: <>A kernel keeps token i while the inclusive sum up to i is below p. On [0.5, 0.25, 0.125, 0.125] with p = 0.75, what goes wrong?</>,
    options: [
      { label: 'It keeps only token 0: the crossing token, whose inclusive sum is exactly 0.75, is dropped', correct: true },
      { label: 'It keeps token 2 as well, one token too many, because 0.75 is not below 0.75', why: 'Token 2 is dropped by both rules; its mass before is 0.75, not below p.' },
      { label: 'Nothing on this row; the two scans only differ when probabilities tie', why: 'They differ at every boundary: the inclusive sum includes the token itself.' },
    ],
    explain: (
      <p>
        The rule keeps the smallest set whose mass reaches p, so the token that crosses p stays: here tokens 0 and 1. The
        inclusive test drops token 1, whose own probability carries the sum to p. That off-by-one is the difference between
        the inclusive and the exclusive scan.
      </p>
    ),
    source: 'ch. 11 §11.1, p. 253; x1-facts D9',
  },
  'x1/X1.5/q-p1': {
    kind: 'choice',
    id: 'x1/X1.5/q-p1',
    prompt: <>Why must p ≥ 1 keep every token explicitly instead of relying on "mass before &lt; p"?</>,
    options: [
      { label: 'In float the mass before the last tokens can round to 1 or above, and p = 1 would then drop them', correct: true },
      { label: 'Because p = 1 is outside the range a scan kernel accepts', why: 'A scan does not know about p; the comparison does.' },
      { label: "Because softmax rows never sum to exactly 1, so the last token's mass before is always above 1", why: 'Not always above 1: it can round either way, which is the problem.' },
    ],
    explain: (
      <p>
        A user who sets p = 1 means "no top-p". llama.cpp's host sampler returns early for p ≥ 1, and lab 11.6's contract
        keeps everything at p ≥ 1 for the same reason.
      </p>
    ),
    source: 'notes/today/ch11-facts.md D2; labs/ch11/lab6_topp',
  },
  'x1/X1.5/q-adaptive': {
    kind: 'choice',
    id: 'x1/X1.5/q-adaptive',
    prompt: <>When does llama.cpp's host top-p have to sort more than its first 256 candidates?</>,
    options: [
      { label: 'When those 256 hold less than p of the mass, which happens on flat distributions', correct: true },
      { label: 'Whenever the vocabulary has more than 1,024 tokens, which is every current model', why: 'More than 1,024 candidates is when it starts with the partial sort of 256, not when it widens.' },
      { label: 'Never: a partial sort of 256 is always enough for any p below 1', why: 'On a flat row thousands of tokens can be needed to reach p.' },
    ],
    explain: (
      <p>
        It partial-sorts the top 256, runs the cumulative sum, and widens only when the sum has not reached p. In the
        explorer, flatten the row until the route switches from 256 to the whole row.
      </p>
    ),
    source: 'notes/today/ch14-facts.md D3',
  },
  'x1/X1.5/q-why-scan': {
    kind: 'choice',
    id: 'x1/X1.5/q-why-scan',
    prompt: <>Why does lab 11.6 compute an exclusive scan rather than an inclusive one for top-p?</>,
    options: [
      { label: 'The keep test needs the mass ranked strictly before each token, which the exclusive scan gives directly', correct: true },
      { label: 'An exclusive scan is cheaper on a GPU than an inclusive one, so the lab saves a pass over the row', why: 'They cost the same; each is the other shifted by one place.' },
      { label: 'An inclusive scan cannot be computed in float without rounding the last value above 1', why: 'Both scans round; that is not what decides the choice.' },
    ],
    explain: (
      <p>
        The token whose inclusive sum first reaches p still has mass before it below p, so the exclusive test keeps it,
        which is the definition. With an inclusive scan you would compare the previous element, the same thing one place
        over.
      </p>
    ),
    source: 'ch. 11 §11.1, p. 253; labs/ch11/lab6_topp',
  },
};

// ---------------------------------------------------------------- the section
export function SX15() {
  return (
    <>
      <p>
        Top-p, the nucleus, keeps the most likely tokens until their mass reaches p. You have the three pieces already:
        lab 14.3 sorts a row largest first and stably, lab 10.3 turns logits into probabilities, and lab 11.6 scans the
        sorted probabilities and keeps a prefix. This page states the rule and its tie-break exactly, composes the three,
        and asks how much of the row really needs sorting.
      </p>
      <Beyond>
        <p>
          Top-p is not in the book. The exclusive scan and the sausage picture are §11.1 (pp. 252–254); the stable key-value
          sort is §14.1 (p. 330) and lab 14.3.
        </p>
      </Beyond>

      <h2>The rule, exactly</h2>
      <p>
        Holtzman et al. define the top-p set as the smallest set of tokens whose total probability is at least p. So the
        token that crosses p is in. On a row sorted largest first that set is a prefix, and the test for each token is: keep
        token i when i = 0, or p ≥ 1, or the mass ranked before it is below p. That is lab 11.6's rule. llama.cpp's host
        and backend samplers, transformers' ascending formulation and vLLM all reduce to it on rows that sum to 1.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D9; ch11-facts D2, D4)">
        <p>
          The paper defines V(p) as "the smallest set such that Σ P ≥ p" and gives no tie rule.
        </p>
      </Today>
      <Predict
        id="x1/X1.5/p-crossing"
        prompt={<>Sorted probabilities [0.5, 0.25, 0.125, 0.0625, 0.0625] and p = 0.875. How many tokens does M5 keep?</>}
        unit="tokens"
      >
        {(v) => {
          const r = topP(CROSSING.probs, CROSSING.p, 'exclusive');
          return (
            <>
              <p data-reveal="crossing">
                <b>{r.kept}</b>{v === r.kept ? ', as you said' : ''}. The masses before the tokens are 0, 0.5, 0.75, 0.875 and
                0.9375; the first three are below p. Token 2's inclusive sum is exactly 0.875: it is the crossing token and
                it stays. Every value here is a power of two, so float arithmetic decides it exactly too.
              </p>
              <CrossingBars />
            </>
          );
        }}
      </Predict>
      <Ask q={questionsSX15['x1/X1.5/q-inclusive-bug']} />
      <Ask q={questionsSX15['x1/X1.5/q-why-scan']} />

      <h2>Ties are broken by the sort</h2>
      <p>
        "Smallest set" is not unique when tokens tie at the boundary. In M5 the tie-break is position, which means the
        sort. The candidates arrive from lab X1.1 in ascending column order, and lab 14.3's sort is stable, so equal
        probabilities stay in order of increasing token id and the lower id survives the boundary.
      </p>
      <Source section="14.1" pages="330" />
      <p>
        The projects differ here. mlx-lm keeps every token tied with the boundary token, so it can keep more. transformers
        sorts ascending with the default, unstable <code>torch.sort</code>, so which tied token it keeps is not specified.
        Chapter 11's <a href={href('ch11', '11.1')}>"Where the nucleus ends"</a> explorer puts the rules side by side on
        rows you choose.
      </p>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md D3, D4; ch14-facts D5)">
        <p>
          mlx-lm <code>apply_top_p</code>: masks <code>logprobs &lt; threshold</code>, where the threshold is the boundary
          token's own log-probability, so its ties survive (our reading).
        </p>
      </Today>
      <Ask q={questionsSX15['x1/X1.5/q-mlx-ties']} label="Work it out" />

      <h2>Composed from three labs</h2>
      <p>
        In M5, top-p runs on the top-k survivors, a [rows × k] tensor of logits with their token ids. Lab 14.3's{' '}
        <code>launchSortRowsDesc</code> sorts each row largest first. Lab 10.3's <code>launchSoftmaxRows</code> turns
        them into probabilities with scale 1, because llama.cpp decides top-p before temperature. Lab 11.6's{' '}
        <code>launchTopPMask</code> writes the mass before each token, the keep mask and the kept count. The probabilities
        are renormalised over the k survivors, as in llama.cpp and transformers.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D1, D4)">
        <p>
          llama.cpp's default chain runs top-k, then top-p and min-p, and temperature last; transformers applies
          temperature first.
        </p>
      </Today>
      <div className="wide" style={{ maxWidth: '58rem', display: 'flex', flexWrap: 'wrap', gap: '0.4rem', alignItems: 'center', margin: '0.4rem 0 1.4rem', fontFamily: 'var(--sans)', fontSize: '0.82rem' }} data-part="chain">
        {[
          ['candidates', '[rows × k] logits and ids, column order (lab X1.1)'],
          ['launchSortRowsDesc', 'lab 14.3: largest first, stable'],
          ['launchSoftmaxRows', 'lab 10.3: scale 1'],
          ['launchTopPMask', 'lab 11.6: mass before, keep, kept'],
          ['kept count', 'a prefix of each sorted row'],
        ].map(([a, b], i, all) => (
          <span key={a} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
            <span className="panel" style={{ padding: '0.35rem 0.6rem' }}>
              <code>{a}</code>
              <br />
              <span className="muted">{b}</span>
            </span>
            {i < all.length - 1 ? <span aria-hidden="true">→</span> : null}
          </span>
        ))}
      </div>

      <h2>How much do you need to sort?</h2>
      <p>
        With top-k first, the sort sees k values, not the vocabulary. Without it, a straightforward top-p sorts the whole
        row, and on a peaked distribution almost all of that sort is wasted: p = 0.95 may need a handful of tokens. llama.cpp's
        host sampler hedges. With more than 1,024 candidates it partial-sorts only the top 256, and widens only if those
        hold less than p. vLLM's default avoids the sort altogether with a pivot search. Section X1.6 shows a third way,
        filtering by min-p before sorting.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md D3, D7)">
        <p>
          llama.cpp <code>llama_sampler_top_p_apply</code>: "try adaptive top-k sorting" of 256 when there are more than
          1,024 candidates. vLLM: <code>apply_top_k_top_p_triton</code>, based on the Qrita paper (not read here).
        </p>
      </Today>
      <p>
        The explorer below draws a whole vocabulary row, finds top-p's kept count, and counts how many values each of
        these strategies hands to the sort.
      </p>
      <Simplification>
        <p>
          The rows are synthetic: seeded Zipf-like logits, rank r getting −s ln(r + 1) plus a small jitter, with s from flat
          (0) to peaked (3). Not a real model's output. Kept counts are computed in double.
        </p>
      </Simplification>
      <SortExplorer />
      <p>
        On a peaked row the nucleus is a few tokens and sorting 150,000 values to find them is almost all waste. On a flat
        row top-p alone keeps tens of thousands, and then it is top-k that caps the sort.
      </p>
      <Ask q={questionsSX15['x1/X1.5/q-adaptive']} />

      <h2>Floats at the boundary</h2>
      <p>
        A decision is only as exact as the mass before the token. Lab 11.6 accepts either answer for a token whose exact
        mass before it lies within 1e-4 of p (relative to that mass), because backends and libraries add in different
        orders. Rows whose probabilities are powers of two decide exactly on every backend, which is why the lab's
        boundary tests use them. And p ≥ 1 must keep everything explicitly.
      </p>
      <Source section="11.1" pages="252–254" />
      <Ask q={questionsSX15['x1/X1.5/q-p1']} />

      <h2>What the sort costs in the chain</h2>
      <p>
        Lab X1.2 composes the chain from the three labs and times each step, with top-k off (the sort sees the whole row)
        and with top-k = 40. Before you run it, commit to a number.
      </p>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md D2 to D4; ch14-facts D3, D7)">
        <p>
          llama.cpp's backend top-p is argsort, softmax, cumsum and a step mask, with one extra token made inclusive.
          transformers sorts ascending and removes tokens whose cumulative sum is at most 1 − p. vLLM's default needs no
          sort.
        </p>
      </Today>
      <Predict
        id="x1/X1.5/p-sort-share"
        prompt={
          <>
            With top-k off, 64 rows of 151,936: what percentage of the composed chain's time is the sort? (The reveal also
            shows the share at top-k = 40.)
          </>
        }
        unit="%"
      >
        {(v) => <MeasuredShare predicted={v} />}
      </Predict>
      <p>
        The kept count is a prefix length per row. The next page, <a href={href('x1', 'X1.6')}>min-p</a>, is a prefix too,
        and the two compose by taking the shorter.
      </p>
    </>
  );
}
