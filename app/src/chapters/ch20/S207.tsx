import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt } from '../../lib/reports';
import { href } from '../../lib/router';
import { aiGQA, aiMLA, gqaTerms, kvHeadOf } from '../../sim/kvmath';
import { peakLive, simulate, type PagedConfig } from '../../sim/paged';
import { KvExplorer } from './KvExplorer';
import { ContiguousPool, PagedBlocksScene, PagedPool } from './PagedBlocksScene';

// ---------------------------------------------------------------- who shares a K/V head

type HeadVariant = 'MHA' | 'GQA' | 'MQA' | 'MLA';
const D_HEAD = 128;
const N_AI = 4096;
const divisorsBetween = (n: number) => Array.from({ length: n }, (_, i) => i + 1).filter((k) => n % k === 0 && k > 1 && k < n);

function HeadsBench() {
  const [hq, setHq] = useState(6);
  const [variant, setVariant] = useState<HeadVariant>('GQA');
  const [gqaPick, setGqaPick] = useState(3);
  const [sel, setSel] = useState<number | null>(4);
  const groups = divisorsBetween(hq);
  const gqaHkv = groups.includes(gqaPick) ? gqaPick : groups[Math.floor((groups.length - 1) / 2)] ?? null;
  const v: HeadVariant = variant === 'GQA' && gqaHkv == null ? 'MHA' : variant;
  const hkv = v === 'MHA' ? hq : v === 'MQA' || v === 'MLA' ? 1 : gqaHkv!;
  const { gq, G } = gqaTerms(hq, hkv);
  const values = v === 'MLA' ? 4 * D_HEAD + D_HEAD / 2 : 2 * hkv * D_HEAD;
  const ai = v === 'MLA' ? aiMLA(N_AI, hq) : aiGQA(N_AI, hq, gq);
  const selected = sel != null && sel < hq ? sel : null;
  const target = selected != null ? kvHeadOf(selected, hq, hkv) : null;

  const W = 720;
  const H = 250;
  const qw = Math.min(56, (W - 40) / hq - 4);
  const qx = (h: number) => 20 + h * ((W - 40) / hq) + ((W - 40) / hq - qw) / 2;
  const kvSlot = (W - 40) / hkv;
  const kw = Math.min(70, kvSlot - 8);
  const kx = (k: number) => 20 + k * kvSlot + (kvSlot - kw) / 2;
  const yQ = 46;
  const yK = 160;
  return (
    <Bench
      title="Who shares a K/V head"
      task="Click a query head to see which K/V head it reads. Then count what is cached."
      foot={
        <>
          The outlined boxes are what is cached during inference (the red boxes of Fig. 20.19). Heads are {D_HEAD} wide
          and the intensity is at N = {N_AI.toLocaleString('en-US')} cached tokens. MLA’s latent uses DeepSeek-V2’s
          proportions, d_c = 4d and d_rope = d/2 (facts E5). Model: <code>src/sim/kvmath.ts</code>.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Query heads h_q = {hq}
          <input type="range" min="1" max="64" value={hq} aria-label="Query heads" onChange={(e) => { setHq(Number(e.target.value)); setSel(null); }} />
        </label>
        <span className="seg" role="group" aria-label="Variant">
          {(['MHA', 'GQA', 'MQA', 'MLA'] as HeadVariant[]).map((x) => (
            <button key={x} aria-pressed={v === x} disabled={x === 'GQA' && gqaHkv == null} onClick={() => setVariant(x)}>{x}</button>
          ))}
        </span>
        {v === 'GQA' ? (
          <label className="field">
            K/V heads
            <select value={hkv} onChange={(e) => setGqaPick(Number(e.target.value))} aria-label="K/V heads">
              {groups.map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
          </label>
        ) : null}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Query heads and the K/V heads they read" style={{ width: '100%', maxWidth: W, display: 'block' }}
        data-hq={hq} data-hkv={hkv} data-variant={v}>
        <text x={20} y={yQ - 10} fontSize="11.5" fontWeight="600">query heads</text>
        {Array.from({ length: hq }, (_, h) => {
          const k = kvHeadOf(h, hq, hkv);
          const on = selected === h;
          const x2 = v === 'MLA' ? W / 2 : kx(k) + kw / 2;
          return (
            <line key={'l' + h} x1={qx(h) + qw / 2} y1={yQ + 26} x2={x2} y2={yK - 4}
              stroke={on ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={on ? 2.4 : 0.9} strokeDasharray={on ? undefined : '3 3'} />
          );
        })}
        {Array.from({ length: hq }, (_, h) => (
          <g key={h} onClick={() => setSel(h)} style={{ cursor: 'pointer' }} data-qhead={h}>
            <rect x={qx(h)} y={yQ} width={qw} height={26} rx="2" fill={selected === h ? 'var(--reg)' : 'var(--reg-wash)'} stroke="var(--reg)" />
            {qw > 14 ? <text x={qx(h) + qw / 2} y={yQ + 17} fontSize={Math.min(11, qw / 2)} textAnchor="middle" style={{ fill: selected === h ? 'var(--page)' : 'var(--ink)' }}>{h}</text> : null}
          </g>
        ))}
        {v === 'MLA' ? (
          <g>
            <rect x={W / 2 - 90} y={yK} width={180} height={34} rx="2" fill="var(--global-wash)" stroke="var(--global)" strokeWidth="2.4" />
            <text x={W / 2} y={yK + 21} fontSize="11.5" textAnchor="middle">one latent, {4 * D_HEAD} + {D_HEAD / 2} values</text>
            <text x={W / 2} y={yK + 54} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
              serves as key and value; K and V are never formed
            </text>
          </g>
        ) : (
          <g>
            <rect x={14} y={yK - 8} width={W - 28} height={66} rx="3" fill="none" stroke="var(--global)" strokeWidth="2.4" />
            <text x={W - 20} y={yK + 72} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--global)' }}>cached</text>
            {Array.from({ length: hkv }, (_, k) => (
              <g key={k} data-kvhead={k}>
                <rect x={kx(k)} y={yK} width={kw / 2 - 1} height={42} fill={target === k ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" />
                <rect x={kx(k) + kw / 2 + 1} y={yK} width={kw / 2 - 1} height={42} fill={target === k ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" />
                {kw > 22 ? (
                  <>
                    <text x={kx(k) + kw / 4} y={yK + 26} fontSize="10" textAnchor="middle" style={{ fill: target === k ? 'var(--page)' : 'var(--ink)' }}>K</text>
                    <text x={kx(k) + (3 * kw) / 4} y={yK + 26} fontSize="10" textAnchor="middle" style={{ fill: target === k ? 'var(--page)' : 'var(--ink)' }}>V</text>
                  </>
                ) : null}
              </g>
            ))}
            <text x={20} y={yK + 72} fontSize="11.5" fontWeight="600">K/V heads</text>
          </g>
        )}
      </svg>
      <table className="data" style={{ background: 'var(--page)', margin: '0.6rem 0 0' }} data-heads="">
        <tbody>
          <tr>
            <td>Query head {selected ?? '(click one)'} reads</td>
            <td className="num" data-reads={target ?? ''}>{selected == null ? '–' : v === 'MLA' ? 'the latent' : `K/V head ${target}`}</td>
            <td className="muted">{v === 'MLA' ? 'every head reads the same latent' : `⌊h ÷ (h_q / h_kv)⌋ = ⌊${selected ?? 'h'} ÷ ${fmt(hq / hkv)}⌋`}</td>
          </tr>
          <tr>
            <td>g_q, query heads per K/V head</td>
            <td className="num" data-gq={v === 'MLA' ? '' : gq}>{v === 'MLA' ? '–' : fmt(gq)}</td>
            <td className="muted">what Eqs. 20.12 and 20.13 use ([13]’s definition)</td>
          </tr>
          <tr>
            <td>G, number of groups</td>
            <td className="num" data-groups={v === 'MLA' ? '' : G}>{v === 'MLA' ? '–' : G}</td>
            <td className="muted">what the GQA paper calls GQA-G; equals h_kv</td>
          </tr>
          <tr>
            <td>Cached per token per layer</td>
            <td className="num" data-values={values}>{values.toLocaleString('en-US')} values</td>
            <td className="muted">{v === 'MLA' ? 'd_c + d_rope' : `2 × ${hkv} × ${D_HEAD}`}</td>
          </tr>
          <tr>
            <td>Decode intensity (multiply-adds per element)</td>
            <td className="num" data-ai={ai.toFixed(3)}>{fmt(ai, 4)}</td>
            <td className="muted">{v === 'MLA' ? '2N h_q / (2h_q + N), about 2h_q' : 'Eq. 20.13, about g_q'}</td>
          </tr>
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- contiguous or paged

const POOLS = [512, 1024, 2048, 4096];
const BLOCKS = [16, 32, 64, 128, 256];
const MAXSEQ = [128, 256, 512, 1024];

function PagedStepper() {
  const [cfg, setCfg] = useState<PagedConfig>({ pool: 1024, block: 16, maxSeq: 256, requests: 16, seed: 1 });
  const run = useMemo(() => simulate(cfg), [cfg]);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const last = run.frames.length - 1;
  const at = Math.min(t, last);
  const f = run.frames[at];
  const peak = useMemo(() => peakLive(run), [run]);
  useEffect(() => { setT(0); setPlaying(false); }, [cfg]);
  useEffect(() => {
    if (!playing) return;
    if (at >= last) { setPlaying(false); return; }
    const id = window.setTimeout(() => setT((x) => x + 1), 120);
    return () => window.clearTimeout(id);
  }, [playing, at, last]);
  const set = (k: keyof PagedConfig, v: number) => setCfg((c) => ({ ...c, [k]: v }));
  const nextEvent = () => {
    for (let i = at + 1; i <= last; i++)
      if (run.frames[i].paged.events.length || run.frames[i].contiguous.events.length) return setT(i);
    setT(last);
  };
  const cRows = run.slots;
  const pRows = Math.ceil(run.blocks / Math.min(16, run.blocks));
  const cH = Math.min(230, cRows * 22) + (run.cfg.pool > cRows * run.cfg.maxSeq ? 14 : 0);
  const pH = Math.min(240, pRows * 26);
  const counters = (s: typeof f.paged) => (
    <table className="data" style={{ background: 'var(--page)', margin: '0.5rem 0 0' }}>
      <tbody>
        <tr><td>reserved</td><td className="num" data-reserved={s.reserved}>{s.reserved}</td><td>used</td><td className="num" data-used={s.used}>{s.used}</td><td>wasted</td><td className="num" data-wasted={s.wasted}>{s.wasted}</td></tr>
        <tr><td>running</td><td className="num" data-running={s.live.length}>{s.live.length}</td><td>waiting</td><td className="num" data-waiting={s.waiting.length}>{s.waiting.length}</td><td>done</td><td className="num" data-done={s.done}>{s.done}</td></tr>
        <tr><td>admitted</td><td className="num" data-admitted={s.admitted}>{s.admitted}</td><td colSpan={2}>swapped out</td><td className="num" colSpan={2} data-evictions={s.evictions}>{s.evictions}</td></tr>
      </tbody>
    </table>
  );
  return (
    <Bench
      title="Contiguous or paged"
      task="Run the same stream of requests through both pools. Which one keeps more requests running, and why?"
      foot={
        <>
          Sizes are in tokens of one layer’s cache. Each request grows one token per step to a final length neither
          allocator knows in advance. Solid: tokens stored; washed: reserved but empty; dashed: free. Model:{' '}
          <code>src/sim/paged.ts</code>.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Pool (tokens)
          <select value={cfg.pool} onChange={(e) => set('pool', Number(e.target.value))} aria-label="Pool size">
            {POOLS.map((x) => <option key={x} value={x}>{x.toLocaleString('en-US')}</option>)}
          </select>
        </label>
        <label className="field">
          Block (tokens)
          <select value={cfg.block} onChange={(e) => set('block', Number(e.target.value))} aria-label="Block size">
            {BLOCKS.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          max_seq
          <select value={cfg.maxSeq} onChange={(e) => set('maxSeq', Number(e.target.value))} aria-label="Maximum sequence length">
            {MAXSEQ.map((x) => <option key={x} value={x}>{x.toLocaleString('en-US')}</option>)}
          </select>
        </label>
        <label className="field">
          Requests = {cfg.requests}
          <input type="range" min="4" max="48" value={cfg.requests} aria-label="Requests" onChange={(e) => set('requests', Number(e.target.value))} />
        </label>
        <button onClick={() => set('seed', cfg.seed + 1)}>Another stream</button>
      </div>
      {run.cfg.pool !== cfg.pool ? <p className="muted" style={{ marginTop: 0 }}>The pool was raised to {run.cfg.pool} tokens so that one max_seq request fits.</p> : null}
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setT(0); }} disabled={at === 0}>Reset</button>
        <button onClick={() => { setPlaying(false); setT(at - 1); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setT(at + 1); }} disabled={at >= last}>Step</button>
        <button onClick={() => { setPlaying(false); nextEvent(); }} disabled={at >= last}>Next event</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= last}>{playing ? 'Pause' : 'Run'}</button>
        <input type="range" min="0" max={last} value={at} aria-label="Step" style={{ flex: '1 1 10rem' }} onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }} />
        <span className="muted" aria-live="polite" data-step={f.step}>step {f.step} of {last}</span>
      </div>
      <div className="cols">
        <div data-pool="contiguous">
          <div className="mx-label t-global">Contiguous: {run.slots} slots of {run.cfg.maxSeq} tokens</div>
          <svg viewBox={`0 0 360 ${cH}`} style={{ width: '100%', maxWidth: 360, display: 'block' }} role="img" aria-label="Contiguous pool">
            <ContiguousPool run={run} side={f.contiguous} x={2} y={2} w={356} h={Math.min(230, cRows * 22)} />
          </svg>
          {counters(f.contiguous)}
        </div>
        <div data-pool="paged">
          <div className="mx-label t-global">Paged: {run.blocks} blocks of {run.cfg.block} tokens, {f.paged.free.length} free</div>
          <svg viewBox={`0 0 360 ${pH}`} style={{ width: '100%', maxWidth: 360, display: 'block' }} role="img" aria-label="Paged pool">
            <PagedPool run={run} side={f.paged} x={2} y={2} w={356} h={pH - 4} />
          </svg>
          {counters(f.paged)}
          <p className="muted" style={{ fontSize: '0.8rem', margin: '0.4rem 0 0' }} data-tables="">
            Block tables: {f.paged.live.length ? f.paged.live.slice(0, 10).map((r) => `r${r.id} → [${r.blocks!.join(', ')}]`).join('; ') : 'none yet'}
            {f.paged.live.length > 10 ? `; … ${f.paged.live.length - 10} more` : ''}
          </p>
        </div>
      </div>
      <div className="narrate" role="status" style={{ marginTop: '0.8rem' }} data-events="">
        {f.step === 0 ? 'Nothing has arrived yet. Press Step, Next event or Run.' : (
          <>Step {f.step}. Paged: {f.paged.events.join(', ') || 'nothing new'}. Contiguous: {f.contiguous.events.join(', ') || 'nothing new'}.</>
        )}
        {at >= last ? (
          <>
            {' '}All {cfg.requests} requests done: contiguous after <b data-finished-contiguous={run.finishedAt.contiguous ?? ''}>{run.finishedAt.contiguous}</b> steps,
            paged after <b data-finished-paged={run.finishedAt.paged ?? ''}>{run.finishedAt.paged}</b>. At most {peak.contiguous} against {peak.paged} running at once
            {f.paged.evictions ? `; the paged pool ran out ${f.paged.evictions} ${f.paged.evictions === 1 ? 'time' : 'times'} and swapped a request out` : ''}.
          </>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS207: Record<string, Question> = {
  'ch20/20.7/q-gq': {
    kind: 'numeric',
    id: 'ch20/20.7/q-gq',
    prompt: <>Mistral 7B has 32 query heads and 8 K/V heads. What is g_q as Eqs. 20.12 and 20.13 use it?</>,
    answer: 4,
    explain: (
      <p>
        In the formulas the number of K/V heads is h_q / g_q, so g_q = 32 / 8 = 4 query heads per K/V head. The GQA paper
        would call this model GQA-8: G = 8 groups. Same model, two numbers; say which one you mean.
      </p>
    ),
    source: 'notes/today/ch20-facts.md E3, E2',
  },
  'ch20/20.7/q-gqa-ai': {
    kind: 'numeric',
    id: 'ch20/20.7/q-gqa-ai',
    prompt: <>Evaluate Eq. 20.13 at N = 4,096 cached tokens, h_q = 32 and g_q = 4.</>,
    answer: 3.996,
    tol: 0.01,
    explain: <p>N h_q / (h_q + (h_q/g_q) N) = 131,072 / (32 + 8 × 4,096) = 131,072 / 32,800 = 3.996, about g_q: each loaded K/V element now meets four queries.</p>,
    source: 'ch. 20 §20.7, p. 509, Eq. 20.13; notes/today/ch20-facts.md F1',
  },
  'ch20/20.7/q-head-map': {
    kind: 'numeric',
    id: 'ch20/20.7/q-head-map',
    prompt: <>A model has 6 query heads and 2 K/V heads (FlashAttention’s own example). Which K/V head does query head 4 read?</>,
    answer: 1,
    explain: <p>Each K/V head serves 6 / 2 = 3 consecutive query heads: 0, 1, 2 read K/V head 0 and 3, 4, 5 read K/V head 1. In code, kv_head = h / 3 rounded down, which llama.cpp writes with its gqa_ratio.</p>,
    source: 'notes/today/ch20-facts.md A8, D1',
  },
  'ch20/20.7/q-mla-bytes': {
    kind: 'numeric',
    id: 'ch20/20.7/q-mla-bytes',
    prompt: <>DeepSeek-V2 caches d_c + d_h^R = 512 + 64 values per token per layer, in 60 layers, at 16 bits. How many bytes per token?</>,
    answer: 69120,
    unit: 'bytes',
    explain: <p>576 × 60 × 2 = 69,120 bytes. The book’s Eq. 20.14 with a latent of width d = 128 gives 60 × 128 × 2 = 15,360, 4.5 times less. The MHA equivalent, 128 heads of 128, would be 3.75 MiB: 57 times more.</p>,
    source: 'notes/today/ch20-facts.md E5, F1',
  },
  'ch20/20.7/q-mla-ai': {
    kind: 'choice',
    id: 'ch20/20.7/q-mla-ai',
    prompt: <>Why is MLA’s decode intensity about 2 h_q, twice MQA’s?</>,
    options: [
      { label: 'Each loaded latent element serves as both key and value.', correct: true },
      { label: 'Because the latent is half the size of an attention head.', why: 'DeepSeek-V2’s latent is 4.5 heads wide, and size does not set the ratio; reuse does.' },
      { label: 'Because MLA decompresses K and V on chip just before use.', why: 'That is the book’s account, and it would not give this figure: decompressed K and V would each be read once, as in MQA.' },
      { label: 'Because it caches only K and recomputes V from it.', why: 'It caches one latent that plays both parts.' },
    ],
    explain: (
      <p>
        At inference the up-projections are absorbed into W_Q and W_O, so attention runs directly on the single latent
        and K and V are never formed. MQA loads a K element and a V element, each used by h_q queries: about h_q
        operations per element. MLA loads one latent element and uses it for the scores and for the weighted sum, by all h_q heads: about 2 h_q. The cited source
        says the latent “serves both the key and the value”.
      </p>
    ),
    source: 'notes/today/ch20-facts.md E4, E5',
  },
  'ch20/20.7/q-paged': {
    kind: 'choice',
    id: 'ch20/20.7/q-paged',
    prompt: <>During normal decoding, where are PagedAttention’s KV blocks, and what does the block table buy?</>,
    options: [
      { label: 'Anywhere in GPU memory; memory is taken one block at a time.', correct: true },
      { label: 'In host memory, copied into the GPU when they are needed.', why: 'That is the book’s account. The paper uses host memory only when it runs out of GPU blocks and must evict a sequence.' },
      { label: 'In one contiguous buffer per sequence, sized for its maximum length.', why: 'That is what paging replaces.' },
      { label: 'On disk, with the block table as an index into the file.', why: 'Nothing in the design goes to disk.' },
    ],
    explain: (
      <p>
        The table maps each sequence&rsquo;s logical blocks to physical ones, so the waste is at most one partly filled
        block per sequence. The vLLM paper: PagedAttention &ldquo;allows storing continuous keys and values in
        non-contiguous memory space&rdquo;, block by block, and swaps evicted sequences&rsquo; blocks to CPU memory only
        when free GPU blocks run out.
      </p>
    ),
    source: 'notes/today/ch20-facts.md E6; book p. 509',
  },
  'ch20/20.7/q-moe': {
    kind: 'choice',
    id: 'ch20/20.7/q-moe',
    prompt: <>Mixtral has 47B parameters but uses 13B of them per token. Which cost does a mixture of experts cut, and which not?</>,
    options: [
      { label: 'Compute per token falls; weight memory does not.', correct: true },
      { label: 'Both the compute and the memory fall.', why: 'Any token may be routed to any expert, so all of them must be loaded.' },
      { label: 'Only the weight memory falls.', why: 'It is the other way round.' },
      { label: 'Neither: the router costs as much as it saves.', why: 'Each token runs 2 of the 8 experts, so its compute does fall.' },
    ],
    explain: <p>Each layer has 8 feed-forward experts and a router picks 2 per token, so the compute falls. Every expert must still be resident, so the weights all live in memory, and the KV cache is unchanged: Mixtral’s attention is Mistral 7B’s.</p>,
    source: 'ch. 20 §20.7, p. 510; notes/today/ch20-facts.md E7',
  },
};

// ---------------------------------------------------------------- the page

export function S207() {
  return (
    <>
      <p>
        Section 20.6 traced the memory problem to one design choice. In multi-head attention every head has its own weight
        matrices, so every head computes, and must cache, its own K and V. Each variant in this section caches less, and,
        as a side effect, does more arithmetic per byte it reads.
      </p>
      <Source section="20.7" pages="508" />

      <h2>Multi-query attention</h2>
      <p>
        Multi-query attention (MQA) lets all the query heads share a single K head and a single V head. The cache shrinks by
        the number of heads, to b × N × 2 × l × d × p (Eq. 20.10).
      </p>
      <Today checked="2026-11-20 (ch20-facts E7)">
        <p>The book does not cite MQA’s source: Shazeer 2019, whose abstract names this very problem, “the memory-bandwidth cost of repeatedly loading the large ‘keys’ and ‘values’ tensors”.</p>
      </Today>
      <p>
        And each loaded K or V element now meets all h_q queries, so the decode intensity becomes N h_q / (h_q + N), about
        h_q (Eq. 20.11). The bytes went down and the arithmetic stayed: that is the whole trick.
      </p>
      <Source section="20.7" pages="508" figs="Eqs. 20.10–20.11" />

      <h2>Grouped-query attention, and what g_q means</h2>
      <p>
        Grouped-query attention (GQA) sits between the two: query heads are divided into groups, and each group shares one K
        head and one V head. The book gives the cache as b × N × 2 × l × (h_q / g_q) × d × p (Eq. 20.12) and the
        intensity as about g_q (Eq. 20.13).
      </p>
      <Source section="20.7" pages="508–509" figs="Eqs. 20.12–20.13, Fig. 20.19" />
      <p>
        Read the formula before the prose. h_q / g_q is the number of K/V heads, so g_q must be the number of query heads
        per K/V head: g_q = 1 is MHA and g_q = h_q is MQA. The book’s prose calls g_q “a few groups of query heads”, which
        is the GQA paper’s G, the number of groups, equal to the number of K/V heads. For Mistral 7B those are 4 and 8. This
        course uses g_q = h_q / h_kv throughout.
      </p>
      <Today checked="2026-11-20 (ch20-facts E3)">
        <p>[13], the book’s source for the formulas: “We define the group size g_q as the number of query heads per distinct KV head.” The GQA paper: “GQA-G refers to grouped-query with G groups.”</p>
      </Today>
      <HeadsBench />
      <p>
        That mapping, consecutive query heads sharing one K/V head, is what the decode kernels you will read implement:
        query head h reads K/V head h / g_q, and K and V are not copied once per query head to make the shapes match.
      </p>
      <Today checked="2026-11-20 (ch20-facts D1, A8, D4)">
        <p>llama.cpp computes <code>gqa_ratio = Q-&gt;ne[2] / K-&gt;ne[2]</code> and processes several Q heads per loaded K/V head. FlashAttention’s <code>flash_attn_with_kvcache</code>: with 6 Q heads and 2 K/V heads, Q heads 0, 1, 2 read K/V head 0. MLX asks that k and v not be pre-tiled to match q.</p>
      </Today>
      <p>
        A good GQA kernel loads each K/V row once and uses it for all g_q query heads of its group, which is what lab 20.5
        asks you to write and measure.
      </p>
      <Beyond>
        <p>Lab 20.5 is ours; see the <a href={href('ch20', 'labs')}>labs page</a>.</p>
      </Beyond>
      <Ask q={questionsS207['ch20/20.7/q-gq']} label="Work it out" />
      <Ask q={questionsS207['ch20/20.7/q-head-map']} label="Work it out" />
      <Ask q={questionsS207['ch20/20.7/q-gqa-ai']} label="Work it out" />
      <p>
        MHA, GQA and MQA all fit the flash-attention tiling of <a href={href('ch20', '20.5')}>section 20.5</a>. They trade
        computation, memory and accuracy differently; the kernel structure stays the same.
      </p>
      <Source section="20.7" pages="509" />

      <h2>Paging the cache</h2>
      <p>
        Even with fewer K/V heads, batch size and context length are limited by memory, and the book adds a second cause:
        cache memory is usually reserved for the largest possible input, so most of it sits fragmented and unused.
        PagedAttention, inspired by paging in operating systems, splits the cache into equal-size blocks which, in the book’s
        words, are loaded into GPU memory from the larger system memory when needed.
      </p>
      <Source section="20.7" pages="509" />
      <p>
        The paper’s design is about where the blocks sit in GPU memory: anywhere. Each sequence has a block table that maps
        its logical blocks to physical ones, so a sequence takes memory one block at a time as it grows, and wastes at most
        one partly filled block. Host memory appears only as vLLM’s fallback when it runs out of blocks and has to evict a
        sequence.
      </p>
      <Today checked="2026-11-20 (ch20-facts E6, A8, D3)">
        <p>vLLM’s own PagedAttention design page is now marked historical: paging lives inside its attention backends, through block tables. FlashAttention takes a <code>block_table</code> (pages a multiple of 256 tokens). llama.cpp’s ring of token slots has no block table: it is not paged.</p>
      </Today>
      <p>
        Run the same stream of requests through a contiguous pool, reserved at max_seq per request as the Llama reference
        does, and a paged one. Before you press Run, predict which side will have requests waiting while memory sits empty.
      </p>
      <Simplification>
        <p>One layer, sizes in tokens. Final lengths are drawn short-skewed (our assumption). Paged admission is first come, first served. When the free list is empty a request is swapped out, as vLLM evicts sequences when it runs out of blocks; which one (here the newest) is our choice, and the copy costs nothing. No prefix sharing.</p>
      </Simplification>
      <PagedStepper />
      <PagedBlocksScene />
      <Ask q={questionsS207['ch20/20.7/q-paged']} />

      <h2>Multi-head latent attention</h2>
      <p>
        The book describes MLA as a low-rank joint compression: each token’s K and V are compressed into one latent vector,
        which is cached instead (Eq. 20.14, b × N × l × d × p), and two up-projection matrices decompress K and V just before
        a head uses them. Its intensity is about 2 h_q (Eq. 20.15).
      </p>
      <Source section="20.7" pages="509–510" figs="Eqs. 20.14–20.15" />
      <p>
        DeepSeek-V2’s paper differs from that account in three ways that matter to a kernel. The latent is one per token per
        layer, shared by all heads, not one across layers. It is d_c + d_h^R = 512 + 64 = 4.5 head widths, not d, so
        Eq. 20.14 understates DeepSeek-V2’s cache 4.5 times. And at inference the up-projections are absorbed into W_Q and
        W_O, so K and V are never formed: attention runs directly on the latent, which serves as both key and value. That
        double use is why the intensity is about 2 h_q, twice MQA’s. Eq. 20.15’s denominator should read 2 h_q + N, not
        h_q + N; the limit is the same.
      </p>
      <Today checked="2026-11-20 (ch20-facts E5, E4, D3)">
        <p>DeepSeek-V2: “W^UK can be absorbed into W^Q, and W^UV can be absorbed into W^O, we even do not need to compute keys and values out for attention.” llama.cpp allocates no V cache at all for MLA models.</p>
      </Today>
      <p>Put the four side by side for one model. DeepSeek-V2 is the one that uses MLA; try the others to see what each variant would do to them.</p>
      <KvExplorer mode="compare" />
      <Ask q={questionsS207['ch20/20.7/q-mla-bytes']} label="Work it out" />
      <Ask q={questionsS207['ch20/20.7/q-mla-ai']} />

      <h2>Mixture of experts</h2>
      <p>
        The last technique is orthogonal to the cache. A mixture of experts gives each layer several specialised
        feed-forward sub-networks and a gate that picks a few of them for each token, so the parameter count can be huge
        while the arithmetic per token stays small.
      </p>
      <Source section="20.7" pages="510" />
      <p>
        What it does not shrink is the memory: any token may be routed to any expert, so all of them must be loaded. Nor
        does it touch the KV cache.
      </p>
      <Today checked="2026-11-20 (ch20-facts E7)">
        <p>Mixtral: Mistral 7B’s architecture with 8 feed-forward experts per layer, 2 chosen per token; “each token has access to 47B parameters, but only uses 13B active parameters during inference.”</p>
      </Today>
      <Ask q={questionsS207['ch20/20.7/q-moe']} />
    </>
  );
}
