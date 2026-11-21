import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, NoteGroup, Simplification, Source, Today } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  DECODE_TOKENS,
  EXPLORE_PAGES,
  EXPLORE_POLICIES,
  POLICY_NAME,
  PRESETS,
  cost,
  explore,
  moved,
  scriptLabel,
  warmTraffic,
  type Counters,
  type Ev,
  type Policy,
  type PresetId,
  type ScriptKind,
  type ScriptOp,
  type SweepPolicy,
  type View,
} from '../../sim/managedmem';
import { PageFaultScene } from './PageFaultScene';

// Section C.8, Page fault handling (book pp. 628–629). Both explorers run on
// src/sim/managedmem.ts: `explore` for "Coherence by invalidation" and `warmTraffic`
// (decodeSweep) for "When the model does not fit".

const POLICY_HOW: Record<Policy, string> = {
  bulk: 'limited: every CPU-held page moves at each launch',
  migrate: 'the default: fault, then move the page',
  map: 'SetPreferredLocation(host) + SetAccessedBy(GPU)',
  replicate: 'SetReadMostly: copies on read, invalidation on write',
  one: '',
};
const ADD: { kind: ScriptKind; label: string }[] = [
  { kind: 'cpu-write', label: 'CPU writes' },
  { kind: 'cpu-read', label: 'CPU reads' },
  { kind: 'kernel-read', label: 'kernel reads' },
  { kind: 'kernel-write', label: 'kernel writes' },
];
const MAX_SCRIPT = 40;
const show = (v: number) => String(Number(v.toPrecision(6)));

/** A sixteen-page strip per memory, for one policy at one step. */
function Strips({ view, events, policy }: { view: View; events: Ev[]; policy: Policy }) {
  const pages = view.m?.pages ?? [];
  const cw = 15;
  const gap = 2.5;
  const x0 = 34;
  const touched = new Set(events.filter((e) => e.page >= 0 && e.e !== 'error').map((e) => e.page));
  const invalidated = new Set(events.filter((e) => e.e === 'invalidate').map((e) => `${e.proc}:${e.page}`));
  const remote = new Set(events.filter((e) => e.e === 'remote').map((e) => e.page));
  const W = x0 + EXPLORE_PAGES * (cw + gap);
  return (
    <svg viewBox={`0 0 ${W} 62`} style={{ width: '100%', maxWidth: 360, display: 'block' }} role="img" aria-label={`Pages under ${POLICY_NAME[policy]}`}>
      {(['cpu', 'gpu'] as const).map((side, r) => (
        <g key={side}>
          <text x={0} y={r * 26 + 19} fontSize="10" fontWeight="600" style={{ fill: side === 'gpu' ? 'var(--global)' : 'var(--ink)' }}>
            {side.toUpperCase()}
          </text>
          {pages.map((p, i) => {
            const here = p.loc === side || p.loc === 'both';
            const mappedRemote = side === 'gpu' && policy === 'map' && p.loc === 'cpu';
            const gone = invalidated.has(`${side}:${i}`);
            const x = x0 + i * (cw + gap);
            const y = r * 26 + 6;
            return (
              <g key={i} data-side={side} data-page={i} data-present={here ? 'yes' : mappedRemote ? 'mapped' : 'no'}>
                <rect x={x} y={y} width={cw} height={18} rx="2"
                  fill={here ? (side === 'gpu' ? 'var(--global)' : 'var(--ink-2)') : 'transparent'}
                  fillOpacity={here ? (side === 'gpu' ? 0.85 : 0.55) : 1}
                  stroke={gone ? 'var(--fail)' : mappedRemote ? 'var(--global)' : here ? 'none' : 'var(--rule)'}
                  strokeDasharray={mappedRemote || gone ? '3 2' : here ? undefined : '2 2'} strokeWidth={gone ? 1.6 : 1} />
                {touched.has(i) && (here || (side === 'gpu' && remote.has(i))) ? (
                  <rect x={x - 1.5} y={y - 1.5} width={cw + 3} height={21} rx="3" fill="none" stroke="var(--ink)" strokeWidth="1.4" />
                ) : null}
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

function stepSummary(events: Ev[]): string {
  const n = (f: (e: Ev) => boolean) => events.filter(f).length;
  const parts: string[] = [];
  const h = n((e) => e.e === 'move' && e.dir === 'h2d');
  const d = n((e) => e.e === 'move' && e.dir === 'd2h');
  const gf = n((e) => e.e === 'fault' && e.proc === 'gpu');
  const cf = n((e) => e.e === 'fault' && e.proc === 'cpu');
  const r = n((e) => e.e === 'remote');
  const iv = n((e) => e.e === 'invalidate');
  if (h) parts.push(`${h} to GPU`);
  if (d) parts.push(`${d} to CPU`);
  if (gf) parts.push(`${gf} GPU ${gf === 1 ? 'fault' : 'faults'}`);
  if (cf) parts.push(`${cf} CPU ${cf === 1 ? 'fault' : 'faults'}`);
  if (r) parts.push(`${r} remote`);
  if (iv) parts.push(`${iv} ${iv === 1 ? 'invalidation' : 'invalidations'}`);
  return parts.length ? parts.join(', ') : 'nothing crossed';
}

function Coherence() {
  const [presetId, setPresetId] = useState<PresetId | 'own'>('figC1');
  const [script, setScript] = useState<ScriptOp[]>(PRESETS[0].script);
  const [n, setN] = useState(0);
  const [from, setFrom] = useState(0);
  const [to, setTo] = useState(15);
  const [perMove, setPerMove] = useState(3);
  const [perRemote, setPerRemote] = useState(1);
  const runs = useMemo(() => Object.fromEntries(EXPLORE_POLICIES.map((p) => [p, explore(p, script)])) as Record<Policy, ReturnType<typeof explore>>, [script]);
  const total = script.length;
  const at = Math.min(n, total);

  const load = (id: PresetId) => {
    setPresetId(id);
    setScript(PRESETS.find((p) => p.id === id)!.script);
    setN(0);
  };
  // A script the reader builds: keep them at the end of what they are building.
  const append = (op: ScriptOp) => {
    if (script.length >= MAX_SCRIPT) return;
    setPresetId('own');
    setScript([...script, op]);
    setN(script.length + 1);
  };
  const add = (kind: ScriptKind) => append({ kind, from: Math.min(from, to), to: Math.max(from, to) });
  const costs = EXPLORE_POLICIES.map((p) => cost(runs[p].steps[at].counters, perMove, perRemote));
  const best = Math.min(...costs);

  return (
    <Bench
      title="Coherence by invalidation"
      task="Load a preset, step it, and compare the four policies. Then build your own script."
      foot={
        <>
          Filled: the page has a valid copy in that memory. Dashed amber in the GPU row: mapped, read over the link. Red dashes:
          a copy invalidated by this step. A dark outline: touched by this step. Costs count pages crossing the link (either
          way) and remote page accesses; invalidations and faults are shown but not priced.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Preset
          <select value={presetId} onChange={(e) => e.target.value !== 'own' && load(e.target.value as PresetId)} aria-label="Preset">
            {PRESETS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
            <option value="own" disabled={presetId !== 'own'}>your script</option>
          </select>
        </label>
        <label className="field">
          Pages from
          <select value={from} onChange={(e) => setFrom(+e.target.value)} aria-label="First page">
            {Array.from({ length: EXPLORE_PAGES }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
        <label className="field">
          to
          <select value={to} onChange={(e) => setTo(+e.target.value)} aria-label="Last page">
            {Array.from({ length: EXPLORE_PAGES }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
        <span className="row" style={{ gap: '0.4rem' }} role="group" aria-label="Add to the script">
          {ADD.map((a) => (
            <button key={a.kind} className="quiet" onClick={() => add(a.kind)} disabled={script.length >= MAX_SCRIPT}>+ {a.label}</button>
          ))}
          <button className="quiet" onClick={() => append({ kind: 'sync', from: 0, to: 0 })} disabled={script.length >= MAX_SCRIPT}>+ synchronise</button>
          <button className="quiet" onClick={() => { setPresetId('own'); setScript([]); setN(0); }} disabled={!script.length}>Clear</button>
        </span>
      </div>

      <ol aria-label="Script" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem 0.4rem', listStyle: 'none', padding: 0, margin: '0 0 0.8rem' }}>
        {script.length === 0 ? <li className="muted">The script is empty. Add an operation.</li> : null}
        {script.map((s, i) => (
          <li key={i}>
            <button className="quiet" aria-pressed={i === at - 1} onClick={() => setN(i + 1)} data-op={i + 1}
              style={{ padding: '0.15rem 0.45rem', fontSize: '0.76rem', opacity: i < at ? 1 : 0.6 }}>
              {i + 1}. {scriptLabel(s)}
            </button>
          </li>
        ))}
      </ol>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setN(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={at >= total}>Step</button>
        <button onClick={() => setN(total)} disabled={at >= total}>Run to the end</button>
        <span className="muted" aria-live="polite">step {at} of {total}</span>
        <label className="field" style={{ marginLeft: 'auto' }}>
          cost per page moved
          <input type="number" min={0} step="any" value={perMove} onChange={(e) => setPerMove(Math.max(0, Number(e.target.value) || 0))} aria-label="cost per page moved" />
        </label>
        <label className="field">
          cost per remote page access
          <input type="number" min={0} step="any" value={perRemote} onChange={(e) => setPerRemote(Math.max(0, Number(e.target.value) || 0))} aria-label="cost per remote page access" />
        </label>
      </div>

      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>
        {at === 0 ? <>Sixteen managed pages, just allocated: none placed yet. Press Step.</> : <>Step {at}: <b>{scriptLabel(script[at - 1])}</b>.</>}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(19rem, 1fr))' }}>
        {EXPLORE_POLICIES.map((p, k) => {
          const st = runs[p].steps[at];
          const c: Counters = st.counters;
          const errs = runs[p].errors.filter((e) => e.step <= at);
          const crashed = st.crashed;
          return (
            <div key={p} className="panel" data-policy={p} data-cost={costs[k]} data-moved={moved(c)}>
              <div className="row" style={{ justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                <b>{POLICY_NAME[p]}</b>
                {at > 0 && costs[k] === best && !crashed ? <span className="chip ok" data-best="">cheapest so far</span> : null}
                {crashed ? <span className="chip fail">segmentation fault</span> : null}
              </div>
              <div className="muted" style={{ fontSize: '0.74rem', marginBottom: '0.4rem' }}>{POLICY_HOW[p]}</div>
              <Strips view={st.view} events={st.events} policy={p} />
              <div style={{ margin: '0.35rem 0', fontSize: '0.78rem' }} data-step-summary="">
                {at === 0 ? '' : st.skipped ? 'did not run: the program had stopped' : `this step: ${stepSummary(st.events)}`}
              </div>
              {errs.map((e, i) => (
                <div key={i} className="t-fail" style={{ fontSize: '0.76rem', marginBottom: '0.3rem' }} data-error={e.kind}>
                  step {e.step}: {e.message}.
                </div>
              ))}
              <table className="threads" aria-label={`Totals under ${POLICY_NAME[p]}`}>
                <tbody>
                  <tr><td>pages to GPU</td><td data-total="h2d">{c.h2d}</td><td>to CPU</td><td data-total="d2h">{c.d2h}</td></tr>
                  <tr><td>GPU faults</td><td data-total="gpuFaults">{c.gpuFaults}</td><td>CPU faults</td><td data-total="cpuFaults">{c.cpuFaults}</td></tr>
                  <tr><td>remote accesses</td><td data-total="remote">{c.remote}</td><td>invalidations</td><td data-total="invalidations">{c.invalidations}</td></tr>
                  <tr><td><b>cost</b></td><td data-total="cost"><b>{show(costs[k])}</b></td><td colSpan={2} className="muted">{moved(c)} × {perMove} + {c.remote} × {perRemote}</td></tr>
                </tbody>
              </table>
            </div>
          );
        })}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- oversubscription
const CAP = 100; // device memory in the model, in pages
const FRACS = Array.from({ length: 151 }, (_, i) => +(0.5 + i * 0.01).toFixed(2));
const SWEEP_LABEL: Record<SweepPolicy, string> = {
  lru: 'fault and migrate, least-recently-used eviction',
  prefix: 'keep a prefix resident, map the rest from host memory',
};

function useDeviceMemory() {
  const { reports } = useReports();
  const gpus: { backend: string; name: string; bytes: number }[] = [];
  let metal: { name: string; bytes: number } | null = null;
  for (const [b, r] of Object.entries(reports)) {
    const d = r.device ?? {};
    const g = Number(d.totalGlobalMem);
    if (b !== 'emu' && b !== 'race' && g > 0) gpus.push({ backend: b, name: String(d.name ?? b), bytes: g });
    const w = Number(d.recommendedMaxWorkingSetSize);
    if (b === 'metal' && w > 0) metal = { name: String(d.name ?? 'Metal device'), bytes: w };
  }
  return { gpus, metal };
}
const gb = (bytes: number) => (bytes / 1e9).toLocaleString('en-US', { maximumFractionDigits: 1 });

function Overflow() {
  const [frac, setFrac] = useState(0.8);
  const [policy, setPolicy] = useState<SweepPolicy>('lru');
  const { gpus, metal } = useDeviceMemory();
  const curves = useMemo(
    () => ({
      lru: FRACS.map((f) => warmTraffic(Math.round(f * CAP), CAP, 'lru').h2d / CAP),
      prefix: FRACS.map((f) => warmTraffic(Math.round(f * CAP), CAP, 'prefix').remote / CAP),
    }),
    [],
  );
  const W = Math.round(frac * CAP);
  const tr = warmTraffic(W, CAP, policy);
  const crossing = policy === 'lru' ? tr.h2d : tr.remote;
  // Plot geometry.
  const PW = 560, PH = 250, pad = { l: 52, r: 16, t: 14, b: 40 };
  const X = (f: number) => pad.l + ((f - 0.5) / 1.5) * (PW - pad.l - pad.r);
  const Y = (v: number) => PH - pad.b - (v / 2) * (PH - pad.t - pad.b);
  const path = (vs: number[]) => FRACS.map((f, i) => `${X(f).toFixed(1)},${Y(vs[i]).toFixed(1)}`).join(' ');
  return (
    <Bench
      title="When the model does not fit"
      task="Drag the weights past 100% of device memory, once with each placement."
      foot={
        <>
          The model's device memory is {CAP} pages. Each token reads every weight page once, in the same order; the numbers are
          for a token once the first few have warmed things up. Amber: pages migrated in per token (and as many evicted back
          out). Ink: pages read over the link per token.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ flex: '1 1 16rem' }}>
          Weights: {Math.round(frac * 100)}% of device memory
          <input type="range" min={0.5} max={2} step={0.01} value={frac} onChange={(e) => setFrac(+e.target.value)} aria-label="Weights as a fraction of device memory" style={{ width: '100%' }} />
        </label>
        <span className="seg" role="group" aria-label="Placement">
          {(['lru', 'prefix'] as SweepPolicy[]).map((p) => (
            <button key={p} aria-pressed={p === policy} onClick={() => setPolicy(p)}>{SWEEP_LABEL[p]}</button>
          ))}
        </span>
      </div>
      <div className="cols plot">
        <svg viewBox={`0 0 ${PW} ${PH}`} role="img" aria-label="Pages crossing the link per token against weight size" style={{ width: '100%', maxWidth: PW, display: 'block' }}>
          <rect x={pad.l} y={pad.t} width={PW - pad.l - pad.r} height={PH - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
          {[0.5, 1, 1.5, 2].map((f) => (
            <g key={f}>
              <line x1={X(f)} x2={X(f)} y1={pad.t} y2={PH - pad.b} stroke="var(--rule-soft)" />
              <text x={X(f)} y={PH - pad.b + 15} fontSize="11" textAnchor="middle">{f * 100}%</text>
            </g>
          ))}
          {[0, 0.5, 1, 1.5, 2].map((v) => (
            <g key={v}>
              <line x1={pad.l} x2={PW - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
              <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{v}×</text>
            </g>
          ))}
          <text x={(PW + pad.l) / 2} y={PH - 6} fontSize="11.5" textAnchor="middle">weights, as a share of device memory</text>
          <text x={13} y={(PH - pad.b + pad.t) / 2} fontSize="11" textAnchor="middle" transform={`rotate(-90 13 ${(PH - pad.b + pad.t) / 2})`}>crossing per token (× device memory)</text>
          <line x1={X(1)} x2={X(1)} y1={pad.t} y2={PH - pad.b} stroke="var(--ink-3)" strokeDasharray="4 3" />
          <polyline points={path(curves.lru)} fill="none" stroke="var(--global)" strokeWidth={policy === 'lru' ? 2.6 : 1.2} opacity={policy === 'lru' ? 1 : 0.45} data-curve="lru" />
          <polyline points={path(curves.prefix)} fill="none" stroke="var(--ink)" strokeWidth={policy === 'prefix' ? 2.6 : 1.2} opacity={policy === 'prefix' ? 1 : 0.45} data-curve="prefix" />
          <line x1={X(frac)} x2={X(frac)} y1={pad.t} y2={PH - pad.b} stroke="var(--ink)" strokeWidth="1" />
          <circle cx={X(frac)} cy={Y(crossing / CAP)} r={4.5} fill={policy === 'lru' ? 'var(--global)' : 'var(--ink)'} data-marker="" />
        </svg>
        <div className="panel" data-readout="" data-weights={W} data-crossing={crossing} data-policy={policy}>
          <p style={{ marginTop: 0 }}>
            <b>{W}</b> weight pages, <b>{CAP}</b> pages of device memory.
          </p>
          {policy === 'lru' ? (
            <p data-line="lru">
              Per token, once warm: <b>{tr.h2d}</b> pages migrated in and <b>{tr.d2h}</b> evicted out, with {tr.faults} GPU faults.
              {tr.h2d === W && W > CAP ? ' Every page, every token.' : tr.h2d === 0 ? ' Nothing: it all fits.' : ''}
            </p>
          ) : (
            <p data-line="prefix">
              Per token, once warm: <b>{tr.remote}</b> pages read over the link, nothing migrated.
              {tr.remote ? ` That is the excess, ${Math.round((tr.remote / W) * 100)}% of the weights.` : ' Nothing: it all fits.'}
            </p>
          )}
          {gpus.length ? (
            gpus.map((g) => (
              <p key={g.backend} className="muted" data-device={g.backend}>
                On your {g.name} ({g.backend}, {gb(g.bytes)} GB of device memory) this would be about{' '}
                <b>{gb((crossing / CAP) * g.bytes)} GB</b> crossing the link for every token.
              </p>
            ))
          ) : (
            <p className="muted" data-device="none">
              No CUDA or HIP device report yet, so no gigabytes: run{' '}
              <code>python3 labs/pmpp.py run appC/lab1 --bench</code> on such a machine to see this ratio in your own
              device's bytes.
            </p>
          )}
          {metal ? (
            <p className="muted" data-device="metal">
              Your {metal.name} reports a <code>recommendedMaxWorkingSetSize</code> of {gb(metal.bytes)} GB. Apple calls it an
              approximation of how much the GPU can allocate without affecting performance, and does not say what happens above
              it. Metal documents no migration, so this model does not describe it.
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

export const questionsSC8: Record<string, Question> = {
  'appC/C.8/q-pages-moved': {
    kind: 'numeric',
    id: 'appC/C.8/q-pages-moved',
    prompt: (
      <>
        Sixty-four managed pages, all written by the CPU. The next kernel reads 4 of them. How many fewer pages move to the GPU
        under fault-driven migration than under the limited mode's bulk transfer at the launch?
      </>
    ),
    answer: 60,
    unit: 'pages',
    explain: (
      <p>
        Bulk transfer moves every managed page the CPU has modified before the kernel starts: 64. With GPU page faults nothing
        moves at the launch, and only the 4 pages the kernel touches fault and migrate. 64 − 4 = 60. Load the preset "kernel
        touches 1 page of 16" in the explorer below to see the same thing at a smaller scale.
      </p>
    ),
    source: 'book §C.6, p. 627 and §C.8, p. 628; notes/today/appC-facts.md A10, A11',
  },
  'appC/C.8/q-pingpong': {
    kind: 'choice',
    id: 'appC/C.8/q-pingpong',
    prompt: <>The CPU and the GPU take turns updating the same small counter page, every step. Which placement keeps traffic lowest?</>,
    options: [
      { label: 'The default, fault and migrate, which puts the page wherever the latest access was.', why: 'The page moves on every turn, once each way: the explorer\'s ping-pong preset moves it 8 times in 4 rounds.' },
      { label: 'Read-mostly replication, so that each side keeps its own copy of the page and reads it locally.', why: 'Every turn writes, and every write invalidates the other side\'s copy, which then has to come back.' },
      { label: 'Keep it in one memory and map it for the other processor (SetPreferredLocation plus SetAccessedBy).', correct: true },
      { label: 'Prefetch it to the GPU before every launch, and back to the CPU before each host turn, so no access faults.', why: 'That is the same migration on every turn, started earlier. It does not remove the trips.' },
    ],
    explain: (
      <p>
        A page that both sides touch in turn migrates on every turn. Pin it to one memory and map it for the other processor,
        and nothing migrates: each GPU access goes over the link instead. The book: mapping "can sometimes be faster than
        migration". NVIDIA documents the combination: pin regions to CPU memory "and establish a direct mapping from the GPU" with{' '}
        <code>cudaMemAdviseSetPreferredLocation</code> and <code>cudaMemAdviseSetAccessedBy</code>.
      </p>
    ),
    source: 'book §C.8, p. 628; notes/today/appC-facts.md A13; the explorer\'s model',
  },
  'appC/C.8/q-cliff': {
    kind: 'choice',
    id: 'appC/C.8/q-cliff',
    prompt: (
      <>
        Weights are 105% of device memory. Decoding reads all the weights in the same order every token, and eviction is least
        recently used. About what fraction of the weight pages migrate per token, once warm?
      </>
    ),
    options: [
      { label: 'About 5%: just the part that does not fit.', why: 'That is what keeping a prefix resident gives you, over the link. Least-recently-used eviction does worse.' },
      { label: 'About 50%, since half of each sweep still finds its pages resident.', why: 'There is no partial hit rate in a cyclic sweep larger than memory under this eviction rule.' },
      { label: 'All of them.', correct: true },
      { label: 'None: once warm, the pages stay.', why: 'Only if they all fit.' },
    ],
    explain: (
      <>
        <p>
          Once memory is full, the least recently used page is the one read longest ago, and in a cyclic sweep that is the
          one needed soonest. Page 100 evicts page 0, the first page the next token reads; bringing page 0 back evicts page 5,
          which that token needs five pages later; and so on. Every access misses. Keep a prefix resident and
          map the rest instead, and only the excess, about 5% of the weights, crosses the link per token.
        </p>
        <p>
          This is a property of least-recently-used eviction on a cyclic sweep, not a documented fact about any driver: our
          sources do not say how NVIDIA's driver chooses what to evict. Treat the cliff as a prediction to test.
        </p>
      </>
    ),
    source: 'the explorer\'s model (src/sim/managedmem.ts, decodeSweep); eviction policy not documented (notes/today/appC-facts.md A11, A13)',
  },
  'appC/C.8/q-hints': {
    kind: 'choice',
    id: 'appC/C.8/q-hints',
    prompt: (
      <>
        A decode loop keeps its logits in managed memory: a kernel writes them, then host code samples the next token.
        With a <code>cudaMemAdvise(..., cudaMemAdviseSetPreferredLocation, ...)</code> call on the logits, every output
        matches the reference. Delete that one call and some tokens come out different. What is the most likely cause?
      </>
    ),
    options: [
      {
        label:
          'A missing synchronise: the host reads logits the kernel may still be writing, and the hint only shifted the timing.',
        correct: true,
      },
      {
        label: 'The hint kept the pages coherent: without a preferred location the driver can migrate a stale copy.',
        why: 'Advice changes where pages live and what moving them costs, never what a load returns.',
      },
      {
        label: 'Advice is binding on some systems, so removing it changes the semantics there.',
        why: 'Advice is a hint wherever it is accepted. NVIDIA says hints only affect performance.',
      },
      {
        label: 'Sampling is random, so different tokens are expected.',
        why: 'Then the version with the hint would differ too. Only the version without it does, so something in the program depends on timing.',
      },
    ],
    explain: (
      <p>
        NVIDIA: "Performance hints never impact the correctness of an application. Performance hints only affect performance."
        So when removing a hint changes the output, the hint was hiding a bug, and in a loop like this the usual one is a
        host read before <code>cudaDeviceSynchronize</code> (or a wait on the kernel's stream). The explorer's four policies
        always agree on the data for the same reason: they differ only in the counters. The book's own rule from
        section C.6 still holds: ordering concurrent access is your job.
      </p>
    ),
    source: 'notes/today/appC-facts.md A13; book §C.6, p. 626 for synchronising concurrent access',
  },
  'appC/C.8/q-signature': {
    kind: 'choice',
    id: 'appC/C.8/q-signature',
    prompt: (
      <>
        Your decode loop prefetches its managed KV pages with <code>cudaMemPrefetchAsync</code>. It must build from one
        source for CUDA 13 and for ROCm 10.2, where a header maps names with lines like{' '}
        <code>#define cudaMemPrefetchAsync hipMemPrefetchAsync</code>. How do you write the prefetch?
      </>
    ),
    options: [
      {
        label: 'Leave it to the rename: write the CUDA 13 call once, and the header turns it into the HIP call.',
        why: 'The names map; the signatures do not. With ROCm 10.2\'s default headers hipMemPrefetchAsync takes an int device, not a location and flags.',
      },
      {
        label: 'Write the CUDA 12 form with an int device, which both CUDA 13 and ROCm 10.2 accept.',
        why: 'CUDA 13.0 removed the int forms: under the plain name it takes a cudaMemLocation and flags. The CUDA build fails.',
      },
      {
        label:
          'Branch with #if: the CUDA 13 call with a cudaMemLocation on CUDA; on HIP the int-device hipMemPrefetchAsync, or hipMemPrefetchAsync_v2.',
        correct: true,
      },
      {
        label: 'Leave the prefetch out of the HIP build; HIP has no prefetch call, only advice.',
        why: 'HIP has the call. It takes the CUDA 12 arguments under the plain name.',
      },
    ],
    explain: (
      <p>
        CUDA 13.0 changed <code>cudaMemPrefetchAsync</code> and <code>cudaMemAdvise</code> to take a <code>cudaMemLocation</code>,
        and gave prefetch a <code>flags</code> argument; the <code>int</code> forms are gone. HIP kept the CUDA 12 shape under the
        plain names, and its location forms are the <code>_v2</code> variants. HIPIFY marks the CUDA 13 forms unsupported. So
        branch with <code>#if</code> on the platform, or map to the <code>_v2</code> name. Leaving the prefetch out of one
        build would still be correct, since NVIDIA says the call "only serves to improve performance"; what is wrong in that
        option is the claim that HIP has none. Quoted from the headers and references; nothing here was compiled.
      </p>
    ),
    source: 'notes/today/appC-facts.md A14, B6',
  },
};

export function SC8() {
  return (
    <>
      <p>
        Starting with Pascal, a GPU can take a page fault: an access to memory it has no valid mapping for stops, the system
        fixes the mapping, and the access runs again. The book calls this the feature that makes Unified Memory seamless. Its
        first effect is to remove the limited mode's worst rule. Nothing has to be flushed to the GPU before a launch any
        more, because anything the kernel needs can be fetched when it is touched.
      </p>
      <Source section="C.8" pages="628" />
      <p>
        Coherence works by invalidation. When the CPU or the GPU modifies a managed page, it may invalidate the other side's
        copy, using page mapping and protection mechanisms. At a launch the runtime no longer
        brings every GPU copy up to date. If the kernel touches a page whose GPU copy the CPU invalidated, the GPU faults, the
        page is brought up to date, and execution resumes.
      </p>
      <Source section="C.8" pages="628" />
      <p>
        A page that is not in GPU memory at all is handled the same way: it faults and migrates. Or, if it will be touched only
        occasionally, it can be mapped into the GPU's address space and read over the interconnect, since "mapping on access can
        sometimes be faster than migration". That generalises the zero-copy access of <a href={href('appC', 'C.2')}>C.2</a>{' '}
        from a special kind of pinned host memory to all of host memory and other GPUs' memory, system-wide. It works the
        other way too: a CPU function that dereferences a page now living in GPU memory is still served, perhaps more slowly,
        which lets you call CPU libraries nobody has ported. And because both sides use the same pointer values, a GPU can
        follow a linked structure the CPU built, and applications that keep hundreds of gigabytes "in core", such as CAD, can
        put it in reach of a GPU.
      </p>
      <Source section="C.8" pages="628–629" />

      <h2>What NVIDIA documents today</h2>
      <p>
        NVIDIA's guide now states the rules the book leaves implicit. Managed memory is usually placed in the memory of the
        processor that first touches it, migrated when another processor uses it, and moved at the granularity of pages
        (software coherence) or cache lines (hardware coherence, as on Grace Hopper). Oversubscription is allowed: you may
        allocate more managed memory than the GPU has. And a software page fault is what you will see in the animation below:
        the owner loses its mapping, the requester gains one, and the page moves, "an expensive operation, and the amount of
        work is proportional to the page size".
      </p>
      <NoteGroup>
        <Today checked="2026-11-20 (notes/today/appC-facts.md, A11, A12)">
          <p>
            Page sizes, as NVIDIA states them now and "subject to change": 4 KiB on x86_64 CPUs, 4 to 64 KiB on Arm CPUs, and
            GPUs "prefer 2MiB physical pages or larger". "TLB misses are generally significantly more expensive on the GPU."
          </p>
          <p>
            Scope: Pascal-style faulting covers memory allocated as managed. Plain <code>malloc</code> or <code>mmap</code> memory
            is GPU-accessible only with HMM (Linux 6.1.24 or later, compute capability 7.5 or later, driver 535 or later with
            open kernel modules) or ATS (Grace Hopper, Grace Blackwell). <code>cudaMalloc</code> memory stays out of the CPU's
            reach everywhere.
          </p>
        </Today>
        <Simplification>
          <p>
            The animation follows one page under fault and migrate with software coherence. No timing, one page size, and
            no hardware-coherent system.
          </p>
        </Simplification>
      </NoteGroup>
      <PageFaultScene />

      <h2>Migrate or map: who decides?</h2>
      <p>
        The book says a page "may be mapped" instead of migrated, and does not say who chooses. NVIDIA does: by default the
        Unified Memory driver decides, by heuristics, and "per default cudaMallocManaged migrates based on a fault-and-migrate
        mechanism". You can steer it. <code>cudaMemAdvise</code> with <code>SetPreferredLocation</code> on the host plus{' '}
        <code>SetAccessedBy</code> for the GPU pins a region to CPU memory and maps it for the GPU. <code>SetReadMostly</code>{' '}
        lets both sides hold a copy, "trading off read bandwidth for write bandwidth". <code>cudaMemPrefetchAsync</code> moves
        a range before anyone faults on it. On hardware-coherent systems a fourth mechanism, access counters, moves pages
        toward the processor that uses them most.
      </p>
      <Today checked="2026-11-20 (notes/today/appC-facts.md, A13, A14)">
        <p>
          The hints arrived with CUDA 8 and Pascal. None is needed for correctness: prefetch "is not required for functionality
          and only serves to improve performance".
        </p>
      </Today>
      <p>
        Before you explore, settle the number the book's two eras disagree on most.
      </p>
      <Ask q={questionsSC8['appC/C.8/q-pages-moved']} label="Work it out" />
      <p>
        The explorer runs one script of CPU and kernel accesses on sixteen managed pages under four policies side by side,
        and keeps a running bill at prices you set.
      </p>
      <Simplification>
        <p>
          Pages, not bytes. Four fixed policies stand in for the driver's undocumented heuristics, and each applies to all
          sixteen pages at once. Costs are your numbers: the book gives none. No prefetching or overlap. Hardware-coherent
          systems, with cache-line granularity and access counters, are not modelled.
        </p>
      </Simplification>
      <Coherence />
      <p>
        What to look for. Load "kernel touches 1 page of 16": bulk transfer moves all sixteen for one touched page. Load the
        ping-pong preset: fault and migrate moves the counter page on every turn, and mapping, which moves nothing, wins at any
        prices where a remote access costs less than two migrations. Then the decode loop, {DECODE_TOKENS} tokens of it:
        mapping pays for every weight page on every token, and read-mostly, which copies the weights once, wins at the default
        prices, while its invalidations land on the one page that is written. The best policy is a property of the access
        pattern, region by region, which is why the hints take an address range.
      </p>
      <Ask q={questionsSC8['appC/C.8/q-pingpong']} />
      <Ask q={questionsSC8['appC/C.8/q-hints']} />

      <h2>The signature trap</h2>
      <p>
        If you use these hints in a <code>.cu</code> file that must also build for HIP, mind the version. CUDA 13.0 changed{' '}
        <code>cudaMemPrefetchAsync</code> and <code>cudaMemAdvise</code> to take a <code>cudaMemLocation</code> instead of a
        device number, and gave prefetch a <code>flags</code> argument. HIP did not follow under the same names.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }} aria-label="Prefetch and advise signatures">
        <thead>
          <tr><th></th><th>CUDA 13.5</th><th>CUDA 12.9</th><th>HIP, ROCm 10.2 default headers</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>prefetch</td>
            <td><code style={{ whiteSpace: 'normal' }}>cudaMemPrefetchAsync(const void *devPtr, size_t count, struct cudaMemLocation location, unsigned int flags, cudaStream_t stream = 0)</code></td>
            <td><code style={{ whiteSpace: 'normal' }}>cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream = 0)</code>; the location form was <code>cudaMemPrefetchAsync_v2</code></td>
            <td><code style={{ whiteSpace: 'normal' }}>hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, hipStream_t stream)</code>; the location form is <code>hipMemPrefetchAsync_v2</code></td>
          </tr>
          <tr>
            <td>advise</td>
            <td><code style={{ whiteSpace: 'normal' }}>cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, struct cudaMemLocation location)</code></td>
            <td><code style={{ whiteSpace: 'normal' }}>cudaMemAdvise(..., int device)</code>, plus <code>cudaMemAdvise_v2</code></td>
            <td><code style={{ whiteSpace: 'normal' }}>hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device)</code> by default (<code>HIP_FORCE_API_VERSION</code> below 800; the default is 717), plus <code>hipMemAdvise_v2</code></td>
          </tr>
        </tbody>
      </table>
      <p>
        Renaming works for <code>cudaMallocManaged</code> and <code>cudaMemLocation</code>, which HIPIFY maps one for one, but
        a plain <code>#define</code> of the prefetch compiles a CUDA 13 call against the wrong overload set. Branch with{' '}
        <code>#if</code> on the platform, or map to the <code>_v2</code> names. These are quoted signatures, unverified: nothing here was compiled.
      </p>
      <Today checked="2026-11-20 (notes/today/appC-facts.md, A14, B6)">
        <p>
          HIPIFY's table marks the CUDA 13 forms of both calls unsupported. llama.cpp's HIP build calls the <code>int</code>{' '}
          form of <code>cudaMemAdvise</code> for its coarse-grain hint.
        </p>
      </Today>
      <Ask q={questionsSC8['appC/C.8/q-signature']} />

      <h2>When the model does not fit</h2>
      <p>
        Oversubscription sounds like the answer to a model larger than your GPU: allocate it managed and let pages come and
        go. Decoding, though, reads every weight once per token, in the same order, and which page leaves when memory is full
        decides everything.
      </p>
      <Beyond>
        <p>
          Least-recently-used eviction is our stand-in. The driver's eviction policy is not documented in our sources, so the
          cliff below is a property of the model to test on a real machine, not a statement about any driver.
        </p>
      </Beyond>
      <p>
        The explorer runs that loop on a device memory of 100 pages, with two placements: the default, fault and migrate with
        least-recently-used eviction, and an advised one that keeps a prefix resident and maps the rest. Before you drag the
        slider: what happens just past 100%?
      </p>
      <Simplification>
        <p>
          Device memory holds weights only: no activations, no KV cache. One page size, no prefetch, no overlap of transfers
          with compute, and the count is pages, not time.
        </p>
      </Simplification>
      <Ask q={questionsSC8['appC/C.8/q-cliff']} />
      <Overflow />
      <p>
        That is the model's answer to why a model a little too big for VRAM need not run only a little slower: with fault
        and migrate and an eviction rule like this one, every token can pay for the whole model at the speed of the
        interconnect. llama.cpp's <code>GGML_CUDA_ENABLE_UNIFIED_MEMORY=1</code> exists to swap to system RAM "instead of
        crashing" on Linux, and this is the trade it offers: running, at the price of the link. On Windows, NVIDIA's driver
        has a "Sysmem Fallback Policy" that lets ordinary allocations spill into system memory (NVIDIA's "shared memory",
        not the on-chip kind), "albeit at lower speeds";
        NVIDIA states no Linux equivalent, so do not count on one. Placement is what decides the cost: keep what fits
        resident and map the rest, and only the excess crosses the link.
      </p>
      <Today checked="2026-11-20 (notes/today/appC-facts.md, A18, C5, D3, D4, D5)">
        <p>
          On a Mac, Metal documents no oversubscription mechanism. <code>recommendedMaxWorkingSetSize</code> is advice, and
          Apple does not say what happens above it. Residency sets (macOS 15) make allocations resident explicitly. llama.cpp
          keeps its memory-mapped weights wired with residency sets and reports <code>recommendedMaxWorkingSetSize</code> as its
          memory budget; MLX exposes <code>set_wired_limit</code>.
        </p>
      </Today>

      <h2>On HIP and Metal</h2>
      <p>
        AMD has the same eras, chosen per GPU and per process rather than by year. With <code>HSA_XNACK=1</code> and an HMM
        driver, a kernel's access to a non-resident managed page faults and migrates, and when device memory is full a page is
        sent back to host memory, so oversubscription works. Some AMD GPUs cannot take page faults at all; there, if you want
        local accesses, every page has to be migrated before the kernel is dispatched, which is the book's limited mode on AMD
        hardware today. On the MI300A, where CPU and GPU share one HBM, nothing migrates, and XNACK is still needed: faults
        there establish mappings, not moves.
      </p>
      <p>
        On Metal, the only page fault Apple documents is an error: <code>MTLCommandBufferErrorPageFault</code>, "an
        unserviceable GPU page fault", caused by a read-write attribute mismatch or an out-of-bounds access. Whether Apple GPUs
        service any faults, as demand paging, is not documented, so this course teaches neither answer.
      </p>
      <Beyond>
        <p>
          The book covers NVIDIA only. AMD from its unified memory page and MI300A post (appC-facts B5, B8); Apple from the
          macOS 27.0 SDK headers (C5).
        </p>
      </Beyond>
      <p>
        Next, <a href={href('appC', 'C.9')}>C.9</a> gives you the address space itself to arrange. To measure what migration
        costs on your machine, lab C.1 on the <a href={href('appC', 'labs')}>labs page</a> feeds one kernel with copies, mapped
        memory and managed memory.
      </p>
      <p>
        Back to <a href={href('appC', 'start')}>the map</a>: this section's step is the one marked C.7–C.8. The step after
        C.9, beyond the book, is where page faults reach plain <code>malloc</code> memory.
      </p>
    </>
  );
}
