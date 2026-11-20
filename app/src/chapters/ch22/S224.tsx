import { useEffect, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source } from '../../components/notes';
import { href } from '../../lib/router';
import {
  COULOMB,
  GiB,
  LLAMA2_7B,
  SERIES,
  advantage,
  crossover,
  curves,
  kvBytes,
  slope,
  tieFactor,
  time,
  type Preset,
  type Series,
} from '../../sim/formulation';

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const WRAP: CSSProperties = { flexWrap: 'wrap' };
const int = (v: number) => Math.round(v).toLocaleString('en-US');
const sig = (v: number, d = 3) => (v >= 1000 ? int(v) : Number(v.toPrecision(d)).toLocaleString('en-US'));
const sci = (v: number) => {
  if (v < 1e4) return sig(v);
  const e = Math.floor(Math.log10(v));
  const m = v / 10 ** e;
  return `${Number(m.toFixed(1))} × 10${String(e).replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c])}`;
};

/**
 * True while the viewport is at most 600 px wide. The plots then use a narrower drawing so
 * that their labels stay legible on a phone. (Chapter 10 has the same helper; a shared one
 * would belong in src/lib.)
 */
function useNarrow(px = 600): boolean {
  const query = `(max-width: ${px}px)`;
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches);
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setNarrow(m.matches);
    on();
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, [query]);
  return narrow;
}

// ---------------------------------------------------------------- the explorer
const PRESET_LABEL: Record<Preset, string> = {
  coulomb: "Electrostatic map (the book's)",
  kv: 'Decoding a sequence (ours, from chapter 20)',
};
const NAMES: Record<Preset, { direct: string; reformed: string; directText: string; reformedText: string; size: string; unit: (x: number) => string }> = {
  coulomb: {
    direct: 'direct summation',
    reformed: 'cutoff with bins',
    directText: 'direct summation',
    reformedText: 'cutoff summation with bins',
    size: 'volume, relative to the reference system',
    unit: (x) => `${sig(x)}× the volume`,
  },
  kv: {
    direct: 'recompute the prefix',
    reformed: 'KV cache',
    directText: 'recomputing the prefix',
    reformedText: 'the KV cache',
    size: 'position of the token being generated',
    unit: (x) => `token ${int(x)}`,
  },
};
const SERIES_STYLE: Record<Series, { stroke: string; dash?: string; width: number }> = {
  'direct-slow': { stroke: 'var(--ink-3)', width: 1.8 },
  'direct-fast': { stroke: 'var(--ink)', width: 2.2 },
  'reformed-slow': { stroke: 'var(--ink-3)', dash: '7 4', width: 1.8 },
  'reformed-fast': { stroke: 'var(--ink)', dash: '7 4', width: 2.2 },
};
const seriesName = (p: Preset, s: Series, S: number) =>
  `${s.startsWith('direct') ? NAMES[p].direct : NAMES[p].reformed}, ${s.endsWith('fast') ? `GPU (${S}×)` : 'CPU'}`;

/** Size sliders: volume in tenths of a decade from 0.01 to 1000; token position in quarter-doublings from 1 to 65,536. */
const SIZE_SLIDER: Record<Preset, { max: number; at: number; of: (v: number) => number }> = {
  coulomb: { max: 50, at: 20, of: (v) => Number((10 ** (v / 10 - 2)).toPrecision(3)) },
  kv: { max: 64, at: 40, of: (v) => Math.round(2 ** (v / 4)) },
};
/** Points along a log axis, for drawing. */
const logOf = (v: number, lo: number, hi: number) => Math.exp(Math.log(lo) + (v / 100) * (Math.log(hi) - Math.log(lo)));
const SPEEDS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000];

function CostPlot({ preset, x, S, cross }: { preset: Preset; x: number; S: number; cross: number | null }) {
  const c = curves(preset);
  const narrow = useNarrow();
  const W = narrow ? 400 : 640;
  const H = narrow ? 300 : 330;
  const pad = { l: narrow ? 48 : 64, r: 12, t: 16, b: 44 };
  const xs = Array.from({ length: 121 }, (_, i) => logOf((100 * i) / 120, c.min, c.max));
  const all = SERIES.flatMap((s) => xs.map((v) => time(c, s, v, S)));
  const yMin = Math.min(...all) / 2;
  const yMax = Math.max(...all) * 2;
  const X = (v: number) => pad.l + ((Math.log10(v) - Math.log10(c.min)) / (Math.log10(c.max) - Math.log10(c.min))) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const xt: number[] = [];
  for (let e = Math.ceil(Math.log10(c.min)); e <= Math.floor(Math.log10(c.max)); e += narrow ? 2 : 1) xt.push(10 ** e);
  const yt: number[] = [];
  const step = Math.max(1, Math.ceil((Math.log10(yMax) - Math.log10(yMin)) / (narrow ? 5 : 7)));
  for (let e = Math.ceil(Math.log10(yMin)); e <= Math.floor(Math.log10(yMax)); e += step) yt.push(10 ** e);
  return (
    <div>
    <div className="ui" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem 1.1rem', fontSize: '0.85em', margin: '0 0 0.3rem' }}>
      {SERIES.map((s) => (
        <span key={s} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }} data-legend={s}>
          <svg width="28" height="8" aria-hidden="true">
            <line x1="1" x2="27" y1="4" y2="4" stroke={SERIES_STYLE[s].stroke} strokeWidth={SERIES_STYLE[s].width} strokeDasharray={SERIES_STYLE[s].dash} />
          </svg>
          {s.startsWith('direct') ? NAMES[preset].direct : NAMES[preset].reformed}, {s.endsWith('fast') ? 'GPU' : 'CPU'}
        </span>
      ))}
    </div>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Cost against problem size, log-log" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {xt.map((v) => (
        <g key={'x' + v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
            {v < 1 ? v : v >= 1000 ? v / 1000 + 'k' : v}
          </text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
            10{String(Math.round(Math.log10(v))).replace(/-?\d/g, (ch) => ('-' === ch ? '⁻' : '⁰¹²³⁴⁵⁶⁷⁸⁹'[+ch]))}
          </text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 8} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
        {NAMES[preset].size}
      </text>
      <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`} style={{ fill: 'var(--ink-2)' }}>
        time (operations ÷ speed)
      </text>
      {SERIES.map((s) => (
        <polyline
          key={s}
          data-series={s}
          points={xs.map((v) => `${X(v)},${Y(time(c, s, v, S))}`).join(' ')}
          fill="none"
          stroke={SERIES_STYLE[s].stroke}
          strokeWidth={SERIES_STYLE[s].width}
          strokeDasharray={SERIES_STYLE[s].dash}
        />
      ))}
      <line x1={X(x)} x2={X(x)} y1={pad.t} y2={H - pad.b} stroke="var(--ink-2)" strokeWidth="1" />
      {cross != null ? (
        <g data-mark="crossover">
          <circle cx={X(cross)} cy={Y(time(c, 'direct-fast', cross, S))} r="5.5" fill="none" stroke="var(--ink)" strokeWidth="2" />
          <text x={X(cross) + 8} y={Y(time(c, 'direct-fast', cross, S)) + 18} fontSize="11.5" style={{ fill: 'var(--ink)' }}>
            tie at {preset === 'coulomb' ? `${sig(cross)}×` : `token ${int(cross)}`}
          </text>
        </g>
      ) : null}
    </svg>
    </div>
  );
}

function MemoryPlot({ length, batch }: { length: number; batch: number }) {
  const W = 300;
  const H = 170;
  const pad = { l: 46, r: 10, t: 10, b: 34 };
  const lo = 1;
  const hi = 65536;
  const yLo = kvBytes(lo, 1) / GiB;
  const yHi = kvBytes(hi, 256) / GiB;
  const X = (v: number) => pad.l + (Math.log10(v / lo) / Math.log10(hi / lo)) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yLo)) / (Math.log10(yHi) - Math.log10(yLo))) * (H - pad.t - pad.b);
  const line = (b: number) => `${X(lo)},${Y(kvBytes(lo, b) / GiB)} ${X(hi)},${Y(kvBytes(hi, b) / GiB)}`;
  const yt = [1 / 1024, 1, 1024];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="KV-cache memory against length" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {yt.map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 5} y={Y(v) + 4} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
            {v < 1 ? '1 MiB' : v === 1 ? '1 GiB' : '1 TiB'}
          </text>
        </g>
      ))}
      {[1, 256, 65536].map((v) => (
        <text key={v} x={X(v)} y={H - pad.b + 14} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
          {v >= 1024 ? v / 1024 + 'k' : v}
        </text>
      ))}
      <text x={(W + pad.l) / 2} y={H - 6} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
        sequence length (tokens)
      </text>
      <polyline points={line(1)} fill="none" stroke="var(--ink-3)" strokeWidth="1.6" />
      {batch > 1 ? <polyline points={line(batch)} fill="none" stroke="var(--ink)" strokeWidth="2" /> : null}
      <circle cx={X(length)} cy={Y(kvBytes(length, batch) / GiB)} r="4" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
      <text x={X(hi) - 4} y={Y(kvBytes(hi, 1) / GiB) + 14} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
        batch 1
      </text>
      {batch > 1 ? (
        <text x={X(hi) - 4} y={Y(kvBytes(hi, batch) / GiB) - 6} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink)' }}>
          batch {batch}
        </text>
      ) : null}
    </svg>
  );
}

function Reformulate() {
  const [preset, setPreset] = useState<Preset>('coulomb');
  const c = curves(preset);
  const [pos, setPos] = useState<Record<Preset, number>>({ coulomb: SIZE_SLIDER.coulomb.at, kv: SIZE_SLIDER.kv.at });
  const [si, setSi] = useState(SPEEDS.indexOf(1000));
  const [batch, setBatch] = useState(8);
  const S = SPEEDS[si];
  const x = SIZE_SLIDER[preset].of(pos[preset]);
  const cross = crossover(c, S);
  const adv = advantage(c, x, S);
  const slopeDirect = slope((v) => time(c, 'direct-fast', v, S), x);
  const slopeReformed = slope((v) => time(c, 'reformed-fast', v, S), x);
  const n = NAMES[preset];

  return (
    <Bench
      title="Reformulate or accelerate?"
      task="Drag the speed-up and watch what moves. Then drag the size past the tie."
      foot={
        preset === 'coulomb' ? (
          <>
            Reference system (size 1), our constants: {int(COULOMB.atoms)} atoms, {int(COULOMB.points)} grid points, a
            cutoff neighbourhood of {int(COULOMB.neighbourhood)} atoms. One unit per atom-grid point pair. Model:{' '}
            <code>src/sim/formulation.ts</code>.
          </>
        ) : (
          <>
            One layer of hidden width {int(LLAMA2_7B.hidden)} (the book&rsquo;s Llama 2 7B example, p. 506); one unit per
            multiply-add, softmax ignored. Recompute: 3·T·d² + 2·T²·d. Cached: 3·d² + 2·T·d. Model:{' '}
            <code>src/sim/formulation.ts</code>.
          </>
        )
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Problem
          <span className="seg" role="group" aria-label="Problem" style={WRAP}>
            {(['coulomb', 'kv'] as Preset[]).map((p) => (
              <button key={p} aria-pressed={preset === p} onClick={() => setPreset(p)}>
                {PRESET_LABEL[p]}
              </button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            Size: <b data-value="size">{n.unit(x)}</b>
          </span>
          <input
            type="range"
            min={0}
            max={SIZE_SLIDER[preset].max}
            value={pos[preset]}
            aria-label="Problem size"
            onChange={(e) => setPos({ ...pos, [preset]: Number(e.target.value) })}
            style={{ width: '14rem' }}
          />
        </label>
        <label className="field">
          <span>
            GPU speed-up over the CPU: <b data-value="speedup">{S}×</b>
          </span>
          <input type="range" min={0} max={SPEEDS.length - 1} value={si} aria-label="GPU speed-up" onChange={(e) => setSi(Number(e.target.value))} style={{ width: '14rem' }} />
        </label>
        {preset === 'kv' ? (
          <label className="field">
            <span>
              Batch: <b data-value="batch">{batch}</b> sequences
            </span>
            <input type="range" min={0} max={8} value={Math.log2(batch)} aria-label="Batch" onChange={(e) => setBatch(2 ** Number(e.target.value))} style={{ width: '10rem' }} />
          </label>
        ) : null}
      </div>
      <div className="cols plot">
        <CostPlot preset={preset} x={x} S={S} cross={cross} />
        <div className="panel" aria-live="polite" data-readout="formulation" data-crossover={cross ?? ''} data-slope-direct={slopeDirect.toFixed(3)} data-slope-reformed={slopeReformed.toFixed(3)}>
          <p style={{ margin: 0 }}>
            At {n.unit(x)}, {n.directText} on the GPU{' '}
            {Math.abs(adv - 1) < 0.005 ? (
              <>
                takes <b data-value="advantage">the same time</b> as
              </>
            ) : (
              <>
                is <b data-value="advantage">{adv > 1 ? `${sig(adv)} times faster` : `${sig(1 / adv)} times slower`}</b> than
              </>
            )}{' '}
            {n.reformedText} on the CPU.
          </p>
          <p style={{ margin: '0.6rem 0 0' }}>
            They tie at <b data-value="crossover">{cross == null ? 'no size shown' : n.unit(cross)}</b>
            {cross != null ? `; beyond it ${n.reformedText} wins even on the slower processor.` : '.'}
          </p>
          <p style={{ margin: '0.6rem 0 0' }}>
            Slope on this plot at this size: {n.direct} <b data-value="slope-direct">{slopeDirect.toFixed(2)}</b>,{' '}
            {n.reformed} <b data-value="slope-reformed">{slopeReformed.toFixed(2)}</b>. The speed-up shifts every GPU curve
            down by the same amount and leaves its slope alone.
          </p>
          <table className="data" style={{ margin: '0.7rem 0 0' }}>
            <tbody>
              {SERIES.map((s) => (
                <tr key={s} data-series-row={s}>
                  <td>{seriesName(preset, s, S)}</td>
                  <td className="num">{sci(time(c, s, x, S))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {preset === 'kv' ? (
        <div className="cols" style={{ marginTop: '0.9rem', alignItems: 'start' }}>
          <div className="panel" data-readout="kv-memory" data-bytes={kvBytes(x, batch)}>
            <p style={{ margin: 0 }}>
              <b>What the cache costs instead</b>: memory. Eq. 20.8 with the book&rsquo;s Llama 2 7B (32 layers, hidden
              width 4096, 2 bytes): {int(kvBytes(1, 1) / 1024)} KiB per token per sequence.
            </p>
            <p style={{ margin: '0.5rem 0 0' }}>
              {batch} × {int(x)} tokens: <b data-value="kv-gib">{sig(kvBytes(x, batch) / GiB)} GiB</b>, growing in
              proportion to both. That memory, with the weights, is what limits the batch (chapter 20, p. 505) in
              section <a href={href('ch22', '22.5')}>22.5</a>.
            </p>
          </div>
          <div className="panel">
            <MemoryPlot length={x} batch={batch} />
          </div>
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const TIE = tieFactor(10, 2, 1);
const EX_S = 1000;
const EX_ADV = advantage(curves('coulomb'), 1, EX_S);
const EX_CROSS = crossover(curves('coulomb'), EX_S)!;

export const questionsS224: Record<string, Question> = {
  'ch22/22.4/q-what-relaxed': {
    kind: 'choice',
    id: 'ch22/22.4/q-what-relaxed',
    prompt: <>Which of these reformulations gives up exactness to cut the complexity?</>,
    options: [
      { label: 'Cutoff summation: far atoms are not summed one by one but treated together.', correct: true },
      {
        label: 'The KV cache: each decode step reuses the keys and values of earlier steps.',
        why: 'It reuses the very numbers it would recompute, so nothing is given up. It spends memory instead.',
      },
      {
        label: 'Splitting a long sum across the warps of a block and adding the partial sums.',
        why: 'That changes the order of the additions, and so the rounding, but not how many there are.',
      },
      {
        label: 'Batching the requests of several users into one matrix product per layer.',
        why: 'It changes how the work is grouped, not the result any user gets.',
      },
    ],
    explain: (
      <p>
        Only cutoff summation changes what is computed: the far atoms&rsquo; individual contributions are dropped and
        handled by a cheaper collective method, and the work falls from the square of the volume to linear. The KV cache
        changes the cost from recomputation to storage without changing a single number.
      </p>
    ),
    source: 'ch. 22 §22.4, p. 538; ch. 20 §20.4, pp. 488–492',
  },
  'ch22/22.4/q-kv-cost': {
    kind: 'choice',
    id: 'ch22/22.4/q-kv-cost',
    prompt: (
      <>
        The KV cache makes the step that generates token T about T times cheaper than recomputing the whole prefix.
        What does it spend instead, and where does that bite next?
      </>
    ),
    options: [
      {
        label: 'Memory that grows with the length of each sequence and with the number of sequences, which caps how many fit in a batch.',
        correct: true,
      },
      {
        label: 'Arithmetic: appending each new key and value to the cache costs about as much as the recomputation it replaces.',
        why: 'An append copies one row of K and one of V per layer. The recomputation it replaces is about T times the whole cached step.',
      },
      {
        label: 'Accuracy, because the cached keys and values were computed for a shorter prefix and grow stale as tokens arrive.',
        why: 'Under the causal mask the keys and values of earlier tokens never change (ch. 20 p. 489), so the cache is exact.',
      },
    ],
    explain: (
      <p>
        Eq. 20.8: b · N · 2 · l · h_q · d · p bytes for b sequences of length N. With the book&rsquo;s Llama 2 7B
        figures, one sequence of 4096 tokens holds 2 GiB of cache. Every sequence in a batch has its own, so the cache
        decides how large the batch of section <a href={href('ch22', '22.5')}>22.5</a> can be. The cache has a second
        consequence: each step is now a vector-matrix product, which chapter 20 calls bandwidth-bound, and that is the
        other reason section 22.5 batches.
      </p>
    ),
    source: 'ours, from ch. 20 §20.4, pp. 488–492, and §20.6, pp. 505–506 (Eq. 20.8)',
  },
  'ch22/22.4/q-why-domain': {
    kind: 'choice',
    id: 'ch22/22.4/q-why-domain',
    prompt: <>Why does the book say problem formulation needs domain insight as well as computer-science skill?</>,
    options: [
      {
        label: 'Which accuracy can go is a physics judgement, and the binning that makes it fast is a computing technique: neither alone gets there.',
        correct: true,
      },
      {
        label: 'Because in practice the domain scientists write the kernels, and the programmer only tunes what they wrote.',
        why: 'The book argues the opposite: the programmer can work with domain scientists to transform the problem itself (p. 529).',
      },
      {
        label: 'Because GPUs cannot run generic numerical methods, so every method must be rewritten for the domain.',
        why: 'They can; the book says that route tends to give high complexity or limited parallelism, not that it fails to run (p. 538).',
      },
    ],
    explain: (
      <p>
        Cutoff binning needs both halves. Knowing that far atoms contribute little, and that their effect can be handled
        collectively, is physics. Turning the remaining computation into a grid-centric kernel that knows which atoms to
        scan is a decomposition and a data structure. The KV cache has the same shape: knowing that earlier keys never
        change is a fact about the causal model, and a cache is a computing technique.
      </p>
    ),
    source: 'ch. 22 §22.4, p. 538',
  },
};

export function S224() {
  const q = questionsS224;
  return (
    <>
      <p>
        The book calls problem formulation arguably the most consequential step in developing a parallel application, and
        cites Wing&rsquo;s paper on computational thinking for it. The earlier sections chose algorithms and decompositions
        for a problem as given. This one asks whether the problem should be given that way at all.
      </p>
      <Source section="22.4" pages="538" />
      <p>
        What not to do, according to the book, is to take a generic numerical method, map the domain problem onto it
        step by step, and implement the steps with the skills of this book. That route tends to give high computational
        complexity, limited parallelism, or both. What to do instead is rethink the method and relax some of the
        constraints on how it is carried out.
      </p>
      <p>
        The book&rsquo;s example is cutoff binning from chapter 21. Deciding that some accuracy can be traded for a
        dramatic drop in complexity takes domain expertise. Making the result fast takes computer science: a grid-centric
        decomposition and the binning techniques that tell each block which atoms to scan.
      </p>
      <Source section="22.4" pages="538" figs="ch. 21 §21.5" />

      <h2>A constant factor against a slope</h2>
      <p>
        The difference between accelerating a method and reformulating it shows up as soon as the problem grows. Predict
        it first.
      </p>
      <Predict
        id="ch22/22.4/p-crossover"
        prompt={
          <>
            At today&rsquo;s problem size, a GPU running direct summation is 10 times faster than a CPU running cutoff
            summation. Direct cost grows with the square of the volume, cutoff cost linearly. By what factor must the
            volume grow before the two take the same time?
          </>
        }
        unit="×"
      >
        {(v) => (
          <p>
            <b>{sig(TIE)}</b>. Write the direct time as k·V² and the cutoff time as 10·k·V; they meet where V = 10.{' '}
            {Math.abs(v - TIE) < 0.05 * TIE
              ? 'That is what you had.'
              : v > TIE
                ? 'Both costs grow, so the ratio between them grows only in proportion to the volume: a lead of 10 is gone at 10 times the volume, not 100.'
                : 'A factor of √10 would be right if the cutoff cost stayed fixed. It grows too, so the ratio between the two grows only in proportion to the volume.'}{' '}
            In the explorer below, a speed-up of {EX_S}× gives exactly this case: the GPU&rsquo;s direct sum is{' '}
            {sig(EX_ADV)} times faster at size 1, and the curves tie at {sig(EX_CROSS)}× the volume.
          </p>
        )}
      </Predict>
      <Reformulate />
      <p>
        Drag the speed-up down from 1000× and back up. Each GPU curve slides with it, and the tie moves to larger sizes
        as the speed-up grows, but the slopes in the readout do not change: a faster processor divides the cost by a
        constant. The reformulation changes the
        slope, so beyond some size it wins even on the slower processor, and no constant can postpone that size forever.
        For the electrostatic map, every factor of 10 in hardware speed only moves the tie by a factor of 10 in volume.
      </p>
      <Simplification>
        <p>
          Costs are operation counts with constants we chose, stated under the plot. No memory traffic, which matters for
          a cached decode step: it is bandwidth-bound. The cutoff&rsquo;s loss of accuracy and the cost of handling far
          atoms are not modelled. The same speed-up applies to both methods on the GPU.
        </p>
      </Simplification>

      <h2>A reformulation you already use</h2>
      <p>
        Chapter 20&rsquo;s KV cache is the same move in your own field. Without it, generating token T recomputes K and V
        for the whole prefix and the attention over all of it, every step. With it, each step computes one new row of Q,
        K and V, appends K and V to the cache, and attends with one row. In the count used here the cached step is
        exactly T times cheaper, so the slope drops by one at every length, and a GPU recomputing the prefix S times
        faster than a CPU with a cache loses to it from token S on. Switch the explorer to the second problem.
      </p>
      <Beyond>
        <p>
          This bridge is ours. The KV cache is in chapter 20 (§20.4, pp. 488–492; Eq. 20.8, pp. 505–506), not in this
          section.
        </p>
      </Beyond>
      <p>
        Unlike cutoff summation, the cache gives up nothing numerically. It spends memory instead: keys and values for
        every token of every sequence, growing with length and with batch. That memory is what limits batching in section{' '}
        <a href={href('ch22', '22.5')}>22.5</a>, so the reformulation that made decoding cheap is also what sets the
        next constraint. Chapter 20&rsquo;s KV-cache sections are{' '}
        <a href={href('ch20', '20.4')}>20.4</a> and <a href={href('ch20', '20.6')}>20.6</a>.
      </p>

      <h2>Where the large gains come from</h2>
      <p>
        The book&rsquo;s closing claim for the section is that formulation is where the largest performance advantages,
        and genuinely new capabilities, come from: a high-fidelity simulation of a biochemical system of a size that
        traditional methods could not reach, for example. It needs insight from both the domain and computer science, and
        the book says the payoff is worth the effort.
      </p>
      <Source section="22.4" pages="538" />
      <Ask q={q['ch22/22.4/q-what-relaxed']} />
      <Ask q={q['ch22/22.4/q-kv-cost']} />
      <Ask q={q['ch22/22.4/q-why-domain']} />
    </>
  );
}
