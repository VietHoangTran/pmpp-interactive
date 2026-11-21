import { useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { useLabStatus } from './status';
import { BOOK_CAPACITY, deviceSlots, residentUpperBound } from '../../sim/coarsen';
import {
  CH6_DEVICE_R,
  FIG1021,
  SOFTMAX_BATCHES,
  SOFTMAX_BLOCK,
  SOFTMAX_ROW,
  coarsenSteps,
  fewestStepsC,
  justEnoughC,
  maxUsefulC,
  softmaxPreset,
  sweepC,
  type CoarsenResult,
} from '../../sim/multiblock';
import { functionSource } from '../../sim/toolchain';

const LAB = 'ch10/lab1_sum';
const fmtInt = (n: number) => n.toLocaleString('en-US');
const pct = (v: number) => `${Math.round(v * 100)}%`;

const SUM_TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', output_ref: 'global',
  partialSums_s: 'shared', input_s: 'shared',
  partialSum: 'reg', segment: 'reg', i: 'reg', t: 'reg', c: 'reg', idx: 'reg',
};

// ---------------------------------------------------------------- the listing
/**
 * 1-based lines of the kernel to mark: the segment, the first load, and the serial
 * loop up to its closing brace.
 */
export function coarseningLines(text: string): number[] {
  const lines = text.split('\n');
  const seg = lines.findIndex((l) => /\bsegment\s*=/.test(l));
  const loop = lines.findIndex((l) => /for\s*\(.*\bc\s*=\s*1/.test(l));
  if (seg < 0 || loop < 0) return [];
  let end = loop;
  for (let d = 0, j = loop; j < lines.length; j++) {
    d += (lines[j].match(/\{/g) ?? []).length - (lines[j].match(/\}/g) ?? []).length;
    if (d <= 0 && j > loop) {
      end = j;
      break;
    }
  }
  const first = lines.slice(seg + 1, loop).findIndex((l) => /partialSum\s*=/.test(l));
  return [seg + 1, ...(first >= 0 ? [seg + 2 + first] : []), ...Array.from({ length: end - loop + 1 }, (_, i) => loop + 1 + i)];
}

function CoarsenListing() {
  const status = useLabStatus(LAB);
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const k = cu ? functionSource(cu, 'CoarsenedSumReductionKernel') : null;
  if (!k)
    return (
      <p className="muted">
        The kernel is loaded from <code>labs/{LAB}/solution/kernel.cu</code> once lab 10.1 is in place.
      </p>
    );
  const loop = coarseningLines(k.text);
  return (
    <Code
      tabs={[{ label: 'CUDA', code: k.text, tiers: SUM_TIERS, status: status.cu, highlight: loop }]}
      caption={
        <>
          <code>CoarsenedSumReductionKernel</code> from lab 10.1's reference solution (<code>labs/{LAB}/solution/kernel.cu</code>,
          lines {k.firstLine} to {k.lastLine}). The marked lines are the coarsening: the segment that grows by the factor, the
          first load, and the serial loop that runs before any barrier.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- interactive: coarsening explorer
const BLOCKS = [8, 32, 64, 128, 256, 512, 1024];
type RSource = 'fig' | 'slider' | 'ch6' | `yours:${string}`;

interface Yours { backend: string; device: string; R: number }
/** CUDA and HIP reports that carry multiProcessorCount and maxBlocksPerMultiProcessor (Metal reports neither). */
function yoursFrom(reports: Reports, block: number): Yours[] {
  return Object.keys(reports)
    .filter((b) => b === 'cuda' || b === 'hip')
    .flatMap((b) => {
      const d = deviceSlots(reports[b].device as Record<string, unknown>);
      return d ? [{ backend: b, device: String(reports[b].device.name ?? b), R: residentUpperBound(d, block) }] : [];
    });
}

function WaveBar({ r, scale, y, label }: { r: CoarsenResult; scale: number; y: number; label: string }) {
  const X0 = 150;
  const width = (steps: number) => (steps / scale) * 520;
  const perWave = r.stepsPerBlock;
  const drawWaves = r.waves <= 48;
  return (
    <g data-bar={label}>
      <text x={X0 - 8} y={y + 14} fontSize="11.5" textAnchor="end">{label}</text>
      {drawWaves ? (
        Array.from({ length: r.waves }, (_, w) => {
          const x = X0 + width(w * perWave);
          return (
            <g key={w}>
              <rect x={x} y={y} width={width(r.fullPerBlock)} height={20} fill="var(--ink-2)" />
              <rect x={x + width(r.fullPerBlock)} y={y} width={width(r.underPerBlock)} height={20} fill="url(#cx-hatch)" stroke="var(--ink-2)" strokeWidth="0.6" />
            </g>
          );
        })
      ) : (
        <>
          <rect x={X0} y={y} width={width(r.fullSteps)} height={20} fill="var(--ink-2)" />
          <rect x={X0 + width(r.fullSteps)} y={y} width={width(r.underSteps)} height={20} fill="url(#cx-hatch)" stroke="var(--ink-2)" strokeWidth="0.6" />
        </>
      )}
      <text x={X0 + width(r.totalSteps) + 6} y={y + 14} fontSize="11.5">
        {fmtInt(r.totalSteps)} steps{drawWaves ? '' : ` (${fmtInt(r.waves)} waves)`}
      </text>
    </g>
  );
}

function SweepChart({ base, maxC, C, onPick }: { base: { N: number; block: number; R: number; rows: number }; maxC: number; C: number; onPick: (c: number) => void }) {
  const sw = sweepC(base, maxC);
  const top = Math.max(...sw.map((r) => r.totalSteps));
  const best = fewestStepsC(base, maxC);
  const rule = justEnoughC(base, maxC);
  const X0 = 40;
  const PW = 640;
  const PH = 120;
  const VW = X0 + PW + 20;
  const bw = PW / sw.length;
  const bx = (c: number) => X0 + (c - 1) * bw;
  // Never drawn smaller than its own size, so the text stays legible: a narrow screen
  // scrolls the chart sideways inside its box instead. The legend is HTML, so it wraps.
  return (
    <>
    <p style={{ margin: '0.6rem 0 0.2rem', fontSize: '0.85rem', fontWeight: 600 }}>
      Total steps for C = 1 to {maxC} (click a bar to pick that factor)
    </p>
    <div style={{ overflowX: 'auto' }} data-part="sweep">
    <svg viewBox={`0 0 ${VW} ${PH + 38}`} role="img" aria-label="Total steps for every coarsening factor" style={{ width: '100%', minWidth: VW, display: 'block' }}
      data-best={best} data-rule={rule}>
      {sw.map((r) => {
        const h = (r.totalSteps / top) * PH;
        const c = r.input.C;
        return (
          <rect key={c} x={bx(c)} y={20 + PH - h} width={Math.max(0.8, bw - (bw > 3 ? 1 : 0))} height={h}
            fill={c === C ? 'var(--ink)' : r.tooFew ? 'var(--rule)' : 'var(--ink-3)'} onClick={() => onPick(c)} style={{ cursor: 'pointer' }}>
            <title>{`C = ${c}: ${r.blocks} blocks, ${r.waves} wave${r.waves === 1 ? '' : 's'}, ${r.totalSteps} steps`}</title>
          </rect>
        );
      })}
      <line x1={X0} x2={X0 + PW} y1={20 + PH} y2={20 + PH} stroke="var(--rule)" />
      <text x={X0 - 4} y={24} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{fmtInt(top)}</text>
      <text x={X0 - 4} y={20 + PH} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>0</text>
      <text x={bx(1)} y={33 + PH} fontSize="10" style={{ fill: 'var(--ink-2)' }}>1</text>
      <text x={bx(sw.length) + bw} y={33 + PH} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{sw.length}</text>
      <polygon points={`${bx(best) + bw / 2},${23 + PH} ${bx(best) + bw / 2 - 5},${32 + PH} ${bx(best) + bw / 2 + 5},${32 + PH}`} fill="var(--ok)" />
      {rule !== best ? (
        <polygon points={`${bx(rule) + bw / 2},${23 + PH} ${bx(rule) + bw / 2 - 5},${32 + PH} ${bx(rule) + bw / 2 + 5},${32 + PH}`} fill="var(--ink)" />
      ) : null}
    </svg>
    </div>
    <p style={{ margin: '0.3rem 0 0', fontSize: '0.85rem' }} data-part="sweep-legend">
      <span style={{ color: 'var(--ok)' }}>▲ fewest steps: C = {best}</span>
      {' · '}
      {rule !== best ? <span>▲ book's rule (largest C with at least R blocks): C = {rule}</span> : <span>the book's rule picks the same C</span>}
      <br />
      <span className="muted">Pale bars: fewer blocks than R, so part of the device idles.</span>
    </p>
    </>
  );
}

function CoarseningExplorer() {
  const { reports } = useReports();
  const [N, setN] = useState(1 << 20);
  const [nText, setNText] = useState(String(1 << 20));
  const [rows, setRows] = useState(1);
  const [block, setBlock] = useState(256);
  const [C, setCRaw] = useState(1);
  const [cText, setCText] = useState('1');
  const [src, setSrc] = useState<RSource>('ch6');
  const [slider, setSlider] = useState(100);

  const yours = yoursFrom(reports, block);
  const R = src === 'fig' ? 1 : src === 'ch6' ? CH6_DEVICE_R : src === 'slider' ? slider : (yours.find((y) => 'yours:' + y.backend === src)?.R ?? CH6_DEVICE_R);
  const setC = (c: number) => {
    const v = Math.max(1, Math.min(1024, Math.floor(c)));
    setCRaw(v);
    setCText(String(v));
  };
  const setLength = (n: number) => {
    setN(n);
    setNText(String(n));
  };
  const r = coarsenSteps({ N, block, C, R, rows });
  const baseline = coarsenSteps({ N, block, C: 1, R, rows });
  const useful = maxUsefulC(N, block);
  const base = { N, block, R, rows };
  // Chart range: wide enough to show the best factor and the current one, narrow enough
  // that the bars near the best stay readable.
  const reach = Math.min(512, useful);
  const maxC = Math.min(reach, Math.max(16, 3 * fewestStepsC(base, reach), C));

  const preset = (p: { N: number; block: number; C: number; R: RSource; rows?: number }) => {
    setLength(p.N);
    setBlock(p.block);
    setC(p.C);
    setSrc(p.R);
    setRows(p.rows ?? 1);
  };

  return (
    <Bench
      title="Coarsening explorer"
      task="Start from Fig. 10.21, then raise C on 2²⁰ floats (about a million) until the warning appears. Find a factor that is worse than both its neighbours."
      foot={
        <>
          Solid: fully used steps, every thread adding its own elements (2C − 1 per block). Hatched: the tree, where threads
          drop out (log2 of the block size per block). Blocks run in waves of R; a wave costs one block's steps, however few
          blocks it holds. The faint bar is C = 1 on the same device.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="ui muted">Presets:</span>
        <button onClick={() => preset({ ...FIG1021, C: 1, R: 'fig' })}>Fig. 10.21 (a)</button>
        <button onClick={() => preset({ ...FIG1021, C: 2, R: 'fig' })}>Fig. 10.21 (b)</button>
        <button onClick={() => preset({ N: 1 << 20, block: 256, C: 1, R: 'ch6' })}>2²⁰ floats</button>
        {SOFTMAX_BATCHES.map((b) => {
          const p = softmaxPreset(b);
          return (
            <button key={b} onClick={() => preset({ N: p.N, block: p.block, C: p.C, R: 'ch6', rows: b })}>
              softmax, batch {b}
            </button>
          );
        })}
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          N per row
          <input type="number" min={1} value={nText} aria-label="N"
            onChange={(e) => { setNText(e.target.value); const v = Math.floor(Number(e.target.value)); if (v >= 1 && v <= 1e10) setN(v); }}
            style={{ width: '8.5rem' }} />
        </label>
        <label className="field">
          Rows
          <input type="number" min={1} value={rows} aria-label="rows" onChange={(e) => setRows(Math.max(1, Math.floor(Number(e.target.value)) || 1))} style={{ width: '5rem' }} />
        </label>
        <label className="field">
          Threads per block
          <select value={block} onChange={(e) => setBlock(+e.target.value)} aria-label="threads per block">
            {BLOCKS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="field">
          Coarsening factor C
          <input type="number" min={1} max={1024} value={cText} aria-label="C"
            onChange={(e) => { setCText(e.target.value); const v = Math.floor(Number(e.target.value)); if (v >= 1 && v <= 1024) setCRaw(v); }}
            style={{ width: '5.5rem' }} />
        </label>
        <input type="range" min={1} max={Math.max(64, Math.min(512, useful))} value={Math.min(C, 512)} aria-label="C slider"
          onChange={(e) => setC(+e.target.value)} style={{ minWidth: '12rem' }} />
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="ui muted">Blocks at once, R:</span>
        <span className="seg" role="group" aria-label="Blocks at once">
          <button aria-pressed={src === 'fig'} onClick={() => setSrc('fig')}>1 (Fig. 10.21)</button>
          <button aria-pressed={src === 'ch6'} onClick={() => setSrc('ch6')}>{CH6_DEVICE_R} (the book's example, ch. 6)</button>
          <button aria-pressed={src === 'slider'} onClick={() => setSrc('slider')}>choose</button>
          {yours.map((y) => (
            <button key={y.backend} aria-pressed={src === 'yours:' + y.backend} onClick={() => setSrc(`yours:${y.backend}`)}>
              yours ({y.backend})
            </button>
          ))}
        </span>
        {src === 'slider' ? (
          <label className="field">
            R = {slider}
            <input type="range" min={1} max={2048} value={slider} onChange={(e) => setSlider(+e.target.value)} aria-label="R" />
          </label>
        ) : null}
      </div>
      <p className="muted" style={{ margin: '0 0 0.8rem' }} data-testid="r-note">
        {src.startsWith('yours:')
          ? (() => {
              const y = yours.find((x) => 'yours:' + x.backend === src)!;
              return <>{y.device}: multiProcessorCount × maxBlocksPerMultiProcessor, capped by thread slots for {block}-thread blocks, gives at most {fmtInt(y.R)}. Registers and shared memory can only lower it.</>;
            })()
          : yours.length
            ? <>A CUDA or HIP report on this machine gives a "yours" option. Metal reports neither number, so a Mac gets none.</>
            : <>"Yours" appears when a CUDA or HIP report carries multiProcessorCount and maxBlocksPerMultiProcessor. Metal reports neither, so a Mac has no such option.</>}
      </p>

      <div className="cols" style={{ marginBottom: '0.9rem' }}>
        <div className="panel" data-testid="outputs" data-blocks={r.blocks} data-waves={r.waves} data-total={r.totalSteps}
          data-full={r.fullSteps} data-under={r.underSteps} data-commits={r.commits} data-toofew={r.tooFew ? 1 : 0}>
          <p style={{ margin: 0 }}>
            Blocks: {rows > 1 ? <>{fmtInt(rows)} rows × </> : null}ceil({fmtInt(N)} / (2·{C}·{block})) = <b>{fmtInt(r.blocks)}</b>
          </p>
          <p style={{ margin: '0.35rem 0 0' }}>
            Waves: ceil({fmtInt(r.blocks)} / {fmtInt(R)}) = <b>{fmtInt(r.waves)}</b>
            {r.waves > 1 && r.lastWave < R ? <> (the last holds {fmtInt(r.lastWave)} blocks)</> : null}
          </p>
          <p style={{ margin: '0.35rem 0 0' }}>
            Per block: {r.fullPerBlock} fully used + {r.underPerBlock} underused = {r.stepsPerBlock} steps
          </p>
          <p style={{ margin: '0.35rem 0 0' }}>
            Total: <b>{fmtInt(r.totalSteps)}</b> steps ({fmtInt(r.fullSteps)} fully used, {fmtInt(r.underSteps)} underused)
          </p>
          <p style={{ margin: '0.35rem 0 0' }}>Atomic commits: <b>{fmtInt(r.commits)}</b>, one per block</p>
        </div>
        <div className="panel">
          {r.tooFew ? (
            <p style={{ margin: 0 }} data-testid="warning">
              <span className="chip fail">too few blocks</span> {fmtInt(r.blocks)} block{r.blocks === 1 ? '' : 's'} for {fmtInt(R)} slots:{' '}
              {pct(r.busyShare)} of the device has work, {pct(1 - r.busyShare)} sits idle for the whole kernel.
            </p>
          ) : (
            <p style={{ margin: 0 }} data-testid="warning">
              <span className="chip ok">the device is full</span> {pct(r.busyShare)} of the block slots are busy, wave by wave.
            </p>
          )}
          <p style={{ margin: '0.5rem 0 0' }}>
            Against C = 1 ({fmtInt(baseline.totalSteps)} steps, {fmtInt(baseline.blocks)} blocks):{' '}
            <b>{r.totalSteps < baseline.totalSteps ? `${fmt(baseline.totalSteps / r.totalSteps)}× fewer steps` : r.totalSteps === baseline.totalSteps ? 'the same' : `${fmt(r.totalSteps / baseline.totalSteps)}× more steps`}</b>.
          </p>
          {C > useful ? (
            <p style={{ margin: '0.5rem 0 0' }} className="muted">
              Past C = {useful} one block already covers a row; a larger factor only adds loads past the end.
            </p>
          ) : null}
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
      <svg viewBox="0 0 760 74" role="img" aria-label="Steps, wave by wave" style={{ width: '100%', minWidth: 760, display: 'block' }}>
        <defs>
          <pattern id="cx-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-2)" strokeWidth="1.3" />
          </pattern>
        </defs>
        <g opacity={0.4}>
          <WaveBar r={baseline} scale={Math.max(baseline.totalSteps, r.totalSteps)} y={6} label="C = 1" />
        </g>
        <WaveBar r={r} scale={Math.max(baseline.totalSteps, r.totalSteps)} y={40} label={`C = ${C}`} />
      </svg>
      </div>
      <SweepChart base={base} maxC={maxC} C={C} onPick={setC} />
    </Bench>
  );
}

// ---------------------------------------------------------------- the measured sweep
function MeasuredCoarsening({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports)
    .filter((b) => b !== 'race')
    .flatMap((b) => {
      const rows = benchRows(reports[b], LAB)
        .filter((r) => r.name === 'any')
        .flatMap((r) => {
          const c = num(r.coarse);
          const g = num(r.gb_per_s);
          return c != null && g != null ? [{ c, g, n: num(r.n), block: num(r.block) }] : [];
        })
        .sort((a, b) => a.c - b.c);
      return rows.length ? [{ b, device: String(reports[b].device.name ?? b), rows }] : [];
    });
  if (!found.length)
    return (
      <p>
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 10.1 --solution --bench</code> on a machine
        with a GPU, then reload the page: this box will chart bandwidth against the coarsening factor.
      </p>
    );
  return (
    <>
      {found.map((f) => {
        const best = f.rows.reduce((a, b) => (b.g > a.g ? b : a));
        const top = best.g;
        const mine = f.rows.reduce((a, b) => (Math.abs(b.c - predicted) < Math.abs(a.c - predicted) ? b : a));
        return (
          <div key={f.b} data-backend={f.b} data-best={best.c} style={{ marginBottom: '0.8rem' }}>
            <p>
              On <b>{f.device}</b> ({f.b})
              {best.n ? <>, {fmtInt(best.n)} floats{best.block ? <> in blocks of {best.block}</> : null},</> : null} the fastest factor
              measured was <b>C = {best.c}</b>, at {fmt(best.g)} GB/s.{' '}
              {mine.c === best.c
                ? 'That is the one you picked.'
                : `Your pick${mine.c !== predicted ? ` (nearest measured: C = ${mine.c})` : ''} ran at ${fmt(mine.g)} GB/s, ${Math.round((mine.g / top) * 100)}% of the fastest.`}
            </p>
            <div style={{ overflowX: 'auto' }}>
            <svg viewBox={`0 0 520 ${f.rows.length * 18 + 4}`} role="img" aria-label={`Bandwidth by coarsening factor on ${f.b}`} style={{ width: '100%', maxWidth: 520, minWidth: 520, display: 'block' }}>
              {f.rows.map((row, i) => (
                <g key={row.c}>
                  <text x={52} y={i * 18 + 13} fontSize="11" textAnchor="end">C = {row.c}</text>
                  <rect x={58} y={i * 18 + 3} width={(row.g / top) * 360} height={13} fill={row.c === best.c ? 'var(--ok)' : 'var(--ink-3)'} />
                  <text x={64 + (row.g / top) * 360} y={i * 18 + 13} fontSize="11">{fmt(row.g)} GB/s{row.c === best.c ? ' (fastest)' : ''}</text>
                </g>
              ))}
            </svg>
            </div>
          </div>
        );
      })}
      <p className="muted">
        Measurements from your machine, not expected values. The step model above has no memory time, and at this size the
        kernel is limited by memory, so the curve you measured need not have the model's shape.
      </p>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS1010: Record<string, Question> = {
  'ch10/10.10/q-fig1021': {
    kind: 'numeric',
    id: 'ch10/10.10/q-fig1021',
    prompt: (
      <>
        Blocks of 8 threads, N = 64, and the device runs one block at a time (R = 1), as in Fig. 10.21. How many steps in
        total with a coarsening factor of 4?
      </>
    ),
    answer: 10,
    unit: 'steps',
    explain: (
      <p>
        One block covers 2 × 4 × 8 = 64 elements, so one block does everything. Each thread adds its 8 elements in 7 fully
        used steps, then the 8 partial sums meet in a tree of log2 8 = 3 steps: 10. Without coarsening it is 4 blocks, run
        one after another, of 1 + 3 = 4 steps each: 16.
      </p>
    ),
    source: 'ch. 10 §10.10, pp. 247–248 (Fig. 10.21)',
  },
  'ch10/10.10/q-just-enough': {
    kind: 'choice',
    id: 'ch10/10.10/q-just-enough',
    prompt: (
      <>
        N = 2<sup>20</sup> floats, 256 threads per block, on a hypothetical device that runs 256 blocks at once. Which
        coarsening factor does the book's rule pick?
      </>
    ),
    options: [
      { label: 'C = 1: the most parallelism.', why: '2048 blocks are 8 waves; every block pays for its own tree and its own commit.' },
      { label: 'C = 4.', why: '512 blocks are two full waves: twice the tree phases and twice the commits of C = 8.' },
      { label: 'C = 8.', correct: true },
      { label: 'C = 16.', why: '128 blocks fill half the device; the other half has nothing to do.' },
    ],
    explain: (
      <p>
        The rule is to leave just enough blocks to fill the hardware. With C = 8 a block covers 2 × 8 × 256 = 4096 elements,
        so 2<sup>20</sup> / 4096 = 256 blocks: exactly one wave. In the step count that is 15 + 8 = 23 steps, against 30 for
        C = 4 and 39 for C = 16.
      </p>
    ),
    source: 'ch. 10 §10.10, p. 248',
  },
  'ch10/10.10/q-batch1': {
    kind: 'choice',
    id: 'ch10/10.10/q-batch1',
    prompt: (
      <>
        A softmax kernel gives each row one block. At batch 1, on a device that runs 100 blocks at once, what fraction of the
        block slots does the reduction keep busy, and what are the ways out?
      </>
    ),
    options: [
      { label: 'All of them: one block uses the whole device if it is large enough, up to 1,024 threads.', why: 'A block runs on one SM; the others have nothing to do.' },
      {
        label: 'About 1%. Split the row over blocks (a second pass or a grid barrier), or accept it when the softmax is a small part of the step.',
        correct: true,
      },
      { label: 'About 1%, and nothing can be done: a row must stay in one block for its max and its sum to agree with each other.', why: 'Section 10.9 is exactly how a row is split across blocks; the second pass makes the pieces agree.' },
      { label: 'About 1%; raise the coarsening factor, so that each thread covers more of the row.', why: 'Coarsening reduces the number of blocks further. It is the cause, not the cure.' },
    ],
    explain: (
      <p>
        One block of 100 slots is 1%. The two exits are the two halves of this chapter: more blocks per row, joined by a
        second pass or, where the whole grid fits on the device, a grid-wide barrier under a cooperative launch (llama.cpp's
        route for long rows); or keep one block per row and accept the idle device because the softmax is a small part of a
        decode step next to the matrix multiplications. Which is right is a measurement, which is what lab 10.3 is for.
      </p>
    ),
    source: 'ch. 10 §10.10, p. 248; notes/today/ch10-facts.md A8, D1',
  },
  'ch10/10.10/q-stride': {
    kind: 'choice',
    id: 'ch10/10.10/q-stride',
    prompt: (
      <>
        In Fig. 10.20 each thread adds <code>input[i + c*BLOCK_DIM]</code> for c = 1 .. 2·COARSE_FACTOR − 1. Why step by{' '}
        <code>BLOCK_DIM</code> instead of giving each thread 2·COARSE_FACTOR consecutive elements?
      </>
    ),
    options: [
      { label: 'So that each thread adds its elements in the same order as the sequential loop would.', why: 'Both layouts add each thread\'s elements in increasing order, so that cannot decide between them; and the order of the sum is not the point.' },
      {
        label: 'So that in each step consecutive threads read consecutive elements and a warp\'s loads coalesce.',
        correct: true,
      },
      { label: 'To avoid shared-memory bank conflicts between the threads of a warp.', why: 'These loads go to global memory; shared memory is used only after the loop.' },
      { label: 'Because the loop has no barrier, and strided indices keep threads from racing.', why: 'Each thread reads its own elements either way; nothing is written in the loop.' },
    ],
    explain: (
      <p>
        This is the interleaved partitioning of chapter 9's coarsened histogram. In step c, thread t reads element{' '}
        <code>segment + t + c·BLOCK_DIM</code>, so the 32 threads of a warp read 32 neighbouring elements: one coalesced
        access, as section 6.1 defined it. With consecutive chunks, a warp's reads in one step would be
        2·COARSE_FACTOR elements apart. A contiguous split is the natural choice on a CPU, where one core streams through its
        own chunk; on a GPU it scatters every warp instruction.
      </p>
    ),
    source: 'ch. 10 §10.10, p. 247 (Fig. 10.20); ch. 9 §9.5; ch. 6 §6.1',
  },
};

export function S1010() {
  return (
    <>
      <p>
        Section 10.9 gives every pair of elements a thread: N elements, N/2 threads, N/2048 blocks of 1024. A large input
        launches far more blocks than the hardware runs at once, and the surplus waits. The hardware has serialised them.
      </p>
      <Source section="10.10" pages="246" />

      <h2>The price of parallelising is paid per block</h2>
      <p>
        Look at what each block spends. First every thread adds its two elements: one step, every thread busy. Then the tree:
        half the threads drop out at each step, and every step needs a barrier or a shuffle. That second phase underuses the
        hardware, and every block launched goes through it. If the blocks really run side by side, it is the cost of
        parallelism. If the hardware will run them one after another anyway, it is waste.
      </p>
      <Source section="10.10" pages="246" />
      <p>
        So when the work will be serialised regardless, serialise it yourself, where it is cheap: give each thread more
        elements to add on its own before the tree starts. That is thread coarsening, from chapter 6.
      </p>
      <Source section="10.10" pages="246" figs="ch. 6 §6.5" />

      <h2>The coarsened kernel</h2>
      <p>
        Two changes to the multi-block kernel. The segment grows by the factor: a block covers{' '}
        <code>COARSE_FACTOR·2·blockDim.x</code> elements. And before any reduction, each thread adds{' '}
        <code>2·COARSE_FACTOR</code> elements in a loop, stepping by <code>BLOCK_DIM</code>. In that loop every thread is busy,
        no thread waits for another, there is no barrier, and in each step the block reads one contiguous run. Only then does
        the usual tree start, over one partial sum per thread.
      </p>
      <Source section="10.10" pages="246–247" figs="Figs. 10.19, 10.20" />
      <CoarsenListing />
      <p>
        The page prints this kernel only in part: the header has no return type, and everything after the loop is an
        ellipsis that stands for Fig. 10.18's reduction and commit. Lab 10.1's version is the whole kernel, compiled and
        tested. Where it differs from the page: it returns <code>void</code>; it takes <code>N</code> and checks every load
        against it, so an index past the end adds 0 (exercise 5); the factor is a run-time argument, <code>coarse</code>,
        and <code>blockDim.x</code> stands in for <code>BLOCK_DIM</code>, so one build can sweep both; and the ellipsis is
        written out as the two-stage warp reduction and the atomic commit of Fig. 10.18. A lane mask typed per vendor, and
        the legacy <code>atomicAdd</code> in place of <code>cuda::atomic_ref</code> under HIP, mean the same file is written
        to compile for HIP; that is unverified, since nothing here has been built with ROCm.
      </p>
      <Source section="10.10" pages="247" figs="Fig. 10.20" />
      <Ask q={questionsS1010['ch10/10.10/q-stride']} />

      <h2>Counting steps</h2>
      <p>
        The book counts the gain in steps. Take 32 elements and 8-thread blocks. Two ordinary blocks, run one after the other,
        take 4 steps each: 1 with every thread busy, 3 in the tree. That is 8 steps, 6 of them underused. One block coarsened
        by two does the same work in 6: 3 busy steps, each thread adding its 4 elements, then the same 3-step tree.
      </p>
      <Source section="10.10" pages="247–248" figs="Fig. 10.21" />
      <p>
        The explorer generalises that count. Per block, coarsening by C gives 2C − 1 busy steps and leaves the tree at log2 of
        the block size. Blocks run in waves of R, the number the device holds at once, and a wave costs one block's steps
        however few blocks it holds. Start with the two Fig. 10.21 presets.
      </p>
      <Simplification>
        <p>
          The book's accounting: every step costs the same, blocks run in whole waves, nothing waits for memory, and launches
          and atomics are free. It shows where coarsening helps and where it stops helping, not how fast a kernel runs.
        </p>
      </Simplification>
      <CoarseningExplorer />
      <Ask q={questionsS1010['ch10/10.10/q-fig1021']} label="Work it out" />

      <h2>How far to coarsen</h2>
      <p>
        Coarsening trades parallelism for efficiency, and the trade runs out. Each step up in the factor launches fewer
        blocks. Once there are fewer blocks than the device can run at once, the idle slots stay idle for the whole kernel,
        and every further step up only lengthens the one wave that remains. The best factor leaves just enough blocks to fill
        the hardware, so it depends on both the input size and the device.
      </p>
      <Source section="10.10" pages="248" />
      <p>
        In the explorer, the book's rule is marked on the chart as the largest factor that still launches at least R blocks.
        On 2<sup>20</sup> floats, with the 264 slots of the book's hypothetical device in chapter 6, it picks C = 7, which launches 293 blocks: one full wave and a second
        wave of 29. C = 8 launches 256 and finishes in one wave, with 8 slots idle. Whole waves make the rule approximate,
        and they make some factors worse than both neighbours.
      </p>
      <Beyond>
        <p>
          Reading the rule as "the largest C with at least R blocks" is ours, as is the comparison with the fewest-steps
          factor. The book states the rule in words only.
        </p>
      </Beyond>
      <Ask q={questionsS1010['ch10/10.10/q-just-enough']} />
      <p>
        Libraries expose the same choice. When you read a production reduction, look for the number of elements each thread
        handles before the tree: that is this section's factor under another name.
      </p>
      <Today checked="2026-11-20 (ch10-facts A13)">
        <p>
          In CUB the knob is <code>items_per_thread</code>; its docs' example policy, not its default, sets 16 or 20.
        </p>
      </Today>

      <h2>One block per row</h2>
      <p>
        Now the case you will meet in practice. A softmax over a vocabulary row needs a max and a sum over every logit of the
        row. Give each row one block of {SOFTMAX_BLOCK} threads, and a {fmtInt(SOFTMAX_ROW)}-logit row (Qwen3-8B's vocabulary)
        leaves each thread about {Math.round(SOFTMAX_ROW / SOFTMAX_BLOCK)} logits to add on its own. That is coarsening at its
        limit, and it needs no commit across blocks at all. It also means one block per row: at batch 1, one block for the
        whole GPU. Try the softmax presets.
      </p>
      <Beyond>
        <p>
          The softmax framing and the vocabulary size are ours (notes/today/ch09-facts.md, D7). The presets keep the book's
          example from chapter 6, a hypothetical device with {BOOK_CAPACITY} slots; on your device the number differs.
        </p>
      </Beyond>
      <p>
        This is why llama.cpp's CUDA softmax has a second path. When a row is too long for one block's shared memory, the device
        supports cooperative launch, and a few other conditions hold (no mask, no scaling), it spreads the row over many blocks: each
        writes its partial max to an array, all meet at a grid-wide barrier, then every block reduces the partials, and the
        same again for the sum. That is section 10.9's second pass with the kernel boundary replaced by a barrier. And it is
        why lab 10.3 measures its one-block-per-row softmax at batch 1, 8 and 64.
      </p>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md A8, D1)">
        <p>
          The grid barrier needs a cooperative launch whose whole grid is resident at once; the HIP build includes a different
          header for it. Metal has no grid barrier.
        </p>
      </Today>
      <Ask q={questionsS1010['ch10/10.10/q-batch1']} />

      <h2>Exercises 4 and 5, and your device</h2>
      <p>
        Exercise 5 asks for the coarsened kernel at any length; that is lab 10.1's kernel, shown above. Exercise 4 turns it
        into a max: inside the block only the operator changes, and the commit is the subject of the table in section 10.9.
        Lab 10.3's row max is the max you will actually use. Both labs are on the <a href={href('ch10', 'labs')}>labs page</a>.
      </p>
      <Source section="10.12" pages="250" figs="exercises 4, 5" />
      <Predict
        id="ch10/10.10/p-coarse"
        prompt={
          <>
            Lab 10.1 times its coarsened kernel on one large array, 2<sup>26</sup> floats or more (the driver doubles n on a
            fast device, up to 2<sup>28</sup>, and starts lower only where memory is short), with 256-thread blocks, for the
            factors 1, 2, 4, and so on up to 128. Which factor will be fastest on <em>your</em> device?
          </>
        }
        unit="(coarsening factor)"
      >
        {(v) => <MeasuredCoarsening predicted={v} />}
      </Predict>
    </>
  );
}
