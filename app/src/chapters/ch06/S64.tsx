import { useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BANK_PRESETS,
  EX5_STRIDES,
  gcdRule,
  halfCase,
  sameWordCase,
  strideCase,
  tileAccess,
  type BankPresetId,
  type BankResult,
} from '../../sim/banks';
import { TurnAndPadScene } from './TurnAndPadScene';

const LAB63 = 'ch06/lab3_banks';

/** Backends that are real GPUs (realBackends() keeps the emulator's race-detection flavour). */
const gpuBackends = (reports: Reports) =>
  realBackends(reports).filter((b) => b !== 'race' && !/emulator/i.test(String(reports[b].device?.name ?? '')));
function labRows(reports: Reports, backend: string, lab: string): BenchRow[] {
  const s = benchRows(reports[backend], lab, 'solution');
  return s.length ? s : benchRows(reports[backend], lab, 'work');
}

// ---------------------------------------------------------------- the bank explorer

type Pattern = 'stride' | 'tile' | 'same' | 'half';

interface BankState {
  preset: BankPresetId;
  pattern: Pattern;
  stride: number;
  T: number;
  pad: number;
  orientation: 'x-first' | 'y-first';
  warp: number;
  lane: number;
}
const START: BankState = { preset: 'nvidia', pattern: 'stride', stride: 1, T: 32, pad: 0, orientation: 'x-first', warp: 0, lane: 1 };

/** Rows of threads in a block T threads wide, keeping it within 1024 threads. */
const blockRows = (T: number) => Math.min(T, Math.floor(1024 / T));

function compute(s: BankState): { r: BankResult; banks: number; lanes: number; describe: (lane: number) => string } | null {
  const p = BANK_PRESETS[s.preset];
  if (p.banks == null || p.warpWidth == null) return null;
  const banks = p.banks;
  const lanes = p.warpWidth;
  if (s.pattern === 'stride') {
    const r = strideCase(s.stride, lanes, banks);
    return {
      r, banks, lanes,
      describe: (l) => `lane ${l}: a[${l}·${s.stride}] is word ${l * s.stride}, bank ${l * s.stride} mod ${banks} = ${r.lanes[l].bank}`,
    };
  }
  if (s.pattern === 'tile') {
    const first = s.warp * lanes;
    const r = tileAccess({ T: s.T, pitch: s.T + s.pad, orientation: s.orientation, ty0: Math.floor(first / s.T), tx0: first % s.T, warpWidth: lanes, banks });
    return {
      r, banks, lanes,
      describe: (l) => {
        const t = r.threads[l];
        return `lane ${l} is thread (tx ${t.tx}, ty ${t.ty}): a[${t.row}][${t.col}] is word ${t.row}·${s.T + s.pad} + ${t.col} = ${t.word}, bank ${t.word} mod ${banks} = ${r.lanes[l].bank}`;
      },
    };
  }
  if (s.pattern === 'same') {
    const r = sameWordCase(0, lanes, banks);
    return { r, banks, lanes, describe: (l) => `lane ${l}: word 0, bank 0, the same word as every other lane` };
  }
  const r = halfCase(lanes, banks);
  return {
    r, banks, lanes,
    describe: (l) => `lane ${l}: a 2-byte element at byte ${l * 2}, inside word ${Math.floor(l / 2)}, bank ${r.lanes[l].bank}`,
  };
}

function BankPicture({ r, banks, lane }: { r: BankResult; banks: number; lane: number }) {
  const W = 700;
  const cw = (W - 40) / banks;
  const rows = Math.max(1, r.passes);
  const dot = Math.max(4, Math.min(9, cw - 2, 260 / rows - 2));
  const H = 44 + rows * (dot + 2) + 20;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Banks and the lanes that land in each" style={{ width: '100%', display: 'block' }}>
      <text x={20} y={12} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>banks (one column each); one dot per distinct word, stacked in the order the bank serves them</text>
      {r.perBank.map((b) => (
        <g key={b.bank} data-bank={b.bank} data-lanes={b.lanes.length} data-words={b.words.length}>
          <rect x={20 + b.bank * cw} y={20} width={cw - 1} height={6} fill={b.lanes.length ? 'var(--shared)' : 'var(--rule)'} />
          {banks <= 32 || b.bank % 8 === 0 ? (
            <text x={20 + b.bank * cw + cw / 2} y={37} fontSize="8" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{b.bank}</text>
          ) : null}
          {b.words.map((w, i) => {
            const shared = r.lanes.filter((l) => l.word === w);
            const mine = shared.some((l) => l.lane === lane);
            return (
              <g key={w}>
                <rect x={20 + b.bank * cw + (cw - 1 - dot) / 2} y={44 + i * (dot + 2)} width={dot} height={dot} rx={dot / 2}
                  fill={b.words.length > 1 ? 'var(--fail)' : 'var(--shared)'} stroke={mine ? 'var(--ink)' : 'none'} strokeWidth={mine ? 2 : 0} />
                {shared.length > 1 && cw >= 14 ? (
                  <text x={20 + b.bank * cw + cw - 1} y={44 + i * (dot + 2) + dot - 1} fontSize="7" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>×{shared.length}</text>
                ) : null}
              </g>
            );
          })}
        </g>
      ))}
      <text x={20} y={H - 4} fontSize="10" style={{ fill: 'var(--ink-3)' }}>
        Ringed: the lane you picked. Red: a bank with more than one word to deliver. “×n”: n lanes on one word.
      </text>
    </svg>
  );
}

function BankExplorer() {
  const [s, setS] = useState<BankState>(START);
  const set = (o: Partial<BankState>) => setS((x) => ({ ...x, ...o }));
  const preset = BANK_PRESETS[s.preset];
  const c = compute(s);
  const lanes = preset.warpWidth ?? 32;
  const lane = Math.min(s.lane, lanes - 1);
  const warpsInBlock = Math.max(1, Math.ceil((s.T * blockRows(s.T)) / lanes));
  return (
    <Bench
      title="Bank explorer"
      task="Find a stride with a 32-way conflict and one with none. Then break the corner-turned tile and fix it."
      foot={
        preset.banks == null ? (
          <>Apple documents no bank structure for threadgroup memory, so there is nothing to count. {preset.facts}.</>
        ) : (
          <>
            NVIDIA's documented rule: bank = 4-byte word index mod the number of banks; passes = the most distinct words
            any bank must deliver. Same word, no conflict. {preset.facts}.
          </>
        )
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          GPU
          <select value={s.preset} onChange={(e) => set({ preset: e.target.value as BankPresetId, warp: 0 })}>
            {Object.values(BANK_PRESETS).map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
          </select>
        </label>
        <label className="field">
          Access
          <select value={s.pattern} onChange={(e) => set({ pattern: e.target.value as Pattern, warp: 0 })}>
            <option value="stride">a[threadIdx.x * stride]</option>
            <option value="tile">a 2D tile, a[..][..]</option>
            <option value="same">every lane on one word</option>
            <option value="half">2-byte elements, a[threadIdx.x]</option>
          </select>
        </label>
        {s.pattern === 'stride' ? (
          <label className="field">
            stride = {s.stride}
            <input type="range" min={1} max={64} value={s.stride} aria-label="stride" onChange={(e) => set({ stride: Number(e.target.value) })} />
          </label>
        ) : null}
        {s.pattern === 'tile' ? (
          <>
            <label className="field">
              Tile width (and blockDim.x)
              <select value={s.T} onChange={(e) => set({ T: Number(e.target.value), warp: 0 })}>
                {[16, 32, 64].map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
            <label className="field">
              Padding columns = {s.pad}
              <input type="range" min={0} max={4} value={s.pad} aria-label="padding columns" onChange={(e) => set({ pad: Number(e.target.value) })} />
            </label>
            <label className="field">
              Index order
              <select value={s.orientation} onChange={(e) => set({ orientation: e.target.value as 'x-first' | 'y-first' })}>
                <option value="x-first">a[threadIdx.x][threadIdx.y] (the turned store)</option>
                <option value="y-first">a[threadIdx.y][threadIdx.x]</option>
              </select>
            </label>
            <label className="field">
              Warp of the block
              <select value={Math.min(s.warp, warpsInBlock - 1)} onChange={(e) => set({ warp: Number(e.target.value) })}>
                {Array.from({ length: warpsInBlock }, (_, i) => <option key={i} value={i}>{i}</option>)}
              </select>
            </label>
          </>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="muted">Exercise 5 strides:</span>
        <span className="seg" role="group" aria-label="Exercise 5 strides">
          {EX5_STRIDES.map((v) => (
            <button key={v} aria-pressed={s.pattern === 'stride' && s.stride === v} onClick={() => set({ pattern: 'stride', stride: v })}>{v}</button>
          ))}
        </span>
        <span className="muted">Tiles:</span>
        <button className="quiet" onClick={() => set({ pattern: 'tile', T: 32, pad: 0, orientation: 'x-first', warp: 0 })}>32 × 32, turned store</button>
        <button className="quiet" onClick={() => set({ pattern: 'tile', T: 32, pad: 1, orientation: 'x-first', warp: 0 })}>padded to 33</button>
        <button className="quiet" onClick={() => set({ pattern: 'tile', T: 16, pad: 0, orientation: 'x-first', warp: 0 })}>16 × 16, turned store</button>
      </div>

      {c ? (
        <>
          <label className="field" style={{ marginBottom: '0.5rem' }}>
            Lane = {lane}
            <input type="range" min={0} max={lanes - 1} value={lane} aria-label="lane" onChange={(e) => set({ lane: Number(e.target.value) })} />
          </label>
          <p className="narrate" style={{ margin: '0 0 0.7rem' }} data-lane-arith>{c.describe(lane)}.</p>
          <BankPicture r={c.r} banks={c.banks} lane={lane} />
          <div className="row" style={{ marginTop: '0.5rem' }} data-readout data-passes={c.r.passes} data-banks-used={c.r.banksUsed} data-max-lanes={c.r.maxLanesPerBank}>
            <span className={'chip ' + (c.r.passes === 1 ? 'ok' : 'fail')}>{c.r.passes === 1 ? 'one pass: no conflict' : `${c.r.passes} passes: ${c.r.passes}-way conflict`}</span>
            <span><b>{c.r.banksUsed}</b> of {c.banks} banks used</span>
            <span>at most <b>{c.r.maxLanesPerBank}</b> lane{c.r.maxLanesPerBank === 1 ? '' : 's'} on one bank</span>
            {c.r.sharedWord ? <span className="muted">some lanes share a word: {preset.sameWord}</span> : null}
          </div>
          {preset.caveat ? <p className="muted" style={{ margin: '0.5rem 0 0' }} data-caveat>{preset.caveat}</p> : null}
        </>
      ) : (
        <p className="narrate" data-readout data-passes="" data-not-documented>
          <b>Not documented.</b> {preset.caveat} On many lanes reading one location, {preset.sameWord}, so NVIDIA's free
          broadcast is not something to count on here.
        </p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- Exercise 5

function Exercise5Table() {
  const [guess, setGuess] = useState<Record<number, string>>({});
  const [shown, setShown] = useState(false);
  const rows = EX5_STRIDES.map((st) => ({ st, r: strideCase(st) }));
  const right = rows.filter(({ st, r }) => Number(guess[st]) === r.passes).length;
  return (
    <Bench
      title="Exercise 5"
      task="For each stride, how many lanes share each bank the warp uses? Fill in all seven, then reveal."
      foot={<><code>__shared__ float a[1024];</code> and <code>a[threadIdx.x*stride] = ...;</code> by a block of one 32-lane warp, <code>a[0]</code> in bank 0 (book p. 155). 32 banks of 4 bytes. Your entries here are not recorded.</>}
    >
      <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>Part</th><th>stride</th><th>lanes per bank used</th>
            {shown ? <><th>model</th><th>banks hit</th></> : null}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ st, r }, i) => (
            <tr key={st} data-stride={st} data-result={shown ? (Number(guess[st]) === r.passes ? 'right' : 'wrong') : ''}>
              <td style={{ paddingLeft: '0.6rem' }}>({'abcdefg'[i]})</td>
              <td>{st}</td>
              <td>
                <input type="number" min={1} max={32} value={guess[st] ?? ''} aria-label={`stride ${st}`}
                  onChange={(e) => { setGuess({ ...guess, [st]: e.target.value }); setShown(false); }} />
              </td>
              {shown ? (
                <>
                  <td><span className={'chip ' + (Number(guess[st]) === r.passes ? 'ok' : 'fail')}>{r.passes === 1 ? '1 (no conflict)' : `${r.passes}-way`}</span></td>
                  <td>{r.banksUsed === 32 ? 'all 32' : r.perBank.filter((b) => b.lanes.length).map((b) => b.bank).join(', ')}</td>
                </>
              ) : null}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" onClick={() => setShown(true)} disabled={rows.some(({ st }) => !guess[st])}>Reveal</button>
        {shown ? <span data-score={right}>{right} of 7 right.</span> : <span className="muted">{rows.filter(({ st }) => guess[st]).length} of 7 filled in</span>}
      </div>
      {shown ? (
        <div style={{ marginTop: '0.8rem' }} data-gcd-rule>
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>The rule behind the table.</b> Lane <i>l</i> lands in bank (<i>l</i> · stride) mod 32, which repeats every
            32 / <i>g</i> lanes, where <i>g</i> is the greatest common divisor of the stride and 32. So the warp uses
            32 / <i>g</i> banks with <i>g</i> lanes on each, all on different words: a <i>g</i>-way conflict.
          </p>
          <p className="muted" style={{ margin: 0 }}>
            {EX5_STRIDES.map((st) => `gcd(${st}, 32) = ${gcdRule(st).g}`).join('; ')}. Any odd stride is conflict-free;
            every factor of 2 in the stride doubles the conflict.
          </p>
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- your device

function YourDevice64() {
  const { reports, loading } = useReports();
  const found = gpuBackends(reports)
    .map((b) => {
      const rows = labRows(reports, b, LAB63);
      return {
        b,
        device: String(reports[b].device.name ?? b),
        probe: rows.filter((r) => r.name === 'bank_probe' && num(r.stride) != null).sort((p, q) => Number(p.stride) - Number(q.stride)),
        transpose: rows.filter((r) => r.name === 'transpose'),
      };
    })
    .filter((x) => x.probe.length || x.transpose.length);
  const strides = [...new Set(found.flatMap((f) => f.probe.map((r) => Number(r.stride))))].sort((a, b) => a - b);
  const rel = (f: (typeof found)[number], st: number) => {
    const row = f.probe.find((r) => Number(r.stride) === st);
    const base = num(f.probe.find((r) => Number(r.stride) === 1)?.ns_per_access);
    return num(row?.relative_to_stride1) ?? (base && num(row?.ns_per_access) ? num(row?.ns_per_access)! / base : undefined);
  };
  return (
    <Bench
      title="Your device: banks"
      task="Does your device's time per access follow the model's pass count?"
      foot="The model column is NVIDIA's documented rule for a 32-lane warp. Measured columns are lab 6.3's time per shared-memory access relative to stride 1; they include loop overhead, so a 32-pass access need not take 32 times as long. On AMD and Apple they are measurements only: neither vendor's documents predict them."
    >
      {loading ? null : !found.length ? (
        <p data-no-measurement>
          No measurement yet. Run <code>python3 labs/pmpp.py run 6.3 --solution --bench</code> on a machine with a GPU
          (or <code>run 6.3 --bench</code> once your own kernel passes) and reload.
        </p>
      ) : (
        <>
          {strides.length ? (
            <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-probe>
              <thead>
                <tr>
                  <th style={{ paddingLeft: '0.6rem' }}>stride</th><th className="num">NVIDIA model: passes</th>
                  {found.map((f) => <th key={f.b} className="num">{f.device} ({f.b}): time ÷ stride 1</th>)}
                </tr>
              </thead>
              <tbody>
                {strides.map((st) => (
                  <tr key={st} data-stride={st}>
                    <td style={{ paddingLeft: '0.6rem' }}>{st}</td>
                    <td className="num">{strideCase(st).passes}</td>
                    {found.map((f) => {
                      const v = rel(f, st);
                      return <td key={f.b} className="num">{v != null ? `${fmt(v)}×` : '–'}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
          {found.some((f) => f.transpose.length) ? (
            <table className="data" style={{ background: 'var(--page)', margin: '0.8rem 0 0' }} data-transpose>
              <thead>
                <tr><th style={{ paddingLeft: '0.6rem' }}>Device</th><th className="num">tile</th><th className="num">pitch T: GB/s</th><th className="num">pitch T + 1: GB/s</th><th className="num">padded ÷ unpadded</th></tr>
              </thead>
              <tbody>
                {found.flatMap((f) =>
                  [...new Set(f.transpose.map((r) => Number(r.tile)))].sort((a, b) => a - b).map((T) => {
                    const a = num(f.transpose.find((r) => Number(r.tile) === T && Number(r.pitch) === T)?.gb_per_s);
                    const b = num(f.transpose.find((r) => Number(r.tile) === T && Number(r.pitch) === T + 1)?.gb_per_s);
                    return (
                      <tr key={f.b + T}>
                        <td style={{ paddingLeft: '0.6rem' }}>{f.device} ({f.b})</td>
                        <td className="num">{T}</td>
                        <td className="num">{a != null ? fmt(a) : '–'}</td>
                        <td className="num">{b != null ? fmt(b) : '–'}</td>
                        <td className="num">{a && b ? `${fmt(b / a)}×` : '–'}</td>
                      </tr>
                    );
                  }),
                )}
              </tbody>
            </table>
          ) : (
            <p className="muted" style={{ margin: '0.8rem 0 0' }}>No tiled-transpose measurement yet: it appears once lab 6.3's transpose has been benchmarked.</p>
          )}
          <p className="muted" style={{ margin: '0.6rem 0 0' }} data-measurement-only={found.filter((f) => f.b !== 'cuda').map((f) => f.b).join(',')}>
            Measurements from your machine, not expected values.
            {found.some((f) => f.b !== 'cuda')
              ? ` No vendor model predicts the ${found.filter((f) => f.b !== 'cuda').map((f) => `${f.device} (${f.b})`).join(' or ')} column${found.filter((f) => f.b !== 'cuda').length === 1 ? '' : 's'}: those are measurements only.`
              : ''}
          </p>
        </>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS64: Record<string, Question> = {
  'ch06/6.4/q-ex5-24': {
    kind: 'numeric',
    id: 'ch06/6.4/q-ex5-24',
    prompt: (
      <>
        Exercise 5(c): <code>__shared__ float a[1024];</code> and one 32-lane warp writes <code>a[threadIdx.x*24]</code>.
        With 32 banks of 4 bytes, how many lanes share each bank the warp uses?
      </>
    ),
    answer: 8,
    explain: <p>Lane l is in bank 24l mod 32: banks 0, 24, 16, 8, then round again. gcd(24, 32) = 8, so 4 banks with 8 lanes each: an 8-way conflict.</p>,
    source: 'book p. 155 (exercise 5), §6.4 pp. 139–140',
  },
  'ch06/6.4/q-ex5-12': {
    kind: 'numeric',
    id: 'ch06/6.4/q-ex5-12',
    prompt: <>Exercise 5(e): the same warp writes <code>a[threadIdx.x*12]</code>. How many lanes share each bank used?</>,
    answer: 4,
    explain: <p>gcd(12, 32) = 4: the banks repeat every 8 lanes, so 8 banks (0, 4, 8, … 28) with 4 lanes each.</p>,
    source: 'book p. 155 (exercise 5)',
  },
  'ch06/6.4/q-ex5-31': {
    kind: 'numeric',
    id: 'ch06/6.4/q-ex5-31',
    prompt: <>Exercise 5(b): <code>a[threadIdx.x*31]</code>. How many lanes share each bank used?</>,
    answer: 1,
    explain: (
      <p>
        31 and 32 share no factor, so the 32 lanes fall in 32 different banks: no conflict. The whole table: 32 → 32,
        31 → 1, 24 → 8, 16 → 16, 12 → 4, 8 → 8, 7 → 1.
      </p>
    ),
    source: 'book p. 155 (exercise 5)',
  },
  'ch06/6.4/q-16col': {
    kind: 'numeric',
    id: 'ch06/6.4/q-16col',
    prompt: (
      <>
        Had the array been 16 floats wide, a warp whose 32 lanes have <code>threadIdx.x</code> = 0 to 31 (a block 32
        threads wide) writing <code>a[threadIdx.x][threadIdx.y]</code> puts how many lanes on each bank it uses?
      </>
    ),
    answer: 16,
    explain: (
      <p>
        The index is 16·tx + ty: even tx land in bank ty, odd tx in bank 16 + ty. Two banks, 16 lanes each, as the book
        says. With a square 16 × 16 block instead, one warp is two rows of threads, and the count becomes four banks with
        8 lanes each: the explorer's 16 × 16 preset.
      </p>
    ),
    source: 'book §6.4 p. 141; the 16 × 16 block case is our arithmetic with the same rule',
  },
  'ch06/6.4/q-padcost': {
    kind: 'numeric',
    id: 'ch06/6.4/q-padcost',
    prompt: <>How many extra bytes of shared memory does padding one 32 × 32 <code>float</code> tile to 32 × 33 cost?</>,
    answer: 128,
    unit: 'bytes',
    explain: <p>One extra column of 32 floats, 4 bytes each. With two such tiles per block, as in a tiled matmul, it is 256 bytes per block.</p>,
    source: 'book §6.4 pp. 140–141',
  },
  'ch06/6.4/q-shift': {
    kind: 'choice',
    id: 'ch06/6.4/q-shift',
    prompt: (
      <>
        The book writes that multiplying <code>i</code> by 33 becomes <code>i &gt;&gt; 5 + i</code>. What does that C
        expression compute?
      </>
    ),
    options: [
      { label: <>33 × <code>i</code>, as intended: a shift by 5, then an add.</>, why: 'Two slips: >> divides by a power of two, and + binds more tightly than a shift.' },
      { label: <><code>i &gt;&gt; (5 + i)</code>, a right shift by 5 + i bits.</>, correct: true },
      { label: <><code>i / 32 + i</code>: a right shift by 5 bits, then an add of i.</>, why: 'That would need parentheses: in C, + binds more tightly than >>.' },
    ],
    explain: (
      <p>
        Multiplying by 32 is a left shift by 5, <code>i &lt;&lt; 5</code>; the book prints the shift the wrong way round,
        and the second expression also needs parentheses. The intended form is <code>(i &lt;&lt; 5) + i</code>. The book's point stands: 32 columns cost one shift, 33 cost a
        shift and an add.
      </p>
    ),
    source: 'book §6.4 pp. 140–141 (printed form); notes/ch06.md "Unclear in the PDF" item 1; facts E1',
  },
  'ch06/6.4/q-broadcast': {
    kind: 'choice',
    id: 'ch06/6.4/q-broadcast',
    prompt: <>Every lane of a warp reads the same word of shared memory. Which statement is right?</>,
    options: [
      { label: 'A 32-way conflict on every GPU: one bank serves the 32 lanes one at a time.', why: 'NVIDIA documents a broadcast for this case, and AMD says it is handled efficiently. A conflict needs different words in one bank.' },
      { label: 'Free on every GPU: the one word is read once and handed to every lane.', why: 'Apple documents no banks, and its tuning advice is to remove many-thread reads of one location.' },
      {
        label: 'Free on NVIDIA and AMD; not known to be free on Metal.',
        correct: true,
      },
    ],
    explain: (
      <p>
        A conflict is lanes on different words of one bank. Lanes on the same word are a different case, and vendors
        treat it differently: NVIDIA and AMD broadcast it, while Apple advises removing such accesses from threadgroup
        reads.
      </p>
    ),
    source: 'notes/today/ch06-facts.md A12, C2; infra-facts.md A10, B10',
  },
  'ch06/6.4/q-amd': {
    kind: 'choice',
    id: 'ch06/6.4/q-amd',
    prompt: (
      <>
        On an AMD GPU with 32 banks and 64-lane wavefronts, lanes <i>i</i> and <i>i</i> + 32 of a unit-stride{' '}
        <code>float</code> access land in the same bank. Should you pad?
      </>
    ),
    options: [
      { label: 'Yes, pad by one column, as for the corner-turned tile of section 6.4.', why: 'Padding changes nothing for a unit stride: the two lanes are still 32 words apart.' },
      {
        label: 'Not on that basis: AMD serves 16 lanes per cycle in any case, so NVIDIA\'s conflict count does not apply. Measure, with the LDS bank-conflict rate or lab 6.3.',
        correct: true,
      },
      { label: 'Yes, pad rows to 65 words to suit the 64-lane wavefront.', why: 'That is a guess at a cost nobody documented. Measure first.' },
    ],
    explain: (
      <p>
        The 32-bank figure is AMD's; the pairing of lanes i and i + 32 is our arithmetic from it. AMD documents that a
        wavefront's LDS access is served 16 lanes per cycle and does not document how it groups lanes for conflicts, so
        a count computed the NVIDIA way is not AMD's cost.
      </p>
    ),
    source: 'notes/today/infra-facts.md B10; ch06-facts.md B4',
  },
};

// ---------------------------------------------------------------- the section

export function S64() {
  const turned = tileAccess({ T: 32, pitch: 32, orientation: 'x-first' });
  const padded = tileAccess({ T: 32, pitch: 33, orientation: 'x-first' });
  return (
    <>
      <p>
        Shared memory is SRAM, not DRAM. Its latency is short, so it has no use for bursts, and it can be split into many
        narrow <b>banks</b> that work in parallel.
      </p>
      <Source section="6.4" pages="139" />
      <p>
        The book's typical layout is 32 banks, each 4 bytes wide, with consecutive 4-byte words in consecutive banks: word
        0 in bank 0, word 31 in bank 31, word 32 back in bank 0. A warp reading 32 consecutive floats is served by 32
        banks at once.
      </p>
      <Today checked="2026-11-20 (notes/today/infra-facts.md A10; ch06-facts.md A12)">
        <p>
          NVIDIA documents 32 banks of 4 bytes for every compute capability. It gives no latency in cycles, only “much
          higher bandwidth and lower latency” than global memory, so the book's “a few clock cycles” is unverified.
        </p>
      </Today>

      <h2>A conflict</h2>
      <p>
        A bank serves one word at a time. When lanes of one warp touch different words that sit in the same bank, the
        hardware splits the access into as many conflict-free passes as it needs and serves them one after another.
      </p>
      <Source section="6.4" pages="139–140" />
      <p>
        That is a <b>bank conflict</b>; k lanes on different words of one bank is a k-way conflict.
      </p>
      <Today checked="2026-11-20 (notes/today/infra-facts.md A10; ch06-facts.md A12)">
        <p>
          Two cases that look like conflicts are not, on NVIDIA GPUs: lanes reading the same word get it broadcast, and
          lanes touching different bytes of the same 32-bit word do not conflict.
        </p>
      </Today>
      <p>
        The book's exercise 5 is the cleanest way in. Fill in the table before you touch the explorer below it.
      </p>
      <Exercise5Table />
      <p>
        Now check your table in the explorer, then go past it: strides the exercise does not list, a whole tile, every
        lane on one word, and 2-byte elements.
      </p>
      <Simplification>
        <p>
          NVIDIA's documented rule, nothing more: no timing. AMD's presets reuse that arithmetic with AMD's bank counts
          but not its 16-lanes-per-cycle service. Apple publishes nothing to model.
        </p>
      </Simplification>
      <BankExplorer />
      <Ask q={questionsS64['ch06/6.4/q-ex5-24']} label="Exercise 5, for review" />
      <Ask q={questionsS64['ch06/6.4/q-ex5-12']} label="Exercise 5, for review" />
      <Ask q={questionsS64['ch06/6.4/q-ex5-31']} label="Exercise 5, for review" />

      <h2>The conflict corner turning makes</h2>
      <p>
        Section 6.1 fixed the global reads of a column-major tile by turning the corner. The tile is declared{' '}
        <code>__shared__ float a[TILE_DIM][TILE_DIM]</code> with <code>TILE_DIM</code> = 32 and written as{' '}
        <code>a[threadIdx.x][threadIdx.y] = ...</code>. A warp of a 32 × 32 block has one <code>threadIdx.y</code> and
        consecutive <code>threadIdx.x</code>, so thread 0 writes <code>a[0][0]</code>, thread 1 <code>a[1][0]</code>,
        thread 2 <code>a[2][0]</code>: a column. Element <code>a[i][j]</code> is word i·32 + j, so the lanes hit words
        0, 32, 64 and so on, and every one of them is in bank {turned.lanes[0].bank}. That is a {turned.passes}-way
        conflict, with 31 banks idle.
      </p>
      <Source section="6.4" pages="139–140" />
      <p>
        The fix is one unused column: declare <code>a[TILE_DIM][TILE_DIM + 1]</code> and keep the same access.
      </p>
      <Source pages="140" />
      <p>
        Element <code>a[i][j]</code> is now word i·33 + j, thread 0 lands in bank 0, thread 1 in bank 1, thread 2 in
        bank 2, and the store takes {padded.passes} pass.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A6, B4)">
        <p>
          NVIDIA's guides and AMD's performance guidelines both print this exact fix: a 32 × 32 array accessed down a
          column, then the same array declared 32 × 33.
        </p>
      </Today>
      <TurnAndPadScene />

      <h2>What padding costs</h2>
      <p>
        Two things. Memory: the extra column holds nothing. And index arithmetic: with 32 columns the multiplication by
        the row length is a single shift, and with 33 it becomes a shift and an add. More padding columns, more work.
      </p>
      <Source section="6.4" pages="140–141" />
      <Ask q={questionsS64['ch06/6.4/q-padcost']} label="Work it out" />
      <p>
        The book prints the shift the wrong way round. It writes <code>i &gt;&gt; 5</code> for multiplying by 32 and{' '}
        <code>i &gt;&gt; 5 + i</code> for 33. Multiplying is a left shift, <code>i &lt;&lt; 5</code> and{' '}
        <code>(i &lt;&lt; 5) + i</code>, and C reads the printed second form as something else again.
      </p>
      <Aside>
        <p>
          A different padding goes by the same name. <code>cudaMallocPitch</code> and <code>hipMallocPitch</code> pad each
          row of a 2D array in global memory so every row starts aligned for coalescing. Section 6.4's padding breaks a
          power-of-two stride in shared memory. (notes/today/ch06-facts.md A3, B2.)
        </p>
      </Aside>
      <Ask q={questionsS64['ch06/6.4/q-shift']} />

      <h2>Narrower arrays conflict less</h2>
      <p>
        How much padding buys depends on the pattern and the size. The book's example: had the array been 16 floats wide,
        the warp's lanes would be 16 words apart, falling on two banks with a 16-way conflict each. Lanes closer together
        conflict less.
      </p>
      <Source pages="141" />
      <p>
        That count assumes the warp's 32 lanes all have different <code>threadIdx.x</code>, as in a block 32 threads wide.
        With a square 16 × 16 block, a warp is two rows of 16 threads, and the same store lands on four banks with 8 lanes
        each (our arithmetic, with the same rule). Try both in the explorer: stride 16, then the 16 × 16 tile preset.
      </p>
      <Ask q={questionsS64['ch06/6.4/q-16col']} label="Work it out" />

      <h2>On other backends</h2>
      <p>
        AMD calls shared memory the local data share (LDS). It has 32 banks on CDNA 1 to 3 and 64 on CDNA 4 and RDNA 2 to
        4, each 4 bytes wide, serves a wavefront 16 lanes per cycle, and handles a broadcast from one address without a
        conflict. AMD's own guide uses the same [32][33] padding. But a count computed for NVIDIA's 32 lanes and 32 banks
        is not AMD's count: switch the explorer to an AMD preset and read the caveat under the result.
      </p>
      <Today checked="2026-11-20 (notes/today/infra-facts.md B10; ch06-facts.md B4)">
        <p>
          AMD's programming-model page says “usually 32 banks”, which contradicts its hardware page for CDNA 4 and RDNA 2
          and later. The hardware page is the specific one.
        </p>
      </Today>
      <Ask q={questionsS64['ch06/6.4/q-amd']} />
      <p>
        Apple documents no bank structure for threadgroup memory. What it does publish is advice: align threadgroup
        allocations to 16 bytes, have neighbouring threads of a quad group touch neighbouring elements, and remove
        accesses to one location by many threads of a threadgroup. That last point is the opposite of NVIDIA's free
        broadcast. On Apple family 9 GPUs, threadgroup memory is allocated out of the L1 cache. So on a Mac, lab 6.3
        measures; nothing predicts.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md C2; infra-facts.md C13)">
        <p>
          From Apple's Xcode documentation, “Reducing shader bottlenecks”, and the 2023 tech talk on profiling M3 and A17
          Pro GPUs.
        </p>
      </Today>
      <Ask q={questionsS64['ch06/6.4/q-broadcast']} />

      <h2>Seeing it in a profiler</h2>
      <p>
        On NVIDIA GPUs, Nsight Compute counts conflicts directly: <code>l1tex__data_bank_conflicts_pipe_lsu.sum</code>,
        and an average N-way figure per shared-memory instruction, where 1-way means no conflict. On AMD Instinct GPUs,
        the ROCm Compute Profiler's LDS panel reports a bank conflict rate, the share of LDS cycles spent on conflicts.
        Apple documents no bank counter; Xcode's tile-memory limiter counters are the indirect evidence.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A12, A17, B4, B9, C5)">
        <p>
          NVIDIA's triage guide flags bank conflicts above about 10 % of shared-memory wavefronts and suggests remapping
          the indexing. ROCm Compute Profiler lists only Instinct GPUs and Ryzen AI Max APUs as tested.
        </p>
      </Today>

      <h2>In the backends you will read</h2>
      <p>
        The two GEMMs we read pad by 16 bytes rather than by one float, and their threads move more than 4 bytes at a
        time. MLX's Metal GEMM pads each threadgroup tile by 16 bytes and shapes the tile by how the operand is stored.
        llama.cpp's quantised matrix multiplication pads every block by 16 bytes and asserts that the row stride, counted
        in 32-bit words, is 4 more than a multiple of 8 on its <code>mma</code> path (odd on its <code>dp4a</code> path).
        Same idea as the extra column, for wider accesses.
      </p>
      <Beyond>
        <p>
          Read from MLX's <code>steel/gemm/gemm.h</code> (<code>tgp_padding_a = 16 / sizeof(T)</code>) and llama.cpp's{' '}
          <code>mmq.cuh</code> on 2026-11-20 (notes/today/ch06-facts.md D1, D3). Why stride mod 8 = 4 suits 16-byte
          accesses is not modelled here.
        </p>
      </Beyond>

      <h2>Your device</h2>
      <p>
        Lab 6.3 has a probe that reads shared memory at each stride of exercise 5 and a tiled transpose with the row
        length as a run-time argument, so the same kernel runs unpadded and padded. See the{' '}
        <a href={href('ch06', 'labs')}>labs page</a>.
      </p>
      <YourDevice64 />
    </>
  );
}
