import { useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import {
  freeLds,
  freePitches,
  sharedBytes,
  stripRead,
  stripWorst,
  wStore,
  wStoreWorst,
  wrongOffsetReads,
  type Accessed,
  type Banks,
  type Worst,
} from '../../sim/gemmbanks';
import { THREADS, type Layout, type Width } from '../../sim/gemmremap';

// ---------------------------------------------------------------- the bank explorer

type Mode = 'A' | 'B' | 'W-scalar' | 'W-float4';
const MODE_LABEL: Record<Mode, string> = {
  A: 'A strip',
  B: 'B strip',
  'W-scalar': 'W tile, scalar loads',
  'W-float4': 'W tile, float4 loads',
};
const MODE_TITLE: Record<Mode, string> = {
  A: 'one register of the A strip, read from A_s',
  B: 'one register of the B strip, read from B_s',
  'W-scalar': "the project's W tile stored transposed into B_s, loaded with scalar loads",
  'W-float4': "the project's W tile stored transposed into B_s, loaded with float4 loads",
};

interface St {
  mode: Mode;
  layout: Layout;
  ld: number;
  pitch: number;
  banks: Banks;
  W: Width;
  warp: number;
  i: number;
  p: number;
  bank: number | null;
  showFree: boolean;
}
const START: St = { mode: 'A', layout: 'remap', ld: 8, pitch: 128, banks: 32, W: 32, warp: 0, i: 0, p: 0, bank: null, showFree: false };

/** "0–3, 8–11": a sorted list of lanes as ranges. */
function ranges(xs: number[]): string {
  const out: string[] = [];
  for (let k = 0; k < xs.length; k++) {
    let j = k;
    while (j + 1 < xs.length && xs[j + 1] === xs[j] + 1) j++;
    out.push(j > k ? `${xs[k]}–${xs[j]}` : String(xs[k]));
    k = j;
  }
  return out.join(', ');
}

const regLabel = (s: St, p: number) => {
  if (s.mode === 'W-scalar') return `sub-tile s = ${p}`;
  if (s.mode === 'W-float4') return `value j = ${p}`;
  const name = s.mode === 'A' ? 'a_r' : 'b_r';
  if (s.layout === 'coarse') return `${name}[${p}]: ${s.mode === 'A' ? 'row tRow' : 'column tCol'} + ${p}`;
  return `${name}[${p}]: quadrant ${s.mode === 'A' ? 'row' : 'column'} ${p >> 2}, ${s.mode === 'A' ? 'row' : 'column'} ${p % 4}`;
};

const nWay = (d: number) => (d <= 1 ? 'no conflict' : `${d}-way conflict`);

function compute(s: St): { acc: Accessed; worst: Worst } {
  if (s.mode === 'W-scalar' || s.mode === 'W-float4') {
    const mapping = s.mode === 'W-float4' ? 'float4' : 'scalar';
    return {
      acc: wStore({ mapping, pitch: s.pitch, banks: s.banks, W: s.W, warp: s.warp, step: s.p }),
      worst: wStoreWorst({ mapping, pitch: s.pitch, banks: s.banks, W: s.W }),
    };
  }
  const strip = { layout: s.layout, strip: s.mode, ld: s.ld, banks: s.banks, W: s.W } as const;
  return { acc: stripRead({ ...strip, warp: s.warp, i: s.i, p: s.p }), worst: stripWorst(strip) };
}

function Histogram({ acc, banks, selected, onPick }: { acc: Accessed; banks: number; selected: number; onPick: (b: number) => void }) {
  const L = 22;
  const cw = banks > 32 ? 8.5 : 14;
  const W = L + banks * cw + 6;
  const top = Math.max(1, ...acc.perBank.map((b) => b.words.length));
  const most = Math.max(4, top); // a fixed minimum height keeps the panel from jumping
  const unit = Math.min(16, 132 / most);
  const base = 6 + most * unit;
  const every = banks > 32 ? 8 : 2;
  return (
    <>
      <p className="ui muted" style={{ margin: '0 0 0.3rem', fontSize: '0.78rem' }}>
        Distinct words per bank, one box each. Click a bank to see its lanes.
      </p>
      <svg viewBox={`0 0 ${W} ${base + 30}`} role="img" aria-label="Distinct words per bank" style={{ width: '100%', maxWidth: W * 1.35, display: 'block' }}
        data-degree={acc.degree} data-banks={banks}>
        <text x={L - 5} y={base + 4} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>0</text>
        <text x={L - 5} y={base - top * unit + 9} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{top}</text>
        {acc.perBank.map((b) => {
          const x = L + b.bank * cw;
          const k = b.words.length;
          return (
            <g key={b.bank} data-bank={b.bank} data-words={k} data-lanes={b.lanes.length} onClick={() => onPick(b.bank)} style={{ cursor: 'pointer' }}>
              <rect x={x} y={0} width={cw} height={base + 16} fill="transparent" />
              {b.words.map((w, j) => (
                <rect key={w} x={x + 1} y={base - (j + 1) * unit + 1} width={Math.max(1, cw - 2)} height={unit - 2} rx="1.5"
                  fill={k > 1 ? 'var(--fail)' : 'var(--ok)'} />
              ))}
              <line x1={x + 1} x2={x + cw - 1} y1={base + 1} y2={base + 1} stroke={k ? 'var(--ink-2)' : 'var(--rule)'} />
              {b.bank === selected ? (
                <rect x={x} y={base - top * unit - 2} width={cw} height={top * unit + 6} fill="none" stroke="var(--ink)" strokeWidth="1.5" />
              ) : null}
              {b.bank % every === 0 ? (
                <text x={x + cw / 2} y={base + 15} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{b.bank}</text>
              ) : null}
            </g>
          );
        })}
        <text x={L} y={base + 28} fontSize="10" style={{ fill: 'var(--ink-3)' }}>bank</text>
      </svg>
    </>
  );
}

function BankExplorer() {
  const [s, setS] = useState<St>(START);
  const set = (o: Partial<St>) => setS((x) => ({ ...x, bank: null, ...o }));
  const isW = s.mode === 'W-scalar' || s.mode === 'W-float4';
  const steps = isW ? 4 : 8;
  const warps = THREADS / s.W;
  const st: St = { ...s, warp: Math.min(s.warp, warps - 1), p: Math.min(s.p, steps - 1) };
  const { acc, worst } = compute(st);
  const busiest = acc.perBank.reduce((a, b) => (b.words.length > a.words.length ? b : a), acc.perBank[0]);
  const sel = acc.perBank[st.bank ?? busiest.bank];
  const distinct = new Set(acc.words.map((w) => w.word)).size;
  const free = st.showFree
    ? isW
      ? freePitches(st.mode === 'W-float4' ? 'float4' : 'scalar', st.banks, st.W)
      : st.mode === 'A'
        ? freeLds(st.layout, st.banks, st.W)
        : null
    : null;
  const arr = st.mode === 'A' ? 'A_s' : 'B_s';
  return (
    <Bench
      title="Banks for one strip read"
      task="Find every leading dimension that removes the A-strip conflict, then check B in both layouts, then the project's W tile."
      foot={
        <>
          bank = 32-bit word index mod the number of banks. Red: a bank that must deliver more than one distinct word, one
          after another. Lanes reading the same word share it (a broadcast). 64 banks uses the same rule on AMD's bank count:
          a model, not AMD's documented cost.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <span className="seg" role="group" aria-label="Access">
          {(Object.keys(MODE_LABEL) as Mode[]).map((m) => (
            <button key={m} aria-pressed={m === s.mode} onClick={() => set({ mode: m })} title={MODE_TITLE[m]}>{MODE_LABEL[m]}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        {!isW ? (
          <label className="field">
            Thread layout
            <select value={s.layout} onChange={(e) => set({ layout: e.target.value as Layout })}>
              <option value="coarse">§15.3/§15.4: one 8 × 8 per thread</option>
              <option value="remap">§15.5: four 4 × 4 per thread</option>
            </select>
          </label>
        ) : null}
        {s.mode === 'A' ? (
          <label className="field">
            A_s leading dimension = {s.ld}
            <input type="range" min={8} max={17} value={s.ld} aria-label="A_s leading dimension" onChange={(e) => set({ ld: Number(e.target.value) })} />
          </label>
        ) : null}
        {s.mode === 'B' ? <span className="muted">B_s rows stay 128 floats (p. 363).</span> : null}
        {isW ? (
          <label className="field">
            B_s pitch = {s.pitch} floats
            <input type="range" min={128} max={148} value={s.pitch} aria-label="B_s pitch" onChange={(e) => set({ pitch: Number(e.target.value) })} />
          </label>
        ) : null}
        <label className="field">
          Banks
          <select value={s.banks} onChange={(e) => set({ banks: Number(e.target.value) as Banks })}>
            <option value={32}>32 (NVIDIA; AMD CDNA 1–3)</option>
            <option value={64}>64 (AMD CDNA 4, RDNA 2–4): a model</option>
          </select>
        </label>
        <label className="field">
          Warp width
          <select value={s.W} onChange={(e) => set({ W: Number(e.target.value) as Width })}>
            <option value={32}>32 lanes</option>
            <option value={64}>64 lanes (AMD CDNA)</option>
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          Warp
          <select value={st.warp} onChange={(e) => set({ warp: Number(e.target.value) })}>
            {Array.from({ length: warps }, (_, k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        {!isW ? (
          <label className="field">
            Step i
            <select value={st.i} onChange={(e) => set({ i: Number(e.target.value) })}>
              {Array.from({ length: 8 }, (_, k) => <option key={k} value={k}>{k}</option>)}
            </select>
          </label>
        ) : null}
        <label className="field">
          {isW ? 'Store' : 'Register'}
          <select value={st.p} onChange={(e) => set({ p: Number(e.target.value) })}>
            {Array.from({ length: steps }, (_, k) => <option key={k} value={k}>{regLabel(st, k)}</option>)}
          </select>
        </label>
        <button className="quiet" onClick={() => set({ ...START })}>Fig. 15.11(a)</button>
        <button className="quiet" onClick={() => set({ ...START, ld: 9 })}>Fig. 15.11(b)</button>
      </div>

      <Histogram acc={acc} banks={st.banks} selected={sel.bank} onPick={(b) => setS((x) => ({ ...x, bank: b }))} />

      <p className="ui" style={{ margin: '0.5rem 0 0.3rem' }} data-this={acc.degree}>
        This instruction: <b className={acc.degree > 1 ? 't-fail' : 't-ok'}>{nWay(acc.degree)}</b>. {st.W} lanes read{' '}
        {distinct} distinct words; the busiest bank, {busiest.bank}, must deliver {busiest.words.length}.
      </p>
      <p className="ui" style={{ margin: '0 0 0.3rem' }} data-worst={worst.degree}>
        Worst over every access of this kind (all {warps} warps{isW ? ', all 4 stores' : ', i = 0 to 7, all 8 registers'}):{' '}
        <b className={worst.degree > 1 ? 't-fail' : 't-ok'}>{nWay(worst.degree)}</b>.
      </p>
      <p className="ui muted" style={{ margin: '0 0 0.3rem' }} data-selected={sel.bank}>
        Bank {sel.bank}:{' '}
        {sel.words.length === 0
          ? 'no lane.'
          : sel.words.map((w, k) => {
              const ls = acc.words.filter((x) => x.word === w);
              const at = isW ? `B_s[${ls[0].row}][${ls[0].col}]` : `${arr}[${ls[0].row}][${ls[0].col}]`;
              return (
                <span key={w}>
                  {k ? '; ' : ''}word {w} = {at}, lane{ls.length > 1 ? 's' : ''} {ranges(ls.map((x) => x.lane))}
                </span>
              );
            })}
        .
      </p>
      {free ? (
        <p className="ui" style={{ margin: 0 }} data-free={free.join(',')}>
          {isW ? 'Conflict-free pitches from 128 to 148' : 'Conflict-free leading dimensions from 8 to 17'} in this setting:{' '}
          <b>{free.length ? free.join(', ') : 'none'}</b>.
        </p>
      ) : s.mode !== 'B' ? (
        <button className="quiet" onClick={() => setS((x) => ({ ...x, showFree: true }))}>
          List every conflict-free {isW ? 'pitch' : 'leading dimension'} for this setting
        </button>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the erratum hunt

// Lines 18 to 36 of labs/ch15/snippets/s156_padded_offset.cu (front-end-checked);
// S156.test.ts fails if this copy drifts from the file.
const PADDED_FIRST_LINE = 18;
const PADDED = `__global__ void mm_tiled_padded(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
    unsigned int bRow = blockIdx.y*bM;
    unsigned int bCol = blockIdx.x*bN;
    unsigned int tRow = (threadIdx.x/(bN/tN))*tM;
    unsigned int tCol = (threadIdx.x%(bN/tN))*tN;
    float C_r[tM][tN] = {};
    for (unsigned int tile = 0; tile < (K + bK - 1)/bK; ++tile) {
        __shared__ float A_s[bM*(bK + 1)];
        __shared__ float B_s[bK*bN];
        loadTile(&A[bRow*K + tile*bK], K, M - bRow, K - tile*bK, &A_s[0], bK + 1, bM, bK);
        loadTile(&B[tile*bK*N + bCol], N, K - tile*bK, N - bCol, &B_s[0], bN, bK, bN);
        __syncthreads();
        mm(tM, tN, bK, &A_s[tRow*bK], bK + 1, &B_s[tCol], bN, C_r);
        __syncthreads();
    }
    unsigned int maxRow = (bRow + tRow < M) ? (M - (bRow + tRow)) : 0;
    unsigned int maxCol = (bCol + tCol < N) ? (N - (bCol + tCol)) : 0;
    writeTile(&C[(bRow + tRow)*N + bCol + tCol], N, maxRow, maxCol, C_r, tM, tN);
}`;
/** File line numbers the hunt refers to. */
const LINE_DECL = 25;
const LINE_LOAD_A = 27;
const LINE_MM = 30;
const TIERS: Record<string, Tier> = {
  A: 'global', B: 'global', C: 'global', A_s: 'shared', B_s: 'shared', C_r: 'reg',
  bRow: 'reg', bCol: 'reg', tRow: 'reg', tCol: 'reg', tile: 'reg', maxRow: 'reg', maxCol: 'reg',
};

function MisReads() {
  const [ty, setTy] = useState(1);
  const reads = wrongOffsetReads(ty);
  const tRow = ty * 8;
  const lo = Math.min(...reads.map((m) => m.row), tRow);
  const hi = Math.max(...reads.map((m) => m.row), tRow + 7);
  const C = 22;
  const X0 = 64;
  const right = reads.filter((m) => m.right).length;
  const pad = reads.filter((m) => m.padding).length;
  const read = new Map(reads.map((m) => [m.row * 9 + m.col, m]));
  return (
    <Bench
      title="What a thread row reads with the old offset"
      task="Pick a thread row. Compare the cells it should read with the cells it does read."
      foot={
        <>
          The padded <code>A_s</code>, 9 floats per row; column 8 is the padding, which <code>loadTile</code> never writes.
          Outlined: the rows thread row ty owns (tRow to tRow + 7, columns 0 to 7). Filled: the 64 words it reads when{' '}
          <code>mm</code> gets <code>&amp;A_s[tRow*bK]</code> with leading dimension bK + 1.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Thread row ty (tRow = 8·ty)
          <select value={ty} onChange={(e) => setTy(Number(e.target.value))}>
            {Array.from({ length: 16 }, (_, k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
      </div>
      <p className="narrate" data-misread={`${reads[0].row},${reads[0].col}`} data-right={right} data-padding={pad} style={{ marginBottom: '0.7rem' }}>
        {ty === 0 ? (
          <>Thread row 0 has tRow = 0, so tRow·bK and tRow·(bK + 1) are both 0: these threads read the right words. The bug hides from any test that only checks the first 8 rows of each block.</>
        ) : (
          <>
            It should start at word tRow·9 = {tRow * 9}, row {tRow}, column 0. It starts at word tRow·8 = {reads[0].word}:
            row {reads[0].row}, column {reads[0].col}, {tRow * 9 - reads[0].word} words early. Right: {right} of 64 reads;
            from the padding column: {pad}.
          </>
        )}
      </p>
      <svg viewBox={`0 0 ${X0 + 9 * C + 120} ${(hi - lo + 1) * C + 26}`} role="img" aria-label="Padded A_s rows and the words read"
        style={{ width: '100%', maxWidth: 420, display: 'block' }}>
        {Array.from({ length: 9 }, (_, c) => (
          <text key={c} x={X0 + c * C + C / 2} y={12} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{c === 8 ? 'pad' : c}</text>
        ))}
        {Array.from({ length: hi - lo + 1 }, (_, k) => lo + k).map((r) => (
          <g key={r}>
            <text x={X0 - 6} y={18 + (r - lo) * C + C / 2 + 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>row {r}</text>
            {Array.from({ length: 9 }, (_, c) => {
              const m = read.get(r * 9 + c);
              const own = r >= tRow && r < tRow + 8 && c < 8;
              return (
                <rect key={c} x={X0 + c * C + 1} y={18 + (r - lo) * C + 1} width={C - 2} height={C - 2} rx="2"
                  data-cell={`${r},${c}`} data-read={m ? (m.right ? 'right' : 'wrong') : ''}
                  fill={m ? (m.right ? 'var(--ok)' : 'var(--fail)') : c === 8 ? 'var(--bench-grid)' : 'var(--cell)'}
                  stroke={own ? 'var(--ok)' : 'var(--rule)'} strokeWidth={own ? 2 : 1} />
              );
            })}
          </g>
        ))}
        <text x={X0 + 9 * C + 10} y={30} fontSize="10" style={{ fill: 'var(--ok)' }}>outlined: should read</text>
        <text x={X0 + 9 * C + 10} y={46} fontSize="10" style={{ fill: 'var(--fail)' }}>red: wrong word read</text>
        <text x={X0 + 9 * C + 10} y={62} fontSize="10" style={{ fill: 'var(--ok)' }}>green: right word</text>
      </svg>
    </Bench>
  );
}

function ErratumHunt() {
  const { progress } = useProgress();
  const [done, setDone] = useState(false);
  const answered = done || !!progress.answers['ch15/15.6/q-offset'];
  return (
    <>
      <Code
        tabs={[
          {
            label: 'Fig. 15.3 with the change of pp. 362–363',
            code: PADDED,
            status: 'front-end-checked; one line deliberately left as printed',
            tiers: TIERS,
          },
        ]}
        firstLine={PADDED_FIRST_LINE}
        highlight={answered ? [LINE_MM] : [LINE_DECL, LINE_LOAD_A, LINE_MM]}
        caption={
          <>
            From <code>labs/ch15/snippets/s156_padded_offset.cu</code>. Highlighted
            {answered ? ': the line that still has to change.' : <>: the three lines that mention A_s's size or leading dimension.</>}{' '}
            Do not copy this loop. Lab 15.4's solution gives the padded leading dimension a name, <code>LDA_S</code>, and
            uses it in every index into <code>A_s</code>.
          </>
        }
      />
      <Ask q={questionsS156['ch15/15.6/q-offset']} label="Erratum hunt: what else must change?" onResult={() => setDone(true)} />
      {answered ? <MisReads /> : null}
    </>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS156: Record<string, Question> = {
  'ch15/15.6/q-why-odd': {
    kind: 'choice',
    id: 'ch15/15.6/q-why-odd',
    prompt: <>In the §15.5 layout, why does any odd leading dimension remove the A-strip conflict on 32 banks?</>,
    options: [
      { label: 'An odd number has no common factor with 32, so consecutive rows fall in different banks.', why: 'True of consecutive rows, but the lane rows read rows 4 apart, not 1 apart. The argument has to be about 4·ld.' },
      { label: 'Lane rows read rows 4 apart; for odd ld, 4g·ld mod 32 takes eight distinct values for g = 0 to 7. An even ld pairs them up.', correct: true },
      { label: 'Odd leading dimensions put the 32 lanes\' words in 32 different banks, one per lane.', why: 'The 32 lanes read only 8 distinct words (four lanes per word): it is enough that those 8 are in different banks.' },
    ],
    explain: (
      <p>
        Lane row g reads row 4g + (fixed offset), so its word is 4g·ld + (fixed). The banks differ between lane rows only
        through 4g·ld mod 32. For odd ld, 4·ld is 4 times an odd number: as g runs over 0 to 7, 4g·ld mod 32 hits eight
        different multiples of 4. For even ld, 4·ld is a multiple of 8, and 8 multiples of 8 fit in only 4 banks: two
        lane rows share each, at least 2-way. The explorer's "list every conflict-free leading dimension" confirms it.
      </p>
    ),
    source: 'ch. 15 §15.6, pp. 361–362; the general rule is ours (notes [inferred], ch15-facts E1)',
  },
  'ch15/15.6/q-64banks': {
    kind: 'choice',
    id: 'ch15/15.6/q-64banks',
    prompt: <>On the 64-bank model, which leading dimension leaves a conflict in the §15.5 A-strip read: 10 or 12?</>,
    options: [
      { label: '10 is still 2-way, as on 32 banks; 12 is free.', why: 'That is closer to the 32-bank picture, where 10 is 2-way and 12 is 4-way.' },
      { label: '12 is 2-way; 10 is free.', correct: true },
      { label: 'Both are free: 64 banks are enough for any even leading dimension.', why: 'Try 12 in the explorer with 64 banks.' },
    ],
    explain: (
      <p>
        With 64 banks the lane rows' words differ by 4g·ld mod 64. For ld = 10 that is 40g mod 64: 0, 40, 16, 56, 32, 8, 48,
        24, eight different banks. For ld = 12 it is 48g mod 64: 0, 48, 32, 16, then the same four again: 2-way. On 32
        banks both conflict: 10 is 2-way and 12 is 4-way. The rule on
        64 banks is "ld not divisible by 4". AMD does not document how a 64-lane access is checked, so this is a model.
      </p>
    ),
    source: 'ours (notes/today/ch15-facts.md E1; bank counts and the caveat from B2)',
  },
  'ch15/15.6/q-early-pad': {
    kind: 'choice',
    id: 'ch15/15.6/q-early-pad',
    prompt: (
      <>
        You pad <code>A_s</code> in the register-tiled kernel of §15.4, before doing the §15.5 remap. What do you expect on
        an NVIDIA GPU?
      </>
    ),
    options: [
      { label: 'The same gain as after the remap: the 8-way conflict on the A strip disappears.', why: 'The 8-way conflict belongs to the remapped layout. Before the remap the A-strip read was 2-way.' },
      { label: 'Little: the A-strip read there was only 2-way, and the 4-way B-strip read is untouched.', correct: true },
      { label: 'A slowdown: padding A_s misaligns the global loads that fill it, so they no longer coalesce.', why: 'The padding is in shared memory; global loads are unchanged.' },
    ],
    explain: (
      <p>
        In the §15.3/§15.4 layout a warp is two rows of 16 threads with 8 × 8 tiles. Its A-strip read touches 2 rows 8
        apart: 2-way at ld 8, removed by padding. Its B-strip read touches 16 distinct words 8 floats apart, which land in 4
        banks: 4-way, and padding A does nothing for it. "B needs no change" holds only after the remap. That is why the labs
        pad after lab 15.3, not before.
      </p>
    ),
    source: 'notes/today/ch15-facts.md E1 (our arithmetic with NVIDIA’s rule); book p. 363',
  },
  'ch15/15.6/q-offset': {
    kind: 'choice',
    id: 'ch15/15.6/q-offset',
    prompt: (
      <>
        The listing above makes exactly the book's two edits: <code>A_s[bM*(bK + 1)]</code>, and bK + 1 as A_s's leading
        dimension in <code>loadTile</code> and <code>mm</code>. What else must change for the kernel to be correct?
      </>
    ),
    options: [
      { label: 'Line 27: the last argument of the A loadTile call, bK, must become bK + 1.', why: 'That argument is the tile’s width, still 8 floats. Only its leading dimension changed, and that argument already says bK + 1.' },
      { label: 'Line 30: the start offset &A_s[tRow*bK] must become &A_s[tRow*(bK + 1)].', correct: true },
      { label: 'Line 26: B_s needs a padding column too.', why: 'B_s is untouched by the edit, so nothing about it became wrong. Whether B conflicts is a speed question, and in this layout padding a row would not fix it anyway.' },
      { label: 'Nothing: the two edits are complete.', why: 'Then every thread row except the first reads the wrong rows of A_s; see below.' },
    ],
    explain: (
      <p>
        <code>mm</code> receives a pointer to the first element of the thread's rows, and the start of row tRow depends on
        the leading dimension: tRow·9 with the padding, not tRow·8. Left as printed, thread row ty starts 8·ty words early.
        Row 0 is unaffected, which is why a small test can pass. The book changes the declaration and the leading-dimension
        arguments and does not mention this line.
      </p>
    ),
    source: 'ch. 15 §15.6, pp. 362–363; notes/ch15.md, "Unclear in the PDF" item 4',
  },
  'ch15/15.6/q-bytes': {
    kind: 'numeric',
    id: 'ch15/15.6/q-bytes',
    prompt: (
      <>
        How many bytes of shared memory do the padded <code>A_s</code> (128 rows, leading dimension 9) and the unpadded{' '}
        <code>B_s</code> (8 × 128) take together, with 4-byte floats?
      </>
    ),
    answer: sharedBytes(9).total,
    unit: 'bytes',
    explain: (
      <p>
        A_s: 128 × 9 × 4 = {sharedBytes(9).A} bytes, up from {sharedBytes(8).A}. B_s: 8 × 128 × 4 = {sharedBytes(9).B}.
        Together {sharedBytes(9).total}: tiny next to what a block may use, which is why §15.7 finds shared memory is not
        this kernel's limit.
      </p>
    ),
    source: 'ch. 15 §15.6 and §15.7, p. 363; the sum is ours (notes [inferred], ch15-facts E1)',
  },
  'ch15/15.6/q-w-pitch': {
    kind: 'choice',
    id: 'ch15/15.6/q-w-pitch',
    prompt: (
      <>
        In the project, the W tile is stored transposed into <code>B_s</code>: with scalar loads one warp's lanes hold 4
        rows × 8 columns of W and store them down 8 rows of <code>B_s</code>. Which B_s pitch makes that store
        conflict-free on 32 banks: 128, 129 or 132 floats?
      </>
    ),
    options: [
      { label: '128, with no padding.', why: 'k·128 mod 32 = 0 for every k: the 8 rows start in the same bank, so each of 4 banks holds 8 words: 8-way.' },
      { label: '129, one padding column, as for A_s.', why: 'k·129 + n mod 32 = k + n: the 32 words fall in banks 0 to 10, up to 4 per bank.' },
      { label: '132.', correct: true },
    ],
    explain: (
      <p>
        Lane (n, k) writes word k·pitch + n, n = 0 to 3, k = 0 to 7. With pitch 132, k·132 mod 32 = 4k, so the banks are 4k
        + n: all 32 different. One padding column is the right fix when lanes go down a column one row at a time; here 4
        lanes sit side by side in each row, so each row has to move by 4 banks. 132 floats is 128 plus 16 bytes, the padding
        MLX gives its threadgroup tiles. The explorer's W-tile mode shows the float4 mapping too.
      </p>
    ),
    source: 'ours (the project layout of lab 6.4; MLX’s 16-byte padding from ch15-facts C2 and ch06-facts D3)',
  },
};

// ---------------------------------------------------------------- the page

export function S156() {
  const first = freeLds('remap', 32, 32)[0];
  return (
    <>
      <p>
        Section 15.5 fixed the stores. The reads from shared memory have a problem too, and it is one the remap made worse.
        A thread that owns a large output tile reads shared memory with a stride, and strided reads can conflict in the
        banks (chapter 6).
      </p>
      <Source section="15.6" pages="361" />
      <p>
        Take the §15.5 layout and the A strip. In each step i, a thread loads one value from each of the rows its
        sub-tiles cover. The four lanes of a lane row own the same rows, so they read the same word: a broadcast, not a
        conflict. The eight lane rows read rows 4 apart. With <code>A_s</code>'s rows 8 floats long, row 4g starts at word
        32g, and 32g mod 32 is 0: every lane row lands in bank 0. Eight distinct words in one bank is an 8-way conflict, and
        it happens on every strip access.
      </p>
      <Source section="15.6" pages="361–362" figs="Fig. 15.11(a)" />

      <h2>Banks for one strip read</h2>
      <Predict
        id="ch15/15.6/p-first-free"
        prompt={<>What is the smallest leading dimension above 8 for A_s that leaves no A-strip conflict on 32 banks, in the §15.5 layout?</>}
      >
        {(v) => (
          <p>
            It is <b>{first}</b>
            {v === first ? ', as you said' : ''}: one padding column, the book's fix. Check it in the explorer, then find the
            others: the list button gives them all once you have a guess.
          </p>
        )}
      </Predict>
      <p>
        The explorer computes the banks of one strip-read instruction of one warp. It starts on the book's case,
        Fig. 15.11(a). Move the leading dimension, then switch the layout and the access.
      </p>
      <Simplification>
        <p>
          NVIDIA's rule (ch15-facts A2): bank = 32-bit word index mod 32; distinct words in one bank are served one after
          another; lanes on the same word share it. 64 banks is AMD's count, used with the same rule: AMD does not document
          how a 64-lane access is checked (B2). Each strip register is one 32-bit load. No timing.
        </p>
      </Simplification>
      <BankExplorer />

      <h2>One padding column</h2>
      <p>
        Give <code>A_s</code> one extra column that nothing uses. The tile is still 8 floats wide, but its leading
        dimension is 9. Row 4g now starts at word 36g, and 36g mod 32 is 4g: the eight lane rows fall in banks 0, 4, 8, ...,
        28. The book's "second elements", in banks 1, 5, ..., 29, are column 1 of the same rows, the next step i.
      </p>
      <Source section="15.6" pages="362" figs="Fig. 15.11(b)" />
      <p>
        The book invites you to check every access. One line does it. Lane row g, register (qr, r) and step i read the word
        (4g + 32·qr + r)·9 + i, which mod 32 is 4g + 9r + i: the quadrant term drops out, because 32·9 is a multiple of 32.
        For fixed r and i that is eight different banks as g runs from 0 to 7. With 8 in place of 9 the same expression is
        8r + i, the same bank for every g.
      </p>
      <Beyond title="Our arithmetic">
        <p>The general check is ours (notes, §15.6 [inferred]); which "second elements" the book means is our reading (notes, Unclear 6).</p>
      </Beyond>
      <Ask q={questionsS156['ch15/15.6/q-why-odd']} />
      <p>
        So the rule, on 32 banks, is any odd leading dimension. On a 64-bank model it is any leading dimension not divisible
        by 4. Set the explorer to 64 banks and try 10 and 12.
      </p>
      <Beyond>
        <p>Both rules are ours (ch15-facts E1), computed with NVIDIA's rule on each bank count.</p>
      </Beyond>
      <Ask q={questionsS156['ch15/15.6/q-64banks']} />

      <h2>"B needs no change", after the remap</h2>
      <p>
        The book says <code>B_s</code> needs nothing: one warp's B-strip read touches consecutive elements of a row, which
        sit in different banks. In the §15.5 layout that holds, and the picture is even better than stated: the lanes of a
        lane column read the same word, so the warp reads 4 distinct words, each broadcast to 8 lanes.
      </p>
      <Source section="15.6" pages="363" />
      <p>
        In the §15.3/§15.4 layout it does not hold. There the warp is two rows of 16 threads with 8 × 8 tiles. Its B-strip
        read touches 16 distinct words 8 floats apart, which land in 4 banks: a 4-way conflict. Its A-strip read was only
        2-way. Padding A in that kernel removes a 2-way conflict and leaves a 4-way one, which is why the labs pad after the
        remap, not before. Switch the explorer's layout to see both.
      </p>
      <Beyond title="Our arithmetic">
        <p>From ch15-facts E1, with NVIDIA's 32-bank rule: "B needs no change" (p. 363) holds for the §15.5 layout only.</p>
      </Beyond>
      <Ask q={questionsS156['ch15/15.6/q-early-pad']} />

      <h2>The code change, and the line the book leaves out</h2>
      <p>
        The book's change is two edits: declare <code>A_s</code> with bM·(bK + 1) floats, and pass bK + 1 instead of bK as{' '}
        <code>A_s</code>'s leading dimension to <code>loadTile</code> and <code>mm</code>. Below is Fig. 15.3's kernel with
        exactly those edits. One more line must change before it computes the right answer. Find it.
      </p>
      <Source section="15.6" pages="362–363" figs="Fig. 15.3" />
      <ErratumHunt />

      <h2>What the padding costs</h2>
      <p>
        <code>A_s</code> grows from 1,024 to 1,152 floats: {sharedBytes(9).A.toLocaleString('en-US')} bytes, and{' '}
        {sharedBytes(9).total.toLocaleString('en-US')} with <code>B_s</code>. That is small; §15.7 explains why shared
        memory is not what limits this kernel. The other cost is subtler. A padded row is 36 bytes long, so most rows do not
        start on a 16-byte boundary and cannot take a float4 store. Lab 15.4 loads A with float4s from global memory and
        writes the four values into <code>A_s</code> as four scalars.
      </p>
      <Beyond title="Our arithmetic">
        <p>The byte counts are ours (notes, §15.6 [inferred]); the scalar stores into A_s follow the notes' Exercise 1 sketch.</p>
      </Beyond>
      <Ask q={questionsS156['ch15/15.6/q-bytes']} label="Work it out" />

      <h2>Banks on other GPUs</h2>
      <p>
        NVIDIA documents 32 banks of 4 bytes. AMD documents 32 banks on CDNA 1 to 3 and 64 on CDNA 4 and RDNA 2 to 4, also
        4 bytes, and an LDS that serves 16 lanes per cycle; it does not say how the lanes of a 64-wide wavefront are checked
        against each other, so the explorer's 64-bank counts are a model, not AMD's cost. Apple documents no banks for
        threadgroup memory at all. MLX pads its threadgroup tiles by 16 bytes in practice, so on Metal padding is something
        you measure, not something you can predict.
      </p>
      <Today checked="2026-11-20 (notes/today/ch15-facts.md A2, B2, C2; ch06-facts.md D3)">
        <p>
          NVIDIA: "Any 32-bit memory read or write request made of 32 addresses that fall in 32 distinct memory banks can
          therefore be serviced simultaneously". AMD: "A single warp can achieve up to 64 bytes per cycle throughput (16
          lanes per cycle)." Metal: no "bank" in the MSL 4.1 specification; MLX: <code>tgp_padding_a = 16 / sizeof(T)</code>.
        </p>
      </Today>
      <p>
        The model has one more limit: it treats each strip register as a 32-bit load. NVIDIA documents no bank rule for
        64- or 128-bit shared-memory accesses, and a compiler may merge neighbouring loads into wider ones. The labs read
        strips as 32-bit values so the model applies, but only a profiler shows what the hardware did:{' '}
        <code>l1tex__data_bank_conflicts_pipe_lsu.sum</code> in Nsight Compute, the LDS "Bank Conflict Rate" in the ROCm
        Compute Profiler, and nothing on Metal.
      </p>
      <Today checked="2026-11-20 (notes/today/ch15-facts.md A2; ch06-facts.md A12, B4)">
        <p>
          Searched and not found: a rule in NVIDIA's current Programming Guide for how 64-bit or 128-bit shared-memory
          accesses map to banks. A lab that reads strips as float4 must measure its conflicts.
        </p>
      </Today>

      <h2>In the project: the W tile</h2>
      <p>
        The project's linear layer computes Y = X·Wᵀ with W stored (out, in), the way <code>nn.Linear</code> and ggml store
        it. So the B tile is a transposed tile of W. Lab 15.6 loads a 128 × 8 tile of W along its rows, which coalesces,
        and stores each value into <code>B_s[k][n]</code>. With scalar loads one warp holds 4 rows × 8 columns of W, and
        its store runs down 8 rows of <code>B_s</code>. With a pitch of 128 floats those 8 rows start in the same bank.
        Use the explorer's W-tile modes to find a pitch that is free, for scalar and for float4 loads.
      </p>
      <Beyond>
        <p>
          Ours: the project layout is lab 6.4's (<a href={href('ch06', 'labs')}>chapter 6 labs</a>), after PyTorch's and
          ggml's (ch03-facts D2, D6); the arithmetic uses NVIDIA's rule. MLX's 16-byte padding is from ch06-facts D3.
        </p>
      </Beyond>
      <Ask q={questionsS156['ch15/15.6/q-w-pitch']} />
      <p>
        Lab 15.6's reference solution uses that pitch, <code>LDB_S = bN + 4</code>, and keeps the A tile at bK + 1. Lab
        15.4 adds the padding and float4 loads to your remapped kernel; lab 15.6 builds the linear layer on top. Both
        are on the <a href={href('ch15', 'labs')}>labs page</a>. Next, §15.7 asks what the large tiles cost in registers:{' '}
        <a href={href('ch15', '15.7')}>occupancy considerations</a>.
      </p>
    </>
  );
}
