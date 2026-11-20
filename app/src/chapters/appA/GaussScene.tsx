import { Scene, type Beat } from '../../components/Scene';
import { system, trace, type Phase, type ThreadState, type Trace } from '../../sim/gauss';

// "One thread per row": the book's parallel elimination as a timeline of thread states,
// every frame taken from src/sim/gauss.ts (trace) with exact fractions and IEEE specials.
// Scripts: content/appA/narration/gauss-threads-N.txt.

// ---------------------------------------------------------------- shared drawing (also used by the stepper in SA7)
export const STATE_LABEL: Record<ThreadState, string> = {
  ready: 'ready',
  working: 'working',
  skipping: 'skips: no such variable',
  idle: 'idle',
  waiting: 'at the barrier',
  searching: 'pivot search',
  finished: 'finished',
};

const BAR = 6;
const CELL = 14;
export const timelineWidth = (phases: Phase[]) => phases.reduce((w, p) => w + (p.kind === 'barrier' ? BAR : CELL), 0);

/** Thread lanes over time: one cell per phase, a dark bar per barrier. */
export function Timeline({ phases, n, x, y, lane = 34 }: { phases: Phase[]; n: number; x: number; y: number; lane?: number }) {
  let cx = x;
  const cols = phases.map((p, i) => {
    const at = cx;
    cx += p.kind === 'barrier' ? BAR : CELL;
    if (p.kind === 'barrier')
      return <rect key={i} x={at + 1.5} y={y - 4} width={BAR - 3} height={n * lane + 2} rx="1" fill="var(--ink)" data-kind="barrier" />;
    return (
      <g key={i} data-kind={p.kind}>
        {p.threads.map((s, t) => (
          <rect key={t} x={at + 1} y={y + t * lane} width={CELL - 3} height={lane - 10} rx="2" data-state={s}
            fill={s === 'working' ? 'var(--ink)' : s === 'finished' ? 'var(--ok-wash)' : s === 'searching' ? 'var(--shared-wash)' : s === 'idle' ? 'var(--rule-soft)' : 'none'}
            stroke={s === 'finished' ? 'var(--ok)' : s === 'searching' ? 'var(--shared)' : s === 'skipping' ? 'var(--ink)' : 'var(--rule)'}
            strokeDasharray={s === 'skipping' ? '3 2' : undefined} />
        ))}
      </g>
    );
  });
  return <g>{cols}</g>;
}

const isBad = (v: number) => !Number.isFinite(v);
const MW = 58; // matrix cell pitch
const cellText = (t: string) => (t === '0' ? '·' : t.replace(/-/g, '−'));

/** The augmented matrix after a phase, one row per thread. */
function Matrix({ phase, x, y, vars, lane = 34 }: { phase: Phase; x: number; y: number; vars: string[]; lane?: number }) {
  const n = phase.m.length;
  const w = MW;
  return (
    <g>
      {[...vars, '='].map((v, c) => (
        <text key={c} x={x + c * w + w / 2} y={y - 8} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{v}</text>
      ))}
      {phase.m.map((row, t) =>
        row.map((cell, c) => {
          const bad = isBad(cell.v);
          return (
            <g key={`${t}-${c}`}>
              <rect x={x + c * w} y={y + t * lane} width={w - 4} height={lane - 10} rx="2"
                fill={bad ? 'var(--fail-wash)' : 'var(--cell)'} stroke={bad ? 'var(--fail)' : c === n ? 'var(--ink-3)' : 'var(--rule)'} />
              <text x={x + c * w + (w - 4) / 2} y={y + t * lane + (lane - 10) / 2 + 4.5} fontSize="12.5" textAnchor="middle"
                style={{ fill: bad ? 'var(--fail)' : 'var(--ink)' }}>{cellText(cell.text)}</text>
            </g>
          );
        }),
      )}
    </g>
  );
}

function opText(p: Phase, t: number): string {
  const s = p.threads[t];
  if (p.kind === 'divide' && p.divisors?.[t]) return '÷ ' + cellText(p.divisors[t]!.text).replace('·', '0');
  if (p.kind === 'subtract' && s === 'working') return `− row ${p.col + 1}`;
  if (s === 'skipping') return 'skips';
  if (p.kind === 'pivot' && s === 'searching') {
    const j = p.pivot!.rows.indexOf(t);
    return `|${cellText(p.pivot!.abs[j].text).replace('·', '0')}|${p.pivot!.winner === t ? ' max' : ''}`;
  }
  if (p.kind === 'swap' && s === 'working') return 'swap';
  if (p.kind === 'backsub' && s === 'working') return 'substitute';
  return '';
}

export function phaseTitle(p: Phase, vars: string[]): string {
  switch (p.kind) {
    case 'start':
      return 'one thread per row, all rows in one block\'s shared memory';
    case 'pivot':
      return `pivot search for ${vars[p.col]}: max |coefficient| is in row ${p.pivot!.winner + 1}`;
    case 'swap':
      return `swap: row ${p.swap![1] + 1} moves to the top`;
    case 'divide':
      return `divide: every remaining row by its own coefficient of ${vars[p.col]}`;
    case 'subtract':
      return `subtract row ${p.col + 1}: ${vars[p.col]} leaves the rows below`;
    case 'barrier':
      return 'barrier: every thread waits';
    case 'backsub':
      return `back substitution: row ${p.col + 1} gives ${vars[p.col]} = ${cellText(p.m[p.col][p.m.length].text)}`;
    case 'done':
      return 'solution: ' + vars.map((v, i) => `${v} = ${cellText(p.m[i][p.m.length].text)}`).join(', ');
  }
}

// ---------------------------------------------------------------- the scene
const T1 = trace(system('book1'), { pivot: false, arith: 'exact' });
const T2 = trace(system('book2'), { pivot: false, arith: 'exact' });
const T3 = trace(system('book2'), { pivot: true, arith: 'exact' });
const VARS = ['X', 'Y', 'Z'];
const W = 720;
const H = 320;
const LANE = 40;
const TY = 74; // top of the lanes and the matrix rows
const TX = 96; // timeline start
const MX = 430; // matrix start
const at = (from: number, to: number, p: number) => Math.min(to, from + Math.floor(p * (to - from + 1)));

function Stage({ t, idx, beat, intro = 1 }: { t: Trace; idx: number; beat: number; intro?: number }) {
  const ph = t.phases[idx];
  const n = t.n;
  const flat = ph.m.flat();
  const nan = flat.filter((c) => Number.isNaN(c.v)).length;
  const inf = flat.filter((c) => Math.abs(c.v) === Infinity).length;
  const nanRows = ph.m.filter((r) => r.some((c) => Number.isNaN(c.v))).length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Threads eliminating one row each" style={{ width: '100%', maxWidth: W, display: 'block' }}
      data-beat={beat} data-system={t.sys.id} data-pivot={t.opts.pivot ? 1 : 0} data-index={idx} data-phase={ph.kind} data-nan={nan} data-inf={inf}
      data-rhs={ph.m.map((r) => r[n].text).join(',')}>
      <text x={20} y={24} fontSize="12" style={{ fill: 'var(--ink-2)' }}>
        {t.sys.id === 'book1' ? "the book's first system" : "the book's second system"}{t.opts.pivot ? ', with pivoting' : ', no pivoting'}
      </text>
      <text x={TX} y={TY - 12} fontSize="11" style={{ fill: 'var(--ink-3)' }}>time →</text>
      {Array.from({ length: n }, (_, k) => (
        <g key={k} opacity={intro}>
          <text x={20} y={TY + k * LANE + 18} fontSize="12.5" fontWeight="600">thread {k}</text>
          <text x={MX - 8} y={TY + k * LANE + 19} fontSize="11.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{opText(ph, k)}</text>
        </g>
      ))}
      <Timeline phases={t.phases.slice(0, idx + 1)} n={n} x={TX} y={TY} lane={LANE} />
      <Matrix phase={ph} x={MX} y={TY} vars={VARS} lane={LANE} />
      {ph.kind === 'barrier' ? (
        <g>
          <rect x={MX - 4} y={TY + n * LANE - 2} width={4 * MW} height={22} rx="3" fill="var(--ink)" />
          <text x={MX + 2 * MW - 4} y={TY + n * LANE + 13} fontSize="12" textAnchor="middle" style={{ fill: 'var(--page)' }}>barrier: all {n} threads wait</text>
        </g>
      ) : null}
      <g transform={`translate(20 ${H - 70})`}>
        {(
          [
            ['working', 'var(--ink)', 'var(--rule)'],
            ['skips', 'none', 'var(--ink)'],
            ['idle', 'var(--rule-soft)', 'var(--rule)'],
            ['pivot search', 'var(--shared-wash)', 'var(--shared)'],
            ['finished', 'var(--ok-wash)', 'var(--ok)'],
          ] as const
        ).map(([l, f, s], i) => (
          <g key={l} transform={`translate(${i * 112} 0)`}>
            <rect width={12} height={12} rx="2" fill={f} stroke={s} strokeDasharray={l === 'skips' ? '3 2' : undefined} />
            <text x={17} y={10} fontSize="11">{l}</text>
          </g>
        ))}
        <rect x={5 * 112} width={4} height={14} fill="var(--ink)" />
        <text x={5 * 112 + 9} y={10} fontSize="11">barrier</text>
      </g>
      <text x={20} y={H - 18} fontSize="13.5" fontWeight="600" style={{ fill: nan || inf ? 'var(--fail)' : 'var(--ink)' }}>
        {nanRows === n
          ? `NaN in all ${n} rows: the answer is lost, though the system has one`
          : nan || inf
            ? 'row 1 divided by 0: infinity, and 0 / 0 is NaN'
            : phaseTitle(ph, VARS)}
      </text>
    </svg>
  );
}

const end1 = T1.phases.length - 1;
const end3 = T3.phases.length - 1;
const beats: Beat[] = [
  { clip: 'gauss-threads-1', label: 'One row each', draw: (p) => <Stage t={T1} idx={0} beat={1} intro={Math.min(1, 0.2 + p * 2)} /> },
  { clip: 'gauss-threads-2', label: 'Divide, barrier', draw: (p) => <Stage t={T1} idx={at(1, 2, p)} beat={2} /> },
  { clip: 'gauss-threads-3', label: 'Subtract, barrier', draw: (p) => <Stage t={T1} idx={at(3, 4, p)} beat={3} /> },
  { clip: 'gauss-threads-4', label: 'The rest', draw: (p) => <Stage t={T1} idx={at(5, end1, p)} beat={4} /> },
  { clip: 'gauss-threads-5', label: 'A zero pivot', draw: (p) => <Stage t={T2} idx={at(0, 4, p)} beat={5} /> },
  { clip: 'gauss-threads-6', label: 'Pivoting', draw: (p) => <Stage t={T3} idx={at(1, end3, p)} beat={6} /> },
];

export function GaussScene() {
  return <Scene chapter="appA" title="One thread per row" beats={beats} />;
}
