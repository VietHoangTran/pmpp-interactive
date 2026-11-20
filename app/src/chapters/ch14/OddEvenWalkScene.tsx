import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { FIG_14_1, LARGEST_FIRST, bookBound, nLogN, runToSorted, step, walk, type StepResult } from '../../sim/oddeven';

// "Why odd-even sort needs N steps": Fig. 14.1's first two steps, the worst case
// [8, 1, ..., 7], the growing bill of comparisons, the shared flag, and the stopping trap.
// Every number drawn comes from src/sim/oddeven.ts. Six beats, one narration clip each
// (content/ch14/narration/oddeven-walk-N.txt).
const FIG = walk(FIG_14_1, 2); // even step, then odd step
const WORST = runToSorted(LARGEST_FIRST, 'ours'); // 7 steps to sorted, 9 launched
const WORST7 = WORST.rows.slice(0, WORST.sortedAfter!);
const REV = step([9, 8, 7, 6, 5, 4, 3, 2], 0); // every thread swaps
const TRAP4 = runToSorted([1, 3, 2, 4], 'ours');
const BIG = 1024;

const W = 720;
const CW = 58; // cell pitch
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const ease = (t: number) => t * t * (3 - 2 * t);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const fmt = (v: number) => v.toLocaleString('en-US');

/** A row of distinct values, moving from `before` to `after` as t goes 0 to 1 (swapped cells pass over and under). */
function Row({ y, before, after, t, hot, dim, name }: {
  y: number; before: number[]; after: number[]; t: number; hot?: (v: number) => boolean; dim?: (pos: number) => boolean; name: string;
}) {
  const n = before.length;
  const x0 = (W - n * CW) / 2;
  const e = ease(clamp(t));
  const shown = t >= 1 ? after : before;
  return (
    <g data-row={name} data-values={shown.join(' ')}>
      {Array.from({ length: n }, (_, k) => (
        <rect key={'slot' + k} x={x0 + k * CW} y={y} width={CW - 6} height={CW - 6} rx="3" fill="var(--cell)" stroke="var(--rule)"
          opacity={dim?.(k) ? 0.4 : 1} />
      ))}
      {before.map((v, k) => {
        const to = after.indexOf(v);
        const x = lerp(x0 + k * CW, x0 + to * CW, e);
        const lift = to === k ? 0 : (to > k ? -1 : 1) * Math.sin(Math.PI * e) * 16;
        const isHot = hot?.(v);
        return (
          <g key={v} transform={`translate(${x}, ${y + lift})`} opacity={dim?.(k) && to === k ? 0.45 : 1}>
            <rect width={CW - 6} height={CW - 6} rx="3" fill={isHot ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" />
            <text x={(CW - 6) / 2} y={(CW - 6) / 2 + 6} fontSize="18" fontWeight="600" textAnchor="middle"
              style={{ fill: isHot ? 'var(--page)' : 'var(--ink)' }}>{v}</text>
          </g>
        );
      })}
    </g>
  );
}

/** Brackets under the pairs a step compares, with the thread that owns each. */
function Brackets({ y, n, s, show, mark }: { y: number; n: number; s: StepResult; show: number; mark?: boolean }) {
  const x0 = (W - n * CW) / 2;
  return (
    <g opacity={show} data-threads={s.threads.filter((t) => t.active).length}>
      {s.threads.map((t) => {
        const left = x0 + t.i * CW + (CW - 6) / 2;
        const right = left + CW;
        if (!t.active)
          return (
            <text key={t.t} x={x0 + Math.min(t.i, n - 1) * CW + (CW - 6) / 2} y={y + 44} fontSize="11.5" textAnchor="middle"
              style={{ fill: 'var(--ink-3)' }}>t{t.t}: i = {t.i}, out of range</text>
          );
        return (
          <g key={t.t} data-pair={`${t.i}-${t.i + 1}`} data-swapped={mark && t.swapped ? 'yes' : 'no'}>
            <path d={`M${left} ${y} v8 H${right} v-8`} fill="none" stroke="var(--ink)" strokeWidth={mark && t.swapped ? 2.6 : 1.4} />
            <text x={(left + right) / 2} y={y + 24} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
              t{t.t}, i = {t.i}{mark && t.swapped ? ': swap' : ''}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function Frame({ beat, headline, children }: { beat: number; headline: string; children: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} 300`} role="img" aria-label="Animation of odd-even sort" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <g data-beat={beat}>{children}</g>
      <text x={W / 2} y={286} fontSize="15" fontWeight="600" textAnchor="middle" data-headline="">{headline}</text>
    </svg>
  );
}

function stepBeat(beat: number, s: StepResult, p: number, label: string) {
  const t = ramp(p, 0.35, 0.75);
  const n = s.before.length;
  const ends = s.parity === 1 ? (k: number) => k === 0 || k === n - 1 : undefined;
  return (
    <Frame beat={beat} headline={p < 0.75 ? label : `${label}: ${s.comparisons} threads compared, ${s.swaps} swapped`}>
      <text x={40} y={40} fontSize="13" style={{ fill: 'var(--ink-2)' }} data-step={s.parity ? 'odd' : 'even'}>
        {s.parity ? 'order odd pairs: (1,2), (3,4), (5,6)' : 'order even pairs: (0,1), (2,3), (4,5), (6,7)'}
      </text>
      <Row y={70} before={s.before} after={s.after} t={t} dim={ends} name="data" />
      <Brackets y={136} n={n} s={s} show={ramp(p, 0.05, 0.25)} mark={p >= 0.75} />
      {s.parity === 1 && p > 0.2 ? (
        <text x={W / 2} y={222} fontSize="13" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
          the two end values sit out; the next even step lets values cross these borders
        </text>
      ) : null}
    </Frame>
  );
}

function worst(p: number) {
  const k = WORST7.length;
  const pos = clamp((p - 0.06) / 0.86) * k; // steps done, fractional
  const done = Math.min(k, Math.floor(pos));
  const s = WORST7[Math.min(k - 1, done)];
  const t = done >= k ? 1 : pos - done;
  const where = (done >= k ? s.after : t > 0.5 ? s.after : s.before).indexOf(8);
  return (
    <Frame beat={3} headline={done >= k ? `sorted after ${k} steps: N = ${LARGEST_FIRST.length} values, about N steps` : `step ${done + 1}: the 8 moves to position ${s.after.indexOf(8)}`}>
      <text x={40} y={40} fontSize="13" style={{ fill: 'var(--ink-2)' }}>worst case: the largest value starts at the front</text>
      <Row y={70} before={s.before} after={s.after} t={t} hot={(v) => v === 8} name="data" />
      <text x={W / 2} y={170} fontSize="14" textAnchor="middle" data-eight={where}>
        the 8 is at position {where}: one place per step, {k - where} to go
      </text>
      <g>
        {WORST7.map((r, j) => (
          <rect key={j} x={W / 2 - (k * 30) / 2 + j * 30} y={196} width={24} height={14} rx="2"
            fill={j < done || (j === done && t > 0.5) ? 'var(--global)' : 'var(--cell)'} stroke="var(--global)" data-done={j < done ? 'yes' : 'no'} />
        ))}
        <text x={W / 2} y={232} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>steps</text>
      </g>
    </Frame>
  );
}

function bill(p: number) {
  const rows = WORST.rows;
  const shownSteps = Math.round(ramp(p, 0.05, 0.4) * rows.length);
  const total = rows.slice(0, shownSteps).reduce((s, r) => s + r.comparisons, 0);
  const g = ramp(p, 0.5, 0.85);
  const big = bookBound(BIG);
  const small = nLogN(BIG);
  const maxW = 520;
  return (
    <Frame beat={4} headline={p < 0.5 ? `${shownSteps} steps on 8 values: ${total} comparisons` : `N = ${fmt(BIG)}: about ${fmt(big)} comparisons against ${fmt(small)}`}>
      <text x={40} y={40} fontSize="13" style={{ fill: 'var(--ink-2)' }}>every step compares about N/2 pairs, swap or not</text>
      <g data-counter={total}>
        <text x={100 + rows.length * 56 + 10} y={100} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>comparisons per step,</text>
        <text x={100 + rows.length * 56 + 10} y={115} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>8 values, worst case</text>
        {rows.map((r, j) => (
          <g key={j} opacity={j < shownSteps ? 1 : 0.15}>
            <rect x={100 + j * 56} y={110 - r.comparisons * 12} width={40} height={r.comparisons * 12} fill="var(--global-wash)" stroke="var(--global)" />
            <text x={120 + j * 56} y={128} fontSize="11.5" textAnchor="middle">{r.comparisons}</text>
          </g>
        ))}
      </g>
      <g opacity={ramp(p, 0.45, 0.55)} data-big={big} data-small={small}>
        <text x={100} y={168} fontSize="12.5">N steps × N/2 comparisons, N = {fmt(BIG)}</text>
        <rect x={100} y={174} width={maxW * g} height={16} fill="var(--global)" />
        <text x={100 + maxW * g + 6} y={187} fontSize="12.5">{fmt(Math.round(big * g))}</text>
        <text x={100} y={218} fontSize="12.5">N log2 N</text>
        <rect x={100} y={224} width={Math.max(1.5, ((maxW * small) / big) * g)} height={16} fill="var(--ink-2)" />
        <text x={100 + ((maxW * small) / big) * g + 8} y={237} fontSize="12.5">{fmt(Math.round(small * g))}</text>
      </g>
    </Frame>
  );
}

function flag(p: number) {
  const n = REV.before.length;
  const x0 = (W - n * CW) / 2;
  const fx = W / 2 - 60;
  const fy = 196;
  const atomic = p >= 0.6;
  const writers = REV.threads.filter((t) => t.swapped);
  const show = ramp(p, 0.15, 0.35);
  return (
    <Frame beat={5} headline={atomic ? 'atomicExch(hasChanged, 1u): the same store, made atomic' : `${writers.length} threads write 1 to one word at once: a data race`}>
      <text x={40} y={40} fontSize="13" style={{ fill: 'var(--ink-2)' }}>an even step on 9 8 7 6 5 4 3 2: every pair swaps</text>
      <Row y={60} before={REV.before} after={REV.after} t={ramp(p, 0.02, 0.15)} name="data" />
      <Brackets y={124} n={n} s={REV} show={1} mark />
      <g opacity={show} data-writers={writers.length} data-atomic={atomic ? 'yes' : 'no'}>
        {writers.map((t) => {
          const sx = x0 + t.i * CW + CW / 2 + (CW - 6) / 2;
          return <line key={t.t} x1={sx} y1={154} x2={fx + 60} y2={fy - 2} stroke={atomic ? 'var(--ok)' : 'var(--fail)'} strokeWidth="1.6" />;
        })}
        <rect x={fx} y={fy} width={120} height={34} rx="3" fill="var(--global-wash)" stroke="var(--global)" />
        <text x={fx + 60} y={fy + 22} fontSize="13" textAnchor="middle">hasChanged = 1</text>
        <text x={fx + 140} y={fy + 22} fontSize="13" style={{ fill: atomic ? 'var(--ok)' : 'var(--fail)' }} data-verdict={atomic ? 'ok' : 'fail'}>
          {atomic ? 'atomic: defined' : 'plain stores: undefined behaviour'}
        </text>
      </g>
    </Frame>
  );
}

function trap(p: number) {
  const rows = TRAP4.rows;
  const k = rows.length;
  const pos = clamp((p - 0.04) / 0.8) * k;
  const done = Math.min(k, Math.floor(pos));
  const s = rows[Math.min(k - 1, done)];
  const t = done >= k ? 1 : ramp(pos - done, 0.3, 0.8);
  const flags = rows.slice(0, done).map((r) => (r.flag ? 1 : 0));
  const firstQuiet = done >= 1 && !rows[0].flag;
  const stopped = done >= k;
  return (
    <Frame beat={6} headline={stopped ? `stop after ${k} steps: two steps in a row, both quiet` : done === 1 ? 'step 1 is quiet, yet 3 > 2: one quiet step proves nothing' : `step ${done + 1}: ${s.parity ? 'odd' : 'even'} pairs`}>
      <text x={40} y={40} fontSize="13" style={{ fill: 'var(--ink-2)' }}>the trap: 1 3 2 4, starting with an even step</text>
      <Row y={70} before={s.before} after={s.after} t={t} name="data" />
      <Brackets y={136} n={4} s={s} show={stopped ? 0.3 : 1} mark={t >= 1} />
      <text x={W / 2} y={208} fontSize="14" textAnchor="middle" data-flags={flags.join(' ')}>
        flag after each step: {flags.length ? flags.join('  ') : '(none yet)'}
      </text>
      {firstQuiet && !stopped && done === 1 ? (
        <text x={W / 2} y={236} fontSize="13.5" textAnchor="middle" style={{ fill: 'var(--fail)' }} data-mark="fail">
          the book read literally would stop here, unsorted
        </text>
      ) : null}
      {stopped ? (
        <text x={W / 2} y={236} fontSize="13.5" textAnchor="middle" style={{ fill: 'var(--ok)' }} data-mark="ok">
          sorted: {TRAP4.out.join(' ')}
        </text>
      ) : null}
    </Frame>
  );
}

const beats: Beat[] = [
  { clip: 'oddeven-walk-1', label: 'Even step', draw: (p) => stepBeat(1, FIG[0], p, 'even step: one thread per disjoint pair') },
  { clip: 'oddeven-walk-2', label: 'Odd step', draw: (p) => stepBeat(2, FIG[1], p, 'odd step: the pairs shift by one') },
  { clip: 'oddeven-walk-3', label: 'Worst case', draw: worst },
  { clip: 'oddeven-walk-4', label: 'The bill', draw: bill },
  { clip: 'oddeven-walk-5', label: 'The flag', draw: flag },
  { clip: 'oddeven-walk-6', label: 'When to stop', draw: trap },
];

export function OddEvenWalkScene() {
  return <Scene chapter="ch14" title="Why odd-even sort needs N steps" beats={beats} />;
}
