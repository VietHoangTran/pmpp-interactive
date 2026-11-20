import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  FALLBACK_WEIGHTS,
  GE_GT,
  SCENE_CFG,
  SCENE_U,
  SCENE_WEIGHTS,
  TOP_U,
  cdfIn,
  fallbackDraw,
  inverseCdf,
  sceneBoundary,
  sceneDraw,
} from '../../sim/draw';
import { f32short } from '../../sim/philox';

// "Finding u in the scan" (section X1.8): one block of 8 threads (display warps of 4, 2
// survivors per thread, chunks of 16) draws from 40 survivors at T = 3 with u = 0.8. Six
// beats, one narration clip each (content/x1/narration/find-u-N.txt). Every sum, flag, ballot
// and token drawn comes from src/sim/draw.ts (drawKernel, sceneBoundary, fallbackDraw),
// pinned in draw.test.ts.
const K = sceneDraw();
const N = SCENE_WEIGHTS.length;
const CHUNK = SCENE_CFG.threads * SCENE_CFG.vec;
const BND = sceneBoundary()!;
const FB = fallbackDraw();

const W = 720;
const H = 340;
const X0 = 20;
const BW = 680;
const MONO = { fontVariationSettings: "'MONO' 1" } as const;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const short = (v: number) => f32short(v);

/** Bar geometry in double precision (drawing only); the numbers printed come from the kernel trace. */
function geometry(w: number[]) {
  const total = w.reduce((s, x) => s + x, 0);
  const edges: number[] = [0];
  for (const x of w) edges.push(edges[edges.length - 1] + x);
  return { total, xs: edges.map((e) => X0 + (e / total) * BW) };
}
const G = geometry(SCENE_WEIGHTS);

function Frame({ beat, headline, children }: { beat: number; headline: string; children: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={headline} data-beat={beat} style={{ display: 'block' }}>
      <text x={14} y={22} fontSize="13" fontWeight="600" data-headline="" style={{ fill: 'var(--ink)' }}>{headline}</text>
      {children}
    </svg>
  );
}

/** The row as one stacked bar; `state(i)` picks each survivor's look. */
function Bar({ y, state, target = true, chunks = false }: { y: number; state?: (i: number) => 'win' | 'skip' | 'scan' | 'plain'; target?: boolean; chunks?: boolean }) {
  const tx = X0 + (K.target / K.total) * BW;
  return (
    <g>
      {SCENE_WEIGHTS.map((_, i) => {
        const st = state ? state(i) : 'plain';
        return (
          <rect key={i} data-survivor={i} data-state={st} x={G.xs[i]} y={y} width={Math.max(0.5, G.xs[i + 1] - G.xs[i])} height={26}
            fill={st === 'win' ? 'var(--ink)' : st === 'skip' ? 'var(--rule)' : i % 2 ? 'var(--global-wash)' : 'var(--cell)'}
            stroke={st === 'win' ? 'var(--ink)' : 'var(--global)'} strokeWidth={0.6} opacity={st === 'skip' ? 0.6 : 1} />
        );
      })}
      {[0, 1, 2, 5].map((i) => (
        <text key={i} x={(G.xs[i] + G.xs[i + 1]) / 2} y={y + 17} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{i}</text>
      ))}
      {chunks
        ? Array.from({ length: Math.ceil(N / CHUNK) }, (_, c) => {
            const a = G.xs[c * CHUNK];
            const b = G.xs[Math.min(N, (c + 1) * CHUNK)];
            return (
              <g key={c}>
                <line x1={a} x2={a} y1={y - 6} y2={y + 34} stroke="var(--ink)" strokeWidth={1.4} />
                <text x={b - a < 120 ? b : a + 3} textAnchor={b - a < 120 ? 'end' : 'start'} y={y + 46 + (c % 2) * 14} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
                  chunk {c}: #{c * CHUNK} to #{Math.min(N, (c + 1) * CHUNK) - 1}
                </text>
              </g>
            );
          })
        : null}
      {target ? (
        <g data-target={K.target}>
          <line x1={tx} x2={tx} y1={y - 14} y2={y + 32} stroke="var(--ink)" strokeWidth={2} strokeDasharray="4 2" />
          <text x={tx} y={y - 18} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink)' }}>target = u × total = {short(K.target)}</text>
        </g>
      ) : null}
    </g>
  );
}

const crossing = K.chunks.find((c) => c.crosses)!;

function LaneGrid({ y, upto, flags }: { y: number; upto: number; flags: boolean }) {
  return (
    <g data-part="lanes">
      <text x={14} y={y - 8} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
        chunk {crossing.c}: aggregate before it {short(crossing.aggBefore)}; each lane: carry + its own inclusive sums
      </text>
      {crossing.lanes.map((l) => {
        const x = 14 + l.thread * 87;
        const on = l.thread < upto;
        return (
          <g key={l.thread} opacity={on ? 1 : 0.25} data-lane={l.thread} data-warp={l.warp}>
            <text x={x + 40} y={y + 10} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
              warp {l.warp}, lane {l.lane}
            </text>
            {l.cum.map((c, j) => {
              const i = l.first + j;
              const f = flags && l.flags[j];
              return (
                <g key={j} data-index={i} data-cum={c} data-flag={flags ? (l.flags[j] ? 1 : 0) : ''}>
                  <rect x={x} y={y + 16 + j * 34} width={80} height={30} rx="3" fill={f ? 'var(--ok-wash)' : 'var(--cell)'} stroke={f ? 'var(--ok)' : 'var(--rule)'} />
                  <text x={x + 5} y={y + 28 + j * 34} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>#{i}{flags ? (l.flags[j] ? '  > target' : '  ≤ target') : ''}</text>
                  <text x={x + 5} y={y + 41 + j * 34} fontSize="11" style={{ ...MONO, fill: 'var(--ink)' }}>{short(c)}</text>
                </g>
              );
            })}
          </g>
        );
      })}
    </g>
  );
}

const ballotBits = (b: number, width: number) =>
  Array.from({ length: width }, (_, l) => ((b >>> l) & 1 ? '1' : '0')).reverse().join('');

const beats: Beat[] = [
  {
    clip: 'find-u-1',
    label: 'The target',
    draw: (p) => (
      <Frame beat={1} headline={`${N} survivors after temperature, total ${short(K.total)}; u = ${SCENE_U}`}>
        <Bar y={90} target={p > 0.35} state={(i) => (p > 0.7 && i === K.index ? 'win' : 'plain')} />
        <text x={14} y={160} fontSize="12" opacity={ramp(p, 0.35, 0.5)} style={{ fill: 'var(--ink)' }}>
          survivor i's width is its weight w_i = exp((l_i − l_0) / T); the bar's length is the total
        </text>
        <text x={14} y={186} fontSize="12" opacity={ramp(p, 0.7, 0.85)} style={{ fill: 'var(--ink)' }} data-token={K.index}>
          the token: the first survivor whose running sum is strictly greater than the target: #{K.index}
        </text>
      </Frame>
    ),
  },
  {
    clip: 'find-u-2',
    label: 'Chunks',
    draw: (p) => {
      const shown = K.chunks.filter((_, c) => p > 0.2 + c * 0.35);
      return (
        <Frame beat={2} headline={`walk the row ${CHUNK} survivors at a time; add each chunk to the aggregate`}>
          <Bar y={90} chunks state={(i) => {
            const c = Math.floor(i / CHUNK);
            const ch = shown.find((x) => x.c === c);
            return ch && !ch.crosses ? 'skip' : 'plain';
          }} />
          {shown.map((ch) => (
            <text key={ch.c} x={14} y={190 + ch.c * 24} fontSize="12" data-chunk={ch.c} data-crosses={ch.crosses ? 1 : 0} data-agg={ch.aggAfter}
              style={{ ...MONO, fill: 'var(--ink)' }}>
              chunk {ch.c}: sum {short(ch.sum)}, aggregate {short(ch.aggBefore)} → {short(ch.aggAfter)}
              {ch.crosses ? ` > ${short(K.target)}: it crosses, scan it` : ` ≤ ${short(K.target)}: skip, no scan`}
            </text>
          ))}
        </Frame>
      );
    },
  },
  {
    clip: 'find-u-3',
    label: 'Scan and flag',
    draw: (p) => {
      const upto = Math.ceil(ramp(p, 0.05, 0.55) * SCENE_CFG.threads);
      return (
        <Frame beat={3} headline={`scan chunk ${crossing.c}; a lane flags a survivor whose sum is greater than ${short(K.target)}`}>
          <LaneGrid y={70} upto={upto} flags={p > 0.6} />
          <text x={14} y={210} fontSize="12" opacity={ramp(p, 0.65, 0.8)} style={{ fill: 'var(--ink)' }}>
            {crossing.lanes.reduce((s, l) => s + l.flags.filter(Boolean).length, 0)} survivors flagged; the first of them is the token
          </text>
        </Frame>
      );
    },
  },
  {
    clip: 'find-u-4',
    label: 'The vote',
    draw: (p) => {
      const nW = crossing.ballots.length;
      return (
        <Frame beat={4} headline="one ballot per warp; the lowest flagged lane of the lowest flagged warp writes the token">
          <LaneGrid y={70} upto={SCENE_CFG.threads} flags />
          {crossing.ballots.map((b, k) => (
            <g key={k} opacity={ramp(p, 0.1 + k * 0.2, 0.25 + k * 0.2)} data-ballot={k} data-bits={b} data-first={crossing.warpFirst[k]}>
              <text x={14} y={220 + k * 22} fontSize="12" style={{ ...MONO, fill: 'var(--ink)' }}>
                warp {k} ballot (lane 0 on the right): {ballotBits(b, SCENE_CFG.warp)} → lowest flagged lane {crossing.warpFirst[k] < 0 ? 'none' : crossing.warpFirst[k]}
              </text>
            </g>
          ))}
          <text x={14} y={226 + nW * 22} fontSize="12.5" fontWeight="700" opacity={ramp(p, 0.65, 0.8)} data-winner={K.index} style={{ fill: 'var(--ok)' }}>
            warp {K.winner!.warp}, lane {K.winner!.lane} (thread {K.winner!.thread}) writes survivor #{K.index}
          </text>
        </Frame>
      );
    },
  },
  {
    clip: 'find-u-5',
    label: 'Near a boundary',
    draw: (p) => {
      const L = cdfIn(SCENE_WEIGHTS, 'f32-loop', SCENE_CFG);
      const i = BND.boundary;
      return (
        <Frame beat={5} headline={`u = ${short(BND.u)}: two orders of addition, one target, neighbouring tokens`}>
          <text x={14} y={70} fontSize="12" style={{ fill: 'var(--ink-2)' }}>the inclusive sum at survivor #{i}, the boundary between #{i} and #{i + 1}</text>
          <g opacity={ramp(p, 0.1, 0.3)} data-order="loop" data-cum={BND.loop.cum} data-index={BND.loop.index}>
            <text x={14} y={104} fontSize="12.5" style={{ ...MONO, fill: 'var(--ink)' }}>
              left to right:    sum {short(BND.loop.cum)}, target {short(BND.loop.target)} → not greater → token #{BND.loop.index}
            </text>
          </g>
          <g opacity={ramp(p, 0.35, 0.55)} data-order="kernel" data-cum={BND.kernel.cum} data-index={BND.kernel.index}>
            <text x={14} y={132} fontSize="12.5" style={{ ...MONO, fill: 'var(--ink)' }}>
              kernel's order:   sum {short(BND.kernel.cum)}, target {short(BND.kernel.target)} → greater → token #{BND.kernel.index}
            </text>
          </g>
          <text x={14} y={176} fontSize="12" opacity={ramp(p, 0.6, 0.75)} style={{ fill: 'var(--ink)' }}>
            the two sums differ in the last bit; the row's total in double is {G.total.toPrecision(9)} and float32 total {short(L.total)}
          </text>
          <text x={14} y={204} fontSize="12" fontWeight="600" opacity={ramp(p, 0.78, 0.92)} style={{ fill: 'var(--ink)' }}>
            a cross-backend test accepts either #{i} or #{i + 1} for a draw this close to the boundary
          </text>
        </Frame>
      );
    },
  },
  {
    clip: 'find-u-6',
    label: 'The two ends',
    draw: (p) => {
      const gt = inverseCdf(GE_GT.weights, GE_GT.u, { rule: 'gt' });
      const ge = inverseCdf(GE_GT.weights, GE_GT.u, { rule: 'ge' });
      const { cum } = cdfIn(FALLBACK_WEIGHTS, 'f32-kernel', SCENE_CFG);
      return (
        <Frame beat={6} headline="u = 0 never picks a zero weight; nothing above the target falls back">
          <g opacity={ramp(p, 0.05, 0.2)} data-end="zero" data-gt={gt.index} data-ge={ge.index}>
            <text x={14} y={66} fontSize="12" style={{ fill: 'var(--ink-2)' }}>weights [0, 0.5, 0.5], u = 0, so the target is 0; inclusive sums 0, 0.5, 1</text>
            <text x={14} y={92} fontSize="12.5" style={{ ...MONO, fill: 'var(--ink)' }}>strictly greater than 0: survivor #{gt.index} (weight 0.5)</text>
            <text x={14} y={114} fontSize="12.5" style={{ ...MONO, fill: 'var(--fail)' }}>greater or equal:   survivor #{ge.index} (weight 0)</text>
          </g>
          <g opacity={ramp(p, 0.45, 0.6)} data-end="fallback" data-index={FB.index} data-fallback={FB.fallback ? 1 : 0}>
            <text x={14} y={168} fontSize="12" style={{ fill: 'var(--ink-2)' }}>
              {FALLBACK_WEIGHTS.length} survivors, M5's largest u = 1 − 2⁻²⁴: total {short(FB.total)} (strided sums, then warp trees)
            </text>
            <text x={14} y={194} fontSize="12.5" style={{ ...MONO, fill: 'var(--ink)' }}>
              target {short(FB.target)}; the scan's last sum {short(cum[cum.length - 1])}
              {cum[cum.length - 1] === FB.target ? ', equal, so not greater' : ', below it'}: nothing qualifies
            </text>
            <text x={14} y={220} fontSize="12.5" fontWeight="700" opacity={ramp(p, 0.7, 0.85)} style={{ fill: 'var(--ok)' }}>
              fallback: the last survivor with positive weight, #{FB.index}
            </text>
            <text x={14} y={246} fontSize="11.5" opacity={ramp(p, 0.75, 0.9)} style={{ fill: 'var(--ink-2)' }}>
              u = {short(TOP_U)}; the total and the scan add the same weights in different orders
            </text>
          </g>
        </Frame>
      );
    },
  },
];

export function FindUScene() {
  return <Scene chapter="x1" title="Finding u in the scan" beats={beats} />;
}
