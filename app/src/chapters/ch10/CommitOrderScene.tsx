import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { f32, f32hex, f32short, f64short, halvingLevels, sceneData, segments, type CommitTrace } from '../../sim/multiblock';

// "Blocks finish in any order" (section 10.9): the same eight block sums committed in
// two seeded orders, then reduced by a fixed second pass. Five beats, one narration
// clip each (content/ch10/narration/commit-order-N.txt). Every value, order and count
// drawn comes from sim/multiblock.ts (sceneData, halvingLevels, segments).
const D = sceneData();
const B = D.seg.blocks; // 8
const PER = D.seg.perBlock; // 8
const LEVELS = halvingLevels(D.sums);
export const MILLION = segments(1_000_000, 1024);

const W = 720;
const H = 348;
const X0 = 30;
const CW = 9; // cell width
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

/** x of input slot i when segments are pulled `gap` pixels apart. */
const cellX = (i: number, gap: number) => X0 + i * CW + Math.floor(i / PER) * gap;
const GAP = 12;
const segMid = (b: number) => cellX(b * PER, GAP) + (PER * CW) / 2;

/** How many commits of a run have happened at progress p. */
const done = (p: number, a = 0.1, b = 0.9) => Math.min(B, Math.floor(ramp(p, a, b) * B + 1e-9));

function Input({ gap, labels, startLabels }: { gap: number; labels: number; startLabels: number }) {
  return (
    <g>
      <text x={X0} y={16} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>
        input: {D.seg.N} logits in global memory, blocks of {D.seg.block} threads, segments of {PER}
      </text>
      {Array.from({ length: B * PER }, (_, i) => {
        const real = i < D.seg.N;
        return (
          <rect key={i} x={cellX(i, gap)} y={26} width={CW - 1} height={20} rx="1"
            fill={real ? 'var(--global-wash)' : 'url(#co-hatch)'} stroke={real ? 'var(--global)' : 'var(--ink-3)'} strokeWidth="0.7" />
        );
      })}
      {Array.from({ length: B }, (_, b) => (
        <g key={b} opacity={labels}>
          <text x={segMid(b)} y={60} fontSize="11" textAnchor="middle">block {b}</text>
          <text x={segMid(b)} y={73} fontSize="10" textAnchor="middle" opacity={startLabels} style={{ fill: 'var(--ink-2)' }}>
            starts at {b * PER}
          </text>
        </g>
      ))}
    </g>
  );
}

function Sums({ show, committed, current }: { show: number; committed?: Map<number, number>; current?: number }) {
  return (
    <g opacity={show}>
      {D.sums.map((s, b) => {
        const k = committed?.get(b);
        const on = current === b;
        return (
          <g key={b} data-block={b}>
            <rect x={segMid(b) - 38} y={84} width={76} height={24} rx="3"
              fill={on ? 'var(--reg)' : 'var(--reg-wash)'} stroke="var(--reg)" opacity={k != null && !on ? 0.45 : 1} />
            <text x={segMid(b)} y={100} fontSize="11" textAnchor="middle" style={{ fill: on ? 'var(--page)' : 'var(--ink)' }}>
              {f32short(s)}
            </text>
            {k != null ? (
              <text x={segMid(b) + 34} y={81} fontSize="10" textAnchor="end" fontWeight="600">#{k + 1}</text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

/** The atomic total and the log of a run's commits up to `n`. */
function Commits({ run, n, label, y0 = 128 }: { run: CommitTrace; n: number; label: string; y0?: number }) {
  const total = n > 0 ? run.running[n - 1] : 0;
  const cur = n > 0 ? run.order[n - 1] : -1;
  return (
    <g>
      <text x={X0} y={y0 + 14} fontSize="12" fontWeight="600">{label}</text>
      <rect x={X0} y={y0 + 22} width={200} height={44} rx="3" fill="var(--global-wash)" stroke="var(--global)" />
      <text x={X0 + 10} y={y0 + 38} fontSize="10.5" style={{ fill: 'var(--global)' }}>*output (global memory)</text>
      <text x={X0 + 10} y={y0 + 58} fontSize="15" fontWeight="600" data-total={f32hex(total)}>{f32short(total)}</text>
      {cur >= 0 ? (
        <text x={X0 + 192} y={y0 + 38} fontSize="10.5" textAnchor="end" fontWeight="600">+ block {cur}</text>
      ) : null}
      {run.order.slice(0, n).map((b, k) => (
        <text key={k} x={270} y={y0 + 38 + k * 17} fontSize="11" data-commit={k}>
          {`commit ${k + 1}: block ${b} adds ${f32short(D.sums[b])}, total ${f32short(run.running[k])}`}
        </text>
      ))}
    </g>
  );
}

/** Two hex strings with the digits where they differ in bold. */
function Hex({ x, y, value, other }: { x: number; y: number; value: number; other: number }) {
  const a = f32hex(value);
  const b = f32hex(other);
  return (
    <text x={x} y={y} fontSize="13" className="mono">
      {[...a].map((ch, i) => (
        <tspan key={i} fontWeight={ch !== b[i] ? 700 : 400} textDecoration={ch !== b[i] ? 'underline' : undefined}>{ch}</tspan>
      ))}
    </text>
  );
}

function Stage({ children, note = true }: { children: ReactNode; note?: boolean }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Blocks committing their sums" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <defs>
        <pattern id="co-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.4" />
        </pattern>
      </defs>
      {children}
      {note ? (
        <text x={W - 10} y={H - 6} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
          commit orders are seeded shuffles from the model, not a timing model
        </text>
      ) : null}
    </svg>
  );
}

const committedMap = (run: CommitTrace, n: number) => new Map(run.order.slice(0, n).map((b, k) => [b, k]));

const beats: Beat[] = [
  {
    clip: 'commit-order-1',
    label: 'Segments',
    draw: (p) => {
      const gap = GAP * ramp(p, 0.08, 0.35);
      const last = ramp(p, 0.62, 0.8);
      return (
        <Stage note={false}>
          <Input gap={gap} labels={ramp(p, 0.3, 0.45)} startLabels={ramp(p, 0.4, 0.55)} />
          <g opacity={last}>
            <rect x={cellX((B - 1) * PER, GAP) - 3} y={22} width={PER * CW + 5} height={28} fill="none" stroke="var(--ink)" strokeWidth="2" />
            <text x={segMid(B - 1)} y={98} fontSize="11.5" textAnchor="middle">
              {D.seg.lastReal} real,
            </text>
            <text x={segMid(B - 1)} y={112} fontSize="11.5" textAnchor="middle">
              {D.seg.pastEnd} past the end
            </text>
          </g>
          <g opacity={ramp(p, 0.8, 0.92)}>
            <text x={X0} y={160} fontSize="13">
              {`block b starts at 2 × ${D.seg.block} × b = ${PER}b; ${D.seg.N} logits need ceil(${D.seg.N} / ${PER}) = ${B} blocks.`}
            </text>
            <text x={X0} y={182} fontSize="13">
              The hatched slots load nothing: those threads add 0 and still join the block's reduction.
            </text>
          </g>
        </Stage>
      );
    },
  },
  {
    clip: 'commit-order-2',
    label: 'Commit',
    draw: (p) => {
      const n = done(p, 0.22, 0.95);
      return (
        <Stage>
          <Input gap={GAP} labels={1} startLabels={0} />
          <Sums show={ramp(p, 0, 0.15)} committed={committedMap(D.runA, n)} current={n > 0 ? D.runA.order[n - 1] : undefined} />
          <Commits run={D.runA} n={n} label="run 1: leaders add with an atomic, in the order they finish" />
        </Stage>
      );
    },
  },
  {
    clip: 'commit-order-3',
    label: 'Run it again',
    draw: (p) => {
      const n = done(p, 0.08, 0.55);
      const compare = ramp(p, 0.6, 0.72);
      return (
        <Stage>
          <Input gap={GAP} labels={1} startLabels={0} />
          <Sums show={1} committed={committedMap(D.runB, n)} current={n > 0 && n < B ? D.runB.order[n - 1] : undefined} />
          <g opacity={1 - compare}>
            <Commits run={D.runB} n={n} label="run 2: the same block sums, another order" />
          </g>
          <g opacity={compare} data-compare="1">
            <text x={X0} y={150} fontSize="12" fontWeight="600">the two runs, side by side</text>
            <text x={X0} y={178} fontSize="13">run 1 ({D.runA.order.join(', ')}):</text>
            <text x={X0 + 260} y={178} fontSize="13">{f32short(D.runA.total)}</text>
            <Hex x={X0 + 400} y={178} value={D.runA.total} other={D.runB.total} />
            <text x={X0} y={204} fontSize="13">run 2 ({D.runB.order.join(', ')}):</text>
            <text x={X0 + 260} y={204} fontSize="13">{f32short(D.runB.total)}</text>
            <Hex x={X0 + 400} y={204} value={D.runB.total} other={D.runA.total} />
            <text x={X0} y={236} fontSize="13" style={{ fill: 'var(--ink-2)' }}>float64 reference:</text>
            <text x={X0 + 260} y={236} fontSize="13" style={{ fill: 'var(--ink-2)' }}>{f64short(D.ref)}</text>
            <text x={X0 + 400} y={236} fontSize="13" className="mono" style={{ fill: 'var(--ink-2)' }}>nearest float32 {f32hex(f32(D.ref))}</text>
          </g>
        </Stage>
      );
    },
  },
  {
    clip: 'commit-order-4',
    label: 'Second pass',
    draw: (p) => {
      // Sums land in the array in run 2's order, each at its own index. Each tree level
      // sits under the slots it overwrites: slot j absorbs slot j + half.
      const written = done(p, 0.05, 0.35);
      const at = new Set(D.runB.order.slice(0, written));
      const lv = p < 0.42 ? 0 : p < 0.55 ? 1 : p < 0.68 ? 2 : 3;
      const BW = 72;
      const slotX = (j: number) => X0 + 52 + j * (BW + 6);
      const last = LEVELS.length - 1;
      return (
        <Stage>
          <Input gap={GAP} labels={1} startLabels={0} />
          <Sums show={1} />
          <text x={X0} y={140} fontSize="12" fontWeight="600">leaders write partial[blockIdx.x]; a second launch, one block, reduces it</text>
          {LEVELS.map((level, d) => {
            const y = 152 + d * 34;
            const half = level.length;
            return (
              <g key={d} data-level={d}>
                <text x={X0} y={y + 16} fontSize="10.5" style={{ fill: d === 0 ? 'var(--global)' : 'var(--ink-2)' }} opacity={d === 0 || d <= lv ? 1 : 0}>
                  {d === 0 ? 'partial' : `j += j+${half}`}
                </text>
                {level.map((v, j) => {
                  const visible = d === 0 ? at.has(j) : d <= lv;
                  return (
                    <g key={j} opacity={visible ? 1 : d === 0 ? 0.3 : 0}>
                      <rect x={slotX(j)} y={y} width={BW} height={24} rx="3"
                        fill={d === 0 ? 'var(--global-wash)' : d === last ? 'var(--ink)' : 'var(--shared-wash)'}
                        stroke={d === 0 ? 'var(--global)' : d === last ? 'var(--ink)' : 'var(--shared)'} />
                      <text x={slotX(j) + BW / 2} y={y + 16} fontSize="11" textAnchor="middle"
                        style={{ fill: d === last ? 'var(--page)' : 'var(--ink)' }}>
                        {d === 0 && !visible ? `[${j}]` : f32short(v)}
                      </text>
                    </g>
                  );
                })}
              </g>
            );
          })}
          <g opacity={ramp(p, 0.75, 0.85)} data-final={f32hex(LEVELS[last][0])}>
            <text x={slotX(2)} y={152 + last * 34 + 11} fontSize="12" className="mono">run 1: {f32hex(D.twoPass.total)}</text>
            <text x={slotX(2)} y={152 + last * 34 + 26} fontSize="12" className="mono">run 2: {f32hex(D.twoPass.total)}</text>
            <rect x={slotX(4) + 10} y={152 + last * 34 + 2} width={120} height={22} rx="11" fill="var(--ok-wash)" stroke="var(--ok)" />
            <text x={slotX(4) + 70} y={152 + last * 34 + 17} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ok)' }}>same every run</text>
          </g>
        </Stage>
      );
    },
  },
  {
    clip: 'commit-order-5',
    label: 'The cost',
    draw: (p) => (
      <Stage note={false}>
        <text x={X0} y={30} fontSize="14" fontWeight="600">what the second pass costs</text>
        <g opacity={ramp(p, 0.05, 0.15)}>
          <text x={X0} y={60} fontSize="13">one more launch, and one float per block:</text>
          <text x={X0 + 20} y={84} fontSize="13">{`here: ${B} blocks, ${B} floats, ${4 * B} bytes`}</text>
        </g>
        <g opacity={ramp(p, 0.22, 0.32)}>
          <text x={X0 + 20} y={108} fontSize="13" data-million={MILLION.blocks}>
            {`1,000,000 logits at 1024 threads: ${MILLION.blocks} blocks, ${MILLION.blocks} floats, ${(4 * MILLION.blocks).toLocaleString('en-US')} bytes`}
          </text>
        </g>
        <g opacity={ramp(p, 0.45, 0.55)}>
          <text x={X0} y={150} fontSize="13" fontWeight="600">committing across blocks with an atomic</text>
          {[
            ['sum', 'float atomic add on CUDA, HIP and Metal'],
            ['max', 'no float atomic max on Metal or in CUDA’s legacy API (HIP has one)'],
            ['argmax', 'value and index must move together: a 64-bit key, or two passes'],
          ].map(([op, what], i) => (
            <g key={op}>
              <text x={X0 + 20} y={176 + i * 24} fontSize="13" fontWeight="600">{op}</text>
              <text x={X0 + 90} y={176 + i * 24} fontSize="13">{what}</text>
            </g>
          ))}
        </g>
        <g opacity={ramp(p, 0.75, 0.85)}>
          <text x={X0} y={270} fontSize="13">two passes work on every backend:</text>
          <text x={X0 + 20} y={294} fontSize="13">llama.cpp's Metal argmax uses two dispatches, a partial pass and a final pass</text>
        </g>
      </Stage>
    ),
  },
];

export function CommitOrderScene() {
  return <Scene chapter="ch10" title="Blocks finish in any order" beats={beats} />;
}
