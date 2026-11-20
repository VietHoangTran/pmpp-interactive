import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { CORRECT, blurImage, patch, spot, type BlurOptions, type BlurResult } from '../../sim/blur';

// ---------------------------------------------------------------- drawing a small picture
// Shared by this scene and by the interactives of section 3.3.
export interface CellLook {
  text?: string;
  fill?: string;
  ink?: string;
  stroke?: string;
  strokeWidth?: number;
  dash?: boolean;
  /** Diagonal hatching: a position that is not part of the picture. */
  hatch?: boolean;
  bold?: boolean;
}

/**
 * The shade of a pixel value. A picture has to look the same in the light and the dark
 * theme, so these greys are computed from the data and are the one place where a colour
 * is not a theme variable. Low values are stretched so that a faint smudge stays visible;
 * the number printed in the cell is the data.
 */
export function grey(v: number): { fill: string; ink: string } {
  if (Number.isNaN(v)) return { fill: 'var(--fail-wash)', ink: 'var(--fail)' };
  const g = Math.round(255 * Math.pow(Math.max(0, Math.min(255, v)) / 255, 0.6));
  return { fill: `rgb(${g},${g},${g})`, ink: g > 128 ? 'rgb(20,20,20)' : 'rgb(242,242,242)' };
}
export const show = (v: number) => (Number.isNaN(v) ? '?' : String(v));

/** Cells of a w by h picture, plus `pad` cells of surroundings on every side, as an SVG group. */
export function PixelCells({
  x = 0, y = 0, w, h, pad = 0, cell = 30, look, hatchId, onCell,
}: {
  x?: number; y?: number; w: number; h: number; pad?: number; cell?: number;
  look: (row: number, col: number, inside: boolean) => CellLook | null;
  hatchId?: string;
  onCell?: (row: number, col: number) => void;
}) {
  const cells: ReactNode[] = [];
  const over: ReactNode[] = [];
  for (let row = -pad; row < h + pad; ++row)
    for (let col = -pad; col < w + pad; ++col) {
      const inside = row >= 0 && row < h && col >= 0 && col < w;
      const k = look(row, col, inside);
      if (!k) continue;
      const cx = x + (col + pad) * cell;
      const cy = y + (row + pad) * cell;
      const emphasised = (k.strokeWidth ?? 1) > 1;
      const box = (
        <g key={row + ',' + col} data-rc={row + ',' + col} onClick={onCell ? () => onCell(row, col) : undefined}>
          <rect x={cx + 1} y={cy + 1} width={cell - 2} height={cell - 2} rx="2"
            fill={k.fill ?? 'var(--cell)'} stroke={k.stroke ?? 'var(--rule)'} strokeWidth={k.strokeWidth ?? 1}
            strokeDasharray={k.dash ? '3 3' : undefined} />
          {k.hatch && hatchId ? <rect x={cx + 1} y={cy + 1} width={cell - 2} height={cell - 2} rx="2" fill={`url(#${hatchId})`} /> : null}
          {k.text != null ? (
            <text x={cx + cell / 2} y={cy + cell / 2 + cell * 0.13} fontSize={cell * 0.36} textAnchor="middle"
              fontWeight={k.bold ? 700 : 400} style={{ fill: k.ink ?? 'var(--ink)', fontVariantNumeric: 'tabular-nums' }}>{k.text}</text>
          ) : null}
        </g>
      );
      (emphasised ? over : cells).push(box); // thick outlines are drawn last so neighbours do not cover them
    }
  return <g>{cells}{over}</g>;
}

export function Hatch({ id }: { id: string }) {
  return (
    <defs>
      <pattern id={id} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink-3)" strokeWidth="1.4" />
      </pattern>
    </defs>
  );
}

// ---------------------------------------------------------------- the scene
// "Why the blur writes somewhere else": a 6x6 picture with one bright pixel, blurred
// into a second buffer and then in place, in two thread orders. Every picture below is
// computed by app/src/sim/blur.ts; nothing is drawn by hand.
const W = 6;
const H = 6;
const N = W * H;
const C = 36;
const LX = 12;   // left picture
const RX = 258;  // right picture
const GY = 46;
const TX = 496;  // text column
const pic = spot(W, H);
const base: BlurOptions = { ...CORRECT, blockW: W, blockH: H };
const states = (over: Partial<BlurOptions>) =>
  Array.from({ length: N + 1 }, (_, limit) => blurImage(pic, { ...base, ...over, limit }));
const separate = states({});
const forward = states({ inPlace: true });
const reversed = states({ inPlace: true, order: 'reversed' });
const ref = separate[N].reference;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const wrongSoFar = (s: BlurResult) => s.out.filter((v, i) => s.written[i] && v !== ref[i]).length;
const inPatch = (t: { row: number; col: number } | undefined, row: number, col: number) =>
  !!t && patch(t.row, t.col, W, H, 1).some((q) => q.inside && q.curRow === row && q.curCol === col);

interface Grid {
  x: number;
  title: string;
  tier?: boolean;
  look: (row: number, col: number) => CellLook;
}
function Stage({ grids, lines, rule }: { grids: Grid[]; lines: { text: string; tone?: 'fail' | 'ok' | 'muted'; bold?: boolean }[]; rule?: number }) {
  return (
    <svg viewBox="0 0 720 330" role="img" aria-label="Animation of a blur kernel writing to a second buffer and then in place"
      style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      {grids.map((g) => (
        <g key={g.title}>
          <text x={g.x} y={GY - 12} fontSize="12" fontWeight="600" style={{ fill: g.tier === false ? 'var(--ink-2)' : 'var(--global)' }}>{g.title}</text>
          <PixelCells x={g.x} y={GY} w={W} h={H} cell={C} look={(r, c) => g.look(r, c)} />
        </g>
      ))}
      {lines.map((l, i) => (
        <text key={i} x={TX} y={GY + 10 + i * 19} fontSize="12" fontWeight={l.bold ? 650 : 400}
          style={{ fill: l.tone === 'fail' ? 'var(--fail)' : l.tone === 'ok' ? 'var(--ok)' : l.tone === 'muted' ? 'var(--ink-2)' : 'var(--ink)' }}>{l.text}</text>
      ))}
      {rule ? (
        <g opacity={rule}>
          <text x={LX} y={GY + H * C + 30} fontSize="13.5" fontWeight="600">Each thread writes only its own element,</text>
          <text x={LX} y={GY + H * C + 50} fontSize="13.5" fontWeight="600">and reads only data that no thread writes.</text>
        </g>
      ) : null}
    </svg>
  );
}

/** A buffer as it stands in state s, with the reads of thread t outlined and wrong stored values marked. */
const buffer = (s: BlurResult, opts: { reader?: { row: number; col: number }; mark?: boolean; blankUnwritten?: boolean; hot?: number } = {}) =>
  (row: number, col: number): CellLook => {
    const i = row * W + col;
    if (opts.blankUnwritten && !s.written[i]) return { dash: true };
    const v = s.out[i];
    const bad = opts.mark && s.written[i] && v !== ref[i];
    const read = inPatch(opts.reader, row, col);
    return {
      text: show(v), ...grey(v),
      stroke: bad ? 'var(--fail)' : read ? 'var(--global)' : i === opts.hot ? 'var(--ink)' : undefined,
      strokeWidth: bad || read || i === opts.hot ? 3 : 1,
    };
  };
const input = (reader?: { row: number; col: number }) => (row: number, col: number): CellLook => {
  const v = pic.data[row * W + col];
  const read = inPatch(reader, row, col);
  return { text: show(v), ...grey(v), stroke: read ? 'var(--global)' : undefined, strokeWidth: read ? 3 : 1 };
};
const plain = (values: number[], mark = false) => (row: number, col: number): CellLook => {
  const v = values[row * W + col];
  const bad = mark && v !== ref[row * W + col];
  return { text: show(v), ...grey(v), stroke: bad ? 'var(--fail)' : undefined, strokeWidth: bad ? 3 : 1 };
};
const at = (t: { row: number; col: number }) => `(${t.row}, ${t.col})`;

const beats: Beat[] = [
  {
    clip: 'in-place-blur-1',
    label: 'Two buffers',
    draw: (p) => {
      const k = Math.min(N, Math.floor(ramp(p, 0.06, 0.78) * (N + 1)));
      const s = separate[k];
      const t = k < N ? separate[N].ran[k] : undefined;
      return (
        <Stage
          grids={[
            { x: LX, title: 'in: every thread reads here', look: input(t) },
            { x: RX, title: 'out: every thread stores one element', look: buffer(s, { blankUnwritten: true, hot: k > 0 ? separate[N].ran[k - 1].row * W + separate[N].ran[k - 1].col : undefined }) },
          ]}
          lines={[
            { text: `${k} of ${N} threads done`, bold: true },
            { text: t ? `thread ${at(t)} reads its patch` : 'all threads done', tone: 'muted' },
            ...(k === N ? [{ text: `${N - separate[N].wrong.length} of ${N} match the reference`, tone: 'ok' as const, bold: true }] : []),
          ]}
        />
      );
    },
  },
  {
    clip: 'in-place-blur-2',
    label: 'One buffer',
    draw: (p) => {
      const k = Math.min(8, Math.floor(ramp(p, 0.3, 0.6) * 9));
      const s = forward[k];
      const next = p >= 0.74 ? forward[N].ran[8] : undefined; // thread (1,2), the neighbour
      return (
        <Stage
          grids={[
            { x: LX, title: 'one buffer: in and out', look: buffer(s, { mark: true, reader: next }) },
            { x: RX, title: 'what it should become', tier: false, look: plain(ref) },
          ]}
          lines={[
            { text: `${k} of ${N} threads done`, bold: true },
            ...(k >= 7 ? [{ text: '7 threads stored a 0 over a 0', tone: 'muted' as const }] : []),
            ...(k >= 8 ? [{ text: `thread (1, 1) stores ${forward[8].out[7]}`, tone: 'fail' as const, bold: true }] : []),
            ...(next ? [{ text: `thread ${at(next)} is next:` }, { text: `it reads ${forward[8].out[7]} where in held ${pic.data[7]}` }] : []),
          ]}
        />
      );
    },
  },
  {
    clip: 'in-place-blur-3',
    label: 'In order',
    draw: (p) => {
      const k = 8 + Math.min(N - 8, Math.floor(ramp(p, 0.04, 0.62) * (N - 7)));
      const s = forward[k];
      const done = k === N;
      return (
        <Stage
          grids={[
            { x: LX, title: 'one buffer, threads in order', look: buffer(s, { mark: true, reader: done ? undefined : forward[N].ran[k] }) },
            { x: RX, title: 'the reference', tier: false, look: plain(ref) },
          ]}
          lines={[
            { text: `${k} of ${N} threads done`, bold: true },
            { text: `wrong so far: ${wrongSoFar(s)}`, tone: 'fail', bold: true },
            ...(done
              ? [
                  { text: `${s.wrong.length} of ${N} pixels are wrong` },
                  { text: 'accesses outside the buffer: ' + s.accesses.length, tone: 'muted' as const },
                ]
              : []),
          ]}
        />
      );
    },
  },
  {
    clip: 'in-place-blur-4',
    label: 'Reversed',
    draw: (p) => {
      const k = Math.min(N, Math.floor(ramp(p, 0.14, 0.62) * (N + 1)));
      const s = reversed[k];
      const done = k === N;
      const differ = forward[N].out.filter((v, i) => v !== reversed[N].out[i]).length;
      return (
        <Stage
          grids={[
            { x: LX, title: 'same kernel, threads reversed', look: buffer(s, { mark: true, reader: done ? undefined : reversed[N].ran[k] }) },
            { x: RX, title: 'the run before, threads in order', look: plain(forward[N].out, true) },
          ]}
          lines={[
            { text: `${k} of ${N} threads done`, bold: true },
            { text: `wrong so far: ${wrongSoFar(s)}`, tone: 'fail', bold: true },
            ...(done ? [{ text: `the two runs differ in ${differ} pixels` }, { text: 'same code, same input', tone: 'muted' as const }] : []),
          ]}
        />
      );
    },
  },
  {
    clip: 'in-place-blur-5',
    label: 'The rule',
    draw: (p) => (
      <Stage
        grids={[
          { x: LX, title: 'in: read by many, written by nobody', look: input() },
          { x: RX, title: 'out: each element written by one thread', look: plain(separate[N].out) },
        ]}
        lines={[
          { text: 'grayscale, blur, matmul:', bold: true },
          { text: 'all three keep the rule', tone: 'muted' },
          ...(p > 0.62 ? [{ text: 'chapter 5: shared memory' }, { text: 'that threads also write,', tone: 'muted' as const }, { text: 'and the barriers it needs', tone: 'muted' as const }] : []),
        ]}
        rule={ramp(p, 0.12, 0.3)}
      />
    ),
  },
];

export function InPlaceBlurScene() {
  return <Scene chapter="ch03" title="Why the blur writes somewhere else" beats={beats} />;
}
