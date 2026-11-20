import { useMemo, useState, type CSSProperties, type KeyboardEvent, type MouseEvent, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import {
  BOOK,
  CHANNELS,
  GRAY_WEIGHTS,
  STYLE_LABEL,
  blockBands,
  blockInfo,
  bookLabel,
  elementAt,
  fmtFloat,
  grayDetail,
  indexingCoverage,
  latticeOf,
  layout,
  offsetOf,
  regions,
  stepsGrayscale,
  stridesFor,
  threadAt,
  type Dims,
  type GrayOp,
  type GrayVariant,
  type Indexing,
  type Launch,
  type RegionId,
  type Style,
  type XY,
} from '../../sim/mapping';
import { GridOverPictureScene } from './GridOverPictureScene';

// ---------------------------------------------------------------- code shown in this section
// Copies of labs/ch03/snippets/*. S32.snippets.test.ts fails if a copy drifts from its file.

/** labs/ch03/snippets/grayscale_kernel.cu: the corrected kernel and its launch. */
export const GRAY_CU = `#define CHANNELS 3   // bytes per colour pixel: r, g, b

// One thread per pixel. Pin holds three bytes for each pixel, Pout one.
__global__ void colorToGrayscaleConversion(unsigned char* Pout,
                                           unsigned char* Pin,
                                           int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        // Offset in the flat gray picture: skip the rows above.
        int grayOffset = row * width + col;
        // The colour picture: the same walk, CHANNELS bytes a pixel.
        int rgbOffset = grayOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        // The cast covers the whole sum. It drops the fraction.
        Pout[grayOffset] =
            (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// n rows (y), m columns (x): x is sized from the width, y from the height.
void launchColorToGrayscale(unsigned char* Pin_d, unsigned char* Pout_d,
                            int m, int n) {
    dim3 dimGrid((m + 15) / 16, (n + 15) / 16, 1);
    dim3 dimBlock(16, 16, 1);
    colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, m, n);
}`;

/** labs/ch03/snippets/grayscale_kernel.metal */
export const GRAY_METAL = `#include <metal_stdlib>
using namespace metal;

#define CHANNELS 3   // bytes per colour pixel: r, g, b

// The same kernel in Metal Shading Language. gid is the thread's position in the
// whole grid: .x is the column, .y the row. Nothing has to be multiplied out.
kernel void color_to_grayscale(device const uchar* Pin  [[buffer(0)]],
                               device uchar*       Pout [[buffer(1)]],
                               constant uint2&     size [[buffer(2)]],   // (width, height)
                               uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x;
    uint row = gid.y;
    // Needed when the launch is dispatchThreadgroups. With dispatchThreads over
    // (width, height) it is never false.
    if (col < size.x && row < size.y) {
        uint grayOffset = row * size.x + col;
        uint rgbOffset = grayOffset * CHANNELS;
        uchar r = Pin[rgbOffset];
        uchar g = Pin[rgbOffset + 1];
        uchar b = Pin[rgbOffset + 2];
        Pout[grayOffset] = (uchar)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}`;

/** labs/ch03/snippets/grayscale_as_printed.cu, after its marker line: Fig. 3.4's code lines, comments reworded. */
export const PRINTED = `// Input bytes are unsigned chars, 0 to 255
// A pixel is 3 chars in a row, one per channel (RGB)
__global__
void colortoGrayscaleConvertion(unsigned char * Pout,
                unsigned char * Pin, int width, int height) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        // 1D offset into the gray picture
        int grayOffset = row*width + col;
        // The colour picture is CHANNELS times
        // as wide, counted in bytes
        int rgbOffset = grayOffset*CHANNELS;
        unsigned char r = Pin[rgbOffset    ]; // red
        unsigned char g = Pin[rgbOffset + 1]; // green
        unsigned char b = Pin[rgbOffset + 2]; // blue
        // Weight the channels, add them up
        // and store one byte
        Pout[grayOffset] = (unsigned char) 0.21f*r+0.71f*g+0.07f*b;
    }
}`;

/** The launch on book p. 50, as printed (a short quote; it is half of one of the defects). */
export const PRINTED_HOST = [
  'dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1);',
  'dim3 dimBlock(16, 16, 1);',
  'colorToGrayscaleConversion<<<dimGrid,dimBlock>>>(Pin_d, Pout_d, m, n);',
];

/** Which line of GRAY_CU each step of the model's stepper is on. */
export const LINE: Record<GrayOp, number> = {
  col: 7, row: 8, guard: 9, grayOffset: 11, rgbOffset: 13, r: 14, g: 15, b: 16, sum: 19, cast: 19, store: 18, done: 21,
};

const TIERS: Record<string, Tier> = {
  Pin: 'global', Pout: 'global', Pin_d: 'global', Pout_d: 'global',
  col: 'reg', row: 'reg', grayOffset: 'reg', rgbOffset: 'reg', r: 'reg', g: 'reg', b: 'reg', gid: 'reg',
};

// ---------------------------------------------------------------- small pieces

const mono: CSSProperties = { margin: '0.18rem 0', overflowWrap: 'anywhere' };
const Reg = ({ children }: { children: ReactNode }) => <b className="t-reg">{children}</b>;
const Glob = ({ children }: { children: ReactNode }) => <b className="t-global">{children}</b>;
/** Inline code that may wrap, for narrow panels. */
const Cw = ({ children }: { children: ReactNode }) => (
  <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>
);
const onKey = (fn: () => void) => (e: KeyboardEvent) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fn();
  }
};

function NumField({ label, value, min, max, onChange }: {
  label: string; value: number; min: number; max: number; onChange: (v: number) => void;
}) {
  const [text, setText] = useState(String(value));
  const [last, setLast] = useState(value);
  if (last !== value) {
    setLast(value);
    setText(String(value));
  }
  return (
    <label className="field">
      {label}
      <input type="number" min={min} max={max} step="1" value={text} style={{ width: '5.2rem' }}
        onChange={(e) => {
          setText(e.target.value);
          const v = Math.round(Number(e.target.value));
          if (e.target.value.trim() !== '' && v >= min && v <= max) {
            setLast(v);
            onChange(v);
          }
        }} />
    </label>
  );
}

// ---------------------------------------------------------------- the thread map

const PRESETS: { id: string; label: string; height: number; width: number; blockW: number; blockH: number }[] = [
  { id: 'book', label: "the book's 62 × 76, blocks 16 × 16", height: 62, width: 76, blockW: 16, blockH: 16 },
  { id: 'small', label: 'small: 10 × 13, blocks 4 × 4', height: 10, width: 13, blockW: 4, blockH: 4 },
  { id: 'ex3', label: 'exercise 3: 150 × 300, blocks 16 wide by 32 high', height: 150, width: 300, blockW: 16, blockH: 32 },
  { id: 'large', label: '1500 × 2000, blocks 16 × 16', height: 1500, width: 2000, blockW: 16, blockH: 16 },
];
const BLOCK_SHAPES: [number, number][] = [[4, 4], [8, 8], [16, 16], [32, 32], [16, 32]];
const STYLES: Style[] = ['blocks', 'threadgroups', 'threads'];
/** Past this many cells along a side, a thread is too small to click and the map draws blocks instead. */
const THREAD_LEVEL_MAX = 128;

const NAMES: Record<Style, { block: string; thread: string; dim: string; unit: string; Unit: string }> = {
  blocks: { block: 'blockIdx', thread: 'threadIdx', dim: 'blockDim', unit: 'block', Unit: 'Block' },
  threadgroups: {
    block: 'threadgroup_position_in_grid', thread: 'thread_position_in_threadgroup', dim: 'threads_per_threadgroup',
    unit: 'threadgroup', Unit: 'Threadgroup',
  },
  threads: {
    block: 'threadgroup_position_in_grid', thread: 'thread_position_in_threadgroup', dim: 'threads_per_threadgroup',
    unit: 'threadgroup', Unit: 'Threadgroup',
  },
};

const clampSel = (l: Launch, p: XY): XY => {
  const L = layout(l);
  return { x: Math.max(0, Math.min(p.x, L.threadsX - 1)), y: Math.max(0, Math.min(p.y, L.threadsY - 1)) };
};

interface Win {
  x0: number;
  y0: number;
  cols: number;
  rows: number;
}

/** A window onto the lattice of threads: one cell per thread, filled where the thread has a pixel. */
function Lattice({ id, launch, win, cell, sel, onPick, hot, hover, onHover, label }: {
  id: string; launch: Launch; win: Win; cell: number; sel: XY; onPick: (p: XY) => void;
  hot?: RegionId | null; hover?: XY | null; onHover?: (b: XY | null) => void; label: string;
}) {
  const L = layout(launch);
  const ML = 54, MT = 24, MR = 8, MB = 8;
  const W = win.cols * cell, H = win.rows * cell;
  const VBW = ML + W + MR, VBH = MT + H + MB;
  const gap = cell >= 6 ? 1.5 : 1;
  const inWin = sel.x >= win.x0 && sel.x < win.x0 + win.cols && sel.y >= win.y0 && sel.y < win.y0 + win.rows;
  const at = (e: MouseEvent<SVGSVGElement>): XY | null => {
    const r = e.currentTarget.getBoundingClientRect();
    const k = VBW / r.width;
    const x = win.x0 + Math.floor(((e.clientX - r.left) * k - ML) / cell);
    const y = win.y0 + Math.floor(((e.clientY - r.top) * k - MT) / cell);
    const ok = x >= win.x0 && y >= win.y0 && x < Math.min(L.threadsX, win.x0 + win.cols) && y < Math.min(L.threadsY, win.y0 + win.rows);
    return ok ? { x, y } : null;
  };
  const xs: number[] = [];
  for (let x = 0; x < L.threadsX; x += launch.blockW) xs.push(x);
  xs.push(L.threadsX);
  const ys: number[] = [];
  for (let y = 0; y < L.threadsY; y += launch.blockH) ys.push(y);
  ys.push(L.threadsY);
  const hotBlocks: XY[] = [];
  if (hot && L.blocks <= 4096)
    for (let by = 0; by < L.gridY; by++)
      for (let bx = 0; bx < L.gridX; bx++) if (blockInfo(launch, bx, by).region === hot) hotBlocks.push({ x: bx, y: by });
  const hb = hover ? blockInfo(launch, hover.x, hover.y) : null;
  const sx = ML + (sel.x - win.x0) * cell, sy = MT + (sel.y - win.y0) * cell;
  const key = (e: KeyboardEvent<SVGSVGElement>) => {
    const d: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    const m = d[e.key];
    if (!m) return;
    e.preventDefault();
    const sx2 = e.shiftKey ? launch.blockW : 1, sy2 = e.shiftKey ? launch.blockH : 1;
    onPick({
      x: Math.max(win.x0, Math.min(Math.min(L.threadsX, win.x0 + win.cols) - 1, sel.x + m[0] * sx2)),
      y: Math.max(win.y0, Math.min(Math.min(L.threadsY, win.y0 + win.rows) - 1, sel.y + m[1] * sy2)),
    });
  };
  return (
    <svg viewBox={`0 0 ${VBW} ${VBH}`} role="application" aria-label={label} tabIndex={0}
      data-lattice={id} data-ml={ML} data-mt={MT} data-cell={cell} data-x0={win.x0} data-y0={win.y0} data-vbw={VBW}
      style={{ width: VBW, maxWidth: '100%', height: 'auto', display: 'block', cursor: 'pointer' }}
      onClick={(e) => { const p = at(e); if (p) onPick(p); }}
      onMouseMove={(e) => {
        if (!onHover) return;
        const p = at(e);
        const b = p ? { x: Math.floor(p.x / launch.blockW), y: Math.floor(p.y / launch.blockH) } : null;
        if (b?.x !== hover?.x || b?.y !== hover?.y) onHover(b);
      }}
      onMouseLeave={() => onHover?.(null)}
      onKeyDown={key}>
      <defs>
        <pattern id={`${id}-f`} width={cell} height={cell} patternUnits="userSpaceOnUse">
          <rect width={cell - gap} height={cell - gap} fill="var(--ink-3)" />
        </pattern>
        <pattern id={`${id}-h`} width={cell} height={cell} patternUnits="userSpaceOnUse">
          <rect x="0.5" y="0.5" width={cell - gap - 1} height={cell - gap - 1} fill="none" stroke="var(--ink-3)" strokeWidth="1" />
        </pattern>
      </defs>
      <svg x={ML} y={MT} width={W} height={H} viewBox={`${win.x0 * cell} ${win.y0 * cell} ${W} ${H}`}>
        <rect x={win.x0 * cell} y={win.y0 * cell} width={W} height={H} fill="var(--page)" />
        {/* every thread, hollow; then the ones that have a pixel, filled */}
        <rect width={L.threadsX * cell} height={L.threadsY * cell} fill={`url(#${id}-h)`} />
        <rect width={L.coveredCols * cell} height={L.coveredRows * cell} fill="var(--page)" />
        <rect width={L.coveredCols * cell} height={L.coveredRows * cell} fill={`url(#${id}-f)`} data-area="working" />
        {/* pixels no thread reaches */}
        {L.unreached > 0 ? (
          <g data-area="unreached">
            {L.coveredCols < launch.width ? (
              <rect x={L.coveredCols * cell} y={0} width={(launch.width - L.coveredCols) * cell} height={launch.height * cell}
                fill="var(--fail-wash)" stroke="var(--fail)" strokeDasharray="4 3" />
            ) : null}
            {L.coveredRows < launch.height ? (
              <rect x={0} y={L.coveredRows * cell} width={L.coveredCols * cell} height={(launch.height - L.coveredRows) * cell}
                fill="var(--fail-wash)" stroke="var(--fail)" strokeDasharray="4 3" />
            ) : null}
          </g>
        ) : null}
        {hotBlocks.map((b) => {
          const i = blockInfo(launch, b.x, b.y);
          return <rect key={`${b.x},${b.y}`} x={i.x0 * cell} y={i.y0 * cell} width={i.w * cell} height={i.h * cell} fill="var(--ink)" fillOpacity="0.2" />;
        })}
        {/* block borders */}
        <g stroke="var(--ink)" strokeWidth={cell >= 6 ? 1.4 : 1}>
          {xs.map((x) => <line key={'x' + x} x1={x * cell} x2={x * cell} y1={0} y2={L.threadsY * cell} />)}
          {ys.map((y) => <line key={'y' + y} y1={y * cell} y2={y * cell} x1={0} x2={L.threadsX * cell} />)}
        </g>
        {hb ? <rect x={hb.x0 * cell} y={hb.y0 * cell} width={hb.w * cell} height={hb.h * cell} fill="none" stroke="var(--ink)" strokeWidth="3" /> : null}
      </svg>
      {inWin ? (
        <g data-sel={`${sel.x},${sel.y}`}>
          <path d={`M${ML - 4} ${sy + cell / 2} H${sx} M${sx + cell / 2} ${MT - 4} V${sy}`} stroke="var(--reg)" strokeWidth="1" fill="none" />
          <rect x={sx} y={sy} width={cell - gap} height={cell - gap} fill="var(--reg)" />
          <rect x={sx - 3} y={sy - 3} width={cell - gap + 6} height={cell - gap + 6} fill="none" stroke="var(--reg)" strokeWidth="2" />
          <text x={ML - 7} y={sy + cell / 2 + 4} fontSize="11" textAnchor="end" style={{ fill: 'var(--reg)' }}>row {sel.y}</text>
          <text x={Math.max(ML + 14, Math.min(ML + W - 14, sx + cell / 2))} y={MT - 8} fontSize="11" textAnchor="middle" style={{ fill: 'var(--reg)' }}>col {sel.x}</text>
        </g>
      ) : null}
    </svg>
  );
}

/** The same launch drawn one cell per block, for pictures too large to draw thread by thread. */
function BlockLattice({ id, launch, cell, selBlock, onPick, hot, hover, onHover, label }: {
  id: string; launch: Launch; cell: number; selBlock: XY; onPick: (b: XY) => void;
  hot?: RegionId | null; hover: XY | null; onHover: (b: XY | null) => void; label: string;
}) {
  const L = layout(launch);
  const { xs, ys } = blockBands(launch);
  const extX = Math.max(L.gridX, Math.ceil(launch.width / launch.blockW));
  const extY = Math.max(L.gridY, Math.ceil(launch.height / launch.blockH));
  const M = 6;
  const VBW = extX * cell + 2 * M, VBH = extY * cell + 2 * M;
  const gap = cell >= 8 ? 2 : 1;
  const at = (e: MouseEvent<SVGSVGElement>): XY | null => {
    const r = e.currentTarget.getBoundingClientRect();
    const k = VBW / r.width;
    const x = Math.floor(((e.clientX - r.left) * k - M) / cell);
    const y = Math.floor(((e.clientY - r.top) * k - M) / cell);
    return x >= 0 && y >= 0 && x < L.gridX && y < L.gridY ? { x, y } : null;
  };
  const key = (e: KeyboardEvent<SVGSVGElement>) => {
    const d: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    const m = d[e.key];
    if (!m) return;
    e.preventDefault();
    onPick({ x: Math.max(0, Math.min(L.gridX - 1, selBlock.x + m[0])), y: Math.max(0, Math.min(L.gridY - 1, selBlock.y + m[1])) });
  };
  const fx = cell / launch.blockW, fy = cell / launch.blockH; // pixels of drawing per thread
  return (
    <svg viewBox={`0 0 ${VBW} ${VBH}`} role="application" aria-label={label} tabIndex={0}
      data-lattice={id} data-ml={M} data-mt={M} data-cell={cell} data-x0={0} data-y0={0} data-vbw={VBW}
      style={{ width: VBW, maxWidth: '100%', height: 'auto', display: 'block', cursor: 'pointer' }}
      onClick={(e) => { const p = at(e); if (p) onPick(p); }}
      onMouseMove={(e) => { const p = at(e); if (p?.x !== hover?.x || p?.y !== hover?.y) onHover(p); }}
      onMouseLeave={() => onHover(null)}
      onKeyDown={key}>
      <defs>
        <pattern id={`${id}-f`} x={M} y={M} width={cell} height={cell} patternUnits="userSpaceOnUse">
          <rect width={cell - gap} height={cell - gap} fill="var(--ink-2)" />
        </pattern>
        <pattern id={`${id}-h`} x={M} y={M} width={cell} height={cell} patternUnits="userSpaceOnUse">
          <rect x="0.5" y="0.5" width={cell - gap - 1} height={cell - gap - 1} fill="none" stroke="var(--ink-3)" strokeWidth="1" />
        </pattern>
      </defs>
      <rect x={M} y={M} width={extX * cell} height={extY * cell} fill="var(--page)" />
      {xs.flatMap((bx) => ys.map((by) => {
        const b = blockInfo(launch, bx.first, by.first);
        const idle = b.working === 0;
        return (
          <rect key={`${bx.first},${by.first}`} data-band={b.region}
            x={M + bx.first * cell} y={M + by.first * cell} width={bx.count * cell} height={by.count * cell}
            fill={`url(#${id}-${idle ? 'h' : 'f'})`}
            opacity={idle ? 1 : hot && hot !== b.region ? 0.25 : 0.35 + 0.65 * (b.working / (launch.blockW * launch.blockH))} />
        );
      }))}
      {L.unreached > 0 ? (
        <g data-area="unreached">
          {L.coveredCols < launch.width ? (
            <rect x={M + L.coveredCols * fx} y={M} width={(launch.width - L.coveredCols) * fx} height={launch.height * fy}
              fill="var(--fail-wash)" stroke="var(--fail)" strokeDasharray="4 3" />
          ) : null}
          {L.coveredRows < launch.height ? (
            <rect x={M} y={M + L.coveredRows * fy} width={L.coveredCols * fx} height={(launch.height - L.coveredRows) * fy}
              fill="var(--fail-wash)" stroke="var(--fail)" strokeDasharray="4 3" />
          ) : null}
        </g>
      ) : null}
      {hover ? <rect x={M + hover.x * cell - 1} y={M + hover.y * cell - 1} width={cell + 1} height={cell + 1} fill="none" stroke="var(--ink)" strokeWidth="2" /> : null}
      <rect data-sel-block={`${selBlock.x},${selBlock.y}`} x={M + selBlock.x * cell - 2} y={M + selBlock.y * cell - 2} width={cell + 3} height={cell + 3}
        fill="none" stroke="var(--reg)" strokeWidth="2.5" />
    </svg>
  );
}

/** A few slots of a flat array around the ones a thread touches. */
function Strip({ name, size, hot, labels, around = 2 }: { name: ReactNode; size: number; hot: number[]; labels?: string[]; around?: number }) {
  const lo = Math.max(0, hot[0] - around);
  const hi = Math.min(size - 1, hot[hot.length - 1] + around);
  const idx = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);
  return (
    <div style={{ marginTop: '0.6rem' }}>
      <div className="mx-label t-global">{name}</div>
      <div className="mx global" style={{ gridAutoFlow: 'column', justifyContent: 'start' }}>
        {lo > 0 ? <span className="void">…</span> : null}
        {idx.map((i) => {
          const k = hot.indexOf(i);
          return (
            <span key={i} className={k >= 0 ? 'hot' : ''} style={{ padding: '0 0.3rem' }}>
              {i}
              {k >= 0 && labels ? <sub style={{ color: 'inherit' }}>{labels[k]}</sub> : null}
            </span>
          );
        })}
        {hi < size - 1 ? <span className="void">…</span> : null}
      </div>
    </div>
  );
}

const REGION_NAME: Record<RegionId, string> = {
  '1': 'inside', '2': 'right edge', '3': 'bottom edge', '4': 'corner', idle: 'outside',
};

function ThreadMap({ launch, setLaunch, sel, setSel }: {
  launch: Launch; setLaunch: (l: Launch) => void; sel: XY; setSel: (p: XY) => void;
}) {
  const [hover, setHover] = useState<XY | null>(null);
  const [hot, setHot] = useState<RegionId | null>(null);
  const L = useMemo(() => layout(launch), [launch]);
  const regs = useMemo(() => regions(launch), [launch]);
  const t = threadAt(launch, sel.x, sel.y);
  const nm = NAMES[launch.style];
  const metal = launch.style !== 'blocks';
  const extX = Math.max(L.threadsX, launch.width), extY = Math.max(L.threadsY, launch.height);
  const threadLevel = extX <= THREAD_LEVEL_MAX && extY <= THREAD_LEVEL_MAX;
  const cell = Math.max(4, Math.min(26, Math.floor(520 / extX)));
  const extBX = Math.max(L.gridX, Math.ceil(launch.width / launch.blockW));
  const blockCell = Math.max(4, Math.min(36, Math.floor(540 / extBX)));
  const preset = PRESETS.find((p) => p.height === launch.height && p.width === launch.width && p.blockW === launch.blockW && p.blockH === launch.blockH);
  const change = (patch: Partial<Launch>) => { setHover(null); setHot(null); setLaunch({ ...launch, ...patch }); };
  const shownBlock = hover ?? t.blockIdx;
  const b = blockInfo(launch, shownBlock.x, shownBlock.y);
  const forX = launch.swapped ? launch.height : launch.width;
  const forY = launch.swapped ? launch.width : launch.height;
  const firstSurplus: XY | null =
    L.threadsX > L.coveredCols ? { x: L.coveredCols, y: 0 } : L.threadsY > L.coveredRows ? { x: 0, y: L.coveredRows } : null;
  const bookThread = L.gridY > 1 ? latticeOf(launch, 0, 1, 0, 0) : null;
  const total = regs.reduce((s, r) => s + r.total, 0);

  return (
    <Bench
      title="The thread map"
      task="Click a thread, or focus the map and use the arrow keys (Shift moves a block at a time)."
      foot={
        <>
          Filled squares are threads with a pixel, hollow ones threads without. A failed guard is not an error, so
          hollow is not red; red marks only pixels that no thread reaches. <TierLegend tiers={['reg', 'global']} />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          Picture and blocks
          <select aria-label="Preset" value={preset?.id ?? 'custom'}
            onChange={(e) => {
              const p = PRESETS.find((x) => x.id === e.target.value);
              if (p) change({ height: p.height, width: p.width, blockW: p.blockW, blockH: p.blockH });
            }}>
            {PRESETS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
            {preset ? null : <option value="custom">your own sizes</option>}
          </select>
        </label>
        <NumField label="Height (rows)" value={launch.height} min={1} max={2000} onChange={(v) => change({ height: v })} />
        <NumField label="Width (columns)" value={launch.width} min={1} max={2000} onChange={(v) => change({ width: v })} />
        <label className="field">
          Block, x by y
          <select aria-label="Block size" value={`${launch.blockW}x${launch.blockH}`}
            onChange={(e) => { const [w, h] = e.target.value.split('x').map(Number); change({ blockW: w, blockH: h }); }}>
            {BLOCK_SHAPES.map(([w, h]) => <option key={`${w}x${h}`} value={`${w}x${h}`}>{w} × {h}</option>)}
          </select>
        </label>
        <label className="field">
          Launch style
          <select aria-label="Launch style" value={launch.style} onChange={(e) => change({ style: e.target.value as Style })}>
            {STYLES.map((s) => <option key={s} value={s}>{STYLE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={!!launch.swapped} onChange={(e) => change({ swapped: e.target.checked })} /> swap the grid extents (a defect)
        </label>
      </div>

      <div className="row" role="status" style={{ marginBottom: '0.8rem', gap: '0.4rem 1.4rem' }}>
        <span data-total="blocks">{nm.unit}s: <b>{L.gridX} across × {L.gridY} down = {L.blocks}</b></span>
        <span data-total="threads">threads: <b>{L.threadsX} across × {L.threadsY} down = {L.threads}</b></span>
        <span data-total="pixels">pixels: <b>{launch.width} across × {launch.height} down = {L.pixels}</b></span>
        <span data-total="surplus">threads with no pixel: <b>{L.surplus}</b></span>
        {L.unreached > 0 ? <span className="chip fail" data-total="unreached">{L.unreached} pixels never written</span> : null}
      </div>
      <p className="muted" style={{ margin: '0 0 0.8rem' }} data-k="launch">
        {launch.style === 'threads' ? (
          <>
            The grid is given in threads: ({forX}, {forY}). Metal cuts it into groups of {launch.blockW} × {launch.blockH}{' '}
            and makes the groups on the far edges smaller.
          </>
        ) : (
          <>
            x: {forX} / {launch.blockW} rounded up = {L.gridX} {nm.unit}s, sized from the {launch.swapped ? 'height' : 'width'}.{' '}
            y: {forY} / {launch.blockH} rounded up = {L.gridY}, sized from the {launch.swapped ? 'width' : 'height'}.
          </>
        )}
        {launch.swapped ? <> <b className="t-fail">The two are the wrong way round.</b></> : null}
        {launch.style === 'threads' && L.surplus > 0 ? (
          <> <b className="t-fail">With no guard, the {L.surplus} threads outside the picture would write to the wrong place or past the end of the array.</b></>
        ) : null}
      </p>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 11fr) minmax(0, 9fr)' }}>
        <div>
          {threadLevel ? (
            <Lattice id="map" launch={launch} win={{ x0: 0, y0: 0, cols: extX, rows: extY }} cell={cell} sel={sel} onPick={setSel}
              hot={hot} hover={hover} onHover={setHover}
              label={`Threads of the launch: ${L.threadsX} across and ${L.threadsY} down. Arrow keys move the selected thread.`} />
          ) : (
            <>
              <p className="muted" style={{ margin: '0 0 0.5rem' }}>
                Too many threads to draw one by one: each square here is a whole {nm.unit}. Pick a {nm.unit}, then a
                thread inside it.
              </p>
              <BlockLattice id="bmap" launch={launch} cell={blockCell} selBlock={t.blockIdx} hot={hot} hover={hover} onHover={setHover}
                onPick={(bk) => setSel(clampSel(launch, latticeOf(launch, bk.x, bk.y, t.threadIdx.x, t.threadIdx.y)))}
                label={`${nm.Unit}s of the launch: ${L.gridX} across and ${L.gridY} down. Arrow keys move the selected ${nm.unit}.`} />
              <div className="mx-label" style={{ marginTop: '0.8rem' }}>
                {metal ? `Threadgroup at (${t.blockIdx.x}, ${t.blockIdx.y})` : `Block ${bookLabel(t.blockIdx)}`}, thread by thread
              </div>
              <Lattice id="inset" launch={launch} sel={sel} onPick={setSel}
                cell={Math.max(5, Math.min(18, Math.floor(288 / launch.blockW), Math.floor(320 / launch.blockH)))}
                win={{ x0: t.blockIdx.x * launch.blockW, y0: t.blockIdx.y * launch.blockH, cols: launch.blockW, rows: launch.blockH }}
                label={`Threads of the selected ${nm.unit}. Arrow keys move the selected thread.`} />
            </>
          )}
          <div className="row" style={{ marginTop: '0.7rem', gap: '0.5rem' }}>
            <span className="muted">Go to</span>
            <button data-jump="book" disabled={!bookThread} onClick={() => bookThread && setSel(bookThread)}>Thread (0,0) of Block (1,0)</button>
            <button data-jump="surplus" disabled={!firstSurplus} onClick={() => firstSurplus && setSel(firstSurplus)}>a thread with no pixel</button>
            <button data-jump="last" onClick={() => setSel({ x: L.threadsX - 1, y: L.threadsY - 1 })}>the last thread</button>
          </div>
        </div>

        <div>
          <div className="panel" data-panel="thread" aria-live="polite">
            <div className="mx-label" data-k="title">
              {metal
                ? <>Thread at grid position (x {t.col}, y {t.row})</>
                : <>Thread {bookLabel(t.threadIdx)} of Block {bookLabel(t.blockIdx)}</>}
            </div>
            <div className="muted" style={{ marginBottom: '0.5rem' }}>
              {metal
                ? <>Metal writes positions x first. The book would call this Thread {bookLabel(t.threadIdx)} of Block {bookLabel(t.blockIdx)}.</>
                : <>The book's labels put y first. The fields below are in x, y order.</>}
            </div>
            <div className="mono" style={mono} data-k="blockIdx">{nm.block} = (x <Reg>{t.blockIdx.x}</Reg>, y <Reg>{t.blockIdx.y}</Reg>)</div>
            <div className="mono" style={mono} data-k="threadIdx">{nm.thread} = (x <Reg>{t.threadIdx.x}</Reg>, y <Reg>{t.threadIdx.y}</Reg>)</div>
            <div className="mono" style={mono} data-k="blockDim">
              {launch.style === 'threads' ? 'dispatch size' : nm.dim} = (x {t.blockDim.x}, y {t.blockDim.y})
              {launch.style === 'threads' && (t.groupSize.x !== t.blockDim.x || t.groupSize.y !== t.blockDim.y)
                ? <>; this edge group is ({t.groupSize.x}, {t.groupSize.y})</> : null}
            </div>
            <hr style={{ border: 0, borderTop: '1px solid var(--rule)', margin: '0.5rem 0' }} />
            {metal ? (
              <>
                <div className="mono" style={mono} data-k="gid">thread_position_in_grid = (<Reg>{t.col}</Reg>, <Reg>{t.row}</Reg>), handed to the kernel</div>
                <div className="mono" style={mono} data-k="row">row = gid.y = <Reg>{t.row}</Reg>{launch.style === 'threadgroups' ? <> (= {t.blockIdx.y}*{t.blockDim.y} + {t.threadIdx.y})</> : null}</div>
                <div className="mono" style={mono} data-k="col">col = gid.x = <Reg>{t.col}</Reg>{launch.style === 'threadgroups' ? <> (= {t.blockIdx.x}*{t.blockDim.x} + {t.threadIdx.x})</> : null}</div>
              </>
            ) : (
              <>
                <div className="mono" style={mono} data-k="row">row = blockIdx.y*blockDim.y + threadIdx.y = {t.blockIdx.y}*{t.blockDim.y} + {t.threadIdx.y} = <Reg>{t.row}</Reg></div>
                <div className="mono" style={mono} data-k="col">col = blockIdx.x*blockDim.x + threadIdx.x = {t.blockIdx.x}*{t.blockDim.x} + {t.threadIdx.x} = <Reg>{t.col}</Reg></div>
              </>
            )}
            <div className="mono" style={mono} data-k="guard">
              {'col < width && row < height: '}
              {t.col} {'<'} {launch.width} is {String(t.colOk)}
              {t.failing === 'col' ? <>, so <b>the thread stops</b> (the row test is not evaluated)</> : (
                <>; {t.row} {'<'} {launch.height} is {String(t.rowOk)}{t.works ? <>, so <b>the thread works</b></> : <>, so <b>the thread stops</b></>}</>
              )}
            </div>
            {launch.style === 'threads' && L.surplus === 0 ? (
              <div className="muted" data-k="noguard">No thread of this launch can fail the test, so the kernel may leave it out.</div>
            ) : null}
            {t.works ? (
              <>
                <div className="mono" style={mono} data-k="gray">grayOffset = row*width + col = {t.row}*{launch.width} + {t.col} = <Glob>{t.grayOffset}</Glob></div>
                <div className="mono" style={mono} data-k="rgb">rgbOffset = grayOffset*{CHANNELS} = <Glob>{t.rgbOffset}</Glob></div>
                <Strip name="Pout, one byte per pixel" size={L.pixels} hot={[t.grayOffset!]} />
                <Strip name="Pin, three bytes per pixel" size={L.pixels * CHANNELS} hot={[t.rgbOffset!, t.rgbOffset! + 1, t.rgbOffset! + 2]} labels={['r', 'g', 'b']} around={1} />
              </>
            ) : (
              <div className="muted" data-k="nomem" style={{ marginTop: '0.4rem' }}>No offsets: this thread never reaches the lines that compute them, and touches no memory.</div>
            )}
          </div>

          <div className="panel" data-panel="block" style={{ marginTop: '0.8rem' }}>
            <div data-k="block">
              <b>{metal ? `${nm.Unit} (x ${b.bx}, y ${b.by})` : `Block ${bookLabel({ x: b.bx, y: b.by })}`}</b>
              {hover ? ' (under the pointer)' : " (the selected thread's)"}: <b>{b.working} of {b.launched}</b> threads work
              {b.working ? <>, {b.workW} across × {b.workH} down</> : null}.
            </div>
            <table className="threads" aria-label="Kinds of block" style={{ marginTop: '0.5rem' }}>
              <thead>
                <tr><th>kind</th><th>{nm.unit}s</th><th>working, each</th><th>product</th></tr>
              </thead>
              <tbody onMouseLeave={() => setHot(null)}>
                {regs.map((r) => (
                  <tr key={r.region} data-region={r.region} className={b.region === r.region ? 'cur' : ''} onMouseEnter={() => setHot(r.region)}>
                    <td>{r.region === 'idle' ? '' : r.region + ': '}{REGION_NAME[r.region]}</td>
                    <td>{r.blocks}</td>
                    <td>{r.working}{r.launched !== r.working ? ` of ${r.launched}` : ''}</td>
                    <td>{r.total}</td>
                  </tr>
                ))}
                <tr data-region="sum">
                  <td colSpan={3}>working threads in all</td>
                  <td><b>{total}</b></td>
                </tr>
              </tbody>
            </table>
            <div className="muted" data-k="sum" style={{ marginTop: '0.4rem' }}>
              {total === L.pixels
                ? <>{total} = {launch.height} × {launch.width}: every pixel has exactly one thread.</>
                : <span className="t-fail">{total} is short of {launch.height} × {launch.width} = {L.pixels} by {L.unreached} pixels.</span>}
              {regs.some((r) => r.region === 'idle') ? <> Blocks of the last kind do no work at all.</> : null}
            </div>
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the kernel stepper

const REG_ORDER = ['col', 'row', 'grayOffset', 'rgbOffset', 'r', 'g', 'b', 'sum', 'byte'];
const KERNEL_FIRST = 3, KERNEL_LAST = 21;

function KernelStepper({ launch, sel, setSel }: { launch: Launch; sel: XY; setSel: (p: XY) => void }) {
  const [n, setN] = useState(0);
  const steps = useMemo(() => stepsGrayscale(launch, sel.x, sel.y), [launch, sel.x, sel.y]);
  const L = layout(launch);
  const t = threadAt(launch, sel.x, sel.y);
  const at = Math.min(n, steps.length);
  const cur = at > 0 ? steps[at - 1] : undefined;
  const regs = cur?.regs ?? {};
  const read = new Map(steps.slice(0, at).filter((s) => s.mem && !s.mem.write).map((s) => [s.mem!.index, s.mem!.value]));
  const stored = steps.slice(0, at).find((s) => s.mem?.write)?.mem;
  const pick = (p: XY | null) => { if (p) { setSel(p); setN(0); } };
  const bookThread = L.gridY > 1 ? latticeOf(launch, 0, 1, 0, 0) : null;
  const right: XY | null = L.threadsX > L.coveredCols ? { x: L.coveredCols, y: Math.min(5, L.threadsY - 1) } : null;
  const below: XY | null = L.threadsY > L.coveredRows ? { x: Math.min(3, L.threadsX - 1), y: L.coveredRows } : null;
  const body = GRAY_CU.split('\n').slice(KERNEL_FIRST - 1, KERNEL_LAST).join('\n');
  return (
    <Bench
      title="Step this thread through the kernel"
      task="Step the book's thread to its store. Then step a thread with no pixel and see where it stops."
      foot={
        <>
          The thread is the one selected in the thread map; changing it here changes it there. Pixel values come from a
          small synthetic picture. <TierLegend tiers={['reg', 'global']} />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', gap: '0.5rem' }}>
        <span data-k="who">
          <b>Thread {bookLabel(t.threadIdx)} of Block {bookLabel(t.blockIdx)}</b>, picture {launch.height} × {launch.width}, blocks {launch.blockW} × {launch.blockH}.
        </span>
        <span className="muted">Use</span>
        <button data-pick="book" disabled={!bookThread} onClick={() => pick(bookThread)}>Thread (0,0) of Block (1,0)</button>
        <button data-pick="right" disabled={!right} onClick={() => pick(right)}>one past the right edge</button>
        <button data-pick="below" disabled={!below} onClick={() => pick(below)}>one below the bottom edge</button>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setN(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={at >= steps.length}>Step</button>
        <button onClick={() => setN(steps.length)} disabled={at >= steps.length}>Run to the end</button>
        <span className="muted" aria-live="polite">step {at} of {steps.length}</span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? cur.text : <>Nothing has run yet. Every thread of the grid starts at the first line of the kernel with its own <code>blockIdx</code> and <code>threadIdx</code>. Press Step.</>}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 5fr)' }}>
        <div>
          <Code tabs={[{ label: 'kernel', code: body, tiers: TIERS }]} firstLine={KERNEL_FIRST} highlight={cur ? [LINE[cur.op]] : []} />
        </div>
        <div>
          <div className="mx-label t-reg">Registers of this thread</div>
          <table className="threads" aria-label="Registers of this thread">
            <tbody>
              {REG_ORDER.map((name) => (
                <tr key={name} data-reg={name} className={cur?.set === name ? 'cur' : ''}>
                  <td>{name === 'sum' ? 'the float sum' : name === 'byte' ? 'after the cast' : name}</td>
                  <td className="t-reg">{regs[name] === undefined ? '·' : name === 'sum' ? fmtFloat(regs[name]) : regs[name]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {t.works ? (
            <div data-k="mem">
              <div className="mx-label t-global" style={{ marginTop: '0.8rem' }}>Pin (global memory): the three bytes of this pixel</div>
              <div className="mx global" style={{ gridTemplateColumns: 'repeat(3, auto)' }}>
                {[0, 1, 2].map((k) => {
                  const i = t.rgbOffset! + k;
                  return <small key={'i' + k} className="muted" style={{ textAlign: 'center', padding: '0 0.3rem' }}>[{i}]</small>;
                })}
                {[0, 1, 2].map((k) => {
                  const i = t.rgbOffset! + k;
                  const hotNow = cur?.mem && !cur.mem.write && cur.mem.index === i;
                  return <span key={k} data-pin={i} className={hotNow ? 'hot' : read.has(i) ? 'on' : ''}>{pixelValue(steps, i)}</span>;
                })}
              </div>
              <div className="mx-label t-global" style={{ marginTop: '0.8rem' }}>Pout (global memory): this thread's one element</div>
              <div className="mx global" style={{ gridTemplateColumns: 'auto' }}>
                <small className="muted" style={{ textAlign: 'center' }}>[{t.grayOffset}]</small>
                <span data-pout={t.grayOffset} className={cur?.mem?.write ? 'hot' : stored ? 'on' : 'void'}>{stored ? stored.value : ''}</span>
              </div>
            </div>
          ) : (
            <p className="muted" data-k="mem" style={{ margin: '0.8rem 0 0' }}>
              This thread reads and writes nothing. Its block still had to be launched whole.
            </p>
          )}
        </div>
      </div>
    </Bench>
  );
}

/** The byte the synthetic picture holds at a Pin index, taken from the trace so the component computes nothing. */
function pixelValue(steps: ReturnType<typeof stepsGrayscale>, index: number): number | string {
  return steps.find((s) => s.mem && !s.mem.write && s.mem.index === index)?.mem!.value ?? '';
}

// ---------------------------------------------------------------- the flattening explorer

const KIND_LABEL: Record<Indexing, string> = {
  row: 'row-major',
  col: 'column-major',
  wrong: 'wrong stride (a defect)',
};

function FlatteningExplorer() {
  const [height, setHeight] = useState(3);
  const [width, setWidth] = useState(5);
  const [depth, setDepth] = useState(1);
  const [kind, setKind] = useState<Indexing>('row');
  const [showStrides, setShowStrides] = useState(false);
  const [pick, setPick] = useState({ plane: 0, row: 1, col: 2 });
  const d: Dims = { depth: kind === 'col' ? 1 : depth, height, width };
  const s = { plane: Math.min(pick.plane, d.depth - 1), row: Math.min(pick.row, height - 1), col: Math.min(pick.col, width - 1) };
  const off = offsetOf(d, kind, s.plane, s.row, s.col);
  const cov = useMemo(() => indexingCoverage({ depth: kind === 'col' ? 1 : depth, height, width }, kind), [depth, height, width, kind]);
  const st = stridesFor(d, kind);
  const three = d.depth > 1;
  const lab = (e: { plane: number; row: number; col: number }) => (three ? `${e.plane},${e.row},${e.col}` : `${e.row},${e.col}`);
  // Where one step along each dimension lands, for the stride markers.
  const next: { slot: number; text: string }[] = [];
  if (showStrides && kind !== 'wrong') {
    if (s.col + 1 < width) next.push({ slot: off + st[2], text: 'col+1' });
    if (s.row + 1 < height) next.push({ slot: off + st[1], text: 'row+1' });
    if (s.plane + 1 < d.depth) next.push({ slot: off + st[0], text: 'plane+1' });
  }
  const firstAt = (slot: number) => {
    for (let plane = 0; plane < d.depth; plane++)
      for (let row = 0; row < height; row++)
        for (let col = 0; col < width; col++) if (offsetOf(d, kind, plane, row, col) === slot) return { plane, row, col };
    return null;
  };
  const pickSlot = (slot: number) => {
    const e = kind === 'wrong' ? firstAt(slot) : elementAt(d, kind, slot);
    if (e) setPick(e);
  };
  const formula =
    kind === 'row'
      ? three
        ? `plane*height*width + row*width + col = ${s.plane}*${height}*${width} + ${s.row}*${width} + ${s.col}`
        : `row*width + col = ${s.row}*${width} + ${s.col}`
      : kind === 'col'
        ? `col*height + row = ${s.col}*${height} + ${s.row}`
        : three
          ? `plane*height*width + row*height + col = ${s.plane}*${height}*${width} + ${s.row}*${height} + ${s.col}`
          : `row*height + col = ${s.row}*${height} + ${s.col}`;
  const shape = three ? [d.depth, height, width] : [height, width];
  const strideTuple = three ? st : st.slice(1);
  const names = three ? ['plane', 'row', 'col'] : ['row', 'col'];
  const idxs = three ? [s.plane, s.row, s.col] : [s.row, s.col];
  const slotBox: CSSProperties = { display: 'grid', justifyItems: 'center', gap: 1 };

  return (
    <Bench
      title="Flatten an array"
      task="Click an element to see its slot, or a slot to get the indices back."
      foot="Slots are numbered from 0. Each element is one slot wide here; in memory, multiply the slot by the element's size in bytes."
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          Height (rows): {height}
          <input type="range" min="1" max="6" value={height} aria-label="Height" onChange={(e) => setHeight(+e.target.value)} />
        </label>
        <label className="field">
          Width (columns): {width}
          <input type="range" min="1" max="8" value={width} aria-label="Width" onChange={(e) => setWidth(+e.target.value)} />
        </label>
        <label className="field">
          Depth (planes): {d.depth}
          <input type="range" min="1" max="3" value={d.depth} aria-label="Depth" disabled={kind === 'col'} onChange={(e) => setDepth(+e.target.value)} />
        </label>
        <span className="seg" role="group" aria-label="Indexing">
          {(['row', 'col', 'wrong'] as Indexing[]).map((k) => (
            <button key={k} aria-pressed={kind === k} onClick={() => setKind(k)}>{KIND_LABEL[k]}</button>
          ))}
        </span>
        <label className="check">
          <input type="checkbox" checked={showStrides} onChange={(e) => setShowStrides(e.target.checked)} /> show the strides
        </label>
      </div>

      <div className="row" style={{ alignItems: 'flex-start', gap: '1.4rem', marginBottom: '0.9rem' }}>
        {Array.from({ length: d.depth }, (_, plane) => (
          <div key={plane}>
            <div className="mx-label">{three ? `plane ${plane}` : `M, ${height} × ${width}`}</div>
            <div className="mx global" data-grid={plane} style={{ gridTemplateColumns: `repeat(${width}, auto)` }}>
              {Array.from({ length: height * width }, (_, i) => {
                const e = { plane, row: Math.floor(i / width), col: i % width };
                const on = e.plane === s.plane && e.row === s.row && e.col === s.col;
                return (
                  <span key={i} role="button" tabIndex={0} aria-pressed={on} data-el={`${e.plane},${e.row},${e.col}`} className={on ? 'hot' : ''}
                    style={{ cursor: 'pointer' }} onClick={() => setPick(e)} onKeyDown={onKey(() => setPick(e))}>
                    {lab(e)}
                  </span>
                );
              })}
            </div>
          </div>
        ))}
        {kind === 'col' ? (
          <div>
            <div className="mx-label">the transpose of M, {width} × {height}</div>
            <div className="mx global" data-grid="transpose" style={{ gridTemplateColumns: `repeat(${height}, auto)` }}>
              {Array.from({ length: height * width }, (_, i) => {
                // row i of the transpose is column i of M
                const e = { plane: 0, row: i % height, col: Math.floor(i / height) };
                const on = e.row === s.row && e.col === s.col;
                return <span key={i} className={on ? 'hot' : ''}>{lab(e)}</span>;
              })}
            </div>
            <div className="muted" style={{ maxWidth: '16rem', marginTop: '0.3rem' }}>Read the transpose row by row: it is the strip below.</div>
          </div>
        ) : null}
      </div>

      <div className="narrate" role="status" style={{ marginBottom: '0.9rem', minHeight: 0 }}>
        <span className="mono" data-k="formula">
          element [{lab(s)}] is at slot {formula} = <Glob>{off}</Glob>
        </span>
        {kind === 'wrong' ? <> <span className="t-fail">The stride should be the width, {width}, not the height, {height}.</span></> : null}
        {showStrides ? (
          <div className="mono" data-k="strides" style={{ marginTop: '0.3rem' }}>
            {kind === 'wrong' ? 'the strides this defect uses' : kind === 'col' ? 'strides' : 'x.stride()'} for shape ({shape.join(', ')}) = ({strideTuple.join(', ')});
            slot = {idxs.map((v, i) => `${v}*${strideTuple[i]}`).join(' + ')} = {off}.{' '}
            <span className="muted">
              One step along {names.join(', ')} moves {strideTuple.join(', ')} slots.
              {kind === 'col' ? ' Not in decreasing order, so PyTorch would not call a tensor with these strides contiguous.' : ''}
            </span>
          </div>
        ) : null}
      </div>

      <div className="mx-label t-global">the flat array, {cov.size} slots</div>
      <div data-strip style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 2px', alignItems: 'flex-start' }}>
        {cov.hits.map((h, i) => {
          const e = kind === 'wrong' ? null : elementAt(d, kind, i);
          const nx = next.find((x) => x.slot === i);
          const here = i === off;
          // With the wrong stride a slot's look says how often it is written, so the selection is an outline.
          const cls = kind === 'wrong' ? (h === 0 ? 'void' : h > 1 ? 'bad' : '') : here ? 'hot' : nx ? 'on' : '';
          return (
            <div key={i} style={slotBox}>
              <small className="muted" style={{ fontSize: '0.64rem' }}>{i}</small>
              <div className="mx global">
                <span role="button" tabIndex={0} data-slot={i} data-hits={h} data-here={here ? '' : undefined} className={cls}
                  style={{ cursor: 'pointer', boxShadow: here && kind === 'wrong' ? '0 0 0 2px var(--ink)' : undefined }}
                  onClick={() => pickSlot(i)} onKeyDown={onKey(() => pickSlot(i))}>
                  {e ? lab(e) : h === 0 ? '' : `×${h}`}
                </span>
              </div>
              <small style={{ fontSize: '0.62rem', minHeight: '0.9rem' }}>{nx ? nx.text : ''}</small>
            </div>
          );
        })}
        {cov.outOfBounds.map((o) => (
          <div key={`o${o.plane},${o.row},${o.col}`} style={slotBox}>
            <small className="t-fail" style={{ fontSize: '0.64rem' }}>{o.offset}</small>
            <div className="mx global">
              <span data-oob={o.offset} className="bad" style={{ borderStyle: 'dashed' }}>{lab(o)}</span>
            </div>
            <small style={{ fontSize: '0.62rem', minHeight: '0.9rem' }} />
          </div>
        ))}
      </div>
      {kind === 'wrong' ? (
        <p data-k="coverage" style={{ margin: '0.8rem 0 0' }}>
          {cov.collisions.length + cov.holes.length + cov.outOfBounds.length === 0 ? (
            <span className="chip ok">every slot written exactly once</span>
          ) : (
            <>
              {cov.collisions.length ? <><span className="chip fail">{cov.collisions.length} slots written twice or more</span>{' '}</> : null}
              {cov.holes.length ? <><span className="chip fail">{cov.holes.length} slots never written</span>{' '}</> : null}
              {cov.outOfBounds.length ? <span className="chip fail">{cov.outOfBounds.length} writes past the end of the array</span> : null}
            </>
          )}{' '}
          {height === width
            ? 'Height equals width, so the wrong stride is the right number and the defect cannot be seen.'
            : 'Each slot shows how many elements land on it; dashed red slots do not exist.'}
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the erratum hunt

type Finding = 'cast' | 'constants' | 'args' | 'spelling';
export interface Spot {
  id: string;
  /** 'k' for the kernel listing, 'h' for the host call. */
  where: 'k' | 'h';
  /** 1-based line in that listing. */
  line: number;
  /** The exact text of the spot on that line. */
  text: string;
  finding?: Finding;
  /** Why a spot that is not a defect is right. */
  fine?: string;
}

export const SPOTS: Spot[] = [
  { id: 'name', where: 'k', line: 4, text: 'colortoGrayscaleConvertion', finding: 'spelling' },
  { id: 'params', where: 'k', line: 4, text: 'unsigned char * Pout', finding: 'args' },
  { id: 'col', where: 'k', line: 6, text: 'blockIdx.x*blockDim.x + threadIdx.x', fine: 'Columns run along x, so col comes from the x fields.' },
  { id: 'row', where: 'k', line: 7, text: 'blockIdx.y*blockDim.y + threadIdx.y', fine: 'Rows run along y, so row comes from the y fields.' },
  { id: 'guard', where: 'k', line: 8, text: 'col < width && row < height', fine: 'Both tests are needed and both comparisons are strict: valid columns are 0 to width - 1.' },
  { id: 'gray', where: 'k', line: 10, text: 'row*width + col', fine: 'Each row above contributes width elements. This is row-major order.' },
  { id: 'rgb', where: 'k', line: 13, text: 'grayOffset*CHANNELS', fine: 'Every pixel before this one took CHANNELS bytes.' },
  { id: 'blue', where: 'k', line: 16, text: 'rgbOffset + 2', fine: 'Blue is the third of the three bytes.' },
  { id: 'cast', where: 'k', line: 19, text: '(unsigned char)', finding: 'cast' },
  { id: 'consts', where: 'k', line: 19, text: '0.21f*r+0.71f*g+0.07f*b', finding: 'constants' },
  { id: 'grid', where: 'h', line: 1, text: 'ceil(m/16.0), ceil(n/16.0)', fine: 'x is sized from the width m and y from the height n, as it must be.' },
  { id: 'block', where: 'h', line: 2, text: '16, 16, 1', fine: '16 × 16 × 1 is 256 threads, a legal block.' },
  { id: 'hname', where: 'h', line: 3, text: 'colorToGrayscaleConversion', finding: 'spelling' },
  { id: 'args', where: 'h', line: 3, text: 'Pin_d, Pout_d, m, n', finding: 'args' },
];

const FINDINGS: { id: Finding; title: string; body: ReactNode; scored: boolean }[] = [
  {
    id: 'cast', scored: true, title: 'The cast covers one constant, not the sum (line 19)',
    body: (
      <>
        A cast applies to the expression right after it, and <Cw>*</Cw> binds less tightly. The line parses as{' '}
        <Cw>((unsigned char)0.21f)*r + 0.71f*g + 0.07f*b</Cw>. <Cw>(unsigned char)0.21f</Cw> is 0, so red is
        multiplied by 0 and what is stored is the truncated value of <Cw>0.71f*g + 0.07f*b</Cw>. It compiles
        without a warning.
      </>
    ),
  },
  {
    id: 'constants', scored: true, title: 'The constants are not the ones in the text (line 19)',
    body: (
      <>
        The book's text on p. 52 gives 0.299, 0.587 and 0.114. The listing has 0.21, 0.71 and 0.07, which add up to
        0.99, so even with the cast fixed a white pixel comes out as 252.
      </>
    ),
  },
  {
    id: 'args', scored: true, title: 'The call passes input and output the wrong way round',
    body: (
      <>
        The kernel is declared <Cw>(Pout, Pin, width, height)</Cw>; the call on p. 50 passes{' '}
        <Cw>(Pin_d, Pout_d, m, n)</Cw>. As printed, the kernel would read from the output buffer and write into the
        picture it was meant to read.
      </>
    ),
  },
  {
    id: 'spelling', scored: false, title: 'The name is spelt two ways',
    body: (
      <>
        <Cw>colortoGrayscaleConvertion</Cw> in the listing, <Cw>colorToGrayscaleConversion</Cw> in the call, the
        text and the caption. This is the only one of the four a compiler would refuse.
      </>
    ),
  },
];

function Listing({ title, lines, where, spots, marked, toggle, shown, firstLine = 1 }: {
  title: string; lines: string[]; where: 'k' | 'h'; spots: Spot[]; marked: Set<string>; toggle: (id: string) => void;
  /** After a check: marked spots are coloured right or wrong; with 'all', missed defects are outlined too. */
  shown: 'none' | 'marks' | 'all'; firstLine?: number;
}) {
  return (
    <div style={{ marginBottom: '0.9rem' }}>
      <div className="mx-label">{title}</div>
      <div className="mono" data-listing={where}
        style={{ background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 4, padding: '0.6rem 0.8rem 0.6rem 0', overflowX: 'auto', fontSize: '0.8rem', lineHeight: 1.75, whiteSpace: 'pre' }}>
        {lines.map((text, i) => {
          const n = i + 1;
          const here = spots.filter((s) => s.where === where && s.line === n).sort((a, b) => text.indexOf(a.text) - text.indexOf(b.text));
          const out: ReactNode[] = [];
          let pos = 0;
          for (const s of here) {
            const at = text.indexOf(s.text, pos);
            if (at < 0) continue;
            out.push(text.slice(pos, at));
            const on = marked.has(s.id);
            const state = shown === 'none' ? (on ? 'marked' : 'idle') : on ? (s.finding ? 'right' : 'wrong') : shown === 'all' && s.finding ? 'missed' : 'idle';
            out.push(
              <span key={s.id} role="button" tabIndex={0} aria-pressed={on} data-spot={s.id} data-state={state}
                onClick={() => toggle(s.id)} onKeyDown={onKey(() => toggle(s.id))}
                style={{
                  cursor: 'pointer', borderRadius: 3, padding: '0.1em 0.15em',
                  borderBottom: state === 'idle' ? '1px dashed var(--ink-3)' : undefined,
                  outline: state === 'marked' ? '2px solid var(--ink)' : state === 'missed' ? '2px dashed var(--fail)' : state === 'right' ? '2px solid var(--ok)' : state === 'wrong' ? '2px solid var(--fail)' : undefined,
                  background: state === 'marked' ? 'var(--rule)' : state === 'right' ? 'var(--ok-wash)' : state === 'wrong' ? 'var(--fail-wash)' : undefined,
                }}>
                {s.text}
              </span>,
            );
            pos = at + s.text.length;
          }
          out.push(text.slice(pos));
          return (
            <div key={i}>
              <span className="muted" style={{ display: 'inline-block', width: '2.6rem', paddingRight: '0.9rem', textAlign: 'right', userSelect: 'none' }}>
                {String(n + firstLine - 1).padStart(2, '0')}
              </span>
              {out}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ErratumHunt() {
  const [marked, setMarked] = useState<Set<string>>(new Set());
  const [checked, setChecked] = useState(false);
  const [answers, setAnswers] = useState(false);
  const toggle = (id: string) => {
    const m = new Set(marked);
    if (m.has(id)) m.delete(id);
    else m.add(id);
    setMarked(m);
    setChecked(false);
  };
  const hit = (f: Finding) => SPOTS.some((s) => s.finding === f && marked.has(s.id));
  const scored = FINDINGS.filter((f) => f.scored);
  const found = scored.filter((f) => hit(f.id));
  const alarms = SPOTS.filter((s) => !s.finding && marked.has(s.id));
  const shown = answers ? 'all' : checked ? 'marks' : 'none';
  return (
    <Bench
      title="Find what is wrong in Fig. 3.4"
      task="Mark every underlined spot you believe is wrong, in the kernel and in the call, then check."
      foot="The code lines are as printed on pp. 50 and 53; the comments are reworded. Fourteen spots can be marked; most of them are right."
    >
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)' }}>
        <div>
          <Listing title="The kernel, as printed" lines={PRINTED.split('\n')} where="k" spots={SPOTS} marked={marked} toggle={toggle} shown={shown} />
          <Listing title="The launch, as printed (n rows, m columns)" lines={PRINTED_HOST} where="h" spots={SPOTS} marked={marked} toggle={toggle} shown={shown} />
        </div>
        <div>
          <div className="row" style={{ marginBottom: '0.7rem' }}>
            <button className="primary" disabled={marked.size === 0} onClick={() => setChecked(true)}>Check my marks</button>
            <button onClick={() => { setMarked(new Set()); setChecked(false); setAnswers(false); }} disabled={marked.size === 0 && !answers}>Clear</button>
            <button onClick={() => { setChecked(true); setAnswers(true); }} disabled={answers}>Show the answers</button>
            <span className="muted">{marked.size} marked</span>
          </div>
          {checked ? (
            <div className="panel" role="status" data-k="hunt" data-found={found.length}>
              <p style={{ margin: '0 0 0.5rem' }}>
                <b>{found.length} of 3 defects found.</b>{' '}
                {found.length < 3 && !answers ? 'Keep looking, or show the answers.' : ''}
              </p>
              {FINDINGS.filter((f) => answers || hit(f.id)).map((f) => (
                <p key={f.id} data-finding={f.id} style={{ margin: '0 0 0.6rem' }}>
                  <span className={'chip ' + (hit(f.id) ? 'ok' : 'fail')}>{hit(f.id) ? 'found' : 'missed'}</span>{' '}
                  <b>{f.title}.</b>{f.scored ? '' : ' (Not scored.)'} {f.body}
                </p>
              ))}
              {alarms.map((s) => (
                <p key={s.id} data-alarm={s.id} style={{ margin: '0 0 0.6rem' }}>
                  <span className="chip fail">not a defect</span> <Cw>{s.text}</Cw>: {s.fine}
                </p>
              ))}
            </div>
          ) : (
            <p className="muted" style={{ margin: 0 }}>
              Three defects change what the program does. One more is a slip in a name. Reading the lines against each
              other is enough; you do not need to run anything.
            </p>
          )}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the pixel tester

const VARIANTS: { id: GrayVariant; name: string; expr: (w: [number, number, number]) => string }[] = [
  { id: 'printed', name: 'Line 19 as printed', expr: (w) => `(unsigned char) ${w[0]}f*r + ${w[1]}f*g + ${w[2]}f*b` },
  { id: 'printedParen', name: 'Printed constants, sum in parentheses', expr: (w) => `(unsigned char)(${w[0]}f*r + ${w[1]}f*g + ${w[2]}f*b)` },
  { id: 'corrected', name: 'Corrected line', expr: (w) => `(unsigned char)(${w[0]}f*r + ${w[1]}f*g + ${w[2]}f*b)` },
];
const COLOURS: { label: string; rgb: [number, number, number] }[] = [
  { label: '(200, 100, 50)', rgb: [200, 100, 50] },
  { label: 'pure red', rgb: [255, 0, 0] },
  { label: 'white', rgb: [255, 255, 255] },
  { label: 'pure green', rgb: [0, 255, 0] },
  { label: 'pure blue', rgb: [0, 0, 255] },
];
const swatch = (css: string): CSSProperties => ({
  display: 'inline-block', width: '1.6rem', height: '1.1rem', border: '1px solid var(--rule)', borderRadius: 2, background: css, verticalAlign: 'middle',
});

function PixelTester() {
  const [rgb, setRgb] = useState<[number, number, number]>([200, 100, 50]);
  const set = (k: number, v: number) => setRgb(rgb.map((x, i) => (i === k ? v : x)) as [number, number, number]);
  const res = VARIANTS.map((v) => ({ ...v, d: grayDetail(rgb[0], rgb[1], rgb[2], v.id) }));
  const right = res[2].d.byte;
  return (
    <Bench
      title="What each version stores"
      task="Try pure red, then white. Which version gets which one wrong, and why?"
      foot="Sums are computed in single precision, one rounded operation at a time, left to right. The swatches show the input colour and each stored gray."
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        {(['r', 'g', 'b'] as const).map((ch, k) => (
          <label key={ch} className="field">
            {ch} = {rgb[k]}
            <input type="range" min="0" max="255" value={rgb[k]} aria-label={ch} onChange={(e) => set(k, +e.target.value)} />
          </label>
        ))}
        <span className="seg" role="group" aria-label="Colours">
          {COLOURS.map((c) => (
            <button key={c.label} aria-pressed={c.rgb.every((v, i) => v === rgb[i])} onClick={() => setRgb(c.rgb)}>{c.label}</button>
          ))}
        </span>
        <span>input <i style={swatch(`rgb(${rgb.join(',')})`)} data-swatch="in" /></span>
      </div>
      <table className="data" style={{ margin: 0, background: 'var(--page)' }} aria-label="Stored byte by version">
        <thead>
          <tr><th style={{ paddingLeft: '0.6rem' }}>Version</th><th>What C++ evaluates</th><th className="num">Float sum</th><th className="num">Byte stored</th><th /></tr>
        </thead>
        <tbody>
          {res.map((v) => (
            <tr key={v.id} data-variant={v.id}>
              <td style={{ paddingLeft: '0.6rem' }}>{v.name}</td>
              <td className="mono" style={{ overflowWrap: 'anywhere' }}>
                {v.id === 'printed'
                  ? <>0*{rgb[0]} + {GRAY_WEIGHTS[v.id][1]}f*{rgb[1]} + {GRAY_WEIGHTS[v.id][2]}f*{rgb[2]}</>
                  : <>{GRAY_WEIGHTS[v.id][0]}f*{rgb[0]} + {GRAY_WEIGHTS[v.id][1]}f*{rgb[1]} + {GRAY_WEIGHTS[v.id][2]}f*{rgb[2]}</>}
                <div className="muted">{v.expr(GRAY_WEIGHTS[v.id])}</div>
              </td>
              <td className="num">{fmtFloat(v.d.sum)}</td>
              <td className="num" data-byte={v.d.byte}>
                <b className={v.d.byte === right ? '' : 't-fail'}>{v.d.byte}</b>
              </td>
              <td><i style={swatch(`rgb(${v.d.byte},${v.d.byte},${v.d.byte})`)} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

const L0 = layout(BOOK);
const SWAPPED = layout({ ...BOOK, swapped: true });
const BOOK_ONE = latticeOf(BOOK, 0, 1, 0, 0);
const T0 = threadAt(BOOK, BOOK_ONE.x, BOOK_ONE.y);

export const questionsS32: Record<string, Question> = {
  'ch03/3.2/q-swapped': {
    kind: 'numeric',
    id: 'ch03/3.2/q-swapped',
    prompt: (
      <>
        The picture has n = 62 rows and m = 76 columns. Someone writes the launch as{' '}
        <code>dim3 dimGrid(ceil(n/16.0), ceil(m/16.0), 1)</code> with 16 × 16 blocks and leaves the kernel unchanged.
        How many pixels are never written?
      </>
    ),
    answer: 744,
    unit: 'pixels',
    explain: (
      <>
        <p>
          The grid's x extent now comes from the height: ceil(62/16) = 4 blocks, 64 thread columns. Its y extent comes
          from the width: 5 blocks, 80 thread rows. All 62 rows are reached, but columns 64 to 75 have no thread:{' '}
          {BOOK.width - SWAPPED.coveredCols} columns × {BOOK.height} rows = {SWAPPED.unreached} pixels keep whatever the
          buffer held.
        </p>
        <p>
          The launch also creates a row of 4 blocks, thread rows 64 to 79, in which no thread works. And on a square
          picture the two launches are identical, so a test on a square input cannot catch this.
        </p>
      </>
    ),
    source: 'ch. 3 §3.2, pp. 49–50',
  },
  'ch03/3.2/q-ex3d': {
    kind: 'numeric',
    id: 'ch03/3.2/q-ex3d',
    prompt: (
      <>
        Book exercise 3. A kernel computes <code>row</code> and <code>col</code> as above and guards its one statement
        with <code>if (row &lt; M &amp;&amp; col &lt; N)</code>. The host sets M = 150, N = 300,{' '}
        <code>dim3 bd(16, 32)</code> and <code>dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1)</code>. How many threads execute
        the guarded statement?
      </>
    ),
    answer: 45000,
    unit: 'threads',
    explain: (
      <p>
        The grid is (19, 5): 95 blocks of 16 × 32 = 512 threads, 48,640 in all, spanning 304 columns and 160 rows. Only
        threads with row &lt; 150 and col &lt; 300 pass: 150 × 300 = 45,000, one per element. The other 3,640 stop at
        the guard. Choose the "exercise 3" preset in the thread map to see the same numbers.
      </p>
    ),
    source: 'ch. 3 exercise 3, p. 65',
  },
  'ch03/3.2/q-ex4a': {
    kind: 'numeric',
    id: 'ch03/3.2/q-ex4a',
    prompt: (
      <>
        Book exercise 4. A matrix 400 wide and 500 high is stored in a flat array in <b>row-major</b> order. What is the
        index of the element at row 20, column 10?
      </>
    ),
    answer: 8010,
    explain: <p>Twenty full rows of 400 come first: 20·400 + 10 = 8010. The height does not appear.</p>,
    source: 'ch. 3 exercise 4, p. 65; §3.2, p. 51',
  },
  'ch03/3.2/q-ex4b': {
    kind: 'numeric',
    id: 'ch03/3.2/q-ex4b',
    prompt: (
      <>
        The same matrix (400 wide, 500 high), now stored in <b>column-major</b> order. What is the index of the element
        at row 20, column 10?
      </>
    ),
    answer: 5020,
    explain: (
      <p>
        Ten full columns of 500 come first: 10·500 + 20 = 5020. Now the width does not appear. This is the macro cuBLAS
        gives C callers, <code>IDX2C(i,j,ld) = j*ld + i</code>, with the leading dimension <code>ld</code> equal to the
        number of rows.
      </p>
    ),
    source: 'ch. 3 exercise 4, p. 65; §3.2, pp. 51–52; cuBLAS data layout (notes/today/ch03-facts.md, A7)',
  },
  'ch03/3.2/q-ex5-stride': {
    kind: 'numeric',
    id: 'ch03/3.2/q-ex5-stride',
    prompt: (
      <>
        A contiguous tensor <code>x</code> has shape (300, 500, 400): depth, height, width. <code>x.stride()</code> is
        (200000, 400, 1). What is the flat offset of <code>x[5, 20, 10]</code>?
      </>
    ),
    answer: 1008010,
    explain: (
      <p>
        Index times stride, summed: 5·200,000 + 20·400 + 10·1 = 1,008,010. It is the book's{' '}
        <code>plane*m*n + row*m + col</code> with m = 400 columns and n = 500 rows, and it is book exercise 5 (x = 10,
        y = 20, z = 5) in PyTorch's words.
      </p>
    ),
    source: 'ch. 3 §3.2, p. 55; exercise 5, p. 65; torch.Tensor.stride (notes/today/ch03-facts.md, D6)',
  },
  'ch03/3.2/q-precedence': {
    kind: 'choice',
    id: 'ch03/3.2/q-precedence',
    prompt: (
      <>
        Find the defect. This line compiles without a warning, with <code>r</code>, <code>g</code>, <code>b</code> of
        type <code>unsigned char</code>. What does it store?
        <br />
        <code>Pout[grayOffset] = (unsigned char) 0.21f*r + 0.71f*g + 0.07f*b;</code>
      </>
    ),
    options: [
      { label: 'The intended luminance, 0.21·r + 0.71·g + 0.07·b, truncated.', why: 'That needs the whole sum inside the parentheses of the cast.' },
      { label: 'The truncated value of 0.71·g + 0.07·b. Red contributes nothing.', correct: true },
      { label: 'Always 0.', why: 'Only the first product is 0. The other two are computed in float and survive.' },
      { label: 'A value that can exceed 255 and wrap around.', why: 'The two remaining weights sum to 0.78, so the value stays below 199.' },
    ],
    explain: (
      <p>
        The cast binds to <code>0.21f</code> alone and turns it into the integer 0, so the first term is 0·r. The sum of
        the other two is a float, and the assignment to an <code>unsigned char</code> truncates it. Pure red is stored
        as 0 where 53 was intended; (200, 100, 50) is stored as 74 instead of 116.
      </p>
    ),
    source: 'ch. 3 Fig. 3.4 line 19, p. 53; C++ grammar and a host compile (notes/today/ch03-facts.md, A9)',
  },
  'ch03/3.2/q-uchar3': {
    kind: 'choice',
    id: 'ch03/3.2/q-uchar3',
    prompt: (
      <>
        A Metal port binds the colour picture, three bytes per pixel, as <code>device const uchar3* Pin</code> and reads
        a pixel as <code>Pin[grayOffset]</code>. What happens?
      </>
    ),
    options: [
      { label: 'It works: one uchar3 is one three-byte pixel.', why: 'In Metal Shading Language uchar3 has size 4 and alignment 4.' },
      { label: 'Each element is 4 bytes, so from the second pixel on it reads the wrong bytes.', correct: true },
      { label: 'It does not compile: a buffer cannot hold a vector type.', why: 'Buffers may hold vector types. The trouble is their size.' },
      { label: 'It reads the right bytes in the order b, g, r.', why: 'Nothing is reordered. The step between elements is wrong.' },
    ],
    explain: (
      <p>
        <code>Pin[i]</code> on a <code>uchar3*</code> is at byte 4·i, and the data has a pixel every 3 bytes. Pixel 0 is
        right, pixel 1 starts one byte late, and it gets worse from there. Keep the book's arithmetic on a{' '}
        <code>uchar*</code> (<code>rgbOffset = grayOffset*3</code>), or declare the buffer{' '}
        <code>packed_uchar3</code>, which is 3 bytes with alignment 1.
      </p>
    ),
    source: 'ours; Metal Shading Language specification, tables 2.3 and 2.4 (notes/today/ch03-facts.md, C5)',
  },
  'ch03/3.2/q-noguard': {
    kind: 'choice',
    id: 'ch03/3.2/q-noguard',
    prompt: (
      <>
        The same Metal kernel body, using <code>thread_position_in_grid</code>, is launched over the 62 × 76 picture
        twice: once with <code>dispatchThreads</code> over a grid of (76, 62), and once with{' '}
        <code>dispatchThreadgroups</code> over (5, 4) groups of 16 × 16. Which launch needs the range test?
      </>
    ),
    options: [
      { label: 'Only dispatchThreads.', why: 'That call creates exactly the threads you ask for.' },
      { label: 'Only dispatchThreadgroups.', correct: true },
      { label: 'Both.', why: 'One of them creates exactly 76 × 62 threads.' },
      { label: 'Neither: Metal discards writes outside a buffer.', why: 'Nothing in the pages we checked says so. Apple tells you to exit early when the grid is larger than the data.' },
    ],
    explain: (
      <p>
        <code>dispatchThreadgroups</code> creates 5 × 4 whole groups: {L0.threads} threads for {L0.pixels} pixels, so{' '}
        {L0.surplus} threads are outside the picture and must be stopped. <code>dispatchThreads</code> over (76, 62)
        creates exactly {L0.pixels} threads and makes the groups on the edges smaller. Apple's documentation says no
        position check is needed with it. Note the order: Metal sizes are width first.
      </p>
    ),
    source: 'ours; Apple, "Calculating threadgroup and grid sizes" (notes/today/ch03-facts.md, C3)',
  },
};

// ---------------------------------------------------------------- the section

export function S32() {
  const [launch, setLaunch] = useState<Launch>(BOOK);
  const [selRaw, setSel] = useState<XY>({ x: 37, y: 21 });
  const sel = clampSel(launch, selRaw);
  const { progress } = useProgress();
  const predictedRed = typeof progress.answers['ch03/3.2/p-red']?.value === 'number';
  const showBookThread = () => {
    setLaunch(BOOK);
    setSel(BOOK_ONE);
    document.querySelector('section.bench[aria-label="The thread map"]')?.scrollIntoView({ block: 'start' });
  };

  return (
    <>
      <p>
        Section 3.1 gave every thread its coordinates: its block's index and its own index within the block, each along
        x and y. This section turns them into the two things a kernel needs, which element of the data the thread owns
        and where that element is in memory. The first takes two multiply-adds, the second one more. Nearly every
        kernel you will write for inference opens with these lines.
      </p>
      <Source section="3.2" pages="48–55" />

      <h2>One thread per pixel</h2>
      <p>
        The book's example is a picture of 62 rows and 76 columns, converted from colour to grey by one thread per
        pixel. The book writes sizes with the vertical dimension first, so "62 × 76" is 62 down and 76 across, the order
        of the subscripts in <code>P[y][x]</code>. Launch sizes go the other way, x first. The book itself calls this a
        source of confusion, and one of the questions below is built on it.
      </p>
      <Source section="3.2" pages="48–49" figs="Fig. 3.2 and footnote 1" />
      <p>
        Blocks are 16 × 16 threads. Covering 76 columns takes 5 blocks along x and covering 62 rows takes 4 along y: 20
        blocks. Before you watch that happen, commit to two numbers.
      </p>
      <Predict
        id="ch03/3.2/p-surplus"
        prompt={<>The launch creates 20 blocks of 256 threads for a picture of 62 × 76 pixels. How many threads have no pixel?</>}
        unit="threads"
      >
        {(v) => (
          <>
            <p>
              {L0.threads} threads for {L0.pixels} pixels leaves <b>{L0.surplus}</b>.{' '}
              {v === L0.surplus ? 'That is what you predicted.' : `You were off by ${Math.abs(v - L0.surplus)}.`}
            </p>
            <p>
              They are not scattered: every one of them is in the last {L0.threadsX - BOOK.width} thread columns or the
              last {L0.threadsY - BOOK.height} thread rows.
            </p>
          </>
        )}
      </Predict>
      <Predict
        id="ch03/3.2/p-offsets"
        prompt={
          <>
            The grey output is one flat array, one byte per pixel, with the picture's rows stored one after another.
            Section 3.1's labels are (y, x). Which element of that array does Thread (0,0) of Block (1,0) write?
          </>
        }
      >
        {(v) => (
          <>
            <p>
              Block (1,0) is one block down and none across, so the thread is in row {T0.blockIdx.y}·{BOOK.blockH} + 0 ={' '}
              {T0.row}, column {T0.col}. {T0.row} full rows of {BOOK.width} come before it: element{' '}
              <b>{T0.grayOffset}</b>. {v === T0.grayOffset ? 'That is what you predicted.' : `You said ${v}.`}
            </p>
            <p>
              Now work out, in your head, where that pixel's red byte is in the colour input, which holds three bytes
              per pixel. Part 5 of the animation gets there, and the thread map shows both numbers.
            </p>
            <p>
              <button onClick={showBookThread}>Select this thread in the map</button>
            </p>
          </>
        )}
      </Predict>
      <p>
        The animation lays the book's launch over the picture block by block, follows one thread to its pixel and into
        the flat output, and ends by sorting the blocks into kinds.
      </p>
      <Simplification>
        <p>
          The animation shows ownership only: which thread gets which pixel, and where that pixel is stored. Blocks do
          not really arrive one at a time in order, and nothing in it is about speed.
        </p>
      </Simplification>
      <GridOverPictureScene />

      <h2>The mapping every later kernel reuses</h2>
      <p>
        A thread's row and column in the data are its position in the whole grid of threads:{' '}
        <code>row = blockIdx.y*blockDim.y + threadIdx.y</code> and <code>col = blockIdx.x*blockDim.x + threadIdx.x</code>.
        Read each as: the blocks before mine, times the threads in a block, plus my place in my own block. The y fields
        give the row and the x fields give the column. For Thread (0,0) of Block (1,0) that is row 1·16 + 0 = 16 and
        column 0·16 + 0 = 0, so it handles pixel (16, 0).
      </p>
      <Source section="3.2" pages="49–50" />
      <p>
        The map below is that launch. Each small square is a thread and the heavy lines are block borders. Select a
        thread and the panel substitutes its numbers into the formulas, down to the bytes it would touch. Find where
        each number in a formula comes from. Then find which blocks contain threads with no pixel, and how many kinds
        of block there are.
      </p>
      <Simplification>
        <p>
          The map shows which thread owns which pixel and nothing else: no timing, no warps, no cost for a thread that
          does nothing. The smaller edge groups of the Metal <code>dispatchThreads</code> style follow Apple's
          description and were not measured on a device.
        </p>
      </Simplification>
      <ThreadMap launch={launch} setLaunch={setLaunch} sel={sel} setSel={setSel} />

      <h2>Surplus threads and the guard</h2>
      <p>
        A grid is a whole number of blocks and every block has the same shape, so the launch rounds up: 5 × 16 = 80
        thread columns for 76 pixel columns, 4 × 16 = 64 thread rows for 62 pixel rows. The {L0.surplus} threads
        outside the picture must do nothing, so the kernel tests both indices before it touches memory:{' '}
        <code>{'if (col < width && row < height)'}</code>. It is the two-dimensional form of the test in chapter 2's
        vector addition, where four blocks of 256 threads left 24 with no element of a 1000-element vector.
      </p>
      <Source section="3.2" pages="50" />
      <p>
        The same test completes the argument that every pixel is handled. Along x there are{' '}
        <code>blockDim.x*gridDim.x</code> threads, and <code>col</code> takes every integer from 0 up to one less than
        that. If that count is at least the width, and the same holds along y, every pixel has a thread. The guard
        removes the threads beyond the picture, and no two threads share a pixel because no two share a (row, col).
      </p>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A5)">
        <p>
          NVIDIA's programming guide documents the same pattern: round the grid up and let out-of-range threads exit. It
          adds a rule the book does not state: idle threads inside a block cost little, but blocks in which no thread
          works should be avoided. A grid rounded up from the right dimensions never contains one.
        </p>
      </Today>
      <p>
        The launch has to size the grid's x extent from the width and its y extent from the height. With n rows and m
        columns the book writes <code>dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1)</code>. Nothing warns you if the two
        are the wrong way round. Tick "swap the grid extents" in the map and look at what changes, then answer.
      </p>
      <Source section="3.2" pages="50, 52–53" />
      <Ask q={questionsS32['ch03/3.2/q-swapped']} label="Work it out" />
      <Ask q={questionsS32['ch03/3.2/q-ex3d']} label="Book exercise 3d" />

      <h2>Why flatten, and what row-major means</h2>
      <p>
        You would like to write <code>Pin_d[row][col]</code>. C++ can index a two-dimensional array that way only when
        it knows the number of columns at compile time, and for a buffer allocated at run time it does not. So the
        buffer is one-dimensional and you compute the offset. Nothing is lost by this. Memory is a flat sequence of
        addressed bytes, so every multidimensional array is flattened somewhere: for a static array the compiler does
        it, for a dynamic one you do.
      </p>
      <Source section="3.2" pages="50–51" figs='sidebar "Memory Space"' />
      <p>
        C and C++ flatten in <b>row-major</b> order: all of row 0, then all of row 1, and so on. Element (j, i) of an
        array <code>Width</code> columns wide is at <code>j*Width + i</code>, where the first term skips the j rows
        above. In a 4 × 4 matrix, M[2][1] is element 2·4 + 1 = 9. The other convention, <b>column-major</b>, stores
        whole columns one after another, and FORTRAN uses it. Column-major storage of a matrix is the row-major storage
        of its transpose, which is why libraries written for FORTRAN callers tell C callers to transpose their inputs.
      </p>
      <Source section="3.2" pages="51–52" figs="Fig. 3.3" />
      <p>
        Check three things in the explorer. A step along one dimension is always the same jump in the flat array,
        wherever you start. Column-major really is the transpose read row by row. And the classic defect, using the
        height where the width belongs, cannot be seen on a square array.
      </p>
      <Simplification>
        <p>
          At most 144 elements, each drawn as one slot. Real offsets are in elements too; the compiler multiplies by the
          element size when it turns <code>P[i]</code> into an address.
        </p>
      </Simplification>
      <FlatteningExplorer />
      <p>
        The book says programmers "need to" linearise by hand in current CUDA C++. That sentence has aged. The habit it
        describes has not: hand-written offsets are what you will find in every backend you read.
      </p>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A6)">
        <p>
          nvcc now accepts C++ up to C++23, and NVIDIA's libcu++ has <code>cuda::std::mdspan</code>, with which a
          kernel can write <code>m(row, col)</code> over a flat buffer (<code>operator()</code>, because{' '}
          <code>operator[]</code> is disabled). Unverified: which toolkit release first shipped it. NVIDIA's own guide,
          llama.cpp, MLX and PyTorch all still flatten by hand, and so do the labs here.
        </p>
      </Today>
      <Ask q={questionsS32['ch03/3.2/q-ex4a']} label="Book exercise 4a" />
      <Ask q={questionsS32['ch03/3.2/q-ex4b']} label="Book exercise 4b" />
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A7, B6, C7)">
        <p>
          The column-major warning applies to the vendors' own BLAS. cuBLAS and rocBLAS store matrices column-major,
          and cuBLAS's documentation gives C callers the macro <code>IDX2C(i,j,ld) = j*ld + i</code>. On Apple's side{' '}
          <code>MPSMatrix</code> is row-major, while the shading language's small <code>floatNxM</code> types are
          column-major.
        </p>
      </Today>

      <h2>What stride() has been telling you</h2>
      <p>
        You have used this layout for years. A PyTorch tensor is a flat buffer plus a tuple of strides, and{' '}
        <code>x.stride()</code> is the jump in that buffer for one step along each dimension. A contiguous tensor of
        shape (2, 5) has stride (5, 1): one step down is 5 elements and one step across is 1, which is{' '}
        <code>row*Width + col</code> with a width of 5. PyTorch's name for the layout, <code>contiguous_format</code>,
        is defined by strides in decreasing order. Tick "show the strides" in the explorer above and you are looking at
        that tuple.
      </p>
      <Beyond title="Bridge: PyTorch">
        <p>
          Ours, not the book's. From the PyTorch 2.15 documentation of <code>Tensor.stride</code> and{' '}
          <code>torch.memory_format</code>, read on 2026-11-20 (notes/today/ch03-facts.md, D6).
        </p>
      </Beyond>

      <h2>The grayscale kernel</h2>
      <p>
        Now the whole kernel. A colour pixel is three consecutive bytes, red, green and blue, and a grey pixel is one
        byte. The thread computes its grey offset as above, then multiplies by three to find its pixel in the colour
        input, because every pixel before it took three bytes: <code>rgbOffset = grayOffset*CHANNELS</code>. It reads
        the three bytes, forms a weighted sum and stores one byte. The book's text gives the weights: L = 0.299·r +
        0.587·g + 0.114·b.
      </p>
      <Source section="3.2" pages="52–54" figs="Fig. 3.4" />
      <Code
        tabs={[
          { label: 'CUDA and HIP', code: GRAY_CU, tiers: TIERS, status: 'front-end-checked; emulator-checked (python3 labs/ch03/snippets/check.py); unverified on CUDA or HIP hardware' },
          { label: 'Metal kernel', code: GRAY_METAL, tiers: TIERS, status: 'unverified: compiles with xcrun metal, not run' },
        ]}
        caption={
          <>
            <code>labs/ch03/snippets/grayscale_kernel.cu</code> and <code>.metal</code>. This is a corrected kernel: it
            differs from the page in the cast, the constants, the argument order of the call and one spelling, which
            you will find for yourself below. The launch also rounds up with integers where the book writes{' '}
            <code>ceil(m/16.0)</code>; NVIDIA's guide uses the integer form. Lab 3.1 has you write and run this kernel,
            renamed <code>colorToGrayscaleKernel</code> and taking the input first, as its launch does.
          </>
        }
      />
      <p>
        The book works the numbers for Thread (0,0) of Block (1,0): it writes <code>Pout[16*76 + 0] = Pout[1216]</code>{' '}
        and reads the three bytes that start at <code>Pin[16*76*3] = Pin[3648]</code>. You predicted the first of these
        before the animation. Step that thread through the kernel line by line, then step a thread with no pixel.
      </p>
      <Source section="3.2" pages="53–54" />
      <p>
        Read the kernel by colour as you step. Violet values live in the thread's own registers and cost nothing to
        reuse. Amber is global memory: this kernel touches it exactly four times per pixel, three reads and one write,
        and everything else is arithmetic on registers.
      </p>
      <Simplification>
        <p>
          One thread at a time, one source line per step, on a synthetic picture. Real threads run in their thousands
          at once. The float sum is computed one rounded single-precision operation at a time, left to right; a GPU
          compiler may fuse or reorder those operations.
        </p>
      </Simplification>
      <KernelStepper key={`${launch.height},${launch.width},${launch.blockW},${launch.blockH},${launch.style},${!!launch.swapped}`}
        launch={launch} sel={sel} setSel={setSel} />

      <h2>Find the errors in the printed listing</h2>
      <p>
        Fig. 3.4 as printed is not the kernel above. One of the differences is a defect that compiles cleanly and gives
        a wrong picture. Below are the figure's code lines and the launch from p. 50, as printed. Mark every spot you
        believe is wrong, then check. There are three defects to find and one slip that is only a spelling.
      </p>
      <Source section="3.2" pages="50, 53" figs="Fig. 3.4" />
      <ErratumHunt />
      <p>
        The cast is the one to remember, because C++ gives you no help with it. A cast applies to the single expression
        that follows it, and multiplication binds less tightly, so the printed line means{' '}
        <code>((unsigned char)0.21f)*r + 0.71f*g + 0.07f*b</code>. Before you open the tester, commit to what that does
        to a red pixel.
      </p>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A9, C6)">
        <p>
          Checked against the C++ grammar and by compiling both forms on a host with all warnings on: no diagnostic.
          Metal Shading Language uses the C++ grammar, so the printed line is the same defect there.
        </p>
      </Today>
      <Predict
        id="ch03/3.2/p-red"
        prompt={<>A pure red pixel is (r, g, b) = (255, 0, 0). What byte does line 19 as printed store for it?</>}
      >
        {(v) => (
          <p>
            It stores <b>{grayDetail(255, 0, 0, 'printed').byte}</b>
            {v === grayDetail(255, 0, 0, 'printed').byte ? ', as you predicted' : `, not ${v}`}. With the sum in
            parentheses the printed constants would give {grayDetail(255, 0, 0, 'printedParen').byte}, and the corrected
            line gives {grayDetail(255, 0, 0, 'corrected').byte}. A red picture comes out black. The tester below is
            now open.
          </p>
        )}
      </Predict>
      <Simplification>
        <p>
          The tester evaluates each version on the CPU in single precision, one rounded operation at a time, left to
          right. It reproduces the host-compiled values recorded in the facts file. A GPU compiler that fuses or
          reorders the operations can differ by one grey level.
        </p>
      </Simplification>
      {predictedRed ? (
        <PixelTester />
      ) : (
        <p className="muted" data-k="tester-closed">The pixel tester opens once you have committed a prediction above.</p>
      )}
      <p>
        The corrected line is{' '}
        <code>{'Pout[grayOffset] = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b);'}</code>. It differs from the page
        in the parentheses and in the constants, which are the ones the book's own text states. They sum to 1, so white
        stays 255. The conversion to a byte drops the fraction, so the stored value can change by one grey level with
        the order in which the three products are added, which is why lab 3.1 compares against its CPU reference with a
        tolerance of 1.
      </p>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A8, A9)">
        <p>
          0.299, 0.587, 0.114 are the luma weights of ITU-R BT.601. The listing's 0.21, 0.71, 0.07 match neither
          BT.601 nor BT.709 (0.2126, 0.7152, 0.0722). In a host test, reordering the additions changed the byte for 40
          of 140,608 sampled colours, by one level each.
        </p>
      </Today>
      <Ask q={questionsS32['ch03/3.2/q-precedence']} label="Find the defect" />

      <h2>Four kinds of block</h2>
      <p>
        Go back to the map, choose the book's picture, and move the pointer over the blocks. The guard sorts the 20
        blocks into four kinds. Twelve lie wholly inside the picture and all 256 of their threads work. The three on the
        right edge lose their last 4 columns, leaving 12 × 16 = 192. The four along the bottom lose their last 2 rows,
        leaving 16 × 14 = 224. The corner block loses both, leaving 12 × 14 = 168. And 12·256 + 3·192 + 4·224 + 168 ={' '}
        {L0.pixels}, the number of pixels: the coverage argument again, as arithmetic.
      </p>
      <Source section="3.2" pages="54–55" figs="Fig. 3.5" />
      <p>
        Chapter 4 comes back to exactly these four regions when it counts the warps whose threads disagree at the
        guard.
      </p>

      <h2>Three dimensions</h2>
      <p>
        A third dimension adds one more line of each kind. The thread gets a plane index,{' '}
        <code>plane = blockIdx.z*blockDim.z + threadIdx.z</code>, and planes are placed one after another in memory.
        With n rows and m columns in a plane, the element is <code>P[plane*m*n + row*m + col]</code>, and the guard
        tests all three indices. Rename the dimensions and this is the tensor you index every day: [batch, seq, hidden]
        is plane, row, col. Set the depth above 1 in the flattening explorer to watch the planes line up.
      </p>
      <Source section="3.2" pages="55" />
      <Ask q={questionsS32['ch03/3.2/q-ex5-stride']} label="Book exercise 5, as PyTorch states it" />
      <p>
        On HIP nothing in this section changes. The <code>__global__</code> qualifier, the four built-in variables, the
        launch statement and the index arithmetic are the same source text, as long as you use the{' '}
        <code>.x</code>, <code>.y</code>, <code>.z</code> members and do not copy a whole built-in into a{' '}
        <code>uint3</code>.
      </p>
      <Beyond title="Not in the book: HIP">
        <p>
          Checked against AMD's HIP documentation and headers on 2026-11-20 (notes/today/ch03-facts.md, B2, B5); not
          compiled on AMD hardware, and the remark about <code>uint3</code> is an inference from AMD's header. One
          thing on AMD's language page should not be copied: its 3D "global unique
          identifier" formula uses the block's width and area as strides, so it is unique only when the grid is one
          block wide. The stride for data is the data's width, as above.
        </p>
      </Beyond>

      <h2>The same mapping on Metal</h2>
      <p>
        Metal hands a kernel its coordinates as attributed parameters instead of global variables, and one of them is
        already the answer. <code>thread_position_in_grid</code> is the thread's position in the whole grid:{' '}
        <code>.x</code> is the column, <code>.y</code> the row, and the origin is the top left. Apple's documentation
        prints the CUDA formula as its definition: threadgroup position times threads per threadgroup, plus position in
        the threadgroup. Change the map's launch style to see every name change and the numbers stay put.
      </p>
      <Beyond title="Not in the book: Metal">
        <p>
          The book uses CUDA only. This part is ours, checked on 2026-11-20 against Apple's documentation, the Metal
          Shading Language specification and the three projects' sources (notes/today/ch03-facts.md, C1, C3, C5, C6,
          D3, D4, D5). The Metal kernel shown above compiles here and has not been run; lab 3.1 runs one.
        </p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>In the kernel</th><th>CUDA and HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr><td>Block's position</td><td><code>blockIdx</code></td><td><code>[[threadgroup_position_in_grid]]</code></td></tr>
          <tr><td>Thread's position in its block</td><td><code>threadIdx</code></td><td><code>[[thread_position_in_threadgroup]]</code></td></tr>
          <tr><td>Block shape</td><td><code>blockDim</code></td><td><code>[[threads_per_threadgroup]]</code></td></tr>
          <tr><td>Grid shape in blocks</td><td><code>gridDim</code></td><td><code>[[threadgroups_per_grid]]</code></td></tr>
          <tr><td>Row and column in the data</td><td>computed from the three above</td><td><code>[[thread_position_in_grid]]</code>, given</td></tr>
          <tr><td>Byte picture</td><td><code>unsigned char*</code></td><td><code>device uchar*</code> with <code>[[buffer(n)]]</code></td></tr>
          <tr><td>Width and height</td><td>passed by value in the launch</td><td><code>constant T&amp;</code>, filled by the host with <code>setBytes</code></td></tr>
          <tr><td>Float literal</td><td><code>0.299f</code></td><td><code>0.299f</code>; there is no <code>double</code></td></tr>
        </tbody>
      </table>
      <p>
        The launch is where Metal offers a choice the book's CUDA does not. <code>dispatchThreadgroups</code> takes a
        number of threadgroups, like a CUDA grid: you round up and you guard. <code>dispatchThreads</code> takes the
        size of the grid in threads. Metal then makes the threadgroups along the edges smaller, creates exactly the
        threads you asked for, and Apple's documentation says the position check is not needed. The four kinds of block
        exist only under the first call. Both are in real use; the list at the end of this page says where.
      </p>
      <Ask q={questionsS32['ch03/3.2/q-noguard']} />
      <p>
        Inside the kernel, three things need care. A byte picture is a plain buffer, <code>device uchar*</code>, indexed
        exactly as in CUDA, but do not reach for <code>uchar3</code> to read a pixel: in Metal Shading Language it
        occupies 4 bytes. Keep the <code>rgbOffset</code> arithmetic on a <code>uchar</code> pointer, or use{' '}
        <code>packed_uchar3</code>, which is 3. There is no <code>double</code>, so write <code>0.299f</code> with its
        suffix. And small scalars such as the width and height arrive as <code>constant T&amp;</code> references that
        the host fills with <code>setBytes</code>.
      </p>
      <Ask q={questionsS32['ch03/3.2/q-uchar3']} />
      <p>
        You now have the mapping, the guard and the flat offset. <a href={href('ch03', '3.3')}>Section 3.3</a> keeps all
        three and gives each thread a loop. On the <a href={href('ch03', 'labs')}>labs page</a>, lab 3.1 has you write
        this kernel and run it on your own GPU, and lab 3.5 hides four index defects for you to find.
      </p>

      <h2>In real backends</h2>
      <p>
        Where the choices of this page turn up in code you will read later. None of it is needed for the labs.
      </p>
      <ul data-list="backends">
        <li>
          <b>Elementwise operations.</b> PyTorch's CUDA kernels under <code>a + b</code> or <code>torch.relu</code>{' '}
          launch a one-dimensional grid whatever the tensor's shape, and each thread computes flat indices only.
          NVIDIA's guide: "The use of multi-dimensional thread blocks and grids is for convenience only and does not
          affect performance." A 2D grid hands you <code>row</code> and <code>col</code> without a division; when the
          operation does not care which row an element is in, flatten first.
        </li>
        <li>
          <b>The two Metal launches.</b> llama.cpp's Metal backend and PyTorch's MPS matmul call{' '}
          <code>dispatchThreadgroups</code>, so they round up and guard. MLX calls <code>dispatchThreads</code> and
          flattens the position itself as <code>index.x + grid_dim.x * index.y</code>, which is{' '}
          <code>row*width + col</code> with the grid's own width as the stride.
        </li>
      </ul>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A2, C3, D3, D4, D5, D6)">
        <p>
          Read in the projects' sources on the check date. The quotation is from NVIDIA's programming guide.
        </p>
      </Today>
    </>
  );
}
