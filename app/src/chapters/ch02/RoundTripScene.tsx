import { type ReactNode, type SVGProps } from 'react';
import { useLanguage } from '../../lib/language';
import { Scene, type Beat } from '../../components/Scene';
import { useNarrow } from './narrow';
import {
  FLOAT_BYTES,
  N,
  expected,
  inputs,
  lanes,
  program,
  run,
  showCell,
  type ArrayId,
  type Cell,
  type LaneStep,
  type Program,
  type State,
} from '../../sim/offload';

// "Three parts, two memories": the host stub of Fig. 2.5 as two lanes (the host
// thread and the device) over two memories, then the same steps with one shared
// memory. Every cell, byte count and lane segment is read from sim/offload.ts;
// six beats, one narration script each (content/ch02/narration/round-trip-N.txt).
//
// The model has no durations. Slot widths and how far the kernel has got when the
// host reaches the copy back are the drawing's choice, and the lesson says so.
//
// Two layouts: the two memories side by side, and on a narrow screen one above the
// other, so that the numbers stay legible on a phone. Both draw the same state.
export const SCENE_CUDA = program('cuda');
export const SCENE_METAL = program('metal', 'buffers');
const WANT = expected();
const IN = inputs();

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const MONO = { fontVariationSettings: "'MONO' 1, 'CASL' 0" };

const CW = 28; // cell pitch
const CH = 26;
const CELLS_DX = 50; // from a row's left edge to its first cell
const BOX_H = 152;
const ROWS_DY = [30, 64, 98]; // from a box's top to each of its three rows
const IDS: ArrayId[] = ['A', 'B', 'C'];
// The order in which the drawing lets the kernel's results appear: not left to
// right, because nothing promises an order among threads.
const FILL_ORDER = [3, 0, 6, 1, 7, 4, 2, 5];

interface Geo {
  narrow: boolean;
  W: number;
  H: number;
  /** Characters per line of a heading or footnote before it wraps. */
  wrap: number;
  headY: number;
  host: { x: number; y: number; w: number };
  dev: { x: number; y: number; w: number };
  /** The boundary between the two memories, and where the byte counter sits. */
  boundary: [number, number, number, number];
  bytes: { x: number; y: number };
  lane: { x: number; w: number; host: number; device: number; h: number; labelSize: number };
  /** Where the line (or, narrow, the three lines) of host and device state starts. */
  statusY: number;
}

const WIDE: Geo = {
  narrow: false, W: 760, H: 364, wrap: 999, headY: 22,
  host: { x: 16, y: 36, w: 300 },
  dev: { x: 444, y: 36, w: 300 },
  boundary: [380, 42, 380, 182],
  bytes: { x: 380, y: 206 },
  lane: { x: 110, w: 634, host: 228, device: 256, h: 20, labelSize: 10 },
  statusY: 314,
};
// Narrow, two memories: host memory on top, the boundary, device memory below.
// The lanes take the full width and are named above, not beside.
const NARROW: Geo = {
  narrow: true, W: 340, H: 614, wrap: 46, headY: 18,
  host: { x: 6, y: 46, w: 328 },
  dev: { x: 6, y: 240, w: 328 },
  boundary: [6, 214, 334, 214],
  bytes: { x: 170, y: 231 },
  lane: { x: 6, w: 328, host: 422, device: 462, h: 20, labelSize: 9.5 },
  statusY: 528,
};
// Narrow, one memory: the lanes move up to sit under the single box.
const NARROW_ONE: Geo = {
  ...NARROW, H: 494,
  bytes: { x: 170, y: 232 },
  lane: { ...NARROW.lane, host: 300, device: 340 },
  statusY: 406,
};


/** Splits a sentence into lines of at most `max` characters, at spaces. */
function wrapText(s: string, max: number): string[] {
  if (s.length <= max) return [s];
  const out: string[] = [];
  let line = '';
  for (const w of s.split(' ')) {
    if (line && line.length + 1 + w.length > max) {
      out.push(line);
      line = w;
    } else line = line ? line + ' ' + w : w;
  }
  if (line) out.push(line);
  return out;
}

/** One <text> whose lines are tspans; its textContent is the whole sentence. */
function Lines({ x, y, lines, dy = 16, ...rest }: { x: number; y: number; lines: string[]; dy?: number } & Omit<SVGProps<SVGTextElement>, 'x' | 'y' | 'dy'>) {
  return (
    <text x={x} y={y} {...rest}>
      {lines.map((l, i) => (
        <tspan key={i} x={x} dy={i === 0 ? 0 : dy}>{i < lines.length - 1 ? l + ' ' : l}</tspan>
      ))}
    </text>
  );
}

type Look = 'plain' | 'read' | 'ok' | 'dev' | 'unknown' | 'hot' | 'gone';
interface Shown {
  text: string;
  look: Look;
}

function CellRow({ x, y, label, cells, opacity = 1, dashed = false }: {
  x: number; y: number; label: string; cells: Shown[]; opacity?: number; dashed?: boolean;
}) {
  return (
    <g opacity={opacity}>
      <text x={x + CELLS_DX - 8} y={y + CH / 2 + 4} fontSize="12" textAnchor="end" fontWeight="600" style={MONO}>{label}</text>
      {cells.map((c, i) => {
        const fill =
          c.look === 'hot' ? 'var(--global)' : c.look === 'dev' ? 'var(--global-wash)' : c.look === 'ok' ? 'var(--ok-wash)'
            : c.look === 'read' ? 'var(--rule)' : c.look === 'gone' ? 'none' : 'var(--cell)';
        const stroke =
          c.look === 'hot' || c.look === 'dev' || c.look === 'unknown' ? 'var(--global)' : c.look === 'ok' ? 'var(--ok)'
            : c.look === 'read' ? 'var(--ink)' : c.look === 'gone' ? 'var(--ink-3)' : 'var(--rule)';
        const ink = c.look === 'hot' ? 'var(--page)' : c.look === 'unknown' || c.look === 'gone' ? 'var(--ink-3)' : 'var(--ink)';
        return (
          <g key={i}>
            <rect x={x + CELLS_DX + i * CW} y={y} width={CW - 3} height={CH} rx="2" fill={fill} stroke={stroke}
              strokeDasharray={dashed || c.look === 'unknown' || c.look === 'gone' ? '3 2' : undefined} />
            <text x={x + CELLS_DX + i * CW + (CW - 3) / 2} y={y + CH / 2 + 4} fontSize="11.5" textAnchor="middle"
              style={{ fill: ink, fontVariantNumeric: 'tabular-nums' }}>{c.text}</text>
          </g>
        );
      })}
    </g>
  );
}

function Box({ at, w, h = BOX_H, title, dashed = false, children }: {
  at: { x: number; y: number }; w: number; h?: number; title: string; dashed?: boolean; children?: ReactNode;
}) {
  return (
    <g>
      <rect x={at.x} y={at.y} width={w} height={h} rx="4" fill="none" stroke="var(--rule)" strokeWidth="1.4"
        strokeDasharray={dashed ? '5 4' : undefined} />
      <text x={at.x + 10} y={at.y + 18} fontSize="12" fontWeight="600">{title}</text>
      {children}
    </g>
  );
}

/** The y of row r (A, B, C) inside a box whose top is at y. */
const rowY = (box: { y: number }, r: number) => box.y + ROWS_DY[r];

/** The two lanes, drawn up to `cursor` (in calls; 2.5 is half way through the third call). */
function Lanes({ g, steps, cursor, labels }: { g: Geo; steps: LaneStep[]; cursor: number; labels: string[] }) {
  const { t } = useLanguage();
  const L = g.lane;
  const n = steps.length;
  const sw = L.w / n;
  const x = (c: number) => L.x + c * sw;
  const upTo = (k: number) => Math.max(0, Math.min(1, cursor - k)); // how much of slot k has been drawn
  const issued = steps.findIndex((s) => s.pending.length > 0);
  const finished = steps.findIndex((s) => s.finishes.length > 0);
  // The kernel occupies the device from just after it is issued until part way into the call that waits for it.
  const k0 = issued + 0.3;
  const k1 = finished + 0.4;
  const kEnd = Math.min(cursor, k1);
  const current = (k: number) => k === Math.min(n - 1, Math.floor(cursor)) && cursor > 0 && cursor < n;
  // Wide: each lane is named to its left. Narrow: above it.
  const name = (label: string, y: number) =>
    g.narrow ? (
      <text x={L.x} y={y - 5} fontSize="11" fontWeight="600">{label}</text>
    ) : (
      <text x={L.x - 10} y={y + 14} fontSize="12" textAnchor="end" fontWeight="600">{label}</text>
    );
  return (
    <g>
      {name(t('host thread', 'luồng host'), L.host)}
      {name(t('device', 'thiết bị'), L.device)}
      <rect x={L.x} y={L.host} width={L.w} height={L.h} fill="none" stroke="var(--rule-soft)" />
      <rect x={L.x} y={L.device} width={L.w} height={L.h} fill="none" stroke="var(--rule-soft)" />
      {steps.map((s, k) => {
        const f = upTo(k);
        if (f <= 0) return null;
        const blocked = s.host === 'blocked';
        return (
          <g key={k}>
            <rect x={x(k) + 1} y={L.host} width={Math.max(0, sw * f - 2)} height={L.h} rx="2"
              fill={blocked ? 'none' : 'var(--ink-2)'} stroke="var(--ink-2)" strokeDasharray={blocked ? '3 2' : undefined} />
            {blocked && f > 0.7 ? (
              <text x={x(k) + sw / 2} y={L.host + 14} fontSize={g.narrow ? 9 : 10.5} textAnchor="middle">{t('waits', 'chờ')}</text>
            ) : null}
          </g>
        );
      })}
      {issued >= 0 && kEnd > k0 ? (
        <g>
          <rect x={x(k0)} y={L.device} width={x(kEnd) - x(k0)} height={L.h} rx="2" fill="var(--ink-2)" stroke="var(--ink-2)" />
          {x(kEnd) - x(k0) > 44 ? (
            <text x={(x(k0) + x(kEnd)) / 2} y={L.device + 14} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>kernel</text>
          ) : null}
        </g>
      ) : null}
      {labels.map((l, k) => {
        // Narrow: a label of two words goes on two lines.
        const parts = g.narrow && l.includes(' ') ? [l.slice(0, l.lastIndexOf(' ')), l.slice(l.lastIndexOf(' ') + 1)] : [l];
        return (
          <Lines key={k} x={x(k) + sw / 2} y={L.device + L.h + 14} lines={parts} dy={11} fontSize={L.labelSize} textAnchor="middle"
            style={{ fill: current(k) ? 'var(--ink)' : 'var(--ink-3)' }} fontWeight={current(k) ? 650 : 400} />
        );
      })}
      {cursor > 0 && cursor < n ? (
        <line x1={x(cursor)} y1={L.host - 5} x2={x(cursor)} y2={L.device + L.h + 5} stroke="var(--ink)" strokeWidth="1.5" />
      ) : null}
    </g>
  );
}

/** The state line under the lanes: one line when wide, three when narrow. */
function Status({ g, parts }: { g: Geo; parts: string[] }) {
  const lines = g.narrow ? parts.map((p, i) => (i < parts.length - 1 ? p + ' ·' : p)) : [parts.join('   ·   ')];
  return <Lines x={g.lane.x} y={g.statusY} lines={lines} fontSize={g.narrow ? 11.5 : 12} style={{ whiteSpace: 'pre' }} />;
}

function slotLabel(p: Program, id: string, t: Translate) {
  const english = id === 'launch' ? (p.backend === 'metal' ? 'commit' : 'launch')
    : id.replace(/-/g, ' ').replace('copy in', 'copy').replace('copy back', 'copy C');
  const vietnamese = english
    .replace(/^alloc /, 'cấp ')
    .replace(/^copy C$/, 'chép C')
    .replace(/^copy /, 'chép ')
    .replace(/^free /, 'giải phóng ')
    .replace(/^fill$/, 'điền')
    .replace(/^launch$/, 'chạy')
    .replace(/^wait$/, 'chờ')
    .replace(/^read$/, 'đọc');
  return t(english, vietnamese);
}

/** How many of `count` elements a step that works element by element has done at fraction f. */
const done = (f: number, count = N) => Math.min(count, Math.floor(clamp(f) * (count + 0.999)));

export interface Picture {
  /** Bytes that have crossed so far, counting the elements of a copy in progress. */
  bytes: number;
  frees: number;
  host: Record<ArrayId, Cell[]> | null;
  device: Record<ArrayId, { cells: Cell[]; freed: boolean; appear: number } | null>;
  /** An element in transit: which array, which index, how far (0 host side, 1 device side). */
  moving: { array: ArrayId; index: number; t: number } | null;
  /** How many results the kernel has produced so far in the drawing. */
  kernelCells: number;
  hostState: 'running' | 'blocked';
  deviceState: 'idle' | 'kernel issued' | 'idle again';
  state: State;
}

/**
 * The model's state at a cursor between calls: the state before the current call,
 * with as many elements of the state after it as the call has got through.
 */
export function pictureAt(p: Program, cursor: number): Picture {
  const n = p.calls.length;
  const k = Math.min(n - 1, Math.floor(cursor));
  const f = cursor >= n ? 1 : cursor - k;
  const before = run(p, k);
  const after = run(p, k + 1);
  const call = p.calls[k];
  const steps = lanes(p);
  const issued = steps.findIndex((s) => s.pending.length > 0);
  const finished = steps.findIndex((s) => s.finishes.length > 0);
  const k0 = issued + 0.3;
  const k1 = finished + 0.4;
  const kernelCells = cursor <= k0 ? 0 : Math.round(clamp((cursor - k0) / (k1 - k0)) * N);
  const finalC = run(p, finished + 1).device.C!.cells;

  const host = before.host ? { A: [...before.host.A], B: [...before.host.B], C: [...before.host.C] } : null;
  const device = {} as Picture['device'];
  for (const a of IDS) {
    const d = before.device[a];
    device[a] = d ? { cells: [...d.cells], freed: d.freed, appear: 1 } : null;
  }
  let bytes = before.bytesMoved;
  let moving: Picture['moving'] = null;
  const o = call.op;
  if (o.op === 'alloc') {
    const d = after.device[o.array]!;
    device[o.array] = { cells: [...d.cells], freed: false, appear: ramp(f, 0.1, 0.5) };
  } else if (o.op === 'fill') {
    const m = done(f);
    const d = device[o.array]!;
    for (let i = 0; i < m; i++) d.cells[i] = after.device[o.array]!.cells[i];
  } else if (o.op === 'copy' && o.kind === 'h2d') {
    const m = done(f);
    const d = device[o.dst.array]!;
    for (let i = 0; i < m; i++) d.cells[i] = after.device[o.dst.array]!.cells[i];
    bytes += m * FLOAT_BYTES;
    if (f > 0 && m < N) moving = { array: o.dst.array, index: m, t: clamp(f * (N + 0.999) - m) };
  } else if (o.op === 'copy' && o.kind === 'd2h' && host) {
    // The call first waits for the kernel (the first 0.4 of the slot), then the elements cross.
    const g = ramp(f, 0.4, 1);
    const m = done(g);
    for (let i = 0; i < m; i++) host.C[i] = after.host!.C[i];
    bytes += m * FLOAT_BYTES;
    if (f > 0.4 && m < N) moving = { array: 'C', index: m, t: 1 - clamp(g * (N + 0.999) - m) };
  } else if (o.op === 'free') {
    if (f > 0.35) device[o.array] = { ...device[o.array]!, freed: true };
  }
  // The kernel's results, as far as the drawing has let it get.
  if (cursor > k0 && k <= finished && device.C) {
    const c = [...before.device.C!.cells];
    for (const i of FILL_ORDER.slice(0, kernelCells)) c[i] = finalC[i];
    device.C = { ...device.C, cells: c };
  } else if (k > finished && device.C && !device.C.freed) device.C = { ...device.C, cells: [...finalC] };
  const frees = before.frees + (o.op === 'free' && f > 0.35 ? 1 : 0);
  return {
    bytes, frees, host, device, moving, kernelCells,
    hostState: after.hostThread,
    deviceState: cursor <= k0 ? 'idle' : cursor < k1 ? 'kernel issued' : 'idle again',
    state: before,
  };
}

const hostLook = (v: Cell, ok = false): Shown => ({ text: showCell(v), look: ok ? 'ok' : 'plain' });
function devLook(cells: Cell[], freed: boolean, hot: number | null): Shown[] {
  return cells.map((v, i) =>
    freed ? { text: '', look: 'gone' } : v === 'uncleared' ? { text: '?', look: 'unknown' } : { text: showCell(v), look: i === hot ? 'hot' : 'dev' },
  );
}

function Frame({ g, label, head, foot, children }: { g: Geo; label: string; head: string; foot?: string; children: ReactNode }) {
  const footLines = foot ? wrapText(foot, g.wrap) : [];
  return (
    <svg viewBox={`0 0 ${g.W} ${g.H}`} role="img" aria-label={label} data-layout={g.narrow ? 'narrow' : 'wide'}
      style={{ width: '100%', maxWidth: g.narrow ? 480 : g.W, display: 'block' }}>
      <Lines x={g.narrow ? 6 : 16} y={g.headY} lines={wrapText(head, g.wrap)} dy={15} fontSize={g.narrow ? 12.5 : 13} fontWeight="600" />
      {children}
      {foot ? (
        <Lines x={g.narrow ? 6 : 16} y={g.H - 14 - (footLines.length - 1) * 16} lines={footLines} fontSize={g.narrow ? 12.5 : 13} fontWeight="600" />
      ) : null}
    </svg>
  );
}

/** Two memories and two lanes for the CUDA stub at `cursor`. */
function Separate({ cursor, head, foot }: { cursor: number; head: string; foot?: string }) {
  const { t } = useLanguage();
  const g = useNarrow() ? NARROW : WIDE;
  const p = SCENE_CUDA;
  const pic = pictureAt(p, cursor);
  const steps = lanes(p);
  const n = p.calls.length;
  const call = p.calls[Math.min(n - 1, Math.floor(cursor))];
  const ptr = run(p, Math.min(n, Math.round(cursor + 0.35))).pointers.filter((x) => x.name.endsWith('_d'));
  const short = (r: string) => (r === 'nothing yet' ? t('not set', 'chưa gán') : r === 'device memory' ? t('device', 'thiết bị') : t('freed', 'đã giải phóng'));
  const anyDevice = ptr.some((x) => x.refersTo === 'device memory');
  const hostText = cursor <= 0 || cursor >= n ? t('running', 'đang chạy') : pic.hostState === 'blocked' ? t(`blocked inside ${call.api[0] ?? 'the call'}`, `đang chờ trong ${call.api[0] ?? 'lời gọi'}`) : t('running', 'đang chạy');
  const devText = pic.deviceState === 'kernel issued' ? t(`vecAddKernel: ${pic.kernelCells} of ${N} results`, `vecAddKernel: ${pic.kernelCells}/${N} kết quả`) : t('nothing to do', 'đang rảnh');
  const [bx1, by1, bx2, by2] = g.boundary;
  return (
    <Frame g={g} label={t('Animation of the host stub over host memory and device memory', 'Hoạt ảnh hàm bao qua bộ nhớ host và thiết bị')} head={head} foot={foot}>
      <Box at={g.host} w={g.host.w} title={t('Host memory', 'Bộ nhớ host')}>
        {IDS.map((a, r) => (
          <CellRow key={a} x={g.host.x} y={rowY(g.host, r)} label={`${a}_h`}
            cells={pic.host![a].map((v, i) => (a === 'C' ? (v === WANT[i] ? hostLook(v, true) : { text: '', look: 'plain' as Look }) : hostLook(v)))} />
        ))}
        <text x={g.host.x + 10} y={g.host.y + BOX_H - 10} fontSize="11" style={{ ...MONO, whiteSpace: 'pre' }}>
          {ptr.map((x) => `${x.name}: ${short(x.refersTo)}`).join('   ')}
        </text>
      </Box>
      <Box at={g.dev} w={g.dev.w} title={t('Device memory', 'Bộ nhớ thiết bị')}>
        {IDS.map((a, r) => {
          const d = pic.device[a];
          if (!d) return null;
          const hot = pic.moving && pic.moving.array === a && a !== 'C' ? pic.moving.index - 1 : null;
          return <CellRow key={a} x={g.dev.x} y={rowY(g.dev, r)} label={`${a}_d`} opacity={d.appear} cells={devLook(d.cells, d.freed, hot)} />;
        })}
        {anyDevice ? (
          <text x={g.dev.x + 10} y={g.dev.y + BOX_H - 10} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{t('the host cannot read or write here', 'host không thể đọc hay ghi ở đây')}</text>
        ) : null}
      </Box>
      <line x1={bx1} y1={by1} x2={bx2} y2={by2} stroke="var(--ink-3)" strokeDasharray="4 4" />
      <text x={g.bytes.x} y={g.bytes.y} fontSize="12" textAnchor="middle" fontWeight="600">{t('bytes across:', 'byte đã chuyển:')} {pic.bytes}</text>
      {pic.moving ? (
        (() => {
          const r = IDS.indexOf(pic.moving.array);
          const from = { x: g.host.x + CELLS_DX + pic.moving.index * CW, y: rowY(g.host, r) };
          const to = { x: g.dev.x + CELLS_DX + pic.moving.index * CW, y: rowY(g.dev, r) };
          const t = pic.moving.t;
          const x = from.x + (to.x - from.x) * t;
          const y = from.y + (to.y - from.y) * t;
          const src = pic.moving.array === 'C' ? run(p, n).device.C!.cells : IN[pic.moving.array as 'A' | 'B'];
          return (
            <g>
              <rect x={x} y={y} width={CW - 3} height={CH} rx="2" fill="var(--global)" stroke="var(--global)" />
              <text x={x + (CW - 3) / 2} y={y + CH / 2 + 4} fontSize="11.5" textAnchor="middle"
                style={{ fill: 'var(--page)' }}>{showCell(src[pic.moving.index])}</text>
            </g>
          );
        })()
      ) : null}
      <Lanes g={g} steps={steps} cursor={cursor} labels={p.calls.map((c) => slotLabel(p, c.id, t))} />
      <Status g={g} parts={[t(`host thread: ${hostText}`, `luồng host: ${hostText}`), t(`device: ${devText}`, `thiết bị: ${devText}`), t(`free calls: ${pic.frees}`, `lượt giải phóng: ${pic.frees}`)]} />
    </Frame>
  );
}

/** One memory and two lanes for the Metal stub with its data created in shared buffers. */
function Shared({ cursor, head, foot }: { cursor: number; head: string; foot?: string }) {
  const { t } = useLanguage();
  const g = useNarrow() ? NARROW_ONE : WIDE;
  const p = SCENE_METAL;
  const pic = pictureAt(p, cursor);
  const steps = lanes(p);
  const n = p.calls.length;
  const x = g.narrow ? 20 : 150;
  const reading = p.calls[Math.min(n - 1, Math.floor(cursor))].id === 'read' ? done(cursor - Math.floor(cursor)) : cursor >= n ? N : 0;
  const hostText = cursor > 0 && cursor < n && pic.hostState === 'blocked' ? t('blocked inside waitUntilCompleted', 'đang chờ trong waitUntilCompleted') : t('running', 'đang chạy');
  const devText = pic.deviceState === 'kernel issued' ? t(`add_arrays: ${pic.kernelCells} of ${N} results`, `add_arrays: ${pic.kernelCells}/${N} kết quả`) : t('nothing to do', 'đang rảnh');
  const cNote = reading > 0 ? t('read after the wait', 'đọc sau khi chờ') : pic.kernelCells > 0 && pic.kernelCells < N ? t('not yet the result', 'kết quả chưa đầy đủ') : '';
  const box = { x: g.host.x, y: g.host.y };
  const side = x + CELLS_DX + N * CW + 18;
  return (
    <Frame g={g} label={t('Animation of the same steps with one shared memory', 'Hoạt ảnh chạy lại với một bộ nhớ dùng chung')} head={head} foot={foot}>
      <Box at={box} w={g.narrow ? g.host.w : g.W - 32} h={g.narrow ? BOX_H + 22 : BOX_H} title={t('One memory, shared by the CPU and the GPU', 'Một bộ nhớ dùng chung cho CPU và GPU')}>
        {IDS.map((a, r) => {
          const d = pic.device[a];
          if (!d) return null;
          return (
            <CellRow key={a} x={x} y={rowY(box, r)} label={`buffer ${a}`} opacity={d.appear}
              cells={d.cells.map((v, i) => (a === 'C' && i < reading && v === WANT[i] ? { text: showCell(v), look: 'ok' as Look } : { text: showCell(v), look: 'dev' as Look }))} />
          );
        })}
        {g.narrow ? (
          <>
            <text x={box.x + 10} y={box.y + BOX_H + 2} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{t('the host reads and writes these through .contents', 'host đọc và ghi qua .contents')}</text>
            <text x={box.x + 10} y={box.y + BOX_H + 16} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{cNote ? `C: ${cNote}` : ''}</text>
          </>
        ) : (
          <>
            <text x={side} y={rowY(box, 0) + 16} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{t('the host reads and writes', 'host đọc và ghi')}</text>
            <text x={side} y={rowY(box, 0) + 31} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{t('these through .contents', 'các buffer qua .contents')}</text>
            <text x={side} y={rowY(box, 2) + 16} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{cNote}</text>
          </>
        )}
      </Box>
      <text x={g.narrow ? g.bytes.x : g.W / 2} y={g.narrow ? g.bytes.y + 14 : g.bytes.y} fontSize="12" textAnchor="middle" fontWeight="600" style={{ whiteSpace: 'pre' }}>
        {t('bytes across:', 'byte đã chuyển:')} {pic.bytes}{'   ·   '}{t('copies:', 'lượt sao chép:')} {run(p, Math.floor(cursor)).copies}
      </text>
      <Lanes g={g} steps={steps} cursor={cursor} labels={p.calls.map((c) => slotLabel(p, c.id, t))} />
      <Status g={g} parts={[t(`host thread: ${hostText}`, `luồng host: ${hostText}`), t(`device: ${devText}`, `thiết bị: ${devText}`), t(`free calls: ${pic.frees}`, `lượt giải phóng: ${pic.frees}`)]} />
    </Frame>
  );
}

/** Beat 1: the loop of Fig. 2.4, with no device involved. */
function HostOnly({ p }: { p: number }) {
  const { t } = useLanguage();
  const g = useNarrow() ? NARROW : WIDE;
  const i = done(ramp(p, 0.08, 0.9));
  const cur = i < N ? i : null;
  const steps: LaneStep[] = Array.from({ length: N }, (_, k) => ({
    id: `i = ${k}`, group: 'fill', host: 'running', finishes: [], pending: [], bytesMoved: 0,
  }));
  const row = (a: 'A' | 'B'): Shown[] => IN[a].map((v, k) => ({ text: String(v), look: k === cur ? 'read' : 'plain' }));
  return (
    <Frame g={g} label={t('Animation of vector addition on the host alone', 'Hoạt ảnh cộng vector chỉ trên host')}
      head={t('C_h[i] = A_h[i] + B_h[i]: one host thread, one memory', 'C_h[i] = A_h[i] + B_h[i]: một luồng host, một bộ nhớ')}
      foot={i >= N ? t(`${N} additions, and no byte left host memory`, `${N} phép cộng, mọi byte vẫn ở bộ nhớ host`) : undefined}>
      <Box at={g.host} w={g.host.w} title={t('Host memory', 'Bộ nhớ host')}>
        <CellRow x={g.host.x} y={rowY(g.host, 0)} label="A_h" cells={row('A')} />
        <CellRow x={g.host.x} y={rowY(g.host, 1)} label="B_h" cells={row('B')} />
        <CellRow x={g.host.x} y={rowY(g.host, 2)} label="C_h" cells={WANT.map((v, k) => (k < i ? { text: String(v), look: 'ok' } : { text: '', look: 'plain' }))} />
      </Box>
      <Box at={g.dev} w={g.dev.w} title={t('Device memory', 'Bộ nhớ thiết bị')} dashed>
        <text x={g.dev.x + g.dev.w / 2} y={g.dev.y + BOX_H / 2 + 8} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t('nothing of ours yet', 'chưa chứa dữ liệu của ta')}</text>
      </Box>
      <Lanes g={g} steps={steps} cursor={ramp(p, 0.08, 0.9) * N} labels={steps.map((s) => s.id.replace(' = ', '='))} />
      <Status g={g} parts={[t(`host thread: ${i < N ? `adding element ${i}` : 'done'}`, `luồng host: ${i < N ? `đang cộng phần tử ${i}` : 'đã xong'}`), t('device: nothing to do', 'thiết bị: đang rảnh')]} />
    </Frame>
  );
}

const CALLS = SCENE_CUDA.calls.length;
const BILL = run(SCENE_CUDA);
const SHARED_END = run(SCENE_METAL);

type Translate = ReturnType<typeof useLanguage>['t'];

const beats: (Omit<Beat, 'draw'> & { draw: (p: number, t: Translate) => ReactNode })[] = [
  { clip: 'round-trip-1', label: 'Host only', vietnamese: { label: 'Chỉ dùng host', caption: 'Bắt đầu khi chưa dùng GPU. Một luồng host duyệt A và B, cộng từng cặp rồi ghi tổng vào C. Cả ba mảng nằm trong bộ nhớ host. Thiết bị bên phải có bộ nhớ riêng và chưa chứa dữ liệu của ta.' }, draw: (p) => <HostOnly p={p} /> },
  {
    clip: 'round-trip-2',
    label: 'Second memory', vietnamese: { label: 'Bộ nhớ thứ hai', caption: 'Muốn dùng thiết bị, dữ liệu cần nằm trong bộ nhớ của nó. Ba vùng nhớ được cấp phát. Chưa có thao tác khởi tạo nên chưa biết nội dung. Host đang giữ ba địa chỉ, mỗi mảng một địa chỉ. Host có thể truyền chúng cho runtime và kernel nhưng không thể tự đọc dữ liệu qua các địa chỉ đó.' },
    draw: (p, t) => (
      <Separate cursor={ramp(p, 0.12, 0.75) * 3} head={t('Three allocations in device memory: not cleared', 'Cấp phát ba vùng bộ nhớ thiết bị, chưa khởi tạo')}
        foot={p > 0.8 ? t('A_d, B_d and C_d hold addresses the host can pass on and cannot read through', 'Host có thể truyền địa chỉ A_d, B_d và C_d nhưng không thể đọc qua chúng') : undefined} />
    ),
  },
  {
    clip: 'round-trip-3',
    label: 'Copy in', vietnamese: { label: 'Chép sang', caption: 'Phần 1. Chép từng phần tử của A và B sang thiết bị. Khi mỗi lượt chép chạy, luồng host phải chờ trong lời gọi. Hãy nhìn bộ đếm. Mỗi mảng có 8 số float, mỗi số 4 byte, nên một lượt chuyển 32 byte. Đã chuyển 64 byte trước khi bắt đầu tính toán.' },
    draw: (p, t) => {
      const cursor = 3 + ramp(p, 0.1, 0.9) * 2;
      return (
        <Separate cursor={cursor} head={t('Part 1: A and B cross the boundary while the host thread waits', 'Phần 1: chép A và B sang trong khi luồng host chờ')}
          foot={p > 0.9 ? t(`${pictureAt(SCENE_CUDA, cursor).bytes} bytes across, and no addition has happened yet`, `đã chuyển ${pictureAt(SCENE_CUDA, cursor).bytes} byte trước phép cộng đầu tiên`) : undefined} />
      );
    },
  },
  {
    clip: 'round-trip-4',
    label: 'Launch', vietnamese: { label: 'Khởi chạy', caption: 'Phần 2. Một câu lệnh khởi chạy kernel rồi trả về ngay. Host không đợi thiết bị kết thúc, thậm chí không đợi nó bắt đầu. Tạm coi thiết bị là nơi tính các giá trị cho C. Mục 2.5 giải thích cách nó thực hiện.' },
    draw: (p, t) => (
      <Separate cursor={5 + (p < 0.25 ? ramp(p, 0.05, 0.25) * 0.3 : 0.3 + ramp(p, 0.25, 1) * 0.7)}
        head={t('Part 2: the launch returns at once', 'Phần 2: lệnh khởi chạy trả về ngay')}
        foot={p > 0.3 ? t('the host has moved on; the device fills C_d in its own memory', 'host tiếp tục, thiết bị ghi C_d trong bộ nhớ của nó') : undefined} />
    ),
  },
  {
    clip: 'round-trip-5',
    label: 'Copy back', vietnamese: { label: 'Chép về', caption: 'Phần 3. Host yêu cầu chép C về. Lượt chép được xếp sau kernel, nên host phải đợi kernel rồi đợi 32 byte nữa chuyển về. Cuối cùng, ba mảng thiết bị được giải phóng. Bên gọi chỉ thấy mảng C của mình đã chứa kết quả.' },
    draw: (p, t) => (
      <Separate cursor={p < 0.6 ? 6 + ramp(p, 0.04, 0.6) : 7 + ramp(p, 0.6, 0.92) * 3}
        head={t('Part 3: the copy back waits for the kernel, then C crosses', 'Phần 3: chờ kernel hoàn thành rồi chép C về')}
        foot={p > 0.92 ? t('three frees, and the caller holds C_h', 'ba lượt giải phóng, kết quả nằm trong C_h của bên gọi') : undefined} />
    ),
  },
  {
    clip: 'round-trip-6',
    label: 'The bill', vietnamese: { label: 'Tổng chi phí', caption: 'Để thực hiện 8 phép cộng, chương trình chuyển 96 byte qua ranh giới bộ nhớ, tức 12 byte mỗi phần tử. Giờ chạy lại với một bộ nhớ dùng chung như trên Apple silicon. Dữ liệu không phải chuyển giữa hai bộ nhớ và mô hình không cần lời gọi giải phóng riêng. Host vẫn phải chờ trước khi đọc kết quả.' },
    draw: (p, t) =>
      p < 0.4 ? (
        <Separate cursor={CALLS} head={t(`The bill: ${N} additions, ${BILL.bytesMoved} bytes across the boundary`, `Tổng chi phí: ${N} phép cộng, chuyển ${BILL.bytesMoved} byte`)}
          foot={p > 0.12 ? t(`${BILL.bytesMoved / N} bytes moved for every element added, ${BILL.copies} copies, ${BILL.frees} free calls`, `mỗi phần tử chuyển ${BILL.bytesMoved / N} byte, ${BILL.copies} lượt sao chép, ${BILL.frees} lượt giải phóng`) : undefined} />
      ) : (
        <Shared cursor={ramp(p, 0.44, 0.9) * SCENE_METAL.calls.length} head={t('The same steps with one shared memory', 'Chạy lại với một bộ nhớ dùng chung')}
          foot={p > 0.9
            ? t(`${SHARED_END.bytesMoved} bytes across, ${SHARED_END.copies} copies, ${SHARED_END.frees} free calls, and still one wait before the read`, `chuyển ${SHARED_END.bytesMoved} byte, ${SHARED_END.copies} lượt sao chép, ${SHARED_END.frees} lượt giải phóng, vẫn chờ một lần trước khi đọc`)
            : undefined} />
      ),
  },
];

export function RoundTripScene() {
  const { t } = useLanguage();
  return <Scene chapter="ch02" title={t('Three parts, two memories', 'Ba phần, hai bộ nhớ')}
    beats={beats.map((beat) => ({ ...beat, draw: (p) => beat.draw(p, t) }))} />;
}
