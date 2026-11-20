import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { moved, runFig, zeroCounters, type Counters, type Ev, type Frame, type Run, type View } from '../../sim/managedmem';

// "Two things called unified memory" (section C.6): Fig. C.1's managed sortfile run by
// src/sim/managedmem.ts in three modes, limited (bulk at launch), full (fault and
// migrate) and one memory (Apple silicon). Every page position and counter below is a
// frame of those runs; nothing is drawn by hand. Six beats, one narration clip each
// (content/appC/narration/two-unified-N.txt).

const N = 8;
const bulk = runFig('managed', 'bulk');
const full = runFig('managed', 'migrate');
const one = runFig('managed', 'one');
const explicit = runFig('explicit', 'migrate');
const line = (role: string) => full.lines.findIndex((l) => l.role === role);

/** The frames of some steps of a run, and the view just before them. */
function slice(run: Run, stepIdx: number[]): { before: View; frames: Frame[]; startCounters: Counters } {
  const first = run.steps[stepIdx[0]];
  const before = first.start > 0 ? run.frames[first.start - 1].view : stepIdx[0] > 0 ? run.steps[stepIdx[0] - 1].view : run.initial;
  const startCounters = stepIdx[0] > 0 ? run.steps[stepIdx[0] - 1].counters : zeroCounters();
  const frames = stepIdx.flatMap((s) => run.frames.slice(run.steps[s].start, run.steps[s].end));
  return { before, frames, startCounters };
}

const clamp = (v: number) => Math.max(0, Math.min(1, v));

/** Where things stand at fraction p of a slice: the view, the counters, and a page in flight. */
function at(sl: ReturnType<typeof slice>, p: number, lead = 0.08, tail = 0.12) {
  const n = sl.frames.length;
  const q = clamp((p - lead) / (1 - lead - tail));
  const t = q * n;
  const k = Math.min(n, Math.floor(t));
  const view = k === 0 ? sl.before : sl.frames[k - 1].view;
  const counters = k === 0 ? sl.startCounters : sl.frames[k - 1].counters;
  const active = k === 0 ? false : sl.frames[k - 1].active;
  const next = k < n ? sl.frames[k] : null;
  const flying = next && next.ev.e === 'move' ? { page: next.ev.page, dir: next.ev.dir!, frac: t - k } : null;
  const ev = k > 0 ? sl.frames[k - 1].ev : null;
  return { view, counters, active, flying, ev, done: k >= n };
}

// Geometry.
const W = 720;
const H = 330;
const BOX = { cpu: { x: 20, y: 64, w: 250, h: 170 }, gpu: { x: 450, y: 64, w: 250, h: 170 } };
const PW = 44;
const PH = 30;
const slot = (side: 'cpu' | 'gpu', i: number) => {
  const b = BOX[side];
  return { x: b.x + 22 + (i % 4) * 54, y: b.y + 52 + Math.floor(i / 4) * 46 };
};
const lerp = (a: number, b: number, f: number) => a + (b - a) * f;

function PageBox({ x, y, i, kind, hot }: { x: number; y: number; i: number; kind: 'cpu' | 'gpu' | 'one'; hot?: boolean }) {
  const fill = kind === 'gpu' ? 'var(--global-wash)' : 'var(--cell)';
  const stroke = kind === 'gpu' ? 'var(--global)' : 'var(--ink-2)';
  return (
    <g>
      <rect x={x} y={y} width={PW} height={PH} rx="3" fill={fill} stroke={stroke} strokeWidth={hot ? 2.4 : 1.2} />
      <text x={x + PW / 2} y={y + PH / 2 + 4} fontSize="11" textAnchor="middle">p{i}</text>
    </g>
  );
}

function Counts({ c, x, y }: { c: Counters; x: number; y: number }) {
  const items: [string, number][] = [
    ['to GPU', c.h2d],
    ['to CPU', c.d2h],
    ['GPU faults', c.gpuFaults],
    ['CPU faults', c.cpuFaults],
  ];
  return (
    <g>
      {items.map(([label, v], i) => (
        <g key={label} data-counter={label} transform={`translate(${x + i * 120}, ${y})`}>
          <text fontSize="11" style={{ fill: 'var(--ink-2)' }}>{label}</text>
          <text y={20} fontSize="17" fontWeight="650" data-value="">{v}</text>
        </g>
      ))}
    </g>
  );
}

function TwoBoxes({
  view,
  counters,
  flying,
  headline,
  note,
  locked,
  beat,
}: {
  view: View;
  counters: Counters;
  flying: { page: number; dir: 'h2d' | 'd2h'; frac: number } | null;
  headline: ReactNode;
  note?: ReactNode;
  locked?: boolean;
  beat: number;
}) {
  const pages = view.data?.pages ?? [];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="CPU memory and GPU memory with the pages of the buffer" style={{ width: '100%', maxWidth: W, display: 'block' }}
      data-beat={beat} data-h2d={counters.h2d} data-d2h={counters.d2h}>
      <text x={20} y={26} fontSize="14" fontWeight="620" data-headline="">{headline}</text>
      {(['cpu', 'gpu'] as const).map((side) => (
        <g key={side}>
          <rect x={BOX[side].x} y={BOX[side].y} width={BOX[side].w} height={BOX[side].h} rx="5"
            fill={side === 'gpu' ? 'var(--global-wash)' : 'var(--page)'} fillOpacity={side === 'gpu' ? 0.45 : 1}
            stroke={side === 'gpu' ? 'var(--global)' : 'var(--ink-3)'} />
          <text x={BOX[side].x + 12} y={BOX[side].y + 24} fontSize="12.5" fontWeight="620"
            style={{ fill: side === 'gpu' ? 'var(--global)' : 'var(--ink)' }}>{side === 'cpu' ? 'CPU memory' : 'GPU memory'}</text>
          {Array.from({ length: N }, (_, i) => {
            const s = slot(side, i);
            return <rect key={i} x={s.x} y={s.y} width={PW} height={PH} rx="3" fill="none" stroke="var(--rule)" strokeDasharray="3 3" />;
          })}
        </g>
      ))}
      {/* the link */}
      <line x1={BOX.cpu.x + BOX.cpu.w} y1={150} x2={BOX.gpu.x} y2={150} stroke="var(--ink-3)" strokeWidth="6" strokeLinecap="round" opacity="0.35" />
      <text x={360} y={228} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>interconnect</text>
      {locked ? (
        <g data-locked="">
          <rect x={BOX.cpu.x} y={BOX.cpu.y} width={BOX.cpu.w} height={BOX.cpu.h} rx="5" fill="var(--fail-wash)" stroke="var(--fail)" strokeDasharray="5 3" />
          <text x={BOX.cpu.x + BOX.cpu.w / 2} y={BOX.cpu.y + BOX.cpu.h - 28} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--fail)' }}>
            CPU: hands off managed memory
          </text>
          <text x={BOX.cpu.x + BOX.cpu.w / 2} y={BOX.cpu.y + BOX.cpu.h - 12} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--fail)' }}>
            until the synchronise
          </text>
        </g>
      ) : null}
      {pages.map((pg, i) => {
        if (flying && flying.page === i) {
          const from = slot(flying.dir === 'h2d' ? 'cpu' : 'gpu', i);
          const to = slot(flying.dir === 'h2d' ? 'gpu' : 'cpu', i);
          return (
            <g key={i} data-page={i} data-where="moving">
              <PageBox x={lerp(from.x, to.x, flying.frac)} y={lerp(from.y, to.y, flying.frac)} i={i} kind={flying.frac > 0.5 ? (flying.dir === 'h2d' ? 'gpu' : 'cpu') : flying.dir === 'h2d' ? 'cpu' : 'gpu'} hot />
            </g>
          );
        }
        if (pg.loc === 'cpu' || pg.loc === 'gpu') {
          const s = slot(pg.loc, i);
          return <g key={i} data-page={i} data-where={pg.loc}><PageBox x={s.x} y={s.y} i={i} kind={pg.loc} /></g>;
        }
        return <g key={i} data-page={i} data-where={pg.loc} />;
      })}
      <Counts c={counters} x={30} y={266} />
      {note ? <text x={20} y={322} fontSize="12" data-note="">{note}</text> : null}
    </svg>
  );
}

function OneBox({ view, waited, headline, note }: { view: View; waited: boolean; headline: ReactNode; note?: ReactNode }) {
  const pages = view.data?.pages ?? [];
  const box = { x: 120, y: 92, w: 480, h: 130 };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="One memory shared by the CPU and the GPU" style={{ width: '100%', maxWidth: W, display: 'block' }}
      data-beat={5} data-h2d={0} data-d2h={0} data-waited={waited ? 'yes' : 'no'}>
      <text x={20} y={26} fontSize="14" fontWeight="620" data-headline="">{headline}</text>
      {(['CPU', 'GPU'] as const).map((who, k) => (
        <g key={who}>
          <rect x={k === 0 ? 150 : 450} y={44} width={120} height={28} rx="4" fill="var(--page)" stroke="var(--ink-3)" />
          <text x={k === 0 ? 210 : 510} y={63} fontSize="12" textAnchor="middle" fontWeight="600">{who}</text>
          <line x1={k === 0 ? 210 : 510} y1={72} x2={k === 0 ? 260 : 460} y2={box.y} stroke="var(--ink-3)" />
        </g>
      ))}
      <rect x={box.x} y={box.y} width={box.w} height={box.h} rx="5" fill="var(--page)" stroke="var(--ink-3)" />
      <text x={box.x + 12} y={box.y + 22} fontSize="12.5" fontWeight="620">one system memory, shared buffer</text>
      {pages.map((pg, i) => {
        const x = box.x + 30 + (i % 8) * 54;
        const y = box.y + 50;
        return (
          <g key={i} data-page={i} data-where={pg.loc}>
            {pg.loc === 'one' ? <PageBox x={x} y={y} i={i} kind="one" /> : <rect x={x} y={y} width={PW} height={PH} rx="3" fill="none" stroke="var(--rule)" strokeDasharray="3 3" />}
          </g>
        );
      })}
      <Counts c={{ h2d: 0, d2h: 0, gpuFaults: 0, cpuFaults: 0, remote: 0, invalidations: 0, evictions: 0, copiedInMemory: 0 }} x={30} y={266} />
      {note ? <text x={20} y={322} fontSize="12" data-note="">{note}</text> : null}
    </svg>
  );
}

const fread = line('read-file');
const sortLine = line('sort');
const syncLine = line('sync');
const useLine = line('use');

const s1 = slice(full.run, [full.stepOf[fread]]);
const s2 = slice(bulk.run, [bulk.stepOf[sortLine]]);
const s3 = slice(full.run, [full.stepOf[sortLine]]);
const s4 = slice(full.run, [full.stepOf[syncLine], full.stepOf[useLine]]);
const s5 = slice(one.run, [one.stepOf[fread], one.stepOf[sortLine], one.stepOf[syncLine]]);

/** What the last event did to its page, in a few words. */
function evText(ev: Ev | null, flying: { page: number } | null, who: 'gpu' | 'cpu'): string {
  if (flying) return `p${flying.page} crosses the interconnect`;
  if (!ev || ev.page < 0) return '';
  const other = who === 'gpu' ? 'CPU' : 'GPU';
  switch (ev.e) {
    case 'fault': return `p${ev.page} faults: no ${who.toUpperCase()} mapping`;
    case 'unmap': return `p${ev.page}: the ${other} loses its mapping`;
    case 'move': return `p${ev.page} has crossed`;
    case 'map': return `p${ev.page}: the ${who.toUpperCase()} gains a mapping`;
    case 'replay': return `p${ev.page}: the access replays and completes`;
    default: return '';
  }
}

const beats: Beat[] = [
  {
    clip: 'two-unified-1',
    label: 'First touch',
    draw: (p) => {
      const f = at(s1, p);
      const placed = f.view.data.pages.filter((x) => x.loc === 'cpu').length;
      return (
        <TwoBoxes beat={1} view={f.view} counters={f.counters} flying={null}
          headline={placed === 0 ? 'cudaMallocManaged: one pointer, no page placed yet' : `fread writes the file: ${placed} of ${N} pages placed in CPU memory`}
          note={f.done ? 'Each page sits where it was first touched.' : undefined} />
      );
    },
  },
  {
    clip: 'two-unified-2',
    label: 'Limited mode',
    draw: (p) => {
      const f = at(s2, p);
      return (
        <TwoBoxes beat={2} view={f.view} counters={f.counters} flying={f.flying} locked={p > 0.08}
          headline={`limited: at the launch, every managed page moves (${f.counters.h2d} of ${N})`}
          note={f.done ? 'CUDA 6 on Kepler and Maxwell; Windows, WSL and some Tegra systems today.' : undefined} />
      );
    },
  },
  {
    clip: 'two-unified-3',
    label: 'Full mode',
    draw: (p) => {
      const f = at(s3, p);
      return (
        <TwoBoxes beat={3} view={f.view} counters={f.counters} flying={f.flying}
          headline={f.counters.gpuFaults === 0 ? 'full: the kernel starts, nothing has moved' : `full: ${evText(f.ev, f.flying, 'gpu')}`}
          note={f.done ? 'One fault per page the kernel touches; an untouched page would stay put.' : undefined} />
      );
    },
  },
  {
    clip: 'two-unified-4',
    label: 'Back to the CPU',
    draw: (p) => {
      const f = at(s4, p);
      const back = f.counters.d2h;
      return (
        <TwoBoxes beat={4} view={f.view} counters={f.counters} flying={f.flying}
          headline={f.counters.cpuFaults === 0 ? 'cudaDeviceSynchronize, then use_data reads on the CPU' : `use_data: ${evText(f.ev, f.flying, 'cpu')} (${back} of ${N} back)`}
          note={f.done ? `${moved(full.run.counters)} pages crossed in all. Two explicit copies move ${moved(explicit.run.counters)}.` : undefined} />
      );
    },
  },
  {
    clip: 'two-unified-5',
    label: 'One memory',
    draw: (p) => {
      const f = at(s5, p);
      const waited = !!f.done;
      return (
        <OneBox view={f.view} waited={waited}
          headline="Apple silicon: the CPU writes, the GPU reads in place"
          note={waited ? 'Nothing moved. Still owed: wait for the command buffer before use_data.' : undefined} />
      );
    },
  },
  {
    clip: 'two-unified-6',
    label: 'Side by side',
    draw: (p) => {
      const cols: { name: string; sub: string; n: number; owe: string }[] = [
        { name: 'CUDA, limited', sub: 'two memories, bulk at launch', n: moved(bulk.run.counters), owe: 'ordering, and no CPU touch while a kernel runs' },
        { name: 'CUDA, full', sub: 'two memories, fault and migrate', n: moved(full.run.counters), owe: 'ordering' },
        { name: 'Apple silicon', sub: 'one memory, shared buffer', n: moved(one.run.counters), owe: 'ordering' },
      ];
      return (
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Pages moved in three modes" style={{ width: '100%', maxWidth: W, display: 'block' }} data-beat={6}>
          <text x={20} y={26} fontSize="14" fontWeight="620" data-headline="">one word, two mechanisms</text>
          {cols.map((c, k) => {
            const x = 20 + k * 233;
            const o = clamp((p - k * 0.15) / 0.25);
            return (
              <g key={c.name} opacity={o} data-mode={c.name} data-moved={c.n}>
                <rect x={x} y={50} width={215} height={230} rx="5" fill="var(--page)" stroke="var(--ink-3)" />
                <text x={x + 14} y={76} fontSize="13" fontWeight="650">{c.name}</text>
                <text x={x + 14} y={95} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{c.sub}</text>
                <text x={x + 14} y={150} fontSize="34" fontWeight="650">{c.n}</text>
                <text x={x + 14} y={172} fontSize="11.5">pages crossed the link</text>
                <text x={x + 14} y={214} fontSize="11" style={{ fill: 'var(--ink-2)' }}>still owed:</text>
                <foreignObject x={x + 12} y={220} width={195} height={56}>
                  <div style={{ fontFamily: 'var(--sans)', fontSize: 11.5, lineHeight: 1.3, color: 'var(--ink)' }}>{c.owe}</div>
                </foreignObject>
              </g>
            );
          })}
          <text x={20} y={316} fontSize="12" opacity={clamp((p - 0.6) / 0.2)} data-note="">
            CUDA's Unified Memory migrates pages between two memories; Apple's unified memory is one memory.
          </text>
        </svg>
      );
    },
  },
];

export function TwoUnifiedScene() {
  return <Scene chapter="appC" title="Two things called unified memory" beats={beats} />;
}
