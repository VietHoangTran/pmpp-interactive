import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { blockHistogram } from '../../sim/convcached';
import {
  SCENE_A,
  SCENE_B,
  SCENE_C,
  SCENE_ONE_L2,
  SCENE_SPLIT_L2,
  eventsOf,
  imageDims,
  lineBytes,
  sceneFacts,
  simulate,
  type CacheConfig,
  type CacheEvent,
  type CacheRun,
} from '../../sim/halocache';

// "Where the halo comes from" (section 7.6). Every frame is drawn from our toy cache model
// (sim/halocache.ts: the two scene runs) and the tap classifier (sim/convcached.ts); the
// numbers the narration says are pinned by halocache.test.ts. Five beats, one clip each
// (content/ch07/narration/halo-from-cache-N.txt).

const F = sceneFacts();
const CFG = SCENE_ONE_L2;
const T = CFG.tile;
const R = CFG.r;
const { width: W, height: H } = imageDims(CFG);
const LINES_PER_ROW = W / CFG.lineFloats;
const B_TAPS = blockHistogram({ tile: T, r: R, width: W, height: H, bx: 1, by: 0 });

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

// Cache contents after k accesses, per run, remembered (each beat asks for the same few dozen moments).
const memo = new Map<string, number[][]>();
function contentsAt(cfg: CacheConfig, k: number): number[][] {
  const key = cfg.slices + ':' + k;
  let c = memo.get(key);
  if (!c) {
    c = simulate(cfg, { stopAfter: k }).contents;
    memo.set(key, c);
  }
  return c;
}

/** Event index reached at progress p through events [s, e), holding the end for the last 15%. */
const at = (p: number, s: number, e: number) => s + Math.round(clamp(p / 0.85) * (e - s));
const range = (run: CacheRun, block: number) => {
  const ix = eventsOf(run, block);
  return [ix[0], ix[ix.length - 1] + 1] as const;
};
const tally = (evs: CacheEvent[], keep?: (e: CacheEvent) => boolean) => {
  const xs = keep ? evs.filter(keep) : evs;
  return { hits: xs.filter((e) => e.hit).length, of: xs.length };
};

interface MapProps {
  x: number;
  y: number;
  cell: number;
  /** Lines resident in this cache now. */
  inL2: Set<number>;
  /** Accesses to show as just made: line -> hit or miss (the latest wins). */
  marks: Map<number, boolean>;
  /** Tiles to outline, with a label. */
  tiles?: { bx: number; by: number; label: string; tone: 'shared' | 'ink' }[];
  title?: string;
  /** Lines to underline (e.g. A's lines in B's halo). */
  ring?: Set<number>;
}

function LineMap({ x, y, cell, inL2, marks, tiles = [], title, ring }: MapProps) {
  const lw = CFG.lineFloats * cell;
  const rows: ReactNode[] = [];
  for (let row = 0; row < H; row++)
    for (let k = 0; k < LINES_PER_ROW; k++) {
      const line = row * LINES_PER_ROW + k;
      const mark = marks.get(line);
      const fill = mark === true ? 'var(--ok-wash)' : mark === false ? 'var(--fail-wash)' : inL2.has(line) ? 'var(--global-wash)' : 'var(--cell)';
      const stroke = mark === true ? 'var(--ok)' : mark === false ? 'var(--fail)' : inL2.has(line) ? 'var(--global)' : 'var(--rule)';
      rows.push(
        <rect key={line} x={x + k * lw + 0.5} y={y + row * cell + 0.5} width={lw - 1} height={cell - 1}
          fill={fill} stroke={stroke} strokeWidth={mark === undefined ? 0.6 : 1.4}
          data-line={line} data-state={mark === true ? 'hit' : mark === false ? 'miss' : inL2.has(line) ? 'l2' : 'none'} />,
      );
      if (ring?.has(line))
        rows.push(<line key={'u' + line} x1={x + k * lw + 1} x2={x + (k + 1) * lw - 1} y1={y + (row + 1) * cell - 0.5} y2={y + (row + 1) * cell - 0.5}
          stroke="var(--ink)" strokeWidth="1.2" strokeDasharray="2 1.5" />);
    }
  return (
    <g>
      {title ? <text x={x} y={y - 8} fontSize="11.5" fontWeight="600">{title}</text> : null}
      {rows}
      {/* tile grid */}
      {Array.from({ length: W / T + 1 }, (_, i) => (
        <line key={'v' + i} x1={x + i * T * cell} x2={x + i * T * cell} y1={y} y2={y + H * cell} stroke="var(--ink-3)" strokeWidth="0.8" />
      ))}
      {Array.from({ length: H / T + 1 }, (_, i) => (
        <line key={'h' + i} x1={x} x2={x + W * cell} y1={y + i * T * cell} y2={y + i * T * cell} stroke="var(--ink-3)" strokeWidth="0.8" />
      ))}
      {tiles.map((t) => (
        <g key={t.label}>
          <rect x={x + t.bx * T * cell - 1} y={y + t.by * T * cell - 1} width={T * cell + 2} height={T * cell + 2} fill="none"
            stroke={t.tone === 'shared' ? 'var(--shared)' : 'var(--ink)'} strokeWidth="2.4" />
          <rect x={x + t.bx * T * cell + (T * cell) / 2 - 8} y={y + t.by * T * cell + (T * cell) / 2 - 8} width={16} height={16} rx={2}
            fill="var(--page)" opacity={0.9} />
          <text x={x + t.bx * T * cell + (T * cell) / 2} y={y + t.by * T * cell + (T * cell) / 2 + 5} fontSize="14" fontWeight="700"
            textAnchor="middle" style={{ fill: t.tone === 'shared' ? 'var(--shared)' : 'var(--ink)' }}>{t.label}</text>
        </g>
      ))}
    </g>
  );
}

function Counter({ y, name, label, hits, of, word = 'found in L2' }: { y: number; name: string; label: string; hits: number; of: number; word?: string }) {
  return (
    <g data-counter={name} data-hits={hits} data-of={of}>
      <text x={432} y={y} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{label}</text>
      <text x={432} y={y + 16} fontSize="13" fontWeight="650">{hits} of {of} {word}</text>
    </g>
  );
}

function Legend({ y }: { y: number }) {
  const items: [string, string, string][] = [
    ['var(--global-wash)', 'var(--global)', 'line in L2'],
    ['var(--fail-wash)', 'var(--fail)', 'just read: miss, from DRAM'],
    ['var(--ok-wash)', 'var(--ok)', 'just read: hit in L2'],
  ];
  return (
    <g>
      {items.map(([fill, stroke, label], i) => (
        <g key={label}>
          <rect x={432} y={y + i * 17 - 9} width={22} height={9} fill={fill} stroke={stroke} />
          <text x={460} y={y + i * 17} fontSize="11">{label}</text>
        </g>
      ))}
      <text x={432} y={y + 3 * 17} fontSize="11" style={{ fill: 'var(--ink-2)' }}>one bar = one cache line = one tile row of {CFG.lineFloats} floats</text>
    </g>
  );
}

function Frame({ beat, headline, children, note }: { beat: number; headline: string; children: ReactNode; note?: string }) {
  return (
    <svg viewBox="0 0 720 330" role="img" aria-label="Our toy cache model: which halo reads hit in L2" data-beat={beat}
      style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <text x={16} y={20} fontSize="13" fontWeight="650" data-headline>{headline}</text>
      {children}
      {note ? <text x={16} y={318} fontSize="11" style={{ fill: 'var(--ink-2)' }} data-note>{note}</text> : null}
    </svg>
  );
}

const MAP = { x: 16, y: 48, cell: 6 };
const NOTE = `Our toy model, not a measurement: a ${CFG.capacity}-line LRU cache, one block at a time, image ${W} × ${H} floats, radius ${R}.`;

function marksOf(run: CacheRun, from: number, to: number, keep?: (e: CacheEvent) => boolean) {
  const m = new Map<number, boolean>();
  for (let i = from; i < to; i++) {
    const e = run.events[i];
    if (!keep || keep(e)) m.set(e.line, e.hit);
  }
  return m;
}

const aSet = new Set(F.aLines);
const aBottom = new Set(F.aBottom);

// ---------------------------------------------------------------- beats

function beatA(p: number) {
  const run = F.one;
  const [s, e] = range(run, SCENE_A);
  const k = at(p, s, e);
  const done = run.events.slice(s, k);
  const loads = tally(done.filter((x) => x.phase === 'load'));
  const halo = tally(done.filter((x) => x.phase === 'halo'));
  return (
    <Frame beat={1} headline={k - s <= 8 ? 'Block A loads its own tile: every line comes from DRAM, through L2' : 'Then A reads its halo from global memory: those lines land in L2 too'} note={NOTE}>
      <LineMap {...MAP} inL2={new Set(contentsAt(CFG, k)[0])} marks={marksOf(run, s, k)} tiles={[{ bx: 0, by: 0, label: 'A', tone: 'shared' }]} title="the image, one bar per cache line" />
      <Counter y={66} name="a-load" label="A's own lines (into shared memory)" hits={loads.hits} of={loads.of} />
      <Counter y={108} name="a-halo" label="A's halo lines (read from global)" hits={halo.hits} of={halo.of} />
      <Legend y={170} />
    </Frame>
  );
}

function beatB(p: number) {
  const run = F.one;
  const [s, e] = range(run, SCENE_B);
  const k = at(p, s, e);
  const done = run.events.slice(s, k);
  const own = tally(done.filter((x) => x.phase === 'load'));
  const left = tally(done.filter((x) => x.phase === 'halo' && aSet.has(x.line)));
  const other = tally(done.filter((x) => x.phase === 'halo' && !aSet.has(x.line)));
  return (
    <Frame beat={2} headline={left.of === F.bLeft ? `B's left halo is A's interior: ${left.hits} of ${left.of} lines found in L2` : 'Block B, next door, loads its own tile'} note={NOTE}>
      <LineMap {...MAP} inL2={new Set(contentsAt(CFG, k)[0])} marks={marksOf(run, s, k)} ring={aSet}
        tiles={[{ bx: 0, by: 0, label: 'A', tone: 'ink' }, { bx: 1, by: 0, label: 'B', tone: 'shared' }]} title="the image, one bar per cache line" />
      <Counter y={66} name="b-load" label="B's own lines" hits={own.hits} of={own.of} />
      <Counter y={108} name="b-left" label="B's halo reads of A's lines (dashed)" hits={left.hits} of={left.of} />
      <Counter y={150} name="b-other" label="B's other halo lines" hits={other.hits} of={other.of} />
      <text x={432} y={200} fontSize="11" style={{ fill: 'var(--ink-2)' }} data-taps={`${B_TAPS.shared},${B_TAPS.global},${B_TAPS.ghost}`}>
        <tspan x={432}>B's {B_TAPS.threads} threads make {B_TAPS.taps.toLocaleString('en-US')} taps:</tspan>
        <tspan x={432} dy={14}>{B_TAPS.shared.toLocaleString('en-US')} shared, {B_TAPS.global} global, {B_TAPS.ghost} ghost</tspan>
      </text>
      <Legend y={250} />
    </Frame>
  );
}

function beatC(p: number) {
  const run = F.one;
  const [, eB] = range(run, SCENE_B);
  const [sC, eC] = range(run, SCENE_C);
  const between = p < 0.35;
  const k = between ? eB + Math.round(ramp(p, 0, 0.3) * (sC - eB)) : at(ramp(p, 0.35, 1), sC, eC);
  const cur = run.events[Math.max(0, k - 1)];
  const done = between ? [] : run.events.slice(sC, k);
  const onA = tally(done.filter((x) => x.phase === 'halo' && aBottom.has(x.line)));
  const halo = tally(done.filter((x) => x.phase === 'halo'));
  const runningBlock = between ? run.blocks[cur.block] : run.blocks[SCENE_C];
  return (
    <Frame beat={3}
      headline={between ? `Blocks ${SCENE_B + 1} to ${SCENE_C - 1} run; the ${CFG.capacity}-line cache turns over`
        : onA.of === F.cOnABottom ? `C reads A's bottom rows as halo: ${onA.hits} of ${onA.of} lines still in L2` : 'C, the first block of the next row'}
      note={NOTE}>
      <LineMap {...MAP} inL2={new Set(contentsAt(CFG, k)[0])} marks={between ? new Map() : marksOf(run, sC, k)} ring={aBottom}
        tiles={[{ bx: 0, by: 0, label: 'A', tone: 'ink' },
          ...(between ? [{ bx: runningBlock.bx, by: runningBlock.by, label: '·', tone: 'shared' as const }] : [{ bx: 0, by: 1, label: 'C', tone: 'shared' as const }])]}
        title="the image, one bar per cache line" />
      <g data-counter="since-b" data-hits={between ? cur.block - SCENE_B : SCENE_C - SCENE_B - 1} data-of={F.evictedBeforeC}>
        <text x={432} y={66} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>blocks run since B</text>
        <text x={432} y={82} fontSize="13" fontWeight="650">{between ? Math.max(0, cur.block - SCENE_B) : SCENE_C - SCENE_B - 1}</text>
        <text x={432} y={104} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>lines evicted between B and C</text>
        <text x={432} y={120} fontSize="13" fontWeight="650">{F.evictedBeforeC}</text>
      </g>
      <Counter y={150} name="c-on-a" label="C's reads of A's bottom rows (dashed)" hits={onA.hits} of={onA.of} />
      <Counter y={192} name="c-halo" label="all of C's halo lines" hits={halo.hits} of={halo.of} />
      <Legend y={250} />
    </Frame>
  );
}

function beatSplit(p: number) {
  const run = F.split;
  const [sA] = range(run, SCENE_A);
  const [, eB] = range(run, SCENE_B);
  const k = at(p, sA, eB);
  const done = run.events.slice(sA, k);
  const left = tally(done.filter((x) => x.block === SCENE_B && x.phase === 'halo' && aSet.has(x.line)));
  const c = contentsAt(SCENE_SPLIT_L2, k);
  const cell = 4.5;
  const die = (s: number, x: number) => {
    const tiles = [
      ...(F.sliceA === s ? [{ bx: 0, by: 0, label: 'A', tone: 'shared' as const }] : []),
      ...(F.sliceB === s ? [{ bx: 1, by: 0, label: 'B', tone: 'shared' as const }] : []),
    ];
    return (
      <LineMap x={x} y={64} cell={cell} inL2={new Set(c[s])} marks={marksOf(run, sA, k, (e) => e.slice === s)} tiles={tiles}
        ring={s === F.sliceB ? aSet : undefined} title={`L2 of die ${s} (${SCENE_SPLIT_L2.capacity / SCENE_SPLIT_L2.slices} lines)`} />
    );
  };
  return (
    <Frame beat={4} headline={left.of === F.splitBLeft ? `Split L2: B's reads of A's lines, ${left.hits} of ${left.of} in its die's L2` : `Split L2: A runs on die ${F.sliceA}, B on die ${F.sliceB}`}
      note={`Our toy model: two dies, each with half of the ${SCENE_SPLIT_L2.capacity} lines; blocks dealt to dies at random (seed ${SCENE_SPLIT_L2.seed}).`}>
      <g data-dies={`${F.sliceA},${F.sliceB}`}>
        {die(0, 16)}
        {die(1, 368)}
      </g>
      <Counter y={252} name="split-b-left" label="B's halo reads of A's lines (dashed)" hits={left.hits} of={left.of} word="found in B's L2" />
      <text x={16} y={258} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>
        <tspan x={16}>A loaded its lines into die {F.sliceA}'s L2.</tspan>
        <tspan x={16} dy={15}>B runs on die {F.sliceB} and looks only there.</tspan>
      </text>
    </Frame>
  );
}

function beatBars(p: number) {
  const b = F.bytes;
  const bars: [string, string, number][] = [
    ['ideal', 'all hits: each line from DRAM once (the ideal)', b.ideal],
    ['one', 'the model, one shared L2 (part 3)', b.one],
    ['split', 'the model, L2 split per die (part 4)', b.split],
    ['none', 'all misses: no reuse between blocks', b.noReuse],
  ];
  const max = Math.max(...bars.map((x) => x[2]));
  const g = ramp(p, 0.05, 0.5);
  return (
    <Frame beat={5} headline="DRAM bytes per output: which bar does your GPU give you?" note={`Bytes per output: cache lines fetched × ${lineBytes(CFG.lineFloats)} bytes, plus one 4-byte store, over the outputs. Our toy model.`}>
      {bars.map(([id, label, v], i) => (
        <g key={id} data-bar={id} data-bytes={v}>
          <text x={16} y={64 + i * 52} fontSize="12">{label}</text>
          <rect x={16} y={72 + i * 52} width={Math.max(1, (520 * v * g) / max)} height={18} fill="var(--global)" />
          <text x={16 + (520 * v * g) / max + 8} y={86 + i * 52} fontSize="12.5" fontWeight="650">{g > 0.98 ? `${v.toFixed(v % 1 ? 1 : 0)} B` : ''}</text>
        </g>
      ))}
      <text x={16} y={290} fontSize="12" style={{ fill: 'var(--ink-2)' }} opacity={ramp(p, 0.55, 0.75)}>
        The book: a "significant probability" of hits. A profiler's L2 hit rate, or lab 7.2's timings, tell you which.
      </text>
    </Frame>
  );
}

const beats: Beat[] = [
  { clip: 'halo-from-cache-1', label: 'Block A', draw: beatA },
  { clip: 'halo-from-cache-2', label: 'Next door', draw: beatB },
  { clip: 'halo-from-cache-3', label: 'A row later', draw: beatC },
  { clip: 'halo-from-cache-4', label: 'Split L2', draw: beatSplit },
  { clip: 'halo-from-cache-5', label: 'Which bar', draw: beatBars },
];

export function HaloFromCacheScene() {
  return <Scene chapter="ch07" title="Where the halo comes from" beats={beats} />;
}
