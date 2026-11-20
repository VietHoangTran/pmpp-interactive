import { Scene, type Beat } from '../../components/Scene';
import { firstReuse, isScattered, peakLive, simulate, type PagedRun, type SideFrame } from '../../sim/paged';

// "Paging the KV cache" (section 20.7). A small stream from src/sim/paged.ts: a pool of
// 256 token slots, blocks of 32 tokens, max_seq 128, 10 requests, seed 12. Every drawing is
// a frame of that simulation, and src/sim/paged.test.ts pins what the narration says about
// this run. Five beats, one narration clip each (content/ch20/narration/paged-blocks-N.txt).
// The two pool drawings are exported for the stepper in section 20.7.

export const SCENE_CFG = { pool: 256, block: 32, maxSeq: 128, requests: 10, seed: 12 };
export const SCENE_RUN = simulate(SCENE_CFG);
/** The first frame in which, on the paged side, a request finishes and a waiting one takes its blocks. */
export const REUSE_FRAME = firstReuse(SCENE_RUN);

const W = 720;
const H = 340;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

/** Contiguous slots of max_seq tokens: the used part solid, the rest of the reservation washed. */
export function ContiguousPool({ run, side, x, y, w, h }: { run: PagedRun; side: SideFrame; x: number; y: number; w: number; h: number }) {
  const rowH = Math.min(22, h / Math.max(1, run.slots));
  const maxSeq = run.cfg.maxSeq;
  const leftover = run.cfg.pool - run.slots * maxSeq;
  return (
    <g data-side="contiguous" data-reserved={side.reserved} data-used={side.used} data-wasted={side.wasted} data-live={side.live.length}>
      {Array.from({ length: run.slots }, (_, s) => {
        const r = side.live.find((q) => q.slot === s);
        const yy = y + s * rowH;
        if (!r)
          return <rect key={s} x={x} y={yy} width={w} height={rowH - 3} fill="none" stroke="var(--rule)" strokeDasharray="3 3" />;
        return (
          <g key={s}>
            <rect x={x} y={yy} width={w} height={rowH - 3} fill="var(--global-wash)" stroke="var(--global)" strokeWidth="0.8" />
            <rect x={x} y={yy} width={(r.len / maxSeq) * w} height={rowH - 3} fill="var(--global)" />
            {rowH > 11 ? (
              <text x={x + w - 4} y={yy + rowH - 7} fontSize={Math.min(10.5, rowH - 5)} textAnchor="end">r{r.id}: {r.len} of {maxSeq} used</text>
            ) : null}
          </g>
        );
      })}
      {leftover > 0 ? (
        <text x={x} y={y + run.slots * rowH + 10} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>{leftover} tokens left over: too few for another slot</text>
      ) : null}
    </g>
  );
}

/** Paged blocks: each block shows its owner and how full it is; free blocks are dashed. */
export function PagedPool({ run, side, x, y, w, h, hot }: { run: PagedRun; side: SideFrame; x: number; y: number; w: number; h: number; hot?: number }) {
  const B = run.cfg.block;
  const nb = run.blocks;
  const cols = Math.min(16, nb);
  const rows = Math.ceil(nb / cols);
  const cw = w / cols;
  const ch = Math.min(28, h / rows);
  const owner = new Map<number, { id: number; fill: number }>();
  for (const r of side.live) r.blocks!.forEach((b, i) => owner.set(b, { id: r.id, fill: Math.min(B, r.len - i * B) / B }));
  return (
    <g data-side="paged" data-reserved={side.reserved} data-used={side.used} data-wasted={side.wasted} data-live={side.live.length} data-free={side.free.length}>
      {Array.from({ length: nb }, (_, b) => {
        const bx = x + (b % cols) * cw;
        const by = y + Math.floor(b / cols) * ch;
        const o = owner.get(b);
        if (!o)
          return <rect key={b} x={bx + 1} y={by + 1} width={cw - 2} height={ch - 2} fill="none" stroke="var(--rule)" strokeDasharray="3 3" />;
        const isHot = hot != null && o.id === hot;
        return (
          <g key={b} data-block={b} data-owner={o.id}>
            <rect x={bx + 1} y={by + 1} width={cw - 2} height={ch - 2} fill="var(--global-wash)" stroke={isHot ? 'var(--ink)' : 'var(--global)'} strokeWidth={isHot ? 2 : 0.8} />
            <rect x={bx + 1} y={by + 1} width={(cw - 2) * o.fill} height={ch - 2} fill="var(--global)" />
            {ch > 12 && cw > 18 ? (
              <text x={bx + cw / 2} y={by + ch / 2 + 3.5} fontSize={Math.min(10, ch - 6)} textAnchor="middle"
                style={{ fill: o.fill > 0.55 ? 'var(--page)' : 'var(--ink)' }}>r{o.id}</text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

function Counters({ side, x, y, label }: { side: SideFrame; x: number; y: number; label: string }) {
  return (
    <text x={x} y={y} fontSize="10.5">
      <tspan fontWeight="600">{label}</tspan>
      <tspan>{`  reserved ${side.reserved}, used ${side.used}, wasted ${side.wasted} · running ${side.live.length}, waiting ${side.waiting.length}, done ${side.done}`}</tspan>
    </text>
  );
}

function Stage({ f, headline, show, hot, tables }: { f: number; headline: string; show: 'contiguous' | 'paged' | 'both'; hot?: number; tables?: boolean }) {
  const frame = SCENE_RUN.frames[Math.min(f, SCENE_RUN.frames.length - 1)];
  const dimC = show === 'paged' ? 0.3 : 1;
  const dimP = show === 'contiguous' ? 0.3 : 1;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Contiguous and paged KV-cache pools" style={{ width: '100%', maxWidth: W, display: 'block' }} data-frame={frame.step}>
      <text x={12} y={24} fontSize="14.5" fontWeight="600" data-headline="">{headline}</text>
      <text x={W - 12} y={24} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>step {frame.step}</text>
      <g opacity={dimC}>
        <text x={12} y={52} fontSize="12" fontWeight="600">contiguous: a max_seq slot per request</text>
        <ContiguousPool run={SCENE_RUN} side={frame.contiguous} x={12} y={60} w={420} h={50} />
        <Counters side={frame.contiguous} x={12} y={124} label="contiguous" />
      </g>
      <g opacity={dimP}>
        <text x={12} y={156} fontSize="12" fontWeight="600">paged: blocks of {SCENE_CFG.block} tokens from a free list</text>
        <PagedPool run={SCENE_RUN} side={frame.paged} x={12} y={164} w={420} h={60} hot={hot} />
        <Counters side={frame.paged} x={12} y={242} label="paged" />
      </g>
      {tables ? (
        <g data-tables="">
          <text x={460} y={156} fontSize="12" fontWeight="600">block tables</text>
          {frame.paged.live.slice(0, 9).map((r, i) => (
            <text key={r.id} x={460} y={174 + i * 15} fontSize="11" fontWeight={r.id === hot ? 700 : 400}>
              r{r.id}: [{r.blocks!.join(', ')}]
            </text>
          ))}
        </g>
      ) : null}
      <text x={460} y={52} fontSize="12" fontWeight="600">queue</text>
      <text x={460} y={70} fontSize="11">contiguous: {frame.contiguous.waiting.map((id) => 'r' + id).join(' ') || 'empty'}</text>
      <text x={460} y={86} fontSize="11">paged: {frame.paged.waiting.map((id) => 'r' + id).join(' ') || 'empty'}</text>
      {frame.paged.events.length || frame.contiguous.events.length ? (
        <text x={12} y={300} fontSize="11.5" data-events="">
          this step, paged: {frame.paged.events.join(', ') || 'nothing'}; contiguous: {frame.contiguous.events.join(', ') || 'nothing'}
        </text>
      ) : null}
    </svg>
  );
}

/** The last beat: wasted token slots over the whole run, both sides, from every frame. */
function Summary({ p }: { p: number }) {
  const frames = SCENE_RUN.frames;
  const end = Math.max(2, Math.round(ramp(p, 0.05, 0.7) * (frames.length - 1)));
  const x0 = 60;
  const x1 = 690;
  const y0 = 226;
  const y1 = 56;
  const X = (t: number) => x0 + (t / (frames.length - 1)) * (x1 - x0);
  const Y = (v: number) => y0 - (v / SCENE_CFG.pool) * (y0 - y1);
  const line = (pick: (f: (typeof frames)[number]) => number) =>
    frames.slice(0, end + 1).map((f, t) => `${X(t).toFixed(1)},${Y(pick(f)).toFixed(1)}`).join(' ');
  const peak = peakLive(SCENE_RUN);
  const { contiguous: fc, paged: fp } = SCENE_RUN.finishedAt;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Wasted cache slots over time, contiguous and paged" style={{ width: '100%', maxWidth: W, display: 'block' }}
      data-finished-contiguous={fc ?? ''} data-finished-paged={fp ?? ''} data-peak-contiguous={peak.contiguous} data-peak-paged={peak.paged}>
      <text x={12} y={24} fontSize="14.5" fontWeight="600" data-headline="">Wasted slots over the run: reservations against blocks</text>
      <rect x={x0} y={y1} width={x1 - x0} height={y0 - y1} fill="var(--page)" stroke="var(--rule)" />
      {[0, 0.5, 1].map((k) => (
        <g key={k}>
          <line x1={x0} x2={x1} y1={Y(k * SCENE_CFG.pool)} y2={Y(k * SCENE_CFG.pool)} stroke="var(--rule-soft)" />
          <text x={x0 - 6} y={Y(k * SCENE_CFG.pool) + 4} fontSize="10" textAnchor="end">{k * SCENE_CFG.pool}</text>
        </g>
      ))}
      <text x={(x0 + x1) / 2} y={y0 + 18} fontSize="11" textAnchor="middle">step</text>
      <text x={x0 - 40} y={y1 - 10} fontSize="11">wasted slots</text>
      <polyline points={line((f) => f.contiguous.wasted)} fill="none" stroke="var(--global)" strokeWidth="2.2" />
      <polyline points={line((f) => f.paged.wasted)} fill="none" stroke="var(--ink)" strokeWidth="2.2" strokeDasharray="5 3" />
      <line x1={x0} x2={x0 + 28} y1={y0 + 40} y2={y0 + 40} stroke="var(--global)" strokeWidth="2.2" />
      <text x={x0 + 36} y={y0 + 44} fontSize="11">contiguous: reserved at max_seq, so most of each slot is wasted</text>
      <line x1={x0} x2={x0 + 28} y1={y0 + 58} y2={y0 + 58} stroke="var(--ink)" strokeWidth="2.2" strokeDasharray="5 3" />
      <text x={x0 + 36} y={y0 + 62} fontSize="11">paged: never more than {SCENE_CFG.block - 1} slots per running request</text>
      <text x={12} y={318} fontSize="11.5">
        all {SCENE_CFG.requests} requests done: contiguous after {fc} steps, paged after {fp}; at most {peak.contiguous} running at once against {peak.paged}
      </text>
    </svg>
  );
}

const reuse = REUSE_FRAME;
/** The request to outline in a frame: the first live one whose blocks are scattered, else the first. */
const hotIn = (f: number) => {
  const live = SCENE_RUN.frames[f].paged.live;
  return (live.find((q) => isScattered(q.blocks!)) ?? live[0])?.id;
};
const GROW_END = Math.min(SCENE_RUN.frames.length - 1, reuse + 26);
const beats: Beat[] = [
  {
    clip: 'paged-blocks-1',
    label: 'Contiguous',
    draw: (p) => <Stage f={Math.round(ramp(p, 0.05, 0.9) * (reuse + 2))} headline="Contiguous: each request reserves max_seq on arrival" show="contiguous" />,
  },
  {
    clip: 'paged-blocks-2',
    label: 'Blocks',
    draw: (p) => {
      const f = Math.max(1, Math.round(ramp(p, 0.1, 0.8) * 3));
      return <Stage f={f} headline="Paged: fixed-size blocks and a block table per request" show="paged" hot={hotIn(f)} tables />;
    },
  },
  {
    clip: 'paged-blocks-3',
    label: 'Finish and reuse',
    draw: (p) => <Stage f={p < 0.35 ? reuse - 1 : reuse} headline={p < 0.35 ? 'A request is about to finish…' : '…its block returns to the free list and a waiting request starts'} show="both" tables />,
  },
  {
    clip: 'paged-blocks-4',
    label: 'Growing',
    draw: (p) => {
      const f = reuse + 1 + Math.round(ramp(p, 0.05, 0.95) * (GROW_END - reuse - 1));
      return <Stage f={f} headline="New blocks come from the free list; the tables keep the order" show="paged" hot={hotIn(f)} tables />;
    },
  },
  { clip: 'paged-blocks-5', label: 'Side by side', draw: (p) => <Summary p={p} /> },
];

export function PagedBlocksScene() {
  return <Scene chapter="ch20" title="Paging the KV cache" beats={beats} />;
}
