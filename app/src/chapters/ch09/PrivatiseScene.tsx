import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { sceneData } from '../../sim/histpriv';

// "Privatise, then commit": 24 pixels of our own (already bin numbers), three blocks of
// eight threads, eight bins. Every number drawn comes from sceneData() in
// src/sim/histpriv.ts, which histpriv.test.ts pins together with the narration scripts
// (content/ch09/narration/privatise-N.txt).
const D = sceneData();
const W = 720;
const H = 360;
const PX = 24; // pixel and private-bin cell width
const GAP = 24;
const bx = (k: number) => (W - (D.blocks * D.tpb * PX + (D.blocks - 1) * GAP)) / 2 + k * (D.tpb * PX + GAP);
const STRIP_Y = 44;
const COPY_Y = 128;
const PUB_W = 44;
const PUB_X = (W - D.bins * PUB_W) / 2;
const PUB_Y = 236;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

const pixelX = (i: number) => bx(Math.floor(i / D.tpb)) + (i % D.tpb) * PX + PX / 2;
const copyX = (k: number, b: number) => bx(k) + b * PX + PX / 2;
const pubX = (b: number) => PUB_X + b * PUB_W + PUB_W / 2;

// Commits drawn bin by bin, so the blocks that share a public bin arrive together.
const COMMITS = [...D.commits].sort((a, b) => a.bin - b.bin || a.block - b.block);

interface Frame {
  /** Arrows from pixels to public bins: [pixel, strength 0..1]. */
  toPublic?: [number, number][];
  /** Arrows from pixels to their block's copy. */
  toCopy?: number[];
  /** Arrows from copy bins to public bins. */
  commits?: typeof COMMITS;
  copies?: number; // opacity of the private copies
  copyValues?: (number | '?' | null)[][];
  hotCopy?: { block: number; bin: number } | null;
  barrier?: string | null;
  publicValues?: (number | null)[];
  hotPublic?: number | null;
  counter?: ReactNode;
  counterData?: Record<string, number>;
  bill?: number; // 0..1 growth of the bars
  headline?: string;
}

function Arrow({ x1, y1, x2, y2, tier, strength = 1 }: { x1: number; y1: number; x2: number; y2: number; tier: 'global' | 'shared'; strength?: number }) {
  return (
    <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={`var(--${tier})`} strokeWidth={strength > 0.5 ? 1.6 : 1}
      opacity={0.18 + 0.82 * strength} markerEnd={`url(#pv-head-${tier})`} />
  );
}

function Stage(f: Frame) {
  const pub = f.publicValues ?? new Array<number | null>(D.bins).fill(null);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Animation: private histograms per block, then a commit to the public histogram"
      style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <defs>
        {(['global', 'shared'] as const).map((t) => (
          <marker key={t} id={`pv-head-${t}`} viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0,0 L6,3 L0,6 z" fill={`var(--${t})`} />
          </marker>
        ))}
      </defs>
      {/* the image strip, one thread per pixel */}
      <text x={bx(0)} y={16} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>
        image: {D.pixels.length} pixels as bin numbers, one thread each
      </text>
      {D.pixels.map((v, i) => (
        <g key={i} data-pixel={v}>
          <rect x={pixelX(i) - PX / 2 + 1} y={STRIP_Y} width={PX - 2} height={PX - 2} rx="2" fill="var(--cell)"
            stroke={v === D.busiest ? 'var(--ink)' : 'var(--rule)'} strokeWidth={v === D.busiest ? 1.3 : 1} />
          <text x={pixelX(i)} y={STRIP_Y + 15} fontSize="11.5" textAnchor="middle">{v}</text>
        </g>
      ))}
      {Array.from({ length: D.blocks }, (_, k) => (
        <g key={k}>
          <line x1={bx(k)} y1={STRIP_Y - 4} x2={bx(k) + D.tpb * PX - 2} y2={STRIP_Y - 4} stroke="var(--ink-2)" />
          <text x={bx(k) + (D.tpb * PX) / 2} y={STRIP_Y - 9} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
            block {k}: {D.tpb} threads
          </text>
        </g>
      ))}

      {/* private copies, one per block, in shared memory */}
      {f.copies ? (
        <g opacity={f.copies}>
          {Array.from({ length: D.blocks }, (_, k) => (
            <g key={k} data-copy={k}>
              <text x={bx(k)} y={COPY_Y + PX + 24} fontSize="11" fontWeight="600" style={{ fill: 'var(--shared)' }}>block {k}'s copy (shared)</text>
              {Array.from({ length: D.bins }, (_, b) => {
                const val = f.copyValues?.[k]?.[b] ?? null;
                const hot = f.hotCopy && f.hotCopy.block === k && f.hotCopy.bin === b;
                return (
                  <g key={b} data-copy-bin={`${k}-${b}`} data-value={val ?? ''}>
                    <rect x={copyX(k, b) - PX / 2 + 1} y={COPY_Y} width={PX - 2} height={PX - 2} rx="2"
                      fill={hot ? 'var(--shared)' : val === '?' ? 'var(--cell)' : 'var(--shared-wash)'} stroke="var(--shared)" />
                    <text x={copyX(k, b)} y={COPY_Y + 15} fontSize="11" textAnchor="middle"
                      style={{ fill: hot ? 'var(--page)' : val === '?' ? 'var(--ink-3)' : 'var(--ink)' }}>{val ?? ''}</text>
                    <text x={copyX(k, b)} y={COPY_Y + PX + 10} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{b}</text>
                  </g>
                );
              })}
            </g>
          ))}
        </g>
      ) : null}
      {f.barrier ? (
        <g data-barrier={f.barrier}>
          <rect x={bx(0)} y={COPY_Y + PX + 32} width={bx(D.blocks - 1) + D.tpb * PX - bx(0)} height={20} rx="3" fill="var(--ink)" />
          <text x={W / 2} y={COPY_Y + PX + 46} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>{f.barrier}</text>
        </g>
      ) : null}

      {/* the public histogram in global memory */}
      <text x={PUB_X - 10} y={PUB_Y + 12} fontSize="11" fontWeight="600" textAnchor="end" style={{ fill: 'var(--global)' }}>public bins</text>
      <text x={PUB_X - 10} y={PUB_Y + 25} fontSize="10" textAnchor="end" style={{ fill: 'var(--global)' }}>(global memory)</text>
      {pub.map((v, b) => (
        <g key={b} data-public-bin={b} data-value={v ?? ''}>
          <rect x={PUB_X + b * PUB_W + 1} y={PUB_Y} width={PUB_W - 2} height={28} rx="2"
            fill={f.hotPublic === b ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" />
          <text x={pubX(b)} y={PUB_Y + 18} fontSize="12" textAnchor="middle"
            style={{ fill: f.hotPublic === b ? 'var(--page)' : 'var(--ink)' }}>{v ?? ''}</text>
          <text x={pubX(b)} y={PUB_Y + 40} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>bin {b}</text>
        </g>
      ))}

      {/* arrows */}
      {(f.toPublic ?? []).map(([i, s]) => (
        <Arrow key={'p' + i} x1={pixelX(i)} y1={STRIP_Y + PX - 2} x2={pubX(D.pixels[i])} y2={PUB_Y - 3} tier="global" strength={s} />
      ))}
      {(f.toCopy ?? []).map((i) => (
        <Arrow key={'c' + i} x1={pixelX(i)} y1={STRIP_Y + PX - 2} x2={copyX(Math.floor(i / D.tpb), D.pixels[i])} y2={COPY_Y - 3} tier="shared" />
      ))}
      {(f.commits ?? []).map((c) => (
        <g key={`m${c.block}-${c.bin}`} data-commit={`${c.block}-${c.bin}`}>
          <Arrow x1={copyX(c.block, c.bin)} y1={COPY_Y + PX + 28} x2={pubX(c.bin)} y2={PUB_Y - 3} tier="global" />
        </g>
      ))}

      {f.bill != null ? (
        <g data-bill={`${D.globalBefore}-${D.globalAfter}`}>
          {[
            { label: 'basic kernel', v: D.globalBefore, y: 292 },
            { label: 'privatised', v: D.globalAfter, y: 314 },
          ].map((r) => (
            <g key={r.y}>
              <text x={PUB_X - 8} y={r.y + 10} fontSize="11" textAnchor="end">{r.label}</text>
              <rect x={PUB_X} y={r.y} width={(r.v / D.globalBefore) * D.bins * PUB_W * f.bill!} height={13} fill="var(--global)" />
              <text x={PUB_X + (r.v / D.globalBefore) * D.bins * PUB_W * f.bill! + 6} y={r.y + 11} fontSize="11.5" fontWeight="600">
                {Math.round(r.v * f.bill!)}
              </text>
            </g>
          ))}
        </g>
      ) : null}
      {f.counter ? (
        <text x={W / 2} y={f.bill != null ? 348 : 312} fontSize="12.5" fontWeight="600" textAnchor="middle"
          {...Object.fromEntries(Object.entries(f.counterData ?? {}).map(([k, v]) => [`data-${k}`, v]))}>{f.counter}</text>
      ) : null}
      {f.headline ? (
        <text x={W / 2} y={f.bill != null ? 290 : 340} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{f.headline}</text>
      ) : null}
    </svg>
  );
}

const hotPixels = D.pixels.flatMap((v, i) => (v === D.busiest ? [i] : []));
/** Each block's copy after its first `t` threads have counted. */
const countedAfter = (t: number) =>
  D.perBlock.map((_, k) => {
    const c = new Array<number>(D.bins).fill(0);
    for (let j = 0; j < Math.min(t, D.tpb); j++) c[D.pixels[k * D.tpb + j]] += 1;
    return c;
  });
const blocksOf = (pixels: number[]) => new Set(pixels.map((i) => Math.floor(i / D.tpb))).size;

const beats: Beat[] = [
  {
    clip: 'privatise-1',
    label: 'One public histogram',
    draw: (p) => {
      const shown = Math.round(ramp(p, 0.25, 0.8) * hotPixels.length);
      const hot = hotPixels.slice(0, shown);
      const faint = ramp(p, 0.05, 0.25);
      return (
        <Stage
          toPublic={[
            ...D.pixels.flatMap((v, i) => (v === D.busiest || faint === 0 ? [] : [[i, 0] as [number, number]])),
            ...hot.map((i) => [i, 1] as [number, number]),
          ]}
          publicValues={p >= 0.85 ? D.total : new Array(D.bins).fill(0)}
          hotPublic={shown ? D.busiest : null}
          counter={<>threads aiming at bin {D.busiest}: {shown}{shown ? <> from {blocksOf(hot)} block{blocksOf(hot) === 1 ? '' : 's'}</> : null}</>}
          counterData={{ contenders: shown }}
          headline="every update is a device-scope atomic on the one public copy"
        />
      );
    },
  },
  {
    clip: 'privatise-2',
    label: 'A copy per block',
    draw: (p) => {
      const zeroed = Math.floor(ramp(p, 0.3, 0.7) * D.bins + 1e-9);
      const vals = Array.from({ length: D.blocks }, () =>
        Array.from({ length: D.bins }, (_, b) => (b < zeroed ? 0 : '?')),
      );
      return (
        <Stage
          copies={ramp(p, 0, 0.2)}
          copyValues={vals}
          barrier={p >= 0.75 ? 'barrier 1: nobody counts until every bin of the copy is zero' : null}
          publicValues={new Array(D.bins).fill(0)}
          headline={p < 0.3 ? 'shared memory starts with leftovers (?)' : 'thread t of each block zeroes bin t'}
        />
      );
    },
  },
  {
    clip: 'privatise-3',
    label: 'Count locally',
    draw: (p) => {
      const t = Math.min(D.tpb, Math.floor(ramp(p, 0.05, 0.75) * D.tpb + 1e-9));
      const c = countedAfter(t);
      const best = c.flatMap((row, k) => row.map((v, b) => ({ k, b, v }))).reduce((a, x) => (x.v > a.v ? x : a), { k: 0, b: 0, v: 0 });
      const arrows = D.pixels.flatMap((_, i) => (i % D.tpb < t ? [i] : []));
      return (
        <Stage
          copies={1}
          copyValues={c}
          toCopy={arrows}
          hotCopy={best.v ? { block: best.k, bin: best.b } : null}
          publicValues={new Array(D.bins).fill(0)}
          counter={<>busiest location of any copy: {best.v} contender{best.v === 1 ? '' : 's'}, all in block {best.k} (was {D.publicContenders})</>}
          counterData={{ contenders: best.v }}
          headline="block-scope atomics on each block's own copy"
        />
      );
    },
  },
  {
    clip: 'privatise-4',
    label: 'Barrier, then commit',
    draw: (p) => {
      const n = Math.round(ramp(p, 0.2, 0.9) * COMMITS.length);
      const done = COMMITS.slice(0, n);
      const pub = new Array<number>(D.bins).fill(0);
      for (const c of done) pub[c.bin] += c.count;
      const last = done[done.length - 1];
      const sameBin = last ? done.filter((c) => c.bin === last.bin).length : 0;
      const vals: (number | '?' | null)[][] = D.perBlock.map((row) => [...row]);
      return (
        <Stage
          copies={1}
          copyValues={vals}
          barrier={p < 0.2 ? 'barrier 2: every update of the block has landed' : null}
          commits={done}
          publicValues={pub}
          hotPublic={last ? last.bin : null}
          counter={last
            ? <>committers to public bin {last.bin} so far: {sameBin}, one per block (at most {D.commitContention})</>
            : <>zero bins are skipped: no atomic for them</>}
          counterData={{ commits: done.length }}
          headline="one thread per bin of each block adds its count with a device-scope atomic"
        />
      );
    },
  },
  {
    clip: 'privatise-5',
    label: 'The bill',
    draw: (p) => (
      <Stage
        copies={0.35}
        copyValues={D.perBlock}
        publicValues={D.total}
        bill={ramp(p, 0.05, 0.45)}
        counter={p >= 0.5 ? <>{D.globalBefore} global atomics become {D.globalAfter}, plus {D.blocks} zeroing passes and {D.blocks} commits</> : null}
        counterData={{ before: D.globalBefore, after: D.globalAfter }}
      />
    ),
  },
];

export function PrivatiseScene() {
  return <Scene chapter="ch09" title="Privatise, then commit" beats={beats} />;
}
