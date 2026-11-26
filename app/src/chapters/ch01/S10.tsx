import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { useLanguage } from '../../lib/language';
import {
  design,
  race,
  regions,
  sweep,
  timeline,
  waves,
  type Design,
  type Finish,
  type Region,
  type SweepPoint,
} from '../../sim/designs';
import { BUDGET, Plan, TwoDesignsScene } from './TwoDesignsScene';
import { useWidth } from './useWidth';

// ---------------------------------------------------------------- helpers
const nf = (v: number) => v.toLocaleString('en-US', { maximumFractionDigits: 3 });
const plural = (n: number, one: string) => `${nf(n)} ${one}${n === 1 ? '' : 's'}`;
const MAX_CHAINS = 4096;
const MAX_LENGTH = 1000;
const HALVINGS = [0, 1, 2, 3];
/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const describe = (d: Design, vi = false) =>
  vi ? `${nf(d.units)} khối, tốc độ ${d.speed}` : `${plural(d.units, 'unit')}, speed ${d.speed}`;

// The pair the prediction and the default explorer use: the animation's two designs.
const FAST = design(BUDGET, 2);
const MANY = design(BUDGET, 0);

// q-chainlen: two jobs where the one with more arithmetic has fewer independent pieces.
// Both fit the explorer's ranges, so the reader can check them there.
export const LONG_FEW = { chains: 8, length: 1000 };
export const SHORT_MANY = { chains: 1024, length: 5 };
const longFew = race(FAST, MANY, LONG_FEW.chains, LONG_FEW.length);
const shortMany = race(FAST, MANY, SHORT_MANY.chains, SHORT_MANY.length);
const ops = (j: { chains: number; length: number }) => nf(j.chains * j.length);

// ---------------------------------------------------------------- Fig. 1.1, redrawn
/**
 * Our own drawing of the idea in the book's Fig. 1.1: the same chip area divided
 * two ways. Schematic only. The unit counts are the two designs used on this page.
 */
function AreaFigure() {
  const { language, t } = useLanguage();
  const vi = language === 'vi';
  const W = 640;
  const chipW = 286;
  const H = 262;
  // Below this width the two chips are drawn one above the other, each at its own size,
  // so that the labels stay legible on a phone.
  const [ref, measured] = useWidth(W);
  const stacked = measured < 600;
  const top = 34;
  const chip = (x: number, title: string, bands: [string, number][], d: Design, channels: number, channelNote: string) => {
    let y = top;
    const rows = d.units > 8 ? 4 : 1;
    const cols = d.units / rows;
    const out = bands.map(([label, h], i) => {
      const y0 = y;
      y += h;
      const arithmetic = i === bands.length - 1;
      return (
        <g key={label}>
          <rect x={x} y={y0} width={chipW} height={h} fill={arithmetic ? 'var(--bench)' : 'var(--cell)'} stroke="var(--ink)" strokeWidth="1" />
          {arithmetic ? (
            Array.from({ length: d.units }, (_, k) => (
              <rect key={k} x={x + 5 + (k % cols) * ((chipW - 10) / cols) + 1.5} y={y0 + 5 + Math.floor(k / cols) * ((h - 10) / rows) + 1.5}
                width={(chipW - 10) / cols - 3} height={(h - 10) / rows - 3} rx="1.5" fill="var(--page)" stroke="var(--ink)" strokeWidth={rows > 1 ? 0.9 : 1.5} />
            ))
          ) : (
            <text x={x + chipW / 2} y={y0 + h / 2 + 4} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{label}</text>
          )}
        </g>
      );
    });
    const bottom = y;
    return (
      <g>
        <text x={x} y={14} fontSize="13" fontWeight="600" style={{ fill: 'var(--ink)' }}>{title}</text>
        <text x={x} y={28} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{bands[bands.length - 1][0]}</text>
        {out}
        {Array.from({ length: channels }, (_, k) => {
          const cx = x + ((k + 0.5) * chipW) / channels;
          return <line key={k} x1={cx} x2={cx} y1={bottom} y2={bottom + 22} stroke="var(--ink)" strokeWidth={channels > 4 ? 2 : 5} />;
        })}
        <rect x={x} y={bottom + 22} width={chipW} height={22} fill="var(--cell)" stroke="var(--ink)" />
        <text x={x + chipW / 2} y={bottom + 37} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{t('memory', 'Bộ nhớ')}</text>
        <text x={x} y={bottom + 60} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{channelNote}</text>
      </g>
    );
  };
  const cpu = (x: number) =>
    chip(x, t('Latency-oriented (a CPU)', 'Ưu tiên độ trễ thấp (CPU)'),
      [[t('control logic and branch prediction', 'Logic điều khiển và dự đoán nhánh'), 48], [t('large caches', 'Bộ nhớ đệm lớn'), 58], [`${t('arithmetic:', 'Tính toán ·')} ${vi ? describe(FAST, vi) : describe(FAST).replace(', speed', ' of speed')}`, 54]], FAST, 2, t('few channels to memory', 'Ít kênh bộ nhớ'));
  const gpu = (x: number) =>
    chip(x, t('Throughput-oriented (a GPU)', 'Ưu tiên thông lượng (GPU)'),
      [[t('control', 'Điều khiển'), 18], [t('small caches', 'Bộ nhớ đệm nhỏ'), 18], [`${t('arithmetic:', 'Tính toán ·')} ${vi ? describe(MANY, vi) : describe(MANY).replace(', speed', ' of speed')}`, 124]], MANY, 8, t('many channels to memory, each allowed to be slow', 'Nhiều kênh bộ nhớ, chấp nhận độ trễ cao'));
  const svgStyle = { width: '100%', display: 'block', fontFamily: 'var(--sans)' } as const;
  return (
    <figure style={{ margin: '0.2rem 0 1.4rem' }} data-stacked={stacked ? '1' : '0'}>
      <div ref={ref} style={{ width: '100%' }}>
        {stacked ? (
          <div style={{ display: 'grid', gap: '1rem' }}>
            <svg viewBox={`-2 0 ${chipW + 4} ${H}`} role="img" aria-label={t('Chip area of a latency-oriented design', 'Diện tích chip của thiết kế ưu tiên độ trễ thấp')} style={{ ...svgStyle, maxWidth: 360 }}>
              {cpu(0)}
            </svg>
            <svg viewBox={`-2 0 ${chipW + 4} ${H}`} role="img" aria-label={t('Chip area of a throughput-oriented design', 'Diện tích chip của thiết kế ưu tiên thông lượng')} style={{ ...svgStyle, maxWidth: 360 }}>
              {gpu(0)}
            </svg>
          </div>
        ) : (
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={t('The same chip area divided two ways', 'Hai cách phân bổ cùng một diện tích chip')} style={{ ...svgStyle, maxWidth: W }}>
            {cpu(0)}
            {gpu(W - chipW)}
          </svg>
        )}
      </div>
      <figcaption className="ui muted" style={{ fontSize: '0.8rem', marginTop: '0.4rem' }}>
        {t(
          "Where the area goes, redrawn from the idea of the book's Fig. 1.1. A schematic: the proportions are not measurements, and the unit counts are the two designs of the animation below.",
          'Sơ đồ minh họa cách phân bổ diện tích chip, vẽ lại theo ý tưởng của hình 1.1. Tỷ lệ chỉ để minh họa. Số khối tương ứng với hai thiết kế trong hoạt ảnh bên dưới.',
        )}
      </figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------- explorer drawings
/** Which unit runs which chain: one row per unit, time across, on a scale shared by both designs. */
function Strip({ d, chains, length, tMax, f }: { d: Design; chains: number; length: number; tMax: number; f: Finish }) {
  const { t } = useLanguage();
  const W = 300;
  const top = 8;
  const hgt = 100;
  const x0 = 2;
  const w = W - 4;
  const X = (t: number) => x0 + (t / tMax) * w;
  const row = hgt / d.units;
  const shade = (wave: number) => (wave % 2 ? 'var(--ink-3)' : 'var(--ink-2)');
  const ws = waves(d, chains, length);
  let marks;
  if (chains <= 512) {
    // few enough to draw every chain
    marks = timeline(d, chains, length).map((s) => (
      <rect key={s.chain} x={X(s.start)} y={top + s.unit * row} width={Math.max(0.8, X(s.end) - X(s.start) - (ws.length > 1 ? 0.8 : 0))}
        height={Math.max(0.8, row - (row > 3 ? 1 : 0.4))} fill={shade(s.wave)} />
    ));
  } else if (ws.length <= 400) {
    // one block per wave
    marks = ws.map((v) => (
      <rect key={v.wave} x={X(v.start)} y={top} width={Math.max(0.6, X(v.end) - X(v.start) - 0.5)} height={(v.units / d.units) * hgt} fill={shade(v.wave)} />
    ));
  } else {
    // too many waves to tell apart: the full ones as one block, then the last
    const lastWave = ws[ws.length - 1];
    marks = [
      <rect key="full" x={X(0)} y={top} width={X(lastWave.start) - X(0)} height={hgt} fill="var(--ink-2)" />,
      <rect key="last" x={X(lastWave.start)} y={top} width={Math.max(0.6, X(lastWave.end) - X(lastWave.start))} height={(lastWave.units / d.units) * hgt} fill="var(--ink-3)" />,
    ];
  }
  const endRight = X(f.last) > W - 130;
  return (
    <svg viewBox={`0 0 ${W} 132`} role="img" aria-label={t(`Which of the ${d.units} units runs which chain, over time`, `Tiến trình xử lý các chuỗi trên ${d.units} khối`)} style={{ width: '100%', maxWidth: 440, display: 'block' }}>
      <rect x={x0} y={top} width={w} height={hgt} fill="var(--cell)" stroke="var(--rule)" />
      {marks}
      <line x1={X(f.first)} x2={X(f.first)} y1={top - 5} y2={top + hgt + 5} stroke="var(--ink)" strokeWidth="1" strokeDasharray="3 3" />
      <line x1={X(f.last)} x2={X(f.last)} y1={top - 5} y2={top + hgt + 5} stroke="var(--ink)" strokeWidth="1.8" />
      <text x={x0} y={top + hgt + 20} fontSize="11" style={{ fill: 'var(--ink-2)' }}>0</text>
      <text x={endRight ? X(f.last) : X(f.last) + 5} y={top + hgt + 20} fontSize="11" textAnchor={endRight ? 'end' : 'start'}>
        {t('last result at', 'Kết quả cuối ở mốc')} {nf(f.last)}
      </text>
    </svg>
  );
}

/** Finishing time of both designs against the number of chains, log scales, with the current point marked. */
function CrossPlot({ pts, regs, chains, a, b }: { pts: SweepPoint[]; regs: Region[]; chains: number; a: Design; b: Design }) {
  const { language, t } = useLanguage();
  const vi = language === 'vi';
  // Drawn at its measured width, so that its text keeps its size on a phone.
  const [ref, W] = useWidth(560);
  const narrow = W < 440;
  // On a wide plot, room on the right for the label of the last point and the legend inside the frame.
  // On a narrow one the label turns left and the legend moves under the axis, where no line runs.
  const H = Math.round(Math.min(330, Math.max(250, W * 0.59))) + (narrow ? 40 : 0);
  const pad = { l: 58, r: narrow ? 26 : 70, t: 14, b: narrow ? 84 : 44 };
  const lo = Math.min(pts[0].a, pts[0].b) / 1.7;
  const hi = Math.max(pts[pts.length - 1].a, pts[pts.length - 1].b) * 1.7;
  const lx = (v: number) => Math.log2(v) / Math.log2(MAX_CHAINS);
  const X = (v: number) => pad.l + lx(v) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (H - pad.t - pad.b);
  // A staircase has long flat runs: keep only the points where a value changes.
  const line = (key: 'a' | 'b') =>
    pts
      .filter((q, i) => i === 0 || i === pts.length - 1 || pts[i - 1][key] !== q[key] || pts[i + 1][key] !== q[key])
      .map((q) => `${X(q.chains).toFixed(1)},${Y(q[key]).toFixed(1)}`)
      .join(' ');
  const xt = [1, 4, 16, 64, 256, 1024, 4096];
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) yt.push(10 ** e);
  const cur = pts[chains - 1];
  const same = cur.a === cur.b;
  const aAbove = cur.a > cur.b;
  // The slower design's label goes above its line and to the left, the faster one's below and to the right,
  // where neither staircase runs.
  const cx = X(chains);
  const upper = cx < pad.l + 84 ? { dx: 9, anchor: 'start' as const } : { dx: -9, anchor: 'end' as const };
  const lower = cx > W - 90 ? { dx: -9, dy: 18, anchor: 'end' as const } : { dx: 9, dy: 18, anchor: 'start' as const };
  const lowest = Math.max(Y(cur.a), Y(cur.b));
  const lg = narrow ? { x: 20, y: H - pad.b + 56 } : { x: W - pad.r - 176, y: H - pad.b - 34 };
  return (
    <div ref={ref} style={{ width: '100%', overflow: 'hidden' }}>
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={t('Time to the last result against the number of independent chains', 'Thời gian đến kết quả cuối theo số chuỗi độc lập')} style={{ display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {regs.filter((g) => g.winner === 'tie' && regs.length > 1).map((g) => (
        <rect key={g.from} x={X(g.from) - 1} y={pad.t} width={X(g.to) - X(g.from) + 2} height={H - pad.t - pad.b} fill="var(--rule-soft)" stroke="var(--rule)" />
      ))}
      {xt.map((v) => (
        <g key={'x' + v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle">{nf(v)}</text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{nf(v)}</text>
        </g>
      ))}
      <text x={(W - pad.r + pad.l) / 2} y={H - pad.b + 36} fontSize="11.5" textAnchor="middle">{t('independent chains', 'Số chuỗi độc lập')}</text>
      <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`}>{t('time to the last result', 'Mốc có kết quả cuối')}</text>
      <polyline points={line('a')} fill="none" stroke="var(--ink)" strokeWidth="2.2" strokeLinejoin="round" />
      <polyline points={line('b')} fill="none" stroke="var(--ink)" strokeWidth="2.2" strokeDasharray="6 4" strokeLinejoin="round" />
      {/* legend */}
      <line x1={lg.x} x2={lg.x + 28} y1={lg.y} y2={lg.y} stroke="var(--ink)" strokeWidth="2.2" />
      <text x={lg.x + 34} y={lg.y + 4} fontSize="11.5">A: {describe(a, vi)}</text>
      <line x1={lg.x} x2={lg.x + 28} y1={lg.y + 18} y2={lg.y + 18} stroke="var(--ink)" strokeWidth="2.2" strokeDasharray="6 4" />
      <text x={lg.x + 34} y={lg.y + 22} fontSize="11.5">B: {describe(b, vi)}</text>
      {/* the current point */}
      <line x1={cx} x2={cx} y1={pad.t} y2={lowest} stroke="var(--ink)" strokeDasharray="3 3" />
      <circle cx={cx} cy={Y(cur.a)} r="4.5" fill="var(--ink)" />
      <circle cx={cx} cy={Y(cur.b)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
      {same ? (
        <text x={cx + upper.dx} y={Y(cur.a) - 9} fontSize="11.5" fontWeight="600" textAnchor={upper.anchor}>{t('A and B:', 'A và B ·')} {nf(cur.a)}</text>
      ) : (
        <>
          <text x={cx + (aAbove ? upper.dx : lower.dx)} y={Y(cur.a) + (aAbove ? -9 : lower.dy)} fontSize="11.5" fontWeight="600"
            textAnchor={aAbove ? upper.anchor : lower.anchor}>A: {nf(cur.a)}</text>
          <text x={cx + (aAbove ? lower.dx : upper.dx)} y={Y(cur.b) + (aAbove ? lower.dy : -9)} fontSize="11.5" fontWeight="600"
            textAnchor={aAbove ? lower.anchor : upper.anchor}>B: {nf(cur.b)}</text>
        </>
      )}
    </svg>
    </div>
  );
}

const leads = (regs: Region[], vi: boolean) =>
  regs
    .map((g) => {
      const span = g.from === g.to ? `at ${nf(g.from)}` : `from ${nf(g.from)} to ${nf(g.to)}`;
      if (vi) {
        const count = g.from === g.to ? nf(g.from) : `${nf(g.from)} đến ${nf(g.to)}`;
        return g.winner === 'tie' ? `hòa ở ${count} chuỗi` : `${g.winner.toUpperCase()} dẫn đầu ở ${count} chuỗi`;
      }
      return g.winner === 'tie' ? `a tie ${span}` : `${g.winner === 'a' ? 'A' : 'B'} leads ${span}`;
    })
    .join(', ');

function BudgetBench() {
  const { language, t } = useLanguage();
  const vi = language === 'vi';
  const [ha, setHa] = useState(2);
  const [hb, setHb] = useState(0);
  const [chains, setChains] = useState(1);
  const [length, setLength] = useState(100);
  const a = design(BUDGET, ha);
  const b = design(BUDGET, hb);
  const r = race(a, b, chains, length);
  const pts = useMemo(() => sweep(design(BUDGET, ha), design(BUDGET, hb), MAX_CHAINS, length), [ha, hb, length]);
  const regs = useMemo(() => regions(pts), [pts]);
  const tMax = Math.max(r.a.last, r.b.last);
  const clampChains = (v: number) => Math.max(1, Math.min(MAX_CHAINS, Math.round(v) || 1));

  const pick = (label: string, h: number, set: (h: number) => void) => (
    <span style={FIELD}>
      {t(`${label}: halve the latency of a unit ${h === 1 ? 'once' : `${h} times`}`, `${label} · Giảm một nửa độ trễ mỗi khối ${h} lần`)}
      <span className="seg" role="group" aria-label={label}>
        {HALVINGS.map((k) => (
          <button key={k} aria-pressed={h === k} onClick={() => set(k)}>
            {describe(design(BUDGET, k), vi)}
          </button>
        ))}
      </span>
    </span>
  );
  const panel = (key: 'a' | 'b', d: Design, f: Finish) => (
    <div className="panel" data-design={key}>
      <p style={{ margin: '0 0 0.5rem' }}>
        <b>{t('Design', 'Thiết kế')} {key.toUpperCase()}</b>: {describe(d, vi)}.{' '}
        {t(`Each unit draws ${d.unitPower} of the ${BUDGET}.`, `Mỗi khối dùng ${d.unitPower} trên ${BUDGET} đơn vị công suất.`)}
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem 0.9rem', alignItems: 'flex-start' }}>
        <svg viewBox="0 0 120 120" role="img" aria-label={t(`Floor plan: ${describe(d)}`, `Sơ đồ phân bổ: ${describe(d, vi)}`)} style={{ width: 120, flex: '0 0 auto', display: 'block' }}>
          <Plan x={0} y={0} size={120} d={d} busy={new Set(Array.from({ length: d.units - f.idle }, (_, i) => i))} />
        </svg>
        <div style={{ flex: '1 1 220px', minWidth: 0 }}>
          <Strip d={d} chains={chains} length={length} tMax={tMax} f={f} />
        </div>
      </div>
      <p className="muted" style={{ margin: '0.4rem 0 0' }} data-out={key + '-used'}>
        {f.idle === 0
          ? t(`Every unit gets a chain. ${plural(f.waves, 'wave')}.`, `Mọi khối đều có việc. Cần ${nf(f.waves)} đợt.`)
          : t(`${plural(d.units - f.idle, 'unit')} working, ${nf(f.idle)} never used. ${plural(f.waves, 'wave')}.`, `${nf(d.units - f.idle)} khối có việc, ${nf(f.idle)} khối không được dùng. Cần ${nf(f.waves)} đợt.`)}
      </p>
    </div>
  );
  const stat = (label: string, key: string, va: number | string, vb: number | string) => (
    <tr>
      <td>{label}</td>
      <td className="num" data-stat={'a-' + key}>{typeof va === 'number' ? nf(va) : va}</td>
      <td className="num" data-stat={'b-' + key}>{typeof vb === 'number' ? nf(vb) : vb}</td>
    </tr>
  );
  return (
    <Bench
      title={t('Spend a power budget', 'Phân bổ ngân sách công suất')}
      task={t('Find the number of chains at which the lead changes hands. Then change the chain length and see what moves.', 'Tìm số chuỗi tại đó thiết kế dẫn đầu thay đổi. Sau đó đổi độ dài chuỗi và quan sát kết quả.')}
      foot={
        <>
          {t(
            `Time is counted in operations of a plain unit (speed 1). In the strips, one row is one unit, alternate waves alternate in shade, the dashed line is the first result and the solid line the last. Both strips share one time scale. The budget of ${BUDGET} is an abstract number, not a device.`,
            `Đơn vị thời gian là thời gian để một khối thường (tốc độ 1) thực hiện một phép toán. Mỗi hàng là một khối, các đợt liên tiếp có màu đậm nhạt xen kẽ. Nét đứt đánh dấu kết quả đầu tiên, nét liền đánh dấu kết quả cuối. Hai biểu đồ dùng chung thang thời gian. Ngân sách ${BUDGET} là giá trị giả định của mô hình.`,
          )}
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        {pick(t('Design A', 'Thiết kế A'), ha, setHa)}
        {pick(t('Design B', 'Thiết kế B'), hb, setHb)}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field">
          {t('Independent chains =', 'Số chuỗi độc lập =')} {nf(chains)}
          <input type="range" min="0" max="120" value={Math.round(10 * Math.log2(chains))} aria-label={t('Independent chains (log scale)', 'Số chuỗi độc lập (thang logarit)')}
            onChange={(e) => setChains(clampChains(2 ** (+e.target.value / 10)))} />
        </label>
        <label className="field">
          {t('exact number', 'Giá trị chính xác')}
          <input type="number" min="1" max={MAX_CHAINS} step="1" value={chains} aria-label={t('Independent chains (exact)', 'Số chuỗi độc lập (giá trị chính xác)')}
            onChange={(e) => setChains(clampChains(+e.target.value))} />
        </label>
        <label className="field">
          {t('Operations in a chain =', 'Số phép toán mỗi chuỗi =')} {nf(length)}
          <input type="range" min="1" max={MAX_LENGTH} value={length} aria-label={t('Operations in a chain', 'Số phép toán trong mỗi chuỗi')} onChange={(e) => setLength(+e.target.value)} />
        </label>
      </div>
      <div className="cols" style={{ marginBottom: '1.2rem' }}>
        {panel('a', a, r.a)}
        {panel('b', b, r.b)}
      </div>
      <div className="cols plot">
        <CrossPlot pts={pts} regs={regs} chains={chains} a={a} b={b} />
        <div className="panel">
          <p style={{ fontSize: '1.15rem', fontWeight: 600, margin: '0 0 0.2rem' }} data-out="verdict">
            {r.winner === 'tie'
              ? t(`A tie: both finish at ${nf(r.a.last)}.`, `Hòa. Cả hai hoàn thành ở mốc ${nf(r.a.last)}.`)
              : t(`Design ${r.winner.toUpperCase()} finishes first, ${nf(r.factor)} times sooner.`, `Thiết kế ${r.winner.toUpperCase()} về trước, nhanh gấp ${nf(r.factor)} lần.`)}
          </p>
          <p className="muted" style={{ margin: '0 0 0.7rem' }}>
            {t(`${plural(chains, 'chain')} of ${plural(length, 'operation')}.`, `${nf(chains)} chuỗi, mỗi chuỗi gồm ${nf(length)} phép toán.`)}
          </p>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th />
                <th className="num">A</th>
                <th className="num">B</th>
              </tr>
            </thead>
            <tbody>
              {stat(t('Units × speed of each', 'Số khối × tốc độ mỗi khối'), 'shape', `${a.units} × ${a.speed}`, `${b.units} × ${b.speed}`)}
              {stat(t('Peak rate, every unit busy', 'Thông lượng đỉnh khi mọi khối đều bận'), 'peak', a.throughput, b.throughput)}
              {stat(t('First result at', 'Mốc có kết quả đầu tiên'), 'first', r.a.first, r.b.first)}
              {stat(t('Last result at', 'Mốc có kết quả cuối cùng'), 'last', r.a.last, r.b.last)}
              {stat(t('Rate delivered on this work', 'Thông lượng thực tế'), 'delivered', r.a.delivered, r.b.delivered)}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.5rem 0 0' }}>{t('Rates are operations per time unit.', 'Thông lượng tính bằng số phép toán trên mỗi đơn vị thời gian.')}</p>
          <p style={{ margin: '0.7rem 0 0' }} data-out="regions">
            {regs.length === 1
              ? t('These two designs are the same, so every number of chains is a tie.', 'Hai thiết kế giống nhau nên luôn hòa, với mọi số chuỗi.')
              : t(`For these two designs, at any chain length: ${leads(regs, false)} chains, the most this explorer runs.`, `Ở mọi độ dài chuỗi, trong phạm vi mô phỏng: ${leads(regs, true)}.`)}
          </p>
        </div>
      </div>
    </Bench>
  );
}

function LessonText({ children, vi }: { children: ReactNode; vi: ReactNode }) {
  return useLanguage().language === 'vi' ? vi : children;
}

// ---------------------------------------------------------------- questions
export const questions10: Record<string, Question> = {
  'ch01/start/q-batch': {
    kind: 'choice',
    id: 'ch01/start/q-batch',
    prompt: (
      <>
        <LessonText vi={<>
          Một máy chủ GPU chạy mô hình cho từng yêu cầu một (kích thước lô bằng 1). Bạn chuyển sang lô 32 và luôn có đủ yêu cầu chờ. Từ lúc bắt đầu xử lý, mỗi yêu cầu mất bao lâu so với trước? Số yêu cầu hoàn thành mỗi giây thay đổi ra sao?
        </>}>
        A GPU server runs a model on one request at a time (batch size 1). You switch it to batches of 32, and there are
        always enough requests waiting to fill a batch. What happens to the time one request takes once it starts
        running, and to the number of requests finished per second?
        </LessonText>
      </>
    ),
    options: [
      { label: 'Each request takes longer; requests per second go up.', labelVi: <>Mỗi yêu cầu mất nhiều thời gian hơn. Số yêu cầu hoàn thành mỗi giây tăng.</>, correct: true },
      {
        label: 'Each request is quicker; requests per second go up.', labelVi: <>Mỗi yêu cầu xử lý nhanh hơn. Số yêu cầu hoàn thành mỗi giây tăng.</>,
        why: 'A pass over 32 requests does more work than a pass over one. No single request gets its answer sooner than it would alone.', whyVi: <>Một lượt 32 yêu cầu làm nhiều việc hơn một lượt duy nhất. Vì vậy từng yêu cầu phải đợi lâu hơn khi được xử lý trong lô.</>,
      },
      {
        label: 'Each request takes the same time; requests per second go up 32 times.', labelVi: <>Mỗi yêu cầu mất chừng ấy thời gian. Số yêu cầu hoàn thành mỗi giây tăng gấp 32 lần.</>,
        why: 'The batched pass is not free. It takes longer than one request alone, though far less than 32 of them in a row.', whyVi: <>Xử lý cả lô tốn nhiều thời gian hơn một yêu cầu riêng lẻ, dù nhanh hơn nhiều so với xử lý tuần tự 32 yêu cầu.</>,
      },
      {
        label: 'Each request takes longer; requests per second stay the same.', labelVi: <>Mỗi yêu cầu mất nhiều thời gian hơn. Số yêu cầu hoàn thành mỗi giây giữ nguyên.</>,
        why: 'Then nobody would batch. The batched pass takes much less time than its 32 requests run one after another.', whyVi: <>Một lượt xử lý cả lô nhanh hơn nhiều so với xử lý lần lượt 32 yêu cầu, nên số yêu cầu hoàn thành mỗi giây tăng.</>,
      },
    ],
    explain: (
      <p>
        <LessonText vi={<>
          Chạy một lượt cho 32 yêu cầu mất nhiều thời gian hơn một lượt cho một yêu cầu, nhưng ít hơn nhiều so với chạy tuần tự 32 lượt. Vì vậy mỗi yêu cầu chờ lâu hơn sau khi bắt đầu, còn cả nhóm hoàn thành sớm hơn. Mục §22.5 của sách liên hệ việc gom lô với đánh đổi trong thiết kế phần cứng. Một thiết kế giảm độ trễ của từng phần việc, thiết kế kia tăng tổng thông lượng khi có nhiều phần việc.
        </>}>
        One pass over 32 requests takes longer than a pass over one, and much less than 32 passes. So each request
        waits longer and the group is done sooner. The book puts it this way in its section on batching, and then says
        it is the same trade as the one between the two kinds of processor: one design gives each piece of work a
        shorter latency, the other gives many pieces a higher throughput. This page is about that trade, built into
        hardware.
        </LessonText>
      </p>
    ),
    source: 'ch. 22 §22.5, pp. 539–540',
  },
  'ch01/start/q-sixteenbit': {
    kind: 'choice',
    id: 'ch01/start/q-sixteenbit',
    prompt: (
      <>
        <LessonText vi={<>
          Sách liệt kê tốc độ 34, 67 và 1,979 TFLOPS cho phép tính 64, 32 và 16 bit trên GPU ví dụ. Giảm từ 64 xuống 32 bit gần như tăng gấp đôi tốc độ. Điều gì giải thích mức tăng khoảng 30 lần khi giảm từ 32 xuống 16 bit?
        </>}>
        The book lists 34, 67 and 1,979 TFLOPS for 64-, 32- and 16-bit arithmetic on its example GPU. Going from 64 to
        32 bits doubles the rate. What explains the step of about 30 times from 32 to 16 bits?
        </LessonText>
      </>
    ),
    options: [
      {
        label: 'Halving the precision again: a 16-bit unit is much smaller than a 32-bit one, so far more of them fit on the chip.', labelVi: <>Độ chính xác giảm thêm một nửa. Khối 16 bit nhỏ hơn nhiều nên chip chứa được nhiều khối hơn.</>,
        why: 'The step from 64 to 32 bits shows what narrower values buy: a factor of 2. The same step again would give about 134, not 1,979.', whyVi: <>Giảm độ rộng từ 64 xuống 32 bit cho hệ số xấp xỉ 2. Lặp lại hệ số đó sẽ cho khoảng 134 TFLOPS.</>,
      },
      {
        label: 'Another kind of unit, and an assumption: it is the tensor-core (matrix-multiply) rate, quoted with sparsity.', labelVi: <>Phần cứng khác và có thêm một giả định. Đây là tốc độ của Tensor Core khi tận dụng tính thưa.</>,
        correct: true,
      },
      {
        label: 'Memory: 16-bit values need half the bandwidth, so twice as many of them arrive per second.', labelVi: <>Dữ liệu 16 bit cần một nửa băng thông, nên mỗi giây truyền được gấp đôi số giá trị.</>,
        why: 'These are peak rates of the arithmetic units. Memory does not enter into them.', whyVi: <>Những con số này là tốc độ đỉnh của các khối tính toán, được tính độc lập với bộ nhớ.</>,
      },
      {
        label: 'A misprint: by the pattern of the first two figures, the 16-bit one should read about 134.', labelVi: <>Đó là lỗi in. Theo hai con số đầu, tốc độ 16 bit phải vào khoảng 134 TFLOPS.</>,
        why: 'The vendor\'s own table prints the same three numbers.', whyVi: <>Bảng thông số của chính NVIDIA cũng ghi đúng ba con số này.</>,
      },
    ],
    explain: (
      <p>
        <LessonText vi={<>
          Giảm một nửa độ rộng dữ liệu làm tốc độ gần gấp đôi ở lần đầu. Chừng đó chưa giải thích được hệ số 30. Con số thứ ba đến từ dòng «FP16 Tensor Core» trong bảng thông số, có chú thích «With sparsity». Tensor Core là khối chuyên dùng cho phép nhân ma trận. Đối với suy luận LLM, điều đáng xem là lợi ích của khối nhân ma trận so với khối tính toán thông thường và cách kernel sử dụng nó. Đó là nội dung chương 15.
        </>}>
        Halving the width of the values doubled the rate once, so a further halving cannot account for a factor of 30.
        The third number has to come from a different row of the vendor's table, and it does: the row is "FP16 Tensor
        Core", and it carries the footnote "With sparsity". Tensor cores are units built for matrix multiplication.
        For LLM inference the comparison that matters is therefore not 16 bits against 32. It is matrix units against
        ordinary ones, and what a kernel has to do to use them, which is chapter 15.
        </LessonText>
      </p>
    ),
    source: 'book p. 3; NVIDIA H100 specification table, checked 2026-11-20 (notes/today/ch01-facts.md, A1)',
  },
  'ch01/start/q-chainlen': {
    kind: 'choice',
    id: 'ch01/start/q-chainlen',
    prompt: (
      <>
        <LessonText vi={<>
          Xét hai công việc trên hai thiết kế trong hoạt ảnh. A có {FAST.units} khối ở tốc độ {FAST.speed}, còn B có {MANY.units} khối ở tốc độ {MANY.speed}. X có {nf(LONG_FEW.chains)} chuỗi độc lập, mỗi chuỗi gồm {nf(LONG_FEW.length)} phép toán, tổng cộng {ops(LONG_FEW)}. Y có {nf(SHORT_MANY.chains)} chuỗi độc lập, mỗi chuỗi gồm {nf(SHORT_MANY.length)} phép toán, tổng cộng {ops(SHORT_MANY)}. Theo mô phỏng, thiết kế nào hoàn thành trước ở mỗi công việc?
        </>}>
        Two jobs for the animation's designs, A ({describe(FAST)}) and B ({describe(MANY)}). Job X is{' '}
        {nf(LONG_FEW.chains)} independent chains of {nf(LONG_FEW.length)} operations: {ops(LONG_FEW)} operations in
        all. Job Y is {nf(SHORT_MANY.chains)} independent chains of {nf(SHORT_MANY.length)} operations:{' '}
        {ops(SHORT_MANY)} in all. In the explorer's model, which design finishes each job first?
        </LessonText>
      </>
    ),
    options: [
      {
        label: `Both on A: its units are ${FAST.speed} times faster.`, labelVi: <>A thắng cả hai công việc, vì mỗi khối của A nhanh gấp {FAST.speed} lần.</>,
        why: `For Y, A needs ${nf(shortMany.a.waves)} waves of ${nf(SHORT_MANY.length / FAST.speed)}, done at ${nf(shortMany.a.last)}. B needs ${nf(shortMany.b.waves)} waves of ${nf(SHORT_MANY.length)}, done at ${nf(shortMany.b.last)}.`, whyVi: <>Với Y, A cần {nf(shortMany.a.waves)} đợt, mỗi đợt {nf(SHORT_MANY.length / FAST.speed)} đơn vị thời gian, và xong ở mốc {nf(shortMany.a.last)}. B cần {nf(shortMany.b.waves)} đợt, mỗi đợt {nf(SHORT_MANY.length)}, và xong ở mốc {nf(shortMany.b.last)}.</>,
      },
      {
        label: `Both on B: it has ${MANY.throughput / FAST.throughput} times the peak rate.`, labelVi: <>B thắng cả hai công việc, vì thông lượng đỉnh của B cao gấp {MANY.throughput / FAST.throughput} lần.</>,
        why: `A peak rate needs every unit busy. For X, B can use only ${nf(LONG_FEW.chains)} of its ${MANY.units} units: one wave of ${nf(LONG_FEW.length)}, done at ${nf(longFew.b.last)}. A runs ${nf(longFew.a.waves)} waves of ${nf(LONG_FEW.length / FAST.speed)} and is done at ${nf(longFew.a.last)}.`, whyVi: <>Muốn đạt tốc độ đỉnh, mọi khối phải có việc. Với X, B chỉ dùng {nf(LONG_FEW.chains)}/{MANY.units} khối và xong sau một đợt ở mốc {nf(longFew.b.last)}. A chạy {nf(longFew.a.waves)} đợt, mỗi đợt {nf(LONG_FEW.length / FAST.speed)}, và xong ở mốc {nf(longFew.a.last)}.</>,
      },
      { label: 'X on A, Y on B: A for the few chains, B for the many.', labelVi: <>X chạy nhanh hơn trên A vì có ít chuỗi. Y chạy nhanh hơn trên B vì có nhiều chuỗi.</>, correct: true },
      {
        label: 'X on B, Y on A: the bigger job goes to the design with the higher peak rate.', labelVi: <>X chạy nhanh hơn trên B, Y chạy nhanh hơn trên A. Công việc lớn hơn phù hợp với thiết kế có thông lượng đỉnh cao hơn.</>,
        why: `That is each job on the design that suits it least: X takes ${nf(longFew.b.last)} on B against ${nf(longFew.a.last)} on A, and Y takes ${nf(shortMany.a.last)} on A against ${nf(shortMany.b.last)} on B.`, whyVi: <>X mất {nf(longFew.b.last)} trên B so với {nf(longFew.a.last)} trên A. Y mất {nf(shortMany.a.last)} trên A so với {nf(shortMany.b.last)} trên B.</>,
      },
    ],
    explain: (
      <p>
        <LessonText vi={<>
          Hãy đếm số phần việc độc lập trước khi đếm phép toán. X có {nf(LONG_FEW.chains)} chuỗi, nhiều hơn {FAST.units} khối của A và ít hơn hẳn {MANY.units} khối của B. A chạy {nf(longFew.a.waves)} đợt và xong ở mốc {nf(longFew.a.last)}. B chạy một đợt và xong ở mốc {nf(longFew.b.last)}. Y có {nf(SHORT_MANY.chains)} chuỗi. B xong sau {nf(shortMany.b.waves)} đợt ở mốc {nf(shortMany.b.last)}. A cần {nf(shortMany.a.waves)} đợt và xong ở mốc {nf(shortMany.a.last)}. Vậy công việc có nhiều phép toán hơn lại phù hợp với thiết kế ưu tiên độ trễ. Với cùng một công việc, độ dài chuỗi thay đổi thời gian của cả hai thiết kế theo cùng tỷ lệ và giữ nguyên thứ tự về đích. Bạn có thể kiểm tra cả hai trường hợp ở mô phỏng trên.
        </>}>
        Count the independent pieces before the operations. X has {nf(LONG_FEW.chains)}: more than A's {FAST.units}{' '}
        units and far fewer than B's {MANY.units}. A finishes it in {nf(longFew.a.waves)} waves, at{' '}
        {nf(longFew.a.last)}; B in one wave, at {nf(longFew.b.last)}. Y has {nf(SHORT_MANY.chains)}: B finishes in{' '}
        {nf(shortMany.b.waves)} waves, at {nf(shortMany.b.last)}, while A needs {nf(shortMany.a.waves)}, until{' '}
        {nf(shortMany.a.last)}. So the job with more arithmetic goes to the latency design. On one job the length of
        a chain scales both designs' times alike: it sets how long the job takes, not which design should take it.
        Both jobs fit the explorer above, if you want to check.
        </LessonText>
      </p>
    ),
    source: 'our model of the book\'s argument, ch. 1 §1.1, pp. 4–5',
  },
  'ch01/start/q-tokens': {
    kind: 'choice',
    id: 'ch01/start/q-tokens',
    prompt: (
      <>
        <LessonText vi={<>
          Một yêu cầu sinh văn bản theo từng token. Tính token t+1 cần token t. Vậy sự độc lập cần thiết để GPU phát huy thông lượng nằm ở đâu?
        </>}>
        One request generates text token by token, and computing token t+1 needs token t. Where is the independence
        that a throughput-oriented processor needs?
        </LessonText>
      </>
    ),
    options: [
      {
        label: 'Across tokens: the GPU works on many future tokens of the request at the same time.', labelVi: <>Giữa các token. GPU xử lý đồng thời nhiều token tương lai của cùng một yêu cầu.</>,
        why: 'Token t+1 needs token t. Along the sequence, the work is one dependent chain.', whyVi: <>Token t+1 cần token t. Xét dọc theo chuỗi, các bước phụ thuộc lẫn nhau.</>,
      },
      {
        label: 'Inside each step: every matrix product is thousands of dot products that do not depend on each other.', labelVi: <>Trong mỗi bước. Mỗi phép nhân ma trận gồm hàng nghìn tích vô hướng độc lập.</>,
        correct: true,
      },
      {
        label: 'Nowhere in one request: generation is sequential, so a GPU only helps once requests are batched.', labelVi: <>Chỉ khi gom nhiều yêu cầu thành lô, bởi một yêu cầu sinh token theo thứ tự.</>,
        why: 'The loop over tokens is sequential. What happens inside one turn of the loop is not.', whyVi: <>Vòng lặp qua các token chạy theo thứ tự. Trong từng bước, nhiều phép tính vẫn chạy được song song.</>,
      },
    ],
    explain: (
      <p>
        <LessonText vi={<>
          Vòng lặp sinh token là chuỗi phụ thuộc. Trong mỗi bước, các phép nhân ma trận lại chia được thành nhiều phần độc lập. Mỗi phần tử đầu ra là một tích vô hướng và có thể tính riêng, như ở đoạn cuối của hoạt ảnh. Gom lô bổ sung tính độc lập giữa các yêu cầu. Một yêu cầu riêng lẻ vẫn phải sinh từng token theo thứ tự và vẫn tận dụng được song song trong từng bước. Đây là cách khóa học áp dụng lập luận của sách.
        </>}>
        The loop over tokens is the chain. The matrix products inside one step are the crowd: each output element is a
        dot product that needs no other output element, which is the picture in the last part of the animation.
        Batching adds a second source of independence, across requests. So a single stream is limited by the chain
        (one step after another) and helped by the crowd inside each step. This mapping is ours.
        </LessonText>
      </p>
    ),
    source: 'our analysis, using ch. 1 §1.1, pp. 4–6; the book treats generation in ch. 20',
  },
  'ch01/start/q-oneunitmore': {
    kind: 'numeric',
    id: 'ch01/start/q-oneunitmore',
    prompt: (
      <>
        <LessonText vi={<>
          Thiết kế B có 64 khối, mỗi khối có tốc độ 1. Công việc gồm 80 chuỗi độc lập, mỗi chuỗi có 100 phép toán. Theo mô phỏng, cần bao nhiêu đơn vị thời gian để hoàn thành chuỗi cuối cùng? Hãy tính trước rồi kiểm tra bằng mô phỏng.
        </>}>
        Design B has 64 units of speed 1. The work is 80 independent chains of 100 operations each. In the explorer's
        model, how many time units pass before the last chain finishes? Work it out, then set the explorer to it.
        </LessonText>
      </>
    ),
    answer: 200,
    unit: 'time units',
    unitVi: 'đơn vị thời gian',
    explain: (
      <p>
        <LessonText vi={<>
          64 chuỗi bắt đầu cùng lúc và xong ở mốc 100. 16 chuỗi còn lại bắt đầu sau đó và xong ở mốc 200, tổng cộng hai đợt. Thêm khối thứ 65 vẫn cần hai đợt. Bớt 16 chuỗi chỉ còn một đợt, nên thời gian giảm một nửa. Khối lượng phần cứng xử lý được mỗi đợt khiến thời gian thay đổi theo từng bậc. Chương 2 cũng dùng cách làm tròn lên này để tính số block luồng khi khởi chạy kernel.
        </>}>
        64 chains start at once and finish at 100. The other 16 start then and finish at 200: two waves. A 65th unit
        would change nothing, because 80 chains would still need two waves. Taking 16 chains away would halve the
        time. Throughput arrives in steps the size of the hardware. You meet the same rounding up in chapter 2, where
        it sets how many blocks of threads a kernel launch needs.
        </LessonText>
      </p>
    ),
    source: 'our model of the book\'s argument, ch. 1 §1.1, pp. 4–5',
  },
};
/** The same record under the name the chapter index convention expects. */
export const questionsS10 = questions10;

// ---------------------------------------------------------------- page
export function S10() {
  const { language, t } = useLanguage();
  const vi = language === 'vi';
  const crowd = race(FAST, MANY, 1024, 100);
  return (
    <>
      <p>
        {vi ? (<>
        Chương đầu của sách trình bày lý do nên học lập trình song song và giới thiệu cấu trúc các chương sau.
        Khóa học giữ lại bốn ý: vì sao CPU và GPU được thiết kế khác nhau (mục này), cách tính giới hạn tăng tốc và bốn
        khó khăn xuyên suốt cuốn sách (<a href={href('ch01', '1.3')}>mục 1.3</a>), cùng lộ trình đọc phù hợp với mục tiêu
        của bạn (<a href={href('ch01', '1.7')}>mục 1.7</a>). Chương này chỉ có phần giải thích.
        </>) : (<>
        The book's first chapter has no code and no exercises. It argues that parallel programming is worth learning
        and lays out the rest of the book. This course keeps four things from it: why the two kinds of processor are
        built differently (this page); how to put a ceiling on a speed-up, and the four difficulties the book is
        organised around (<a href={href('ch01', '1.3')}>section 1.3</a>); and a map of the book for your goals (
        <a href={href('ch01', '1.7')}>section 1.7</a>).
        </>)}
      </p>
      <Source pages="1–20" />
      <p>
        {vi ? (<>
        Các chương sau đều bắt đầu bằng một bài kiểm tra ngắn về kiến thức đã học. Ở chương đầu, hãy thử câu hỏi
        về việc phục vụ mô hình dựa trên kinh nghiệm của bạn với framework. Phần còn lại của bài sẽ giải thích
        câu trả lời từ góc độ phần cứng.
        </>) : (<>
        Every later chapter opens with a check on what the chapters before it taught. Nothing comes before this one,
        so there is no such check here. One question from the framework side stands in for it. Answer it from what
        you know about serving models. The rest of the page explains why the answer is also a statement about
        hardware.
        </>)}
      </p>
      <Ask q={questions10['ch01/start/q-batch']} label={t('From what you already know', 'Dựa trên những gì bạn đã biết')} />

      <h2>{t('Why there are two kinds of processor', 'Vì sao có hai hướng thiết kế bộ xử lý')}</h2>
      <p>
        {vi ? (<>
        Cho đến khoảng năm 2003, cùng một chương trình thường chạy nhanh hơn trên bộ xử lý đời mới vì bộ xử lý
        thực thi một chuỗi lệnh nhanh hơn trước. Điện năng tiêu thụ và nhiệt tỏa ra đã giới hạn cả tần số xung nhịp
        lẫn lượng công việc xử lý được trong một chu kỳ. Từ đó, tốc độ tăng chủ yếu nhờ thực thi đồng thời nhiều luồng
        lệnh, và chương trình hưởng lợi khi công việc có thể chia cho các luồng đó. Hai hướng thiết kế hình thành.
        Bộ xử lý <b>đa lõi</b> giữ tốc độ cao cho từng luồng và tăng số lõi. Bộ xử lý <b>nhiều luồng</b> như GPU chấp
        nhận tốc độ thấp hơn trên từng luồng để hoàn thành nhiều việc hơn khi chạy rất nhiều luồng cùng lúc.
        </>) : (<>
        Until about 2003 a program got faster with each new processor without being touched, because each new
        processor ran a single sequence of instructions faster than the last. Energy use and heat ended that: they
        capped both the clock rate and the work that fits in one clock period. Since then, more speed has come from
        running several instruction sequences (threads) at the same time, and a program benefits only if its work is
        split that way. Two lines of design followed. <b>Multi-core</b> processors keep each thread fast and add
        cores. <b>Many-thread</b> processors, the GPUs, give up speed per thread to maximise the total work done by a
        very large number of them.
        </>)}
      </p>
      <Source pages="1–3" />
      <p>
        {vi ? (<>
        Sách lấy NVIDIA Hopper H100 làm ví dụ cho hướng thứ hai. H100 giữ được hàng trăm nghìn luồng trên nhiều
        pipeline đơn giản. Theo số liệu đỉnh năm 2022 mà sách trích dẫn, GPU này đạt 34, 67 và 1,979 TFLOPS tương ứng
        với phép tính 64, 32 và 16 bit. Một TFLOPS tương ứng 10<sup>12</sup> phép tính dấu phẩy động mỗi giây. CPU máy
        chủ cùng năm đạt vài TFLOPS. Đây là tốc độ đỉnh của phần cứng. Tốc độ của ứng dụng còn phụ thuộc vào việc nó
        sử dụng phần cứng đó hiệu quả đến đâu.
        </>) : (<>
        The book's example of the second kind is NVIDIA's Hopper H100, with hundreds of thousands of threads on many
        simple pipelines. It quotes peak rates, as of 2022, of 34, 67 and
        1,979 TFLOPS for 64-, 32- and 16-bit arithmetic (one TFLOPS is 10<sup>12</sup> floating-point operations per
        second), against "a few TFLOPS" for a server CPU of the same year. It adds its own caveat: these are peak
        rates of the hardware, not the speed of any application.
        </>)}
      </p>
      <Source section="1.1" pages="3" />
      <Ask q={questions10['ch01/start/q-sixteenbit']} />
      <p>
        {vi ? (<>
        Khi đọc các số liệu như trên, hãy nhớ hai điều. Tốc độ đỉnh giả định khối tính toán luôn có việc để làm.
        Những con số cao nhất trên bảng thông số thường đến từ phần cứng chuyên nhân ma trận. Chương 5 giải thích
        cách đưa kernel tiến gần tốc độ đỉnh. Chương 15 trình bày cách tận dụng phần cứng nhân ma trận.
        </>) : (<>
        Read every figure of this kind with both points in mind. A peak rate is what the arithmetic units could do if
        they never had to wait for anything, and the largest figures on a specification sheet belong to the matrix
        units. How close a kernel gets to the first is the subject of chapter 5. What it takes to use the second is
        in chapter 15.
        </>)}
      </p>
      <Today checked="2026-11-20">
        <p>
          {vi ? (<>
        Bảng thông số H100 của NVIDIA ghi cả ba con số cho biến thể SXM. Hai số 34 và 67 nằm ở các dòng FP64
          và FP32 thông thường. Số 1,979 nằm ở dòng «FP16 Tensor Core», với chú thích «With sparsity» (khi tận dụng
          tính thưa của dữ liệu).
        </>) : (<>
        NVIDIA's H100 page still prints all three figures, for the SXM variant. 34 and 67 are its plain FP64 and
          FP32 rows. 1,979 is the row "FP16 Tensor Core", footnoted "With sparsity".
        </>)}
        </p>
      </Today>

      <h2>{t('Where the chip area goes', 'Diện tích chip được dùng vào đâu')}</h2>
      <p>
        {vi ? (<>
        Sự chênh lệch đến từ mục tiêu của từng thiết kế. CPU <b>ưu tiên độ trễ thấp</b>, tức là hoàn thành từng
        phép toán trong một luồng nhanh nhất có thể. Thiết kế này dùng các khối tính toán có độ trễ thấp, bộ nhớ đệm
        lớn để xử lý nhanh các lần truy cập bộ nhớ và logic điều khiển có dự đoán nhánh để duy trì luồng lệnh.
        Những thành phần này dùng diện tích chip và điện năng vốn có thể dành cho thêm khối tính toán và kênh bộ nhớ.
        </>) : (<>
        The gap comes from what each design is asked to be good at. A CPU is a <b>latency-oriented</b> design: it is
        built so that each operation of a single thread finishes as soon as possible. That takes arithmetic units
        designed for short latency, large caches that turn slow memory accesses into fast ones, and control logic
        with branch prediction to keep one stream of instructions moving. All of it uses chip area and power that
        could have gone to more arithmetic units and more channels to memory.
        </>)}
      </p>
      <Source section="1.1" pages="4–5" figs="Fig. 1.1" />
      <AreaFigure />
      <p>
        {vi ? (<>
        GPU <b>ưu tiên thông lượng</b>. Thiết kế này bắt nguồn từ trò chơi điện tử, nơi mỗi khung hình cần rất
        nhiều phép tính dấu phẩy động và lần truy cập bộ nhớ. Diện tích và điện năng chủ yếu dành cho nhiều khối
        tính toán đơn giản và nhiều kênh bộ nhớ. GPU dùng ít logic điều khiển hơn và bộ nhớ đệm nhỏ hơn, đồng thời
        chấp nhận độ trễ cao cho từng phép toán.
        </>) : (<>
        A GPU is a <b>throughput-oriented</b> design. Its shape comes from video games, which need a very large number
        of floating-point calculations and memory accesses for every frame. So the area and the power go to many
        simple arithmetic units and many memory channels, with little control logic and small caches, and each single
        operation is allowed to take a long time.
        </>)}
      </p>
      <Today checked="2026-11-20">
        <p>
          {vi ? (<>
        Tài liệu lập trình CUDA của NVIDIA mô tả cùng một đánh đổi. GPU chấp nhận hiệu năng mỗi luồng thấp
          hơn để đạt tổng thông lượng cao hơn nhiều, bằng cách dành nhiều transistor hơn cho các khối xử lý dữ liệu.
        </>) : (<>
        NVIDIA's current CUDA programming guide states the same split: a GPU trades "lower single-thread
          performance" for "much greater total throughput", and GPUs "devote more transistors to data processing
          units".
        </>)}
        </p>
      </Today>
      <p>
        {vi ? (<>
        Vì sao khó làm một chip giỏi cả hai? Sách đưa ra một quy tắc về chi phí. Muốn thông lượng gấp đôi,
        hãy dùng gấp đôi số khối tính toán, tức gấp đôi diện tích và công suất. Muốn giảm một nửa độ trễ của một khối,
        có thể phải tăng dòng điện lên gấp đôi, dùng hơn gấp đôi diện tích và gấp bốn lần công suất. Theo lập luận
        này, cải thiện độ trễ tốn kém hơn nhiều so với tăng thông lượng. Đây là quy tắc lập luận của sách và chưa
        được kiểm chứng trong tài liệu của nhà sản xuất. Hoạt ảnh và mô phỏng bên dưới cho thấy hệ quả của giả định đó.
        </>) : (<>
        Why not build one chip that is good at both? The book's answer is a cost rule. To double the throughput,
        double the number of units: twice the area and twice the power. To halve the latency of a unit, you may need
        twice the current, more than twice the area and four times the power. Latency is far more expensive to buy
        than throughput, so a design has to choose. Treat this rule as the book's argument: no vendor document states
        it, and it is unverified. The animation and the explorer below take that one rule and follow where it leads.
        </>)}
      </p>
      <Source section="1.1" pages="5" />

      <h2>{t('One budget, two designs', 'Cùng ngân sách, hai thiết kế')}</h2>
      <p>
        {vi ? (<>
        <b>Độ trễ</b> là thời gian để hoàn thành một phần việc. <b>Thông lượng</b> là lượng công việc hoàn
        thành trong mỗi đơn vị thời gian, tính trên tất cả các phần việc. Hoạt ảnh giao cùng một ngân sách công suất
        và cùng một khối lượng việc cho hai thiết kế rồi đo trên cùng một trục thời gian. Hãy xem một lượt trước khi
        thử các nút điều khiển bên dưới.
        </>) : (<>
        Two terms first. <b>Latency</b> is how long one piece of work takes from start to finish.{' '}
        <b>Throughput</b> is how much work is finished per unit of time, over all the pieces. The animation gives two
        designs the same power budget and the same work, and runs them against one clock. Watch it once before you
        use the controls further down.
        </>)}
      </p>
      {vi ? <p className="status-line">Bật phụ đề để đọc bản tiếng Việt của lời thuyết minh.</p> : null}
      <TwoDesignsScene />
      <Predict
        id="ch01/start/p-crowd"
        prompt={
          <>
            {vi ? (
              <>Thiết kế A có {FAST.units} khối, mỗi khối nhanh gấp {FAST.speed} lần khối thường. Thiết kế B có {MANY.units} khối thường.
                Theo quy tắc của sách, hai thiết kế dùng cùng công suất. Có 1,024 chuỗi độc lập, cùng độ dài. B hoàn thành nhanh gấp mấy lần A?</>
            ) : (
              <>Design A has {FAST.units} units, each {FAST.speed} times faster than a plain one. Design B has {MANY.units}{' '}
            plain units. By the book's rule both draw the same power. The work is 1,024 independent chains of equal
            length. How many times sooner does B finish?</>
            )}
          </>
        }
        unit={t('times sooner', 'lần nhanh hơn')}
      >
        {(v) => vi ? (
          <p>
            <b>{nf(crowd.factor)} lần.</b> B chạy {MANY.units} chuỗi đồng thời, nên cần {nf(crowd.b.waves)} đợt.
            A chạy {FAST.units} chuỗi đồng thời và cần {nf(crowd.a.waves)} đợt. Mỗi đợt của A nhanh gấp {FAST.speed} lần,
            nên tổng thời gian của A bằng {nf(crowd.a.waves / FAST.speed)} đợt của B. Lấy số đó chia cho {nf(crowd.b.waves)}
            đợt của B, bạn được {nf(crowd.factor)}.{' '}
            {v === crowd.factor
              ? 'Đó cũng là số bạn đã dự đoán.'
              : v === MANY.units / FAST.units
                ? `${nf(v)} là tỷ lệ số khối, cũng là tỷ lệ số đợt. Cần tính thêm mỗi khối của A nhanh gấp ${FAST.speed} lần.`
                : v > 0 && v < 1
                  ? 'Số nhỏ hơn 1 nghĩa là A thắng. A chỉ thắng khi số chuỗi quá ít để tận dụng các khối của B.'
                  : v === MANY.units
                    ? `${nf(v)} chỉ là số khối của B. Hãy so với số khối và tốc độ của A.`
                    : 'Hãy đếm số đợt mỗi thiết kế cần, rồi tính thời gian của mỗi đợt.'}{' '}
            Đặt mô phỏng thành 1,024 chuỗi để kiểm tra. Sau đó giảm còn 1 chuỗi và quan sát bên nào thắng.
          </p>
        ) : (
          <p>
            <b>{nf(crowd.factor)} times.</b> B runs {MANY.units} chains at a time, so 1,024 chains take{' '}
            {nf(crowd.b.waves)} waves. A runs {FAST.units} at a time and needs {nf(crowd.a.waves)} waves, each{' '}
            {FAST.speed} times shorter, which is the time of {nf(crowd.a.waves / FAST.speed)} of B's waves.{' '}
            {nf(crowd.a.waves / FAST.speed)} against {nf(crowd.b.waves)} is {nf(crowd.factor)}.{' '}
            {v === crowd.factor
              ? 'That is what you had.'
              : v === MANY.units / FAST.units
                ? `${nf(v)} is the ratio of the unit counts, or of the wave counts. It forgets that each of A's units is ${FAST.speed} times faster.`
                : v > 0 && v < 1
                  ? 'A number below 1 backs the faster units. They win only while there are too few chains to keep B\'s units busy.'
                  : v === MANY.units
                    ? `${nf(v)} counts B's units and leaves A out of the comparison.`
                    : 'Count the waves each design needs, then how long one wave takes.'}{' '}
            Set the explorer below to 1,024 chains and check it. Then bring the chains back down to 1 and watch the
            factor turn round.
          </p>
        )}
      </Predict>
      <BudgetBench />
      <p>
        {vi ? (<>
        Với cùng ngân sách, hệ số thắng của thiết kế ít khối nhanh trên một chuỗi bằng hệ số thắng của thiết kế nhiều khối thường khi các khối đều bận. Vài khối nhanh
        thắng khi chỉ có một chuỗi. Nhiều khối thường thắng khi có đủ chuỗi để chạy đồng thời. Thay đổi độ dài chuỗi
        nhân thời gian của cả hai thiết kế theo cùng tỷ lệ, nên không đổi bên thắng hay hệ số tăng tốc. Số phần
        việc độc lập mới quyết định bên nào thắng. Trong cùng ngân sách, mỗi lần giảm một nửa độ trễ làm thông lượng
        đỉnh lần lượt đi từ {HALVINGS.map((h) => design(BUDGET, h).throughput).join(' xuống ')}.
        </>) : (<>
        Three things to take from it. On one budget the two designs differ by the same factor in opposite directions:
        the fast units win a single chain by exactly the factor by which the many units win a crowd. The length of a
        chain cancels out, so it never changes the winner or the factor. What decides the winner is how many pieces
        of the work are independent of each other. And latency bought on a fixed budget is paid for in peak rate:
        each halving takes the rate with every unit busy from {HALVINGS.map((h) => design(BUDGET, h).throughput).join(' to ')}.
        </>)}
      </p>
      <Simplification>
        <p>
          {vi ? (<>
        Giảm một nửa độ trễ với công suất gấp bốn lần là quy tắc trong sách (trang 5). Việc áp dụng quy tắc
          này nhiều lần là phép ngoại suy của khóa học. Mạch điện thực tế có quan hệ phức tạp hơn.
        </>) : (<>
        One step, half the latency for four times the power, is the book's rule (p. 5). Repeating it is our
          extrapolation; real circuits do not follow one exponent.
        </>)}
        </p>
        <p>
          {vi ? (<>
        Mô hình bỏ qua chi phí diện tích, bộ nhớ đệm, kênh bộ nhớ và pipeline. Mỗi khối chỉ chạy một chuỗi tại
          một thời điểm. GPU thật giữ được nhiều luồng trên phần cứng hơn số khối tính toán (chương 4). Ngân sách
          {BUDGET} là một giá trị giả định của mô hình.
        </>) : (<>
        Left out: the area cost, caches, memory channels, pipelining. A unit runs one chain at a time; a real GPU
          keeps many more threads resident than units (chapter 4). The budget of {BUDGET} is an abstract number, not
          a device.
        </>)}
        </p>
      </Simplification>
      <p>
        {vi ? (<>
        Hãy xem lại bảng. Tốc độ đỉnh của một thiết kế cố định với mọi khối lượng công việc. Thông lượng
        thực tế thì thay đổi. Khi chỉ có một chuỗi, nó chỉ bằng một phần nhỏ của tốc độ đỉnh. Đây chính là lưu ý
        của sách về H100, minh họa bằng mô hình đủ đơn giản để bạn kiểm tra bằng tay.
        </>) : (<>
        Look at the table once more before you leave it. The peak rate of a design never moves, whatever work you
        give it. The rate delivered does, and with a single chain it is a small fraction of the peak. That is the
        book's caveat about the H100 figures, in a model small enough to check by hand.
        </>)}
      </p>
      <Ask q={questions10['ch01/start/q-chainlen']} />
      <Ask q={questions10['ch01/start/q-oneunitmore']} />

      <h2>{t('What the design asks of your code', 'Thiết kế GPU đòi hỏi gì ở mã của bạn')}</h2>
      <p>
        {vi ? (<>
        Trong mô phỏng, một khối tính toán sẽ nhàn rỗi nếu số chuỗi ít hơn số khối. Trên GPU thật, khối còn
        có thể nhàn rỗi trong khi luồng chờ bộ nhớ hoặc chờ kết quả của phép toán trước. Vì thế, mã GPU cần tạo
        nhiều luồng để phần cứng luôn tìm được luồng sẵn sàng chạy trong lúc các luồng khác chờ.
        </>) : (<>
        In the explorer a unit sits idle whenever there are fewer chains than units. On a real GPU there is a second
        reason for a unit to have nothing to do. The design lets memory accesses and arithmetic take a long time, and
        a thread that is waiting for one of them has nothing to run. The book draws the conclusion for software: GPU
        code is expected to supply a very large number of threads, so that the hardware always has some that are
        ready while others wait.
        </>)}
      </p>
      <Source section="1.1" pages="5" />
      <p>
        {vi ? (<>
        Chương 4 giải thích cơ chế này tại <a href={href('ch04', '4.6')}>§4.6, lập lịch warp và che giấu độ
        trễ</a> (trang 82–85). Phần cứng giữ nhiều luồng thường trú hơn số luồng có thể thực thi cùng một lúc. Ở mỗi
        bước, nó chọn công việc đang sẵn sàng. Hàng trăm nghìn luồng của H100 mà sách nêu là số luồng thường trú.
        Số luồng thực sự thực thi tại một thời điểm thấp hơn nhiều.
        </>) : (<>
        Chapter 4 shows the mechanism (<a href={href('ch04', '4.6')}>§4.6, warp scheduling and latency tolerance</a>,
        pp. 82–85): the hardware keeps many more threads resident than it can
        execute at one instant, and at each step it picks one that is not waiting. The hundreds of thousands of
        threads the book gives the H100 are threads of this kind: resident, with far fewer executing at any moment.
        </>)}
      </p>
      <Today checked="2026-11-20 (notes/today/ch01-facts.md, A2)">
        <p>
          {vi ? (<>
        Tích của hai thông số NVIDIA công bố cho H100 cho ra số luồng thường trú ở quy mô này. Phép tính do
          khóa học thực hiện.
        </>) : (<>
        Two of NVIDIA's own H100 figures, multiplied together, give a count of that size for threads held resident.
          The multiplication is ours.
        </>)}
        </p>
      </Today>
      <p>
        {vi ? (<>
        Hai thiết kế bổ sung cho nhau. Các đoạn tuần tự của chương trình chạy trên CPU, còn phần nặng về
        tính toán được chuyển sang GPU. Đó là lý do mô hình lập trình ở chương 2 có hai vai trò, host và thiết bị.
        </>) : (<>
        Neither design replaces the other. The sequential parts of a program stay on the CPU and the numerically
        heavy parts go to the GPU, which is why the programming model you meet in chapter 2 has a host and a device.
        </>)}
      </p>
      <Source section="1.1 and 1.3" pages="5–6, 10" />

      <h2>{t('The trade you already make', 'Đánh đổi bạn thường gặp khi chọn kích thước lô')}</h2>
      <p>
        {vi ? (<>
        Mỗi lần chọn kích thước lô (batch size), bạn đều thực hiện đánh đổi này. Với lô 1, mỗi yêu cầu được
        xử lý riêng, nên có thời gian xử lý ngắn nhất. Với lô lớn, từng yêu cầu mất nhiều thời gian hơn nhưng GPU
        tạo được nhiều token hơn mỗi giây. Phục vụ ở lô 1 là dùng một bộ xử lý ưu tiên thông lượng để đáp ứng mục
        tiêu độ trễ. Mục <a href={href('ch22', '22.5')}>§22.5</a> của sách cũng liên hệ việc gom lô với hai thiết kế
        trong chương này.
        </>) : (<>
        You make this trade every time you choose a batch size. At batch size 1 a request waits for nothing except
        its own arithmetic, which is the shortest wait one request can have. With a large batch every request takes
        longer, and the GPU finishes more tokens per second in total. Serving at batch size 1 asks a
        throughput-oriented processor to behave like a latency-oriented one. The book draws the same parallel
        itself: its section on batching, <a href={href('ch22', '22.5')}>§22.5</a>, points back to the two designs of
        this chapter.
        </>)}
      </p>
      <Source section="22.5" pages="539–540" />
      <Ask q={questions10['ch01/start/q-tokens']} />

      <h2>{t('Where the other backends fit', 'HIP và Metal trong bức tranh này')}</h2>
      <p>
        {vi ? (<>
        Sách dùng phần cứng NVIDIA để trình bày lập luận này. Theo tài liệu của AMD và Apple, đặc điểm ưu
        tiên thông lượng cũng xuất hiện ở hai backend còn lại của khóa học, GPU AMD lập trình qua HIP và GPU Apple
        lập trình qua Metal.
        </>) : (<>
        The book makes this argument with NVIDIA hardware, and nowhere in the chapter mentions HIP or Metal. By our
        reading of the vendors' own documents, the split between the two designs is not NVIDIA's alone. It carries
        over to the other two backends of this course: AMD GPUs, programmed through HIP, and Apple GPUs, programmed
        through Metal.
        </>)}
      </p>
      <Beyond>
        <p>
          {vi ? (<>
        HIP cung cấp một mô hình lập trình tương tự CUDA cho GPU AMD. AMD lưu ý rằng phần mềm CUDA cần
          được điều chỉnh khi chuyển sang HIP. Metal là API đồ họa và tính toán của Apple. Kernel tính toán được
          viết bằng Metal Shading Language (MSL) và chạy trên một lưới các luồng.
        </>) : (<>
        HIP is CUDA's programming model on AMD GPUs. AMD documents it for AMD GPUs and says it is "not intended to
          be a drop-in replacement" for CUDA. Metal is Apple's graphics and compute API, with kernels written in the
          Metal Shading Language (MSL): a compute function is "also known as a compute kernel" and runs on "a grid of
          threads".
        </>)}
        </p>
        <p>
          {vi ? (<>
        Hình 1.1 minh họa GPU có DRAM riêng, phù hợp với các GPU rời dùng CUDA và HIP. Trên Apple silicon,
          CPU và GPU dùng chung bộ nhớ hệ thống. Chương 2 giải thích điểm này ở <a href={href('ch02', '2.4')}>§2.4,
          bộ nhớ thiết bị và truyền dữ liệu</a>.
        </>) : (<>
        One difference to hold on to. Fig. 1.1 draws the GPU with its own DRAM. That fits CUDA and HIP devices. It
          does not fit Apple silicon, where CPU and GPU share system memory. Chapter 2 takes it up in{' '}
          <a href={href('ch02', '2.4')}>§2.4, device memory and data transfer</a>.
        </>)}
        </p>
        <p className="muted">{t('Checked 2026-11-20.', 'Kiểm tra ngày 20/11/2026.')}</p>
      </Beyond>
    </>
  );
}
