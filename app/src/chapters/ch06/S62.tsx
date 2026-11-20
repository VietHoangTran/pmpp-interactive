import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, num, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  AMD_CDNA_L2, BUS_PRESETS, DEMAND, FIG_6_9, banksNeeded, busBandwidth, busesFor, bytesInFlight, channelTimeline, deal,
  elementsToFill, phaseChannels, utilisationBound, type Bus, type Mapping,
} from '../../sim/dram';
import { ChannelDrawing, DramBanksScene } from './DramBanksScene';

const pct = (fraction: number) => Math.round(fraction * 1000) / 10;
const round2 = (v: number) => Math.round(v * 100) / 100;
const gb = (v: number) => round2(v).toLocaleString('en-US', { maximumFractionDigits: 2 });
const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;

// ---------------------------------------------------------------- Fig. 6.7, redrawn
function ChannelsFigure() {
  const chW = 156;
  const gap = 20;
  const x0 = 26;
  const bankW = 30;
  return (
    <figure className="wide" style={{ maxWidth: '58rem', margin: '0.4rem 0 1.6rem' }}>
      <svg viewBox="0 0 720 214" role="img" aria-label="A processor with four channels; each channel is a memory controller with its own bus, and each bus serves four banks" style={{ width: '100%', maxWidth: 720, display: 'block', fontFamily: 'var(--sans)' }}>
        <rect x={10} y={146} width={700} height={60} rx={4} fill="var(--page)" stroke="var(--ink)" />
        <text x={20} y={200} fontSize="11.5" fontWeight="600">processor</text>
        {[0, 1, 2, 3].map((c) => {
          const cx = x0 + c * (chW + gap);
          return (
            <g key={c}>
              <rect x={cx} y={154} width={chW} height={28} rx={3} fill="var(--global-wash)" stroke="var(--global)" />
              <text x={cx + chW / 2} y={172} fontSize="11.5" textAnchor="middle">channel {c}: controller</text>
              <line x1={cx + chW / 2} x2={cx + chW / 2} y1={154} y2={118} stroke="var(--global)" strokeWidth="5" />
              <text x={cx + chW / 2 + 8} y={140} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>bus</text>
              <line x1={cx + 12} x2={cx + chW - 12} y1={118} y2={118} stroke="var(--global)" strokeWidth="3" />
              {[0, 1, 2, 3].map((b) => {
                const bx = cx + 12 + b * ((chW - 24 - bankW) / 3);
                return (
                  <g key={b}>
                    <line x1={bx + bankW / 2} x2={bx + bankW / 2} y1={118} y2={100} stroke="var(--global)" strokeWidth="1.5" />
                    <rect x={bx} y={40} width={bankW} height={60} rx={2} fill="var(--cell)" stroke="var(--global)" />
                    <text x={bx + bankW / 2} y={74} fontSize="10" textAnchor="middle">b{b}</text>
                  </g>
                );
              })}
            </g>
          );
        })}
        <text x={x0} y={28} fontSize="11" style={{ fill: 'var(--ink-2)' }}>banks: cell array, sense amplifiers, burst interface</text>
      </svg>
      <figcaption className="caption muted" style={{ fontFamily: 'var(--sans)', fontSize: '0.8rem' }}>
        The structure of the book's Fig. 6.7, redrawn: four channels, each a memory controller with its own bus, four
        banks on each bus. Real devices have more of both.
      </figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------- bus calculator
const PRESET_KEYS = ['ddr5', 'hbm3e'] as const;
const sameBus = (a: Bus, b: Bus) => a.widthBits === b.widthBits && a.clockGHz === b.clockGHz && a.ddr === b.ddr && a.buses === b.buses;

function BusCalculator() {
  const [bus, setBus] = useState<Bus>({ widthBits: 64, clockGHz: 3.2, ddr: true, buses: 1 });
  const [target, setTarget] = useState<number>(DEMAND.cpu);
  const per = busBandwidth(bus.widthBits, bus.clockGHz, bus.ddr);
  const total = busBandwidth(bus.widthBits, bus.clockGHz, bus.ddr, bus.buses);
  const need = busesFor(target, per);
  const isHbm = sameBus(bus, BUS_PRESETS.hbm3e);
  const field = (label: string, key: 'widthBits' | 'clockGHz' | 'buses', step: number, min: number) => (
    <label className="field">
      {label}
      <input
        type="number"
        step={step}
        min={min}
        value={bus[key]}
        aria-label={label}
        onChange={(e) => setBus({ ...bus, [key]: Math.max(min, Number(e.target.value) || min) })}
      />
    </label>
  );
  return (
    <Bench
      title="Bus bandwidth"
      task="Find how many DDR5 buses a CPU and a GPU need. Then load the book's HBM3E inputs."
      foot="Width × clock, × 2 for double data rate (the book's formula, p. 133; NVIDIA documents the same one). GB here is 10⁹ bytes. Peak figures: a bus delivers this only when it is never idle, which is the rest of this section."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }} role="group" aria-label="Presets">
        <span className="muted">Presets</span>
        {PRESET_KEYS.map((k) => (
          <button key={k} className="quiet" data-preset={k} aria-pressed={sameBus(bus, BUS_PRESETS[k])}
            onClick={() => { const p = BUS_PRESETS[k]; setBus({ widthBits: p.widthBits, clockGHz: p.clockGHz, ddr: p.ddr, buses: p.buses }); }}>
            {BUS_PRESETS[k].label}
          </button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '1rem', alignItems: 'flex-end' }}>
        {field('Bus width (bits)', 'widthBits', 8, 1)}
        {field('Bus clock (GHz)', 'clockGHz', 0.1, 0.1)}
        <label className="field">
          Transfers per clock
          <span className="seg">
            <button aria-pressed={!bus.ddr} onClick={() => setBus({ ...bus, ddr: false })}>1 (single)</button>
            <button aria-pressed={bus.ddr} onClick={() => setBus({ ...bus, ddr: true })}>2 (double data rate)</button>
          </span>
        </label>
        {field('Number of buses', 'buses', 1, 1)}
      </div>
      <div className="cols">
        <div className="panel" aria-live="polite">
          <p style={{ margin: '0 0 0.4rem' }}>
            One bus: {bus.widthBits / 8} bytes × {bus.ddr ? '2 × ' : ''}
            {bus.clockGHz} GHz = <b><output data-out="per" data-value={round2(per)}>{gb(per)}</output> GB/s</b>
          </p>
          <p style={{ fontSize: '1.6rem', fontWeight: 600, margin: '0.3rem 0' }}>
            <output data-out="total" data-value={round2(total)}>{gb(total)}</output> GB/s
          </p>
          <p className="muted" style={{ margin: 0 }}>from {plural(bus.buses, 'bus', 'buses')}</p>
          {isHbm ? (
            <p style={{ margin: '0.7rem 0 0' }} data-out="erratum">
              <span className="chip fail">the book prints 1 TB/s</span> for these inputs (p. 133), and "4.9 GHz/s" for the
              clock. Its own formula gives {gb(total)} GB/s, about 1.25 TB/s, and that is one HBM stack.
            </p>
          ) : null}
        </div>
        <div className="panel">
          <label className="field">
            Bandwidth you need (GB/s)
            <input type="number" min={1} value={target} aria-label="Bandwidth you need" onChange={(e) => setTarget(Math.max(1, Number(e.target.value) || 1))} />
          </label>
          <div className="row" style={{ margin: '0.5rem 0' }}>
            <button className="quiet" data-target="cpu" aria-pressed={target === DEMAND.cpu} onClick={() => setTarget(DEMAND.cpu)}>200: a CPU (p. 133)</button>
            <button className="quiet" data-target="gpu" aria-pressed={target === DEMAND.gpu} onClick={() => setTarget(DEMAND.gpu)}>1000: a GPU (p. 133)</button>
          </div>
          <p style={{ margin: 0 }}>
            {target} / {gb(per)} = {fmt(target / per)}, so{' '}
            <b><output data-out="need">{need}</output> {need === 1 ? 'bus' : 'buses'}</b> of this kind.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- channel and banks explorer
function ChannelExplorer() {
  const [R, setR] = useState(20);
  const [banks, setBanks] = useState(1);
  const [share, setShare] = useState(0);
  const [seed, setSeed] = useState(1);
  const [chan, setChan] = useState(16);
  const tl = useMemo(() => channelTimeline({ R, banks, conflictShare: share / 100, seed }), [R, banks, share, seed]);
  const span = Math.max(43, Math.min(124, 3 * (R + 1) + 1));
  const bound = utilisationBound(R, banks);
  const need = banksNeeded(R);
  const conflicts = tl.accesses.filter((a) => a.conflict).length;
  const book = R === 20 && banks === 1 && share === 0 && chan === 16;
  const slider = (name: string, value: number, set: (v: number) => void, min: number, max: number, step = 1, unit = '') => (
    <label className="field">
      {name} = {value}
      {unit}
      <input type="range" min={min} max={max} step={step} value={value} aria-label={name} style={{ width: '11rem' }} onChange={(e) => set(+e.target.value)} />
    </label>
  );
  return (
    <Bench
      title="One channel, its banks, in time"
      task="Start at the book's case. Add banks until the bus is full; then send some requests to a busy bank."
      foot="Time runs left to right in units of one burst. Each request waits R units in its bank, then sends one burst. The percentage is measured once the start-up (the first latency, which nothing hides) has passed."
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        {slider('Latency-to-burst ratio R', R, setR, 1, 40)}
        {slider('Banks', banks, setBanks, 1, 48)}
        {slider('Requests sent to a busy bank', share, setShare, 0, 50, 5, '%')}
        <label className="field">
          Channel bus (GB/s)
          <input type="number" min={1} value={chan} aria-label="Channel bus bandwidth" onChange={(e) => setChan(Math.max(1, Number(e.target.value) || 1))} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '1rem' }}>
        <button className="quiet" data-preset="book" aria-pressed={book} onClick={() => { setR(20); setBanks(1); setShare(0); setChan(16); }}>
          The book's case: R = 20, one bank, 16 GB/s
        </button>
        <button className="quiet" data-preset="full" aria-pressed={banks === need && share === 0} onClick={() => { setBanks(need); setShare(0); }}>
          R + 1 banks
        </button>
        {share > 0 ? (
          <button className="quiet" onClick={() => setSeed(seed + 1)}>Draw other conflicts</button>
        ) : null}
      </div>
      <div className="panel" style={{ marginBottom: '0.9rem', overflowX: 'auto' }}>
        <ChannelDrawing minWidth={540} tl={tl} span={span} label={`Timeline of ${banks} banks and the bus over ${span} burst times`} data={{ util: pct(tl.utilisation) }} />
      </div>
      <div className="cols">
        <div className="panel" aria-live="polite">
          <p className="muted" style={{ margin: 0 }}>Bus busy</p>
          <p style={{ fontSize: '2rem', fontWeight: 600, margin: '0 0 0.2rem', lineHeight: 1.1 }}>
            <output data-out="util" data-value={pct(tl.utilisation)}>{pct(tl.utilisation)}%</output>
          </p>
          <p style={{ margin: 0 }}>
            delivers <b><output data-out="gbs" data-value={round2(tl.utilisation * chan)}>{gb(tl.utilisation * chan)}</output> GB/s</b> of the bus's {chan}
          </p>
        </div>
        <div className="panel">
          <p style={{ margin: '0 0 0.6rem' }}>
            The book's bound, min(1, banks / (R + 1)) = min(1, {banks} / {R + 1}) ={' '}
            <b><output data-out="bound" data-value={pct(bound)}>{pct(bound)}%</output></b>
          </p>
          <p style={{ margin: '0 0 0.6rem' }}>
            Banks that fill the bus: R + 1 = <b><output data-out="need">{need}</output></b>
          </p>
          <p style={{ margin: 0 }} data-out="verdict">
            {conflicts > 0 ? (
              <>
                <b>{plural(conflicts, 'conflict')}</b> in {tl.accesses.length} requests. Each one waited for a bank that
                was still busy, everything behind it waited too, and the bus sat idle meanwhile: {pct(bound - tl.utilisation)}{' '}
                points below the bound.
              </>
            ) : banks < need ? (
              <>
                <b>{plural(need - banks, 'bank')} short.</b> The bus is idle {pct(1 - tl.utilisation)}% of the time,
                waiting for some bank to finish its latency.
              </>
            ) : banks === need ? (
              <>
                <b>Exactly enough.</b> As one burst ends, the next bank's latency is over.
              </>
            ) : (
              <>
                <b>{plural(banks - need, 'bank')} more than needed.</b> No gain in this model. In a real channel the
                extra banks make it less likely that two requests arriving together want the same bank; here that
                likelihood is the conflict slider.
              </>
            )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the dealer
const elems = (cell: number[]): string => {
  const runs: [number, number][] = [];
  for (const i of cell) {
    const last = runs[runs.length - 1];
    if (last && last[1] === i - 1) last[1] = i;
    else runs.push([i, i]);
  }
  const show = runs.slice(0, 3).map(([a, b]) => (a === b ? `M[${a}]` : `M[${a}..${b}]`));
  return runs.length > 3 ? `${show.join(', ')} and ${runs.length - 3} more runs` : show.join(', ');
};
const sameMap = (a: Mapping, b: Mapping) => a.elemBytes === b.elemBytes && a.burstBytes === b.burstBytes && a.channels === b.channels && a.banks === b.banks;

function Dealer() {
  const [m, setM] = useState<Mapping>({ ...FIG_6_9 });
  const [n, setN] = useState(16);
  const cells = useMemo(() => deal(n, m), [n, m]);
  const fill = elementsToFill(m);
  const usedChannels = cells.filter((ch) => ch.some((b) => b.length > 0)).length;
  const usedCells = cells.flat().filter((c) => c.length > 0).length;
  const seg = (label: string, key: keyof Mapping, values: number[], unit = '') => (
    <label className="field">
      {label}
      <span className="seg">
        {values.map((v) => (
          <button key={v} aria-pressed={m[key] === v} onClick={() => setM({ ...m, [key]: v })}>
            {v}
            {unit}
          </button>
        ))}
      </span>
    </label>
  );
  return (
    <Bench
      title="Dealing an array to channels and banks"
      task="Reproduce Fig. 6.9. Then load AMD's interleave and find how big an array must be before it uses every channel."
      foot="Consecutive bursts go to consecutive channels; after the last channel the next round moves to the next bank (the book's rule, p. 135). The array starts at channel 0, bank 0."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }} role="group" aria-label="Presets">
        <span className="muted">Presets</span>
        <button className="quiet" data-preset="fig69" aria-pressed={sameMap(m, FIG_6_9)} onClick={() => { setM({ ...FIG_6_9 }); setN(16); }}>
          Fig. 6.9: 8-byte bursts, 4 channels, 2 banks
        </button>
        <button className="quiet" data-preset="amd" aria-pressed={sameMap(m, AMD_CDNA_L2)} onClick={() => setM({ ...AMD_CDNA_L2 })}>
          AMD CDNA: 256-byte interleave across 32 L2 channels
        </button>
      </div>
      <div className="row" style={{ marginBottom: '1rem', alignItems: 'flex-end' }}>
        {seg('Element size', 'elemBytes', [2, 4, 8], ' B')}
        <label className="field">
          Bytes per channel turn (burst)
          <select value={m.burstBytes} aria-label="Bytes per channel turn" onChange={(e) => setM({ ...m, burstBytes: +e.target.value })}>
            {[8, 16, 32, 64, 128, 256].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          Channels
          <select value={m.channels} aria-label="Channels" onChange={(e) => setM({ ...m, channels: +e.target.value })}>
            {[1, 2, 4, 8, 16, 32].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        {seg('Banks per channel', 'banks', [1, 2, 4])}
        <label className="field">
          Array length (elements)
          <select value={n} aria-label="Array length" onChange={(e) => setN(+e.target.value)}>
            {[8, 16, 32, 64, 256, 1024, 2048, 4096].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
      </div>
      <div style={{ display: 'grid', gap: '0.5rem', gridTemplateColumns: `repeat(auto-fill, minmax(${m.channels > 8 ? 7.5 : 10.5}rem, 1fr))`, marginBottom: '0.9rem' }}>
        {cells.map((ch, c) => (
          <div key={c} className="panel" data-channel={c} style={{ padding: '0.45rem 0.6rem', borderColor: ch.some((b) => b.length) ? 'var(--global)' : 'var(--rule)' }}>
            <div className="mx-label">channel {c}</div>
            {ch.map((cell, b) => (
              <div key={b} data-cell={`${c},${b}`} data-count={cell.length} style={{ fontSize: '0.74rem', color: cell.length ? 'var(--ink)' : 'var(--ink-3)' }}>
                {m.banks > 1 ? <span className="muted">bank {b}: </span> : null}
                {cell.length ? elems(cell) : 'nothing'}
              </div>
            ))}
          </div>
        ))}
      </div>
      <p style={{ margin: 0 }} aria-live="polite">
        {n} elements of {m.elemBytes} bytes ({n * m.elemBytes} bytes) touch{' '}
        <b><output data-out="channels">{usedChannels}</output> of {m.channels} channels</b>
        {m.banks > 1 ? <> and {usedCells} of {m.channels * m.banks} channel-bank pairs</> : null}. Every channel gets a whole
        turn from <output data-out="fill-channels">{fill.channels}</output> elements ({fill.channels * m.elemBytes} bytes)
        {m.banks > 1 ? <>, every bank of every channel from {fill.banks}</> : null}.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the toy multiplication, phase by phase
function ToyReplay() {
  const { progress } = useProgress();
  const unlocked = progress.answers['ch06/6.2/p-8x8'] != null;
  const [width, setWidth] = useState(4);
  const [ph, setPh] = useState(0);
  const phases = useMemo(() => phaseChannels(width, 2, FIG_6_9), [width]);
  const cur = phases[Math.min(ph, phases.length - 1)];
  const loaded = new Set(cur.blocks.flatMap((b) => b.elements));
  const tiles = width / 2;
  const rows = Array.from({ length: tiles }, (_, r) => cur.blocks.filter((b) => b.row === r));
  const chOf = (i: number) => Math.floor((i * FIG_6_9.elemBytes) / FIG_6_9.burstBytes) % FIG_6_9.channels;
  return (
    <Bench
      title="The tiled multiplication, phase by phase"
      task={unlocked ? 'Step through the phases at both sizes and count the channels each phase keeps busy.' : 'Step through the 4 × 4 case. The 8 × 8 case opens once you commit the prediction below.'}
      foot="Figs. 6.10 and 6.11's setting: row-major M, 2 × 2 tiles, every block running at once, Fig. 6.9's mapping (2-element bursts, 4 channels, 2 banks). Only M's loads are traced, as in the book. The subscript in each cell is its channel."
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Matrix
          <span className="seg">
            <button aria-pressed={width === 4} onClick={() => { setWidth(4); setPh(0); }}>4 × 4</button>
            <button aria-pressed={width === 8} disabled={!unlocked} title={unlocked ? undefined : 'Commit your prediction below first'} onClick={() => { setWidth(8); setPh(0); }}>8 × 8</button>
          </span>
        </label>
        <label className="field">
          Phase
          <span className="seg">
            {phases.map((p) => (
              <button key={p.ph} aria-pressed={p.ph === cur.ph} onClick={() => setPh(p.ph)}>
                phase {p.ph}
              </button>
            ))}
          </span>
        </label>
      </div>
      <div className="cols">
        <div>
          <div className="mx-label">M ({width} × {width}, row-major; this phase's loads filled)</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${width}, auto)` }} data-out="grid">
            {Array.from({ length: width * width }, (_, i) => (
              <span key={i} className={loaded.has(i) ? 'hot' : ''} data-i={i}>
                {i}
                <sub>c{chOf(i)}</sub>
              </span>
            ))}
          </div>
        </div>
        <div className="panel" aria-live="polite">
          <p style={{ margin: '0 0 0.5rem' }}>
            <b>Phase {cur.ph}</b>: channels busy{' '}
            <output data-out="phase-channels" data-value={cur.channels.join(',')}>{cur.channels.join(' and ')}</output>
            {' '}({cur.channels.length} of {FIG_6_9.channels})
          </p>
          <div className="row" style={{ marginBottom: '0.6rem', gap: '0.3rem' }}>
            {Array.from({ length: FIG_6_9.channels }, (_, c) => (
              <span key={c} className={'chip' + (cur.channels.includes(c) ? ' global' : '')} data-lit={cur.channels.includes(c) ? 'yes' : 'no'}>
                channel {c}
              </span>
            ))}
          </div>
          <div style={{ overflowX: 'auto' }}>
          <table className="threads">
            <tbody>
              {rows.map((bs) => (
                <tr key={bs[0].row}>
                  <td>
                    {bs.length === 2
                      ? `Blocks ${bs[0].row},0 and ${bs[0].row},1`
                      : `Blocks ${bs[0].row},0 to ${bs[0].row},${bs.length - 1}`}
                  </td>
                  <td>{bs[0].elements.map((e) => `M[${e}]`).join(', ')}</td>
                  <td>ch. {bs[0].channels.join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
          <p className="muted" style={{ margin: '0.5rem 0 0' }}>
            Blocks in the same row of tiles load the same M elements; the book's block names give the tile row first.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- bytes in flight (ours)
const BOOK_BW = { key: 'book', label: "The book's example, H100 SXM: 3.35 TB/s (p. 94)", gbs: 3350 };

function BytesInFlight() {
  const { reports } = useReports();
  const measured = useMemo(
    () =>
      realBackends(reports).flatMap((b) => {
        const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
        const v = num(r?.bandwidth_gb_per_s);
        return v ? [{ key: b, label: `Yours: ${String(reports[b].device.name ?? b)} (${b}), lab 5.1: ${fmt(v)} GB/s`, gbs: v }] : [];
      }),
    [reports],
  );
  const options = [...measured, BOOK_BW];
  const [key, setKey] = useState<string | null>(null);
  const [custom, setCustom] = useState(500);
  const [lat, setLat] = useState(50);
  const [bytes, setBytes] = useState(4);
  const [warp, setWarp] = useState(32);
  const chosen = key ?? options[0].key;
  const bw = chosen === 'custom' ? custom : (options.find((o) => o.key === chosen) ?? BOOK_BW).gbs;
  const f = bytesInFlight(bw, lat, bytes, warp);
  const wider = bytesInFlight(bw, lat, 16, warp);
  return (
    <Bench
      title="Bytes in flight (ours, not the book's)"
      task="Pick a bandwidth and a latency, then see how many loads must be outstanding at every moment. Switch to 16-byte loads."
      foot="Little's law: bytes outstanding = bandwidth × latency. One GB/s is one byte per nanosecond. The latency is an input: no vendor documents one in nanoseconds; NVIDIA describes global-memory latency as hundreds of clock cycles (facts A1)."
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field">
          Bandwidth to sustain
          <select value={chosen} aria-label="Bandwidth to sustain" onChange={(e) => setKey(e.target.value)}>
            {options.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
            <option value="custom">Custom</option>
          </select>
        </label>
        {chosen === 'custom' ? (
          <label className="field">
            GB/s
            <input type="number" min={1} value={custom} aria-label="Custom bandwidth" onChange={(e) => setCustom(Math.max(1, Number(e.target.value) || 1))} />
          </label>
        ) : null}
        <label className="field">
          Latency = {lat} ns
          <input type="range" min={10} max={1000} step={10} value={lat} aria-label="Latency in nanoseconds" style={{ width: '11rem' }} onChange={(e) => setLat(+e.target.value)} />
        </label>
        <label className="field">
          Bytes per load (per thread)
          <span className="seg">
            {[4, 8, 16].map((b) => (
              <button key={b} aria-pressed={bytes === b} onClick={() => setBytes(b)}>
                {b === 4 ? '4 (float)' : b === 8 ? '8 (float2)' : '16 (float4)'}
              </button>
            ))}
          </span>
        </label>
        <label className="field">
          Warp width
          <span className="seg">
            {[32, 64].map((w) => (
              <button key={w} aria-pressed={warp === w} onClick={() => setWarp(w)}>{w}</button>
            ))}
          </span>
        </label>
      </div>
      <div className="cols">
        <div className="panel" aria-live="polite">
          <p style={{ margin: 0 }}>
            {fmt(bw)} GB/s × {lat} ns =
          </p>
          <p style={{ fontSize: '1.6rem', fontWeight: 600, margin: '0.2rem 0' }}>
            <output data-out="bytes" data-value={Math.round(f.bytes)}>{fmt(Math.round(f.bytes))}</output> bytes
          </p>
          <p className="muted" style={{ margin: 0 }}>about {fmt(f.bytes / 1024)} KiB on their way, all the time</p>
        </div>
        <div className="panel">
          <p style={{ margin: '0 0 0.5rem' }}>
            That is <b><output data-out="loads" data-value={Math.round(f.loads)}>{fmt(Math.round(f.loads))}</output></b> loads of {bytes} bytes
            outstanding, or <b><output data-out="warps" data-value={Math.round(f.warpLoads)}>{fmt(Math.round(f.warpLoads))}</output></b>{' '}
            warp-wide load instructions of {warp} lanes.
          </p>
          <p style={{ margin: 0 }} className="muted">
            {bytes === 16
              ? 'With 16 bytes per lane each instruction carries four times what a float load carries.'
              : `With 16-byte loads it would take ${fmt(Math.round(wider.warpLoads))} warp-wide instructions instead.`}{' '}
            On Metal the SIMD-group width comes from the pipeline, not from a constant.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS62: Record<string, Question> = {
  'ch06/6.2/q-buses': {
    kind: 'numeric',
    id: 'ch06/6.2/q-buses',
    prompt: <>How many 64-bit DDR5-6400 buses (51.2 GB/s each) does it take to supply 1000 GB/s?</>,
    answer: 20,
    explain: (
      <p>
        1000 / 51.2 = 19.5, and half a bus is not on offer: 20, the book's own count. A CPU at 200 GB/s needs 3.9, so
        four. Twenty buses of wires across a board is the problem HBM solves.
      </p>
    ),
    source: 'ch. 6 §6.2, p. 133',
  },
  'ch06/6.2/q-hbm': {
    kind: 'numeric',
    id: 'ch06/6.2/q-hbm',
    prompt: (
      <>
        The book's HBM3E example: a 4.9 GHz bus clock, double data rate, and 16 buses of 64 bits. Using the formula on
        the same page, what bandwidth do these inputs give?
      </>
    ),
    answer: 1254.4,
    tol: 0.01,
    unit: 'GB/s',
    explain: (
      <p>
        8 bytes × 2 × 4.9 GHz × 16 = 1254.4 GB/s, about 1.25 TB/s. The book prints "1 TB/s" (and "4.9 GHz/s" for the
        clock). It is also the figure for one HBM stack: memory makers list 1.18 to more than 1.23 TB/s per HBM3E stack,
        and a GPU carries several stacks (facts A8).
      </p>
    ),
    source: 'ch. 6 §6.2, p. 133; erratum (notes item 3, facts A8 and E1)',
  },
  'ch06/6.2/q-util': {
    kind: 'numeric',
    id: 'ch06/6.2/q-util',
    prompt: (
      <>
        A channel's bus can move 32 GB/s. Each access spends 15 units of latency in its bank, then 1 unit of burst on the
        bus. The channel has two banks and requests are always waiting. How much does it deliver?
      </>
    ),
    answer: 4,
    tol: 0.01,
    unit: 'GB/s',
    explain: (
      <p>
        One bank keeps the bus busy 1 unit in every 15 + 1 = 16. Two banks overlap their latencies and send two bursts
        in every 16 units: 2/16 = 12.5 %, and 12.5 % of 32 GB/s is 4 GB/s. Filling the bus would take R + 1 = 16 banks.
      </p>
    ),
    source: 'ch. 6 §6.2, p. 134 (the R + 1 reasoning)',
  },
  'ch06/6.2/q-banks': {
    kind: 'numeric',
    id: 'ch06/6.2/q-banks',
    prompt: <>A channel's latency-to-burst ratio is 30. What is the smallest number of banks that keeps its bus busy all the time?</>,
    answer: 31,
    explain: (
      <p>
        R + 1 = 31: while one bank sends its burst, the other 30 are each somewhere in their 30-unit latency. Real
        channels have more, because two requests that need the same bank at once cannot overlap (a bank conflict), and
        because a bank's cell array cannot grow without its latency growing, so capacity alone needs many banks.
      </p>
    ),
    source: 'ch. 6 §6.2, pp. 134–135',
  },
  'ch06/6.2/q-onechannel': {
    kind: 'choice',
    id: 'ch06/6.2/q-onechannel',
    prompt: (
      <>
        A CDNA GPU interleaves addresses across 32 L2 channels, 256 bytes at a time. Many warps each read one element
        down column 0 of a row-major float matrix 2048 wide (thread r reads row r). How many of the 32 L2 channels do
        these reads use?
      </>
    ),
    options: [
      { label: 'One.', correct: true },
      { label: 'All 32: interleaving spreads every array over every channel.', why: 'It spreads consecutive addresses. These are 8192 bytes apart.' },
      { label: 'Eight.', why: 'Work out the distance between two consecutive reads in bytes, and divide by 256.' },
      { label: 'It depends on the warp width.', why: 'The channel depends only on the address; which warp issues the read does not change it.' },
    ],
    explain: (
      <p>
        Each row is 2048 × 4 = 8192 bytes = 32 × 256. Element (r, 0) starts 32r chunks of 256 bytes into the matrix, so
        it always falls in the same channel: every one of these reads queues on one channel while 31 sit idle. Use the
        dealer with AMD's preset to see the 2048-element pattern. Spreading simultaneous accesses over channels is half
        of the symbiosis on p. 137; NVIDIA and Apple do not document their mapping, so there you can only measure.
      </p>
    ),
    source: "our arithmetic on AMD's documented interleave (facts A9); ch. 6 §6.2, p. 137",
  },
  'ch06/6.2/q-occupancy': {
    kind: 'choice',
    id: 'ch06/6.2/q-occupancy',
    prompt: (
      <>
        A kernel's loads are perfectly coalesced, but it reaches only half the bandwidth you measured for your device
        with a simple copy. Its occupancy is 25 %. What is the most likely reason?
      </>
    ),
    options: [
      { label: 'Too few requests in flight to keep every channel and bank busy.', correct: true },
      { label: 'Its accesses must be uncoalesced after all.', why: 'Coalescing decides how much each request carries, not how many are outstanding. The premise says each request is full.' },
      { label: 'DRAM latency is too long for any kernel to reach that bandwidth.', why: 'Your simple copy reached it, against the same latency. Enough requests in flight hide it.' },
      { label: 'The caches are thrashing.', why: 'Possible in a kernel that reuses data, but nothing here points to reuse; a quarter of the warps points to too little in flight.' },
    ],
    explain: (
      <p>
        With a quarter of the possible warps resident, the kernel has far fewer loads outstanding than it could, and
        too few to keep every channel's banks busy: the bus idles while banks wait. The fixes: more warps (occupancy),
        or more bytes in each load (section 6.3). NVIDIA's triage guide calls this "bytes-in-flight starvation".
      </p>
    ),
    source: "ch. 6 §6.2, pp. 135–136; NVIDIA's Compute Triage Guide (facts A10)",
  },
};

// ---------------------------------------------------------------- the 8x8 reveal
function EightByEight({ committed }: { committed: number }) {
  const eight = phaseChannels(8, 2, FIG_6_9);
  const per = eight[0].channels.length;
  return (
    <>
      <p data-out="p-8x8">
        <b>{per}.</b> {committed === per ? 'That is what you had.' : 'The trace shows why.'} Element 8r + c of M sits in
        burst 4r + ⌊c/2⌋. Phase ph reads columns 2ph and 2ph + 1, so every M load of the phase is in burst 4r + ph, which
        is channel ph: channel {eight.map((p) => p.channels[0]).join(', then ')}. All four channels are used over the run,
        but one at a time, which is less spread than the 4 × 4 case manages.
      </p>
      <p className="muted">
        The book's "use all four channels" (p. 137) holds across phases, not at once. This derivation is ours (notes item
        8). The 8 × 8 option is now open in the explorer above.
      </p>
    </>
  );
}

// ---------------------------------------------------------------- page
function Muted({ children }: { children: ReactNode }) {
  return <span className="muted">{children}</span>;
}

export function S62() {
  return (
    <>
      <p>
        Section 6.1 made each request efficient: a warp's neighbouring addresses become one burst. That is still not
        enough. The book puts a DRAM access at tens of nanoseconds (p. 124) and the burst itself at very little time;
        a processor that waits for one burst at a time gets a small fraction of what the memory could supply. DRAM
        systems add two more levels of parallelism on top of bursts: channels and banks. This section is about what
        they need from your kernel.
      </p>
      <Source section="6.2" pages="132–133" figs="Fig. 6.7" />

      <h2>Channels and banks</h2>
      <p>
        A channel is a memory controller with its own bus, and each bus has several banks behind it. A bank holds an
        array of DRAM cells, the sense amplifiers that read them, and the interface that puts bursts on the bus. A
        processor has one or more channels.
      </p>
      <ChannelsFigure />
      <p>
        Two different things limit a channel. Its bus can move only so many bytes per second, and each bank needs a long
        time before it has anything to send. The first is arithmetic; the second is solved by having many banks.
      </p>

      <h2>What one bus can carry</h2>
      <p>
        A bus moves its width in bytes on every clock edge it uses: width × clock, twice that for double data rate (DDR),
        which transfers on both the rising and the falling edge. The book's example is a 64-bit DDR5-6400 bus clocked at
        3.2 GHz: 8 bytes × 2 × 3.2 GHz = 51.2 GB/s. The book also puts a modern CPU's appetite at 200 GB/s or more and a
        GPU's at more than 1000 GB/s. Work out the bus counts before you touch the calculator.
      </p>
      <Source section="6.2" pages="133" />
      <Ask q={questionsS62['ch06/6.2/q-buses']} />
      <Ask q={questionsS62['ch06/6.2/q-hbm']} />
      <BusCalculator />
      <p>
        Four buses for the CPU and twenty for the GPU. Twenty 64-bit buses driven across a circuit board draw too much
        power, which is why GPUs moved to High Bandwidth Memory (HBM): the DRAM is packaged with the processor, the
        connections are short, and both the clock and the number of buses go up.
      </p>
      <Source section="6.2" pages="133" />
      <p>
        The book's HBM3E example has two slips. It gives the bus clock as "4.9 GHz/s", which should be GHz. And it says
        the result is "1 TB/s", but its own formula on its own inputs gives 1254.4 GB/s, about 1.25 TB/s, as the
        calculator shows. That is for one HBM stack, and a GPU carries several. The book's "one to eight channels" per
        processor also undercounts GPUs: its own example device, the H100 SXM5, has 10 memory controllers and 5 HBM3
        stacks.
      </p>
      <Today checked="2026-11-20 (ch06-facts.md A7, A8; ch04-facts.md A4)">
        <p>
          NVIDIA documents the same width × clock × 2 formula; AMD calls 8 to 16 memory controllers typical. Memory
          makers list 1.18 to more than 1.23 TB/s per HBM3E stack. Not checked: DDR5's two 32-bit subchannels.
        </p>
      </Today>

      <h2>Why one bank is not enough</h2>
      <p>
        Inside a channel, every access to a bank has two parts: a long wait while the bank reads its cell array, then a
        short burst on the bus. The bus is in use only during the burst. If the wait is R times as long as the burst,
        one bank keeps the bus busy at most one unit in every R + 1. The book's illustration takes R = 20: 1/21, or 4.8 %
        of the time, so a channel whose bus could move 16 GB/s delivers at most 0.76 GB/s.
      </p>
      <Source section="6.2" pages="133–134" figs="Fig. 6.8" />
      <DramBanksScene />
      <p>
        The cure is to start an access in a second bank while the first is still waiting. Their latencies overlap and
        their bursts go out back to back: two banks, twice the utilisation. Each further bank adds the same amount until
        R + 1 of them fill the bus. Check that in the explorer, then make some requests land on a bank that is still
        busy.
      </p>
      <Source section="6.2" pages="134" />
      <p>
        The explorer below is the book's model of one channel, run forward in time. The narrated animation above was
        drawn from the same model.
      </p>
      <Simplification>
        <p>
          Every access has the same latency and the same burst; the controller starts requests in arrival order, at most
          one per burst time. Real DRAM also refreshes its cells, and treats an access to an already open row
          differently; neither is here.
        </p>
        <p>R = 20 is the book's illustration, not a measured ratio. Banks per HBM3E channel could not be checked (facts A9).</p>
      </Simplification>
      <ChannelExplorer />
      <p>
        Real channels have more than R + 1 banks, for two reasons the book gives. Requests do not arrive in a tidy
        rotation: two that want the same bank at the same time cannot overlap, which is a <b>bank conflict</b>, and more
        banks make one less likely. And a bank's cell array cannot grow without its latency growing, so capacity alone
        calls for many banks.
      </p>
      <Source section="6.2" pages="134–135" />
      <Ask q={questionsS62['ch06/6.2/q-util']} />
      <Ask q={questionsS62['ch06/6.2/q-banks']} />

      <h2>Spreading addresses over channels and banks</h2>
      <p>
        Banks and channels help only if requests that arrive together go to different ones. The hardware arranges that
        by dealing the address space out like cards: consecutive bursts go to consecutive channels, and once every
        channel has had one, the next round goes to the next bank. In the book's toy (bursts of two 4-byte elements, four
        channels, two banks) M[0] and M[1] land in channel 0, bank 0; M[2] and M[3] in channel 1; M[8] and M[9] back in
        channel 0, bank 1. Even a 16-element array is spread over every channel and bank.
      </p>
      <Source section="6.2" pages="135" figs="Fig. 6.9" />
      <Dealer />
      <p>
        One vendor documents its version. Load AMD's preset and notice how coarse it is: 64 floats share a channel before
        the next one gets any, so a small array sits in one channel and it takes 8192 bytes to give all 32 channels a
        turn.
      </p>
      <Today checked="2026-11-20 (ch06-facts.md A9, B3)">
        <p>
          AMD: the L2 of CDNA GPUs has "32 ... channels at 256-byte interleaving", addresses interleaved across them for
          load balancing. These are L2 channels; AMD does not document the DRAM mapping beneath them. NVIDIA and Apple
          publish no address mapping at all.
        </p>
      </Today>
      <p>
        Interleaving spreads consecutive addresses. Whether the accesses your threads make at the same moment are
        consecutive is a different question, and it is yours.
      </p>
      <Ask q={questionsS62['ch06/6.2/q-onechannel']} />

      <h2>Threads and DRAM need each other</h2>
      <p>
        Put the pieces together. Each bank works on one access at a time, each channel needs about R + 1 of them going to
        keep its bus full, and a GPU has many channels. The memory reaches its rated bandwidth only when a great many
        requests are under way at the same moment, spread over the channels and banks. Requests come from threads, so
        this is the second reason to want high occupancy. Chapter 4 used resident warps to keep the execution units busy
        while one warp waited; here the same warps keep the DRAM busy, because each one has loads outstanding. And each
        access must still be coalesced, or a request carries less than a full burst.
      </p>
      <Source section="6.2" pages="135–136" />
      <p>
        The book traces this through chapter 5's tiled multiplication at toy size: 4 × 4 matrices, 2 × 2 tiles, four
        blocks running at once, and Fig. 6.9's mapping. In phase 0, blocks 0,0 and 0,1 both load M[0], M[1], M[4] and
        M[5], and blocks 1,0 and 1,1 load M[8], M[9], M[12] and M[13]. Step through it.
      </p>
      <Source section="6.2" pages="136–137" figs="Figs. 6.10 and 6.11" />
      <ToyReplay />
      <p>
        Phase 0 keeps channels 0 and 2 busy, phase 1 channels 1 and 3. The book then leaves you a check: that 8 × 8
        matrices with the same 2 × 2 tiles "use all four channels".
      </p>
      <Source section="6.2" pages="137" />
      <Predict
        id="ch06/6.2/p-8x8"
        prompt={
          <>
            Same mapping (2-element bursts, 4 channels, 2 banks), 8 × 8 matrices, 2 × 2 tiles, all sixteen blocks running
            at once. In one phase, how many channels do the M loads of all the blocks hit?
          </>
        }
        unit="channels"
      >
        {(v) => <EightByEight committed={v} />}
      </Predict>
      <p>
        Two blocks loading the same M elements at about the same time need not both go to DRAM: if their timing is close
        enough, a cache can merge them into one access. The book says GPU caches are mainly designed to combine such
        accesses.
      </p>
      <Source section="6.2" pages="136–137" />
      <p>
        How often they manage it on your device is a measurement, and the vendors describe their caches more narrowly
        than the book does.
      </p>
      <Today checked="2026-11-20 (ch06-facts.md A4)">
        <p>
          No vendor page we read says caches are "mainly designed" for this. NVIDIA documents 128-byte lines of four
          32-byte sectors and warns a line may be evicted before reuse; AMD documents hit-on-miss, a request for a line
          already being fetched counting as a hit.
        </p>
      </Today>
      <p>
        The book's closing warning is the one that matters in practice: if all the threads running at one moment access
        data in the same channel, throughput drops sharply, however many of them there are.
      </p>
      <Source section="6.2" pages="137" />
      <p>
        The opposite failure, too few requests at once, has a name in NVIDIA's profiler guide: "bytes-in-flight starvation", a kernel that "is not
        presenting enough outstanding loads to the bus". The guide's remedies are more instruction-level parallelism,
        vectorised loads (section 6.3), or asynchronous copies. NVIDIA's Best Practices Guide adds the counterweight the
        book saves for chapter 15: with enough instruction-level parallelism, a kernel can cover latency at low
        occupancy.
      </p>
      <Today checked="2026-11-20 (ch06-facts.md A10, C5; ch04-facts.md B10)">
        <p>
          The two NVIDIA quotes are from the Nsight Compute Compute Triage Guide and the Best Practices Guide. AMD:
          memory-bound kernels benefit more from high occupancy. Apple's Xcode documentation defines occupancy and shows a
          counter for it, with no API to query it.
        </p>
      </Today>
      <Ask q={questionsS62['ch06/6.2/q-occupancy']} />

      <h2>How many bytes must be in flight</h2>
      <p>
        The book stops at "enough threads". A rule from queueing theory, Little's law, puts a number on it: to sustain a
        bandwidth B against a latency L, B × L bytes must be on their way at every moment. At 1 GB/s that is one byte per
        nanosecond of latency. Batch-1 decoding, which streams every weight once per token with little arithmetic on each
        (section 5.1), lives or dies by this number.
      </p>
      <Beyond>
        <p>
          Little's law and this explorer are ours, not the book's. The default latency, 50 ns, is an illustrative value in
          the book's "tens of nanoseconds" (p. 124), which is the DRAM alone; NVIDIA describes a global load's latency as
          "hundreds of clock cycles" (facts A1).
        </p>
      </Beyond>
      <BytesInFlight />
      <p>
        Two things move the answer, and they are the two levers of this section and the next. More resident warps put
        more loads in flight at once; wider loads put more bytes in each one. Section 6.3 takes the second.
      </p>

      <h2>On Apple silicon</h2>
      <p>
        The book's picture is a discrete GPU with its own DRAM or HBM. An Apple GPU shares system memory with the CPU, so
        both draw on one memory system, and Apple warns that heavy GPU traffic can slow the CPU's own memory accesses. Apple prints a memory bandwidth for each product configuration; the lesson does not use it, because it
        is a shared peak, not what your kernels get. Your number is the one lab 5.1 measured{' '}
        <Muted>(<a href={href('ch05', 'labs')}>chapter 5 labs</a>)</Muted>.
      </p>
      <Today checked="2026-11-20 (ch06-facts.md C6; ch05-facts.md C6; infra-facts.md C13)">
        <p>
          "Apple GPUs have a unified memory model in which the CPU and the GPU share system memory." Apple's developer
          documentation gives no bandwidth figures; its Xcode counters measure the GPU's reads and writes in GB/s while
          your app runs.
        </p>
      </Today>
    </>
  );
}
